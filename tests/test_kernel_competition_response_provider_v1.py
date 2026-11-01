"""Synthetic clock-law integration in the existing event engine, CPU only.

The fit solver is stubbed with an explicit fixed law: these tests validate
adapters/integration, not that actual kernel training identified these slopes.
"""
import copy
import math
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.kernel_provider import KernelResourceProvider
from gsdeploy.forecasting.simulator import FullHorizonModel
from gsdeploy.resources import identity
from tests.test_kernel_competition_features_v2 import capture, provider
from tests.test_kernel_provider import ALIASES
from tools.predictor_validation_20261027 import kernel_competition_features_v2 as f
from tools.predictor_validation_20261027 import kernel_competition_response_provider_v1 as m


def bundle(p, *, beta=None, reference_a=100, reference_b=200, sigma=.2):
    beta = [math.log(2), 0, 0, 0, 0, 0] if beta is None else beta
    info = dict(features=list(f.FEATURES), coefficients=beta,
        feature_schema=f.SCHEMA, response_domain=m.DOMAIN, numerical_qualification=True,
        lognormal_reference_sigma=sigma, baselines=[
            dict(group=[owner, owner, 'kernel', 'context', 'synthetic-nsys'], log_reference_ms=math.log(ref))
            for owner, ref in [('a', reference_a), ('b', reference_b)]])
    with patch.object(f, 'fit_response', return_value=(info, None, None, None)):
        return m.training_bundle(p, [capture(f.KernelCompetitionFeatureProvider(p))], allow_cpu_fixture=True)


def response(p=None, document=None, **options):
    p = provider().provider if p is None else p
    document = bundle(p) if document is None else document
    scenarios = options.pop('scenarios', {
        'mean': dict(catalog_scenario='mean', reference_functional='mean'),
        'median': dict(catalog_scenario='mean', reference_functional='quantile', quantile=.5)})
    return m.KernelCompetitionResponseProvider(p, document, scenarios=scenarios,
        allow_unvalidated_profiled_response=True, allow_cpu_fixture=True, **options)


def request(name, arrival=0):
    return dict(request_id=name, task='task', pool_id='input', arrival_ns=arrival)


def active(p, aliases=ALIASES, scenario='median'):
    rows = {}
    for alias in aliases:
        row = dict(request(alias), alias=alias)
        plan = p.plan(row, aliases, scenario)
        seg = plan['segments'][0]
        row.update(profile_id=plan['profile_id'], segment_id=seg['id'], segment_kind=seg['kind'],
                   segment_duration_ms=seg['duration_ms'])
        rows[alias] = row
    return rows


class KernelResponseProviderTests(unittest.TestCase):
    def test_finite_but_extreme_sigma_stays_unknown_without_overflow(self):
        catalog = provider().provider
        p = response(catalog, bundle(catalog, sigma=1e200))
        with self.assertRaisesRegex(UnknownEvidence, 'NONFINITE_FITTED_KERNEL_REFERENCE'):
            p.plan(dict(request('a'), alias='a'), ALIASES, 'mean')

    def test_mean_reference_is_exp_mu_plus_half_variance_not_median(self):
        p = response()
        rows = active(p, scenario='mean')
        value = p.service(rows['a'], rows, ALIASES, 'mean')
        self.assertAlmostEqual(value['reference_clock_ms'], 100 * math.exp(.2**2 / 2))
        self.assertAlmostEqual(value['service_ms'], 200 * math.exp(.2**2 / 2))
        median = active(p)
        self.assertAlmostEqual(p.service(median['a'], median, ALIASES, 'median')['service_ms'], 200)
        self.assertFalse(value['joint_request_distribution_identified'])
        self.assertFalse(value['static_occupancy_used_as_throughput'])

    def test_existing_engine_generates_peer_completion_and_conserves_reference_work(self):
        p = response()
        with patch.object(p.catalog, 'service', side_effect=AssertionError('no occupancy speed')):
            result = FullHorizonModel(ALIASES, p).forecast(
                {a: [request(a)] for a in ALIASES}, 0, 1_000_000_000, 'median', require_energy=False)
        self.assertEqual(result['finishes']['a']['finish_ns'], 200_000_000)
        self.assertEqual(result['finishes']['b']['finish_ns'], 300_000_000)
        intervals = [r for r in result['transitions'] if r['active_requests']]
        self.assertAlmostEqual(intervals[0]['service_ms']['b'], 400)
        self.assertAlmostEqual(intervals[1]['service_ms']['b'], 200)

    def test_future_arrival_changes_active_kernel_set_and_fifo_wait_without_future_end(self):
        p = response()
        future = [dict(request('b', 50_000_000), alias='b')]
        with patch.object(p.catalog, 'service', side_effect=AssertionError('no analytical service')):
            result = FullHorizonModel(ALIASES, p).forecast(
                {'a': [request('a0'), request('a1')], 'b': []},
                0, 1_000_000_000, 'median', future_arrivals=future,
                route_future=lambda row, queues, clock: row['alias'], require_energy=False)
        self.assertEqual(result['finishes']['a0']['finish_ns'], 150_000_000)
        self.assertEqual(result['finishes']['a1']['start_ns'], 150_000_000)
        self.assertEqual(result['finishes']['a1']['finish_ns'], 350_000_000)
        self.assertEqual(result['finishes']['b']['finish_ns'], 400_000_000)
        self.assertEqual(len(result['finishes']), 3)

    def test_cursor_conserves_observed_remaining_reference_work(self):
        p = response()
        row = request('a')
        row['forecast_cursor'] = dict(profile_id='a', segment_index=0, remaining_fraction=.25,
                                     observed_at_ns=0, evidence='synthetic-current-progress')
        result = FullHorizonModel(ALIASES, p).forecast(
            {'a': [row], 'b': []}, 0, 1_000_000_000, 'median', require_energy=False)
        self.assertEqual(result['finishes']['a']['finish_ns'], 25_000_000)
        row.pop('forecast_cursor')
        row['head_age_ms'] = 5
        with self.assertRaisesRegex(UnknownEvidence, 'OBSERVED_SEGMENT_PROGRESS_REQUIRED'):
            FullHorizonModel(ALIASES, p).forecast(
                {'a': [row]}, 0, 1_000_000_000, 'median', require_energy=False)

    def test_target_future_end_is_not_allowed_by_original_forecast_request_gate(self):
        row = request('a')
        row['nsys_end_ns'] = 1
        with self.assertRaisesRegex(ValueError, 'no labels or outputs'):
            FullHorizonModel(ALIASES, response()).forecast(
                {'a': [row]}, 0, 1_000_000_000, 'median', require_energy=False)

    def test_failed_real_structural_fit_is_retained_and_cannot_forecast(self):
        catalog = provider().provider
        document = m.training_bundle(catalog, [capture(f.KernelCompetitionFeatureProvider(catalog))],
                                     allow_cpu_fixture=True)
        self.assertEqual(document['response_fit']['reason'], 'STRUCTURAL_KERNEL_COVARIATE_RANK_FAILURE')
        p = response(catalog, document)
        with self.assertRaisesRegex(UnknownEvidence, 'NUMERICALLY_QUALIFIED'):
            p.plan(dict(request('a'), alias='a'), ALIASES, 'median')

    def test_opt_in_catalog_context_and_bundle_mutation_gates(self):
        catalog = provider().provider
        document = bundle(catalog)
        p = m.KernelCompetitionResponseProvider(catalog, document, allow_cpu_fixture=True)
        with self.assertRaisesRegex(UnknownEvidence, 'OPT_IN'):
            p.plan(dict(request('a'), alias='a'), ALIASES, 'mean')
        with self.assertRaisesRegex(ValueError, 'fixture'):
            m.KernelCompetitionResponseProvider(catalog, document)
        altered = copy.deepcopy(document)
        altered['response_fit']['coefficients'][0] += 1
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            response(catalog, altered)
        foreign = copy.deepcopy(catalog.document)
        foreign['profiles'][0]['execution']['backend'] = 'changed-backend'
        other = KernelResourceProvider(foreign, 'context', allow_cpu_fixture=True)
        with self.assertRaisesRegex(ValueError, 'catalog/context/hardware'):
            response(other, document)

    def test_missing_baseline_and_unsupported_static_peer_remain_unknown(self):
        catalog = provider().provider
        document = bundle(catalog)
        document['response_fit']['baselines'].pop()
        document.pop('bundle_sha256')
        document['bundle_sha256'] = identity(document)
        p = response(catalog, document)
        with self.assertRaisesRegex(UnknownEvidence, 'UNKNOWN_OWNER_PROFILE'):
            active(p)
        p = response()
        rows = active(p)
        with patch.object(p.features, 'features', return_value=dict(known=False, reason='STATIC_CONFIGURATION_UNKNOWN')):
            self.assertFalse(p.service(rows['a'], rows, ALIASES, 'median')['known'])
        p.catalog.segments['b', 'kernel']['kernel'].pop('allocated_registers_per_block')
        with self.assertRaisesRegex(ValueError, 'catalog/context changed'):
            p.plan(dict(request('a'), alias='a'), ALIASES, 'median')

    def test_catalog_snapshot_and_boundaries_do_not_copy_whole_plans(self):
        catalog = provider().provider
        p = response(catalog)
        rows = active(p)
        catalog.document['profiles'][0]['segments'][0]['kernel']['name'] = 'changed-external-catalog'
        with patch.object(p, 'plan', side_effect=AssertionError('whole-plan rebuild at kernel boundary')):
            self.assertTrue(p.service(rows['a'], rows, ALIASES, 'median')['known'])

    def test_host_and_gap_need_explicit_provider_even_when_gpu_fit_exists(self):
        catalog = provider().provider
        doc = copy.deepcopy(catalog.document)
        doc['profiles'][0]['segments'].append(dict(id='host', kind='host', duration_ms=dict(mean=10)))
        other = KernelResourceProvider(doc, 'context', allow_cpu_fixture=True)
        p = response(other)
        with self.assertRaisesRegex(UnknownEvidence, 'NON_GPU'):
            p.plan(dict(request('a'), alias='a'), ALIASES, 'median')
        with self.assertRaisesRegex(ValueError, 'Non-GPU response execution'):
            response(other, non_gpu_provider=object())

    def test_scenarios_are_predeclared_and_quantile_bound(self):
        for scenario in ({'catalog_scenario': 'mean', 'reference_functional': 'quantile', 'quantile': 1},
                         {'catalog_scenario': 'mean', 'reference_functional': 'mean', 'quantile': .5}):
            with self.subTest(scenario=scenario), self.assertRaises(ValueError):
                response(scenarios={'bad': scenario})
        p = response()
        with self.assertRaisesRegex(UnknownEvidence, 'SCENARIO'):
            p.plan(dict(request('a'), alias='a'), ALIASES, 'unknown')


if __name__ == '__main__':
    unittest.main()
