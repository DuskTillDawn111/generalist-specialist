import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.kernel_provider import KernelResourceProvider, KERNEL_PROFILE_SCHEMA
from gsdeploy.forecasting.simulator import FullHorizonModel
from tests.test_kernel_resources import hardware, kernel


ALIASES = {name: dict(family=name, supported_tasks=['task']) for name in ('a', 'b')}


def document(shared=0):
    return dict(schema=KERNEL_PROFILE_SCHEMA, execution_context_sha256='context',
                provenance=dict(kind='cpu_fixture', split='training', sources=['synthetic-only']),
                hardware=hardware(), concurrency_evidence='synthetic-capability',
                policy='equal_block_round_robin', profiles=[
                    dict(id=name, family=name, task='task', pool_id='input',
                         execution=dict(batch_size=1, backend='synthetic', weight_reuse=False),
                         segments=[dict(id='kernel', kind='gpu', duration_ms=dict(mean=100 if name == 'a' else 200),
                                        kernel=kernel(shared=shared), solo_timing_basis='synthetic',
                                        timing_evidence='synthetic-isolated-timing')]) for name in ALIASES])


def provider(doc=None, **options):
    return KernelResourceProvider(document() if doc is None else doc, 'context',
                                  allow_cpu_fixture=True, allow_analytical_hypothesis=True, **options)


def active_rows(p):
    rows = {}
    for name in ALIASES:
        row = dict(alias=name, task='task', pool_id='input', request_id=name, arrival_ns=0)
        plan = p.plan(row, ALIASES, 'mean'); seg = plan['segments'][0]
        row.update(profile_id=plan['profile_id'], segment_id=seg['id'], segment_kind=seg['kind'],
                   segment_duration_ms=seg['duration_ms'])
        rows[name] = row
    return rows


class KernelProviderTests(unittest.TestCase):
    def test_fixture_training_context_and_opt_in_gates(self):
        with self.assertRaises(ValueError):
            KernelResourceProvider(document(), 'context')
        doc = document(); doc['provenance']['split'] = 'validation'
        with self.assertRaises(ValueError): provider(doc)
        with self.assertRaises(ValueError):
            KernelResourceProvider(document(), 'other', allow_cpu_fixture=True)
        p = KernelResourceProvider(document(), 'context', allow_cpu_fixture=True)
        rows = active_rows(p)
        self.assertFalse(p.service(rows['a'], rows, ALIASES, 'mean')['known'])
        self.assertFalse(p.physical_evidence)
        doc = document(); doc['provenance']['kind'] = 'measured'
        for profile in doc['profiles']:
            profile['segments'][0]['solo_timing_basis'] = 'unprofiled_isolated'
        measured_static = KernelResourceProvider(doc, 'context', allow_analytical_hypothesis=True)
        self.assertFalse(measured_static.physical_evidence)

    def test_plan_is_ordered_and_configuration_bound_without_throughput_claim(self):
        p = provider(); rows = active_rows(p)
        plan = p.plan(rows['a'], ALIASES, 'mean')
        self.assertEqual(plan['segments'][0]['demand'], dict(sm=None, dram=None, l2=None))
        self.assertEqual(p.demand_summary('a')['classification'], 'unknown')
        bad = {**rows['a'], 'pool_id': 'other'}
        with self.assertRaises(UnknownEvidence): p.plan(bad, ALIASES, 'mean')
        with self.assertRaises(UnknownEvidence): p.plan(rows['a'], ALIASES, 'unknown')
        bad = copy.deepcopy(rows); bad['a']['segment_duration_ms'] = 1
        with self.assertRaises(ValueError): p.service(bad['a'], bad, ALIASES, 'mean')

    def test_service_reprices_the_same_block_allocation_after_neighbor_completion(self):
        p = provider(); rows = active_rows(p)
        together = p.service(rows['b'], rows, ALIASES, 'mean')
        solo = p.service(rows['b'], {'b': rows['b']}, ALIASES, 'mean')
        self.assertEqual(together['service_ms'], 400)
        self.assertEqual(solo['service_ms'], 200)
        self.assertFalse(together['physically_validated'])

    def test_host_independent_and_gpu_unknown_for_missing_static_or_timing(self):
        doc = document(); doc['profiles'][0]['segments'][0]['kind'] = 'host'
        doc['profiles'][1]['segments'][0]['kernel'].pop('allocated_registers_per_block')
        p = provider(doc); rows = active_rows(p)
        self.assertEqual(p.service(rows['a'], rows, ALIASES, 'mean')['service_ms'], 100)
        self.assertFalse(p.service(rows['b'], rows, ALIASES, 'mean')['known'])
        doc = document(); doc['profiles'][0]['segments'][0]['solo_timing_basis'] = 'profiled_kernel'
        p = provider(doc); rows = active_rows(p)
        self.assertFalse(p.service(rows['a'], rows, ALIASES, 'mean')['known'])

    def test_small_grid_and_concurrency_evidence_are_not_fabricated(self):
        for change in ('small_grid', 'no_concurrency'):
            doc = document()
            if change == 'small_grid': doc['profiles'][0]['segments'][0]['kernel']['grid'] = [1, 1, 1]
            else: doc['concurrency_evidence'] = None
            p = provider(doc); rows = active_rows(p)
            self.assertFalse(p.service(rows['a'], rows, ALIASES, 'mean')['known'])

    def test_profiled_isolated_baseline_needs_explicit_domain_and_opt_in(self):
        doc = document()
        for profile in doc['profiles']:
            profile['segments'][0].update(solo_timing_basis='profiled_isolated',
                profiler_id='nsys-cuda-light-training', profiler_perturbed=True,
                perturbation_flags=['CUPTI observation; independent natural validation required'])
        p = provider(doc); rows = active_rows(p)
        self.assertEqual(p.service(rows['a'], rows, ALIASES, 'mean')['reason'], 'PROFILED_BASELINE_OPT_IN_REQUIRED')
        p = provider(doc, allow_profiled_baselines=True); rows = active_rows(p)
        result = p.service(rows['a'], rows, ALIASES, 'mean')
        self.assertTrue(result['known'])
        self.assertEqual(result['timing_domains']['a']['basis'], 'profiled_isolated')
        self.assertFalse(result['natural_timing_validated'])
        self.assertFalse(p.physical_evidence)
        model = FullHorizonModel(ALIASES, p)
        queues = {name: [dict(request_id=name, task='task', pool_id='input', arrival_ns=0)] for name in ALIASES}
        predicted = model.forecast(queues, 0, 1000000000, require_energy=False)
        self.assertEqual(predicted['finishes']['b']['finish_ns'], 300000000)
        doc['profiles'][0]['segments'][0].pop('profiler_id')
        p = provider(doc, allow_profiled_baselines=True); rows = active_rows(p)
        self.assertEqual(p.service(rows['a'], rows, ALIASES, 'mean')['reason'], 'EXPLICIT_PROFILER_DOMAIN_REQUIRED')

    def test_zero_blocks_explicitly_stall_and_neighbor_keeps_progress(self):
        p = provider(document(shared=40000)); rows = active_rows(p)
        a = p.service(rows['a'], rows, ALIASES, 'mean')
        b = p.service(rows['b'], rows, ALIASES, 'mean')
        self.assertEqual(a['service_ms'], 100)
        self.assertTrue(b['known']); self.assertTrue(b['stalled']); self.assertIsNone(b['service_ms'])

    def test_power_and_memory_are_unknown_or_independently_delegated(self):
        p = provider(); rows = active_rows(p)
        self.assertFalse(p.power(rows, ALIASES, 'mean')['known'])
        self.assertFalse(p.memory(rows, ALIASES, 'mean')['known'])
        class Independent:
            def power(self, *args): return dict(known=True, watts=37, evidence='independent')
            def memory(self, *args): return dict(known=True, peak_memory_mib=123, evidence='independent')
        p = provider(power_provider=Independent(), memory_provider=Independent())
        self.assertEqual(p.power(rows, ALIASES, 'mean')['watts'], 37)
        self.assertEqual(p.memory(rows, ALIASES, 'mean')['peak_memory_mib'], 123)

    def test_full_horizon_real_interface_conserves_remaining_kernel_work(self):
        p = provider(); model = FullHorizonModel(ALIASES, p)
        queues = {name: [dict(request_id=name, task='task', pool_id='input', arrival_ns=0)] for name in ALIASES}
        result = model.forecast(queues, 0, 1000000000, require_energy=False)
        self.assertEqual(result['finishes']['a']['finish_ns'], 200000000)
        self.assertEqual(result['finishes']['b']['finish_ns'], 300000000)

    def test_full_horizon_stalled_kernel_resumes_when_neighbor_finishes(self):
        p = provider(document(shared=40000)); model = FullHorizonModel(ALIASES, p)
        queues = {name: [dict(request_id=name, task='task', pool_id='input', arrival_ns=0)] for name in ALIASES}
        result = model.forecast(queues, 0, 1000000000, require_energy=False)
        self.assertEqual(result['finishes']['a']['finish_ns'], 100000000)
        self.assertEqual(result['finishes']['b']['finish_ns'], 300000000)
        self.assertIsNone(result['transitions'][0]['service_ms']['b'])


if __name__ == '__main__': unittest.main()
