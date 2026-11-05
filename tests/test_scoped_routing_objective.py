import copy
import unittest
from unittest.mock import patch

from experiments.scoped_routing_objective import candidate_plan, predict_candidate, scoped_reference


TASKS = ('asr', 'image', 'ocr', 'text')


class ScopedObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.system = dict(workload=dict.fromkeys(TASKS, 1.), gpu_memory_mib=100.,
            latency_reference_ms=dict.fromkeys(TASKS, 100.),
            weights=dict.fromkeys(('Q', 'T', 'L', 'P', 'M', 'E'), 1/6),
            catalog=dict(safe_factor=1.110854, utility_normalization=dict(
                schema='six_term_reference_v2', status='synthetic_fixture', provenance={'test': True},
                workload_rps=dict.fromkeys(TASKS, 1.), horizon_seconds=1800., memory_budget_mib=100.,
                reference_receipts=['test'], episode_reference_joules=1000.,
                episode_protocol={'original': True},
                tasks={t: dict(best_loss=0., worst_loss=1., deadline_ms=100., provenance={'test': True}) for t in TASKS})))
        aliases = {m+'#0': m for m in (*TASKS, 'generalist_2')}
        requests = [dict(task=t, pool_id=str(i), offset_ms=i*100., alias=t+'#0') for i in range(2) for t in TASKS]
        self.plan = dict(execution_context_sha256='context', aliases=aliases,
            jobs=[dict(id='single', duration_ms=1000., requests=requests, design={})])
        self.fit = dict(profiles={'execution_context_sha256': 'context'})
        self.evidence = dict(execution_context_sha256='context', resident_aliases=aliases,
            raw_peak_estimate_mib=50., capacity_service_ms={m: {t: 1. for t in TASKS if m in (t, 'generalist_2')}
                                                         for m in aliases.values()},
            quality=[dict(family=m, task=t, pool_id=str(i), quality_totals=[i, 1 if i == 0 else 9])
                     for m in aliases.values() for t in TASKS if m in (t, 'generalist_2') for i in range(2)])
        self.routing = {t: {t: 1., 'generalist_2': 1.} for t in TASKS}

    @staticmethod
    def forecast(fit, plan):
        return dict(complete=True, predicted_episode_board_joules=100.,
            jobs=[dict(requests=[dict(r, predicted_e2e_ms=50.) for r in plan['jobs'][0]['requests']])])

    def test_original_six_terms_weighted_quality_and_raw_memory_preserved(self):
        with patch('experiments.scoped_routing_objective.predict_plan', self.forecast):
            result = predict_candidate(self.fit, self.plan, self.routing, self.evidence, self.system)
        self.assertAlmostEqual(result['Q_norm'], .1)  # ratio of totals, not mean(0/1, 1/9)
        self.assertEqual(result['M_norm'], .5)
        self.assertAlmostEqual(result['safe_memory_mib'], 50*1.110854)
        self.assertEqual(result['E_norm'], .1)
        self.assertEqual(result['T_norm'], 0.)
        self.assertAlmostEqual(result['U'], -(.1+0+.5+.5+.5+.1)/6)
        self.assertFalse(result['candidate_ranking_qualified'])

    def test_reference_rebinding_preserves_numbers_and_original_source(self):
        before = copy.deepcopy(self.system)
        data, rates = scoped_reference(self.system, self.plan)
        spec = data['utility_normalization']
        self.assertEqual(self.system, before)
        self.assertEqual(spec['tasks'], self.system['catalog']['utility_normalization']['tasks'])
        self.assertEqual(spec['episode_reference_joules'], 1000.)
        self.assertEqual(spec['horizon_seconds'], 1.)
        self.assertEqual(rates, dict.fromkeys(TASKS, 2.))
        self.assertFalse(spec['provenance']['scoped_reference_reuse']['new_reference_measurement'])

    def test_same_offered_cohort_original_router_and_residency(self):
        routed = candidate_plan(self.plan, self.routing, dict.fromkeys(TASKS, 2.))
        self.assertEqual(routed['aliases'], self.plan['aliases'])
        for old, new in zip(self.plan['jobs'][0]['requests'], routed['jobs'][0]['requests']):
            self.assertEqual({k: old[k] for k in ('task', 'pool_id', 'offset_ms')},
                             {k: new[k] for k in ('task', 'pool_id', 'offset_ms')})
        for t in TASKS:
            self.assertEqual({r['alias'] for r in routed['jobs'][0]['requests'] if r['task'] == t},
                             {t+'#0', 'generalist_2#0'})

    def test_no_multi_route_episode_or_unconserved_route(self):
        bad = copy.deepcopy(self.plan)
        bad['jobs'] *= 2
        with self.assertRaises(ValueError):
            scoped_reference(self.system, bad)
        bad_route = copy.deepcopy(self.routing)
        bad_route['text']['text'] += 1.
        with self.assertRaises(ValueError):
            candidate_plan(self.plan, bad_route, dict.fromkeys(TASKS, 2.))

    def test_original_memory_and_rho_limits_remain_active(self):
        with patch('experiments.scoped_routing_objective.predict_plan', self.forecast):
            self.evidence['raw_peak_estimate_mib'] = 91.
            self.assertFalse(predict_candidate(self.fit, self.plan, self.routing, self.evidence, self.system)['constraints_eligible'])
            self.evidence['raw_peak_estimate_mib'] = 50.
            self.evidence['capacity_service_ms']['generalist_2']['asr'] = 801.
            self.assertFalse(predict_candidate(self.fit, self.plan, self.routing, self.evidence, self.system)['constraints_eligible'])


if __name__ == '__main__':
    unittest.main()
