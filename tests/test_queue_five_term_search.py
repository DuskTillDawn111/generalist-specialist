import copy
import unittest
from unittest.mock import patch

from tools.queue_five_term_20261023 import search as module
from tools.queue_five_term_20261023.policy import FIXED_WEIGHTS, OBJECTIVE_SCHEMA
from tests.test_queue_coupled_search import Provider


class FiveTermSearchTests(unittest.TestCase):
    def contract(self):
        return dict(deployment={'text':2,'generalist_2':1},aliases={
            'text#0':dict(family='text',supported_tasks=['text']),
            'text#1':dict(family='text',supported_tasks=['text']),
            'generalist_2#0':dict(family='generalist_2',supported_tasks=['text'])},
            text_rate=1.85,quality_loss={'text':{'text':0.},'generalist_2':{'text':.02}},
            weights=dict(FIXED_WEIGHTS),latency_refs_ms={'text':100.},background_routing={},horizon_seconds=1.)

    def cohort(self):
        return [dict(request_id=str(i),task='text',pool_id='x',arrival_ns=i*5_000_000) for i in range(6)]

    def test_new_objective_domain_and_frozen_sources_are_separate(self):
        items=list(module.candidates(self.contract()['deployment']))
        self.assertEqual(len(items),24)
        self.assertEqual(len({module.identity(c) for c in items}),24)
        self.assertTrue(all(c['objective_schema']==OBJECTIVE_SCHEMA and c['weights']==FIXED_WEIGHTS for c in items))
        self.assertTrue(module.CoupledUtilityPolicy.__module__.startswith('tools.queue_five_term_20261023.'))
        self.assertTrue(module.SharedRateEnvironment.__module__.startswith('tools.queue_coupled_utility_20261023.'))

    def test_real_simulation_excludes_P_from_score_and_retains_diagnostic(self):
        contract=self.contract();candidate=next(module.candidates(contract['deployment']))
        first=module.evaluate_cohort(contract,{},self.cohort(),candidate,provider=Provider(),memory_norm=.8)
        with patch.object(module,'percentile',return_value=1.e12):
            huge_tail=module.evaluate_cohort(contract,{},self.cohort(),candidate,provider=Provider(),memory_norm=.8)
        self.assertEqual(first['U'],huge_tail['U'])
        self.assertNotEqual(first['terms']['P'],huge_tail['terms']['P'])
        self.assertEqual(first['route_trace_sha256'],huge_tail['route_trace_sha256'])
        self.assertEqual(first['completion_trace_sha256'],huge_tail['completion_trace_sha256'])
        self.assertEqual(first['weighted_terms'],list('QTLME'))
        self.assertTrue(first['actual_new_policy_simulated'])
        self.assertFalse(first['old_score_table_reused'])
        self.assertEqual(first['P_evaluation_role'],'diagnostic_only')
        expected=-sum(.2*first['terms'][term] for term in 'QTLME')
        self.assertAlmostEqual(first['U'],expected)

    def test_old_weights_are_rejected_before_any_simulation(self):
        contract=self.contract();candidate=next(module.candidates(contract['deployment']))
        contract['weights']={term:1/6 for term in 'QTLPME'}
        with self.assertRaisesRegex(ValueError,'five-term weights'):
            module.evaluate_cohort(contract,{},self.cohort(),candidate,provider=Provider())
        old=copy.deepcopy(candidate);old['weights']['P']=.2
        with self.assertRaisesRegex(ValueError,'five-term weights'):
            module.search([old],(413,414),lambda *args:None)

    def test_exhaustive_and_quotient_run_new_policies_and_match_every_result(self):
        contract=self.contract();calls=[]
        def evaluate(candidate,seed):
            calls.append((module.identity(candidate),seed))
            return module.evaluate_cohort(contract,{},self.cohort(),candidate,provider=Provider(),memory_norm=.8)
        dfs,optimized,summary=module.compare_domain(list(module.candidates(contract['deployment'])),(413,414),evaluate)
        self.assertEqual(dfs['oracle_calls'],48);self.assertEqual(optimized['oracle_calls'],28)
        self.assertEqual(len(calls),76)
        self.assertTrue(summary['same_nonempty_optimum_and_all_ties'])
        self.assertFalse(summary['empty_result_sets_agree_without_optimality_claim'])
        self.assertEqual(len(summary['full_run_crosschecks']),24)
        self.assertEqual(summary['strict_bound_candidates_pruned'],0)
        self.assertFalse(summary['physical_optimality'])

    def test_missing_background_preserved_unknown_not_empty_optimum_claim(self):
        class Missing(Provider):
            def power(self,*args):return dict(known=False,watts=None,status='UNKNOWN_BACKGROUND_POWER')
        contract=self.contract();calls=[]
        def evaluate(candidate,seed):
            calls.append(seed)
            return module.evaluate_cohort(contract,{},self.cohort(),candidate,provider=Missing())
        _,_,summary=module.compare_domain(list(module.candidates(contract['deployment'])),(413,414),evaluate)
        self.assertFalse(summary['nonempty_optimum_identified'])
        self.assertFalse(summary['same_nonempty_optimum_and_all_ties'])
        self.assertTrue(summary['empty_result_sets_agree_without_optimality_claim'])
        self.assertFalse(summary['whole_declared_cpu_domain_certified'])
        self.assertEqual(summary['unknown_candidate_count'],24)
        self.assertEqual(sum(summary['unknown_run_reason_counts'].values()),48)

    def test_existing_observational_coverage_is_not_silently_activated(self):
        from tools.queue_coupled_utility_20261023.resources import SCHEMA,key
        historical=dict(schema=SCHEMA,reserved_seeds_used=False,quality_used=False,training_seeds=[113],
            minimum_power_seconds=.25,minimum_service_observations=2,input_prior=16.,context_semantics='host-pending only',
            deployments={key(dict(sorted(module.PRIMARY.items()))):dict(runs=[],service_strata={'task_context':{},'input_context':{}},power_states={})})
        audit=module.coverage_audit({'schema':'controlled_text_only','training_seeds':[301,302]},historical)
        self.assertFalse(audit['full_workload_candidate_selection_supported'])
        self.assertFalse(audit['hybrid_provider_used'])
        inventory=audit['historical_observational_inventory']
        self.assertFalse(inventory['may_fill_controlled_actual_overlap_gap'])
        self.assertEqual(len(inventory['missing_service_contexts']),18)
        self.assertEqual(len(inventory['power_states']),12)


if __name__=='__main__':unittest.main()
