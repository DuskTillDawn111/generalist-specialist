"""CPU-only finite-domain contracts; synthetic resources are not native evidence."""
import copy
from dataclasses import replace
from fractions import Fraction as F
import json
import math
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_forecast_ensemble_v1 import arguments
from tests.test_actual_worker_twoFIFO_reference_v4 import decode
from tools.predictor_validation_20261027 import twoFIFO_forecast_ensemble_v1 as ensemble
from tools.predictor_validation_20261027 import twoFIFO_candidate_ranking_v1 as module


def generated(owner='owner', *, drain=None):
    requests, draws, args = arguments()
    if owner != 'owner':
        for row in requests:
            row['owner'] = owner
        joint = draws[0]['inputs']; old_plans = joint['plans']; remapped = {}
        for plans in old_plans.values():
            for plan in plans:
                remapped[plan] = replace(plan, ready_producer_id=owner,
                    macros=tuple(replace(macro, marks=tuple(replace(mark, producer_id=owner)
                        for mark in macro.marks)) for macro in plan.macros))
        joint['plans'] = {(owner,*key[1:]):tuple(remapped[p] for p in plans)
                          for key,plans in old_plans.items()}
        joint['rates'] = {(owner,*key[1:]):value for key,value in joint['rates'].items()}
        joint['protocol_slacks'] = {remapped[p]:value for p,value in joint['protocol_slacks'].items()}
        joint['transport_draws'] = {(owner,*key[1:]):value for key,value in joint['transport_draws'].items()}
        for field in ('initial_sender_available_ns','initial_worker_read_release_ns'):
            args[field] = {owner:value for value in args[field].values()}
        args['producer_fifo'] = tuple(owner for _ in args.get('producer_fifo', ()))
        args['scoring_plan']['aliases'] = {owner:args['scoring_plan']['aliases']['owner']}
    if drain is not None:
        args['drain_end_ns'] = drain
    draws[0]['weight'] = F(1,3)
    draws.append(dict(copy.deepcopy(draws[0]), draw_id='draw1', block_id='training-block-2', weight=F(2,3)))
    return ensemble.evaluate(requests, draws, **args)


def supply_synthetic_resources(document, memory):
    """Isolate ranking arithmetic, retaining all real CPU timing terms."""
    document = copy.deepcopy(document)
    for draw in document['draws']:
        plan=document['score_comparison_contract']['scoring_plan']
        raw_memory=memory*plan['memory_budget_mib']/plan['memory_safe_factor']
        prediction=dict(schema='gsdeploy.twoFIFO_multiwindow_resource_prediction.v3',
            board_energy_joules=0.,partial_known_peak_memory_mib=raw_memory,
            startup_energy_joules=0.,premeasurement_owned_sampled_peak_mib=raw_memory,
            memory_unknown_observation_duration_ns=0,startup_energy_already_amortized=False,
            safe_factor_already_applied=False,power_unknown_measurement_duration_ns=0,
            total_measurement_duration_ns=2,total_observation_duration_ns=document['original_drain_end_ns'])
        draw['forecast']['scoring']['resource_prediction']=prediction
        score = draw['forecast']['scoring']['original_score']
        score['terms'].update(M=plan['memory_safe_factor']*raw_memory/plan['memory_budget_mib'],E=0.)
        complete = all(v is not None for v in score['terms'].values())
        score.update(six_term_comparison_complete=complete, absolute_utility_complete=complete,
            cost=sum(v/6 for v in score['terms'].values()) if complete else None)
        score['full_U'] = -score['cost'] if complete else None
        draw['forecast']['scoring']['original_score_bridge']['score'] = copy.deepcopy(score)
    return document


def change_score(document, transform, draw=0):
    score = document['draws'][draw]['forecast']['scoring']['original_score']
    transform(score)
    document['draws'][draw]['forecast']['scoring']['original_score_bridge']['score'] = copy.deepcopy(score)


class CandidateRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generated = generated()
        cls.known = supply_synthetic_resources(cls.generated, 1.)

    def test_three_real_cpu_routes_share_draws_and_rank_original_equal_six_cost(self):
        candidates = {'slow':supply_synthetic_resources(self.generated,3.),
            'best':supply_synthetic_resources(generated('second'),1.),
            'middle':supply_synthetic_resources(generated('third'),2.)}
        before = copy.deepcopy(candidates)
        output = decode(module.rank_candidates(candidates))
        self.assertEqual(candidates,before)
        self.assertEqual(output['winner_candidate_ids'],['best'])
        self.assertEqual([g['candidate_ids'] for g in output['complete_domain_ranking']],
                         [['best'],['middle'],['slow']])
        self.assertEqual(output['candidate_count'],3)
        self.assertTrue(all(row['offered_requests']==2 for row in output['candidates']))
        self.assertFalse(output['final_gain_eligible'])

    def test_all_pairs_reuse_exact_original_compare_outputs(self):
        candidates = {str(i):supply_synthetic_resources(self.generated,i) for i in range(4)}
        with patch.object(module.ensemble,'compare', wraps=ensemble.compare) as compare:
            result = module.rank_candidates(candidates)
        self.assertEqual(compare.call_count,10)  # four self validation plus six pairs
        for pair in result['paired_comparisons']:
            self.assertEqual(pair['comparison'],ensemble.compare(candidates[pair['baseline_id']],
                                                               candidates[pair['candidate_id']]))

    def test_unknown_world_and_missing_generated_resources_keep_full_mass(self):
        unknown = copy.deepcopy(self.known)
        unknown['draws'][1].update(status='UNKNOWN_WORLD',forecast=None,
            all_public_request_ids_retained=['r0','r1'],reason='synthetic missing evidence')
        output = decode(module.rank_candidates({'complete':self.known,
            'unknown_world':unknown,'missing_M_E':self.generated}))
        rows = {r['candidate_id']:r for r in output['candidates']}
        self.assertEqual(rows['unknown_world']['known_utility_probability_mass'],F(1,3))
        self.assertEqual(rows['unknown_world']['unknown_utility_probability_mass'],F(2,3))
        self.assertEqual(rows['missing_M_E']['unknown_utility_probability_mass'],1)
        self.assertIsNone(rows['unknown_world']['complete_utility'])
        self.assertIsNotNone(rows['unknown_world']['conditional_on_known_utility_mean'])
        self.assertEqual(rows['missing_M_E']['per_draw_term_status'][0]['terms']['M'],'UNKNOWN')
        self.assertEqual(output['complete_domain_ranking_status'],'UNKNOWN')
        self.assertEqual(output['winner_status'],'UNKNOWN')
        self.assertIsNone(output['complete_domain_ranking'])
        self.assertIsNone(output['winner_candidate_ids'])
        self.assertEqual(output['complete_subdomain_ranking_diagnostic'][0]['candidate_ids'],['complete'])
        self.assertFalse(output['conditional_means_used_for_ranking'])
        pair = output['paired_comparisons'][0]['comparison']
        self.assertEqual(pair['unknown_delta_probability_mass'],F(2,3))
        self.assertIsNone(pair['candidate_better'])
        self.assertEqual(pair['positive_delta_probability_bounds'],[0,F(2,3)])

    def test_all_unknown_keeps_empty_subdomain_and_no_winner(self):
        output=decode(module.rank_candidates({'a':self.generated,'b':self.generated}))
        self.assertEqual(output['complete_subdomain_ranking_diagnostic'],[])
        self.assertIsNone(output['winner_candidate_ids'])

    def test_score_terms_and_full_U_cannot_be_forged_or_reweighted(self):
        mutations = {
            'missing_term':lambda s:s['terms'].pop('M'),
            'nan_term':lambda s:s['terms'].update(M=math.nan),
            'bool_term':lambda s:s['terms'].update(M=True),
            'wrong_cost':lambda s:s.update(cost=0.),
            'wrong_U':lambda s:s.update(full_U=0.),
            'unknown_M_but_U_known':lambda s:s['terms'].update(M=None),
            'conditional_U':lambda s:s.update(absolute_utility_complete=False),
            'completion_flag':lambda s:s.update(six_term_comparison_complete=False),
            'cohort_size':lambda s:s.update(cohort_size=1),
            'cohort_ids':lambda s:s.update(cohort_ids=['r0']),
            'incomplete_ids':lambda s:s.update(incomplete_ids=['r0']),
            'T_subset':lambda s:s['terms'].update(T=.5),
            'P_completed_subset':lambda s:s['terms'].update(P=0.),
            'L_wrong_endpoint':lambda s:s['terms'].update(L=0.),
        }
        for name,mutation in mutations.items():
            bad=copy.deepcopy(self.known);change_score(bad,mutation)
            with self.subTest(name=name),self.assertRaises(ValueError):
                module.rank_candidates({'only':bad})

    def test_fixed_drain_miss_and_missing_quality_are_not_dropped_or_zero_filled(self):
        incomplete=generated(drain=5)
        output=decode(module.rank_candidates({'short':incomplete}))
        row=output['candidates'][0]
        self.assertEqual(row['offered_requests'],2)
        self.assertTrue(all(d['terms']['P']=='UNKNOWN' for d in row['per_draw_term_status']))
        self.assertIsNone(row['complete_utility'])
        self.assertTrue(output['mean_of_world_P95_is_not_mixture_latency_P95'])
        bad=copy.deepcopy(incomplete)
        change_score(bad,lambda s:s['terms'].update(P=0.))
        with self.assertRaises(ValueError):module.rank_candidates({'bad':bad})
        unknown=copy.deepcopy(self.known)
        def lose_quality(score):
            score['terms']['Q']=None
            score.update(cost=None,full_U=None,six_term_comparison_complete=False,absolute_utility_complete=False)
        for index in range(2):change_score(unknown,lose_quality,index)
        row=decode(module.rank_candidates({'quality_missing':unknown}))['candidates'][0]
        self.assertEqual(row['unknown_utility_probability_mass'],1)
        self.assertTrue(all(d['terms']['Q']=='UNKNOWN' for d in row['per_draw_term_status']))

    def test_raw_resource_absence_scope_and_single_application_cannot_be_bypassed(self):
        for mutation in ('absent','startup_unknown','memory_unknown','power_unknown',
                         'already_safe','already_amortized','measurement_includes_drain','drain_truncated'):
            bad=copy.deepcopy(self.known)
            scoring=bad['draws'][0]['forecast']['scoring'];pred=scoring['resource_prediction']
            if mutation=='absent':scoring['resource_prediction']=None
            elif mutation=='startup_unknown':pred['startup_energy_joules']=None
            elif mutation=='memory_unknown':pred['memory_unknown_observation_duration_ns']=1
            elif mutation=='power_unknown':pred['power_unknown_measurement_duration_ns']=1
            elif mutation=='already_safe':pred['safe_factor_already_applied']=True
            elif mutation=='already_amortized':pred['startup_energy_already_amortized']=True
            elif mutation=='measurement_includes_drain':pred['total_measurement_duration_ns']=150
            else:pred['total_observation_duration_ns']=2
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):module.rank_candidates({'bad':bad})

    def test_shared_public_cohort_score_window_drain_and_joint_draw_contracts(self):
        for mutation in ('arrival','task','pool_id','request_id','request_order','window','drain',
                         'quality','normalizer','weights','draw_order','draw_id','block','draw_weight'):
            bad=copy.deepcopy(self.known)
            if mutation in ('task','pool_id','request_id'):bad['public_requests'][0][mutation]='changed'
            elif mutation=='arrival':bad['public_requests'][0]['arrival_ns']=[1,1]
            elif mutation=='request_order':bad['public_requests'].reverse()
            elif mutation=='window':bad['original_window'][1]=3
            elif mutation=='drain':bad['original_drain_end_ns']=151
            elif mutation=='quality':bad['score_comparison_contract']['quality']['text']['text']=.5
            elif mutation=='normalizer':bad['score_comparison_contract']['scoring_plan']['memory_budget_mib']=100
            elif mutation=='weights':bad['score_comparison_contract']['scoring_plan']['weights']['M']=.5
            elif mutation=='draw_order':bad['draws'].reverse()
            elif mutation=='draw_id':bad['draws'][0]['draw_id']='changed'
            elif mutation=='block':bad['draws'][0]['block_id']='changed'
            else:
                bad['draws'][0]['weight']=dict(numerator=1,denominator=2)
                bad['draws'][1]['weight']=dict(numerator=1,denominator=2)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                module.rank_candidates({'baseline':self.known,'bad':bad})

    def test_exact_positive_probability_scheme_and_offered_denominators(self):
        for mutation in ('zero','float','total','draw_offered','offered','duplicate_draw','unknown_request_loss'):
            bad=copy.deepcopy(self.known)
            if mutation=='zero':bad['draws'][0]['weight']=0
            elif mutation=='float':bad['draws'][0]['weight']=1/3
            elif mutation=='total':bad['draws'][0]['weight']=1
            elif mutation=='draw_offered':bad['draws'][0]['offered_requests']=1
            elif mutation=='offered':bad['offered_requests']=1
            elif mutation=='duplicate_draw':bad['draws'][1]['draw_id']='draw0'
            else:bad['draws'][0].update(status='UNKNOWN_WORLD',forecast=None,
                                      all_public_request_ids_retained=['r0'])
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):module.rank_candidates({'bad':bad})

    def test_original_stability_and_rho_gn_are_only_detached_copies(self):
        qualification={'stability_state_v2':{'status':'ORIGINAL_UNSTABLE'},'rho':1.2,
            'g(n)':.7,'stability_status':'UNSTABLE','reasons':['original rule']}
        output=module.rank_candidates({'a':self.known},qualifications={'a':qualification})
        self.assertEqual(output['candidates'][0]['qualification'],qualification)
        self.assertEqual(output['candidates'][0]['stability_status'],'UNSTABLE')
        qualification['reasons'].append('later mutation')
        self.assertEqual(output['candidates'][0]['qualification']['reasons'],['original rule'])
        self.assertFalse(output['original_stability_evaluated'])
        self.assertFalse(output['final_gain_eligible'])
        self.assertTrue(output['qualifications_copied_not_authenticated'])

    def test_caps_and_partial_weight_inputs_rejected_even_all_worlds_unknown(self):
        for field in ('global_admission_cap','global_concurrency_cap','cap1','cap2','global_cap'):
            bad=copy.deepcopy(self.generated);bad[field]=1
            with self.subTest(field=field),self.assertRaises(ValueError):module.rank_candidates({'bad':bad})
        with self.assertRaises(ValueError):
            module.rank_candidates({'a':self.known},qualifications={'a':{'global_admission_cap':1}})
        with self.assertRaises(ValueError):
            module.rank_candidates({'a':self.known},qualifications={'wrong_candidate':{}})

    def test_anchor_ties_avoid_transitive_chain_and_preserve_exact_ties(self):
        # Memory cost deltas are 1/6 of M: chain spacings .09 utility, tolerance .1.
        candidates={name:supply_synthetic_resources(self.generated,memory)
                    for name,memory in [('low',2.08),('middle',1.54),('high',1.)]}
        output=decode(module.rank_candidates(candidates,utility_tie_tolerance=.1))
        self.assertEqual([g['candidate_ids'] for g in output['complete_domain_ranking']],
                         [['high','middle'],['low']])
        self.assertEqual(output['winner_candidate_ids'],['high','middle'])
        result=decode(module.rank_candidates({'a':self.known,'b':copy.deepcopy(self.known)}))
        self.assertEqual(result['winner_candidate_ids'],['a','b'])
        self.assertEqual(result['paired_comparisons'][0]['strict_draw_relation'],'EXACTLY_TIED')
        self.assertFalse(result['paired_comparisons'][0]['comparison']['candidate_better'])

    def test_strict_dominance_uses_every_draw_and_strict_tolerance_boundary(self):
        worse=supply_synthetic_resources(self.generated,7.)
        result=decode(module.rank_candidates({'worse':worse,'better':self.known}))
        pair=result['paired_comparisons'][0]
        self.assertEqual(pair['strict_draw_relation'],'CANDIDATE_DOMINATES')
        delta=pair['comparison']['draws'][0]['delta_U']
        result=decode(module.rank_candidates({'worse':worse,'better':self.known},utility_tie_tolerance=delta))
        self.assertEqual(result['paired_comparisons'][0]['strict_draw_relation'],'NO_STRICT_DRAW_DOMINANCE')

    def test_invalid_tolerance_and_empty_domains_rejected(self):
        for tolerance in (-1,math.nan,math.inf,True,'0'):
            with self.subTest(tolerance=tolerance),self.assertRaises(ValueError):
                module.rank_candidates({'a':self.known},utility_tie_tolerance=tolerance)
        for candidates in ({},[],{'':self.known}):
            with self.subTest(candidates=type(candidates)),self.assertRaises(ValueError):module.rank_candidates(candidates)

    def test_JSON_round_trip_preserves_fraction_mass_and_never_calls_generator(self):
        candidates=json.loads(json.dumps({'a':self.known,'b':self.generated}))
        with patch.object(ensemble,'forecast',side_effect=AssertionError('ranking must not forecast')):
            output=json.loads(json.dumps(module.rank_candidates(candidates)))
        self.assertEqual(decode(output)['candidates'][1]['unknown_utility_probability_mass'],1)
        self.assertFalse(output['GPU_started']);self.assertFalse(output['validation_labels_read'])
        self.assertFalse(output['worlds_generated']);self.assertFalse(output['model_fit'])


if __name__=='__main__':
    unittest.main()
