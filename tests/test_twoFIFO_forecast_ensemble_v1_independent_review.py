"""Independent finite mass, fair paired comparison and joint-law CPU tests."""
import copy
from fractions import Fraction as F
import json
import unittest

from tests.test_twoFIFO_forecast_ensemble_v1 import arguments
from tests.test_actual_worker_twoFIFO_reference_v4 import decode
from tools.predictor_validation_20261027 import twoFIFO_forecast_ensemble_v1 as m
from tools.predictor_validation_20261027.twoFIFO_paired_transport_v3 import PairedTransportConstraints as C


class EnsembleIndependentReview(unittest.TestCase):
    def three_worlds(self):
        requests,draws,args=arguments();known=draws[0];known['weight']=F(1,3)
        incomplete=copy.deepcopy(known);incomplete.update(draw_id='incomplete',weight=F(1,6))
        # Later work always exceeds the original fixed drain while transport
        # itself remains identified; this is a simulated miss, not UNKNOWN.
        incomplete['inputs']['rates']={key:F(1,100) for key in incomplete['inputs']['rates']}
        unknown=copy.deepcopy(known);unknown.update(draw_id='unknown',weight=F(1,2))
        unknown['inputs']['transport_draws'][('owner','text','input',0)]=C(0,0,5,10,9)
        return requests,[known,incomplete,unknown],args

    def test_encoded_fraction_mass_distinguishes_unknown_world_and_known_fixed_drain_miss(self):
        r,d,a=self.three_worlds();out=json.loads(json.dumps(m.evaluate(r,d,**a)));v=decode(out)
        self.assertEqual(v['offered_requests'],2)
        self.assertEqual(v['world_unknown_probability_mass'],F(1,2))
        self.assertEqual(v['draws'][1]['status'],'GENERATED_WORLD')
        self.assertEqual(v['draws'][1]['forecast']['world']['unfinished_request_ids'],['r0','r1'])
        self.assertEqual(v['six_terms']['P']['known_probability_mass'],F(1,3))
        self.assertEqual(v['six_terms']['P']['unknown_probability_mass'],F(2,3))
        self.assertIsNone(v['six_terms']['P']['expected_value'])
        self.assertAlmostEqual(v['six_terms']['P']['conditional_on_known_mean'],28.5)
        self.assertTrue(all(row['fixed_drain_incomplete_probability_bounds']==[F(1,6),F(2,3)] for row in v['fixed_drain_risk']))
        for row in v['draws']:self.assertEqual(row['offered_requests'],2)
        self.assertEqual(v['draws'][2]['all_public_request_ids_retained'],['r0','r1'])
        self.assertEqual(v['draws'][2]['evidence']['offered_requests'],2)
        self.assertFalse(v['unknown_worlds_renormalized'])

    def test_fractional_paired_delta_bounds_keep_unsupported_mass(self):
        r,d,a=self.three_worlds();baseline=m.evaluate(r,d,**a);candidate=copy.deepcopy(baseline)
        baseline['draws'][0]['forecast']['scoring']['original_score']['full_U']=0
        candidate['draws'][0]['forecast']['scoring']['original_score']['full_U']=3
        v=decode(m.compare(json.loads(json.dumps(baseline)),json.loads(json.dumps(candidate))))
        self.assertIsNone(v['expected_delta_U']);self.assertIsNone(v['candidate_better'])
        self.assertEqual(v['unknown_delta_probability_mass'],F(2,3))
        self.assertEqual(v['positive_delta_probability_bounds'],[F(1,3),1])
        self.assertEqual([row['weight'] for row in v['draws']],[F(1,3),F(1,6),F(1,2)])

    def test_comparison_rejects_input_task_time_id_order_and_boundary_changes(self):
        r,d,a=arguments();base=m.evaluate(r,d,**a)
        for field in ('pool_id','task','request_id','arrival_ns','order','window','drain','weight','block'):
            candidate=copy.deepcopy(base)
            if field in ('pool_id','task','request_id'):candidate['public_requests'][0][field]='changed'
            elif field=='arrival_ns':candidate['public_requests'][0][field]=[1,1]
            elif field=='order':candidate['public_requests'].reverse()
            elif field=='window':candidate['original_window'][1]=12345
            elif field=='drain':candidate['original_drain_end_ns']=12345
            elif field=='weight':candidate['draws'][0]['weight']=dict(numerator=1,denominator=2)
            else:candidate['draws'][0]['block_id']='different-block'
            with self.subTest(field=field),self.assertRaises(ValueError):m.compare(base,candidate)
        # Route changes are the intended candidate intervention.
        candidate=copy.deepcopy(base);candidate['public_requests'][0]['owner']='candidate-owner'
        self.assertIsNone(m.compare(base,candidate)['expected_delta_U'])

    def test_comparison_requires_shared_original_score_normalizers_and_quality(self):
        r,d,a=arguments();base=m.evaluate(r,d,**a)
        for field in ('workload_rps','latency_reference_ms','memory_budget_mib','memory_safe_factor','energy_reference_watts','quality'):
            changed=copy.deepcopy(a)
            if field=='quality':changed[field]['text']['text']+=.1
            elif field in ('workload_rps','latency_reference_ms'):changed['scoring_plan'][field]['text']*=2
            else:changed['scoring_plan'][field]*=2
            other=m.evaluate(r,d,**changed)
            with self.subTest(field=field),self.assertRaises(ValueError):m.compare(base,other)

    def test_jointlaw_v4_draw_compile_to_ensemble_keeps_joint_marks_and_ordinal(self):
        from tests.test_twoFIFO_joint_request_law_v2 import source_fixture
        from tests.test_actual_worker_world_forecast_v4 import inputs
        from tools.predictor_validation_20261027 import twoFIFO_joint_request_law_v4 as law_module
        _,raw,_,_,vector=source_fixture();second=copy.deepcopy(vector)
        vector['training_block_id']='training-block-1';second['training_block_id']='training-block-2'
        law=dict(schema=law_module.SCHEMA,blocks=['training-block-1','training-block-2'],vectors=[vector,second])
        row=raw['request_records'][0];previous=raw['priming_request_records'][-1]
        origin=row['scheduled_time_ns'];key=tuple(vector['public_key']);owner=key[0];draws=[]
        for index,q in enumerate((.1,.9)):
            chosen=law_module.choose_vector(law,key,0,block_quantile=q,within_block_quantile=0)
            compiled=law_module.compile_draw_v4(chosen['vector'],chosen['public_key_ordinal'])
            plans={key:compiled['plan']};rates={(owner,n.phase,0,0):F(1) for n in compiled['plan'].macros}
            draws.append(dict(draw_id='joint'+str(index),block_id=chosen['block_id'],weight=F(1,2),
                inputs=dict(plans=plans,rates=rates,protocol_slacks={compiled['plan']:compiled['protocol_slacks']},
                    transport_draws={(*key,0):compiled['transport_draw']})))
        initial=dict(requests=[dict(request_id='new-public-request',owner=owner,task=key[1],pool_id=key[2],arrival_ns=origin)],
            plans=draws[0]['inputs']['plans'],rates=draws[0]['inputs']['rates'],
            protocol_slacks=draws[0]['inputs']['protocol_slacks'],transport_draws=draws[0]['inputs']['transport_draws'],
            window_start_ns=origin,window_end_ns=origin+1,drain_end_ns=origin+10**10,
            initial_sender_available_ns={owner:previous['submit_write_completed_time_ns']},
            initial_worker_read_release_ns={owner:previous['previous_handler_next_request_ready']['perf_ns']},
            initial_protocol_state=dict(reader_free={},collector_free=origin,last_collector_arrival=origin,pending_messages=[]))
        args=inputs(initial);requests=args.pop('requests')
        for field in m.INPUT_FIELDS:args.pop(field)
        args['scoring_plan']['aliases']={owner:dict(family='text',supported_tasks=['text'])}
        out=decode(m.evaluate(requests,draws,**args))
        self.assertEqual(out['world_unknown_probability_mass'],0)
        self.assertEqual(out['offered_requests'],1)
        for result in out['draws']:
            world=result['forecast']['world']
            self.assertEqual(world['starts'],{'new-public-request':row['observed_worker_start_time_ns']})
            self.assertEqual(world['public_key_arrival_ordinals'],{'new-public-request':0})
            emitted=[e for e in world['trace'] if e['kind']=='latent_occurrence']
            self.assertEqual(len(emitted),sum(len(n['marks']) for n in vector['nodes'])+1)
            self.assertTrue(all('new-public-request'==e['request_id'] for e in emitted))
            self.assertIsNone(world['transport_schedule']['new-public-request']['stdin_line_available_ns'])

    def test_serialized_compare_revalidates_probability_and_offered_denominators(self):
        r,d,a=arguments();base=json.loads(json.dumps(m.evaluate(r,d,**a)))
        for mutation in ('weight2','weight0','weightfloat','offered','drawoffered','drawduplicate'):
            left=copy.deepcopy(base);right=copy.deepcopy(base)
            for out in (left,right):
                if mutation=='weight2':out['draws'][0]['weight']=2
                elif mutation=='weight0':out['draws'][0]['weight']=0
                elif mutation=='weightfloat':out['draws'][0]['weight']=1.0
                elif mutation=='offered':out['offered_requests']=1
                elif mutation=='drawoffered':out['draws'][0]['offered_requests']=1
                else:
                    out['draws'][0]['weight']=dict(numerator=1,denominator=2)
                    out['draws'].append(copy.deepcopy(out['draws'][0]))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.compare(left,right)

    def test_typed_missing_law_evidence_skips_forecast_and_keeps_detached_full_mass(self):
        from unittest.mock import patch
        r,d,a=arguments();evidence=dict(public_key=['owner','text','unseen'])
        d[0]['inputs']=m.UnknownForecastInputs('no donor for exact public input',evidence)
        with patch.object(m,'forecast',side_effect=AssertionError('must not generate fake empty plan')):
            out=m.evaluate(r,d,**a)
        evidence['public_key'][2]='later-mutated'
        v=decode(out)
        self.assertEqual(v['world_unknown_probability_mass'],1)
        self.assertEqual(v['draws'][0]['evidence']['public_key'][2],'unseen')
        self.assertEqual(v['draws'][0]['all_public_request_ids_retained'],['r0','r1'])
        self.assertTrue(all(row['fixed_drain_incomplete_probability_bounds']==[0,1] for row in v['fixed_drain_risk']))
        self.assertIsNone(v['six_terms']['P']['conditional_on_known_mean'])
        self.assertIsNone(v['utility']['expected_value'])
        for kwargs in ({'global_admission_cap':1},{'cutoff_ns':0},{'resource_memroy_typo':{}}):
            with self.subTest(kwargs=kwargs),self.assertRaises((TypeError,ValueError)):m.evaluate(r,d,**a,**kwargs)
