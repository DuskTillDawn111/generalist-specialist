"""Frozen-law reconstruction diagnostics; no training or GPU execution."""
import copy
import unittest

from tools.predictor_validation_20261027 import flow_forensics as module


class ConstantProvider:
    def __init__(self, unknown_multiway=False):
        self.unknown_multiway = unknown_multiway

    def plan(self,row,aliases,scenario):
        return {'profile_id':'fixed','execution':{},'segments':[
            {'id':phase,'kind':'mixed','duration_ms':100.,'demand':{'sm':None,'dram':None,'l2':None}}
            for phase in ('prepare','prefill')]}

    def service(self,row,active,aliases,scenario):
        if self.unknown_multiway and len(active)>1:
            return {'known':False,'reason':'UNSEEN_FULL_MIXTURE'}
        return {'known':True,'context_supported':True,'service_ms':100.,'interpolated':row['segment_id']=='prefill'}

    def power(self,*args):
        return {'known':False}

    def memory(self,*args):
        return {'known':False}


def row(rid='a',alias='s#0',start=0,prepare_end=100_000_000,end=300_000_000):
    return {'request_id':rid,'native_request_id':rid,'alias':alias,'family':'text','task':'text','pool_id':'input',
            'handler_start_ns':start,'handler_end_ns':end,'parent_receipt_ns':end+1_000_000,
            'segments':[{'phase':'prepare','start_ns':start,'end_ns':prepare_end},
                        {'phase':'prefill','start_ns':prepare_end,'end_ns':end}]}


class FlowForensicsTests(unittest.TestCase):
    def test_reconstructs_work_on_observed_phases_without_refitting(self):
        trace={'id':'validation-run','split':'validation','requests':[row()]}
        report=module.observed_reconstruction(trace,ConstantProvider(),{'s#0':{'family':'text'}},origin_ns=0,horizon_ns=1_000_000_000,
            forecast={'a':{'service_ms':190.,'start_ns':0}})
        phases={r['phase']:r for r in report['phases']}
        self.assertEqual(phases['prepare']['reconstructed_work'],1.)
        self.assertEqual(phases['prefill']['reconstructed_work'],2.)
        request=report['requests'][0]
        self.assertEqual(request['actual_service_ms'],300.)
        self.assertEqual(request['observed_context_service_proxy_ms'],200.)
        self.assertEqual(request['parent_tail_ms'],1.)
        self.assertEqual(report['phase_reconstruction_coverage']['interpolated_duration_fraction'],2/3)

    def test_unknown_partial_context_is_not_imputed_or_dropped(self):
        trace={'split':'validation','requests':[row(),row('b','s#1',0,50_000_000,150_000_000)]}
        report=module.observed_reconstruction(trace,ConstantProvider(True),{'s#0':{'family':'text'},'s#1':{'family':'text'}},
            origin_ns=0,horizon_ns=1_000_000_000)
        self.assertEqual(len(report['phases']),4)
        self.assertEqual(report['phase_reconstruction_coverage']['known'],0)
        self.assertTrue(all(p['reconstructed_work'] is None for p in report['phases']))
        self.assertTrue(all(r['observed_context_service_proxy_ms'] is None for r in report['requests']))

    def test_pure_joint_vectors_preserve_phase_order_and_validation_split(self):
        trace={'id':'validation-run','split':'validation','sources':[{'path':'immutable-source','sha256':'x'}],'requests':[row()]}
        before=copy.deepcopy(trace)
        vectors=module.reconstruct_work([trace],ConstantProvider())
        self.assertEqual(trace,before)
        item=vectors['requests'][0]
        self.assertEqual(item['phase_order'],['prepare','prefill'])
        self.assertEqual(item['work_vector'],[1.,2.])
        self.assertTrue(item['not_for_training'])
        self.assertEqual(vectors['training_eligible_requests'],0)
        trace['split']='training'
        self.assertEqual(module.reconstruct_work([trace],ConstantProvider())['training_eligible_requests'],1)

    def test_same_alias_sequential_requests_allowed_and_overlap_rejected(self):
        first=row(end=200_000_000)
        second=row('b','s#0',200_000_000,300_000_000,400_000_000)
        trace={'split':'validation','requests':[first,second]}
        report=module.observed_reconstruction(trace,ConstantProvider(),{'s#0':{'family':'text'}},origin_ns=0,horizon_ns=1_000_000_000)
        self.assertEqual([r['worker_request_ordinal'] for r in report['requests']],[1,2])
        self.assertEqual(report['requests'][1]['prior_worker_busy_ms'],200.)
        second['handler_start_ns']=100_000_000
        second['segments'][0]['start_ns']=100_000_000
        with self.assertRaisesRegex(ValueError,'one native alias'):
            module.observed_reconstruction(trace,ConstantProvider(),{'s#0':{'family':'text'}},origin_ns=0,horizon_ns=1_000_000_000)

    def test_semantic_handler_gap_is_rejected(self):
        request=row();request['segments'][1]['start_ns']+=1
        with self.assertRaisesRegex(ValueError,'continuously cover'):
            module.reconstruct_work([{'id':'run','split':'validation','requests':[request]}],ConstantProvider())

    def test_replay_must_exactly_match_original_frozen_predictions(self):
        candidate={'routes':{'a':'s#0'}}
        plan={'aliases':{'s#0':{'family':'text','supported_tasks':['text']}},'window_seconds':1.,'drain_seconds':1.,
              'trace':[{'request_id':'a','task':'text','pool_id':'input','offset_ns':0}]}
        frozen={'requests':[{'request_id':'a','finish_ns':200_000_000,'service_ms':200.}]}
        forecast,phases,result=module.replay(frozen,plan,candidate,ConstantProvider())
        self.assertEqual(phases[('a','prepare')],100.)
        self.assertEqual(forecast['a']['start_ns'],0)
        frozen['requests'][0]['service_ms']=199.
        with self.assertRaisesRegex(ValueError,'differs from original'):
            module.replay(frozen,plan,candidate,ConstantProvider())

    def test_timeline_phase_disagreement_keeps_start_drift_scope(self):
        trace={'requests':[row(end=200_000_000)]}
        simulated={'transitions':[{'time_ns':0,'until_ns':200_000_000,
            'active_requests':{'s#0':'a'},'segments':{'s#0':'prefill'}}]}
        result=module.timeline_disagreement(trace,simulated,0,200_000_000)
        self.assertEqual(result['absolute_time_state_disagreement_fraction'],.5)
        self.assertEqual(result['same_request_different_phase_ms'],100.)

    def test_utility_terms_are_original_algebra_not_new_predictions(self):
        def candidate(cid,p,a):
            return {'candidate_id':cid,'prediction_metrics':{'terms':dict.fromkeys('QTLPME',p)},
                    'observed_metrics':{'terms':dict.fromkeys('QTLPME',a)}}
        report={'candidates':[candidate('base',1,2),candidate('spill',2,1)]}
        result=module.utility_attribution(report,dict.fromkeys('QTLPME',1/6))[0]
        self.assertEqual(result['terms']['P']['predicted_delta_contribution'],-1/6)
        self.assertEqual(result['terms']['P']['observed_delta_contribution'],1/6)


if __name__=='__main__':
    unittest.main()
