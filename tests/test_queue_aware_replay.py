"""Closed-loop visibility tests; synthetic values test causality only."""
import unittest
from tools.queue_aware_20261023.replay import simulate,frozen_tables,metrics,ns,DEPLOYMENT
from tools.spillover_context_20261022.development_oracle import DevelopmentOracle

class ClosedLoopTests(unittest.TestCase):
    def test_completed_events_only_and_estimate_environment_mismatch(self):
        aliases={'s#0':dict(family='s',supported_tasks=['text'])}
        requests=[dict(request_id=str(i),arrival_ns=t,task='text',pool_id='p') for i,t in enumerate([0,5_000_000,12_000_000,21_000_000])]
        snapshots=[]
        def choose(req,snapshot):
            snapshots.append(snapshot)
            # The policy estimates1ms; the environment takes10ms. Neither its
            # future finish nor actual service appears in the visible snapshot.
            for state in snapshot.values():
                for r in state['outstanding']:
                    self.assertEqual(set(r),{'request_id','task','pool_id','enqueued_ns','actual_submit_ns'})
                    self.assertLessEqual(r['enqueued_ns'],req['arrival_ns'])
                    if r['actual_submit_ns'] is not None:self.assertLessEqual(r['actual_submit_ns'],req['arrival_ns'])
            return dict(alias='s#0',status='MOCK_ESTIMATE_1MS')
        rows,decisions=simulate(requests,aliases,{'s':{'text':{'p':10.}}},choose,estimated_services={'s':{'text':{'p':1.}}})
        self.assertEqual(snapshots[1]['s#0']['last_completed_ns'],None)
        self.assertEqual(decisions[1]['observed_completed_ids'],[])
        self.assertEqual(decisions[2]['observed_completed_ids'],['0'])
        self.assertEqual(decisions[3]['observed_completed_ids'],['0','1'])
        self.assertEqual([r['finish_ns'] for r in rows],[10_000_000,20_000_000,30_000_000,40_000_000])

    def test_arrivals_at_completion_observe_it_before_routing(self):
        aliases={'s#0':dict(family='s',supported_tasks=['text'])};seen=[]
        requests=[dict(request_id=str(i),arrival_ns=i*10_000_000,task='text',pool_id='p') for i in range(2)]
        def choose(req,snapshot):seen.append(snapshot);return {'alias':'s#0'}
        rows,decisions=simulate(requests,aliases,{'s':{'text':{'p':10.}}},choose)
        self.assertEqual(seen[1]['s#0']['outstanding'],[])
        self.assertEqual(seen[1]['s#0']['last_completed_ns'],10_000_000)
        self.assertEqual(decisions[1]['newly_observed_completed_ids'],['0'])

    def test_queued_host_submit_is_visible_but_worker_start_is_not(self):
        aliases={'s#0':dict(family='s',supported_tasks=['text'])};seen=[]
        requests=[dict(request_id=str(i),arrival_ns=i*1_000_000,task='text',pool_id='p') for i in range(3)]
        def choose(req,snapshot):seen.append(snapshot);return {'alias':'s#0'}
        rows,_=simulate(requests,aliases,{'s':{'text':{'p':10.}}},choose)
        self.assertEqual(seen[2]['s#0']['outstanding'][1]['actual_submit_ns'],1_000_000)
        self.assertEqual(rows[1]['start_ns'],10_000_000)
        self.assertNotIn('start_ns',seen[2]['s#0']['outstanding'][1])

    def test_real_frozen_nosplit_reproduces_original_point_metrics(self):
        o=DevelopmentOracle();services,quality,context=frozen_tables(o);plan=context['reference_plan']
        aliases={f'{m}#{i}':dict(family=m,supported_tasks=o.catalog['models'][m]['supported_tasks']) for m,n in DEPLOYMENT.items() for i in range(n)}
        _,cohort=o._cohort(107);requests=[dict(request_id=str(i),arrival_ns=ns(a),task=t,pool_id=sid) for i,(a,t,sid) in enumerate(cohort)]
        _,route=o.api['trace'].expand_routes(DEPLOYMENT,plan['routing']);router=o.api['routing'].Router(route)
        def choose(req,snapshot):return {'alias':router.choose(req['task'])}
        rows,_=simulate(requests,aliases,services,choose);actual=metrics(o,rows)
        expected=o.evaluate(plan,107,True,strict_context_checks=False)['diagnostic_extrapolation_only']
        for term in 'QTLP':self.assertAlmostEqual(actual['terms'][term],expected['terms'][term],places=8)
        self.assertIsNone(actual['U']);self.assertIsNone(actual['terms']['M']);self.assertIsNone(actual['terms']['E'])
        self.assertTrue(next(p for p in context['paths'] if p['family']=='generalist_2' and p['task']=='text')['context_extrapolation'])
        o.verify_inputs()

if __name__=='__main__':unittest.main()
