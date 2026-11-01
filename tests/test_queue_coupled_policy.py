import unittest
from tools.queue_coupled_utility_20261023.policy import CoupledUtilityPolicy,percentile


class Provider:
    def service(self,row,active,aliases,scenario):
        base=100. if aliases[row['alias']]['family']=='text' else 10.
        return dict(known=True,context_supported=True,service_ms=base*len(active))
    def residual(self,row,age,active,aliases,scenario):return dict(known=age<1000,tail_supported=age<1000,remaining_ms=100.)
    def power(self,active,aliases,scenario):return dict(known=True,watts=50+100*len(active))

class CoupledPolicyTests(unittest.TestCase):
    def aliases(self):return {a:dict(family='text' if a.startswith('t') else 'generalist_2',supported_tasks=['text']) for a in ['t0','t1','g']}
    def policy(self,provider=None,**kwargs):
        p=CoupledUtilityPolicy(self.aliases(),provider or Provider(),{'text':{'text':0.},'generalist_2':{'text':.01}},
            {t:1/6 for t in 'QTLPME'},{'text':100.},{'text':1.},**kwargs)
        p.start_phase(0,1_000_000_000);return p
    def snapshot(self):
        return {a:dict(outstanding=[],last_completed_ns=None) for a in self.aliases()}
    def test_cohort_p95_deadline_and_memory_unknown_flags(self):
        p=self.policy();s=self.snapshot();p.set_arrival('a',0);result=p.choose('text','x',0,s)
        self.assertEqual(result['alias'],'g');self.assertTrue(result['actual_spill_selected'])
        ids=[v['cohort_ids'] for a in result['audit'] for v in a['scenarios'].values()]
        self.assertTrue(all(v==['a'] for v in ids))
        for a in result['audit']:
            for m in a['scenarios'].values():
                self.assertEqual(m['terms']['P'],m['terms']['L'])
                self.assertEqual(m['terms']['T'],0.)
        self.assertIsNone(result['M_delta']);self.assertIsNone(result['full_U'])
        self.assertFalse(result['actual_peak_memory_invariance_proved'])
        self.assertAlmostEqual(percentile([1.,10.,100.]),91.)
    def test_missing_resources_fallback_to_specialists(self):
        class Missing(Provider):
            def power(self,*args):return dict(known=False,watts=None)
        result=self.policy(Missing()).choose('text','x',0,self.snapshot())
        self.assertEqual(result['status'],'SPECIALIST_EVIDENCE_FALLBACK');self.assertTrue(result['alias'].startswith('t'))
        self.assertTrue(any('UNKNOWN_WHOLE_BOARD_POWER' in x for a in result['audit'] for x in a['unknown_reasons']))
    def test_unsupported_residual_and_control_do_not_spill(self):
        p=self.policy();s=self.snapshot()
        s['t0']['outstanding']=[dict(request_id='old',task='text',pool_id='x',enqueued_ns=0,actual_submit_ns=0)]
        result=p.choose('text','x',2_000_000_000,s)
        self.assertFalse(result['actual_spill_selected'])
        c=self.policy(allow_spill=False);self.assertFalse(c.choose('text','x',0,self.snapshot())['actual_spill_selected'])
    def test_bucket_and_completed_feedback_cooldown_are_causal(self):
        p=self.policy(exposure_rate_rps=.1,cooldown_ms=500.,drift_ratio_limit=2.)
        p.set_arrival('a',0);first=p.choose('text','x',0,self.snapshot());self.assertEqual(first['alias'],'g')
        p.observe_admission(dict(request_id='a',model='g',task='text',pool_id='x',arrival_ns=0))
        p.observe_completed([dict(request_id='a',model='g',task='text',pool_id='x',enqueued_ns=0,completion_time_ns=100_000_000,worker_service_ms=100.)])
        self.assertEqual(p.cooldown_until_ns,600_000_000)
        p.set_arrival('b',100_000_000);second=p.choose('text','x',100_000_000,self.snapshot())
        self.assertFalse(second['actual_spill_selected'])
        self.assertIn('COMPLETION_FEEDBACK_DRIFT_COOLDOWN',second['gates']);self.assertIn('EXPOSURE_BUDGET_EXHAUSTED',second['gates'])
        self.assertTrue(all('a' in m['cohort_ids'] for a in second['audit'] for m in a['scenarios'].values()))
    def test_deadline_term_and_future_feedback_rejected(self):
        p=self.policy();p.deadline_ns=5_000_000;p.set_arrival('a',0)
        result=p.choose('text','x',0,self.snapshot())
        self.assertTrue(all(m['terms']['T']==1. for a in result['audit'] for m in a['scenarios'].values()))
        p=self.policy();p.observe_completed([dict(request_id='a',alias='t0',task='text',pool_id='x',arrival_ns=0,completion_ns=10)])
        with self.assertRaisesRegex(ValueError,'Future completion'):p.choose('text','x',0,self.snapshot())
    def test_deterministic_budget_skips_forecast_without_truncating_cohort(self):
        class Counted(Provider):
            def __init__(self):self.calls=0
            def service(self,*args):self.calls+=1;return super().service(*args)
        provider=Counted();p=self.policy(provider,maximum_pending_for_forecast=2);snapshot=self.snapshot()
        for alias,count in [('t0',2),('t1',1),('g',1)]:
            snapshot[alias]['outstanding']=[dict(request_id=f'{alias}:{i}',task='text',pool_id='x',enqueued_ns=0) for i in range(count)]
        original=repr(snapshot);p.set_arrival('new',0)
        result=p.choose('text','x',0,snapshot)
        self.assertEqual(result['alias'],'t1');self.assertEqual(result['status'],'SPECIALIST_COMPUTE_BUDGET_FALLBACK')
        self.assertEqual(provider.calls,0);self.assertEqual(result['audit'],[])
        self.assertFalse(result['actual_spill_selected']);self.assertEqual(repr(snapshot),original)
        self.assertIn('DETERMINISTIC_PENDING_FORECAST_BUDGET_EXCEEDED',result['gates'])
        self.assertNotIn('NO_COMPLETE_SHARED_RESOURCE_FORECAST',result['gates'])
        self.assertEqual(result['forecast_computation_budget'],dict(maximum_pending_for_forecast=2,pending_count=4,
            exceeded=True,forecast_skipped=True,cohort_not_truncated=True,wall_clock_timeout_used=False))
        replay=self.policy(Counted(),maximum_pending_for_forecast=2);replay.set_arrival('new',0)
        self.assertEqual(result,replay.choose('text','x',0,snapshot))
    def test_budget_boundary_is_inclusive_and_input_requires_integer(self):
        p=self.policy(maximum_pending_for_forecast=1);snapshot=self.snapshot()
        snapshot['t0']['outstanding']=[dict(request_id='old',task='text',pool_id='x',enqueued_ns=0)]
        result=p.choose('text','x',0,snapshot)
        self.assertFalse(result['forecast_computation_budget']['exceeded']);self.assertTrue(result['audit'])
        for value in (0,-1,1.,True,None):
            with self.assertRaisesRegex(ValueError,'pending forecast budget'):self.policy(maximum_pending_for_forecast=value)

if __name__=='__main__':unittest.main()
