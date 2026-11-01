"""Causal decisions from observable queue snapshots; no native/GPU execution."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('queue_policy',ROOT/'tools/queue_aware_20261023/policy.py')
p=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(p)


class QueueAware(unittest.TestCase):
    def policy(self,*,mode='quality_aware',generalist=120.,qualities=(.05,.2),specialists=1):
        aliases={f'text#{i}':dict(family='text',supported_tasks=['text']) for i in range(specialists)}
        aliases['generalist_2#0']=dict(family='generalist_2',supported_tasks=['text','image','asr'])
        service={'text':{'text':{'x':100.}},'generalist_2':{'text':{'x':generalist},'image':{'i':500.},'asr':{'a':400.}}}
        return p.QueueAwarePolicy(aliases,service,{'text':{'text':qualities[0]},'generalist_2':{'text':qualities[1]}},
            {'L':1.,'Q':1.},{'text':100.},mode=mode)

    def state(self,specialists=1):
        return {**{f'text#{i}':dict(outstanding=[],last_completed_ns=None) for i in range(specialists)},
            'generalist_2#0':dict(outstanding=[],last_completed_ns=None)}

    def request(self,request_id='r',task='text',pool_id='x',enqueued=0,submit=0):
        return dict(request_id=request_id,task=task,pool_id=pool_id,enqueued_ns=enqueued,actual_submit_ns=submit)

    def test_specialist_blocked_generalist_idle_spills(self):
        state=self.state();state['text#0']['outstanding']=[self.request('r0'),self.request('r1',submit=None)]
        r=self.policy().choose('text','x',0,state)
        self.assertEqual(r['alias'],'generalist_2#0')
        specialist=next(c for c in r['audit'] if c['alias']=='text#0')
        self.assertEqual(specialist['predicted_waiting_ms'],200.)
        self.assertEqual(specialist['waiting_evidence']['submitted_count'],1)
        self.assertEqual(specialist['waiting_evidence']['queued_count'],1)

    def test_specialist_idle_returns_to_specialist(self):
        self.assertEqual(self.policy().choose('text','x',0,self.state())['alias'],'text#0')

    def test_generalist_busy_with_all_background_tasks_not_selected(self):
        state=self.state();state['text#0']['outstanding']=[self.request()]
        state['generalist_2#0']['outstanding']=[self.request('a','image','i'),self.request('b','asr','a',submit=None)]
        r=self.policy().choose('text','x',0,state)
        self.assertEqual(r['alias'],'text#0')
        g=next(c for c in r['audit'] if c['alias']=='generalist_2#0')
        self.assertEqual(g['predicted_waiting_ms'],900.)

    def test_two_specialists_balance_before_spilling(self):
        state=self.state(2);policy=self.policy(specialists=2,generalist=300.)
        self.assertEqual(policy.choose('text','x',0,state)['alias'],'text#0')
        state['text#0']['outstanding']=[self.request()]
        self.assertEqual(policy.choose('text','x',0,state)['alias'],'text#1')

    def test_slow_generalist_not_forced_when_idle(self):
        state=self.state();state['text#0']['outstanding']=[self.request()]
        self.assertEqual(self.policy(generalist=1000.).choose('text','x',0,state)['alias'],'text#0')

    def test_higher_quality_changes_decision_without_absolute_gate(self):
        state=self.state();policy=self.policy(generalist=80.,qualities=(0.,.5))
        self.assertEqual(policy.choose('text','x',0,state)['alias'],'text#0')
        latency=self.policy(mode='latency',generalist=80.,qualities=(0.,.5))
        self.assertEqual(latency.choose('text','x',0,state)['alias'],'generalist_2#0')
        zero=p.QueueAwarePolicy(policy.aliases,policy.service_ms,policy.quality_loss,{'L':1.,'Q':0.},{'text':100.})
        self.assertEqual(zero.choose('text','x',0,state)['audit'],latency.choose('text','x',0,state)['audit'])

    def test_future_labels_ignored_and_no_snapshot_mutation(self):
        state=self.state();state['text#0']['outstanding']=[self.request()];before=copy.deepcopy(state)
        baseline=self.policy().choose('text','x',50_000_000,state)
        altered=copy.deepcopy(state)
        for alias in altered:
            altered[alias].update(worker_latency_ms=-999,completion_ns=0,reference='SECRET',actual_future_service=0)
            for row in altered[alias]['outstanding']:
                row.update(worker_latency_ms=1e9,completion_time_ns=0,predicted_answer='SECRET',actual_service_ms=0)
        self.assertEqual(baseline,self.policy().choose('text','x',50_000_000,altered))
        self.assertEqual(state,before)

    def test_overdue_submitted_head_has_floor_and_suffix_full_service(self):
        state=self.state();state['text#0']['outstanding']=[self.request('a'),self.request('b',submit=0)]
        r=self.policy().choose('text','x',1_000_000_000,state)
        s=next(c for c in r['audit'] if c['alias']=='text#0')
        self.assertEqual(s['predicted_waiting_ms'],105.)
        self.assertTrue(s['waiting_evidence']['outstanding'][0]['overdue_floor_applied'])
        self.assertEqual(s['waiting_evidence']['outstanding'][1]['predicted_remaining_ms'],100.)

    def test_last_received_completion_delays_inferred_head_start(self):
        state=self.state();state['text#0']['outstanding']=[self.request()]
        state['text#0']['last_completed_ns']=90_000_000
        r=self.policy().choose('text','x',100_000_000,state)
        s=next(c for c in r['audit'] if c['alias']=='text#0')
        self.assertEqual(s['predicted_waiting_ms'],90.)

    def test_queued_head_not_aged_and_future_enqueue_does_not_add_work(self):
        state=self.state();state['text#0']['outstanding']=[self.request(submit=None)]
        state['generalist_2#0']['outstanding']=[self.request('future','unknown','missing',2_000_000_000,2_000_000_000)]
        r=self.policy().choose('text','x',1_000_000_000,state)
        s=next(c for c in r['audit'] if c['alias']=='text#0')
        g=next(c for c in r['audit'] if c['alias']=='generalist_2#0')
        self.assertEqual(s['predicted_waiting_ms'],100.);self.assertEqual(g['predicted_waiting_ms'],0.)

    def test_unknown_candidate_audited_no_optimistic_zero(self):
        state=self.state();state['generalist_2#0']['outstanding']=[self.request(task='image',pool_id='missing')]
        r=self.policy().choose('text','x',0,state)
        g=next(c for c in r['audit'] if c['alias']=='generalist_2#0')
        self.assertIsNone(g['score']);self.assertIn('UNKNOWN_OUTSTANDING_SERVICE',g['unknown_reasons'])
        self.assertEqual(r['alias'],'text#0')
        self.assertIsNone(self.policy().choose('text','unknown',0,state)['alias'])

    def test_missing_snapshot_and_invalid_future_submit_unknown(self):
        state=self.state();del state['text#0']
        state['generalist_2#0']['outstanding']=[self.request(submit=1)]
        self.assertIsNone(self.policy().choose('text','x',0,state)['alias'])


if __name__=='__main__':unittest.main()
