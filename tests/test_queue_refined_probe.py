import copy
import unittest
from tools.queue_refined_calibration_20261023.probe_policy import ProbePolicy
from tools.queue_refined_calibration_20261023.guarded_policy import RateLimitedPolicy
from tools.queue_refined_calibration_20261023.policy import QueueAwarePolicy


class RandomProbeTests(unittest.TestCase):
    def base(self):
        aliases={f'{m}#0':dict(family=m,supported_tasks=['text']) for m in ['text','generalist_2']}
        return QueueAwarePolicy(aliases,{m:{'text':{'a':1.,'b':20.}} for m in ['text','generalist_2']},
            {m:{'text':.1} for m in ['text','generalist_2']},dict(L=1,Q=1),dict(text=1000))

    def policy(self,probability=.0625,phase='measurement',seed=201):
        return ProbePolicy(self.base(),dict(schema='randomized_text_probe_v1',probability=probability,
            namespace='queue-refined-calibration-20261023'),seed=seed,phase=phase)

    def snapshot(self,p):return {a:dict(outstanding=[],last_completed_ns=None) for a in p.aliases}

    def test_coins_replay_independent_of_inputs_and_future_labels(self):
        a=self.policy();b=self.policy();sa=self.snapshot(a);sb=self.snapshot(b);sb['generalist_2#0']['future_service_ms']=999999
        for i in range(1000):
            da=a.choose('text','a',i*1000000,sa);db=b.choose('text','b',i*1000000,sb)
            self.assertEqual(da['probe'],db['probe'])
            self.assertEqual(da['probe']['requested_probe'],da['alias'].startswith('generalist_2'))

    def test_phase_and_seed_distinct_streams(self):
        ps=[self.policy(),self.policy(phase='warmup'),self.policy(seed=202)]
        self.assertEqual(len({p.rng_seed for p in ps}),3)

    def test_zero_probe_never_spills_and_guard_fallback_consumes_one_coin(self):
        p=self.policy(0);s=self.snapshot(p)
        self.assertTrue(all(p.choose('text','a',i,s)['alias']=='text#0' for i in range(20)))
        p=self.policy(1);g=RateLimitedPolicy(p,dict(schema='queue_aware_token_bucket_v1',task='text',family='generalist_2',rate_rps=.23125,burst=1))
        a=g.choose('text','a',0,s);b=g.choose('text','a',1,s)
        self.assertEqual(a['alias'],'generalist_2#0');self.assertEqual(b['alias'],'text#0')
        self.assertEqual(b['probe']['draw_index'],2);self.assertTrue(b['probe']['requested_probe']);self.assertFalse(b['rate_guard']['consume'])

    def test_actual_mock_workers_replay_probe_phases(self):
        from tests.test_queue_refined_probe_native import NativeRunner
        fixture=NativeRunner();fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.contract.update(strategy='queue_latency',mode='latency',guard=dict(schema='queue_aware_token_bucket_v1',task='text',family='generalist_2',rate_rps=10,burst=1),
            probe=dict(schema='randomized_text_probe_v1',probability=.5,namespace='queue-refined-calibration-20261023'))
        fixture.test_original_cpu_workers_priming_warmup_measurement_shutdown()

if __name__=='__main__':unittest.main()
