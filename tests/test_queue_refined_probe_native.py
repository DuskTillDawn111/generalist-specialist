"""CPU worker protocol end-to-end and phase/identity contracts; never GPU."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.queue_refined_calibration_20261023 import native_runner as native,protocol
from tools.queue_refined_calibration_20261023.runtime_window import queue_snapshot


class NativeRunner(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        asset=self.root/'asset';asset.write_text('cpu')
        self.samples=self.root/'pool.jsonl'
        rows=[dict(pool_id='t',task_type='text',sample=dict(context='c',question='q',label='ok'))]
        rows += [dict(pool_id=t,task_type=t,sample=dict(input=str(asset),label='ok')) for t in ['image','asr','ocr']]
        self.samples.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        models={'text':dict(supported_tasks=['text'],service_time_ms=1.),
            'generalist_2':dict(supported_tasks=['text','image','asr','ocr'],service_time_ms=1.),
            'ocr':dict(supported_tasks=['ocr'],service_time_ms=1.)}
        self.config=dict(catalog=dict(models=models,safe_factor=1.1,memory_table_mib={m:{'1':10.,'2':20.} for m in models}),
            workload={'text':40.,'image':10.,'asr':10.,'ocr':10.},weights={t:1/6 for t in 'QTLPME'},
            latency_reference_ms={t:100. for t in ['text','image','asr','ocr']},gpu_memory_mib=1000.)
        self.config['catalog']['quality']=dict(identity=dict(input_pool_sha256=protocol.receipt(self.samples)['sha256']))
        self.system=SimpleNamespace(**self.config,as_dict=lambda:copy.deepcopy(self.config))
        services={'text':{'text':{'t':1.}},'generalist_2':{'text':{'t':2.},**{t:{t:1.} for t in ['image','asr','ocr']}},'ocr':{'ocr':{'ocr':1.}}}
        self.contract=protocol.make_policy_contract({'text':2,'generalist_2':1,'ocr':1},services,
            {m:{t:.1 for t in spec['supported_tasks']} for m,spec in models.items()},self.config['weights'],self.config['latency_reference_ms'],
            strategy='queue_quality',models=models,text_rate=40.,background_routing={t:{'ocr' if t=='ocr' else 'generalist_2':10.} for t in ['image','asr','ocr']})

    def test_original_cpu_workers_priming_warmup_measurement_shutdown(self):
        from tools.queue_refined_calibration_20261023 import admission
        # Evaluation math has its own tests. This test exercises actual original
        # CPU worker subprocesses/collector/queues and callback raw contract.
        def evaluate(system,raw):
            from tools.queue_refined_calibration_20261023.evaluation import audit_phase
            self.assertTrue(raw['request_records'])
            self.assertEqual(raw['energy_evidence']['kind'],'unmeasured')
            self.assertTrue(audit_phase(system,raw)['verified'],audit_phase(system,raw))
            self.assertTrue(audit_phase(system,raw,'warmup')['verified'],audit_phase(system,raw,'warmup'))
            return dict(utility=dict(U=None,eligible=False,reason='MOCK_BACKEND'),stability={})
        with patch('subprocess.run',side_effect=AssertionError('No GPU/system query in mock')):
            raw=native.run(self.system,protocol.nominal_plan(self.contract),self.samples,self.contract,backend='mock',
                duration=.15,warmup=.10,seed=113,arrival_process='deterministic_periodic',startup_timeout=10.,drain_timeout=10.,evaluator=evaluate)
        self.assertEqual(raw['status'],'completed',raw['errors'])
        self.assertTrue(raw['priming_request_records']);self.assertTrue(raw['warmup_request_records'])
        self.assertTrue(all(r['status']=='completed' for r in raw['request_records']))
        self.assertTrue(all(not v['forced'] and v['exit_code']==0 and v['bye_received'] for v in raw['shutdown'].values()))
        first=raw['request_records'][0]['routing_decision']
        self.assertTrue(all(not s['outstanding'] and s['last_completed_ns'] is None for s in first['snapshot'].values()))
        self.assertEqual(first['newly_observed_completions'],[])
        self.assertFalse(raw['memory_evidence']['measured']);self.assertFalse(raw['energy_evidence']['coverage_valid'])
        self.assertIsNone(raw['evaluation']['utility']['U'])
        self.assertEqual(raw['energy_evidence']['profile_key'],admission.measurement_identity(self.system,raw))
        for r in raw['request_records']:
            self.assertGreaterEqual(r['routing_decision']['decision_time_ns'],r['scheduled_time_ns'])
            if r['task'] in ['image','asr']:self.assertEqual(r['model'],'generalist_2#0')
            if r['task']=='ocr':self.assertEqual(r['model'],'ocr#0')

    def test_priming_failure_preserved_and_shutdown_bounded(self):
        from tools.queue_refined_calibration_20261023 import admission
        rows=[json.loads(l) for l in self.samples.read_text().splitlines()]
        for r in rows:r['sample']['mock_error']=True
        self.samples.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with patch.object(admission,'measurement_identity',return_value='failed-mock',create=True):
            raw=native.run(self.system,protocol.nominal_plan(self.contract),self.samples,self.contract,backend='mock',
                duration=.05,warmup=0,seed=113,drain_timeout=5.,startup_timeout=10.,evaluator=lambda *a:dict(utility=dict(U=None)))
        self.assertEqual(raw['status'],'failed');self.assertTrue(raw['priming_request_records'])
        self.assertTrue(any('Priming failed' in e for e in raw['errors']))
        self.assertTrue(raw['shutdown']);self.assertFalse(raw['request_records'])

    def test_phase_snapshot_excludes_previous_phase_even_past_completions(self):
        pending={'w':dict(phase='warmup',model='text#0',status='completed',completion_time_ns=10),
            'p':dict(phase='priming',model='text#0',status='submitted',request_id='p',task='text',pool_id='t',enqueued_time_ns=1,actual_submit_time_ns=2)}
        result=queue_snapshot(SimpleNamespace(pending=pending),['text#0'],100,'measurement')
        self.assertEqual(result,{'text#0':dict(outstanding=[],last_completed_ns=None)})

    def test_contract_parameter_replica_and_source_changes_identity(self):
        c=copy.deepcopy(self.contract);original=protocol.digest(c);c['overdue_remaining_ms']=6.
        self.assertNotEqual(original,protocol.digest(c));c=copy.deepcopy(self.contract);c['service_replica_states']['text']=1
        with self.assertRaisesRegex(ValueError,'Replica'):protocol.validate_contract(c)
        c=copy.deepcopy(self.contract);c['source_fingerprints']={str(self.samples):'0'*64}
        with self.assertRaisesRegex(ValueError,'source changed'):protocol.validate_contract(c)

    def test_each_strategy_fresh_dispatcher_and_fixed_backgrounds(self):
        for strategy in protocol.STRATEGIES:
            c=copy.deepcopy(self.contract);c['strategy']=strategy;c['mode']='quality_aware' if strategy in {'queue_quality','queue_specialist_only'} else 'latency'
            a=native.make_dispatcher(c);b=native.make_dispatcher(c)
            snapshot={alias:dict(outstanding=[],last_completed_ns=None) for alias in a.aliases}
            self.assertEqual(a.choose('text','t',0,snapshot),b.choose('text','t',0,snapshot))
            self.assertEqual(a.choose('image','image',0,snapshot)['alias'],'generalist_2#0')
            self.assertEqual(a.choose('ocr','ocr',0,snapshot)['alias'],'ocr#0')

    def test_native_binding_failure_preserves_terminal_phase_and_claim(self):
        base=self.root/'base.json';base.write_text(json.dumps(dict(jobs=[dict(identity={})])))
        identity=dict(horizon_seconds=600,warmup_seconds=30,arrival_process='poisson',measurement_claim='development only')
        job=dict(id='job',phase='development',seed=113,strategy='queue_quality',identity=identity,
            nominal_plan=protocol.nominal_plan(self.contract),policy_contract=self.contract)
        packet=dict(base_packet=dict(path=str(base)),jobs=[job]);output=self.root/'failed-native'
        static=SimpleNamespace(bind_native_job=lambda *args:(_ for _ in ()).throw(ValueError('checkpoint unavailable')))
        with patch.object(native,'validate_packet',return_value=dict(valid=True)),\
             patch.object(native,'frozen_runtime',return_value=SimpleNamespace(__package__='frozen.runtime')),\
             patch.object(native.importlib,'import_module',side_effect=[static,SimpleNamespace()]),\
             patch.object(native,'run',side_effect=AssertionError('No worker start after binding failure')):
            raw=native.execute_job(packet,'job',output,execute=True)
        self.assertEqual(json.loads((output/'run.json').read_text()),raw)
        self.assertEqual(raw['status'],'failed');self.assertEqual(raw['experiment_phase'],'development')
        self.assertEqual(raw['measurement_claim'],'development only')
        self.assertIn('checkpoint unavailable',raw['errors'][0]);self.assertIsNone(raw['evaluation']['utility']['U'])
        self.assertEqual(raw['memory_evidence'],{});self.assertEqual(raw['energy_evidence'],{})
        self.assertEqual(raw['request_records'],[]);self.assertEqual(raw['shutdown'],{})


if __name__=='__main__':unittest.main()
