"""Audit falsification and real CPU subprocess admission; never GPU evidence."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

from gsdeploy.resources import identity, prepare_registry, validate_resources
from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.runner import record
from gsdeploy.runtime.schema import MeasurementConfig
from tools.predictor_validation_20261027 import mechanism_audit as module
from tools.predictor_validation_20261027.background_collect import context_identity


def save(path, value):
    Path(path).write_text(json.dumps(value))


class MechanismAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.aliases = copy.deepcopy(module.RESIDENTS)
        self.context = {'residents': self.aliases, 'worker_bindings': {}, 'execution_sources': []}
        for name in module.SEMANTIC_PATHS:
            path = module.ROOT / name
            frozen = self.root / (name.replace('/', '_'))
            shutil.copyfile(path, frozen)
            self.context['execution_sources'].append({'original_path': str(path), 'frozen': module.receipt(frozen)})
        for family in set(self.aliases.values()):
            spec = self.root / (family + '.json')
            save(spec, {'family': family})
            self.context['worker_bindings'][family] = {'command': [sys.executable, 'unused-worker.py', '--spec', str(spec)]}
            self.context['execution_sources'].append({'original_path': str(spec), 'frozen': module.receipt(spec)})
        self.context['sources'] = [r['frozen'] for r in self.context['execution_sources']]
        self.context_path = self.root / 'context.json'
        save(self.context_path, self.context)
        self.ctx = context_identity(self.context)

    def fixture(self, shared=False, capacity=1):
        resources = validate_resources({'schema': 'gsdeploy.resource_configuration.v1',
            'mode': 'shared' if shared else 'request_admission', 'device': '0',
            'allocations': {a: {} if shared else {'max_inflight': 1} for a in self.aliases},
            **({} if shared else {'groups': {'gpu': {'aliases': sorted(self.aliases), 'max_active': capacity}}})}, self.aliases)
        trace = [{'request_id': 'measurement:0', 'offset_ns': 0, 'task': 'text', 'pool_id': 'input'},
                 {'request_id': 'measurement:1', 'offset_ns': 10, 'task': 'text', 'pool_id': 'input'}]
        candidate = {'id': 'shared' if shared else 'admission', 'execution_context_sha256': self.ctx,
                     'routes': {'measurement:0': 'text#0', 'measurement:1': 'generalist_2#0'},
                     'resource_configuration': resources}
        self.plan = {'schema': 'gsdeploy.native_flow_plan.v1', 'execution_context_sha256': self.ctx,
                     'context': module.receipt(self.context_path), 'runner_source': module.receipt(module.ROOT/'tools/predictor_validation_20261027/native_flow.py'),
                     'trace': trace, 'trace_sha256': identity(trace), 'window_seconds': .000001,
                     'aliases': {a: {'family': f} for a,f in self.aliases.items()}, 'candidates': [candidate]}
        self.plan_path = self.root/'plan.json'
        self.ready = {}
        for index, (alias, family) in enumerate(self.aliases.items()):
            self.ready[alias] = {'pid': 100+index, 'worker': family,
                                 'tasks': ['ocr'] if family == 'ocr' else ['text', 'image', 'asr'],
                                 'spec_sha256': module.receipt(self.root/(family+'.json'))['sha256']}
        records = [self.row(0, 'text#0', 1000, 1012, 1014, 1040, 1042, shared),
                   self.row(1, 'generalist_2#0', 1010, 1043, 1045, 1070, 1072, shared)]
        self.raw = {'schema': 'gsdeploy.native_flow_collection.v1', 'kind': 'measured', 'backend': 'native',
                    'status': 'completed', 'collection_complete': True, 'errors': [], 'runtime_errors': [],
                    'telemetry_errors': [], 'qualification_issues': [], 'execution_context_sha256': self.ctx,
                    'context': self.plan['context'], 'candidate_id': candidate['id'],
                    'trace_sha256': identity(trace), 'resource_configuration': resources,
                    'resident_aliases': list(self.aliases), 'replica_to_family': self.aliases,
                    'ready': self.ready, 'startup_begin_ns': 1, 'startup_end_ns': 999,
                    'phase_start_ns': 1000, 'deadline_ns': 2000, 'arrival_loop_end_ns': 2001,
                    'observation_end_ns': 2100, 'request_records': records, 'priming_request_records': [],
                    'warmup_request_records': [],
                    'shutdown': {a: {'forced': False, 'exit_code': 0, 'bye_received': True} for a in self.aliases},
                    'admission_snapshot_after': None if shared else {'active': {}, 'waiting': [], 'closed': True, 'peak_active': 1},
                    'board_samples': [{'time_ns': 2000, 'owned_aliases': list(self.aliases), 'foreign_pids': [],
                                       'processes_mib': {str(r['pid']): 10 for r in self.ready.values()}}]}
        self.raw_path = self.root/'raw.json'
        self.write()
        return self.raw

    def row(self, index, alias, scheduled, grant, start, end, receipt, shared=False):
        rid = 'measurement:'+str(index)
        value = {'request_id': rid, 'phase': 'measurement', 'task': 'text', 'pool_id': 'input',
                 'model': alias, 'status': 'completed', 'scheduled_time_ns': scheduled,
                 'enqueued_time_ns': scheduled+1, 'actual_submit_time_ns': grant+1,
                 'submit_write_completed_time_ns': grant+2, 'observed_worker_start_time_ns': start,
                 'completion_time_ns': receipt,
                 'worker_response': {'request_id': rid, 'task': 'text', 'task_type': 'text',
                                     'success': True, 'start_perf_ns': start, 'end_perf_ns': end},
                 'execution_events': [{'request_id': rid, 'pid': self.ready[alias]['pid'],
                                       'perf_ns': start, 'received_time_ns': start+1}]}
        if not shared:
            value.update(admission_wait_start_time_ns=scheduled+2, admission_granted_time_ns=grant)
        return value

    def write(self):
        self.plan['plan_sha256'] = identity({k:v for k,v in self.plan.items() if k!='plan_sha256'})
        save(self.plan_path, self.plan)
        self.raw['plan'] = module.receipt(self.plan_path)
        save(self.raw_path, self.raw)

    def check(self):
        self.write()
        return module.audit(self.raw_path)

    def test_admission_covers_complete_request_and_reports_observation_gaps(self):
        self.fixture()
        report = self.check()
        self.assertEqual(report['group_peaks']['gpu']['observed_peak'], 1)
        self.assertEqual(report['permit_intervals'], 2)
        self.assertEqual(report['fifo']['status'], 'NOT_IDENTIFIABLE')
        self.assertIn('lower bound', report['observation_limits'][0])
        self.assertEqual(report['ownership']['start_pid_matches_ready'], 2)

    def test_shared_has_no_permits_or_admission_snapshot(self):
        self.fixture(shared=True)
        report = self.check()
        self.assertEqual(report['permit_intervals'], 0)
        self.assertIsNone(report['observed_permit_peak'])
        self.raw['request_records'][0]['admission_granted_time_ns'] = 1012
        with self.assertRaisesRegex(ValueError, 'fabricate'):
            self.check()

    def test_group_capacity_violation_is_rejected(self):
        self.fixture()
        self.raw['request_records'][1] = self.row(1, 'generalist_2#0', 1010, 1015, 1017, 1070, 1072)
        with self.assertRaisesRegex(ValueError, 'group max_active'):
            self.check()

    def test_per_alias_capacity_violation_is_rejected(self):
        self.fixture(capacity=2)
        self.plan['candidates'][0]['routes']['measurement:1'] = 'text#0'
        self.raw['request_records'][1] = self.row(1, 'text#0', 1010, 1015, 1017, 1070, 1072)
        with self.assertRaisesRegex(ValueError, 'per-alias max_inflight'):
            self.check()

    def test_wait_start_inversion_does_not_claim_fifo_failure(self):
        self.fixture(capacity=2)
        self.raw['request_records'][0] = self.row(0, 'text#0', 1000, 1030, 1032, 1080, 1082)
        self.raw['request_records'][1] = self.row(1, 'generalist_2#0', 1010, 1020, 1022, 1070, 1072)
        self.raw['admission_snapshot_after']['peak_active'] = 2
        report = self.check()
        self.assertEqual(report['fifo']['observed_wait_start_order_inversions'], 1)
        self.assertEqual(report['fifo']['status'], 'NOT_IDENTIFIABLE')

    def test_worker_receipt_timing_and_pid_are_bound(self):
        self.fixture()
        self.raw['request_records'][0]['completion_time_ns'] = 1039
        with self.assertRaisesRegex(ValueError, 'order violated'):
            self.check()
        self.raw['request_records'][0]['completion_time_ns'] = 1042
        self.raw['request_records'][0]['execution_events'][0]['pid'] = 999
        with self.assertRaisesRegex(ValueError, 'PID binding'):
            self.check()

    def test_resource_plan_route_and_source_changes_are_rejected(self):
        self.fixture()
        self.raw['resource_configuration'] = copy.deepcopy(self.raw['resource_configuration'])
        self.raw['resource_configuration']['groups']['gpu']['max_active'] = 2
        with self.assertRaisesRegex(ValueError, 'configuration'):
            self.check()
        self.fixture()
        self.plan['candidates'][0]['routes']['measurement:1'] = 'text#1'
        with self.assertRaisesRegex(ValueError, 'route'):
            self.check()
        self.fixture()
        frozen = Path(self.context['execution_sources'][0]['frozen']['path'])
        frozen.write_text('# changed frozen runtime\n')
        with self.assertRaisesRegex(ValueError, 'source receipt'):
            self.check()

    def test_drain_clean_shutdown_and_owned_telemetry_are_required(self):
        self.fixture()
        self.raw['admission_snapshot_after']['active'] = {'leak': 'text#0'}
        with self.assertRaisesRegex(ValueError, 'drained'):
            self.check()
        self.raw['admission_snapshot_after']['active'] = {}
        self.raw['shutdown']['text#0']['forced'] = True
        with self.assertRaisesRegex(ValueError, 'shutdown'):
            self.check()
        self.raw['shutdown']['text#0']['forced'] = False
        self.raw['board_samples'][0]['foreign_pids'] = [900]
        with self.assertRaisesRegex(ValueError, 'ownership'):
            self.check()

    def test_native_source_annotation_cannot_change_timing(self):
        self.fixture()
        native_path = self.root/'native.json'
        save(native_path, self.raw)
        self.raw['native_raw'] = module.receipt(native_path)
        self.raw['quality_evidence'] = {'ignored': 'inference unchanged'}
        self.assertEqual(self.check()['status'], 'PASS_OBSERVED_CONTRACT')
        self.raw['request_records'][0]['completion_time_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'Annotated raw differs'):
            self.check()

    def test_real_cpu_runtime_admission_receipts_pass_audit(self):
        """Real sender/collector/gate/shutdown; synthetic worker, no CUDA imports."""
        self.fixture(capacity=1)
        worker_path = self.root/'cpu_worker.py'
        worker_path.write_text('''import hashlib,json,os,sys,time
from pathlib import Path
spec=Path(sys.argv[sys.argv.index('--spec')+1]);family=json.loads(spec.read_text())['family']
def emit(kind,data):print(kind+'|'+json.dumps(data),flush=True)
emit('READY',dict(pid=os.getpid(),worker=family,tasks=['text','image','asr','ocr'],spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest()))
for line in sys.stdin:
    r=json.loads(line)
    if r.get('op')=='shutdown':emit('BYE',dict(pid=os.getpid()));break
    start=time.perf_counter_ns();emit('START',dict(request_id=r['request_id'],pid=os.getpid(),perf_ns=start))
    time.sleep(.01);end=time.perf_counter_ns()
    emit('RESULT',dict(request_id=r['request_id'],task_type=r['task_type'],task=r['task_type'],success=True,start_perf_ns=start,end_perf_ns=end))
''')
        for entry in self.context['worker_bindings'].values():
            entry['command'][1] = str(worker_path)
        save(self.context_path, self.context)
        self.ctx = context_identity(self.context)
        self.plan['execution_context_sha256'] = self.ctx
        self.plan['context'] = module.receipt(self.context_path)
        self.plan['candidates'][0]['execution_context_sha256'] = self.ctx
        self.raw.update(context=self.plan['context'], execution_context_sha256=self.ctx)
        resources = self.plan['candidates'][0]['resource_configuration']
        registry = prepare_registry({a: copy.deepcopy(self.context['worker_bindings'][f]) for a,f in self.aliases.items()}, resources)
        config = MeasurementConfig(deployment=dict.fromkeys(self.aliases,1), workload_rps={'text':1.},
            routing_rps={'text':dict.fromkeys(self.aliases,.25)},startup_timeout_seconds=10,shutdown_timeout_seconds=5)
        runtime = DeploymentRuntime(config,registry)
        self.raw['startup_begin_ns'] = time.perf_counter_ns()
        try:
            runtime.start(); self.raw['startup_end_ns'] = time.perf_counter_ns()
            start=time.perf_counter_ns();self.raw['phase_start_ns']=start
            self.raw['deadline_ns']=start+1000
            rows=[]
            for i,declared in enumerate(self.plan['trace']):
                row=record('measurement',i,'text',self.plan['candidates'][0]['routes'][declared['request_id']],
                           {'pool_id':'input','sample':{'id':'input'}},start+declared['offset_ns'])
                rows.append(row);runtime.enqueue(row,{'id':'input'})
            self.raw['arrival_loop_end_ns']=time.perf_counter_ns()
            runtime.drain(rows,5)
            self.raw['observation_end_ns']=time.perf_counter_ns()
            self.raw['request_records']=rows;self.raw['ready']=copy.deepcopy(runtime.ready)
            self.raw['runtime_errors']=runtime.errors
            self.raw['board_samples']=[{'time_ns':self.raw['observation_end_ns'],
                'owned_aliases':list(self.aliases),'foreign_pids':[],
                'processes_mib':{str(r['pid']):10 for r in runtime.ready.values()}}]
        finally:
            runtime.close()
        self.raw['shutdown']=runtime.shutdown_report
        self.raw['admission_snapshot_after']=runtime.admission.snapshot()
        report=self.check()
        self.assertEqual(report['permit_intervals'],2)
        self.assertEqual(report['observed_permit_peak'],1)
        self.assertGreater(report['admission_wait_ms']['max'],1.)


if __name__ == '__main__':
    unittest.main()
