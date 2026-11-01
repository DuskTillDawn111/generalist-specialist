import copy
import json
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import twoFIFO_fresh181_execute_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_fresh181_prefix_v1 as p
from tools.predictor_validation_20261027.stage_response_features_v1 import checked, receipt
from tests.test_twoFIFO_owned_collect_v2 import FakeMonitor

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'artifacts/predictor-twoFIFO-prospective-public-contract-v1-20261029-cpu02-prepared/report.json'
MATCHING=ROOT/'artifacts/predictor-twoFIFO-matching-owned-producer-v1-20261029-cpu02-prepared/manifest.json'


class Clock:
    def __init__(self):self.now=100
    def clock(self):return self.now
    def wait(self,at):self.now=max(at,self.now)


class Runtime:
    """Protocol-complete ordered prior history, CPU transport only."""
    def __init__(self,clock,registry,*,incomplete=False,unsettled=False,wrong_spec=False):
        self.clock=clock;self.lock=threading.RLock();self.events=queue.Queue();self.pending={};self.errors=[]
        self.ready={owner:dict(pid=i+1,clock='time.perf_counter_ns',spec_sha256=b['spec_sha256'])
            for i,(owner,b) in enumerate(registry.items())}
        if wrong_spec:self.ready[m.OWNERS[0]]['spec_sha256']='0'*64
        self.enqueued=[];self.drain_counts=[];self.closed=False;self.shutdown_report={}
        self.incomplete=incomplete;self.unsettled=unsettled
    def start(self):pass
    def enqueue(self,row,sample):
        at=self.clock.clock();self.pending[row['request_id']]=row;self.enqueued.append(row['request_id'])
        row.update(status='submitted',enqueued_time_ns=at,actual_submit_time_ns=at,
            submit_write_completed_time_ns=at+1)
    def drain(self,rows,timeout):
        self.drain_counts.append(len(self.enqueued))
        for row in rows:
            if row['status']!='submitted':continue
            if row['phase']!='priming':
                if not self.incomplete:row.update(status='completed',completion_time_ns=self.clock.clock())
                continue
            at=self.clock.clock();pid=self.ready[row['model']]['pid']
            row.update(status='completed',observed_worker_start_time_ns=at+2,
                worker_response=dict(success=True,start_perf_ns=at+2,end_perf_ns=at+7),
                completion_time_ns=at+8,response_reader_event_enqueue_time_ns=at+8,
                response_collector_dequeued_time_ns=at+9,response_collector_processing_time_ns=at+10,
                execution_events=[dict(pid=pid,perf_ns=at+2,received_time_ns=at+3,
                    collector_dequeued_time_ns=at+4,collector_processing_time_ns=at+5)])
            if not self.unsettled:row['previous_handler_next_request_ready']=dict(
                request_id=row['request_id'],pid=pid,perf_ns=at+11,received_time_ns=at+12,
                collector_dequeued_time_ns=at+13,collector_processing_time_ns=at+14,
                clock='host_perf_counter_ns',boundary='previous_handler_next_request_ready',
                scope='after_response_flush_before_next_read')
            self.clock.now=at+15
    def close(self):self.closed=True


def forecast(gate):
    return dict(schema=p.FORECAST_SCHEMA,validation_labels_read=False,target_future_labels_read=False,
        gate_request_sha256=p.digest(gate),public_requests=gate['public_requests'],model_sources=gate['model_sources'],
        forecast_generator_sources=gate['forecast_generator_sources'],resource_horizon_qualified=False,
        full_U_qualified=False,draws=[dict(draw_id='all-unknown',weight=dict(numerator=1,denominator=1),
            status='UNKNOWN_WORLD',requests=[dict(request_id=r['request_id'],completion_time_ns=None)
                for r in gate['public_requests']])])


class FreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.prepared=Path(cls.tmp.name)/'prepared'
        cls.manifest_ref=m.prepare(PUBLIC,MATCHING,'781ace6cfa7f4bc285129dd30e8a39f1',cls.prepared)
        cls.manifest=checked(cls.manifest_ref);cls.plan=checked(cls.manifest['attempts'][0]['plan'])
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_exact_preparation_preserves_original_public181_backend_and_priming(self):
        manifest=m.audit(self.manifest_ref['path']);contract=checked(self.plan['public_contract'])
        training=checked(receipt(MATCHING));training_ids={r['execution_id'] for r in training['attempts']}
        self.assertEqual((manifest['candidate_count'],manifest['measured_offered'],manifest['priming_history_offered']),(4,724,72))
        self.assertEqual(Path(manifest['context']['path']).read_bytes(),Path(self.plan['matching_parent_context']['path']).read_bytes())
        for attempt,candidate in zip(manifest['attempts'],contract['candidates']):
            plan=m.audit_plan(attempt['plan']['path']);self.assertNotIn(plan['execution_id'],training_ids)
            self.assertEqual((plan['measurement_ns'],plan['drain_ns']),(90_000_000_000,120_000_000_000))
            self.assertEqual(plan['candidate'],candidate);self.assertEqual(plan['runtime_config']['workload_rps'],contract['workload_rps'])
            for actual,original in zip(plan['cases'][0]['arrivals'],contract['trace']):
                self.assertEqual((actual['request_id'],actual['arrival_ns'],actual['owner']),
                    (original['request_id'],original['offset_ns'],candidate['routes'][original['request_id']]))
            self.assertEqual(plan['priming_semantics']['sha256'],self.plan['priming_semantics']['sha256'])
            self.assertEqual(len(plan['priming']),18);self.assertFalse(plan['final_gain_eligible'])
        self.assertEqual(contract['weights'],{k:1/6 for k in 'QTLPME'})

    def test_rehashed_plan_mutations_cannot_change_forecast_window_admission_or_sources(self):
        path=Path(self.manifest['attempts'][0]['plan']['path']);original=path.read_bytes()
        mutations=[lambda d:d.update(measurement_ns=10_000_000_000),
            lambda d:d['runtime_config'].update(max_pending_requests=2),
            lambda d:d['cases'][0]['arrivals'][0].update(arrival_ns=17),
            lambda d:d['forecast_barrier'].update(no_target_future_label_input=False),
            lambda d:d['priming'][0].update(pool_id='foreign'),lambda d:d.update(final_gain_eligible=True),
            lambda d:d.update(sources=[]),lambda d:d['runtime_policy'].update(global_admission_cap=2)]
        try:
            for mutate in mutations:
                value=copy.deepcopy(self.plan);mutate(value);value.pop('execution_context_sha256');value['execution_context_sha256']=m.digest(value)
                path.write_text(json.dumps(value))
                with self.subTest(value=value.get('measurement_ns')),self.assertRaises(ValueError):m.audit(self.manifest_ref['path'])
        finally:path.write_bytes(original)
        self.assertEqual(receipt(path),self.manifest['attempts'][0]['plan'])

    def run_collector(self,callback='seal',*,incomplete=False,unsettled=False,wrong_spec=False,checkpoint_late=False):
        clock=Clock();holder={};seen={}
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);output=folder/'attempt'/'collection';plan=copy.deepcopy(self.plan)
            for key,name in [('prefix_path','precohort-prefix.json'),('gate_request_path','prediction-gate.json'),('seal_path','prediction-seal.json')]:
                plan['forecast_barrier'][key]=str(output/name)
            path=folder/'attempt'/'plan.json';p.save(path,plan);p.save(folder/'manifest.json',dict(CPU_TEST_ONLY=True))
            def factory(config,registry):
                holder['runtime']=Runtime(clock,registry,incomplete=incomplete,unsettled=unsettled,wrong_spec=wrong_spec)
                return holder['runtime']
            def gate_callback(gate_ref):
                gate=checked(gate_ref);prefix=checked(gate['prefix']);runtime=holder['runtime']
                self.assertEqual(len(runtime.enqueued),18);self.assertTrue(Path(gate['prefix']['path']).is_file())
                self.assertEqual(len(prefix['prefix_snapshot']['request_records']),18)
                self.assertEqual(len(gate['public_requests']),181);self.assertNotIn('raw',gate)
                seen.update(gate=copy.deepcopy(gate),prefix=copy.deepcopy(prefix))
                if callable(callback):callback(gate_ref,plan,clock)
                if callback=='none':return
                if callback=='late':clock.wait(gate['forecast_origin_ns'])
                pred=p.save(output/'forecast.json',forecast(gate))
                p.seal_prediction(gate_ref['path'],pred['path'],gate['expected_seal_path'],clock_ns=clock.clock)
                if checkpoint_late:clock.wait(gate['forecast_origin_ns'])
            def ownership(values,*args,**kwargs):
                pid=next(iter(values));return dict(owned_aliases=[m.OWNERS[pid-1]],unresolved_nvml_pids=[],
                    foreign_pids=[],ownership_evidence=[dict(CPU_TEST_ONLY=True,live=True)])
            with patch.object(m,'audit_plan',return_value=plan),patch.object(m.supervisor,'validate_lease'):
                raw=m.collect(path,output,lease_fd=-1,runtime_factory=factory,monitor_factory=FakeMonitor,
                    clock_ns=clock.clock,wait_until_ns=clock.wait,ownership_check=ownership,gate_callback=gate_callback)
            saved=checked(receipt(output/'raw.json'));self.assertEqual(raw,saved)
            if 'precohort_prefix' in raw:
                seen['prefix_after']=p.audit_prefix(raw['precohort_prefix'],receipt(path),plan,receipt(folder/'manifest.json'),allow_cpu_fixture=True)
            return copy.deepcopy(raw),holder['runtime'],clock,seen

    def test_actual_control_flow_seals_before_first_target_and_enqueues_full181_independently(self):
        raw,runtime,clock,seen=self.run_collector()
        self.assertEqual(raw['status'],'completed',raw['errors']);self.assertTrue(runtime.closed)
        self.assertEqual(len(runtime.enqueued),199);self.assertEqual(runtime.drain_counts[-1],199)
        self.assertTrue(raw['predictions_durable_before_cohort']);self.assertTrue(raw['all_requests_completed'])
        self.assertEqual(raw['backend'],'CPU_TEST_ONLY');self.assertFalse(raw['main_queue_provider_eligible'])
        epoch=raw['planned_cohort_epoch_ns'];window=raw['measurement_windows'][0]
        self.assertLess(raw['prediction_seal_accepted_ns'],epoch)
        self.assertEqual(window,dict(case_id='original181',start_ns=epoch,end_ns=epoch+90_000_000_000,
            fixed_drain_deadline_ns=epoch+210_000_000_000))
        self.assertEqual([r['actual_submit_time_ns']-epoch for r in raw['request_records']],
            [r['arrival_ns'] for r in self.plan['cases'][0]['arrivals']])
        self.assertTrue(raw['no_mutating_checkpoint_during_case']);self.assertEqual(raw['board_samples'][-1]['phase'],'final_drain')
        self.assertEqual(seen['prefix'],seen['prefix_after']);self.assertEqual(clock.now,epoch+210_000_000_001)

    def test_no_seal_late_seal_or_late_accept_never_sends_target(self):
        for callback,late in [('none',False),('late',False),('seal',True)]:
            with self.subTest(callback=callback,late=late):
                raw,runtime,_,_=self.run_collector(callback,checkpoint_late=late)
                self.assertEqual(raw['status'],'failed');self.assertEqual(len(runtime.enqueued),18)
                self.assertTrue(runtime.closed);self.assertFalse(raw['target_response_labels_collected'])
                self.assertEqual(len(raw['request_records']),181)
                self.assertTrue(all(r['status']=='not_started' for r in raw['request_records']))

    def test_missing_ready_or_foreign_spec_never_coerces_busy_prefix(self):
        for options in [dict(unsettled=True),dict(wrong_spec=True)]:
            raw,runtime,_,seen=self.run_collector(**options)
            self.assertEqual(raw['status'],'failed');self.assertEqual(len(runtime.enqueued),18)
            self.assertFalse(seen);self.assertFalse(raw['predictions_durable_before_cohort'])
            if options.get('unsettled'):self.assertEqual(raw['causal_prefix_status'],'UNKNOWN_BUSY_OR_UNPROVED_PRIOR_READY')

    def test_incomplete_preserves_all_offered_and_original_fixed_deadline(self):
        raw,runtime,_,_=self.run_collector(incomplete=True)
        self.assertEqual(raw['status'],'completed',raw['errors']);self.assertFalse(raw['all_requests_completed'])
        self.assertEqual(len(raw['request_records']),181);self.assertEqual(len(runtime.enqueued),199)
        self.assertTrue(all(r['status']=='incomplete' for r in raw['request_records']))
        self.assertEqual(raw['cases'][0]['status'],'incomplete');self.assertIsNone(raw['U'])

    def test_prefix_requires_complete_priming_exact_spec_and_no_future_or_unsettled_messages(self):
        _,_,_,seen=self.run_collector();source=seen['prefix']['prefix_snapshot'];epoch=seen['gate']['forecast_origin_ns']
        changes=[lambda s:s['request_records'].pop(),lambda s:s['request_records'][0].update(request_id='measurement:0'),
            lambda s:s.update(queued_runtime_events=[{}]),lambda s:s.update(pending_request_ids=[]),
            lambda s:s['request_records'][0].pop('previous_handler_next_request_ready'),
            lambda s:s['request_records'][0].update(submit_write_completed_time_ns=epoch+1),
            lambda s:s['ready'][m.OWNERS[0]].update(spec_sha256='0'*64)]
        for change in changes:
            snapshot=copy.deepcopy(source);change(snapshot)
            with self.subTest(change=change),self.assertRaises((ValueError,UnknownEvidence)):p.derive(self.plan,snapshot,epoch)

    def test_fixed_full_world_mass_and_model_binding_before_numeric_forecast(self):
        _,_,_,seen=self.run_collector();gate=seen['gate'];valid=forecast(gate);p.validate_forecast(valid,gate)
        changes=[lambda d:d['draws'][0]['requests'].pop(),lambda d:d['draws'][0]['weight'].update(numerator=2),
            lambda d:d['draws'][0]['weight'].update(numerator=0),lambda d:d['draws'][0].update(status='GENERATED_WORLD'),
            lambda d:d['draws'][0]['requests'][0].update(completion_time_ns=123),
            lambda d:d.update(model_sources=[dict(path='/foreign',sha256='0'*64)]),lambda d:d.update(full_U_qualified=True),
            lambda d:d.update(validation_labels_read=True),lambda d:d['draws'][0]['requests'][0].update(owner='foreign'),
            lambda d:d['draws'][0].update(qualified=True),lambda d:d.update(transfer_qualified=True)]
        for change in changes:
            value=copy.deepcopy(valid);change(value)
            with self.subTest(change=change),self.assertRaises(ValueError):p.validate_forecast(value,gate)

    def test_resealed_gate_cannot_change_plan_calendar_or_six_term_contract(self):
        def probe(gate_ref,plan,clock):
            path=Path(gate_ref['path']);original=path.read_bytes();gate=checked(gate_ref)
            changes=[lambda d:d['public_requests'][0].update(arrival_ns=d['forecast_origin_ns']+99),
                lambda d:d.update(execution_id='0'*32),lambda d:d['six_term_weights'].update(P=0),
                lambda d:d.update(stability_policy='changed'),lambda d:d.update(expected_seal_path='/foreign/seal.json'),
                lambda d:d.update(forecast_origin_ns=d['forecast_origin_ns']+1)]
            try:
                for change in changes:
                    value=copy.deepcopy(gate);change(value);path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):p.audit_gate(receipt(path),allow_cpu_fixture=True)
            finally:path.write_bytes(original)
        raw,_,_,_=self.run_collector(probe);self.assertEqual(raw['status'],'completed',raw['errors'])

    def test_rehashed_prefix_seed_or_lifetime_does_not_replace_observed_snapshot(self):
        def probe(gate_ref,plan,clock):
            gate=checked(gate_ref);path=Path(gate['prefix']['path']);original=path.read_bytes();document=checked(gate['prefix'])
            changes=[lambda d:d['seed'].update(sequence=d['seed']['sequence']+1),
                lambda d:d.update(execution_id='0'*32),lambda d:d.update(independent_accuracy_qualified=True),
                lambda d:d.update(source_authenticated_by_independent_execution_review=True)]
            try:
                for change in changes:
                    value=copy.deepcopy(document);change(value);path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):p.audit_prefix(receipt(path),gate['plan'],plan,gate['manifest'],allow_cpu_fixture=True)
            finally:path.write_bytes(original)
        raw,_,_,_=self.run_collector(probe);self.assertEqual(raw['status'],'completed',raw['errors'])

    def test_seal_persistence_crossing_origin_is_rejected_and_never_reused(self):
        def probe(gate_ref,plan,clock):
            gate=checked(gate_ref);prediction=p.save(Path(gate_ref['path']).parent/'crossing-forecast.json',forecast(gate))
            times=iter((clock.now,gate['forecast_origin_ns']))
            with self.assertRaisesRegex(ValueError,'persistence crossed'):
                p.seal_prediction(gate_ref['path'],prediction['path'],gate['expected_seal_path'],clock_ns=lambda:next(times))
            with self.assertRaisesRegex(ValueError,'before the exact'):
                p.audit_seal(receipt(gate['expected_seal_path']),gate_ref,now_ns=gate['forecast_origin_ns'],allow_cpu_fixture=True)
            Path(gate['expected_seal_path']).unlink()
        raw,_,_,_=self.run_collector(probe);self.assertEqual(raw['status'],'completed',raw['errors'])

    def test_cpu_interface_world_cannot_authenticate_as_native_source_or_forecast(self):
        def probe(gate_ref,plan,clock):
            with self.assertRaisesRegex(ValueError,'role/source'):
                p.audit_gate(gate_ref)
            gate=checked(gate_ref)
            with self.assertRaisesRegex(ValueError,'actual READY|pinned actual|ancestry'):
                p.inspect_native_ownership(checked(gate['prefix'])['prefix_snapshot'],
                    checked(gate['prefix'])['ready_ownership_evidence'])
        raw,_,_,_=self.run_collector(probe);self.assertEqual(raw['status'],'completed',raw['errors'])

    def test_change_restore_unselected_plan_during_first_audit_blocks_launch_before_claim(self):
        selected=self.manifest['attempts'][0];other=Path(self.manifest['attempts'][1]['plan']['path']);original=other.read_bytes()
        original_audit=m.audit
        def change_and_audit(path):
            other.write_bytes(original+b' ');other.write_bytes(original)
            return original_audit(path)
        with patch.object(m,'audit',side_effect=change_and_audit),patch.object(m,'require_native_forecast_bridge'),\
                patch.object(m.supervisor,'public_environment_gate'),\
                patch.object(m.supervisor,'supervise') as launch:
            with self.assertRaises(ValueError):m.execute(selected['plan']['path'],selected['collection'],manifest_path=self.manifest_ref['path'])
            launch.assert_not_called()
        self.assertFalse(Path(selected['claim']).exists());self.assertEqual(receipt(other),self.manifest['attempts'][1]['plan'])

    def test_native_missing_transfer_bridge_holds_before_claim_environment_or_runtime(self):
        item=self.manifest['attempts'][0]
        with patch.object(m.supervisor,'public_environment_gate') as env,patch.object(m.supervisor,'supervise') as launch:
            with self.assertRaisesRegex(UnknownEvidence,m.NATIVE_BRIDGE_HOLD):
                m.execute(item['plan']['path'],item['collection'],manifest_path=self.manifest_ref['path'])
            env.assert_not_called();launch.assert_not_called()
        self.assertFalse(Path(item['claim']).exists());self.assertFalse(Path(item['collection']).exists())
        with patch.object(m.supervisor,'validate_lease'),patch.object(m,'audit_plan',return_value=self.plan),\
                patch.object(m,'MeasurementConfig') as config:
            with self.assertRaisesRegex(UnknownEvidence,m.NATIVE_BRIDGE_HOLD):
                m.collect(item['plan']['path'],item['collection'],lease_fd=-1)
            config.assert_not_called()

    def test_generated_word_and_arbitrary_receipts_cannot_bypass_native_bridge_hold(self):
        _,_,_,seen=self.run_collector();gate=copy.deepcopy(seen['gate'])
        # The generic CPU hypothesis schema binds sources, but is explicitly
        # not the missing native law/replay authenticator.
        gate.update(model_sources=[dict(path='/arbitrary',sha256='0'*64)],
            forecast_generator_sources=[dict(path='/also-arbitrary',sha256='1'*64)])
        value=forecast(gate);value['draws'][0]['status']='GENERATED_WORLD'
        with self.assertRaisesRegex(ValueError,'all-UNKNOWN'):p.validate_forecast(value,gate)
        value['draws'][0]['requests'][0]['completion_time_ns']=gate['public_requests'][0]['arrival_ns']
        p.validate_forecast(value,gate)
        plan=copy.deepcopy(self.plan);plan.update(qualified=True,transfer_qualified=True,native_forecast_transfer_bridge='PASS')
        with self.assertRaisesRegex(UnknownEvidence,m.NATIVE_BRIDGE_HOLD):m.require_native_forecast_bridge(plan)

    def test_completed_execution_requires_exact_candidate_and_no_retry_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            item=copy.deepcopy(self.manifest['attempts'][0]);item['claim']=str(Path(temporary)/'claim.json')
            claim=p.save(item['claim'],dict(schema=m.SCHEMA+'.claim',manifest=self.manifest_ref,plan=item['plan'],
                execution_id=item['execution_id'],attempt_id=item['attempt_id'],candidate_id=item['candidate_id'],no_retries=True))
            execution=dict(profile_execution=dict(schema=m.SCHEMA+'.execution_binding',manifest=self.manifest_ref,plan=item['plan'],
                execution_id=item['execution_id'],attempt_id=item['attempt_id'],candidate_id=item['candidate_id'],
                raw_schema=m.RAW_SCHEMA,claim=claim,predictions_required_before_cohort=True,automatic_fit=False),
                budget=m.supervisor.BUDGET,status='CHILD_EXITED_REQUIRES_PROFILE_REVIEW',collection_subprocess_launched=True,
                process_group=123,child_reaped=True,live_group_pids=[],experiment_lock_held_through_group_exit=True,
                all_owned_descendants_exited_and_reaped=True,final_source_audit='PASS',watchdog=dict(status='EXITED',return_code=0))
            m.completed_execution(execution,self.manifest_ref,item)
            changes=[lambda d:d['profile_execution'].update(candidate_id='foreign'),lambda d:d.update(final_source_audit='FAIL'),
                lambda d:d['watchdog'].update(return_code=1),lambda d:d.update(live_group_pids=[321]),
                lambda d:d['profile_execution'].update(predictions_required_before_cohort=False)]
            for change in changes:
                value=copy.deepcopy(execution);change(value)
                with self.assertRaises(ValueError):m.completed_execution(value,self.manifest_ref,item)


if __name__=='__main__':unittest.main()
