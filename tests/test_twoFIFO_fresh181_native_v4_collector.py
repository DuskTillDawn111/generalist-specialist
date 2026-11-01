"""CPU transport tests for native4's exact shared dispatcher and live barrier.

The expected producer's numerical internals have their own author suite. Here
only that CPU computation is injected; actual dispatch, prefix, ledger, durable
files and change/restore guards remain real. No native source is qualified.
"""
import copy
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from tools.predictor_validation_20261027 import twoFIFO_fresh181_native_v4 as m
from tools.predictor_validation_20261027.stage_response_features_v1 import receipt,checked
from tests.test_twoFIFO_fresh181_execute_v1 import PUBLIC,MATCHING,Runtime,Clock
from tests.test_twoFIFO_owned_collect_v2 import FakeMonitor


class NativeV4CollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.fit_manifest=m.save(cls.root/'fit-manifest.json',dict(executor_manifest=receipt(MATCHING)))
        cls.fit_ref=m.save(cls.root/'fit.json',dict(CPU_SYNTHETIC_ONLY=True))
        cls.fit=dict(manifest=cls.fit_manifest,sources=[])
        cls.resource_ref=m.save(cls.root/'resource.json',dict(CPU_SYNTHETIC_ONLY=True))
        cls.resource_reviews=[m.save(cls.root/('review'+str(i)+'.json'),dict(CPU_SYNTHETIC_ONLY=True)) for i in range(8)]
        cls.budget_ref=m.save(cls.root/'budget.json',dict(CPU_SYNTHETIC_ONLY=True))
        cls.budget=dict(public_preparation=receipt(PUBLIC),fits_predeclared_budget=False,sources=[],
            full_U_live_pipeline_included=True,resource_manifest=cls.resource_ref,resource_reviews=cls.resource_reviews,
            evaluation_protocol=m.expected.EVALUATION_PROTOCOL)
        with patch.object(m.response,'audit_fit',return_value=dict(report=cls.fit)),patch.object(m.timing,'audit',return_value=cls.budget):
            cls.manifest_ref=m.prepare(PUBLIC,MATCHING,cls.fit_ref['path'],'675977027c2b4a57ae6da9527fc8f251',cls.root/'prepared',
                budget_report_path=cls.budget_ref['path'],resource_manifest_path=cls.resource_ref['path'],
                resource_review_paths=[r['path'] for r in cls.resource_reviews])
        cls.manifest=checked(cls.manifest_ref);cls.plan=checked(cls.manifest['attempts'][0]['plan']);cls.counter=0
        m.save(cls.root/'manifest.json',dict(CPU_SYNTHETIC_ONLY=True))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def collect(self,mode=None,*,incomplete=False,unsettled=False):
        type(self).counter+=1;folder=self.root/('run'+str(self.counter));folder.mkdir()
        plan=copy.deepcopy(self.plan);out=folder/'collection';out.mkdir()
        barrier=plan['forecast_barrier']
        for key,name in [('prefix_path','prefix.json'),('gate_request_path','gate.json'),('expected_report_path','expected.json'),('seal_path','seal.json')]:
            barrier[key]=str(out/name)
        plan_ref=m.save(folder/'plan.json',plan)
        clock=Clock();holder={};calls=[];sessions=(SimpleNamespace(queue=object()),SimpleNamespace(queue=object()))
        def factory(config,registry):
            holder['registry']=registry;holder['runtime']=Runtime(clock,registry,incomplete=incomplete,unsettled=unsettled)
            return holder['runtime']
        def ownership(values,*args,**kwargs):
            pid=next(iter(values));return dict(owned_aliases=[m.OWNERS[pid-1]],unresolved_nvml_pids=[],foreign_pids=[],
                ownership_evidence=[dict(CPU_TEST_ONLY=True,live=True)])
        def produce(session,gate_path,output,**kwargs):
            self.assertIs(session,sessions[0]);calls.append('produce');gate=checked(receipt(gate_path));holder['epoch']=gate['forecast_origin_ns']
            result=m.save(output,dict(CPU_TEST_ONLY=True,gate=receipt(gate_path),prefix=gate['prefix']))
            if mode=='late_producer':clock.wait(gate['forecast_origin_ns'])
            return result
        def seal(session,gate_path,report_path,output,**kwargs):
            self.assertIs(session,sessions[1]);calls.append('seal_two_pass_auditor')
            self.assertEqual(kwargs['report_reference'],receipt(report_path));self.assertFalse(any(r['phase']!='priming' for r in holder['runtime'].pending.values()))
            result=m.save(output,dict(CPU_TEST_ONLY=True,expected_report=receipt(report_path),seal_begin_ns=clock.clock()))
            if mode=='late_seal':clock.wait(holder['epoch'])
            if mode in ('report_restore','prefix_restore'):
                ref=receipt(report_path) if mode=='report_restore' else checked(receipt(gate_path))['prefix']
                path=Path(ref['path']);before=path.read_bytes();path.write_bytes(before+b' ');path.write_bytes(before)
            return result
        def accept(session,gate_ref,seal_ref,**kwargs):
            calls.append('accept');self.assertIs(session,sessions[1])
            m.require(kwargs['now_ns']<holder['epoch'],'CPU injected seal crossed origin')
        atomic=m.atomic_save
        def atomic_save(path,value):
            result=atomic(path,value)
            if mode=='late_checkpoint' and value.get('checkpoint_context',{}).get('kind')=='precase':clock.wait(holder['epoch'])
            return result
        def wait(at):
            if mode=='postaccept_restore' and at==holder.get('epoch'):
                path=Path(barrier['expected_report_path']);before=path.read_bytes();path.write_bytes(before+b' ');path.write_bytes(before)
            clock.wait(at)
        with patch.object(m,'audit_plan',return_value=plan),patch.object(m.supervisor,'validate_lease'),\
                patch.object(m.expected,'produce',side_effect=produce),patch.object(m.expected,'seal',side_effect=seal),\
                patch.object(m.expected,'audit_seal_structure',side_effect=accept),patch.object(m,'atomic_save',side_effect=atomic_save):
            raw=m.collect_cpu(plan_ref['path'],out,lease_fd=-1,sessions=(sessions[0],sessions[0] if mode=='same_session' else sessions[1]),
                runtime_factory=factory,monitor_factory=FakeMonitor,clock_ns=clock.clock,wait_until_ns=wait,ownership_check=ownership)
        return raw,holder['runtime'],clock,calls,checked(receipt(out/m.ACTIVITY_FILE)),plan

    def test_shared_dispatcher_full181_exact_snapshot300_90plus120_and_two_pass(self):
        raw,runtime,clock,calls,ledger,plan=self.collect()
        self.assertEqual(raw['status'],'completed',raw['errors']);self.assertEqual(len(runtime.enqueued),199)
        self.assertEqual(calls,['produce','seal_two_pass_auditor','accept']);self.assertEqual(raw['live_expected_forecast_passes'],2)
        self.assertEqual(raw['planned_cohort_epoch_ns'],raw['case_histories'][0]['prior_history_snapshot']['snapshot_time_ns']+300_000_000_000)
        self.assertEqual(raw['executed_worker_registry'],m.resource.campaign.planned_worker_registry(checked(plan['context'])))
        self.assertEqual(raw['measurement_windows'][0]['fixed_drain_deadline_ns'],raw['planned_cohort_epoch_ns']+210_000_000_000)
        self.assertEqual(len(ledger['checkpoint_spans']),17);self.assertTrue(runtime.closed)
        self.assertTrue(raw['no_mutating_checkpoint_during_case']);self.assertTrue(raw['predictions_durable_before_cohort'])
        from tools.predictor_validation_20261027 import twoFIFO_fresh181_validation_review_v1 as review
        self.assertEqual(review.inspect_ledger(plan,raw,ledger)['complete_spans'],17)

    def test_all_deadline_boundaries_and_same_session_preserve_zero_target(self):
        for mode in ('late_producer','late_seal','late_checkpoint','same_session'):
            with self.subTest(mode=mode):
                raw,runtime,_,_,_,_=self.collect(mode)
                self.assertEqual(raw['status'],'failed',raw['errors']);self.assertEqual(len(runtime.enqueued),18)
                self.assertTrue(runtime.closed);self.assertEqual(len(raw['request_records']),181)
                self.assertTrue(all(r['status']=='not_started' for r in raw['request_records']))

    def test_report_prefix_change_restore_and_postaccept_change_restore_zero_target(self):
        for mode in ('report_restore','prefix_restore','postaccept_restore'):
            with self.subTest(mode=mode):
                raw,runtime,_,_,_,_=self.collect(mode)
                self.assertEqual(raw['status'],'failed');self.assertEqual(len(runtime.enqueued),18);self.assertTrue(runtime.closed)

    def test_missing_READY_settle10s_and_original_censoring_remain_explicit(self):
        raw,runtime,clock,calls,_,_=self.collect(unsettled=True)
        self.assertEqual(raw['status'],'failed');self.assertEqual(len(runtime.enqueued),18);self.assertEqual(calls,[])
        self.assertGreaterEqual(clock.now,10_000_000_000)
        raw,runtime,_,_,_,_=self.collect(incomplete=True)
        self.assertEqual(raw['status'],'completed',raw['errors']);self.assertEqual(len(runtime.enqueued),199)
        self.assertTrue(all(r['status']=='incomplete' for r in raw['request_records']));self.assertFalse(raw['all_requests_completed'])

if __name__=='__main__':unittest.main()
