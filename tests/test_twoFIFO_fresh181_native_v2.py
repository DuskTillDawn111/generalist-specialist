"""CPU v2 typed registry/control flow; no native fit or GPU work."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import twoFIFO_fresh181_native_v2 as m
from tools.predictor_validation_20261027 import twoFIFO_matching_prospective_bridge_v2 as typed
from tools.predictor_validation_20261027 import twoFIFO_matching_prospective_bridge_v1 as core
from tools.predictor_validation_20261027 import twoFIFO_fresh181_prefix_v1 as prefix
from tools.predictor_validation_20261027.stage_response_features_v1 import checked,receipt
from tests.test_twoFIFO_fresh181_execute_v1 import PUBLIC,MATCHING,Runtime,Clock
from tests.test_twoFIFO_owned_collect_v2 import FakeMonitor


class NativeV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_twoFIFO_matching_prospective_bridge_v1 import BridgeTests
        BridgeTests.setUpClass();cls.fixture=BridgeTests
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.continuation=prefix.save(cls.root/'continuation.json',dict(CPU_SYNTHETIC_ONLY=True))
        cls.intake=dict(CPU_SYNTHETIC_ONLY=True,sources=[])
        cls.fit=dict(manifest=BridgeTests.fit_manifest,continuation=cls.continuation,
            fit=dict(BridgeTests.fitted,response_law_compilation_eligible=False),intake_evidence=cls.intake,
            sources=[BridgeTests.fit_manifest,cls.continuation])
        with patch.object(m.response,'audit_fit',return_value=dict(report=cls.fit)):
            cls.manifest_ref=m.prepare(PUBLIC,MATCHING,BridgeTests.fit_ref['path'],'a497fe19341d442d9189caa4dc4d4f19',cls.root/'prepared')
        cls.manifest=checked(cls.manifest_ref);cls.item=cls.manifest['attempts'][0];cls.plan=checked(cls.item['plan'])
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup();cls.fixture.tearDownClass()
    def authentication(self):
        def verified(a,b,callback):return callback(self.fixture.bundle,self.intake)
        return patch.object(m.response,'audit_fit',return_value=dict(report=self.fit)),\
            patch.object(m.response,'with_verified_bundle',side_effect=verified)

    def test_real_v2_preparation_has_exact_fixed_auditor_schema_and_budget(self):
        with patch.object(m.response,'audit_fit',return_value=dict(report=self.fit)):
            document=m.audit(self.manifest_ref['path']);plan=m.audit_plan(self.item['plan']['path'])
        self.assertEqual(document['candidate_count'],4);self.assertEqual(document['measured_offered'],724)
        self.assertEqual(plan['schema'],m.SCHEMA+'.attempt');self.assertEqual(plan['forecast_barrier']['lead_seconds'],300)
        self.assertEqual(plan['target_plan_auditor']['module'],typed.TARGET_REGISTRY)
        self.assertEqual(plan['forecast_barrier']['typed_producer_source'],receipt(typed.__file__))
        self.assertEqual(plan['forecast_barrier']['budget']['real_fit_budget_qualification'],'UNKNOWN_NOT_BENCHMARKED')
        self.assertFalse(document['native_launch_enabled'])

    def test_old_fresh_or_caller_auditor_cannot_be_substituted(self):
        with patch.object(m.response,'audit_fit',return_value=dict(report=self.fit)):
            with self.assertRaises(ValueError):m.audit_plan(self.fixture.plan_ref['path'])
        with self.assertRaisesRegex(ValueError,'exact native-v2 auditor'):typed.TargetSession(None,{},None,None)
        for op in (m.collect,m.execute):
            with self.assertRaisesRegex(UnknownEvidence,m.EXECUTION_HOLD):op('unused','unused',lease_fd=-1)
        with self.assertRaisesRegex(ValueError,'injected transport'):
            m.collect_cpu('unused','unused',lease_fd=-1,sessions=(None,None))

    def collect_cpu(self,*,late=False,tamper=False,audit_restore=False,same_session=False):
        clock=Clock();holder={};pa,pb=self.authentication()
        def factory(config,registry):holder['runtime']=Runtime(clock,registry);return holder['runtime']
        def ownership(values,*args,**kwargs):
            pid=next(iter(values));return dict(owned_aliases=[m.OWNERS[pid-1]],unresolved_nvml_pids=[],
                foreign_pids=[],ownership_evidence=[dict(CPU_TEST_ONLY=True,live=True)])
        original=typed.produce
        original_audit=typed.audit_report
        def produce(session,gate_path,output,**kwargs):
            result=original(session,gate_path,output,**kwargs)
            if late:clock.wait(checked(receipt(gate_path))['forecast_origin_ns'])
            if tamper:
                document=checked(result);document['replay']['forecast_sha256']='0'*64
                Path(result['path']).write_text(core.json.dumps(document))
            return result
        def audit_report(*args,**kwargs):
            result=original_audit(*args,**kwargs)
            if audit_restore:
                path=Path(args[1]);original=path.read_bytes();path.write_bytes(original+b' ');path.write_bytes(original)
            return result
        with pa,pb,patch.object(m.supervisor,'validate_lease'),patch.object(typed,'produce',side_effect=produce),\
                patch.object(typed,'audit_report',side_effect=audit_report):
            with typed.open_target_session(self.fixture.fit_ref['path'],self.manifest_ref['path'],self.item['plan']['path']) as producer:
                with typed.open_target_session(self.fixture.fit_ref['path'],self.manifest_ref['path'],self.item['plan']['path']) as auditor:
                    raw=m.collect_cpu(self.item['plan']['path'],self.item['collection'],lease_fd=-1,sessions=(producer,producer if same_session else auditor),
                        runtime_factory=factory,monitor_factory=FakeMonitor,clock_ns=clock.clock,wait_until_ns=clock.wait,
                        ownership_check=ownership)
        return raw,holder['runtime'],clock

    def test_complete_real_collector_typed_producer_full_auditor_and_seal_before181(self):
        raw,runtime,clock=self.collect_cpu()
        self.assertEqual(raw['status'],'completed',raw['errors']);self.assertEqual(len(runtime.enqueued),199)
        self.assertTrue(raw['predictions_durable_before_cohort']);self.assertEqual(raw['backend'],'CPU_TEST_ONLY')
        self.assertTrue(runtime.closed);self.assertEqual(runtime.drain_counts[-1],199)
        report=checked(raw['typed_forecast_report']);self.assertEqual(report['schema'],typed.SCHEMA)
        self.assertFalse(report['source_supported_native_hypothesis']);self.assertFalse(report['native_execution_enabled'])
        self.assertEqual(len(core.decode(report['replay']['forecast'])['ensemble']['draws']),12)
        seal=checked(raw['prediction_seal']);self.assertEqual(seal['schema'],typed.SCHEMA+'.prediction_seal')
        self.assertLess(seal['seal_begin_ns'],raw['planned_cohort_epoch_ns'])
        self.assertEqual(raw['measurement_windows'][0]['fixed_drain_deadline_ns'],raw['planned_cohort_epoch_ns']+210_000_000_000)

    def test_late_typed_producer_stops_before_target_enqueue(self):
        # Use the second independent candidate so no output/claim is reused.
        original=self.item;self.item=self.manifest['attempts'][1]
        try:raw,runtime,_=self.collect_cpu(late=True)
        finally:self.item=original
        self.assertEqual(raw['status'],'failed');self.assertEqual(len(runtime.enqueued),18)
        self.assertFalse(raw['predictions_durable_before_cohort']);self.assertEqual(len(raw['request_records']),181)
        self.assertTrue(all(r['status']=='not_started' for r in raw['request_records']))

    def test_typed_report_tampering_cannot_release_target_even_with_qualified_flags(self):
        original=self.item;self.item=self.manifest['attempts'][2]
        try:raw,runtime,_=self.collect_cpu(tamper=True)
        finally:self.item=original
        self.assertEqual(raw['status'],'failed');self.assertEqual(len(runtime.enqueued),18)
        self.assertTrue(runtime.closed);self.assertFalse(raw['predictions_durable_before_cohort'])
        self.assertTrue(raw['errors'])

    def test_audit_return_to_seal_report_change_restore_stops_before_any_target(self):
        original=self.item;self.item=self.manifest['attempts'][3]
        try:raw,runtime,_=self.collect_cpu(audit_restore=True)
        finally:self.item=original
        self.assertEqual(raw['status'],'failed');self.assertEqual(len(runtime.enqueued),18)
        self.assertFalse(raw['predictions_durable_before_cohort']);self.assertTrue(runtime.closed)

    def test_same_producer_auditor_session_rejected_before_forecast_and_targets(self):
        # Each test starts a separate prepared namespace to avoid capture retry.
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(m.response,'audit_fit',return_value=dict(report=self.fit)):
                manifest_ref=m.prepare(PUBLIC,MATCHING,self.fixture.fit_ref['path'],'3b13609aa52946b889ed981840e7a58d',Path(temporary)/'prepared')
            original_item=self.item;original_manifest=self.manifest_ref
            self.manifest_ref=manifest_ref;self.item=checked(manifest_ref)['attempts'][0]
            try:raw,runtime,_=self.collect_cpu(same_session=True)
            finally:self.item=original_item;self.manifest_ref=original_manifest
        self.assertEqual(raw['status'],'failed');self.assertEqual(len(runtime.enqueued),18)
        self.assertTrue(any('separately authenticated' in error for error in raw['errors']))


if __name__=='__main__':unittest.main()
