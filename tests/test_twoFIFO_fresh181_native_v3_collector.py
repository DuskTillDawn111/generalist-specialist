"""CPU v3 typed registry/control flow; no native fit or GPU work."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import twoFIFO_fresh181_native_v3 as m
from tools.predictor_validation_20261027 import twoFIFO_matching_prospective_bridge_v3 as typed
from tools.predictor_validation_20261027 import twoFIFO_matching_prospective_bridge_v1 as core
from tools.predictor_validation_20261027 import twoFIFO_fresh181_prefix_v1 as prefix
from tools.predictor_validation_20261027.stage_response_features_v1 import checked,receipt
from tests.test_twoFIFO_fresh181_execute_v1 import PUBLIC,MATCHING,Runtime,Clock
from tests.test_twoFIFO_owned_collect_v2 import FakeMonitor


class NativeV3CollectorTests(unittest.TestCase):
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
        cls.budget_ref=prefix.save(cls.root/'budget.json',dict(CPU_SYNTHETIC_ONLY=True))
        cls.budget=dict(public_preparation=receipt(PUBLIC),fits_predeclared_budget=False,sources=[])
        with patch.object(m.response,'audit_fit',return_value=dict(report=cls.fit)),patch.object(m.timing,'audit',return_value=cls.budget):
            cls.manifest_ref=m.prepare(PUBLIC,MATCHING,BridgeTests.fit_ref['path'],'a497fe19341d442d9189caa4dc4d4f19',cls.root/'prepared',budget_report_path=cls.budget_ref['path'])
        cls.manifest=checked(cls.manifest_ref);cls.item=cls.manifest['attempts'][0];cls.plan=checked(cls.item['plan'])
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup();cls.fixture.tearDownClass()
    def authentication(self):
        def verified(a,b,callback):return callback(self.fixture.bundle,self.intake)
        return patch.object(m.response,'audit_fit',return_value=dict(report=self.fit)),\
            patch.object(m.response,'with_verified_bundle',side_effect=verified),patch.object(m.timing,'audit',return_value=self.budget)
    def collect_cpu(self,*,late=False,tamper=False,audit_restore=False,same_session=False):
        clock=Clock();holder={};pa,pb,pc=self.authentication()
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
        with pa,pb,pc,patch.object(m.supervisor,'validate_lease'),patch.object(typed,'produce',side_effect=produce),\
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
    def test_audit_return_to_seal_report_change_restore_stops_before_any_target(self):
        original=self.item;self.item=self.manifest['attempts'][3]
        try:raw,runtime,_=self.collect_cpu(audit_restore=True)
        finally:self.item=original
        self.assertEqual(raw['status'],'failed');self.assertEqual(len(runtime.enqueued),18)
        self.assertFalse(raw['predictions_durable_before_cohort']);self.assertTrue(runtime.closed)

if __name__=='__main__':unittest.main()
