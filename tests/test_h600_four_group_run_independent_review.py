"""Independent CPU F8 provenance regression, real workers always mocked."""
from pathlib import Path
import unittest
from unittest.mock import patch

from tests import test_h600_four_group_run_v1 as fixtures
from tools.predictor_validation_20261027 import h600_four_group_run_v1 as m


class FinalRunIndependentReviewTests(unittest.TestCase):
    def setUp(self):
        fixtures.FinalRunTests.setUp(self)
        # Keep this control fixture wholly synthetic. The parent fixture's
        # original disk plan names a real reservation before in-memory edits.
        self.plan_path.unlink();m.save(self.plan_path,self.plan)
        self.sealed['plan']=m.receipt(self.plan_path)
        self.sealed['seal_sha256']=m.digest({k:v for k,v in self.sealed.items() if k!='seal_sha256'})
        self.sealed_path.unlink();m.save(self.sealed_path,self.sealed)

    controlled=fixtures.FinalRunTests.controlled
    collector=fixtures.FinalRunTests.collector
    audited=fixtures.FinalRunTests.audited
    report=fixtures.FinalRunTests.report

    def test_rehashed_seal_cannot_omit_authenticated_upstream_source_closure(self):
        self.controlled()
        upstream=self.root/'training-native-source.json';m.save(upstream,{'immutable':'upstream training evidence'})
        self.prerequisite['source_receipts']=[m.receipt(upstream),m.receipt(m.__file__)]
        sealed=m.read(self.sealed_path);sealed['source_receipts']=[]
        sealed['seal_sha256']=m.digest({k:v for k,v in sealed.items() if k!='seal_sha256'})
        self.sealed_path.unlink();m.save(self.sealed_path,sealed)
        with patch.object(m.final.native_flow,'execute',side_effect=self.collector) as execute,\
                patch.object(m.final,'audit_actual',side_effect=self.audited),\
                patch.object(m.final,'report_actual',side_effect=self.report):
            with self.assertRaises(ValueError):m.run(self.sealed_path,self.root/'execution',diagnostic=True)
            execute.assert_not_called()

    def test_rehashed_gate_status_label_cannot_turn_original_failure_into_pass(self):
        self.controlled()
        gate_path=self.root/'original-gate.json';m.save(gate_path,dict(status='FAIL'))
        sealed=m.read(self.sealed_path);sealed['independent_gate']=m.receipt(gate_path)
        sealed['independent_gate_status']='PASS'
        sealed['seal_sha256']=m.digest({k:v for k,v in sealed.items() if k!='seal_sha256'})
        self.sealed_path.unlink();m.save(self.sealed_path,sealed)
        with self.assertRaises(ValueError):m.check_sealed(self.sealed_path)

    def test_preserved_upstream_source_change_stops_before_next_configuration(self):
        self.controlled()
        upstream=self.root/'training-native-source.json';m.save(upstream,{'immutable':'upstream training evidence'})
        upstream_ref=m.receipt(upstream)
        self.prerequisite['source_receipts']=[upstream_ref]
        # The immutable source is available in the full freshly authenticated
        # prerequisite even if an old/partial sealed closure omitted it.
        sealed=m.read(self.sealed_path)
        sealed['source_receipts'].append(upstream_ref)
        sealed['seal_sha256']=m.digest({k:v for k,v in sealed.items() if k!='seal_sha256'})
        self.sealed_path.unlink();m.save(self.sealed_path,sealed)
        def collect(plan,cid,output):
            self.collector(plan,cid,output)
            upstream.write_text('{"changed":true}')
        with patch.object(m.final.native_flow,'execute',side_effect=collect) as execute,\
                patch.object(m.final,'audit_actual',side_effect=self.audited):
            with self.assertRaises(ValueError):m.run(self.sealed_path,self.root/'execution',diagnostic=True)
        self.assertEqual(execute.call_count,1)

    def independent_fixture(self,*,actual_known=1200):
        base=self.root/'finite';base.mkdir()
        final_folder=base/'four-group-diagnostic-v1';final_folder.mkdir()
        plan_path=final_folder/'plan.json';m.save(plan_path,dict(cpu_fixture=True))
        report_folder=base/'admission-validation-v1/report';report_folder.mkdir(parents=True)
        cids=list(m.capacity.ROUTES)
        report=dict(candidate_coverage={'requested':4,'observed':4},candidates=[
            dict(candidate_id=cid,known=0,requested=1200) for cid in cids])
        report_path=report_folder/'report.json';m.save(report_path,report)
        gate_path=report_folder/'gate.json';m.save(gate_path,dict(status='INCOMPLETE',unknown_checks=['delta_sign_accuracy']))
        protocol_path=base/'training-protocol.json';m.save(protocol_path,dict(campaign=m.receipt(self.model_path)))
        training=dict(protocol=m.receipt(protocol_path),manifests={
            k:m.receipt(self.model_path) for k in ('capacity_protocol','resource_protocol','quality_protocol')})
        seal_path=base/'seal-audit.json';m.save(seal_path,dict(protocol=m.receipt(protocol_path),
            **{k:m.receipt(self.model_path) for k in ('capacity_manifest','resource_manifest','quality_manifest')}))
        evidence=[dict(candidate_id=cid,requested=1200,known=actual_known,
            quality_audit=dict(count=1200,independently_recomputed=True),
            actual_metrics=dict(six_term_comparison_complete=True),
            semantic_validation_trace=dict(split='validation',not_for_training=True,requests=[{}]*1200),
            resource_validation_evidence=dict(split='validation',not_for_training=True),not_for_training=True)
            for cid in cids]
        m.save(report_folder/'validation-audit.json',dict(seal_audit=m.receipt(seal_path),gate=m.receipt(gate_path),evidence=evidence))
        plan=dict(role_reservation=m.receipt(self.role_path))
        patches=[patch.object(m,'BASE',base),patch.object(m.final,'audit_plan',return_value=plan),
            patch.object(m,'_training',return_value=training),
            patch.object(m.validation,'verify_report',return_value={'verified':True}),
            patch.object(m.validation,'audit_protocol',return_value=dict(campaign=m.receipt(self.model_path),
                role_reservation=plan['role_reservation'])),
            patch.object(m.continuation,'json_source_closure',return_value=[])]
        for p in patches:p.start();self.addCleanup(p.stop)
        completed=base/'unused-completed.json';m.save(completed,dict(cpu_fixture=True))
        return plan_path,report_path,completed

    def test_prediction_UNKNOWN_does_not_hide_complete_actual_V4_from_diagnostic(self):
        args=self.independent_fixture()
        result=m._prerequisites(*args)
        self.assertEqual(result['independent_gate_status'],'INCOMPLETE')

    def test_actual_V4_incomplete_still_blocks_even_for_diagnostic(self):
        args=self.independent_fixture(actual_known=1199)
        with self.assertRaises(ValueError):m._prerequisites(*args)


if __name__=='__main__':unittest.main()
