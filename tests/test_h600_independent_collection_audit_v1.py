"""Collection audits keep domain, chronology and evidence separate from fit gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import h600_independent_collection_audit_v1 as m


class CollectionAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.runroot=self.root/'runs'
        self.suite=dict(created_ns=20,jobs=[dict(candidate_id=cid,output=str(self.runroot/cid))
            for cid in m.selected.CANDIDATES])

    def test_inventory_keeps_missing_until_final_and_rejects_extra_or_nested_attempt(self):
        self.runroot.mkdir();first=self.runroot/m.selected.CANDIDATES[0];first.mkdir()
        self.assertEqual(len(m.inventory(self.suite)['missing']),3)
        with self.assertRaisesRegex(ValueError,'Missing declared'):m.inventory(self.suite,complete=True)
        extra=self.runroot/'retry';extra.mkdir()
        with self.assertRaisesRegex(ValueError,'Extra candidate or attempt'):m.inventory(self.suite)
        extra.rmdir();nested=first/'retry';nested.mkdir();(nested/'invocation.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Nested duplicate attempt'):m.inventory(self.suite)

    def test_startup_requires_complete_seal_suite_and_prospective_audit(self):
        seal=dict(information_cutoff_ns=10,created_ns=15);prospective=dict(audited_ns=25)
        result=m.chronology({'startup_begin_ns':26},seal,self.suite,prospective)
        self.assertTrue(result['seal_and_audit_before_startup'])
        for t in (15,20,25,None):
            with self.subTest(t=t),self.assertRaisesRegex(ValueError,'precede native startup'):
                m.chronology({'startup_begin_ns':t},seal,self.suite,prospective)

    def test_recursive_source_receipts_reject_drift_without_any_fit_or_gpu(self):
        path=self.root/'source';path.write_text('frozen');ref=m.receipt(path)
        value={'nested':[{'evidence':ref},ref]}
        self.assertEqual(m.source_refs(value),[ref]);path.write_text('drift')
        with self.assertRaises(ValueError):m.source_refs(value)

    def fixture(self):
        cid=m.selected.CANDIDATES[0];folder=self.runroot/cid;folder.mkdir(parents=True)
        path=folder/'raw.json';path.write_text('{}');scored=folder/'scored-raw.json';scored.write_text('{}')
        planpath=self.root/'plan.json';planpath.write_text('{}');ref=m.receipt(planpath)
        self.suite['sources']={'plan':ref};suitepath=self.root/'suite.json';suitepath.write_text('{}')
        catalog=self.root/'catalog.json';catalog.write_text('{}')
        plan=dict(catalog=m.receipt(catalog),aliases={'worker':{}},
            candidates=[dict(id=cid,resource_configuration={'mode':'shared'})])
        raw=dict(candidate_id=cid,native_raw=m.receipt(path),plan=ref,startup_begin_ns=26,
            quality_evidence={})
        mapped=dict(traces=[dict(split='validation',not_for_training=True,
            requests=[{'split':'validation'} for _ in range(1200)])],resource_evidence={'split':'validation'},mapping_report={})
        checks=[patch.object(m,'inputs',return_value=(self.suite,plan,
            dict(created_ns=15,information_cutoff_ns=10),dict(audited_ns=25),[])),
            patch.object(m,'scored_heldout',return_value=(raw,m.receipt(scored))),
            patch.object(m.flow_profiles,'map_flow_collection',return_value=mapped),
            patch.object(m,'verify_quality_annotation',return_value={'count':1200}),
            patch.object(m.mechanism_audit,'audit',return_value={'status':'PASS_OBSERVED_CONTRACT'}),
            patch.object(m.flow_eligibility,'native_stability',return_value={'original_stability_passed':True}),
            patch.object(m.flow_metrics,'score_flow',return_value={'utility_complete':True,'terms':{'Q':.1}})]
        for p in checks:p.start();self.addCleanup(p.stop)
        return suitepath,path,mapped,raw

    def test_single_audit_remaps_validation_and_preserves_scoring_stability_and_exclusive_output(self):
        suite,path,mapped,raw=self.fixture();before=copy.deepcopy(raw);out=self.root/'audit.json'
        result=m.audit_one(suite,path,out)
        m.flow_profiles.map_flow_collection.assert_called_once_with(path,allow_validation_diagnostics=True)
        self.assertEqual(result['status'],'PASS');self.assertFalse(result['prediction_gate_assessed'])
        self.assertFalse(result['fitting_performed']);self.assertFalse(result['GPU_started'])
        self.assertEqual(result['actual_six_term_score']['terms'],{'Q':.1});self.assertEqual(raw,before)
        with self.assertRaises(FileExistsError):m.audit_one(suite,path,out)

    def test_training_mapping_or_wrong_attempt_never_saved_and_original_stability_failure_retained(self):
        suite,path,mapped,raw=self.fixture()
        mapped['traces'][0]['split']='training'
        with self.assertRaisesRegex(ValueError,'validation-only'):m.audit_one(suite,path,self.root/'denied.json')
        self.assertFalse((self.root/'denied.json').exists());mapped['traces'][0]['split']='validation'
        with patch.object(m.flow_eligibility,'native_stability',return_value={'original_stability_passed':False}):
            result=m.audit_one(suite,path,self.root/'stability-failed.json')
        self.assertEqual(result['status'],'FAIL_ORIGINAL_STABILITY');self.assertFalse(result['final_gain_eligible'])
        other=self.root/'foreign.json';other.write_text('{}')
        with self.assertRaisesRegex(ValueError,'exact declared'):m.audit_one(suite,other,self.root/'foreign-audit.json')

    def test_complete_requires_four_distinct_audited_attempts_and_keeps_prediction_gate_unassessed(self):
        suite,path,mapped,raw=self.fixture();first=self.root/'audit-0.json'
        m.audit_one(suite,path,first);template=json.loads(first.read_text());paths=[]
        for i,job in enumerate(self.suite['jobs']):
            folder=Path(job['output']);folder.mkdir(exist_ok=True)
            native=folder/'raw.json';native.write_text(json.dumps({'startup_begin_ns':26}))
            scored=folder/'scored-raw.json';scored.write_text('{}')
            report=copy.deepcopy(template);report.update(candidate_id=job['candidate_id'],
                raw=m.receipt(native),scored_raw=m.receipt(scored),source_receipts=[])
            audit=self.root/('complete-audit-'+str(i)+'.json');audit.write_text(json.dumps(report));paths.append(audit)
        with self.assertRaisesRegex(ValueError,'All four'):m.complete(suite,paths[:3],self.root/'missing.json')
        with self.assertRaisesRegex(ValueError,'Unique declared'):m.complete(suite,[*paths[:3],paths[0]],self.root/'duplicate.json')
        result=m.complete(suite,paths,self.root/'complete.json')
        self.assertEqual(result['audited_requests'],4800);self.assertEqual(result['status'],'PASS')
        self.assertFalse(result['prediction_gate_assessed']);self.assertFalse(result['final_gain_eligible'])


if __name__=='__main__':unittest.main()
