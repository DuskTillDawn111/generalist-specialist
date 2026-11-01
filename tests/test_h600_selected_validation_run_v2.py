"""H600 suite binds capacity/predictor sources before any native execution."""
import unittest
from unittest.mock import patch
from tests import test_h600_selected_validation_v2 as fixtures
from tools.predictor_validation_20261027 import h600_selected_validation_run_v2 as run
from tools.predictor_validation_20261027 import h600_selected_validation_v2 as entry
from tools.predictor_validation_20261027.native_flow import receipt


class H600RunTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.H600ValidationTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.start_patches();self.fixture=fixture;self.f=fixture.f;f=self.f
        capacity=f.write('capacity-protocol.json',{})
        protocol=entry.read(fixture.protocol);protocol['capacity_protocol']=receipt(capacity)
        f.write(fixture.protocol.name,protocol)
        sealed=fixture.seal()
        self.capacity_seal=f.write('capacity-seal.json',dict(protocol=receipt(capacity),
            target_plan=receipt(f.plan_path),created_ns=sealed['created_ns']))
        self.patches=[patch.object(run,'audit_plan',return_value=f.plan),
            patch.object(run,'audit_protocol',side_effect=lambda path:entry.read(path)),
            patch.object(run,'verify_seal',side_effect=lambda path:entry.read(path)),
            patch('tools.predictor_validation_20261027.flow_suite_audit.condition',return_value={}),
            patch('tools.predictor_validation_20261027.flow_suite_audit._generation',return_value={})]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def prepare(self):
        f=self.f
        return run.prepare(f.plan_path,f.root/'sealed',self.capacity_seal,f.root/'suite')

    def test_real_entry_and_complete_suite_audit_never_read_runtime_outputs(self):
        with patch.object(run,'execute',side_effect=AssertionError('GPU forbidden')):
            suite=self.prepare();audit=run.audit_inputs(self.f.root/'suite/suite.json')
        self.assertEqual(suite['sources']['plan'],receipt(self.f.plan_path));self.assertEqual(audit['status'],'PASS')
        self.assertFalse(audit['runtime_paths_opened']);self.assertEqual(audit['training_verification']['training_flow_count'],16)

    def test_capacity_other_plan_or_other_protocol_stops_before_execution(self):
        f=self.f
        for key in ('target_plan','protocol'):
            original=entry.read(self.capacity_seal);changed=dict(original);changed[key]=receipt(f.model_path)
            f.write(self.capacity_seal.name,changed)
            with self.subTest(key=key),patch.object(run,'execute') as execute:
                with self.assertRaisesRegex(ValueError,'same original reserved plan'):self.prepare()
                execute.assert_not_called()
            f.write(self.capacity_seal.name,original)
        self.assertFalse((f.root/'suite').exists())

    def test_empty_candidate_predictions_stop_prospective_prepare(self):
        f=self.f;path=f.root/'sealed/predictions.json';sealed=entry.read(path)
        sealed['candidates'][0]['requests']=[];f.write('sealed/predictions.json',sealed)
        audit=entry.read(f.root/'sealed/training-audit.json');audit['predictions']=receipt(path);f.write('sealed/training-audit.json',audit)
        with self.assertRaisesRegex(ValueError,'complete unique offered cohort'):self.prepare()
        self.assertFalse((f.root/'suite').exists())

    def test_bad_generator_or_first_failure_never_retries(self):
        self.prepare();f=self.f;path=f.root/'suite/suite.json';suite=entry.read(path)
        changed=dict(suite);changed['sources']=dict(suite['sources'],source=receipt(f.plan_path));f.write('suite/suite.json',changed)
        with patch.object(run,'execute') as execute:
            with self.assertRaisesRegex(ValueError,'generator source'):run.run(path)
            execute.assert_not_called()
        f.write('suite/suite.json',suite)
        with patch.object(run,'execute',return_value={'status':'failed','qualification_issues':[]}) as execute:
            with self.assertRaisesRegex(RuntimeError,'without retry'):run.run(path)
            self.assertEqual(execute.call_count,1)

    def test_sources_are_rechecked_before_each_candidate_startup(self):
        self.prepare();f=self.f
        def completed_then_source_drift(*args):
            f.write(f.quality_path.name,{'changed_after_first_candidate':True})
            return {'status':'completed','qualification_issues':[]}
        with patch.object(run,'execute',side_effect=completed_then_source_drift) as execute:
            with self.assertRaises(ValueError):run.run(f.root/'suite/suite.json')
            self.assertEqual(execute.call_count,1)


if __name__=='__main__':unittest.main()
