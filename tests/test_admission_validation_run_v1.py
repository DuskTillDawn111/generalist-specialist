"""CPU-only launcher control tests; fake collection is never GPU evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import admission_validation_run_v1 as m


class AdmissionValidationRunTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.folder=self.root/'seal';self.folder.mkdir()
        self.code=self.root/'frozen.py';self.code.write_text('frozen\n')
        self.leaf=self.root/'leaf.json';m.save(self.leaf,dict(source=m.receipt(self.code)))
        self.plan=self.root/'plan.json';m.save(self.plan,dict(candidates=[dict(id=c) for c in m.ORDER],leaf=m.receipt(self.leaf)))
        self.protocol=dict(target_plan=m.receipt(self.plan))
        self.sealed=dict(created_ns=20,candidates=[dict(candidate_id=c,unknown_reason='missing support',
            requests=[],metrics=None) for c in m.ORDER])
        m.save(self.folder/'predictions.json',self.sealed)
        m.save(self.folder/'seal-audit.json',dict(protocol=m.receipt(self.plan)))
        self.packet=dict(protocol=self.protocol,sealed=self.sealed,plan=m.read(self.plan),
            source_closure=m.closure([m.receipt(self.plan),m.receipt(self.folder/'seal-audit.json'),
                m.receipt(self.folder/'predictions.json'),m.receipt(m.__file__)]),
            prediction_statuses=[dict(candidate_id=c,requested=1200,prediction_status='UNKNOWN',unknown_reason='missing support') for c in m.ORDER])

    def execute(self,plan,cid,output):
        output.mkdir();m.save(output/'raw.json',dict(candidate_id=cid));m.save(output/'scored-raw.json',dict(candidate_id=cid))

    def actual(self,path,protocol,sealed):
        return dict(native_raw=m.receipt(path),scored_raw=m.receipt(path.with_name('scored-raw.json')),
            known=1200,requested=1200,original_stability=dict(original_stability_passed=False))

    def test_recursive_source_and_original_snapshot_mutation_is_rejected(self):
        refs=m.closure([m.receipt(self.plan)])
        self.assertIn(m.receipt(self.code),refs)
        self.code.write_text('changed\n')
        with self.assertRaises(ValueError):m.recheck(refs)
        original=self.root/'original.py';original.write_text('x\n')
        frozen=self.root/'snapshot.py';frozen.write_text('x\n')
        snapshot=self.root/'context.json';m.save(snapshot,dict(execution_sources=[dict(original_path=str(original),frozen=m.receipt(frozen))]))
        refs=m.closure([m.receipt(snapshot)]);self.assertIn(m.receipt(original),refs)
        original.write_text('y\n')
        with self.assertRaises(ValueError):m.recheck(refs)
        with self.assertRaisesRegex(ValueError,'original execution'):m.closure([m.receipt(snapshot)])

    def test_audit_has_one_full_verification_and_closes_all_nested_sources(self):
        with (patch.object(m.validation,'check_sealed',return_value=dict(protocol=m.receipt(self.plan))) as verify,
              patch.object(m.validation,'audit_protocol',return_value=self.protocol)):
            result=m.audit(self.folder)
        self.assertEqual(verify.call_count,1)
        self.assertIn(m.receipt(self.code),result['source_closure'])
        self.assertTrue(all(s['prediction_status']=='UNKNOWN' for s in result['prediction_statuses']))

    def test_unknown_needs_diagnostic_then_runs_all_four_without_skip_or_pass(self):
        output=self.root/'run'
        with (patch.object(m,'audit',return_value=self.packet) as audit,
              patch.object(m.validation.training.native_flow,'execute',side_effect=self.execute) as execute,
              patch.object(m.validation,'actual_evidence',side_effect=self.actual)):
            with self.assertRaisesRegex(ValueError,'explicit diagnostic'):m.run(self.folder,output)
            self.assertFalse(output.exists());self.assertEqual(execute.call_count,0)
            state=m.run(self.folder,output,diagnostic=True)
        self.assertEqual(audit.call_count,2)  # one proof per invocation, never per job
        self.assertEqual([call.args[1] for call in execute.call_args_list],list(m.ORDER))
        self.assertEqual(state['requested_requests'],4800);self.assertEqual(len(state['completed']),4)
        self.assertEqual(state['prediction_gate_status'],'UNKNOWN');self.assertEqual(state['remaining'],[])
        self.assertEqual(state['status'],'COLLECTED_DIAGNOSTIC');self.assertFalse(state['final_gain_eligible'])

    def test_collection_failure_keeps_fixed_denominator_stops_and_prohibits_retry(self):
        calls=[]
        def execute(plan,cid,output):
            calls.append(cid)
            if len(calls)==2:raise RuntimeError('native bounded drain failure')
            self.execute(plan,cid,output)
        output=self.root/'failed'
        with (patch.object(m,'audit',return_value=self.packet),
              patch.object(m.validation.training.native_flow,'execute',side_effect=execute),
              patch.object(m.validation,'actual_evidence',side_effect=self.actual)):
            with self.assertRaisesRegex(RuntimeError,'bounded drain'):m.run(self.folder,output,diagnostic=True)
            with self.assertRaises(FileExistsError):m.run(self.folder,output,diagnostic=True)
            with self.assertRaisesRegex(FileExistsError,'another output'):m.run(self.folder,self.root/'another-output',diagnostic=True)
            another_seal=self.root/'another-seal';another_seal.mkdir()
            with self.assertRaisesRegex(FileExistsError,'another output or seal'):m.run(another_seal,self.root/'another-output-again',diagnostic=True)
        state=m.read(output/'run-state.json')
        self.assertEqual(calls,list(m.ORDER[:2]));self.assertEqual(len(state['completed']),1)
        self.assertEqual(state['remaining'],list(m.ORDER[1:]));self.assertEqual(state['requested_requests'],4800)
        self.assertEqual(state['status'],'INCOMPLETE_NO_RETRY')

    def test_between_job_source_mutation_stops_before_second_launch(self):
        calls=[]
        def execute(plan,cid,output):
            calls.append(cid);self.execute(plan,cid,output);self.code.write_text('changed\n')
        output=self.root/'mutated'
        with (patch.object(m,'audit',return_value=self.packet),
              patch.object(m.validation.training.native_flow,'execute',side_effect=execute),
              patch.object(m.validation,'actual_evidence',side_effect=self.actual)):
            with self.assertRaises(ValueError):m.run(self.folder,output,diagnostic=True)
        self.assertEqual(calls,[m.ORDER[0]])
        state=m.read(output/'run-state.json');self.assertEqual(state['status'],'INCOMPLETE_NO_RETRY')
        self.assertEqual(state['requested_requests'],4800)


if __name__=='__main__':unittest.main()
