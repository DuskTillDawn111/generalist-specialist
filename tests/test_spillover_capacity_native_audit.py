"""CPU gates and independent trapezoidal integration for terminal audits."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('capacity_native_audit',ROOT/'tools/spillover_capacity_20261022/audit_native.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)


class NativeAuditGates(unittest.TestCase):
    def test_running_queue_rejects_before_any_terminal_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);packet=root/'packet.json';queue=root/'queue.json'
            packet.write_text('{}');queue.write_text(json.dumps(dict(status='RUNNING',jobs=[{'raw':{'path':'/must/not/read/run.json'}}])))
            with mock.patch.object(audit,'audit_job') as reader, self.assertRaisesRegex(ValueError,'All six'):
                audit.audit(packet,queue,root/'out')
            reader.assert_not_called();self.assertFalse((root/'out').exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with mock.patch.object(Path,'read_text') as reader,self.assertRaises(FileExistsError):
                audit.audit(root/'packet',root/'queue',root)
            reader.assert_not_called()

    def test_independent_integration_interpolates_exact_boundaries(self):
        samples=[{'time_ns':0,'watts':10},{'time_ns':1_000_000_000,'watts':20},{'time_ns':2_000_000_000,'watts':30}]
        self.assertAlmostEqual(audit.integrate(samples,500_000_000,1_500_000_000),20)

    def test_power_requires_full_coverage_and_bounded_gaps(self):
        for samples,begin,end in [
            ([{'time_ns':1,'watts':10},{'time_ns':1_000_000_000,'watts':10}],0,1_000_000_000),
            ([{'time_ns':0,'watts':10},{'time_ns':3_000_000_000,'watts':10}],0,3_000_000_000),
            ([{'time_ns':0,'watts':-1},{'time_ns':1_000_000_000,'watts':10}],0,1_000_000_000)]:
            with self.subTest(samples=samples),self.assertRaises(AssertionError):audit.integrate(samples,begin,end)

    def test_native_protocol_uses_merged_canonical_hash_not_historical_path(self):
        # Real frozen worker metadata reproduces the failed-v1 assertion without
        # reading a GPU run. Historical task quality uses another protocol scope.
        base=ROOT/'artifacts/static-study-v6-20261021/inputs'
        workers=json.loads((base/'worker-registry.json').read_text())
        catalog=json.loads((base/'catalog.json').read_text())
        for family in ['text','generalist_2']:
            spec=json.loads(Path(workers[family]['spec']).read_text())
            expected=audit.runtime_protocol_digest(spec)
            historical=catalog['quality']['paths'][family]['text']['protocol_sha256']
            self.assertNotEqual(expected,historical)
            response={'protocol_sha256':expected}
            audit.check_response_protocol(response,spec,'fixture')
            with self.assertRaisesRegex(AssertionError,'runtime_worker_protocol_canonical_sha256'):
                audit.check_response_protocol({'protocol_sha256':historical},spec,'fixture')

    def test_failed_job_preserves_traceback_and_never_claims_resources_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);queue_path=root/'queue.json';packet_path=root/'packet.json'
            jobs=[{'job_id':str(i),'active_aliases':['text#0']} for i in range(6)]
            packet={'protocol':'balanced_closed_loop_capacity_v1','U':None,'diagnostic_only':True,'jobs':jobs}
            packet_path.write_text(json.dumps(packet));registered=[]
            for job in jobs:
                path=root/(job['job_id']+'-attempt1')/'run.json';path.parent.mkdir();path.write_text('{}')
                registered.append(dict(job_id=job['job_id'],status='completed',raw=dict(path=str(path),sha256=audit.sha(path))))
            queue_path.write_text(json.dumps(dict(status='ALL_SIX_DIAGNOSTIC_ATTEMPTS_COMPLETE',packet_sha256=audit.sha(packet_path),jobs=registered)))
            with mock.patch.object(audit,'audit_job',side_effect=AssertionError()):
                result=audit.audit(packet_path,queue_path,root/'out')
            self.assertEqual(result['status'],'AUDIT_FAILED')
            self.assertFalse(result['resources_and_energy_verified_independently'])
            self.assertFalse(result['resource_independently_verified'])
            self.assertFalse(result['energy_independently_verified'])
            self.assertTrue(all(r['traceback'] and r['error_location'] and r['error'] for r in result['jobs']))


if __name__=='__main__':unittest.main()
