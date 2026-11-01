"""Independent sustained-cohort audit gates, including adversarial role changes."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('capacity_sustained_audit',ROOT/'tools/spillover_capacity_20261022/audit_sustained.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)


def fixture():
    aliases=['text#0','text#1','generalist_2#0']
    job=dict(active_aliases=aliases,planned_requests={},support_requests={})
    measured=[];last={};counts={}
    for n, alias in enumerate(aliases):
        primary=[];support=[]
        for i in range(96):
            template=dict(request_id=f'primary:{alias}:{i}',model=alias,task='text',pool_id=f'p{i%32}',
                          role='primary',sequence=i,arrival_index=i,repetition=i//32)
            primary.append(template)
            measured.append(dict(template,phase='measurement',status='completed',scheduled_time_ns=i*10+1,
                                 completion_time_ns=i*10+5+n))
        for i in range(1024):
            support.append(dict(request_id=f'support:{alias}:{i}',model=alias,task='text',pool_id=f'p{i%32}',
                                role='support',support_sequence=i,sequence=96+i,arrival_index=96+i,repetition=i//32))
        # Fast aliases each release a single support before the final alias completes.
        count=1 if n<2 else 0
        if count:
            measured.append(dict(support[0],phase='measurement',status='completed',scheduled_time_ns=956,
                                 completion_time_ns=960))
        counts[alias]=count;last[alias]=955+n
        job['planned_requests'][alias]=primary;job['support_requests'][alias]=support
    window=dict(start_time_ns=0,end_time_ns=960,duration_seconds=960/1e9,primary_completion_time_ns=957,
                support_admission_stop_time_ns=958,alias_completion_time_ns=last,
                primary_completed_by_alias={a:96 for a in aliases},support_released_by_alias=counts,
                released_by_alias={a:96+counts[a] for a in aliases})
    return job,measured,window


class SustainedCohortAudit(unittest.TestCase):
    def test_accepts_complete_primary_with_drained_support(self):
        primary,support,planned=audit.audit_measurement_cohorts(*fixture())
        self.assertEqual((len(primary),len(support),len(planned)),(288,2,290))

    def test_support_cannot_replace_missing_primary(self):
        job,rows,window=fixture();rows.pop(0)
        with self.assertRaises(AssertionError):audit.audit_measurement_cohorts(job,rows,window)

    def test_support_must_be_exact_independent_prefix(self):
        job,rows,window=fixture()
        row=next(r for r in rows if r['role']=='support');row['support_sequence']=1
        with self.assertRaises(AssertionError):audit.audit_measurement_cohorts(job,rows,window)

    def test_no_support_after_admission_stop_or_before_own_primary(self):
        for admission in [954,958]:
            job,rows,window=fixture()
            next(r for r in rows if r['role']=='support')['scheduled_time_ns']=admission
            with self.subTest(admission=admission),self.assertRaises(AssertionError):
                audit.audit_measurement_cohorts(job,rows,window)

    def test_primary_window_and_support_drain_are_distinct(self):
        job,rows,window=fixture();window['primary_completion_time_ns']=960
        with self.assertRaises(AssertionError):audit.audit_measurement_cohorts(job,rows,window)
        job,rows,window=fixture();window['end_time_ns']=959
        with self.assertRaises(AssertionError):audit.audit_measurement_cohorts(job,rows,window)

    def test_duplicate_or_unreleased_support_rejected(self):
        job,rows,window=fixture();rows.append(copy.deepcopy(rows[-1]))
        with self.assertRaises(AssertionError):audit.audit_measurement_cohorts(job,rows,window)

    def test_total_wall_and_per_request_timeouts_remain_bounded(self):
        job,rows,window=fixture();window['end_time_ns']=600_000_000_001
        with self.assertRaises(AssertionError):audit.audit_measurement_cohorts(job,rows,window)
        job,rows,window=fixture();rows[0]['scheduled_time_ns']=-300_000_000_000
        with self.assertRaises(AssertionError):audit.audit_measurement_cohorts(job,rows,window)

    def test_nonterminal_queue_rejects_before_raw_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);packet=root/'packet.json';queue=root/'queue.json'
            packet.write_text('{}');queue.write_text(json.dumps(dict(status='RUNNING')))
            with mock.patch.object(audit,'audit_job') as reader,self.assertRaisesRegex(ValueError,'All four'):
                audit.audit(packet,queue,root/'out')
            reader.assert_not_called();self.assertFalse((root/'out').exists())

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with mock.patch.object(Path,'read_text') as reader,self.assertRaises(FileExistsError):
                audit.audit(root/'packet',root/'queue',root)
            reader.assert_not_called()

    def test_power_independent_interpolation_and_coverage(self):
        samples=[dict(time_ns=0,watts=10),dict(time_ns=1_000_000_000,watts=20),dict(time_ns=2_000_000_000,watts=30)]
        self.assertAlmostEqual(audit.integrate(samples,500_000_000,1_500_000_000),20)
        with self.assertRaises(AssertionError):audit.integrate(samples,-1,1_000_000_000)

    def test_failed_native_audit_never_claims_resource_or_energy_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);packet_path=root/'packet.json';queue_path=root/'queue.json'
            jobs=[dict(job_id=str(i),block=block,state=state,seed=seed,active_aliases=['generalist_2#0'])
                  for i,(block,state,seed) in enumerate([(1,'B',103),(1,'C',103),(2,'C',104),(2,'B',104)])]
            packet=dict(protocol='sustained_joint_primary_support_v1',U=None,diagnostic_only=True,jobs=jobs,bindings={})
            packet_path.write_text(json.dumps(packet));registered=[]
            for job in jobs:
                path=root/(job['job_id']+'-attempt1')/'run.json';path.parent.mkdir();path.write_text('{}')
                registered.append(dict(job_id=job['job_id'],status='completed',raw=dict(path=str(path),sha256=audit.sha(path))))
            queue_path.write_text(json.dumps(dict(schema='gsdeploy.sustained_capacity_queue.v1',
                status='ALL_FOUR_SUSTAINED_DIAGNOSTIC_ATTEMPTS_COMPLETE',packet_sha256=audit.sha(packet_path),jobs=registered)))
            with mock.patch.object(audit,'audit_job',side_effect=AssertionError('fixture native failure')):
                result=audit.audit(packet_path,queue_path,root/'out')
            self.assertEqual(result['status'],'AUDIT_FAILED')
            self.assertFalse(result['resource_independently_verified'])
            self.assertFalse(result['energy_independently_verified'])
            self.assertTrue(all(r['traceback'] and r['error_location'] for r in result['jobs']))


if __name__=='__main__':unittest.main()
