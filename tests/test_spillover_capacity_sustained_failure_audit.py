"""Failure evidence cannot turn shutdown completions into a complete cohort."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sustained_failure_audit',ROOT/'tools/spillover_capacity_20261022/audit_sustained_failure.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture():
    alias='generalist_2#0';templates=[];rows=[]
    for i in range(96):
        template=dict(request_id=f'measurement:{alias}:{i}',model=alias,role='primary',sequence=i,pool_id=f'p{i%32}',repetition=i//32)
        templates.append(template)
        if i<77:
            completion=(i+1)*1_000_000_000 if i<76 else 601_000_000_000
            rows.append(dict(template,status='completed',scheduled_time_ns=i*1_000_000_000,
                completion_time_ns=completion,worker_response=dict(end_ns=completion+100)))
        else:rows.append(dict(template,status='incomplete',scheduled_time_ns=None,completion_time_ns=None,release_state='not_released'))
    job=dict(active_aliases=[alias],planned_requests={alias:templates},support_requests={alias:[{}]*1024})
    window=dict(start_time_ns=0,start_unix_ns=100,end_time_ns=600_000_000_000,duration_seconds=600,
        primary_completion_time_ns=None,support_admission_stop_time_ns=None,primary_released_by_alias={alias:77},
        support_released_by_alias={alias:0},released_by_alias={alias:77},primary_completed_by_alias={alias:76},alias_completion_time_ns={})
    return job,rows,window


class FailureEvidence(unittest.TestCase):
    def test_shutdown_completion_is_tail_and_unreleased_stays_unknown(self):
        primary,support,planned,classes,counts=audit.classify_measurement(*fixture())
        self.assertEqual(len(primary),96);self.assertFalse(support)
        self.assertEqual(list(classes.values()).count('WITHIN_DECLARED_600S'),76)
        self.assertEqual(list(classes.values()).count('AFTER_DECLARED_600S'),1)
        self.assertEqual(list(classes.values()).count('UNRELEASED_UNKNOWN'),19)

    def test_cannot_claim_all_final_completions_observed_by_deadline(self):
        job,rows,window=fixture();window['primary_completed_by_alias']['generalist_2#0']=77
        with self.assertRaises(AssertionError):audit.classify_measurement(job,rows,window)

    def test_native_tail_cannot_be_hidden_by_host_clock(self):
        job,rows,window=fixture();rows[76]['completion_time_ns']=599_000_000_000
        _,_,_,classes,_=audit.classify_measurement(job,rows,window)
        self.assertEqual(classes[rows[76]['request_id']],'AFTER_DECLARED_600S')

    def test_unreleased_requires_unknown_response_and_time(self):
        job,rows,window=fixture();rows[-1]['completion_time_ns']=601_000_000_000
        with self.assertRaises(AssertionError):audit.classify_measurement(job,rows,window)

    def test_missing_registered_primary_cannot_be_dropped(self):
        job,rows,window=fixture();rows.pop()
        with self.assertRaises(AssertionError):audit.classify_measurement(job,rows,window)

    def test_immutable_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):audit.audit('/not/read','/not/read',tmp)


if __name__=='__main__':unittest.main()
