"""Fault-injection of serial collection and terminal summary; no GPU access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.spillover_context_20261022 import run_search_discrimination as runner


class Finished:
    pid=99123
    def __init__(self,code):self.returncode=code
    def poll(self):return self.returncode


class SearchDiscriminationRunnerTests(unittest.TestCase):
    def run_case(self,mode):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);packet=root/'packet.json';out=root/'execution'
            packet.write_text(json.dumps(dict(jobs=[dict(id='a'),dict(id='b')])))
            calls=[]
            def child(command,**kwargs):
                calls.append(command)
                dest=Path(command[command.index('--output')+1]);dest.mkdir(parents=True)
                if len(calls)==1:
                    if mode=='collection_failure':return Finished(1)
                    rows=[dict(job_id='a')]+([] if mode=='incomplete_queue' else [dict(job_id='b')])
                    (dest/'queue-progress.json').write_text(json.dumps(dict(
                        status='ALL_DECLARED_DEVELOPMENT_ATTEMPTS_COLLECTED',jobs=rows)))
                elif len(calls)==2:
                    status='AUDIT_INCOMPLETE_OR_FAILED' if mode=='audit_evidence_failure' else 'VERIFIED_TERMINAL_EVIDENCE'
                    (dest/'report.json').write_text(json.dumps(dict(observations=[dict(job_id=job,
                        audit_status=status,utility=dict(eligible=mode!='ineligible')) for job in ['a','b']])))
                else:
                    (dest/'report.json').write_text('{}')
                return Finished(0)
            with patch('sys.argv',['run','--packet',str(packet),'--output',str(out),'--execute']),patch.object(
                    runner.subprocess,'Popen',side_effect=child):
                result=runner.main()
            progress=json.loads((out/'progress.json').read_text())
            return result,progress,len(calls)

    def test_collection_failure_retains_status_without_retry_or_audit(self):
        code,state,calls=self.run_case('collection_failure')
        self.assertEqual(code,1);self.assertEqual(calls,1)
        self.assertEqual(state['status'],'COLLECTION_STOPPED_FOR_DIAGNOSIS_NO_RETRY')
        self.assertIsNone(state['audit'])

    def test_success_exit_without_all_terminals_cannot_trigger_audit(self):
        code,state,calls=self.run_case('incomplete_queue')
        self.assertEqual(code,1);self.assertEqual(calls,1)
        self.assertIn('all declared terminal',state['error'])

    def test_failed_evidence_cannot_become_complete_pair_summary(self):
        code,state,calls=self.run_case('audit_evidence_failure')
        self.assertEqual(code,1);self.assertEqual(calls,2)
        self.assertIsNotNone(state['audit']['report']);self.assertIsNone(state['uncertainty'])

    def test_ineligible_outcome_is_retained_without_filtering_into_pair_summary(self):
        code,state,calls=self.run_case('ineligible')
        self.assertEqual(code,0);self.assertEqual(calls,2)
        self.assertEqual(state['status'],'COMPLETE_AUDITED_INELIGIBLE_CONDITIONS_RETAINED')

    def test_complete_requires_all_three_phases_and_receipts(self):
        code,state,calls=self.run_case('success')
        self.assertEqual(code,0);self.assertEqual(calls,3)
        self.assertEqual(state['status'],'COMPLETE_COLLECTION_AUDIT_AND_PAIR_SUMMARY')
        self.assertIsNotNone(state['uncertainty']['report'])


if __name__=='__main__':unittest.main()
