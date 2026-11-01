"""CPU-only sequence, predecessor, identity, capacity and no-retry checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import kernel_dag_response_replication_continuation_v1 as m
from tools.predictor_validation_20261027.native_collect import receipt


class ContinuationTests(unittest.TestCase):
    def test_queue_pid_start_ticks_command_and_canonical_claim_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);proc=root/'proc/77';proc.mkdir(parents=True);cwd=root/'work';cwd.mkdir();(proc/'cwd').symlink_to(cwd)
            out=cwd/'queue';out.mkdir();native=cwd/'matching';native.mkdir();manifest=cwd/'matching-manifest.json';manifest.write_text('{}')
            claim=dict(schema=m.QUEUE_SCHEMA,matching_manifest=receipt(manifest),matching_output=str(native),
                       fit_manifest=receipt(manifest),independent_review=receipt(manifest))
            (out/'claim.json').write_text(json.dumps(claim));claim_ref=receipt(out/'claim.json')
            command=['python','-m',m.QUEUE_MODULE,'--matching-manifest','matching-manifest.json','--fit-manifest','matching-manifest.json',
                '--independent-review','matching-manifest.json','--output','queue','--matching-output','matching','--execute']
            (proc/'cmdline').write_bytes(b'\0'.join(v.encode() for v in command)+b'\0')
            (proc/'stat').write_text('77 (queue) '+' '.join(['S']+['0']*18+['ticks']+['0']*4))
            actual=m.queue_identity(77,claim_ref,native/'report.json',proc_root=root/'proc')
            self.assertEqual(actual['start_ticks'],'ticks')
            (proc/'cmdline').write_bytes(b'\0'.join(v.encode() for v in command[:-1]))
            with self.assertRaises(ValueError):m.queue_identity(77,claim_ref,native/'report.json',proc_root=root/'proc')

    def test_independent_review_requires_exact_campaign_and_zero_blockers(self):
        ref={'path':'root-review'};campaign={'path':'campaign','sha256':'a'*64}
        good=dict(schema=m.REVIEW_SCHEMA,status=m.REVIEW_STATUS,campaign=campaign,
                  unresolved_blocking_findings=[],test_summary=dict(status='PASS'))
        for key in ('schema','status','campaign','unresolved_blocking_findings','test_summary'):
            bad=dict(good);bad[key]='wrong'
            with self.subTest(key=key),patch.object(m,'read',return_value=bad),self.assertRaises(ValueError):m.checked_review(ref,campaign)

    def fixture(self,root):
        root=Path(root);claim=root/'queue-claim.json';claim.write_text('{}')
        identity=dict(pid=77,start_ticks='ticks',command=['pinned'],claim=receipt(claim),final_path=str(root/'matching-final.json'))
        manifest=dict(queue_identity=identity,queue_claim=receipt(claim),matching_final_path=str(root/'matching-final.json'),
            queue_final_path=str(root/'queue-final.json'),output=str(root/'output'))
        return manifest,dict(after_matching_manifest={'path':'manifest'})

    def test_wait_handles_partial_json_only_with_exact_live_writer_and_never_needs_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest,campaign=self.fixture(temp);calls=[]
            def pause(seconds):
                calls.append(seconds);Path(manifest['matching_final_path']).write_text('{}')
                Path(manifest['queue_final_path']).write_text(json.dumps(dict(schema=m.QUEUE_SCHEMA,status=m.QUEUE_STATUS,
                    matching_manifest=campaign['after_matching_manifest'],matching_report=receipt(manifest['matching_final_path']))))
            Path(manifest['matching_final_path']).write_text('{')
            with patch.object(m.producer,'require_matching_completion') as native_gate:
                final,queue=m.wait_matching(manifest,campaign,lambda:None,poll_seconds=1,
                    identity_reader=lambda *args:manifest['queue_identity'],pause=pause)
            self.assertEqual(calls,[1]);native_gate.assert_called_once();self.assertEqual(final,receipt(manifest['matching_final_path']))

    def test_wait_rejects_pid_reuse_and_writer_exit_before_complete_final(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest,campaign=self.fixture(temp)
            changed=dict(manifest['queue_identity'],start_ticks='reused')
            with self.assertRaisesRegex(ValueError,'identity changed'):
                m.wait_matching(manifest,campaign,lambda:None,identity_reader=lambda *args:changed,pause=lambda _:None)
            def absent(*args):raise FileNotFoundError('exited')
            with self.assertRaisesRegex(ValueError,'exited before complete'):
                m.wait_matching(manifest,campaign,lambda:None,identity_reader=absent,pause=lambda _:None)

    def test_capacity_waits_without_claim_or_gpu_and_uses_only_observed_required_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest,_=self.fixture(temp);waits=[];checks=[]
            free=iter([99,100])
            result=m.wait_capacity(manifest,100,lambda:checks.append(True),phase='native',poll_seconds=1,
                pause=lambda value:waits.append(value),disk_usage=lambda _:SimpleNamespace(free=next(free)))
            self.assertEqual(waits,[1]);self.assertEqual(len(checks),2)
            self.assertEqual(result['extra_arbitrary_reserve_bytes'],0);self.assertEqual(result['free_bytes'],100)

    def test_cpu_commands_cover_three_reviews_audits_intakes_and_instance_bindings(self):
        campaign=dict(runs=[dict(capture_id='c'+str(i),run='/native/'+str(i)) for i in range(3)])
        actions=m.cpu_stages('/campaign',campaign,'/device','/output')
        self.assertEqual(len(actions),12)
        self.assertEqual([a['name'] for a in actions],
            [str(i)+'-'+phase for i in range(3) for phase in ('native-review','occupancy','intake','producer-review')])
        for action in actions:self.assertNotIn('--fit',action['command']);self.assertNotIn('--execute',action['command'])
        audit=actions[1]
        with self.assertRaises(ValueError):m.validate_cpu_result(audit,dict(schema=audit['schema'],sensitivity=dict(all_passed=True),
            traces=[dict(launch_count=3,known_launch_count=2,unknown_launch_count=1)]))

    def test_started_capture_failure_preserves_prefix_and_never_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest,campaign=self.fixture(root);manifest_path=root/'manifest.json';manifest_path.write_text('{}')
            storage=root/'storage.json';storage.write_text(json.dumps(dict(new_chain_peak_scenario_bytes=10,
                estimated_native_capture_bytes=3,original_GPU_collector_floor_bytes=1,original_GPU_collector_preflight_scenario_bytes=1)))
            manifest.update(storage_scenario=receipt(storage),sources=[],campaign={'path':'campaign'})
            campaign['runs']=[]
            for index in range(3):
                run=root/('native'+str(index));run.mkdir();campaign['runs'].append(dict(capture_id='c'+str(index),run=str(run)))
            final=root/'final.json';final.write_text('{}');queue=root/'queue-final.json';queue.write_text('{}')
            calls=[]
            def child(command,**kwargs):
                calls.append(command);entry=campaign['runs'][len(calls)-1]
                if len(calls)==1:
                    value=dict(schema=m.producer.SCHEMA+'.execution',capture_id=entry['capture_id'],status='COLLECTED',outer_source_guard_passed=True)
                    (Path(entry['run'])/'response-replication-result.json').write_text(json.dumps(value))
                    return SimpleNamespace(returncode=0)
                return SimpleNamespace(returncode=1)
            class Guard:
                def __init__(self,*args):pass
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def check(self):pass
            with patch.object(m,'audit',return_value=(manifest,campaign)),patch.object(m,'SourceChangeGuard',Guard),\
                 patch.object(m,'wait_matching',return_value=(receipt(final),receipt(queue))),patch.object(m,'idle'),\
                 patch.object(m.producer,'audit',return_value=campaign),patch.object(m,'wait_capacity',return_value={}):
                result=m.run(manifest_path,run_command=child)
            self.assertEqual(len(calls),2);self.assertEqual(len(result['completed_captures']),1)
            self.assertEqual(result['status'],'STOPPED_NO_RETRY_PREFIX_PRESERVED')
            self.assertTrue((Path(manifest['output'])/'capture-1-started.json').exists())
            self.assertFalse((Path(manifest['output'])/'capture-2-started.json').exists())
            self.assertFalse(result['completed_cpu_stages'])
            with patch.object(m,'audit',return_value=(manifest,campaign)),self.assertRaises(FileExistsError):m.run(manifest_path,run_command=child)


if __name__=='__main__':unittest.main()
