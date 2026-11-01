"""CPU-only queue identity, publication and ordered downstream checks."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_public181_after_kernel_v1 as m


class AfterKernelTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        parent = dict(output=str(root/'kernel-run'), queue_identity=dict(pid=12), device_attributes={'path': '/device'}, campaign={'path': '/campaign'})
        parent_ref = m.save(root/'kernel-manifest.json', parent)
        campaign = dict(runs=[dict(capture_id=str(i)) for i in range(3)])
        resource = dict(attempts=[dict(attempt_id=str(i), claim=str(root/('claim'+str(i))),
            collection=str(root/('collection'+str(i)))) for i in range(8)])
        resource_ref = m.save(root/'resource-manifest.json', resource)
        claim = Path(parent['output']); claim.mkdir()
        claim_ref = m.save(claim/'claim.json', dict(manifest=parent_ref, queue_identity=parent['queue_identity'], no_retries=True))
        identity = dict(pid=55, start_ticks='555', command=['python', '-m', m.MODULE, 'run', '--manifest', parent_ref['path'], '--execute'],
            manifest=parent_ref, claim=claim_ref)
        manifest = dict(schema=m.SCHEMA, resource_manifest=resource_ref, kernel_manifest=parent_ref,
            kernel_claim=claim_ref, kernel_identity=identity, kernel_final_path=str(claim/'report.json'),
            output=str(root/'output'), sources=[])
        return manifest, resource, parent, campaign

    def final(self, manifest, parent, campaign):
        return dict(schema=m.kernel.SCHEMA+'.run', manifest=manifest['kernel_manifest'],
            status='COMPLETE_THREE_NATIVE_CAPTURES_AND_CPU_DIAGNOSTICS_NO_FIT', fit_started=False,
            automatic_retry=False, validation_labels_read=False,
            completed_captures=[dict(capture_id=r['capture_id']) for r in campaign['runs']],
            completed_cpu_stages=[dict(name=str(i)) for i in range(12)])

    def test_exact_writer_command_claim_and_start_ticks(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, _, _, _ = self.fixture(folder)
            proc = Path(folder)/'proc/55'; proc.mkdir(parents=True); (proc/'cwd').symlink_to(Path(folder))
            command = manifest['kernel_identity']['command']
            (proc/'cmdline').write_bytes(b'\0'.join(part.encode() for part in command)+b'\0')
            (proc/'stat').write_text('55 (python worker) '+' '.join(['S']+['0']*18+['555']))
            identity = m.kernel_identity(55, manifest['kernel_manifest'], manifest['kernel_claim'], proc_root=proc.parent)
            self.assertEqual(identity, manifest['kernel_identity'])
            for change in ('module', 'manifest', 'execute'):
                value = list(command)
                value[2 if change=='module' else 5 if change=='manifest' else 6] = 'foreign'
                (proc/'cmdline').write_bytes(b'\0'.join(part.encode() for part in value))
                with self.assertRaises(ValueError):
                    m.kernel_identity(55, manifest['kernel_manifest'], manifest['kernel_claim'], proc_root=proc.parent)

    def test_stopped_missing_duplicate_capture_or_stage_final_never_launches(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, _, parent, campaign = self.fixture(folder)
            for mode in ('status', 'capture', 'stage', 'fit', 'role'):
                final = self.final(manifest, parent, campaign)
                if mode=='status': final['status']='STOPPED_NO_RETRY_PREFIX_PRESERVED'
                if mode=='capture': final['completed_captures'][2]['capture_id']='0'
                if mode=='stage': final['completed_cpu_stages'].pop()
                if mode=='fit': final['fit_started']=True
                if mode=='role': final['validation_labels_read']=True
                Path(manifest['kernel_final_path']).write_text(json.dumps(final))
                with self.subTest(mode=mode), patch.object(m.kernel, 'cpu_stages', return_value=[dict(name=str(i)) for i in range(12)]):
                    with self.assertRaises(ValueError): m.completed_kernel(manifest, parent, campaign)

    def test_pid_reuse_rejected_even_if_final_is_complete_and_writer_exit_without_final_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, _, parent, campaign = self.fixture(folder)
            with patch.object(m, 'completed_kernel', return_value=({'path': '/complete'}, {})):
                with self.assertRaisesRegex(ValueError, 'start ticks'):
                    m.wait_kernel(manifest, parent, campaign, lambda: None,
                        identity_reader=lambda *a: dict(manifest['kernel_identity'], start_ticks='new'))
            with patch.object(m, 'completed_kernel', return_value=None):
                with self.assertRaisesRegex(ValueError, 'exited without complete'):
                    m.wait_kernel(manifest, parent, campaign, lambda: None,
                        identity_reader=lambda *a: (_ for _ in ()).throw(FileNotFoundError()))

    def test_complete_final_can_be_read_after_exact_writer_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, _, parent, campaign = self.fixture(folder)
            expected = ({'path': '/complete'}, {})
            with patch.object(m, 'completed_kernel', return_value=expected):
                result = m.wait_kernel(manifest, parent, campaign, lambda: None,
                    identity_reader=lambda *a: (_ for _ in ()).throw(FileNotFoundError()))
            self.assertEqual(result, expected)

    def test_ordered_native_then_intake_clears_cuda_and_never_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, resource, parent, campaign = self.fixture(folder)
            source = m.save(Path(folder)/'manifest.json', manifest)
            parent_ref = m.save(Path(folder)/'parent-final.json', dict(status='CPU_FIXTURE_ONLY'))
            calls=[]
            def command(command, **kwargs):
                calls.append(command)
                output = Path(command[command.index('--output')+1]); output.mkdir()
                if len(calls)==1:
                    reviews=[]
                    for a in resource['attempts']:
                        reviews.append(m.save(output/('review'+a['attempt_id']+'.json'), {'CPU_TEST_ONLY':True}))
                    document=dict(schema=m.executor.CONT_SCHEMA, manifest=manifest['resource_manifest'],
                        status='COMPLETE_EIGHT_NATIVE_RESOURCE_ATTEMPTS_REVIEWED_NO_FIT', fit_started=False,
                        validation_labels_read=False, completed_attempts=[dict(attempt_id=a['attempt_id'],
                            training_review=review) for a,review in zip(resource['attempts'],reviews)])
                else:
                    self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                    self.assertEqual(command.count('--review'), 8)
                    document=dict(schema=m.intake.SCHEMA, manifest=manifest['resource_manifest'],
                        status='COMPLETE_EIGHT_NATIVE_RESOURCE_TUPLES_REDERIVED_NO_FIT',
                        observed_resource_attempts=8, unknown_resource_attempts=0)
                m.save(output/'report.json', document)
                return types.SimpleNamespace(returncode=0)
            with patch.object(m, 'audit', return_value=(manifest,resource,parent,campaign)), \
                    patch.object(m, 'wait_kernel', return_value=(parent_ref, {})), \
                    patch.object(m.executor, 'audit', return_value=resource):
                ref = m.run(source['path'], run_command=command)
                result = m.checked(ref)
                self.assertEqual(result['status'], 'COMPLETE_EIGHT_NATIVE_RESOURCE_TRAINING_AND_INTAKE_NOT_VALIDATION')
                self.assertEqual(len(calls), 2); self.assertFalse(result['fit_started'])
                with self.assertRaises(FileExistsError): m.run(source['path'], run_command=command)
                self.assertEqual(len(calls), 2)

    def test_exit_zero_with_native_hold_cannot_start_intake(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, resource, parent, campaign = self.fixture(folder)
            source = m.save(Path(folder)/'manifest.json', manifest)
            parent_ref = m.save(Path(folder)/'parent-final.json', {})
            def held(command, **kwargs):
                output=Path(command[command.index('--output')+1]);output.mkdir()
                m.save(output/'report.json',dict(status='HOLD_CLAIMED_ATTEMPT_NO_RETRY'))
                return types.SimpleNamespace(returncode=0)
            with patch.object(m, 'audit', return_value=(manifest,resource,parent,campaign)), \
                    patch.object(m, 'wait_kernel', return_value=(parent_ref,{})), \
                    patch.object(m.executor, 'audit', return_value=resource), patch.object(m, 'actions', wraps=m.actions):
                result=m.checked(m.run(source['path'],run_command=held))
            self.assertEqual(result['status'],'STOPPED_NO_RETRY_PREFIX_PRESERVED')
            self.assertFalse((Path(manifest['output'])/'intake-started.json').exists())


if __name__=='__main__': unittest.main()
