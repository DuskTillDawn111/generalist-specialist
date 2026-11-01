"""Fault-injected CPU control checks; never starts an upstream or GPU process."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import admission_validation_continuation_v1 as m


class AdmissionValidationContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(m,'BASE',self.base).start()
        self.paths=m._paths()
        for key in ('training_protocol','training_review','validation_protocol'):
            path=Path(self.paths[key]);path.parent.mkdir(parents=True,exist_ok=True);m.save(path,dict(kind=key))
        self.campaign=self.base/'campaign.json';m.save(self.campaign,dict(kind='fixed campaign'))
        self.role=self.base/'roles.json';m.save(self.role,dict(kind='fixed roles'))
        self.tp=dict(campaign=m.receipt(self.campaign),role_reservation=m.receipt(self.role))
        for key in m.tc.MODULES:
            path=self.base/(key+'.json');m.save(path,dict(kind=key));self.tp[key]=m.receipt(path)
        self.vp=copy.deepcopy(self.tp)
        for key,value in [('training_protocol',self.tp),('validation_protocol',self.vp)]:
            Path(self.paths[key]).write_text(__import__('json').dumps(value))
        m.save(self.paths['training_started'],dict(source=m.receipt(m.tc.__file__),protocol=m.receipt(self.paths['training_protocol']),
            root_review=m.receipt(self.paths['training_review']),pid=m.PID))
        patch.object(m.tc,'audit_root_review',return_value=(self.tp,{})).start()
        patch.object(m.validation,'audit_protocol',return_value=self.vp).start()
        self.identity=patch.object(m.tc,'process_identity',return_value=m.expected_identity()).start()
        self.processes=patch.object(m,'_module_processes',return_value=[m.expected_identity()]).start()
        self.output=self.base/m.FOLDER

    def prepare(self):
        self.protocol=m.prepare(self.output);self.path=self.output/'protocol.json';return self.protocol

    def review(self):
        p=self.protocol;self.review_path=self.output/'root-review.json'
        value=dict(schema=m.SCHEMA+'.main_thread_review',decision='APPROVED_FOR_CONDITIONAL_V4_ONLY',
            reviewer_role='main_thread',protocol=m.receipt(self.path),created_ns=p['created_ns']+1,
            final_gain_eligible=False,final_launch=False,diagnostic_reason='Preserve FAIL; collect fixed V4 only',
            **{k:p[k] for k in ('source','training_protocol','training_review','validation_protocol','campaign',
                'role_reservation','paths','training_process_identity')})
        m.save(self.review_path,value);return value

    def rehash(self,value):
        value['protocol_sha256']=m.digest({k:v for k,v in value.items() if k!='protocol_sha256'})
        self.path.write_text(__import__('json').dumps(value))

    def complete_training(self):
        p=self.protocol;folder=self.base/m.T_FOLDER
        review=folder/'training-review.json';m.save(review,dict(kind='materialized training review'))
        jobs=[];completions=[];mapped={}
        for i in range(8):
            output=self.base/('training-job-'+str(i));output.mkdir();native=output/'raw.json';m.save(native,dict(kind='native',index=i))
            job=dict(job_id='job-'+str(i),output=str(output),controller_sha256='exact-controller')
            jobs.append(job);mapped[job['job_id']]=dict(native_raw=m.receipt(native))
            packet=dict(schema=m.validation.training.SCHEMA+'.completed',campaign=p['campaign'],job=job,
                review=m.receipt(review),source=m.receipt(m.validation.training.__file__),request_count=1200,
                controller_sha256=job['controller_sha256'],native_raw=m.receipt(native),GPU_started=True,final_gain_eligible=False)
            m.save(output/'training-complete.json',packet)
            completions.append(dict(job_id=job['job_id'],completion=m.receipt(output/'training-complete.json')))
        manifests={}
        for key,module in m.tc.MODULES.items():
            path=Path(p['paths']['manifests'][key]);path.parent.mkdir();m.save(path,dict(protocol=self.tp[key]))
            manifests[key]=m.receipt(path)
            patch.object(module,'verify_manifest',side_effect=lambda path: m.read(path)).start()
        upstream=dict(stage2_gate_status='FAIL',stage3_status='DIAGNOSTIC_INCOMPLETE',stage3_conclusion=False)
        m.save(folder/'upstream-audit.json',upstream)
        done=dict(schema=m.tc.SCHEMA+'.completed',source=m.receipt(m.tc.__file__),protocol=p['training_protocol'],
            root_review=p['training_review'],job_count=8,jobs=completions,training_review=m.receipt(review),
            manifests=manifests,upstream_audit=m.receipt(folder/'upstream-audit.json'),upstream_statuses=upstream,
            source_receipts=[],service_refitted=False,independent_or_final_launched=False,final_gain_eligible=False,GPU_started=True)
        m.save(p['paths']['training_completed'],done)
        patch.object(m.validation.training,'verify',return_value=dict(jobs=jobs)).start()
        patch.object(m.validation.training,'_review',return_value={}).start()
        self.map_training=patch.object(m.validation.training,'map_training',side_effect=lambda job,path:mapped[job['job_id']]).start()
        return done,jobs

    def fake_seal(self,protocol,resources,quality,capacity,output):
        import os
        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'')
        output=Path(output);output.mkdir();m.save(output/'seal-audit.json',dict(kind='fake CPU seal'))

    def fake_run(self,sealed,output,*,diagnostic):
        import os
        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'0');self.assertTrue(diagnostic)
        output=Path(output);output.mkdir();m.save(output/'run-state.json',dict(attempted=list(m.runner.ORDER),GPU_started=True))

    def fake_report(self,sealed,execution,output):
        import os
        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'')
        output=Path(output);output.mkdir()
        m.save(output/'report.json',dict(status='INCOMPLETE',candidate_coverage=dict(requested=4,observed=4)))
        m.save(output/'gate.json',dict(status='FAIL'))
        m.save(output/'capacity-qualification.json',dict(final_gain_eligible=False))
        m.save(output/'validation-audit.json',dict(evidence=[dict(known=1200) for _ in range(4)]))

    def test_prepare_only_freezes_sources_exact_process_and_fixed_outputs(self):
        with patch.object(m.runner,'run',side_effect=AssertionError('No GPU prepare')):
            p=self.prepare();self.assertEqual(m.audit_protocol(self.path),p)
        self.assertEqual(p['training_process_identity'],m.expected_identity())
        self.assertEqual(p['paths']['outputs']['execution'],str(self.base/m.V_FOLDER/'execution-v1'))
        self.assertFalse(p['training_outcomes_read']);self.assertFalse(p['independent_outcomes_read'])
        self.assertFalse(p['GPU_started']);self.assertFalse((self.output/'started.json').exists())

    def test_original_pid_argv_cwd_start_tick_recycle_or_duplicate_never_pass(self):
        p=self.prepare()
        for key,value in [('start_ticks',m.START_TICKS+1),('cwd','/foreign'),('argv',['python','other.py'])]:
            with self.subTest(key=key):
                changed=copy.deepcopy(m.expected_identity());changed[key]=value;self.identity.return_value=changed
                with self.assertRaisesRegex(ValueError,'identity changed'):m.upstream_status(p)
        self.identity.return_value=m.expected_identity();self.processes.return_value=[m.expected_identity(),dict(pid=m.PID+1)]
        with self.assertRaisesRegex(ValueError,'Duplicate'):m.upstream_status(p)

    def test_missing_completion_and_exited_original_is_terminal_even_without_gpu(self):
        p=self.prepare();self.identity.return_value=None;self.processes.return_value=[]
        with self.assertRaisesRegex(RuntimeError,'exited without completed'):m.upstream_status(p)

    def test_rehashed_protocol_cannot_replace_pid_argv_paths_roles_or_output_policy(self):
        p=self.prepare()
        for mutate in [lambda x:x['training_process_identity'].update(start_ticks=m.START_TICKS+1),
                lambda x:x['paths']['outputs'].update(execution=str(self.base/'replacement')),
                lambda x:x['policy'].update(final_launch=True),lambda x:x.update(independent_outcomes_read=True)]:
            changed=copy.deepcopy(p);mutate(changed);self.rehash(changed)
            with self.assertRaisesRegex(ValueError,'policy/process'):m.audit_protocol(self.path)

    def test_source_changed_after_prepare_is_rejected(self):
        self.prepare();self.campaign.write_text('changed source')
        with self.assertRaises(ValueError):m.audit_protocol(self.path)

    def test_independent_source_bound_root_review_required_before_wait_or_seal(self):
        self.prepare();review=self.review();self.assertEqual(m.audit_root_review(self.path,self.review_path)[1],review)
        for key,value in [('decision','APPROVED_FOR_FINAL'),('source',m.receipt(m.tc.__file__)),('final_launch',True),('diagnostic_reason','')]:
            changed=copy.deepcopy(review);changed[key]=value;self.review_path.write_text(__import__('json').dumps(changed))
            with patch.object(m,'wait_upstream',side_effect=AssertionError('No wait without exact review')):
                with self.assertRaisesRegex(ValueError,'conditional V4-only'):m.run(self.path,self.review_path)
        self.assertFalse((self.output/'started.json').exists())

    def test_wait_only_polls_original_process_no_launch_and_interval_is_bounded(self):
        self.prepare()
        with patch.object(m,'upstream_status',side_effect=['WAITING','COMPLETED']) as status,patch.object(m.time,'sleep') as sleep:
            m.wait_upstream(self.path,poll_seconds=.1)
        self.assertEqual(status.call_count,2);sleep.assert_called_once_with(.1)
        with self.assertRaises(ValueError):m.wait_upstream(self.path,poll_seconds=61)

    def test_wait_rehashes_complete_sources_without_reparsing_historical_closure(self):
        self.prepare()
        with patch.object(m,'source_closure',side_effect=AssertionError('No repeated old JSON parse')):
            self.assertEqual(m.audit_protocol(self.path),self.protocol)
            self.assertEqual(m.audit_protocol(self.path),self.protocol)
        self.campaign.write_text('changed historical source')
        with self.assertRaises(ValueError):m.audit_protocol(self.path)

    def test_real_duplicate_receipts_hash_each_unique_path_once_and_bind_original_snapshot(self):
        from collections import Counter
        original=self.base/'original.py';original.write_text('frozen source\n')
        snapshot=self.base/'snapshot.py';snapshot.write_text('frozen source\n')
        raw=self.base/'native.json';m.save(raw,dict(payload='real native bytes'*1000))
        context=self.base/'context.json';m.save(context,dict(execution_sources=[dict(original_path=str(original),frozen=m.receipt(snapshot))],
            duplicated=[m.receipt(raw)]*100))
        packet=self.base/'parent.json';m.save(packet,dict(context=m.receipt(context),same_context=[m.receipt(context)]*100,
            same_raw=[m.receipt(raw)]*100))
        original_checked=m.checked;hashed=[]
        def checked(ref):
            hashed.append(ref['path']);return original_checked(ref)  # real on-disk SHA verification
        with patch.object(m,'checked',side_effect=checked):
            refs=m.source_closure([m.receipt(packet),m.receipt(context),m.receipt(raw)]*20)
        self.assertEqual(set(Counter(hashed).values()),{1})
        self.assertEqual(len(refs),5);self.assertIn(m.receipt(original),refs)
        self.assertEqual({(r['path'],r['sha256']) for r in refs},
            {(r['path'],r['sha256']) for r in m.runner.closure([m.receipt(packet)])})
        changed={**m.receipt(raw),'sha256':'0'*64}
        with self.assertRaises(ValueError):m.source_closure([changed,m.receipt(packet)])
        original.write_text('changed original source\n')
        with self.assertRaises(ValueError):m.source_closure([m.receipt(packet)])

    def test_failed_wait_preserves_exclusive_started_and_never_retries_or_runs_gpu(self):
        self.prepare();self.review()
        with patch.object(m,'wait_upstream',side_effect=RuntimeError('original T8 failed')),patch.object(m.runner,'run') as launch:
            with self.assertRaisesRegex(RuntimeError,'original T8 failed'):m.run(self.path,self.review_path)
            with self.assertRaises(FileExistsError):m.run(self.path,self.review_path)
        self.assertEqual(launch.call_count,0);self.assertTrue((self.output/'started.json').exists())
        self.assertTrue(m.read(self.output/'failed.json')['no_retry'])

    def test_foreign_training_completion_or_partial_T8_never_seals(self):
        p=self.prepare();done=Path(p['paths']['training_completed'])
        wrong=dict(schema=m.tc.SCHEMA+'.completed',source=m.receipt(m.tc.__file__),protocol=p['training_protocol'],
            root_review=p['training_review'],job_count=7,service_refitted=False,independent_or_final_launched=False,
            final_gain_eligible=False,GPU_started=True)
        m.save(done,wrong)
        with self.assertRaisesRegex(ValueError,'finite source-bound'):m.training_evidence(p)
        wrong['job_count']=8;wrong['service_refitted']=True;done.write_text(__import__('json').dumps(wrong))
        with self.assertRaisesRegex(ValueError,'finite source-bound'):m.training_evidence(p)

    def test_cpu_seal_and_report_environment_is_restored_before_gpu_step(self):
        import os
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'3'}):
            def cpu():
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(os.environ['OMP_NUM_THREADS'],'1');return 'ok'
            self.assertEqual(m._cpu(cpu),'ok');self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'0')
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'3')
            with self.assertRaisesRegex(RuntimeError,'cpu failure'):m._cpu(lambda:(_ for _ in ()).throw(RuntimeError('cpu failure')))
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'0')

    def test_training_audit_requires_exact_eight_native_completions_and_three_fits(self):
        self.prepare();done,jobs=self.complete_training();evidence=m.training_evidence(self.protocol)
        self.assertEqual(self.map_training.call_count,8)
        self.assertEqual(evidence['upstream_statuses']['stage2_gate_status'],'FAIL')
        self.assertEqual(evidence['upstream_statuses']['stage3_status'],'DIAGNOSTIC_INCOMPLETE')
        done['jobs'][0],done['jobs'][1]=done['jobs'][1],done['jobs'][0]
        Path(self.paths['training_completed']).write_text(__import__('json').dumps(done))
        with self.assertRaisesRegex(ValueError,'all-eight ordered'):m.training_evidence(self.protocol)
        done['jobs'][0],done['jobs'][1]=done['jobs'][1],done['jobs'][0]
        Path(self.paths['training_completed']).write_text(__import__('json').dumps(done))
        packet_path=Path(jobs[0]['output'])/'training-complete.json';packet=m.read(packet_path);packet['request_count']=1199
        packet_path.write_text(__import__('json').dumps(packet))
        with self.assertRaisesRegex(ValueError,'completion packets'):m.training_evidence(self.protocol)

    def test_training_manifest_wrong_protocol_never_reaches_V4(self):
        self.prepare();self.complete_training()
        path=Path(self.paths['manifests']['quality_protocol']);value=m.read(path);value['protocol']=self.tp['resource_protocol']
        path.write_text(__import__('json').dumps(value))
        done=m.read(self.paths['training_completed']);done['manifests']['quality_protocol']=m.receipt(path)
        Path(self.paths['training_completed']).write_text(__import__('json').dumps(done))
        with self.assertRaisesRegex(ValueError,'changed mode protocol'):m.training_evidence(self.protocol)

    def test_once_successful_pipeline_seals_then_diagnostic_V4_reports_preserving_FAIL(self):
        import os
        self.prepare();self.review();self.complete_training()
        with (patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),patch.object(m,'wait_upstream'),
              patch.object(m.validation,'seal',side_effect=self.fake_seal) as seal,
              patch.object(m.runner,'run',side_effect=self.fake_run) as launch,
              patch.object(m.runner,'report',side_effect=self.fake_report),
              patch.object(m.validation,'verify_report',return_value=dict(verified=True,GPU_started=False))):
            value=m.run(self.path,self.review_path)
            with self.assertRaises(FileExistsError):m.run(self.path,self.review_path)
        self.assertEqual(seal.call_count,1);self.assertEqual(launch.call_count,1)
        self.assertEqual(self.map_training.call_count,8)  # one training proof, no per-job refit
        self.assertEqual(value['native_requested'],4800);self.assertEqual(value['native_known'],4800)
        self.assertEqual(value['independent_gate_status'],'FAIL');self.assertEqual(value['report_status'],'INCOMPLETE')
        self.assertFalse(value['final_launched']);self.assertFalse(value['training_launched']);self.assertFalse(value['service_refitted'])
        m.runner.recheck(value['source_receipts'])

    def test_collection_failure_reports_incomplete_once_and_does_not_retry(self):
        import os
        self.prepare();self.review();self.complete_training()
        def failed(sealed,output,*,diagnostic):
            self.fake_run(sealed,output,diagnostic=diagnostic)
            state=Path(output)/'run-state.json';state.write_text(__import__('json').dumps(dict(attempted=[m.runner.ORDER[0]],GPU_started=True)))
            raise RuntimeError('bounded collection failure')
        with (patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),patch.object(m,'wait_upstream'),
              patch.object(m.validation,'seal',side_effect=self.fake_seal),
              patch.object(m.runner,'run',side_effect=failed) as launch,
              patch.object(m.runner,'report',side_effect=self.fake_report),
              patch.object(m.validation,'verify_report',return_value=dict(verified=True))):
            value=m.run(self.path,self.review_path)
        self.assertEqual(launch.call_count,1);self.assertEqual(value['status'],'INCOMPLETE_NO_RETRY')
        self.assertEqual(value['independent_runs_launched'],1);self.assertEqual(value['native_requested'],4800)
        self.assertIn('bounded collection failure',value['collection_error']);self.assertFalse(value['final_gain_eligible'])

    def test_seal_failure_stops_without_launch_and_preserves_no_retry(self):
        self.prepare();self.review();self.complete_training()
        with (patch.object(m,'wait_upstream'),patch.object(m.validation,'seal',side_effect=ValueError('unusable evidence')),
              patch.object(m.runner,'run') as launch):
            with self.assertRaisesRegex(ValueError,'unusable evidence'):m.run(self.path,self.review_path)
        self.assertEqual(launch.call_count,0);self.assertTrue((self.output/'failed.json').exists())


if __name__=='__main__':unittest.main()
