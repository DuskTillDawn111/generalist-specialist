"""Complete collector-to-law CPU fixtures, including missing physical evidence."""
import copy
import unittest

from gsdeploy.forecasting.native_training import (build_template,digest,fit_native_captures,
                                                 normalize_native,validate_capture)


POOL=[dict(task_type='text',pool_id='t'),dict(task_type='image',pool_id='i')]
ALIASES={'generalist_2#0':'generalist_2','image#0':'image'}


def capture(run,mode):
    context=dict(resident_aliases=ALIASES)
    context_id=digest(context);jobs=[];rows=[]
    inputs=[('generalist_2#0','text','t'),('generalist_2#0','image','i'),('image#0','image','i')]
    groups=[[item] for item in inputs] if mode=='solo' else [
        [inputs[0],inputs[2]],[inputs[1],inputs[2]]]
    for index,group in enumerate(groups):
        start=1_000_000_000+index*1_000_000_000
        job=dict(id=str(index),requests=[]);jobs.append(job)
        for alias,task,pool in group:
            rid=str(len(rows));pid=1 if alias.startswith('generalist') else 2
            job['requests'].append(dict(alias=alias,task=task,pool_id=pool))
            native_ms=10 if mode=='solo' else 20
            spans=[];events=[];left=start
            for name,kind,ms in [('setup','host',1),('native','gpu',native_ms),('tail','host',1)]:
                right=left+ms*1_000_000
                spans.append(dict(id=name,kind=kind,start_ns=left,end_ns=right))
                for event,timestamp in [('start',left),('end',right)]:
                    events.append(dict(request_id=rid,event=event,segment_id=name,kind=kind,
                                       perf_ns=timestamp,received_time_ns=timestamp+1))
                left=right
            rows.append(dict(request_id=rid,phase='training',status='completed',profile_job_id=str(index),
                model=alias,task=task,pool_id=pool,scheduled_time_ns=start,completion_time_ns=left+100,
                response_reader_event_enqueue_time_ns=left+100,execution_events=events,
                worker_response=dict(start_perf_ns=start,end_perf_ns=left,observed_stages=spans),
                previous_handler_next_request_ready=dict(request_id=rid,pid=pid,perf_ns=left+200,
                    received_time_ns=left+300,clock='host_perf_counter_ns',
                    boundary='previous_handler_next_request_ready',scope='after_response_flush_before_next_read')))
    raw=dict(schema='gsdeploy.native_activity_capture.v1',kind='cpu_fixture',status='completed',split='training',
        run_id=run,execution_context_sha256=context_id,admission_enabled=False,resident_aliases=ALIASES,
        errors=[],runtime_errors=[],telemetry_errors=[],request_records=rows,
        ready={alias:dict(clock='time.perf_counter_ns',pid=i+1) for i,alias in enumerate(ALIASES)},
        shutdown={alias:dict(forced=False,exit_code=0,bye_received=True) for alias in ALIASES},
        observation_start_ns=0,observation_end_ns=4_000_000_000,
        batches=[dict(id=str(index),start_ns=1_000_000_000+index*1_000_000_000,
            request_ids=[r['request_id'] for r in rows if r['profile_job_id']==str(index)]) for index in range(len(groups))],
        board_samples=[dict(time_ns=i*20_000_000,watts=50.,owned_used_mib=200.,foreign_pids=[])
                       for i in range(201)])
    plan=dict(split='training',mode=mode,run_id=run,execution_context_sha256=context_id,aliases=ALIASES,jobs=jobs)
    return dict(raw=raw,plan=plan,context=context,source='analytic-fixture:'+run)


class NativeTrainingTests(unittest.TestCase):
    def test_complete_full_pool_pipeline_fits_real_overlap_contract(self):
        bundles=[capture('solo1','solo'),capture('solo2','solo'),
                 capture('overlap1','overlap'),capture('overlap2','overlap')]
        result=fit_native_captures(bundles,POOL,block_seconds=.5)
        law=result['profiles']['native_activity_law']
        self.assertEqual(len(law['progress_groups']),3)
        self.assertTrue(all(d['accepted'] for d in law['fit_diagnostics']))
        self.assertEqual(len(result['lifecycle']['profiles']),3)
        self.assertFalse(result['admission_enabled'])
        self.assertFalse(result['independent_accuracy_qualified'])
        self.assertEqual(len(result['normalized_runs']),4)
        self.assertTrue(all(r['delays']['permit_release_ms'] is None
                            for d in result['normalized_runs'] for r in d['lifecycle_observations']))

    def test_dropped_solo_input_cannot_be_hidden_by_editing_plan(self):
        one,two=capture('one','solo'),capture('two','solo')
        two['raw']['request_records'].pop();two['plan']['jobs'].pop();two['raw']['batches'].pop()
        with self.assertRaisesRegex(ValueError,'full frozen input pool'):
            build_template([one,two],POOL)

    def test_capacity_and_heldout_data_cannot_enter_training(self):
        for split in ('development_capacity','confirmation'):
            bad=capture('one','solo');bad['raw']['split']=bad['plan']['split']=split
            with self.assertRaisesRegex(ValueError,'training-only'):
                validate_capture(bad)

    def test_phase_result_disagreement_or_nonlocal_ready_rejected(self):
        one,two=capture('one','solo'),capture('two','solo')
        template,_=build_template([one,two],POOL)
        bad=copy.deepcopy(one);bad['raw']['request_records'][0]['execution_events'][1]['perf_ns']+=1
        with self.assertRaisesRegex(ValueError,'Contiguous|disagree'):
            normalize_native(bad,template)
        bad=copy.deepcopy(one);bad['raw']['request_records'][0]['previous_handler_next_request_ready']['pid']=99
        with self.assertRaisesRegex(ValueError,'resident worker'):
            normalize_native(bad,template)

    def test_unobserved_input_or_same_run_not_independent_solo_evidence(self):
        one=capture('one','solo')
        with self.assertRaisesRegex(ValueError,'Distinct solo runs'):
            build_template([one,copy.deepcopy(one)],POOL)
        with self.assertRaisesRegex(ValueError,'full frozen input pool'):
            build_template([one,capture('two','solo')],POOL+[dict(task_type='text',pool_id='missing')])


if __name__=='__main__':unittest.main()
