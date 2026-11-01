"""Synthetic CPU contracts; no real smoke/training/validation data are fitted."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import stage_response_features_v1 as f


def event(rid, edge, at, available, **fields):
    return dict(request_id=rid,event=edge,perf_ns=at,received_time_ns=available-2,
        collector_dequeued_time_ns=available-1,collector_processing_time_ns=available,**fields)


def language_row(owner,rid,start,end,phase,index,query_true):
    ident=phase+'_'+str(index)
    meta=dict(kind='gpu',segment_id=ident,semantic_phase=phase,forward_index=index,
        input_sequence_length=16 if phase=='prefill' else 1,input_batch_size=1)
    begin=event(rid,'start',start,start+10,**meta)
    finish=event(rid,'end',end,end+10,**meta)
    qbase=dict(kind='device_event_query',segment_id=ident,phase_id=ident)
    false=event(rid,'observation',end+1,end+11,end_event_complete=False,
        host_query_begin_ns=end+1,host_query_end_ns=end+1,recorded_stream_elapsed_ms=None,**qbase)
    true=event(rid,'observation',query_true,query_true+10,end_event_complete=True,
        host_query_begin_ns=query_true,host_query_end_ns=query_true,recorded_stream_elapsed_ms=.00015,**qbase)
    record=dict(id=ident,start_ns=start,end_ns=end,semantic_phase=phase,forward_index=index,
        input_sequence_length=meta['input_sequence_length'],input_batch_size=1)
    device=dict(phase_id=ident,end_record_host_bracket_ns=[end-1,end],recorded_stream_elapsed_ms=.00015)
    return dict(model=owner,request_id=rid,pool_id='pool-'+owner,task='text',profile_job_id='job',
        split='training',validation_view='training',status='completed',execution_events=[begin,finish,false,true],
        worker_response=dict(semantic_forward_observations=dict(records=[record],instrumentation_errors=[],
            device_progress=dict(records=[device]))))


class FeatureTests(unittest.TestCase):
    def fixture(self,root,mode='natural',split='training'):
        rows=[language_row('text#0','a',100,200,'prefill',0,220),
              language_row('generalist_2#0','b',120,140,'decode',1,250)]
        job=dict(id='job',split=split,validation_view=split,
            inputs={r['model']:r['pool_id'] for r in rows},tasks={r['model']:'text' for r in rows},
            start_offsets_ms={'text#0':0,'generalist_2#0':20})
        plan=dict(schema='gsdeploy.paired_profile_campaign.v6',split=split,mode=mode,
            progress_observer=True,mode_switch_at_execute=False,no_retries=True,final_gain_eligible=False,
            execution_context_sha256='context',jobs=[job])
        context=dict(protocol='paired_profile_v6_'+mode,paired_profile_mode=mode,
            worker_variant='semantic_progress_worker_v1.py',
            monitoring_scope='startup_through_final_drain_sample_before_worker_close')
        executed=dict(split=None,contexts=None,limit=None,trace=mode=='kernel',jobs=[job])
        for name,value in [('plan',plan),('context',context),('executed',executed)]:
            (root/(name+'.json')).write_text(json.dumps(value))
        raw=dict(plan=f.receipt(root/'plan.json'),context=f.receipt(root/'context.json'),
            executed_jobs=f.receipt(root/'executed.json'),execution_context_sha256='context',status='completed',
            performance_scope='unprofiled_with_phase_markers' if mode=='natural' else 'profiler_perturbed',
            monitoring_scope=context['monitoring_scope'],shutdown_memory_sampled=False,
            request_records=rows,priming_request_records=[{'forbidden':'do not inspect me'}],errors=[],
            runtime_errors=[],telemetry_errors=[])
        path=root/'raw.json';path.write_text(json.dumps(raw));return path,raw

    def test_labels_domains_and_parent_prefix_separate(self):
        with tempfile.TemporaryDirectory() as d:
            path,raw=self.fixture(Path(d));result=f.extract(path)
        a,b=result['stages']
        self.assertEqual(a['runtime_features_at_begin']['peer_visible_unconfirmed_owner_count'],0)
        # The first peer END is visible at150, but its successful query is only
        # available260. False query is evidence, not proof of truth-now.
        state=next(x for x in a['processed_covariate_path'] if x['cutoff_processing_ns']==151)
        self.assertEqual(state['peer_visible_unconfirmed_owner_count'],1)
        self.assertFalse(state['unconfirmed_is_actual_unfinished'])
        self.assertFalse(state['peer_visible_unconfirmed_phases'][0]['last_processed_query']['end_event_complete'])
        self.assertEqual(b['runtime_features_at_begin']['peer_same_semantic_phase_owner_count'],0)
        self.assertEqual(a['response_labels']['host_call_wall_ns'],100)
        self.assertAlmostEqual(a['response_labels']['recorded_current_stream_elapsed_ns'],150)
        self.assertFalse(a['response_labels']['host_and_stream_durations_additive'])
        self.assertEqual(a['response_labels']['host_label_domain'],'natural_host_call_wall_ns')
        self.assertEqual(result['priming_rows_excluded'],1)
        self.assertFalse(result['fit_performed'])

    def test_future_response_mutation_does_not_change_BEGIN_feature(self):
        with tempfile.TemporaryDirectory() as d:
            path,raw=self.fixture(Path(d));before=f.extract(path)['stages'][0]['runtime_features_at_begin']
            raw['request_records'][1]['worker_response']['generated_tokens']=999999
            raw['request_records'][1]['worker_response']['prediction']='future output'
            path.write_text(json.dumps(raw));after=f.extract(path)['stages'][0]['runtime_features_at_begin']
        self.assertEqual(before,after)
        self.assertIsNone(after['nominal_flops']);self.assertIsNone(after['resource_similarity_proxy'])

    def test_query_completion_only_available_after_parent_processed(self):
        rows=[language_row('text#0','a',100,200,'prefill',0,220)]
        events=f.ordered_events(rows)
        before=f.visible_prefix(events,229,'other','prefill')
        after=f.visible_prefix(events,230,'other','prefill')
        self.assertEqual(before['peer_visible_unconfirmed_owner_count'],1)
        self.assertEqual(after['peer_visible_unconfirmed_owner_count'],0)

    def test_completion_interval_uses_false_query_begin_not_success_as_point(self):
        with tempfile.TemporaryDirectory() as d:
            path,raw=self.fixture(Path(d));labels=f.extract(path)['stages'][0]['response_labels']
        interval=labels['current_stream_completion_interval_ns']
        self.assertEqual((interval['lower_ns'],interval['upper_ns']),(201,220))
        self.assertFalse(interval['right_censored'])

    def test_smoke_and_validation_refused_before_response_inspection(self):
        for split in ('smoke','validation'):
            with self.subTest(split=split),tempfile.TemporaryDirectory() as d:
                path,raw=self.fixture(Path(d),split=split)
                raw['request_records']=[{'worker_response':'bad response must never be parsed'}]
                path.write_text(json.dumps(raw))
                with self.assertRaisesRegex(ValueError,'smoke/validation forbidden'):f.extract(path)

    def test_profiler_label_cannot_masquerade_as_natural(self):
        with tempfile.TemporaryDirectory() as d:
            path,raw=self.fixture(Path(d),mode='kernel');result=f.extract(path)
            self.assertEqual(result['stages'][0]['response_labels']['host_label_domain'],
                'profiler_perturbed_host_call_wall_ns')
            raw['performance_scope']='unprofiled_with_phase_markers';path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'timing-domain'):f.extract(path)

    def test_changed_plan_receipt_and_incomplete_coverage_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path,raw=self.fixture(Path(d));(Path(d)/'plan.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Changed'):f.extract(path)
        with tempfile.TemporaryDirectory() as d:
            path,raw=self.fixture(Path(d));raw['request_records'].pop();path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'coverage'):f.extract(path)

    def test_out_of_order_availability_and_old_errors_not_filtered(self):
        with tempfile.TemporaryDirectory() as d:
            path,raw=self.fixture(Path(d));raw['request_records'][0]['execution_events'][0]['collector_processing_time_ns']=0
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'availability'):f.extract(path)
        with tempfile.TemporaryDirectory() as d:
            path,raw=self.fixture(Path(d));raw['telemetry_errors']=['unknown pid'];path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'errors cannot be filtered'):f.extract(path)

    def test_rank_rejects_collinear_phase_candidate_and_tracks_capture_blocks(self):
        def stage(n,same,sha):
            return dict(owner='text#0',semantic_phase='prefill',capture_sha256=sha,
                processed_covariate_path=[dict(peer_visible_unconfirmed_owner_count=n,
                    peer_same_semantic_phase_owner_count=same)])
        deficient=f.design_check([stage(1,1,'one'),stage(2,2,'one')])[-1]
        self.assertFalse(deficient['full_column_rank'])
        self.assertEqual(deficient['whole_capture_blocks'],1)
        good=f.design_check([stage(1,1,'one'),stage(2,0,'two'),stage(1,0,'three')])[-1]
        self.assertTrue(good['full_column_rank']);self.assertTrue(good['minimum_initial_repetition_design_met'])
        self.assertFalse(good['nonlinear_integrated_hazard_Jacobian_checked'])

    def test_right_censoring_is_preserved_and_no_fraction_invented(self):
        with tempfile.TemporaryDirectory() as d:
            path,raw=self.fixture(Path(d));row=raw['request_records'][0]
            row['execution_events'].pop();path.write_text(json.dumps(raw));stage=f.extract(path)['stages'][0]
        self.assertTrue(stage['response_labels']['current_stream_completion_interval_ns']['right_censored'])
        self.assertIsNone(stage['response_labels']['recorded_current_stream_elapsed_ns'])
        self.assertFalse(stage['device_frontier_ready'])


if __name__=='__main__':unittest.main()
