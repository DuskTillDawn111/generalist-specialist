"""CPU independent audit of joint-notice contract and frozen macro03 labels.

No GPU, validation capture, model fitting, or primary forecast mutation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/predictor-profile-training-evidence-v1-natural-20261028-review02'


def receipt(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require(ok,message):
    if not ok:raise ValueError(message)


def checked(ref):
    require(receipt(ref['path'])==ref,'Changed source: '+ref['path'])
    return json.loads(Path(ref['path']).read_text())


def independent_visible(events,cutoff,owner,phase):
    # Reduce only accepted observations, never occurrence-time future labels.
    active={}
    for event in events:
        if event['collector_processing_time_ns']>cutoff:continue
        alias=event['owner'];segment=event.get('segment_id')
        if event.get('event')=='start':
            semantic=event.get('semantic_phase')
            if segment in ('native_inference','native_ocr'):
                semantic='entry' if segment=='native_inference' else 'ocr_inference'
            elif semantic not in ('prefill','decode'):continue
            active[alias]=dict(owner=alias,request_id=event['request_id'],phase=semantic,
                phase_id=segment,available_ns=event['collector_processing_time_ns'])
        elif event.get('event')=='end' and segment in ('native_inference','native_ocr'):
            require(alias in active and active[alias]['request_id']==event['request_id'],
                'Native END must match visible request')
            del active[alias]
    peers=[active[alias] for alias in sorted(active) if alias!=owner]
    return dict(cutoff_processing_ns=cutoff,peer_visible_macro_owner_count=len(peers),
        peer_same_visible_macro_phase_owner_count=sum(x['phase']==phase for x in peers),
        peer_macros=peers,query_success_ends_macro=False,
        visible_macro_is_actual_SM_activity=False,
        macro_absence_or_presence_certifies_physical_completion=False)


def audit(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    macro_ref=receipt(BASE/'response-macros-v1-03.json');macro=checked(macro_ref)
    prior=checked(receipt(ROOT/'artifacts/predictor-stage-response-training-v2-natural-20261028-independentreview01/report.json'))
    stage=checked(prior['stage_training'])
    review=checked(prior['qualified_training_review']);raw_ref=review['inputs']['raw.json'];raw=checked(raw_ref)
    sources=[macro_ref,*macro['sources'],prior['stage_training'],prior['host_partition'],
        receipt(__file__),receipt(ROOT/'tools/predictor_validation_20261027/response_notification_posterior_reference_v1.py'),
        receipt(ROOT/'docs/native_prefix_joint_response_notification_contract_v1_20261028.md')]
    for ref in macro['sources']:checked(ref) if ref['path'].endswith('.json') else require(receipt(ref['path'])==ref,'Source drift')
    for ref in stage['sources']:require(receipt(ref['path'])==ref,'Stage source drift')
    rows={row['request_id']:row for row in raw['request_records']}
    require(len(rows)==len(raw['request_records']),'Unique raw request identities required')
    ids={row['request_id'] for row in macro['requests']}
    require(len(ids)==macro['measured_requests']==50,'Exact unique 50 training requests required')
    require(ids=={row['request_id'] for row in stage['stages']},'Original reviewed training request set changed')
    jobs={rows[ident]['profile_job_id'] for ident in ids}
    by_job={}
    for job in jobs:
        events=[]
        for ident in ids:
            row=rows[ident]
            if row['profile_job_id']!=job:continue
            for ordinal,event in enumerate(row['execution_events']):
                events.append(dict(event,owner=row['model'],_ordinal=ordinal))
        events.sort(key=lambda e:(e['collector_processing_time_ns'],e['owner'],e['request_id'],e['_ordinal']))
        by_job[job]=events
    macros_count=exposure_count=milestone_count=late_query_count=late_milestone_count=occupancy=0
    owner_count=Counter();g2_final=[]
    for record in macro['requests']:
        row=rows[record['request_id']];events=row['execution_events']
        forwards=[e for e in events if e.get('event')=='start' and e.get('semantic_phase') in ('prefill','decode')]
        native='native_inference' if forwards else 'native_ocr'
        starts=forwards or [e for e in events if e.get('event')=='start' and e.get('segment_id')==native]
        ends=[e['perf_ns'] for e in events if e.get('event')=='end' and e.get('segment_id')==native]
        require(len(ends)==1,'Unique native END required')
        left=row['worker_response']['start_perf_ns'];right=row['previous_handler_next_request_ready']['perf_ns']
        bounds=[left,*[e['perf_ns'] for e in starts],ends[0],right]
        intervals=[(a,b) for a,b in zip(bounds,bounds[1:]) if a!=b]
        require(intervals==[(m['left_ns'],m['right_ns']) for m in record['macros']],
            'Independent raw interval reconstruction differs')
        require(record['worker_start_to_ready_ns']==right-left,'Worker clock differs')
        require(sum(m['response_wall_ns'] for m in record['macros'])==right-left,'Macro wall not conserved')
        occupancy+=right-left
        for m in record['macros']:
            macros_count+=1;owner_count[m['owner']]+=1
            require(m['response_wall_ns']==m['right_ns']-m['left_ns'],'Response wall differs')
            for flag in ('additional_stream_elapsed_or_sync_wait_may_be_added','true_SM_service_identified'):
                require(m[flag] is False,'Unqualified physical/additive claim')
            require(m['prospective_shape_template_qualification_required'] is True,'Missing prospective qualification')
            expected=[(i,e) for i,e in enumerate(events) if m['left_ns']<=e['perf_ns']<m['right_ns']
                and e.get('event') in ('start','end','observation')]
            require(len(expected)==len(m['internal_training_milestones']),'Milestones missing or duplicated')
            for (ordinal,event),label in zip(expected,m['internal_training_milestones']):
                milestone_count+=1
                require(label['raw_event']==event and label['raw_event_ordinal']==ordinal,'Raw identity changed')
                require(label['raw_event_pointer']==dict(capture_sha256=raw_ref['sha256'],
                    request_id=row['request_id'],execution_event_ordinal=ordinal),'Raw pointer changed')
                require(label['runtime_outcome_known_at_start'] is False,'Future outcome treated as known')
                clocks=label['clocks']
                require([clocks[k] for k in ('occurrence_ns','reader_receipt_ns','dequeued_ns','processed_ns')]==
                    [event[k] for k in ('perf_ns','received_time_ns','collector_dequeued_time_ns','collector_processing_time_ns')],
                    'Milestone clock identity differs')
                if clocks['processed_ns']>m['right_ns']:
                    late_milestone_count+=1
                    if event.get('kind')=='device_event_query':late_query_count+=1
            path=m['causal_covariate_path'];cursor=m['left_ns']
            for piece in path:
                exposure_count+=1
                require(piece['left_ns']==cursor and piece['right_ns']>cursor,'Exposure overlap/gap')
                require(piece['duration_ns']==piece['right_ns']-cursor,'Exposure clock differs')
                require(piece['state']==independent_visible(by_job[row['profile_job_id']],cursor,m['owner'],m['semantic_phase']),
                    'Processed-prefix state differs from independent reducer')
                cursor=piece['right_ns']
            require(cursor==m['right_ns'],'Full macro exposure not covered')
        if row['model']=='generalist_2#0':
            final=next(m for m in record['macros'] if m['right_ns']==ends[0] and m['kind']=='language_forward_macro')
            g2_final.append(dict(request_id=row['request_id'],macro_id=final['macro_id'],
                final_forward_to_native_END_ns=final['response_wall_ns'],
                prospective_TTS_work_template_qualified=False))
    require(macros_count==macro['macros']==334,'Exact 334 macros required')
    late=[];phase_counts=Counter();ended_before_reader=0
    for s in stage['stages']:
        begin,end=s['response_labels']['host_observed_interval_ns']
        if end>=s['BEGIN_available_ns']:continue
        row=rows[s['request_id']]
        b=next(e for e in row['execution_events'] if e.get('segment_id')==s['phase_id'] and e.get('event')=='start')
        e=next(e for e in row['execution_events'] if e.get('segment_id')==s['phase_id'] and e.get('event')=='end')
        require(e['collector_processing_time_ns']>b['collector_processing_time_ns'],'END already visible at BEGIN cutoff')
        require(s['stage_kind']=='host_stage','Counterexamples are host stages, not language forwards')
        ended_before_reader+=end<b['received_time_ns'];phase_counts[s['phase_id']]+=1
        late.append(dict(request_id=s['request_id'],phase_id=s['phase_id'],BEGIN=dict(b),END=dict(e)))
    require(len(late)==137 and ended_before_reader==98,'Reviewed latency decomposition changed')
    require(dict(phase_counts)==dict(pre_sync=46,post_sync=44,final_sync=46,cache_reclamation=1),'Phase decomposition changed')
    v59_ref=receipt(ROOT/'artifacts/predictor-lifecycle-engine-checkpoint-v59-20261028-cpu01/report.json')
    v59=checked(v59_ref);frozen=[*v59['module_source_closure'],*v59['test_fixture_sources'],*v59['frozen_profile_sources']]
    for ref in frozen:require(receipt(ref['path'])==ref,'v59/frozen source changed: '+ref['path'])
    modules=['tests.test_composable_response_macros_v1','tests.test_response_notification_posterior_reference_v1',
        'tests.test_joint_macro_notification_contract_independent_v1']
    for name in modules:sources.append(receipt(ROOT/(name.replace('.','/')+'.py')))
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    (output/'tests.txt').write_text(log.getvalue());require(result.wasSuccessful(),'CPU tests failed')
    for ref in [*sources,*frozen]:require(receipt(ref['path'])==ref,'Source changed during audit')
    (output/'actual_delayed_BEGIN_cases.json').write_text(json.dumps(late,indent=2)+'\n')
    report=dict(schema='gsdeploy.native_prefix_joint_response_notification_contract_independent_review.v1',
        status='PASS_CPU_CONTRACT_AND_DISJOINT_TRAINING_MACROS_ONLY',sources=sources,
        actual=dict(requests=len(ids),macros=macros_count,total_worker_START_to_READY_ns=occupancy,
            macros_by_owner=dict(owner_count),processed_exposure_pieces=exposure_count,
            lossless_raw_milestones=milestone_count,milestone_notifications_after_physical_macro_END=late_milestone_count,
            query_notifications_after_physical_macro_END=late_query_count,stages=len(stage['stages']),
            host_stages_ended_before_BEGIN_processed=137,of_these_ended_before_BEGIN_reader=98,
            host_stages_crossing_END_during_reader_to_processing=39,late_host_phase_counts=dict(phase_counts),
            G2_final_macros=g2_final),
        tests=dict(modules=modules,run=result.testsRun,failures=len(result.failures),errors=len(result.errors),log=receipt(output/'tests.txt')),
        independent_raw_interval_reconstruction=True,independent_processed_prefix_reduction=True,
        raw_milestone_identity_preserved=True,query_true_does_not_clear_macro_tail=True,
        frozen_v59_report=v59_ref,frozen_v59_module_source_count=len(v59['module_source_closure']),
        frozen_v59_test_fixture_source_count=len(v59['test_fixture_sources']),
        frozen_profile_sources=v59['frozen_profile_sources'],frozen_receipts_unchanged=True,
        response_and_notification_joint_model_fitted=False,posterior_calibration_validated=False,
        future_G2_TTS_template_qualified=False,native_assimilation_implemented=False,
        physical_SM_service_observed=False,all_stream_completion_observed=False,
        main_queue_provider_eligible=False,final_gain_eligible=False,GPU_started=False,
        validation_labels_read=False,
        limitations=['Finite posterior weights are synthetic counterexample masses, not fitted probabilities.',
            'Macro labels are disjoint host response clocks including async tails, not resource-free or SM service evidence.',
            'Training labels do not qualify prospective future shapes or G2 generation/stop/TTS work templates.',
            '137/464 are one-capture host-stage latency facts, not language-forward or general mixture probabilities.',
            'Joint response/internal milestone/notification law, capture holdout and full-arrival queue validation remain required.'])
    (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return receipt(output/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(audit(parser.parse_args().output)))
