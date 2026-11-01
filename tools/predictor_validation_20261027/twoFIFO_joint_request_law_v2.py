"""New-schema joint request work/notice law with explicit UNKNOWN transport.

Whole-attempt protocol recurrence includes priming history. Blocks, never raw
SHAs, are equal-weight latent units. Complete request vectors move jointly.
"""
import argparse
from collections import defaultdict
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from . import twoFIFO_phase12_fit_v1 as response
from . import phase_heterogeneous_protocol_candidate_v1 as rates
from .joint_request_response_transport_candidate_v2 import fraction_record,fraction_read,lags
from .protocol_timing_slack_v3 import derive,checked as checked_slacks
from .worker_stdin_fifo_contract_v1 import TransportDraw,sender_fifo
from .macro_response_queue_reference_v1 import Mark,Macro,RequestPlan,validate_plan
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.twoFIFO_whole_request_joint_training_law.v2'
require=response.require


def source_messages(raw):
    messages=[]
    for row in raw['priming_request_records']+raw['request_records']:
        owner,rid=row['model'],row['request_id']
        for ordinal,event in enumerate(row['execution_events']):
            messages.append(dict(identity=(owner,rid,'event:'+str(ordinal)),producer=owner,
                occurrence_ns=event['perf_ns'],reader_ns=event['received_time_ns'],
                dequeue_ns=event['collector_dequeued_time_ns'],processing_ns=event['collector_processing_time_ns']))
        ready=row['previous_handler_next_request_ready']
        for identity,occurrence,reader,dequeue,processing in (
            ('RESULT',row['worker_response']['end_perf_ns'],row['response_reader_event_enqueue_time_ns'],row['response_collector_dequeued_time_ns'],row['response_collector_processing_time_ns']),
            ('worker_READY',ready['perf_ns'],ready['received_time_ns'],ready['collector_dequeued_time_ns'],ready['collector_processing_time_ns'])):
            messages.append(dict(identity=(owner,rid,identity),producer=owner,occurrence_ns=occurrence,
                reader_ns=reader,dequeue_ns=dequeue,processing_ns=processing))
    return messages


def public_source_order(raw,plan):
    by_id={r['request_id']:r for r in raw['request_records']}
    return raw['priming_request_records']+[by_id[a['request_id']] for c in plan['cases']
        for a in sorted(c['arrivals'],key=lambda a:(a['arrival_ns'],a['arrival_index']))]


def transport_constraints(row,previous):
    require(previous is not None,'Measured request requires actual priming/prior owner history')
    submit=row['actual_submit_time_ns'];enqueue=row['enqueued_time_ns'];ready=previous['previous_handler_next_request_ready']['perf_ns']
    turn=submit-max(enqueue,previous['submit_write_completed_time_ns'])
    require(turn>=0,'Actual original sender release order required')
    return dict(enqueue_after_arrival_ns=enqueue-row['scheduled_time_ns'],sender_turn_extra_ns=turn,
        submit_to_write_record_ns=row['submit_write_completed_time_ns']-submit,
        observed_submit_to_START_ns=row['observed_worker_start_time_ns']-submit,
        previous_READY_minus_submit_ns=ready-submit,
        observed_previous_handler_busy_at_submit=submit<ready,
        submit_to_stdin_line_available_ns=None,worker_release_to_START_extra_ns=None,
        latent_split_identified=False,fitted_transport_probability_law=False)


def request_vector(request,row,fit,slacks,constraints):
    nodes=[];used=set();baselines={tuple(b['group']):b['log_reference_ms'] for b in fit['baselines']}
    events=[('event:'+str(i),e['perf_ns'],e) for i,e in enumerate(row['execution_events'])]
    events.append(('RESULT',row['worker_response']['end_perf_ns'],None))
    for index,macro in enumerate(request['macros']):
        left,right=macro['left_ns'],macro['right_ns'];marks=[]
        for identity,at,event in events:
            if not(left<=at<right or (index==len(request['macros'])-1 and at==right)):continue
            require(identity not in used,'Protocol mark must occur in exactly one complete macro')
            used.add(identity)
            if identity=='RESULT':action='result';visible_phase=None
            else:
                edge=event.get('event');segment=event.get('segment_id')
                query=event.get('kind')=='device_event_query'
                if query:
                    require(event.get('evidence_scope')=='recorded_current_stream_only'
                        and type(event.get('end_event_complete')) is bool
                        and all(k in event for k in ('event_id','phase_id','stream_id','device_index','host_query_begin_ns','host_query_end_ns')),
                        'Complete explicit query outcome/current-stream scope required')
                    action='query'
                elif edge=='start' and (segment in ('native_inference','native_ocr') or event.get('semantic_phase') in ('prefill','decode')):action='phase'
                elif edge=='end' and segment in ('native_inference','native_ocr'):action='clear'
                else:action='diagnostic'
                visible_phase=('entry' if segment=='native_inference' else 'ocr_inference' if segment=='native_ocr' else event.get('semantic_phase')) if action=='phase' else None
            marks.append(dict(identity=identity,reference_work_at=fraction_record(rates.progress(macro,at,fit)),
                protocol_slacks=slacks[identity],action=action,phase=visible_phase,
                query_complete=event.get('end_event_complete') if event and action=='query' else None,
                event_reference_phase_id=event.get('phase_id') if event and action=='query' else None,
                clear_scope='request' if action=='clear' else None))
        marks.sort(key=lambda m:fraction_read(m['reference_work_at']))
        shape=macro['forward_shape'] or {};group=(macro['owner'],macro['pool_id'],macro['task'],macro['macro_id'],macro['semantic_phase'],macro['kind'],
            *(shape.get(k) for k in ('input_sequence_length','input_batch_size','forward_index')))
        require(group in baselines,'Exact complete response nuisance group required')
        work=rates.progress(macro,right,fit)
        nodes.append(dict(macro_id=macro['macro_id'],phase=macro['semantic_phase'],kind=macro['kind'],
            reference_work_ns=fraction_record(work),marks=marks,baseline_reference_ms=math.exp(baselines[group]),
            log_reference_work_residual=math.log(float(work)/1e6)-baselines[group],forward_shape=macro['forward_shape']))
    require(used==set(slacks)-{'worker_READY'},'All original START/PHASE/RESULT messages must be retained once')
    starts=[e for e in row['execution_events'] if e.get('event') is None]
    require(len(starts)==1 and starts[0]['perf_ns']==row['observed_worker_start_time_ns'],'Explicit actual START message required')
    vector=dict(public_key=[row['model'],row['task'],row['pool_id']],nodes=nodes,
        ready_protocol_slacks=slacks['worker_READY'],transport_constraints=constraints,
        dispatch_to_worker_start_ns=0,dispatch_delay_is_structural_zero_transport_is_separate=True,
        all_entry_forward_OCR_READY_nodes_retained=True,query_outcomes_and_notice_coordinates_joint=True)
    vector['vector_id']=hashlib.sha256(json.dumps(vector,sort_keys=True).encode()).hexdigest()
    return vector


def build_joint_law_v2(documents,plans,raws,qualified_fit,*,heldout_block_id=None):
    denominator=response.intake.complete_documents(plans,documents)
    require(denominator['measured_requests']==546 and len(raws)==12,'Complete twelve-attempt law required')
    require(qualified_fit.get('response_law_compilation_eligible') is True
        and qualified_fit.get('features')==list(response.FEATURES)
        and qualified_fit.get('fitted_root_numerical_qualification') is True
        and qualified_fit.get('denominator')==denominator,'Full and whole-block fitted-root qualified phase12 response required')
    fit=qualified_fit['full_fit'];require(fit['numerical_qualification'] is True and set(fit['whole_blocks'])==set(response.intake.campaign_v2.strict.CAPTURES),'Exact three-block fitted response required')
    folds=qualified_fit['whole_block_leaveout']
    require(len(folds)==3 and {f['heldout_block_id'] for f in folds}==set(response.intake.campaign_v2.strict.CAPTURES)
        and all(f['training_fit']['numerical_qualification'] is True
            and f['evaluation']['summary']['unknown_rows']==0
            and set(f['training_fit']['whole_blocks'])==set(response.intake.campaign_v2.strict.CAPTURES)-{f['heldout_block_id']}
            for f in folds),'Every complete whole-block fitted root and heldout nuisance support required')
    all_blocks=list(response.intake.campaign_v2.strict.CAPTURES)
    require(heldout_block_id is None or heldout_block_id in all_blocks,'Explicit planned whole-block holdout required')
    blocks=[b for b in all_blocks if b!=heldout_block_id]
    if heldout_block_id is not None:fit=next(f['training_fit'] for f in folds if f['heldout_block_id']==heldout_block_id)
    vectors=[];provenance=[];decompositions=[]
    for doc,plan,raw in zip(documents,plans,raws):
        if plan['capture_id']==heldout_block_id:continue
        require(all(raw[k]==plan[k]==doc[k] for k in ('capture_id','attempt_id','execution_id')),'Raw/document/plan identity mismatch')
        require([r['request_id'] for r in raw['request_records']]==[r['request_id'] for r in doc['requests']]
            and len(raw['priming_request_records'])==18,'Complete raw measured/priming denominator required')
        messages=source_messages(raw);derived=derive(messages)
        require(len(messages)==len(derived)==sum(len(r['execution_events'])+2 for r in raw['priming_request_records']+raw['request_records']),
            'Dynamic whole-attempt original message count mismatch')
        indexed={tuple(m['identity']):m['protocol_slacks'] for m in derived};prior={};constraints={}
        priming_ids={r['request_id'] for r in raw['priming_request_records']}
        for row in public_source_order(raw,plan):
            if row['request_id'] not in priming_ids:constraints[row['request_id']]=transport_constraints(row,prior.get(row['model']))
            prior[row['model']]=row
        rows={r['request_id']:r for r in raw['request_records']}
        for request in doc['requests']:
            row=rows[request['request_id']];slacks={identity[2]:values for identity,values in indexed.items() if identity[:2]==(row['model'],row['request_id'])}
            vector=request_vector(request,row,fit,slacks,constraints[row['request_id']]);vector['training_block_id']=plan['capture_id']
            vector['vector_id']=hashlib.sha256(json.dumps({k:v for k,v in vector.items() if k!='vector_id'},sort_keys=True).encode()).hexdigest()
            vectors.append(vector);provenance.append(dict(vector_id=vector['vector_id'],block_id=plan['capture_id'],
                attempt_id=plan['attempt_id'],execution_id=plan['execution_id'],request_id=row['request_id'],case_id=row['profile_job_id'],
                raw_sha256=request['macros'][0]['capture_sha256']))
        decompositions.append(dict(attempt_id=plan['attempt_id'],block_id=plan['capture_id'],messages=len(messages),
            priming_messages=sum(len(r['execution_events'])+2 for r in raw['priming_request_records']),
            priming_history_requests=len(raw['priming_request_records']),all_observed_clocks_reconstructed=True,
            stdout_producer_scope='one_persistent_producer_per_owner_per_attempt'))
    return dict(schema=SCHEMA,status='COMPLETE_REQUEST_WORK_PROTOCOL_VECTORS_BUSY_TRANSPORT_UNKNOWN',
        features=list(response.FEATURES),coefficients=fit['coefficients'],vectors=vectors,
        provenance_not_used_for_lookup=provenance,whole_attempt_protocol_decompositions=decompositions,
        blocks=blocks,block_weights=[1/len(blocks)]*len(blocks),heldout_block_id=heldout_block_id,
        measured_request_vectors=len(vectors),authenticated_campaign_startup_attempts=12,
        donor_startup_attempts=len(decompositions),training_replication_units=len(blocks),
        public_lookup_key='owner_task_exact_input_and_public_key_arrival_ordinal',
        common_block_latent_across_inputs=True,conditional_request_draws_independent_given_block_assumption=True,
        complete_request_vector_drawn_jointly=True,per_macro_or_lag_independent_sampling=False,
        source_raw_ids_are_provenance_only=True,protocol_slacks_are_measured_CPU_service=False,
        startup_control_unobserved_collector_work_remains_limitation=True,
        observed_submit_to_START_is_worker_demand=False,busy_transport_law='UNKNOWN_NOT_IDENTIFIED',
        prospective_template_transfer_qualified=False,main_queue_provider_eligible=False,
        final_gain_eligible=False,validation_labels_read=False,GPU_started=False)


def choose_vector(law,public_key,public_key_ordinal,*,block_quantile,within_block_quantile):
    require(law['schema']==SCHEMA and type(public_key_ordinal) is int and public_key_ordinal>=0,'Exact law and public-key ordinal required')
    require(type(public_key) in (tuple,list) and len(public_key)==3,'Public owner/task/exact-input key required')
    for q in (block_quantile,within_block_quantile):require(type(q) in (int,float) and math.isfinite(q) and 0<=q<1,'Explicit quantile in[0,1) required')
    blocks=law['blocks'];block=blocks[min(int(block_quantile*len(blocks)),len(blocks)-1)]
    donors=[v for v in law['vectors'] if v['training_block_id']==block and v['public_key']==list(public_key)]
    if not donors:return dict(status='UNKNOWN_EXACT_PUBLIC_INPUT_IN_SELECTED_WHOLE_BLOCK',public_key=list(public_key),public_key_ordinal=public_key_ordinal)
    vector=copy.deepcopy(donors[min(int(within_block_quantile*len(donors)),len(donors)-1)])
    return dict(status='COMPLETE_REQUEST_JOINT_VECTOR_DRAW',vector=vector,block_id=block,
        public_key_ordinal=public_key_ordinal,block_probability=1/len(blocks),transport_status='UNKNOWN_WITHOUT_EXPLICIT_PAIRED_DRAW')


def compile_draw_v2(vector,public_key_ordinal,transport_support=None):
    if transport_support is None:return dict(status='UNKNOWN_EXPLICIT_COMPLETE_PAIRED_TRANSPORT_DRAW_REQUIRED')
    require(type(public_key_ordinal) is int and public_key_ordinal>=0,'Public-key arrival ordinal required')
    require(transport_support.get('status')=='SUPPLIED_FEASIBLE_LATENT_SCENARIO'
        and transport_support.get('vector_id')==vector['vector_id']
        and transport_support.get('public_key_ordinal')==public_key_ordinal,'Explicit transport must stay paired with complete request vector/public ordinal')
    draw=transport_support.get('draw');require(type(draw) is TransportDraw,'All five explicit transport fields required')
    sender_fifo(0,0,draw);c=vector['transport_constraints']
    for key in ('enqueue_after_arrival_ns','sender_turn_extra_ns','submit_to_write_record_ns'):
        require(getattr(draw,key)==c[key],'Supplied scenario violates observed paired sender components')
    line=draw.submit_to_stdin_line_available_ns;extra=draw.worker_release_to_START_extra_ns
    require(max(Fraction(c['previous_READY_minus_submit_ns']),line)+extra==c['observed_submit_to_START_ns'],
        'Paired latent line/continuation must reconstruct source START exactly')
    owner=vector['public_key'][0];producer=owner+':modeled_stdout_lifetime';nodes=[];timings={}
    for node in vector['nodes']:
        marks=[]
        for mark in node['marks']:
            s=checked_slacks(mark['protocol_slacks']);require(mark['identity'] not in timings,'Unique generated message identity required')
            timings[mark['identity']]=s
            marks.append(Mark(identity=mark['identity'],work_ns=fraction_read(mark['reference_work_at']),processing_delay_ns=Fraction(0),
                action=mark['action'],producer_id=producer,phase=mark['phase'],reader_delay_ns=Fraction(0),
                event_id=owner+':key-slot:'+str(public_key_ordinal)+':'+mark['event_reference_phase_id']+':stream_end' if mark['action']=='query' else None,
                stream_id=owner+':symbolic_recorded_current_stream' if mark['action']=='query' else None,
                query_complete=mark['query_complete'],clear_scope=mark['clear_scope']))
        nodes.append(Macro(node['macro_id'],node['phase'],fraction_read(node['reference_work_ns']),tuple(marks)))
    timings['worker_READY']=checked_slacks(vector['ready_protocol_slacks'])
    plan=RequestPlan(tuple(nodes),Fraction(0),producer,dispatch_to_worker_start_ns=Fraction(0),ready_reader_delay_ns=Fraction(0))
    validate_plan(plan)
    return dict(status='COMPILED_EXPLICIT_CONDITIONAL_TRANSPORT_SCENARIO_NOT_FITTED_BUSY_LAW',plan=plan,
        protocol_slacks=timings,transport_draw=draw,public_key=tuple(vector['public_key']),public_key_ordinal=public_key_ordinal,
        transport_scenario_probability_qualified=False,changed_schedule_transport_transfer_qualified=False)


def run(campaign_path,review_paths,fit_path,output,*,heldout_block_id=None):
    bundle=response.load_complete(campaign_path,review_paths)
    require(bundle.get('status')!=response.CONTAMINATION_HOLD,response.CONTAMINATION_HOLD)
    require(bundle['documents'] is not None,'No partial joint-law compilation')
    ref=response.receipt(fit_path);report=response.checked(ref)
    require(report['schema']==response.SCHEMA and report['numerical_fit_run'] is True and report['validation_labels_read'] is False,'Exact new fitted response report required')
    for src in report['sources']:require(response.receipt(src['path'])==src,'Fit source drift')
    rebuilt=response.fit_new_blocks(bundle['documents'],bundle['plans']);require(rebuilt==report['fit'],'Fitted response must rederive exactly from these complete documents')
    reviews={response.checked(response.checked(response.receipt(p))['inputs']['plan.json'])['attempt_id']:response.checked(response.receipt(p)) for p in review_paths}
    raws=[response.checked(reviews[p['attempt_id']]['raw']) for p in bundle['plans']]
    result=build_joint_law_v2(bundle['documents'],bundle['plans'],raws,rebuilt,heldout_block_id=heldout_block_id)
    result['sources']=strict_references([bundle['sources'],ref,report['sources'],implementation_sources(__file__)])
    require(all(response.receipt(src['path'])==src for src in result['sources']),'Joint law source drift')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with (output/'report.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return response.receipt(output/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',required=True)
    parser.add_argument('--training-reviews',nargs='+',required=True);parser.add_argument('--fit',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--heldout-block',choices=response.intake.campaign_v2.strict.CAPTURES)
    args=parser.parse_args();print(json.dumps(run(args.campaign,args.training_reviews,args.fit,args.output,heldout_block_id=args.heldout_block)))
