"""CPU verification including original protocol-normalized explicit no-answer.

No CUDA imports or calls. Three language owners must have successful probes;
the frozen worker does not instrument OCR with device events. Collection
correctness, exact outputs/actual work, nsys identity and API authenticity need
their independent reviews. This verifier never manufactures a GPU smoke.
"""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import re

from .implementation_source_closure_v27 import implementation_sources,strict_references
from .native_collect import receipt
from .smoke_verifier_v3 import load_cpu_adapter,validate_output
from .unknown_preflight_archive_v1 import save_synced

OWNERS=('generalist_2#0','ocr#0','text#0','text#1')
LANGUAGE=tuple(o for o in OWNERS if o!='ocr#0')
SCHEMA='gsdeploy.stream_event_forward_observation.v1'
FROZEN={'device_progress_observer_v1.py':'7b5e9466824c277e2923eea216bcf4fa3325f52caa6e9c2cf989b784f281d4f0',
        'semantic_progress_worker_v1.py':'53faf12524c44cb495337b55a639292da1160297f3651c5f7adf5aa0630bffb6'}
AVAILABILITY=('received_time_ns','collector_dequeued_time_ns','collector_processing_time_ns')
BASE_NO_ANSWER='original_tasks_exact_no_answer'


def stamp(value):return type(value) is int and value>=0
def finite(value):return type(value) in (int,float) and math.isfinite(value) and value>=0


def valid_prediction(row,policy=None):
    response=row.get('worker_response',{});prediction=response.get('prediction');raw=response.get('raw_output')
    if not isinstance(raw,str) or not raw.strip() or not isinstance(prediction,str):return False
    if prediction.strip():return True
    if row.get('task')!='text' or prediction!='':return False
    if policy in (BASE_NO_ANSWER,'optional_brackets_terminal_punctuation_v2') and raw.strip().casefold() in {'<no answer>','no answer>'}:return True
    return (policy=='optional_brackets_terminal_punctuation_v2'
        and re.fullmatch(r'<?\s*no\s+answer\s*>?[.!]?',raw.strip(),flags=re.I) is not None)


def validate_request(row,ready,output_policy=None):
    issues=[];rid=row.get('request_id');response=row.get('worker_response',{});owner=row.get('model')
    def require(value,code):
        if not value:issues.append(code+':'+str(rid))
        return bool(value)
    begin,end=response.get('start_perf_ns'),response.get('end_perf_ns')
    require(row.get('status')=='completed' and response.get('success') is True and response.get('request_id')==rid,
            'COMPLETED_SUCCESSFUL_REQUEST_REQUIRED')
    require(valid_prediction(row,output_policy),'VALID_ORIGINAL_PROTOCOL_OUTPUT_REQUIRED')
    if not require(stamp(begin) and stamp(end) and begin<end,'HANDLER_BRACKET_REQUIRED'):return issues,dict(request_id=rid)
    received=row.get('response_reader_event_enqueue_time_ns');dequeued=row.get('response_collector_dequeued_time_ns');processed=row.get('response_collector_processing_time_ns')
    require(all(stamp(x) for x in (received,dequeued,processed)) and end<=received<=dequeued<=processed,'RESULT_AVAILABILITY_CHAIN')
    require(row.get('completion_time_ns')==received,'RESULT_COMPLETION_IDENTITY')
    starts=[];observations=[];phase_starts={};phase_pairs={}
    for event in row.get('execution_events',[]):
        require(event.get('request_id')==rid,'EVENT_REQUEST_IDENTITY')
        at=event.get('perf_ns');parent=[event.get(k) for k in AVAILABILITY]
        require(stamp(at) and begin<=at<=end and all(stamp(x) for x in parent)
                and at<=parent[0]<=parent[1]<=parent[2],'EVENT_HOST_PARENT_AVAILABILITY_CHAIN')
        kind=event.get('event')
        if kind is None:
            starts.append(event);continue
        if kind=='observation':
            observations.append(event);continue
        key=event.get('kind'),event.get('segment_id')
        if kind=='start':
            require(key not in phase_starts and key not in phase_pairs,'DUPLICATE_PHASE_START');phase_starts[key]=at
        elif kind=='end':
            lo=phase_starts.pop(key,None)
            require(stamp(lo) and stamp(at) and lo<=at,'UNPAIRED_PHASE_END');phase_pairs[key]=(lo,at)
        else:require(False,'UNRECOGNIZED_EVENT_EDGE')
    require(len(starts)==1 and starts[0].get('pid')==ready.get('pid') and starts[0].get('perf_ns')==begin,'EXACT_WORKER_START')
    require(not phase_starts,'UNFINISHED_PHASE_BOUNDARIES')
    nxt=row.get('previous_handler_next_request_ready',{});times=[nxt.get(k) for k in ('perf_ns',*AVAILABILITY)]
    require(nxt.get('request_id')==rid and nxt.get('pid')==ready.get('pid') and nxt.get('clock')=='host_perf_counter_ns'
            and nxt.get('boundary')=='previous_handler_next_request_ready' and all(stamp(x) for x in times)
            and end<=times[0]<=times[1]<=times[2]<=times[3],'EXACT_WORKER_NEXT_READY')
    if owner=='ocr#0':
        require(not observations,'OCR_FROZEN_WORKER_HAS_NO_DEVICE_OBSERVATIONS')
        return issues,dict(request_id=rid,owner=owner,device_probe='UNKNOWN_NOT_INSTRUMENTED',raw_output_accepted_as_structure_only=True)
    document=response.get('semantic_forward_observations',{});progress=document.get('device_progress',{})
    require(document.get('schema')=='gsdeploy.semantic_forward_observation.v1' and document.get('request_id')==rid,'SEMANTIC_DOCUMENT_IDENTITY')
    require(progress.get('schema')==SCHEMA and progress.get('cuda_synchronization_added') is False
            and progress.get('unfinished_fraction_identified') is False and progress.get('completion_is_single_recorded_stream_only') is True
            and progress.get('physical_SM_service_identified') is False and progress.get('progress_model_qualified') is False,'EXPLICIT_LIMITED_DEVICE_SCOPE')
    errors=copy.deepcopy(document.get('instrumentation_errors',[]))
    # A real smoke is intended to exercise the probes: errors remain in the
    # report and fail this probe, rather than being hidden as ordinary UNKNOWN.
    require(isinstance(errors,list) and not errors,'PROBE_INSTRUMENTATION_ERROR_PRESERVED')
    phases=document.get('records',[]);records=progress.get('records',[]);queries=progress.get('queries',[])
    require(bool(phases) and len(records)==len(phases),'EACH_FORWARD_REQUIRES_DEVICE_RECORD')
    phases_by_id={p.get('id'):p for p in phases};records_by_id={p.get('phase_id'):p for p in records}
    require(len(phases_by_id)==len(phases) and len(records_by_id)==len(records) and set(phases_by_id)==set(records_by_id),'EXACT_FORWARD_DEVICE_COVERAGE')
    brackets=progress.get('nvtx_host_brackets',[]);brackets_by_name={p.get('name'):p for p in brackets}
    require(len(brackets)==len(phases) and len(brackets_by_name)==len(brackets),'EACH_FORWARD_REQUIRES_NVTX_BRACKET')
    for index,phase in enumerate(phases):
        require(type(phase.get('forward_index')) is int and phase.get('forward_index')==index
                and phase.get('semantic_phase')==('prefill' if index==0 else 'decode'),'FORWARD_ORDINAL_PHASE')
        require(all(type(phase.get(k)) is int and phase[k]>0 for k in ('input_sequence_length','input_batch_size')),'BEGIN_TENSOR_SHAPE_REQUIRED')
        require(phase.get('nvtx_name')=='GS_STAGE:'+str(rid)+':gpu:'+str(phase.get('id')),'EXACT_NVTX_FORWARD_NAME')
        start,finish=phase.get('start_ns'),phase.get('end_ns');ident=phase.get('id');record=records_by_id.get(ident,{})
        require(stamp(start) and stamp(finish) and begin<=start<=finish<=end,'FORWARD_HOST_BRACKET')
        require(phase_pairs.get(('gpu',ident))==(start,finish),'FORWARD_PHASE_TRANSPORT_PAIR')
        require(record.get('event_id')==str(rid)+':'+str(ident)+':stream_end' and record.get('forward_index')==index
                and stamp(record.get('stream_id')) and stamp(record.get('device_index')),'EVENT_STREAM_DEVICE_IDENTITY')
        require(record.get('declared_stream_scope')=='caller_current_stream_at_forward_BEGIN'
                and record.get('other_stream_dependencies_certified') is False and record.get('forward_success_certified_by_probe') is False,'RECORD_SCOPE')
        require('observation_error' not in record,'PROBE_RECORD_ERROR_PRESERVED')
        a=record.get('begin_record_host_bracket_ns',[]);b=record.get('end_record_host_bracket_ns',[])
        require(isinstance(a,list) and isinstance(b,list) and len(a)==len(b)==2 and all(stamp(x) for x in a+b)
                and stamp(start) and stamp(finish) and start<=a[0]<=a[1]<=b[0]<=b[1]<=finish,'EVENT_RECORD_HOST_BRACKETS')
        bracket=brackets_by_name.get(phase.get('nvtx_name'),{});times=[bracket.get(k) for k in ('push_before_ns','push_after_ns','pop_before_ns','pop_after_ns')]
        require(all(stamp(x) for x in times) and stamp(start) and stamp(finish)
                and begin<=times[0]<=times[1]<=start<=finish<=times[2]<=times[3]<=end,'DUAL_SIDED_NVTX_HOST_BRACKETS')
    require([q.get('observation_sequence') for q in queries]==list(range(1,len(queries)+1)),'EXACT_QUERY_SEQUENCE')
    stripped=[{k:v for k,v in event.items() if k not in AVAILABILITY} for event in observations]
    require(stripped==queries,'EXACT_QUERIES_RETAINED_IN_PARENT_TRANSPORT')
    completed=set();last_query_end=-1;query_counts=Counter()
    for query in queries:
        require({'in_flight_remaining_fraction','completion_time_ns','physical_SM_service_ns','recorded_stream_elapsed_ms',
                 'all_request_GPU_work_complete','host_query_begin_ns','host_query_end_ns'}<=set(query),'EXPLICIT_QUERY_UNKNOWN_FIELDS_REQUIRED')
        phase_id=query.get('phase_id');record=records_by_id.get(phase_id,{})
        require(phase_id in records_by_id and query.get('segment_id')==phase_id and query.get('schema')==SCHEMA
                and query.get('request_id')==rid and query.get('event')=='observation' and query.get('kind')=='device_event_query','QUERY_IDENTITY')
        require(all(query.get(k)==record.get(k) for k in ('event_id','stream_id','device_index','forward_index')),'QUERY_RECORD_BINDING')
        lo,hi=query.get('host_query_begin_ns'),query.get('host_query_end_ns');record_end=record.get('end_record_host_bracket_ns',[])
        require(stamp(lo) and stamp(hi) and lo<=hi and lo>=last_query_end and query.get('perf_ns')==hi
                and len(record_end)==2 and stamp(record_end[1]) and record_end[1]<=lo,'QUERY_HOST_BRACKET_ORDER')
        if stamp(hi):last_query_end=hi
        require(query.get('all_request_GPU_work_complete') is False and query.get('evidence_scope')=='recorded_current_stream_only'
                and all(query.get(k) is None for k in ('in_flight_remaining_fraction','completion_time_ns','physical_SM_service_ns')),'NO_GLOBAL_OR_FRACTIONAL_COMPLETION_CLAIM')
        done=query.get('end_event_complete');require(type(done) is bool and phase_id not in completed,'COMPLETION_BOOLEAN_AND_NO_REQUERY')
        query_counts[phase_id]+=1
        if done is True:
            require(finite(query.get('recorded_stream_elapsed_ms')) and record.get('recorded_stream_elapsed_ms')==query.get('recorded_stream_elapsed_ms')
                    and record.get('completion_query_bracket_ns')==[lo,hi],'COMPLETED_INTERVAL_QUERY_BINDING');completed.add(phase_id)
        else:require(query.get('recorded_stream_elapsed_ms') is None,'UNFINISHED_ELAPSED_UNKNOWN')
    require(all(query_counts[p]>0 for p in phases_by_id),'EACH_RECORDED_FORWARD_REQUIRES_QUERY')
    require(completed==set(phases_by_id) and progress.get('pending_event_ids')==[],'ALL_PROBES_COMPLETE_AFTER_EXISTING_SYNC')
    return issues,dict(request_id=rid,owner=owner,forward_records=len(phases),queries=len(queries),completed_recorded_stream_events=len(completed),
        errors_preserved=errors,record_errors_preserved=[copy.deepcopy(r) for r in records if 'observation_error' in r],
        device_probe='FAIL' if issues else 'RAW_REPORTED_RECORDED_STREAM_EVENTS_COMPLETE',all_request_GPU_work_complete=False)


def validate_raw(raw,output_policies=None):
    issues=[];ready=raw.get('ready',{});rows=raw.get('request_records',[])
    if raw.get('status')!='completed' or any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors')):issues.append('COMPLETED_CLEAN_RAW_REQUIRED')
    if set(ready)!=set(OWNERS) or len({r.get('pid') for r in ready.values()})!=4:issues.append('FOUR_DISTINCT_OWNER_READY_REQUIRED')
    for owner,value in ready.items():
        if not stamp(value.get('pid')) or not isinstance(value.get('torch'),str) or not value['torch'] or value.get('clock')!='time.perf_counter_ns':issues.append('READY_TORCH_PID_CLOCK_REQUIRED:'+owner)
    if set(r.get('model') for r in rows)!=set(OWNERS) or len({r.get('request_id') for r in rows})!=len(rows):issues.append('FOUR_OWNER_UNIQUE_REQUEST_COVERAGE_REQUIRED')
    reports=[]
    for row in rows:
        try:
            found,result=validate_request(row,ready.get(row.get('model'),{}),(output_policies or {}).get(row.get('model')));issues.extend(found);reports.append(result)
        except (KeyError,TypeError,ValueError,IndexError) as exc:
            issues.append('MALFORMED_REQUEST:'+str(row.get('request_id'))+':'+type(exc).__name__)
    shutdown=raw.get('shutdown',{})
    if set(shutdown)!=set(OWNERS) or any(r.get('forced') or r.get('exit_code')!=0 or r.get('bye_received') is not True for r in shutdown.values()):issues.append('FOUR_CLEAN_WORKER_EXITS_REQUIRED')
    return dict(status='FAIL' if issues else 'RAW_STRUCTURAL_CONTRACT_ACCEPTED_DEVICE_API_NOT_AUTHENTICATED',issues=issues,
        requests=reports,expected_device_event_owners=LANGUAGE,uninstrumented_owner=dict(owner='ocr#0',status='UNKNOWN_NOT_INSTRUMENTED'))


def checked(reference):
    if (not isinstance(reference,dict) or set(reference)!={'path','sha256'}
            or receipt(reference['path'])!=reference):
        raise ValueError('Exact raw plan/context references required')
    value=json.loads(Path(reference['path']).read_text())
    if receipt(reference['path'])!=reference:raise ValueError('Referenced role document changed during read')
    return value


def qualify_roles(raw):
    """Owned collector stores roles in checked plan/jobs and measured rows.

    Qualify these roles before looking at response/probe bodies. A missing
    top-level raw split is the actual collector schema, not an inferred role.
    """
    plan=checked(raw.get('plan'));context=checked(raw.get('context'))
    split,mode=plan.get('split'),plan.get('mode')
    if (plan.get('schema')!='gsdeploy.paired_profile_campaign.v4' or split not in ('smoke','training')
            or mode not in ('natural','kernel') or plan.get('progress_observer') is not True
            or plan.get('mode_switch_at_execute') is not False
            or plan.get('final_gain_eligible') is not False):
        raise ValueError('Explicit non-validation paired-v4 observer plan schema/mode/split required')
    if (raw.get('schema')!='gsdeploy.native_phase_collection.v1' or raw.get('kind')!='measured'
            or raw.get('backend')!='native' or raw.get('split',split)!=split
            or raw.get('not_for_training') or raw.get('validation_labels_used')
            or raw.get('performance_scope')!=('profiler_perturbed' if mode=='kernel' else 'unprofiled_with_phase_markers')
            or raw.get('execution_context_sha256')!=plan.get('execution_context_sha256')
            or not isinstance(plan.get('execution_context_sha256'),str)):
        raise ValueError('Raw collection schema/mode/optional top-level role conflicts with checked plan')
    if (context.get('paired_profile_mode')!=mode or context.get('protocol')!='paired_profile_v4_'+mode
            or context.get('worker_variant')!='semantic_progress_worker_v1.py'):
        raise ValueError('Checked context observer/mode role conflicts with paired plan')
    jobs=plan.get('jobs',[]);rows=raw.get('request_records',[])
    expected_view='correctness_clock_only_not_fit' if split=='smoke' else 'training'
    if (not jobs or not rows or any(job.get('split')!=split or job.get('validation_view')!=expected_view for job in jobs)
            or any(row.get('split')!=split for row in rows)):
        raise ValueError('Every declared job and measured row must have exact checked non-validation split')
    by_job={job.get('id'):job for job in jobs}
    if len(by_job)!=len(jobs) or None in by_job:raise ValueError('Unique declared role-bearing job identities required')
    for row in rows:
        job=by_job.get(row.get('profile_job_id'))
        if job is None or row.get('validation_view')!=job.get('validation_view'):
            raise ValueError('Measured row job/validation-view role must equal checked plan')
    return dict(plan_schema=plan['schema'],split=split,mode=mode,
        role_source='exact_receipted_plan_jobs_and_all_measured_row_splits',
        raw_top_level_split_present='split' in raw,validation_roles_forbidden=True)


def reparse_outputs(raw):
    """Rebuild original CPU requests and finish without consulting task labels."""
    context=checked(raw['context']);sources=[];issues=[];policies={};summaries=[]
    snapshots=context.get('execution_sources',[])
    for member in snapshots:
        ref=member['frozen'];checked_bytes=receipt(ref['path'])
        if checked_bytes!=ref or receipt(member['original_path'])['sha256']!=ref['sha256']:
            raise ValueError('Frozen executable/input snapshot differs from actual source')
        sources.extend([ref,receipt(member['original_path'])])
    pool_entries=[r for r in snapshots if r['original_path'].endswith('/runtime-inputs.jsonl')]
    if len(pool_entries)!=1:raise ValueError('One exact frozen original input pool required')
    pool={row['pool_id']:row for row in map(json.loads,Path(pool_entries[0]['frozen']['path']).read_text().splitlines())}
    for owner in OWNERS:
        family='generalist_2' if owner=='generalist_2#0' else 'ocr' if owner=='ocr#0' else 'text'
        binding=context['worker_bindings'][family];spec_ref=receipt(binding['spec']);sources.append(spec_ref)
        if spec_ref['sha256']!=binding['spec_sha256'] or raw['ready'][owner]['spec_sha256']!=spec_ref['sha256']:
            raise ValueError('Actual READY worker spec does not match frozen binding')
        spec=json.loads(Path(binding['spec']).read_text());adapter=None if family=='ocr' else load_cpu_adapter(binding,spec)
        if adapter is not None:
            policies[owner]=adapter.policy['tasks']['text'].get('no_answer_normalization') or BASE_NO_ANSWER
            source_root=Path(binding['command'][binding['command'].index('--source-root')+1])
            sources.extend(receipt(path) for path in (source_root/'gsdeploy/backends/compact.py',source_root/'gsdeploy/tasks.py'))
        for row in raw.get('priming_request_records',[])+raw['request_records']:
            if row.get('model')!=owner:continue
            item=pool.get(row.get('pool_id'))
            if item is None:raise ValueError('Original exact input pool request required')
            found=validate_output(row,item,spec,adapter);issues.extend(found)
            summaries.append(dict(owner=owner,request_id=row['request_id'],sample_id=row['worker_response'].get('sample_id'),
                protocol_sha256=row['worker_response'].get('protocol_sha256'),request_sha256=row['worker_response'].get('request_sha256'),
                explicit_no_answer_normalized=(row['worker_response'].get('prediction')=='' and valid_prediction(row,policies.get(owner))),
                original_CPU_reparse_passed=not found))
    return dict(issues=issues,policies=policies,requests=summaries,sources=strict_references(sources),
        task_quality_evaluated=False,raw_outputs_modified=False)


def run(raw_path,output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    root=Path(__file__).parent;frozen=[receipt(root/name) for name in FROZEN]
    if any(r['sha256']!=FROZEN[Path(r['path']).name] for r in frozen):raise ValueError('Reviewed observer/worker bytes changed')
    raw_ref=receipt(raw_path);raw=json.loads(Path(raw_path).read_text())
    # Reject validation roles before examining request response/probe bodies.
    role=qualify_roles(raw)
    refs=[raw_ref,*frozen]
    for name in ('plan','context'):
        ref=raw.get(name)
        if not isinstance(ref,dict) or receipt(ref['path'])!=ref:raise ValueError('Exact raw plan/context references required')
        refs.append(ref)
    output_review=reparse_outputs(raw)
    result=validate_raw(raw,output_review['policies']);result['issues'].extend(output_review['issues'])
    if result['issues']:result['status']='FAIL'
    result.update(schema='gsdeploy.device_progress_smoke_CPU_raw_contract_review.v2',raw=raw_ref,
        output_protocol_review=output_review,
        verifier_role_binding_revision=2,qualified_role=role,
        reviewed_frozen_sources=frozen,sources=strict_references([refs,output_review['sources'],implementation_sources(__file__)]),
        GPU_started_by_verifier=False,validation_labels_read=False,device_API_executed_by_verifier=False,
        limitations=['JSON self-reports do not authenticate actual torch/CUDA Event execution. Raw provenance, exact worker launch/source binding and device identity need independent execution evidence.',
            'Original source-bound CPU output parsing checks request/protocol/generation identity including explicit no-answer. Task quality and paired natural/kernel equality remain separate.',
            'The frozen worker probes three language owners only. OCR READY/output/exit are checked, while OCR device progress remains uninstrumented.',
            'A successful query certifies only work captured on its recorded stream before query return; other streams, complete request readiness, device completion timestamp and SM fraction remain unknown.',
            'PHASE receipt chain demonstrates recorded observer availability on the host; it does not establish zero transport delay, bounded native clock drift or exact NVTX/CUDA clock alignment.',
            'Observer overhead, paired performance perturbation and native event association are not tested by this CPU verifier.'])
    if receipt(raw_path)!=raw_ref:raise ValueError('Raw changed during review')
    for ref in result['sources']:
        if receipt(ref['path'])!=ref:raise ValueError('Review source changed')
    out.mkdir(parents=True,exist_ok=False);save_synced(out/'report.json',result);return receipt(out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.raw,args.output)))
