"""CPU-only preparation and injected dispatch of original-runtime FIFO cohorts.

No worker startup, GPU import, execute CLI, profiler or model mutation exists.
The injected dispatcher can be called by a future separately authorized owned
collector after startup/priming; it uses runtime.enqueue/drain unchanged.
"""
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
from pathlib import Path

from .stage_response_features_v1 import receipt
from .implementation_source_closure_v27 import implementation_sources,strict_references

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='gsdeploy.original_twoFIFO_training_manifest.v1'
OWNERS=('text#0','text#1','generalist_2#0','ocr#0')
MEASUREMENT_NS=10_000_000_000
DRAIN_NS=60_000_000_000


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def collection_fields():
    return dict(schema='gsdeploy.original_twoFIFO_collection_fields.v1',clock='host_perf_counter_ns',
        retain_existing=['scheduled_time_ns','enqueued_time_ns','actual_submit_time_ns',
            'submit_write_completed_time_ns','observed_worker_start_time_ns','worker_response.start_perf_ns',
            'worker_response.end_perf_ns','previous_handler_next_request_ready.perf_ns',
            'previous_handler_next_request_ready.received_time_ns','previous_handler_next_request_ready.collector_dequeued_time_ns',
            'previous_handler_next_request_ready.collector_processing_time_ns','response_reader_event_enqueue_time_ns',
            'response_collector_dequeued_time_ns','response_collector_processing_time_ns','all_original_execution_events'],
        parent_extension=['sender_dequeue_ns','serialize_begin_ns','serialize_end_ns','stdin_write_begin_ns',
            'stdin_write_return_ns','stdin_flush_begin_ns','stdin_flush_return_ns','write_record_lock_acquired_ns',
            'serialized_line_byte_length','serialized_line_sha256','actual_text_encoding','sender_public_owner_ordinal'],
        worker_extension=['worker_pid_start_identity','worker_stdin_read_ordinal','stdin_read_enter_ns',
            'stdin_line_read_return_ns','parsed_request_id','parse_begin_ns','parse_end_ns','task_checks_end_ns',
            'peak_memory_reset_begin_ns','peak_memory_reset_end_ns','READY_emit_begin_ns','READY_flush_return_ns'],
        parent_notification_extension=['stdout_line_read_return_ns','protocol_parse_end_ns','event_queue_put_enter_ns',
            'event_queue_put_return_ns','collector_dequeue_ns','collector_lock_acquired_ns'],
        optional_public_pipe_metadata=['stdin_pipe_capacity_bytes_via_F_GETPIPE_SZ_or_UNKNOWN',
            'pipe_identifier_and_fd_lifetime','userspace_text_buffer_configuration'],
        resource_boundaries=['startup_begin_ns','startup_all_residents_ready_ns','priming_begin_ns','priming_end_ns',
            'case_epoch_ns','measurement_end_ns','fixed_drain_deadline_ns','actual_finalization_ns',
            'last_owned_sample_before_worker_close_ns','worker_close_begin_ns','cleanup_end_ns'],
        resource_samples=['query_begin_ns','query_end_ns','sample_midpoint_ns','board_power_watts',
            'owned_used_mib','owned_pid_start_identity_and_ancestry','all_resident_aliases','foreign_or_unknown_owner_pids',
            'read_errors_and_missingness','measurement_or_drain_or_startup_scope'],
        marking_policy='New instrumented worker/transport variant only; preserve frozen originals and record overhead/domain separately.',
        exact_pipe_line_availability_identified_even_with_these_markers=False,
        CPU_service_vs_scheduling_preemption_identified=False,hardware_counter_permissions_assumed=False)


def cases(inputs):
    definitions=[]
    def item(owner,length,at):return dict(owner=owner,task='ocr' if owner=='ocr#0' else 'text',
        pool_id=inputs['ocr' if owner=='ocr#0' else 'text'][length]['pool_id'],arrival_ns=at)
    def add(name,aim,rows):definitions.append(dict(case_id=name,intended_stratum=aim,arrivals=rows))
    for owner in OWNERS:
        name=owner.replace('#','-')
        add('spaced-'+name,'spaced_short_medium_long_reference_idle_not_guaranteed',
            [item(owner,size,i*3_000_000_000) for i,size in enumerate(('short','medium','long'))])
        add('HOL-'+name,'same_owner_long_short_and_reverse_order_head_of_line',
            [item(owner,size,at) for size,at in [('long',0),('short',1_000_000),('medium',2_000_000),
                ('short',3_000_000_000),('long',3_001_000_000),('medium',3_002_000_000)]])
    for owner in ('text#0','generalist_2#0'):
        name=owner.replace('#','-')
        add('worker-busy-'+name,'target_worker_busy_sender_completed_before_prior_READY',
            [item(owner,'long',0),item(owner,'short',50_000_000),item(owner,'short',100_000_000)])
        add('sender-pressure-'+name,'target_sender_busy_and_pipe_pressure_not_certified_by_payload_size',
            [item(owner,'long',0) for _ in range(16)]+[item(owner,'short',1_000_000),item(owner,'medium',2_000_000)])
    # Controlled phase-offset attempts: measured phase identities, not these
    # offsets or input lengths, decide whether a phase stratum was achieved.
    for peer_count,offset,name in [(1,0,'one-peer-simultaneous'),(1,300_000_000,'one-peer-staggered'),
                                  (2,50_000_000,'two-peers-staggered'),(3,300_000_000,'three-peers-staggered')]:
        rows=[item('text#0','long',0),item('text#0','short',1_000_000)]
        for i,owner in enumerate(OWNERS[1:peer_count+1]):
            rows.extend([item(owner,'long',offset+i*50_000_000),item(owner,'short',offset+i*50_000_000+1_000_000)])
        add(name,'mixed_owner_phase_overlap_intent_requires_observed_phase_coverage',rows)
    return definitions


def validate(manifest,*,verify_sources=True):
    require(manifest.get('schema')==SCHEMA and manifest.get('split')=='training','Training-only FIFO manifest required')
    require(manifest['measurement_ns']==MEASUREMENT_NS and manifest['drain_ns']==DRAIN_NS,
            'Fixed measurement and drain boundaries required')
    require(manifest['resident_owners']==list(OWNERS) and manifest['max_pending_per_model']>=18,
            'Original four-owner roster and enough host queue capacity required')
    require(manifest['runtime_policy']==dict(admission=None,batching=False,weight_reuse=False,
        enqueue_all_without_waiting_for_RESULT_or_READY=True),'Original FIFO policy required')
    require(len(manifest['capture_blocks'])==3,'Three planned training capture blocks required')
    ids=set();counts=Counter();sample_ids={r['pool_id']:r for group in manifest['inputs'].values() for r in group.values()}
    require(len(sample_ids)==6,'Three exact public text inputs and three OCR inputs required')
    for block in manifest['capture_blocks']:
        require(len(block['cases'])==16,'Complete fixed sixteen-case matrix required')
        for case in block['cases']:
            rows=case['arrivals'];require(rows and len(rows)==case['offered'],'Every offered row must remain indexed')
            for row in rows:
                require(set(row)=={'request_id','owner','task','pool_id','arrival_ns','arrival_index'},'Complete public arrival row required')
                require(row['request_id'] not in ids and row['owner'] in OWNERS and row['pool_id'] in sample_ids,
                        'Distinct source-bound public arrivals required')
                require(type(row['arrival_ns']) is int and 0<=row['arrival_ns']<MEASUREMENT_NS,'Arrival outside fixed window')
                require(sample_ids[row['pool_id']]['task']==row['task'] and (row['owner']=='ocr#0')==(row['task']=='ocr'),
                        'Owner/task/input compatibility required')
                ids.add(row['request_id']);counts[block['capture_id']]+=1
            require([r['arrival_index'] for r in rows]==list(range(len(rows))),'Complete offered ordinal sequence required')
            require(case['public_arrivals_sha256']==digest(rows),'Public arrival identity changed')
    if verify_sources:
        require(all(receipt(r['path'])==r for r in manifest['sources']),'Manifest source changed')
        require(receipt(manifest['prepared_pool']['path'])==manifest['prepared_pool'],'Prepared pool changed')
    return dict(planned_capture_blocks=3,case_attempts=48,total_offered=len(ids),offered_per_capture=dict(counts),
        minimum_same_owner_burst=16,actual_sender_busy_coverage='UNKNOWN_UNTIL_MEASURED',
        actual_worker_busy_coverage='UNKNOWN_UNTIL_MEASURED',actual_phase_overlap_coverage='UNKNOWN_UNTIL_MEASURED',
        hardware_counters_required=False,GPU_started=False)


def prepare(output,context_path,pool_path):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    context_ref=receipt(context_path);context=json.loads(Path(context_path).read_text())
    declared=context['declared_inputs'];inputs={};wanted=set()
    for task in ('text','ocr'):
        choices=sorted((row for row in declared if row['task']==task),key=lambda row:row['input_size'])
        selected=[choices[0],choices[1],choices[-1]]
        inputs[task]={size:dict(row,task=task,size_measure='text_input_characters' if task=='text' else 'source_image_bytes')
            for size,row in zip(('short','medium','long'),selected)}
        wanted.update(row['pool_id'] for row in selected)
    selected=[]
    # Only parse lines containing the already-declared training identities.
    for line in Path(pool_path).open():
        if any(pool_id in line for pool_id in wanted):
            row=json.loads(line)
            if row['pool_id'] in wanted:selected.append(row)
    require({row['pool_id'] for row in selected}==wanted,'Exact declared training input bytes required')
    declarations={row['pool_id']:row for group in inputs.values() for row in group.values()}
    media=[]
    for row in selected:
        sample=row['sample'];declared_row=declarations[row['pool_id']]
        if row['task_type']=='text':
            exact=hashlib.sha256(json.dumps({k:sample[k] for k in ('context','question')},
                sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            size=len(sample['context'])+len(sample['question'])
        else:
            media_ref=receipt(sample['input']);media.append(media_ref)
            exact=media_ref['sha256'];size=Path(sample['input']).stat().st_size
        require(exact==declared_row['exact_input_sha256'] and size==declared_row['input_size'],
                'Actual training content/size differs from frozen declared input')
        declared_row['sample_JSON_UTF8_bytes']=len(json.dumps(sample,ensure_ascii=False).encode())
        declared_row['wire_byte_measure']='Conditional UTF-8 text encoding; actual full envelope and encoding must be recorded at future send.'
    pool_file=output/'runtime-inputs.jsonl'
    with pool_file.open('x') as stream:
        for row in selected:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    definitions=cases(inputs);blocks=[]
    for repeat in range(3):
        capture_id='training-block-'+str(repeat+1);bound=[]
        # Rotate the fixed case order without changing public arrivals or inputs.
        ordered=definitions[repeat:]+definitions[:repeat]
        for definition in ordered:
            case=copy.deepcopy(definition)
            for i,row in enumerate(case['arrivals']):
                row.update(request_id=capture_id+':'+case['case_id']+':'+str(i),arrival_index=i)
            case.update(offered=len(case['arrivals']),public_arrivals_sha256=digest(case['arrivals']))
            bound.append(case)
        blocks.append(dict(capture_id=capture_id,cases=bound))
    refs=[context_ref,receipt(pool_path),*media,*context['sources'],*implementation_sources(__file__)]
    sources=strict_references(refs)
    manifest=dict(schema=SCHEMA,status='CPU_PREPARED_NOT_COLLECTED',split='training',
        resident_owners=list(OWNERS),inputs=inputs,prepared_pool=receipt(pool_file),sources=sources,
        measurement_ns=MEASUREMENT_NS,drain_ns=DRAIN_NS,max_pending_per_model=64,capture_blocks=blocks,
        runtime_policy=dict(admission=None,batching=False,weight_reuse=False,enqueue_all_without_waiting_for_RESULT_or_READY=True),
        input_generation_configuration='Unchanged frozen owner bindings, tokenizer and generation limits; no sample padding or new generation cap.',
        initial_case_policy='Future supervisor must bind startup/priming and prior actual READY/notification history; no unrecorded empty-initial assumption.',
        prior_case_incomplete_policy='Stop subsequent unstarted cases; retain their NOT_STARTED status and every offered row of the started case.',
        attempt_policy='No automatic retries; three planned blocks are training repetitions, not new validation labels.',
        collector_compatibility='Frozen profile_owned_collect_v3 owner-keyed job cannot express repeated owner; use dispatch_case with an already-started original DeploymentRuntime.',
        supervisor_entry='CPU prepare/check only; future owned lease/startup/monitor/cleanup supervisor is required and not invoked here.',
        instrumentation_extension_required_for_exact_line_decomposition=True,
        planned_phase_targets=['OCR inference','generalist_2 prefill','text#0/text#1 prefill','generalist_2 final decode'],
        structurally_unavailable_phase_strata=['same_phase_at_OCR_inference_with_only_one_OCR_owner',
            'same_phase_at_READY_tail_after_visible_phase_CLEAR'],
        phase_design_scope='Target visible peer-count freedoms with real phase evidence; no claim all family-phase-same interactions are estimable.',
        resource_design=dict(
            phase_fullstates_to_exhaustively_cover=False,existing_exact_phase_input_fullstates=144,
            existing_power_states_with_two_capture_support=48,existing_memory_states_with_two_capture_support=58,
            short_entry_or_tail_below_sampling_period='UNKNOWN_NOT_ZERO',
            candidate_pooling_key='Complete resident roster + actual START-to-READY owner/input multiset on one global board clock',
            pooling_is_explicit_hypothesis_not_qualified_provider=True,
            matched_phase_composition_cases=['one-peer-simultaneous','one-peer-staggered'],
            matched_case_meaning='Same four residents and same offered owner/input multiset; offsets 0 vs 300ms change possible phase composition.',
            scoring='Leave whole capture block out; compare coarse pooling residuals versus observed phase composition; no state deletion.',
            power_scope='Integrate board power on fixed ten-second measurement only; exclude drain; startup energy separate.',
            memory_scope='Owned sampled peak startup through fixed drain before worker close, including inactive residents.',
            missing_state_or_sample_policy='UNKNOWN with sample bracket/query-duration metadata; never zero-fill or claim instantaneous peak.'),
        collection_field_contract=collection_fields(),
        hardware_counters_required=False,native_CUDA_API_numeric_authentication='UNQUALIFIED',
        validation_labels_read=False,GPU_started=False,main_queue_provider_eligible=False)
    audit=validate(manifest)
    for name,data in [('manifest',manifest),('collection-fields',collection_fields()),('validation',audit)]:
        with (output/(name+'.json')).open('x') as stream:json.dump(data,stream,ensure_ascii=False,indent=2);stream.write('\n')
    return receipt(output/'manifest.json')


def dispatch_case(manifest,case,runtime,pool,*,epoch_ns,clock_ns,wait_until_ns,checkpoint):
    """Inject into an already owned, started, primed runtime; no startup here.

    wait_until_ns must observe the supplied monotonic clock. All offered rows
    are checkpointed before any enqueue. Fixed drain never extends from a late
    dispatch or an early RESULT, and completed rows are retained until cutoff.
    """
    require(type(epoch_ns) is int and clock_ns()<=epoch_ns,'Explicit future/current epoch required')
    require(case in [c for block in manifest['capture_blocks'] for c in block['cases']]
            and case['public_arrivals_sha256']==digest(case['arrivals']),
            'Exact complete prepared case required')
    rows=[]
    for arrival in case['arrivals']:
        item=pool[arrival['pool_id']]
        rows.append(dict(request_id=arrival['request_id'],phase='profile',arrival_index=arrival['arrival_index'],
            task=arrival['task'],model=arrival['owner'],pool_id=arrival['pool_id'],sample_id=item['sample'].get('id'),
            scheduled_time_ns=epoch_ns+arrival['arrival_ns'],actual_submit_time_ns=None,completion_time_ns=None,
            worker_latency_ms=None,worker_queue_wait_ms=None,status='planned'))
    end=epoch_ns+manifest['measurement_ns'];deadline=end+manifest['drain_ns']
    raw=dict(schema='gsdeploy.original_twoFIFO_injected_case.v1',case_id=case['case_id'],offered=len(rows),
        public_arrivals_sha256=case['public_arrivals_sha256'],request_records=rows,
        observation_start_ns=epoch_ns,measurement_end_ns=end,fixed_drain_deadline_ns=deadline,
        all_offered_registered_before_dispatch=True,errors=[])
    checkpoint(raw)
    for row in sorted(rows,key=lambda r:(r['scheduled_time_ns'],r['arrival_index'])):
        wait_until_ns(row['scheduled_time_ns']);row['arrival_observed_time_ns']=clock_ns()
        if clock_ns()>=deadline:
            row.update(status='incomplete',error='arrival dispatcher missed fixed drain; offered row retained')
        else:
            try:runtime.enqueue(row,pool[row['pool_id']]['sample'])
            except Exception as exc:row.update(status='error',error=type(exc).__name__+': '+str(exc))
        checkpoint(raw)
    wait_until_ns(end)
    runtime.drain(rows,max(0,deadline-clock_ns())/1e9)
    wait_until_ns(deadline)
    for row in rows:
        if row['status'] in ('planned','queued','submitted'):
            row.update(status='incomplete',error='fixed drain reached; offered row retained')
        row['RESULT_reader_within_fixed_drain']=(row.get('completion_time_ns') is not None
            and row['scheduled_time_ns']<=row['completion_time_ns']<=deadline)
    raw.update(observation_finalized_ns=clock_ns(),fixed_cutoff_overshoot_ns=max(0,clock_ns()-deadline))
    checkpoint(raw);return raw


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='op',required=True)
    cmd=commands.add_parser('prepare')
    for key in ('output','context','pool'):cmd.add_argument('--'+key,required=True)
    cmd=commands.add_parser('check');cmd.add_argument('--manifest',required=True)
    args=parser.parse_args()
    if args.op=='prepare':result=prepare(args.output,args.context,args.pool)
    else:result=validate(json.loads(Path(args.manifest).read_text()))
    print(json.dumps(result,ensure_ascii=False))
