"""Independent native4 validation review; labels never become fitting inputs.

Explicit validation derivative of the resource cohort audit, retaining partial
outcomes and all199 original request payloads. Frozen training/native modules
are imported only through role-neutral helpers, never relabelled or patched.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from fractions import Fraction as F
from gsdeploy.quality import sample_totals
from gsdeploy.tasks import page_ocr_text,score_result
from gsdeploy.prediction import components
from . import twoFIFO_fresh181_native_v4 as executor
from . import twoFIFO_matching_prospective_bridge_v4 as typed
from . import twoFIFO_fresh181_validation_metrics_v1 as metrics
from . import twoFIFO_public181_expected_score_v2 as expected
from . import twoFIFO_matching_solo_reference_v1 as solo
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard
from .twoFIFO_public181_partial_protocol_v1 import inspect_protocol
from .twoFIFO_public181_resource_review_v1 import resource_facts
from .twoFIFO_native_quality_labels_v1 import prepared_references
from .twoFIFO_owned_training_review_v3 import (require,inspect_request_clock,inspect_histories,inspect_samples,attempt_events)
from .twoFIFO_owned_collect_v3 import ACTIVITY_FILE,ACTIVITY_SCHEMA,TIME_DOMAIN
from .twoFIFO_training_manifest_v1 import OWNERS,digest
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .stage_response_features_v1 import receipt,checked,stamp
from .macro_six_term_bridge_v1 import whole_score
from .macro_response_queue_reference_v1 import encode

SCHEMA='gsdeploy.twoFIFO_fresh181_independent_validation_review.v1'
RAW_SCHEMA=executor.RAW_SCHEMA
PLAN_SCHEMA=executor.SCHEMA+'.attempt'
MEASUREMENT_NS=90_000_000_000
DRAIN_NS=120_000_000_000


def inspect_cohort(plan, raw):
    require(plan.get('schema') == PLAN_SCHEMA and plan.get('split') == 'validation'
            and raw.get('schema') == RAW_SCHEMA and raw.get('split') == 'validation',
            'Exact native4 validation plan/raw role required')
    for name in ('execution_id', 'capture_id', 'attempt_id'):
        require(isinstance(plan.get(name), str) and plan[name] and raw.get(name) == plan[name],
                'Plan/raw identity mismatch: ' + name)
    cases = plan['cases']; rows = raw['request_records']; observed_cases = raw['cases']
    require(0 < len(cases) <= 13 and len(observed_cases) == len(cases),
            'Exact bounded complete case list required')
    require(len({c['case_id'] for c in cases}) == len(cases), 'Unique planned case IDs required')
    expected = [(c['case_id'], a) for c in cases for a in c['arrivals']]
    require(len({a['request_id'] for _, a in expected}) == len(expected), 'Unique planned request IDs required')
    require([r.get('request_id') for r in rows] == [a['request_id'] for _, a in expected],
            'Every offered row must be retained in original plan order')
    by_id = {r['request_id']: r for r in rows}
    issues = []; outcome_diagnostics=[];late_writes = 0; prior_end = None; saw_unstarted = False
    for case, observed in zip(cases, observed_cases):
        cid = case['case_id']; arrivals = case['arrivals']
        require(observed.get('case_id') == cid and case['public_arrivals_sha256'] == digest(arrivals)
                and observed.get('public_arrivals_sha256') == case['public_arrivals_sha256'],
                'Exact ordered case and public arrival digest required')
        require(observed.get('request_ids') == [a['request_id'] for a in arrivals],
                'Case must retain all offered identities')
        epoch = observed.get('epoch_ns', observed.get('observation_start_ns'))
        if epoch is None:
            saw_unstarted = True
            for a in arrivals:
                r = by_id[a['request_id']]
                require(r.get('status') == 'not_started' and r.get('scheduled_time_ns') is None,
                        'Unstarted offered rows must be explicit, not fabricated arrivals')
            issues.append('UNSTARTED_CASE:' + cid)
        else:
            require(not saw_unstarted and stamp(epoch) and (prior_end is None or prior_end <= epoch),
                    'Started cases must form an ordered prefix')
            end, deadline = epoch + MEASUREMENT_NS, epoch + MEASUREMENT_NS + DRAIN_NS
            require(observed.get('measurement_end_ns') == end
                    and observed.get('fixed_drain_deadline_ns') == deadline,
                    'Original fixed measurement/drain required')
            final = observed.get('observation_finalized_ns')
            if not stamp(final) or final < deadline:
                issues.append('FIXED_DRAIN_NOT_OBSERVED:' + cid)
            prior_end = final if stamp(final) else deadline
        for a in arrivals:
            r = by_id[a['request_id']]
            require(r.get('profile_job_id') == cid and r.get('split') == 'validation'
                    and r.get('model') == a['owner'] and r.get('task') == a['task']
                    and r.get('pool_id') == a['pool_id'] and r.get('arrival_index') == a['arrival_index']
                    and r.get('arrival_ns') == a['arrival_ns'], 'Exact public owner/task/input/arrival required')
            if epoch is not None:
                require(r.get('scheduled_time_ns') == epoch + a['arrival_ns'], 'Exact scheduled arrival required')
                if r.get('status') == 'completed':
                    clock = inspect_request_clock(r); late_writes += clock['late_write_after_RESULT']
                    if (r.get('worker_response', {}).get('success') is not True
                            or r['completion_time_ns'] > deadline):
                        outcome_diagnostics.append('FAILED_OR_AFTER_CUTOFF_RESULT:' + r['request_id'])
                else:
                    outcome_diagnostics.append('INCOMPLETE_OFFERED_REQUEST:' + r['request_id'])
    priming = raw['priming_request_records']; public_priming = plan['priming']
    require(len(public_priming) == 18 and len(priming)==18, 'Exactly eighteen planned priming requests required per attempt')
    require([(r['model'], r['task'], r['pool_id']) for r in priming]
            == [(a['owner'], a['task'], a['pool_id']) for a in public_priming[:len(priming)]], 'Exact ordered priming prefix required')
    require([r.get('request_id') for r in priming] == [r['request_id'] for r in public_priming],
            'Exact predeclared priming identities required')
    all_rows = priming + rows
    require(len({r.get('request_id') for r in all_rows}) == len(all_rows), 'Priming/measured IDs must be disjoint')
    for r in priming:
        if r.get('status') != 'completed' or r.get('worker_response', {}).get('success') is not True:
            issues.append('INCOMPLETE_PRIMING:' + str(r.get('request_id')))
        else:
            inspect_request_clock(r)
    # Source order constrains two independent serial branches. Sender can have
    # several submitted requests while its worker is still in an earlier one.
    owners = defaultdict(list)
    public_order=priming+[by_id[a['request_id']] for c in cases
        for a in sorted(c['arrivals'],key=lambda a:(a['arrival_ns'],a['arrival_index']))]
    for r in public_order:
        if r.get('actual_submit_time_ns') is not None:
            owners[r['model']].append(r)
    for owner, members in owners.items():
        require(owner in OWNERS, 'Unknown resident owner')
        for previous, current in zip(members, members[1:]):
            require(previous['enqueued_time_ns']<=current['enqueued_time_ns'],
                    'Dispatcher must preserve public per-owner arrival and tie order')
            if stamp(previous.get('submit_write_completed_time_ns')):
                require(previous['submit_write_completed_time_ns'] <= current['actual_submit_time_ns'],
                        'Original sender FIFO release order required')
            if stamp(current.get('observed_worker_start_time_ns')):
                ready = previous.get('previous_handler_next_request_ready', {})
                require(stamp(ready.get('perf_ns')) and ready['perf_ns'] <= current['observed_worker_start_time_ns'],
                        'Original worker FIFO must release at prior READY')
                starts=[e for e in current.get('execution_events',[]) if e.get('event') is None]
                require(len(starts)==1 and starts[0].get('perf_ns')==current['observed_worker_start_time_ns'],
                        'Exact next handler START notification required')
                for key in ('received_time_ns','collector_dequeued_time_ns','collector_processing_time_ns'):
                    require(stamp(ready.get(key)) and stamp(starts[0].get(key)) and ready[key]<=starts[0][key],
                            'Same producer READY must precede next START on every parent clock')
    return dict(issues=issues, offered_requests=len(rows), planned_cases=len(cases),
                outcome_diagnostics=outcome_diagnostics,
                completed_requests=sum(r.get('status') == 'completed' for r in rows),
                unstarted_requests=sum(r.get('status') == 'not_started' for r in rows),
                priming_requests=len(priming), late_write_after_RESULT_requests=late_writes,
                complete_offered_denominator_preserved=True, global_admission_cap_added=False)


def inspect_ledger(plan,raw,ledger):
    """Exact fresh-v4 ledger, with accepted forecasts before first arrival."""
    require(raw.get('no_mutating_checkpoint_during_case') is True
        and raw.get('checkpoint_spans_file')==ACTIVITY_FILE and raw.get('checkpoint_spans_schema')==ACTIVITY_SCHEMA
        and raw.get('checkpoint_spans_execution_id')==plan['execution_id'] and raw.get('checkpoint_time_domain')==TIME_DOMAIN,
        'Exact protected fresh-v4 timing path required')
    require(ledger.get('schema')==ACTIVITY_SCHEMA and ledger.get('execution_id')==plan['execution_id']
        and ledger.get('time_domain')==TIME_DOMAIN and ledger.get('all_spans_complete') is True,
        'Complete canonical checkpoint activity sidecar required')
    spans=ledger['checkpoint_spans'];windows=raw['measurement_windows']
    require(len(windows)==1 and [s['span_id'] for s in spans]==list(range(len(spans))),'Complete ordered spans required')
    window=windows[0];case=plan['cases'][0];origin=window['start_ns'];deadline=window['fixed_drain_deadline_ns']
    fields={'span_id','kind','case_id','phase','stage','time_domain','start_ns','end_ns'}
    require(all(set(s)==fields and s['time_domain']==TIME_DOMAIN and stamp(s['start_ns']) and stamp(s['end_ns'])
        and s['start_ns']<=s['end_ns'] and (s['end_ns']<origin or s['start_ns']>deadline) for s in spans),
        'No full history/checkpoint operation may intersect original90+120')
    require([s['start_ns'] for s in spans]==sorted(s['start_ns'] for s in spans),'Actual span issue order required')
    wrappers=[s for s in spans if s['kind']=='checkpoint']
    require([(s['stage'],s['case_id']) for s in wrappers]==[('before_startup',None),('postpriming',None),
        ('precase',case['case_id']),('postdrain',case['case_id']),('shutdown',None)],
        'Exact five fresh-v4 durable checkpoints required')
    for parent in wrappers:
        children=spans[parent['span_id']+1:parent['span_id']+3]
        require([s['kind'] for s in children]==['checkpoint_locked_snapshot','checkpoint_serialization_and_atomic_save']
            and all((s['stage'],s['case_id'])==(parent['stage'],parent['case_id'])
                and parent['start_ns']<=s['start_ns']<=s['end_ns']<=parent['end_ns'] for s in children)
            and children[0]['end_ns']<=children[1]['start_ns'],'Full checkpoint lock/serialization children required')
    histories=[s for s in spans if s['kind']=='precase_history_snapshot']
    persistence=[s for s in spans if s['kind']=='precase_expected_forecast']
    require(len(histories)==len(persistence)==1 and len(spans)==17,
        'Exactly one actual prior snapshot and prefix persistence operation required')
    history=histories[0];persist=persistence[0];pre=wrappers[2]
    snapshot=raw['case_histories'][0]['prior_history_snapshot']
    accepted=raw.get('prediction_seal_accepted_ns')
    require(history['case_id']==persist['case_id']==case['case_id']
        and history['start_ns']<=snapshot['snapshot_time_ns']<=history['end_ns']<=persist['start_ns']<=persist['end_ns']
        and stamp(accepted) and persist['start_ns']<=accepted<=persist['end_ns']<=pre['start_ns']<=pre['end_ns']<origin,
        'Snapshot, complete six-term forecast/independent seal and durable precase checkpoint must precede origin')
    require(raw.get('public_case_calendars')==plan['cases'] and raw.get('priming_public_calendar')==plan['priming'],
        'Original calendars persisted before execution required')
    require(raw.get('case_preparations')==[dict(case_id=case['case_id'],public_arrivals_sha256=case['public_arrivals_sha256'],
        public_arrivals=case['arrivals'],prior_history_index=0,prepared_before_epoch=True,
        durable_checkpoint_return_ns=raw['case_preparations'][0]['durable_checkpoint_return_ns'])],
        'Exact original shared dispatcher preparation required')
    require(pre['end_ns']<=raw['case_preparations'][0]['durable_checkpoint_return_ns']<origin,
        'Actual durable dispatcher checkpoint return must precede origin')
    require(origin==snapshot['snapshot_time_ns']+300_000_000_000
        and raw.get('shared_dispatch_protocol')==expected.shared_dispatch_protocol()
        and raw.get('executed_worker_registry')==executor.resource.campaign.planned_worker_registry(checked(plan['context'])),
        'Exact shared worker registry and resource dispatcher snapshot+300s protocol required')
    executor.resource.settled_priming(plan,snapshot,origin)
    return dict(complete_spans=len(spans),checkpoints=5,history_snapshots=1,prefix_persistence=1,
        critical_interval_overlaps=0,acceptance_checkpoint_return_before_origin=True,zero_observation_overhead_proven=False)


def inspect_prediction(plan,raw,manifest_path,plan_path,execution):
    barrier=plan['forecast_barrier'];gate_ref=receipt(barrier['gate_request_path']);prefix_ref=receipt(barrier['prefix_path'])
    report_ref=receipt(barrier['expected_report_path']);seal_ref=receipt(barrier['seal_path'])
    require(raw.get('causal_prefix_status')=='PERSISTED_BEFORE_FIRST_COHORT_SEND'
        and raw.get('predictions_durable_before_cohort') is True
        and raw.get('precohort_prefix')==prefix_ref and raw.get('prediction_gate')==gate_ref
        and raw.get('expected_forecast_report')==report_ref and raw.get('prediction_seal')==seal_ref,
        'Actual target raw must bind the exact persisted prefix, report and seal')
    refs=[gate_ref,prefix_ref,report_ref,seal_ref]
    with SourceChangeGuard(refs) as guard:
        with expected.open_target_session(plan['saved_matching_fit']['path'],manifest_path,plan_path) as session:
            _,_,document,gate=typed._prefix_and_gate(session.queue,gate_ref)
            require(document['prefix_snapshot']==raw['case_histories'][0]['prior_history_snapshot']
                and document['prefix_snapshot']['ready']==raw['ready'], 'Actual full prior snapshot must match sealed prefix')
            executor.prefix.inspect_native_ownership(document['prefix_snapshot'],document['ready_ownership_evidence'],
                collector_pid=execution['process_group'])
            report=expected.audit_report(session,report_ref['path'],require_native=True)
            executor.verify(report_ref);guard.check()
            seal=expected.audit_seal_structure(session,gate_ref,seal_ref,now_ns=raw['prediction_seal_accepted_ns'])
            window=raw['measurement_windows'][0]
            require(raw['planned_cohort_epoch_ns']==gate['forecast_origin_ns']==window['start_ns']
                and gate['measurement_end_ns']==window['end_ns'] and gate['fixed_drain_deadline_ns']==window['fixed_drain_deadline_ns']
                and seal['expected_report']==report_ref,'Actual original horizon must match pre-origin full six-term prediction')
            guard.check();return dict(refs=refs,report=report,gate=gate,prefix=document,
                source_authenticated_full_replay=True,forecast_accuracy_not_assumed=True)


def actual_quality(raw,plan,catalog_ref,requests,deadline):
    require(len(raw['request_records'])==len(requests)==181
        and [r['request_id'] for r in raw['request_records']]==[r['request_id'] for r in requests],
        'Actual quality requires every original181 offered outcome in order')
    references=prepared_references(plan,catalog_ref);rates=checked(plan['public_contract'])['workload_rps']
    totals={t:dict(numerator=0.,denominator=0.,offered=0,scored=0,missing=[]) for t in rates};outputs=[]
    for row,public in zip(raw['request_records'],requests):
        require((row['model'],row['task'],row['pool_id'])==(public['owner'],public['task'],public['pool_id']),
            'Actual quality exact owner/task/input identity required')
        task=public['task'];target=totals[task];target['offered']+=1
        complete=(row.get('status')=='completed' and type(row.get('completion_time_ns')) is int
            and public['arrival_ns']<=row['completion_time_ns']<=deadline and (row.get('worker_response') or {}).get('success') is True)
        record=dict(request_id=public['request_id'],task=task,pool_id=public['pool_id'],status='UNKNOWN_NATIVE_OUTCOME')
        if complete:
            reference=references[task,public['pool_id']];response=copy.deepcopy(row['worker_response'])
            require(response['request_id']==public['request_id'] and response['task']==task
                and row['sample_id']==reference['id'],'Exact original native output/reference identity required')
            if task=='ocr':response['prediction']=page_ocr_text(response['prediction'])
            num,den=sample_totals(task,score_result(reference,response))
            require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (num,den)),
                'Finite native quality counts required')
            target['numerator']+=num;target['denominator']+=den;target['scored']+=1
            record.update(status='MEASURED_VALIDATION_OUTPUT',numerator=num,denominator=den)
        else:target['missing'].append(public['request_id'])
        outputs.append(record)
    for total in totals.values():total['loss']=total['numerator']/total['denominator'] if total['offered'] and not total['missing'] and total['denominator']>0 else None
    Q=0. if all(v['loss'] is not None for v in totals.values()) else None
    if Q is not None:
        total_rate=sum(rates.values())
        # Match the native original scorer's task order and multiplication/
        # division order exactly, with observed counts replacing expectations.
        for task,rate in rates.items():
            Q+=(rate/total_rate)*totals[task]['numerator']/totals[task]['denominator']
    return dict(schema=SCHEMA+'.actual_quality',measured_Q=Q,per_task=totals,requests=outputs,
        offered_requests=181,catalog=catalog_ref,numerator_denominator_summed_before_division=True,
        training_expectation_fallback_used=False,validation_labels_read=True,training_feature_extraction_eligible=False)


def actual_score(raw,plan,requests,labels,quality,resource,source_evidence):
    contract=checked(plan['public_contract']);window=raw['measurement_windows'][0];a=window['start_ns'];b=window['end_ns'];d=window['fixed_drain_deadline_ns']
    known={r['request_id']:r['arrival_ns']+r['reader_E2E_ns'] for r in labels['requests'] if r['completed_within_fixed_drain']}
    frame=dict(original_window=encode([F(a),F(b)]),original_drain_end_ns=encode(F(d)),cutoff_ns=encode(F(d)),
        RESULT_receipts=encode({rid:F(at) for rid,at in known.items()}),offered_requests=181,
        unfinished_request_ids=[r['request_id'] for r in requests if r['request_id'] not in known])
    resources=None
    if resource is not None:
        joint=resource['joint_tuple'];resources=dict(original_window_ns=[a,b],original_drain_end_ns=d,resident_owners=list(OWNERS),
            startup_energy_joules=joint['startup_joules'],startup_through_phase_start_owned_peak_mib=joint['premeasurement_owned_sampled_peak_mib'],
            phase_start_through_drain_owned_peak_mib=joint['measurement_through_drain_owned_sampled_peak_mib'],
            measurement_board_energy_joules=joint['measurement_board_joules'],measurement_energy_includes_drain=False,
            memory_safety_factor_already_applied=False,startup_energy_already_amortized=False,source_evidence=source_evidence)
    score=whole_score(frame,requests,contract,{},resources)['score'];score['terms']['Q']=quality['measured_Q']
    score['quality_totals_by_task']={task:dict(numerator=v['numerator'],denominator=v['denominator'],
        requests=v['offered'],basis='actual_native_validation_counts_summed_before_division')
        if not v['missing'] else None for task,v in quality['per_task'].items()}
    score['unknown_reasons']=[r for r in score['unknown_reasons'] if not r.startswith('UNKNOWN_AGGREGATE_QUALITY:')]
    score['unknown_reasons']+=['UNKNOWN_ACTUAL_VALIDATION_QUALITY:'+t for t,v in quality['per_task'].items() if v['loss'] is None]
    complete=all(v is not None for v in score['terms'].values());score['six_term_comparison_complete']=complete
    score['absolute_utility_complete']=complete;score['cost']=sum(contract['weights'][k]*v for k,v in score['terms'].items()) if complete else None
    score['full_U']=-score['cost'] if complete else None
    terms=score['terms'];score['metric_components']=components(score['per_task'],contract['workload_rps'],contract['latency_reference_ms'],
        terms['T'],terms['M'],contract['weights'],E=terms['E'],Q=terms['Q']) if terms['L'] is not None and terms['P'] is not None else None
    score['quality_scope']='Actual validation outputs; original task summed counts and offered rates'
    return dict(schema=SCHEMA+'.observed_score',score=score,original_equal_six_terms_retained=True,
        resources=resources,validation_labels_read=True,used_for_fit=False,predicted_full_U=None,delta_U=None,
        predicted_full_U_unknown_reason=None,final_gain_eligible=False)


def actual_resources(raw):
    from .twoFIFO_observed_resources_v2 import observed
    facts=dict(observed(raw),validation_labels_read=True,training_feature_extraction_eligible=False)
    window=raw['measurement_windows'][0];startup=next(b['time_ns'] for b in raw['lifecycle_boundaries'] if b['boundary']=='startup_begin')
    before=[s['owned_used_mib'] for s in raw['board_samples'] if startup<=s['time_ns']<=window['start_ns']]
    case=facts['cases'][0]
    return dict(schema=SCHEMA+'.actual_resources',joint_tuple=dict(startup_joules=facts['startup']['energy_joules'],
        premeasurement_owned_sampled_peak_mib=max(before) if before else None,
        measurement_through_drain_owned_sampled_peak_mib=case['local_measurement_through_drain_owned_sampled_peak_mib'],
        measurement_board_joules=case['measurement_energy_joules']),observed_original_resources=facts,
        measurement_includes_drain=False,startup_amortization_applied=False,safe_factor_applied=False,
        sampled_memory_is_continuous_capacity_bound=False,validation_labels_read=True,used_for_fit=False)


def _protocol_source_refs(context):
    refs=[]
    for member in context.get('execution_sources',[]):refs.extend([member['frozen'],receipt(member['original_path'])])
    for binding in context['worker_bindings'].values():
        refs.append(receipt(binding['spec']))
        command=binding['command']
        if '--source-root' in command:
            root=Path(command[command.index('--source-root')+1])
            refs.extend(receipt(root/name) for name in ('gsdeploy/backends/compact.py','gsdeploy/tasks.py'))
    return refs


def review(raw_path,output,*,manifest_path,execution_path):
    raw_ref=receipt(raw_path);raw=checked(raw_ref)
    require(raw.get('schema')==RAW_SCHEMA and raw.get('collector_schema')==RAW_SCHEMA
        and raw.get('split')=='validation' and raw.get('backend')=='native' and raw.get('kind')=='measured',
        'Exact native4 validation raw required before evaluation labels are inspected')
    plan_ref=raw['plan'];plan=checked(executor.verify(plan_ref));context=checked(executor.verify(raw['context']))
    require(plan.get('schema')==PLAN_SCHEMA and plan.get('split')=='validation','Exact validation plan role required')
    manifest_ref=receipt(manifest_path);manifest_snapshot=checked(manifest_ref);execution_ref=receipt(execution_path)
    declared=next(a for a in manifest_snapshot['attempts'] if a['plan']==plan_ref)
    claim_ref=receipt(declared['claim'])
    activity_ref=receipt(Path(raw_path).resolve().parent/ACTIVITY_FILE)
    barrier=plan['forecast_barrier'];prefix_ref=receipt(barrier['prefix_path']);gate_ref=receipt(barrier['gate_request_path'])
    report_ref=receipt(barrier['expected_report_path']);seal_ref=receipt(barrier['seal_path'])
    public=checked(executor.verify(plan['public_preparation']));public_plan=checked(executor.verify(public['inputs']['public_plan']))
    catalog_ref=public_plan['catalog'];executor.verify(catalog_ref)
    watched=strict_references([raw_ref,raw,plan_ref,plan,context,_protocol_source_refs(context),manifest_ref,manifest_snapshot,
        execution_ref,claim_ref,activity_ref,prefix_ref,gate_ref,report_ref,seal_ref,catalog_ref,public,public_plan,implementation_sources(__file__)])
    with SourceChangeGuard(watched) as guard:
        guard.check();manifest=executor.audit(manifest_path)
        require(manifest==manifest_snapshot and executor.audit_plan(plan_ref['path'])==plan,'Whole canonical native4 target must rederive')
        item=next(a for a in manifest['attempts'] if a['plan']==plan_ref)
        require(Path(raw_path).resolve()==Path(item['collection'])/'raw.json'
            and Path(execution_path).resolve()==Path(item['collection'])/'execution.json',
            'Canonical actual raw and supervisor receipt required')
        execution=checked(executor.verify(execution_ref));executor.completed_execution(execution,manifest_ref,item)
        require(raw['context']==plan['context'] and raw.get('execution_context_sha256')==plan['execution_context_sha256']
            and raw.get('prepared_pool')==plan['prepared_pool'] and raw.get('source_bindings')==plan['sources']
            and raw.get('source_freeze')==plan['source_freeze']
            and raw.get('fresh181_executor_sources')==implementation_sources(executor.__file__),
            'Actual context, pool, complete native4 collector closure and freeze required')
        require(checked(executor.verify(raw['executed_cases']))==dict(attempt_id=plan['attempt_id'],execution_id=plan['execution_id'],cases=plan['cases']),
            'Complete original executed181 calendar binding required')
        cohort=inspect_cohort(plan,raw);issues=list(cohort['issues']);histories=inspect_histories(plan,raw)
        ledger=inspect_ledger(plan,raw,checked(executor.verify(activity_ref)))
        protocol=inspect_protocol(raw);issues.extend(protocol['issues'])
        require(protocol['all_offered_and_priming_prefixes']==199 and len(protocol['requests'])==199,
            'Every199 original stdout payload/phase/READY prefix must be inspected')
        require(all(ref in watched for ref in protocol['sources']),'No output parser input may escape the outer source watch')
        attempt_events(raw)
        shutdown=raw.get('shutdown',{})
        require(set(shutdown)==set(OWNERS) and all(type(v.get('exit_code')) is int and type(v.get('forced')) is bool
            and type(v.get('bye_received')) is bool for v in shutdown.values()),'Actual four-resident cleanup report required')
        if raw.get('status')!='completed' or any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors')):
            issues.append('CLEAN_FULL_HORIZON_COLLECTION_REQUIRED')
        telemetry=inspect_samples(raw,execution['process_group']);prediction=inspect_prediction(plan,raw,manifest_path,plan_ref['path'],execution)
        from .natural_macro_resources_candidate_v1 import priming_identity
        require(priming_identity(raw,context)==plan['priming_semantics']['sha256'],
            'Actual original eighteen priming contents must match saved law semantics')
        gate=prediction['gate'];requests=gate['public_requests'];origin=gate['forecast_origin_ns'];deadline=gate['fixed_drain_deadline_ns']
        labels=quality=resources=score=forecast_errors=timeline=six_term_errors=solo_reference=solo_comparison=None
        if not issues:
            labels=metrics.observed_labels(raw,requests,deadline)
            solo_reference=solo.build(plan['saved_matching_fit']['path'],plan_ref['path'])
            require(all(ref in watched for ref in strict_references(solo_reference['sources'])),
                'No solo-reference input may escape the independent outer source watch')
            contract=checked(executor.verify(plan['public_contract']))
            sample_times=[origin+t for t in contract['queue_sample_offsets_ns']]
            timeline=metrics.actual_timeline(raw,requests,origin,deadline,sample_times)
            forecast_errors=metrics.prediction_errors(prediction['report']['source_queue_forecast'],labels,timeline,requests,origin,deadline,
                sample_times=sample_times)
            solo_comparison=solo.slowdown_labels(labels,solo_reference,forecast_errors)
            quality=actual_quality(raw,plan,catalog_ref,requests,deadline);resources=actual_resources(raw)
            score=actual_score(raw,plan,requests,labels,quality,resources,[raw_ref,execution_ref,activity_ref])
            from .actual_worker_twoFIFO_reference_v5_2 import decode
            cid=plan['candidate_id'];candidate_ids=[a['candidate_id'] for a in manifest['attempts']]
            score['predicted_full_U']=decode(prediction['report']['expected_score'])['expected_U_under_supplied_marginals']
            six_term_errors=expected.evaluate_validation({cid:prediction['report']},
                {cid:dict(terms=score['score']['terms'],full_U=score['score']['full_U'],weights=contract['weights'],
                    primitives=resources['joint_tuple'])},candidate_ids=candidate_ids)
        guard.check()
        result=dict(schema=SCHEMA,status='PASS_CONDITIONAL_NATIVE_VALIDATION_EVIDENCE_AND_DESCRIPTIVE_ERRORS' if not issues else 'NOT_ELIGIBLE_NATIVE_VALIDATION',
            manifest=manifest_ref,plan=plan_ref,raw=raw_ref,execution=execution_ref,claim=claim_ref,issues=issues,
            cohort=cohort,histories=histories,checkpoint_activity=ledger,checkpoint_activity_source=activity_ref,
            stdout_payload_protocol_review=protocol,telemetry=telemetry,source_authenticated_prediction=dict(
                report=report_ref,seal=seal_ref,gate=gate_ref,prefix=prefix_ref,full_queue_replay_rederived=True,full_six_term_replay_rederived=True),
            observed=labels,queue_timeline=timeline,forecast_errors=forecast_errors,
            solo_host_reference=solo_reference,conditional_solo_host_comparison=solo_comparison,
            actual_quality=quality,actual_resources=resources,actual_six_term_score=score,six_term_forecast_errors=six_term_errors,
            offered_requests=181,priming_requests=18,predefined_forecast_worlds=12,
            recorded_stdout_payloads_reparsed=True,verbatim_stdout_transcript_available=False,
            independent_stdout_raw_bytes_reparsed=False,stdout_provenance_scope='SOURCE_AUTHENTICATED_COLLECTOR_PARSED_PAYLOADS_AND_FOUR_CLOCKS',
            validation_labels_read=True,validation_labels_used_for_fit=False,numerical_fit_run=False,
            training_feature_extraction_eligible=False,main_queue_provider_eligible=False,
            independent_prediction_accuracy_validated=False,original_stability_evaluated=False,
            predicted_full_U=None if score is None else score['predicted_full_U'],delta_U=None,candidate_ranking_evaluated=False,final_gain_eligible=False,GPU_started=False,
            sources=watched)
        out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
        ref=executor.save(out/'report.json',encode(result));guard.check();executor.verify(report_ref);return ref


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','manifest','execution','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();print(json.dumps(review(args.raw,args.output,manifest_path=args.manifest,execution_path=args.execution)))
