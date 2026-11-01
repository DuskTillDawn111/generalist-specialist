"""Evaluation-only full offered cohort diagnostics; no fit or policy changes."""
from __future__ import annotations
from fractions import Fraction as F
import bisect
from .twoFIFO_training_manifest_v1 import require
from .macro_response_queue_reference_v1 import encode
from .actual_worker_twoFIFO_reference_v5_2 import decode

SCHEMA='gsdeploy.twoFIFO_fresh181_validation_metrics.v1'
QUEUES=('arrived_not_STARTED','enqueued_not_submitted','submitted_waiting_START',
        'running_START_to_READY','arrived_RESULT_unfinished')


def signed_error_bounds(predicted,observed_lower,observed_upper):
    """None lower means -infinity; None upper means +infinity."""
    if predicted is None:
        return dict(signed_lower=None,signed_upper=None,absolute_lower=None,absolute_upper=None)
    return dict(signed_lower=None if observed_upper is None else predicted-observed_upper,
        signed_upper=predicted-observed_lower,
        absolute_lower=max(observed_lower-predicted,0,
            0 if observed_upper is None else predicted-observed_upper),
        absolute_upper=None if observed_upper is None else max(abs(predicted-observed_lower),abs(predicted-observed_upper)))


def percentile(values,q=F(95,100)):
    require(values,'Nonempty complete offered percentile population required')
    ordered=sorted(values);pos=q*(len(ordered)-1);lo=pos.numerator//pos.denominator;part=pos-lo
    if not part:return ordered[lo]
    a,b=ordered[lo],ordered[lo+1]
    return float('inf') if b==float('inf') else a+(b-a)*part


def observed_labels(raw,requests,deadline):
    require(len(requests)==len(raw['request_records'])==181,'Keep every original181 offered label')
    require([r['request_id'] for r in requests]==[r['request_id'] for r in raw['request_records']],
        'Exact ordered offered labels required')
    labels=[]
    for public,row in zip(requests,raw['request_records']):
        arrival=public['arrival_ns'];reader=row.get('completion_time_ns')
        complete=(row.get('status')=='completed' and (row.get('worker_response') or {}).get('success') is True
            and type(reader) is int and arrival<=reader<=deadline)
        start=row.get('observed_worker_start_time_ns');ready=(row.get('previous_handler_next_request_ready') or {}).get('perf_ns')
        service=ready-start if type(start) is int and type(ready) is int and start<=ready<=deadline else None
        labels.append(dict(request_id=public['request_id'],owner=public['owner'],task=public['task'],pool_id=public.get('pool_id'),
            arrival_ns=arrival,status=row.get('status'),completed_within_fixed_drain=complete,
            reader_E2E_ns=reader-arrival if complete else None,
            reader_E2E_lower_ns=reader-arrival if complete else deadline-arrival,
            reader_E2E_upper_ns=reader-arrival if complete else None,
            actual_START_to_READY_ns=service,
            actual_arrival_to_START_ns=start-arrival if type(start) is int and start<=deadline else None,
            solo_slowdown_ratio=None,solo_slowdown_unknown_reason='NO_SOURCE_AUTHENTICATED_EXACT_INPUT_SOLO_BASELINE',
            host_START_READY_is_not_physical_SM_service=True))
    def summary(group):
        lower=percentile([F(r['reader_E2E_lower_ns']) for r in group]);upper=percentile([
            float('inf') if r['reader_E2E_upper_ns'] is None else F(r['reader_E2E_upper_ns']) for r in group])
        return dict(offered=len(group),successful_within_drain=sum(r['completed_within_fixed_drain'] for r in group),
            P95_lower_ns=lower,P95_upper_ns=None if upper==float('inf') else upper,
            P95_ns=lower if lower==upper else None,
            P95_identifiable=lower==upper,censored_survivor_P95_substitution=False)
    return dict(schema=SCHEMA+'.observed',requests=labels,overall=summary(labels),
        per_task={task:summary([r for r in labels if r['task']==task]) for task in sorted({r['task'] for r in labels})},
        censored_upper_bound_None_means_unbounded=True,validation_labels_read=True,training_feature_extraction_eligible=False)


def _active_bounds(begin,end,end_lower,at):
    if begin is None or begin>at:return 0,0
    if end is not None:return (1,1) if at<end else (0,0)
    return (1,1) if at<end_lower else (0,1)


def actual_snapshot(raw,requests,at):
    lower={k:0 for k in QUEUES};upper=dict(lower);uncertain={k:[] for k in QUEUES}
    for public,row in zip(requests,raw['request_records']):
        rid=public['request_id'];arrival=public['arrival_ns']
        if arrival>at:continue
        enqueue=row.get('enqueued_time_ns');submit=row.get('actual_submit_time_ns');start=row.get('observed_worker_start_time_ns')
        ready=(row.get('previous_handler_next_request_ready') or {}).get('perf_ns')
        # ERROR and post-drain RESULT still release this stdout-reader queue;
        # neither releases the worker, which requires the separate READY edge.
        reader=row.get('response_reader_event_enqueue_time_ns')
        if reader is None:reader=row.get('completion_time_ns') or row.get('late_response_time_ns')
        payload=row.get('worker_response') or row.get('late_response_after_drain') or {}
        known_last=max([t for t in [start,payload.get('end_perf_ns')]+[
            e.get('perf_ns') for e in row.get('execution_events',[])] if type(t) is int],default=arrival)
        values={
            'arrived_not_STARTED':_active_bounds(arrival,start,submit or arrival,at),
            'enqueued_not_submitted':_active_bounds(enqueue,submit,enqueue or arrival,at),
            'submitted_waiting_START':_active_bounds(submit,start,submit or arrival,at),
            'running_START_to_READY':_active_bounds(start,ready,known_last,at),
            # This queue describes received stdout, not unseen GPU progress.
            'arrived_RESULT_unfinished':(int(reader is None or reader>at),)*2}
        if start is None and type(submit) is int and submit<=at:values['running_START_to_READY']=(0,1)
        if enqueue is None:values['enqueued_not_submitted']=(0,1)
        if submit is None and (enqueue is None or enqueue<=at):values['submitted_waiting_START']=(0,1)
        for key,(a,b) in values.items():
            lower[key]+=a;upper[key]+=b
            if a!=b:uncertain[key].append(rid)
    return dict(at_ns=at,lower=lower,upper=upper,uncertain_request_ids=uncertain,
        no_READY_release_inferred_from_RESULT=True)


def actual_timeline(raw,requests,origin,deadline,sample_times):
    cuts={origin,deadline,*sample_times}
    for public,row in zip(requests,raw['request_records']):
        cuts.add(public['arrival_ns'])
        for at in [row.get(k) for k in ('enqueued_time_ns','actual_submit_time_ns','observed_worker_start_time_ns',
                'completion_time_ns','late_response_time_ns','response_reader_event_enqueue_time_ns')]+[
                (row.get('previous_handler_next_request_ready') or {}).get('perf_ns'),
                (row.get('worker_response') or row.get('late_response_after_drain') or {}).get('end_perf_ns')]+[
                e.get('perf_ns') for e in row.get('execution_events',[])]:
            if type(at) is int and origin<=at<=deadline:cuts.add(at)
    return [actual_snapshot(raw,requests,at) for at in sorted(cuts) if origin<=at<=deadline]


def _predicted_state(state):
    life=state['lifecycle_count']
    return {'arrived_not_STARTED':life['arrived_not_submitted']+life['submitted_waiting_START'],
        'enqueued_not_submitted':sum(len(h['enqueued_waiting_request_ids']) for h in state['parent_sender_FIFO_heads']),
        'submitted_waiting_START':life['submitted_waiting_START'],
        'running_START_to_READY':life['running_START_to_READY'],
        'arrived_RESULT_unfinished':len(state['arrived_RESULT_unfinished_request_ids'])}


def queue_errors(actual,diagnostics,origin,deadline):
    predicted=[s['state'] for s in diagnostics['event_slices']]+[diagnostics['cutoff_state']]
    require(predicted and predicted[0]['at_ns']==origin and predicted[-1]['at_ns']==deadline,
        'Prediction must cover the original full210s queue horizon')
    ac=[s['at_ns'] for s in actual];pc=[s['at_ns'] for s in predicted]
    cuts=sorted(set(ac+pc));integral={k:[F(0),F(0)] for k in QUEUES};peak={k:[0,0] for k in QUEUES}
    for a,b in zip(cuts,cuts[1:]):
        truth=actual[bisect.bisect_right(ac,a)-1];point=_predicted_state(predicted[bisect.bisect_right(pc,a)-1])
        for key in QUEUES:
            p=point[key];lo=truth['lower'][key];hi=truth['upper'][key]
            low=max(lo-p,0,p-hi);high=max(abs(p-lo),abs(p-hi))
            integral[key][0]+=(b-a)*low;integral[key][1]+=(b-a)*high
            peak[key]=[max(peak[key][0],low),max(peak[key][1],high)]
    return dict(full_horizon_ns=deadline-origin,queue_definitions=list(QUEUES),
        time_mean_absolute_error_bounds_requests={k:[v/(deadline-origin) for v in x] for k,x in integral.items()},
        peak_absolute_error_bounds_requests=peak,actual_censoring_propagated=True,
        RESULT_and_READY_not_interchanged=True)


def prediction_errors(report,observed,timeline,requests,origin,deadline,*,sample_times=()):
    value=decode(report['replay']['forecast']);ensemble=value['ensemble'];draws=ensemble['draws'];projections=value['complete_draw_projection']
    require(len(draws)==len(projections)==12 and sum(F(d['weight']) for d in draws)==1,'Retain all12 weighted worlds')
    require([r['request_id'] for r in requests]==[r['request_id'] for r in ensemble['public_requests']],
        'Forecast/evaluation offered population differs')
    actual={r['request_id']:r for r in observed['requests']};by_request={r['request_id']:[] for r in requests};worlds=[]
    for draw,projection in zip(draws,projections):
        require((draw['draw_id'],draw['weight'],draw['status'])==(projection['draw_id'],projection['weight'],projection['status'])
            and [r['request_id'] for r in projection['requests']]==list(actual),'Exact full181 projection in every world required')
        known=draw['status']=='GENERATED_WORLD';starts={};ready={};queue=None
        if known:
            diagnostic=draw['queue_diagnostics'];queue=queue_errors(timeline,diagnostic,origin,deadline)
            for state in [s['state'] for s in diagnostic['event_slices']]+[diagnostic['cutoff_state']]:
                for rid in state['worker_lifecycle']['running_START_to_READY']+state['worker_lifecycle']['worker_released']:
                    starts.setdefault(rid,state['at_ns'])
                for rid in state['worker_lifecycle']['worker_released']:ready.setdefault(rid,state['at_ns'])
        for public,row in zip(requests,projection['requests']):
            rid=public['request_id'];truth=actual[rid];at=row['completion_time_ns']
            e2e=at-public['arrival_ns'] if known and at is not None else None
            service=ready[rid]-starts[rid] if rid in ready and rid in starts else None
            by_request[rid].append(dict(draw_id=draw['draw_id'],weight=draw['weight'],status=draw['status'],
                predicted_reader_E2E_ns=e2e,error_ns=None if e2e is None or truth['reader_E2E_ns'] is None else e2e-truth['reader_E2E_ns'],
                signed_error_bounds_ns=signed_error_bounds(e2e,truth['reader_E2E_lower_ns'],truth['reader_E2E_upper_ns']),
                predicted_START_to_READY_ns=service,service_error_ns=None if service is None or truth['actual_START_to_READY_ns'] is None else service-truth['actual_START_to_READY_ns']))
        predicted_p95=draw['queue_diagnostics']['reader_E2E_P95_ns'] if known else None
        actual_p95=observed['overall']['P95_ns']
        worlds.append(dict(draw_id=draw['draw_id'],weight=draw['weight'],status=draw['status'],queue_errors=queue,
            predicted_complete_cohort_P95_ns=predicted_p95,
            P95_error_ns=None if predicted_p95 is None or actual_p95 is None else predicted_p95-actual_p95,
            P95_signed_error_bounds_ns=signed_error_bounds(predicted_p95,observed['overall']['P95_lower_ns'],observed['overall']['P95_upper_ns']),
            original_six_terms=None if not known else draw['scoring']['original_score']['terms'],
            predicted_full_U=None if not known else draw['scoring']['original_score']['full_U']))
    rows=[]
    for rid,values in by_request.items():
        mass=sum((F(v['weight']) for v in values if v['predicted_reader_E2E_ns'] is not None),F(0))
        total=sum((F(v['weight'])*v['predicted_reader_E2E_ns'] for v in values if v['predicted_reader_E2E_ns'] is not None),F(0))
        point=total if mass==1 else None;truth=actual[rid]['reader_E2E_ns']
        service_mass=sum((F(v['weight']) for v in values if v['predicted_START_to_READY_ns'] is not None),F(0))
        service_mean=sum((F(v['weight'])*v['predicted_START_to_READY_ns'] for v in values if v['predicted_START_to_READY_ns'] is not None),F(0)) if service_mass==1 else None
        service_truth=actual[rid]['actual_START_to_READY_ns']
        rows.append(dict(request_id=rid,task=actual[rid]['task'],draws=values,predicted_mean_reader_E2E_ns=point,
            known_complete_probability_mass=mass,remaining_probability_mass=1-mass,
            error_ns=None if point is None or truth is None else point-truth,
            signed_error_bounds_ns=signed_error_bounds(point,actual[rid]['reader_E2E_lower_ns'],actual[rid]['reader_E2E_upper_ns']),
            absolute_relative_error=None if point is None or truth in (None,0) else abs(point-truth)/truth,
            predicted_mean_START_to_READY_ns=service_mean,service_known_probability_mass=service_mass,
            service_absolute_relative_error=None if service_mean is None or service_truth in (None,0) else abs(service_mean-service_truth)/service_truth))
    error_values=[r['error_ns'] for r in rows if r['error_ns'] is not None]
    service_errors=[abs(r['service_error_ns']) for values in by_request.values() for r in values if r['service_error_ns'] is not None]
    relative=[r['absolute_relative_error'] for r in rows if r['absolute_relative_error'] is not None]
    service_relative=[r['service_absolute_relative_error'] for r in rows if r['service_absolute_relative_error'] is not None]
    per_task={}
    for task,truth in observed['per_task'].items():
        population=[r for r in requests if r['task']==task];predictions=[]
        for draw,projection in zip(draws,projections):
            values=[r['completion_time_ns']-p['arrival_ns'] for p,r in zip(requests,projection['requests'])
                if p['task']==task and r['completion_time_ns'] is not None] if draw['status']=='GENERATED_WORLD' else []
            predictions.append(dict(draw_id=draw['draw_id'],weight=draw['weight'],P95_ns=percentile(values) if len(values)==len(population) else None))
        mass=sum((F(p['weight']) for p in predictions if p['P95_ns'] is not None),F(0))
        mean=sum((F(p['weight'])*p['P95_ns'] for p in predictions if p['P95_ns'] is not None),F(0)) if mass==1 else None
        per_task[task]=dict(offered=len(population),draws=predictions,known_P95_probability_mass=mass,
            expected_draw_P95_ns=mean,observed_P95_ns=truth['P95_ns'],
            absolute_relative_error=None if mean is None or truth['P95_ns'] in (None,0) else abs(mean-truth['P95_ns'])/truth['P95_ns'],
            P95_signed_error_bounds_ns=signed_error_bounds(mean,truth['P95_lower_ns'],truth['P95_upper_ns']))
    grid=[];actual_cuts=[p['at_ns'] for p in timeline]
    for at in sample_times:
        require(origin<=at<=deadline,'Predeclared queue sample must lie in original horizon')
        truth=timeline[bisect.bisect_right(actual_cuts,at)-1];means={k:F(0) for k in QUEUES};mass=F(0)
        for draw in draws:
            if draw['status']!='GENERATED_WORLD':continue
            diagnostic=draw['queue_diagnostics'];states=[s['state'] for s in diagnostic['event_slices']]+[diagnostic['cutoff_state']]
            point=_predicted_state(states[bisect.bisect_right([s['at_ns'] for s in states],at)-1]);mass+=F(draw['weight'])
            for key in QUEUES:means[key]+=F(draw['weight'])*point[key]
        grid.append(dict(at_ns=at,known_world_probability_mass=mass,queues={key:dict(
            predicted_mean=means[key] if mass==1 else None,observed_lower=truth['lower'][key],observed_upper=truth['upper'][key],
            error_bounds=signed_error_bounds(means[key] if mass==1 else None,truth['lower'][key],truth['upper'][key])) for key in QUEUES}))
    grid_errors={key:dict(mean_absolute_error_lower=None if not grid or any(p['queues'][key]['predicted_mean'] is None for p in grid)
        else sum((p['queues'][key]['error_bounds']['absolute_lower'] for p in grid),F(0))/len(grid),
        mean_absolute_error_upper=None if not grid or any(p['queues'][key]['predicted_mean'] is None for p in grid)
        else sum((p['queues'][key]['error_bounds']['absolute_upper'] for p in grid),F(0))/len(grid)) for key in QUEUES}
    return encode(dict(schema=SCHEMA+'.forecast_errors',requests=rows,worlds=worlds,offered_requests=181,world_count=12,
        world_unknown_probability_mass=ensemble['world_unknown_probability_mass'],
        mean_absolute_point_reader_error_ns=sum(abs(e) for e in error_values)/len(error_values) if error_values else None,
        point_error_request_coverage=len(error_values)/181,
        point_reader_MARE=sum(relative)/len(relative) if relative else None,
        point_reader_MARE_request_coverage=len(relative)/181,
        point_START_to_READY_MARE=sum(service_relative)/len(service_relative) if service_relative else None,
        point_START_to_READY_MARE_request_coverage=len(service_relative)/181,
        per_task_P95=per_task,censored_error_bounds_retained=True,
        predeclared_queue_sample_grid=grid,point_queue_grid_errors=grid_errors,
        original_queue_grid_scope='arrived_RESULT_unfinished',
        mean_prediction_error_is_not_mean_world_absolute_error=True,
        service_comparison_scope='HOST_START_TO_READY_NOT_SOLO_OR_PHYSICAL_GPU_SERVICE',
        diagnostic_unweighted_supported_service_absolute_error_ns=sum(service_errors)/len(service_errors) if service_errors else None,
        unknown_worlds_or_censored_requests_dropped=False,unknown_worlds_renormalized=False,
        delta_U=None,delta_U_unknown_reason='QUEUE_SEAL_HAS_NO_AUTHENTICATED_FULL_U_PREDICTION',
        candidate_ranking_evaluated=False,independent_accuracy_qualified=False,final_gain_eligible=False))
