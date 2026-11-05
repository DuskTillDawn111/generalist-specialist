"""Independent development checks of frozen native laws and actual FIFO replay."""
from __future__ import annotations
from collections import defaultdict
import statistics

import numpy as np

from ..energy import integrate_power
from ..forecast_errors import UnknownEvidence
from .lifecycle import LifecycleProvider
from .native_context import NativeContextProvider, counts, design, within_support
from .native_training import digest, stage_layout, validate_capture
from .simulator import FullHorizonModel


DEFAULT_CRITERIA=dict(schema='gsdeploy.native_development_accuracy_criteria.v1',
    minimum_independent_runs=2,maximum_mean_e2e_relative_error=.15,
    maximum_p95_e2e_relative_error=.30,maximum_p95_native_work_error=.20,
    maximum_integrated_energy_relative_error=.15,all_observed_contexts_supported=True,
    scope='controlled independent development capture accuracy; no optimizer or formal utility admission')


def quantile(values,q):
    values=sorted(values)
    return values[min(len(values)-1,round(q*(len(values)-1)))] if values else None


def metrics(values):
    return dict(n=len(values),mean=statistics.mean(values) if values else None,p95=quantile(values,.95),
                maximum=max(values) if values else None)


def validate_native_fit(fit,bundles,criteria):
    return _score_native_fit(fit,bundles,criteria,training_replay=False)


def replay_native_training(fit,bundles):
    """Executable replay diagnostic, explicitly unable to qualify validation."""
    return _score_native_fit(fit,bundles,DEFAULT_CRITERIA,training_replay=True)


def _score_native_fit(fit,bundles,criteria,*,training_replay):
    if criteria!=DEFAULT_CRITERIA:raise ValueError('Explicit frozen default development criteria required')
    template=fit['profiles'];identity=template['execution_context_sha256']
    provider=NativeContextProvider(template,identity,allow_cpu_fixture=True)
    lifecycle=LifecycleProvider(fit['lifecycle'],identity,allow_cpu_fixture=True)
    training_runs={r['run_id'] for r in fit['normalized_runs']}
    val_ids=[b['raw']['run_id'] for b in bundles]
    if training_replay:
        if (len(set(val_ids))!=len(val_ids) or not set(val_ids)<=training_runs
                or not {b['source'] for b in bundles}<=set(fit['source_receipts'])):
            raise ValueError('Training replay must use the original fitted training receipts')
    elif (len(set(val_ids))!=len(val_ids) or set(val_ids)&training_runs
              or set(fit['source_receipts'])&{b['source'] for b in bundles}):
        raise ValueError('Independent nontraining validation run receipts required')
    reports=[];all_latency=[];all_work=[];reasons=[]
    for bundle in bundles:
        if validate_capture(bundle,allowed_splits=('training',) if training_replay else ('development_validation',))!=identity:
            raise ValueError('Frozen predictor and validation execution mismatch')
        raw=bundle['raw'];alias_family=bundle['context']['resident_aliases']
        from .native_order import request_orders
        orders=request_orders(raw)
        if raw['kind']!=template['provenance']['kind']:
            raise ValueError('Validation cannot mix CPU fixtures with physical training evidence')
        aliases={a:dict(family=f,supported_tasks=sorted({p['task'] for p in provider.profiles.values() if p['family']==f}))
                 for a,f in alias_family.items()}
        lookup={(p['family'],p['task'],p['pool_id']):p for p in template['profiles']}
        spans=[];work_errors=[];unsupported=[]
        for row in raw['request_records']:
            profile=lookup.get((alias_family[row['model']],row['task'],row['pool_id']))
            if profile is None:raise ValueError('Validation input is outside frozen input domain')
            layout=stage_layout(row)
            if [(s['id'],s['kind']) for s in layout]!=[(s['id'],s['kind']) for s in profile['segments']]:
                raise ValueError('Validation native stage layout differs from frozen template')
            for event in row['execution_events']:
                if (event.get('request_id')!=row['request_id'] or type(event.get('received_time_ns')) is not int
                        or event['received_time_ns']<event['perf_ns']):
                    raise ValueError('Validation event identity/common clock visibility mismatch')
            response=row['worker_response'];ready=row['previous_handler_next_request_ready']
            if (row['completion_time_ns']!=row['response_reader_event_enqueue_time_ns']
                    or row['completion_time_ns']<response['end_perf_ns']
                    or ready.get('request_id')!=row['request_id']
                    or ready.get('clock')!='host_perf_counter_ns'
                    or ready.get('boundary')!='previous_handler_next_request_ready'
                    or ready.get('scope')!='after_response_flush_before_next_read'
                    or not response['end_perf_ns']<=ready['perf_ns']<=ready['received_time_ns']):
                raise ValueError('Validation lifecycle boundary mismatch')
            for s in layout:
                spans.append(dict(alias=row['model'],profile_id=profile['id'],segment_id=s['id'],
                    start_ns=s['start_ns'],end_ns=s['end_ns'],kind=s['kind'],request_id=row['request_id']))
        window=[raw['observation_start_ns'],raw['observation_end_ns']]
        by_alias=defaultdict(list)
        for span in spans:
            if not window[0]<=span['start_ns']<span['end_ns']<=window[1]:
                raise ValueError('Validation handler outside observation window')
            by_alias[span['alias']].append(span)
        for owned in by_alias.values():
            owned.sort(key=lambda s:s['start_ns'])
            if any(a['end_ns']>b['start_ns'] for a,b in zip(owned,owned[1:])):
                raise ValueError('Overlapping validation work on one owner')
        cuts={*window,*[s[k] for s in spans for k in ('start_ns','end_ns')]}
        temporal=provider.native_law.temporal_progress
        if temporal is not None:
            age_edges=temporal.spec['age_edges']
            for span in spans:
                if span['kind']!='gpu':continue
                profile=provider.profiles[span['profile_id']]
                segment=next(s for s in profile['segments'] if s['id']==span['segment_id'])
                baseline=segment['duration_ms']['mean']
                if provider.native_order_law is not None:
                    baseline*=provider.native_order_law.factor(profile,segment,orders[span['request_id']])
                span['native_baseline_ms']=baseline
                for edge in age_edges[1:]:
                    at=span['start_ns']+round(edge*baseline*1e6)
                    if at<span['end_ns']:cuts.add(at)
        cuts=sorted(cuts)
        intervals=[];predicted_energy=0.;resources=provider.native_law.spec.get('resources')
        causal_energy=provider.requires_energy_history
        initial_energy_state=None
        if causal_energy:
            prefix=[s['end_ns'] for row in raw.get('priming_request_records',[])
                    for s in stage_layout(row) if s['kind']=='gpu']
            if prefix and max(prefix)<=window[0]:initial_energy_state=max(prefix)
        energy_supported=resources is not None or causal_energy
        changes=defaultdict(list)
        for span in spans:
            changes[span['start_ns']].append(('start',span));changes[span['end_ns']].append(('end',span))
        executing={};native_intervals=defaultdict(list)
        for left,right in zip(cuts,cuts[1:]):
            for kind,span in sorted(changes[left],key=lambda pair:pair[0]):
                if kind=='end':
                    if executing.pop(span['alias']) is not span:raise ValueError('Validation owner transition mismatch')
                else:
                    if span['alias'] in executing:raise ValueError('Validation owner overlap')
                    executing[span['alias']]=span
            members={s['alias']:dict(profile_id=s['profile_id'],segment_id=s['segment_id']) for s in executing.values()}
            if temporal is not None:
                for span in executing.values():
                    if span['kind']=='gpu':
                        members[span['alias']].update(native_age_ms=(left-span['start_ns'])/1e6,
                            native_baseline_ms=span['native_baseline_ms'])
            intervals.append((left,right,members))
            for span in executing.values():
                if span['kind']=='gpu':native_intervals[span['request_id'],span['segment_id']].append((left,right,members))
            if resources is not None and not causal_energy:
                x=counts(members,provider.profiles,provider.native_law.names)
                if not within_support(x,resources['support_vertices']):energy_supported=False
                else:predicted_energy+=(right-left)/1e9*float(design(x,resources['terms'])@np.array(resources['coefficients']))
        if causal_energy:
            try:
                predicted_energy,_=provider.idle_age_energy.integrate(
                    [(left,right,counts(members,provider.profiles,provider.native_law.names))
                     for left,right,members in intervals],initial_energy_state)
            except UnknownEvidence:
                energy_supported=False
        for target in spans:
            if target['kind']!='gpu':continue
            work=0.;known=True
            baseline=next(s for s in provider.profiles[target['profile_id']]['segments'] if s['id']==target['segment_id'])['duration_ms']['mean']
            if provider.native_order_law is not None:
                baseline*=provider.native_order_law.factor(provider.profiles[target['profile_id']],
                    dict(id=target['segment_id']),orders[target['request_id']])
            for left,right,members in native_intervals[target['request_id'],target['segment_id']]:
                try:work+=(right-left)/1e6/baseline*provider.native_law.progress(target['alias'],members[target['alias']],members)
                except UnknownEvidence as exc:
                    known=False;unsupported.append(str(exc))
            if known:work_errors.append(abs(work-1.))
        model=FullHorizonModel(aliases,provider,lifecycle_provider=lifecycle)
        rows={r['request_id']:r for r in raw['request_records']};seen=set();latency_errors=[];batch_results=[]
        fifo_batch_energy=0.;actual_batch_duration_ns=0;fifo_energy_supported=resources is not None or causal_energy
        energy_carry=initial_energy_state;energy_clock=window[0];previous_actual_end=window[0]
        if len(rows)!=len(raw['request_records']):raise ValueError('Unique validation request ids required')
        for batch in raw['batches']:
            observed=[rows[rid] for rid in batch['request_ids']]
            if seen&set(batch['request_ids']):raise ValueError('Repeated validation batch request')
            seen.update(batch['request_ids']);now=batch['start_ns']
            actual_end=batch.get('end_ns')
            if actual_end is None:
                actual_end=max([now+1,*[max(r['completion_time_ns'],r['previous_handler_next_request_ready']['perf_ns']) for r in observed]])
            if not window[0]<=now<actual_end<=window[1]:raise ValueError('Controlled batch outside capture energy window')
            actual_batch_duration_ns+=actual_end-now
            replay=[dict(request_id=r['request_id'],task=r['task'],pool_id=r['pool_id'],
                         arrival_ns=r['scheduled_time_ns'],alias=r['model']) for r in observed]
            if provider.native_order_law is not None:
                for row in replay:row['native_order']=orders[row['request_id']]
            queues=defaultdict(list);future=[]
            for row in replay:
                if row['arrival_ns']<now:raise ValueError('Validation arrival precedes its controlled batch')
                if row['arrival_ns']==now:queues[row['alias']].append(row)
                else:future.append(row)
            # A declared idle-duration job is a real controlled observation
            # window, even when no requests arrive. Its energy cannot disappear.
            deadline=max([now,batch.get('arrival_end_ns') or now,*[r['arrival_ns'] for r in replay]])+1
            try:
                batch_energy_state=None
                if causal_energy:
                    # Service resets remain conditional on each declared batch.
                    # Energy history never resets: preserve fixed controller gaps
                    # and rebase predicted idle age to the next batch's clock.
                    gap=now-previous_actual_end
                    if gap<0:raise ValueError('Overlapping controlled batches')
                    if gap:
                        gap_joules,energy_carry=provider.energy_interval({},energy_clock,energy_clock+gap,energy_carry)
                        fifo_batch_energy+=gap_joules;energy_clock+=gap
                    batch_energy_state=now-(energy_clock-energy_carry) if energy_carry is not None else None
                forecast=model.forecast(dict(queues),now,deadline,future_arrivals=future,
                    route_future=lambda row,pending,at_ns:row['alias'],drain_deadline_ns=deadline+120_000_000_000,
                    require_energy=False,require_memory=False,require_episode_energy=True,
                    energy_state=batch_energy_state)
                if forecast['incomplete_ids']:raise UnknownEvidence('INCOMPLETE_VALIDATION_FIFO_REPLAY')
                errors=[]
                for row in observed:
                    prediction=forecast['finishes'][row['request_id']]
                    actual_ms=(row['completion_time_ns']-row['scheduled_time_ns'])/1e6
                    predicted_ms=(prediction['finish_ns']-row['scheduled_time_ns'])/1e6
                    error=abs(predicted_ms-actual_ms)/actual_ms;latency_errors.append(error)
                    errors.append(dict(request_id=row['request_id'],observed_e2e_ms=actual_ms,predicted_e2e_ms=predicted_ms,
                                       relative_error=error))
                predicted_batch_energy=forecast['through_drain_board_energy_joules']
                if predicted_batch_energy is None:fifo_energy_supported=False
                else:fifo_batch_energy+=predicted_batch_energy
                if causal_energy:
                    energy_clock=forecast['drain_finish_ns'];energy_carry=forecast['final_energy_state']
                batch_results.append(dict(id=batch['id'],status='predicted',requests=errors,
                    predicted_batch_energy_joules=predicted_batch_energy,
                    predicted_batch_duration_ms=(forecast['drain_finish_ns']-now)/1e6))
            except UnknownEvidence as exc:
                fifo_energy_supported=False
                unsupported.append(str(exc));batch_results.append(dict(id=batch['id'],status='UNKNOWN',reason=str(exc)))
            previous_actual_end=actual_end
        if seen!=set(rows):raise ValueError('Validation batches do not cover full observed cohort')
        measured_energy=integrate_power(raw['board_samples'],*window,max_gap_seconds=.25)
        if measured_energy is None:raise ValueError('Independent validation energy lacks telemetry coverage')
        energy_error=abs(predicted_energy-measured_energy)/measured_energy if energy_supported and measured_energy>0 else None
        idle_ns=window[1]-window[0]-actual_batch_duration_ns
        if idle_ns<0:raise ValueError('Controlled batches overlap in capture energy window')
        fifo_energy=None
        if fifo_energy_supported and causal_energy:
            try:
                tail=window[1]-previous_actual_end
                if tail:
                    tail_joules,_=provider.energy_interval({},energy_clock,energy_clock+tail,energy_carry)
                    fifo_batch_energy+=tail_joules
                fifo_energy=fifo_batch_energy
            except UnknownEvidence:
                fifo_energy_supported=False
        elif fifo_energy_supported and resources is not None:
            idle=[0.]*len(provider.native_law.names)
            if within_support(idle,resources['support_vertices']):
                idle_watts=float(design(idle,resources['terms'])@np.array(resources['coefficients']))
                fifo_energy=fifo_batch_energy+idle_watts*idle_ns/1e9
        fifo_energy_error=abs(fifo_energy-measured_energy)/measured_energy if fifo_energy is not None and measured_energy>0 else None
        if unsupported:reasons.append(raw['run_id']+':UNSUPPORTED_CONTEXT_OR_REPLAY')
        if energy_error is None or energy_error>criteria['maximum_integrated_energy_relative_error']:
            reasons.append(raw['run_id']+':INTEGRATED_ENERGY_ERROR_OR_UNKNOWN')
        if fifo_energy_error is None or fifo_energy_error>criteria['maximum_integrated_energy_relative_error']:
            reasons.append(raw['run_id']+':FIFO_INTEGRATED_ENERGY_ERROR_OR_UNKNOWN')
        reports.append(dict(run_id=raw['run_id'],source=bundle['source'],latency_errors=metrics(latency_errors),
            native_work_errors=metrics(work_errors),unsupported=sorted(set(unsupported)),batches=batch_results,
            energy=dict(observed_joules=measured_energy,predicted_joules=predicted_energy if energy_supported else None,
                        relative_error=energy_error,scope='heldout actual stage exposure; separate from FIFO timing'),
            fifo_energy=dict(observed_joules=measured_energy,predicted_joules=fifo_energy,relative_error=fifo_energy_error,
                scope=('autonomous conditional FIFO batches with continuous predicted idle-age history and fixed controller gaps; observed priming prefix only'
                       if causal_energy else 'autonomous FIFO batch energy plus measured controller idle gaps; independent batch resets; no observed stage exposure'))))
        all_latency.extend(latency_errors);all_work.extend(work_errors)
    latency,work=metrics(all_latency),metrics(all_work)
    if len(bundles)<criteria['minimum_independent_runs']:reasons.append('INSUFFICIENT_INDEPENDENT_VALIDATION_RUNS')
    if (not all_latency or latency['mean']>criteria['maximum_mean_e2e_relative_error']
            or latency['p95']>criteria['maximum_p95_e2e_relative_error']):reasons.append('FIFO_E2E_ERROR_OR_UNKNOWN')
    if not all_work or work['p95']>criteria['maximum_p95_native_work_error']:reasons.append('NATIVE_WORK_ERROR_OR_UNKNOWN')
    return dict(schema='gsdeploy.native_training_replay.v1' if training_replay else 'gsdeploy.native_activity_development_validation.v1',
        fit_sha256=digest(fit),criteria=criteria,training_replay=training_replay,
        development_accuracy_passed=bool(not training_replay and not reasons),admission_enabled=False,formal_utility_qualified=False,
        **(dict(training_replay_accuracy_within_threshold=not reasons) if training_replay else {}),
        independent_run_ids=val_ids,latency_errors=latency,native_work_errors=work,reasons=reasons,runs=reports)
