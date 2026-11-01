"""Read-only heldout flow diagnosis with frozen coefficients, never fitting.

Observed-context work reconstruction uses future *observations* retrospectively.
It is neither an online prediction nor a counterfactual intervention. Its purpose
is to separate a conditional-service-law mismatch from trajectory feedback.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
from pathlib import Path
import statistics

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting import FullHorizonModel
from .background_statistical import BackgroundStatisticalProvider, family_task
from .mechanism_audit import receipt, checked
from .statistical_profiles import full_count
from .validation import errors, percentile


SCHEMA = 'gsdeploy.read_only_flow_forensics.v1'


def read(path):
    return json.loads(Path(path).read_text())


def stats(values):
    values = list(values)
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'p95': percentile(values) if values else None, 'max': max(values) if values else None}


def period(timestamp, origin, horizon_ns):
    relative = (timestamp-origin)/horizon_ns
    return 'early' if relative < 1/3 else 'middle' if relative < 2/3 else 'late' if relative < 1 else 'drain'


def age_bucket(age_ms):
    return '0..50ms' if age_ms < 50 else '50..200ms' if age_ms < 200 else '200..1000ms' if age_ms < 1000 else '>=1000ms'


def request_groups(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        for label in ('all', 'task:'+row['task'], 'alias:'+row['alias'], 'input:'+row['task']+':'+row['pool_id'],
                      'period:'+row['period'], 'worker_request_ordinal:'+row['ordinal_bucket']):
            groups[label].append(row)
    result = {}
    for label, selected in sorted(groups.items()):
        result[label] = {'requests': len(selected),
            'forecast_service_error': errors([(r['forecast_service_ms'], r['actual_service_ms']) for r in selected if r['forecast_service_ms'] is not None]),
            'observed_context_proxy_error': errors([(r['observed_context_service_proxy_ms'], r['actual_service_ms']) for r in selected if r['observed_context_service_proxy_ms'] is not None]),
            'handler_only_context_proxy_error': errors([(r['handler_context_service_proxy_ms'], r['actual_service_ms']) for r in selected if r['handler_context_service_proxy_ms'] is not None]),
            'parent_tail_ms': stats(r['parent_tail_ms'] for r in selected if r['parent_tail_ms'] is not None),
            'forecast_start_error_ms': stats(r['forecast_start_error_ms'] for r in selected if r['forecast_start_error_ms'] is not None)}
    return result


def phase_groups(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        for label in ('all', 'phase:'+row['phase'], 'task:'+row['task'], 'alias:'+row['alias'],
                      'task_phase:'+row['task']+':'+row['phase'], 'input_phase:'+row['task']+':'+row['pool_id']+':'+row['phase'],
                      'period_phase:'+row['period']+':'+row['phase']):
            groups[label].append(row)
    result = {}
    for label, selected in sorted(groups.items()):
        known = [r for r in selected if r['reconstructed_work'] is not None]
        result[label] = {'phases': len(selected), 'known': len(known),
            'reconstructed_work': stats(r['reconstructed_work'] for r in known),
            'mean_absolute_work_residual': statistics.mean(abs(r['reconstructed_work']-1) for r in known) if known else None,
            'observed_duration_ms': stats(r['actual_ms'] for r in selected),
            'forecast_phase_duration_error': errors([(r['simulated_ms'],r['actual_ms']) for r in selected if r['simulated_ms'] is not None]),
            'direct_context_duration_fraction': sum(r['direct_context_ms'] for r in selected)/sum(r['actual_ms'] for r in selected),
            'unknown_duration_ms': sum(r['unknown_context_ms'] for r in selected)}
    return result


def observed_reconstruction(trace, provider, aliases, *, origin_ns, horizon_ns, forecast=None,
                            simulated_phase_ms=None, actual_metadata=None):
    """Integrate frozen d(work)/dt on actual complete semantic wall intervals.

    Every complete observed phase has nominal work 1. A value above 1 means
    the frozen law predicts it should have completed sooner on that observed
    trajectory. No actual within-phase progress or GPU utilization is observed.
    """
    forecast, simulated_phase_ms, actual_metadata = forecast or {}, simulated_phase_ms or {}, actual_metadata or {}
    requests = trace['requests']
    if len({r['request_id'] for r in requests}) != len(requests):
        raise ValueError('Unique scoped observed request IDs required')
    phases, segments, request_info = {}, [], {}
    plans = {}
    handler_work = collections.defaultdict(float)
    handler_known = collections.defaultdict(lambda: True)
    for row in requests:
        rid = row['request_id']
        native = row.get('native_request_id', rid)
        request_info[rid] = row
        try:
            plans[rid] = provider.plan({'alias':row['alias'],'task':row['task'],'pool_id':row['pool_id']},aliases,'mean')
        except UnknownEvidence:
            plans[rid] = None
        cursor = row['handler_start_ns']
        for segment in row['segments']:
            if not cursor == segment['start_ns'] < segment['end_ns'] <= row['handler_end_ns']:
                raise ValueError('Observed semantic phases must continuously cover each handler')
            cursor = segment['end_ns']
            phase = segment['phase']
            key = rid,phase
            if key in phases:
                raise ValueError('Repeated semantic phase in one request')
            phases[key] = {'request_id':rid,'native_request_id':native,'alias':row['alias'],'family':row['family'],
                'task':row['task'],'pool_id':row['pool_id'],'phase':phase,'start_ns':segment['start_ns'],'end_ns':segment['end_ns'],
                'period':period(segment['start_ns'],origin_ns,horizon_ns),
                'actual_ms':(segment['end_ns']-segment['start_ns'])/1e6,'work_sum':0.,'known_context_ms':0.,
                'direct_context_ms':0.,'interpolated_context_ms':0.,'unknown_context_ms':0.,
                'unknown_reasons':collections.Counter(),'exposures':collections.defaultdict(float),
                'simulated_ms':simulated_phase_ms.get((native,phase))}
            segments.append((row,segment))
        if cursor != row['handler_end_ns']:
            raise ValueError('Observed semantic phases leave an uncovered handler tail')
    cuts = sorted({s[k] for _,s in segments for k in ('start_ns','end_ns')})
    interval_count = 0
    for left,right in zip(cuts,cuts[1:]):
        active_observed = [(row,segment) for row,segment in segments if segment['start_ns']<=left and right<=segment['end_ns']]
        if not active_observed:
            continue
        if len({r['alias'] for r,_ in active_observed}) != len(active_observed):
            raise ValueError('Observed simultaneous handlers on one native alias')
        interval_count += 1
        active = {}
        for row,segment in active_observed:
            plan = plans[row['request_id']]
            active[row['alias']] = {'request_id':row['request_id'],'alias':row['alias'],'task':row['task'],
                'pool_id':row['pool_id'],'profile_id':plan['profile_id'] if plan else None,'segment_id':segment['phase']}
        full_members = [{'family':family_task(row['family'],row['task'])} for row,_ in active_observed]
        full_key,_ = full_count(full_members)
        public_counts = collections.Counter((row['family'],row['task']) for row,_ in active_observed)
        context_label = json.dumps([{'family':f,'task':t,'count':n} for (f,t),n in sorted(public_counts.items())],separators=(',',':'))
        duration_ms = (right-left)/1e6
        for row,segment in active_observed:
            rid, phase = row['request_id'],segment['phase']
            record = phases[(rid,phase)]
            value = provider.service(active[row['alias']],active,aliases,'mean') if plans[rid] else {'known':False,'reason':'UNKNOWN_INPUT_PLAN'}
            service = value.get('service_ms')
            if value.get('known') is True and type(service) in (int,float) and math.isfinite(service) and service>0:
                record['work_sum'] += duration_ms/service
                record['known_context_ms'] += duration_ms
                record['interpolated_context_ms' if value.get('interpolated') is True else 'direct_context_ms'] += duration_ms
            else:
                record['unknown_context_ms'] += duration_ms
                record['unknown_reasons'][value.get('reason','UNKNOWN')] += 1
            age = ((left+right)/2-row['handler_start_ns'])/1e6
            for label in ('active_count:'+str(len(active)),'handler_age:'+age_bucket(age),
                          'interval_period:'+period((left+right)//2,origin_ns,horizon_ns),
                          'full_task_context:'+context_label):
                record['exposures'][label] += duration_ms
            baseline = sum(s['duration_ms'] for s in plans[rid]['segments']) if plans[rid] else None
            base = getattr(provider,'base',None)
            target = family_task(row['family'],row['task']),row['task'],row['pool_id']
            speed = base.handler.get(target,{}).get('progress',{}).get(full_key) if base is not None else None
            if baseline and type(speed) in (int,float) and speed>0:
                handler_work[rid] += duration_ms/baseline*speed
            else:
                handler_known[rid] = False
    exposure_groups = collections.defaultdict(lambda:{'duration_ms':0.,'known_duration_ms':0.,'weighted_work_residual':0.,'phase_keys':set()})
    phase_rows = []
    for key,record in phases.items():
        work = record.pop('work_sum')
        known = record['unknown_context_ms'] < 1e-8 and abs(record['known_context_ms']-record['actual_ms']) < 1e-6
        record['reconstructed_work'] = work if known else None
        record['observed_context_service_proxy_ms'] = record['actual_ms']/work if known and work>0 else None
        record['unknown_reasons'] = dict(record['unknown_reasons'])
        for label,duration in record['exposures'].items():
            group = exposure_groups[label]
            group['duration_ms'] += duration
            group['phase_keys'].add(key)
            if known:
                group['known_duration_ms'] += duration
                group['weighted_work_residual'] += duration*(work-1)
        record['exposures'] = dict(record['exposures'])
        phase_rows.append(record)
    for label,value in exposure_groups.items():
        value['phase_count'] = len(value.pop('phase_keys'))
        denominator = value['known_duration_ms']
        value['phase_work_residual_weighted_by_exposure'] = value.pop('weighted_work_residual')/denominator if denominator else None
    ordinals, busy = {}, collections.defaultdict(float)
    request_rows = []
    for row in sorted(requests,key=lambda r:r['handler_start_ns']):
        rid,alias = row['request_id'],row['alias']
        native = row.get('native_request_id',rid)
        ordinal = ordinals.get(alias,0)+1
        ordinals[alias] = ordinal
        actual = (row['handler_end_ns']-row['handler_start_ns'])/1e6
        pieces = [p for p in phase_rows if p['request_id']==rid]
        complete = all(p['observed_context_service_proxy_ms'] is not None for p in pieces)
        pred = forecast.get(native,{})
        metadata = actual_metadata.get(native,{})
        parent = row.get('parent_receipt_ns',metadata.get('completion_time_ns'))
        request_rows.append({'request_id':rid,'native_request_id':native,'alias':alias,'family':row['family'],
            'task':row['task'],'pool_id':row['pool_id'],'period':period(row['handler_start_ns'],origin_ns,horizon_ns),
            'worker_request_ordinal':ordinal,'ordinal_bucket':'1..5' if ordinal<=5 else '6..20' if ordinal<=20 else '>20',
            'prior_worker_busy_ms':busy[alias],'measurement_execution_age_ms':(row['handler_start_ns']-origin_ns)/1e6,
            'actual_service_ms':actual,'forecast_service_ms':pred.get('service_ms'),
            'observed_context_service_proxy_ms':sum(p['observed_context_service_proxy_ms'] for p in pieces) if complete else None,
            'handler_reconstructed_work':handler_work[rid] if handler_known[rid] else None,
            'handler_context_service_proxy_ms':actual/handler_work[rid] if handler_known[rid] and handler_work[rid]>0 else None,
            'parent_tail_ms':(parent-row['handler_end_ns'])/1e6 if parent is not None else None,
            'forecast_start_error_ms':(pred['start_ns']-(row['handler_start_ns']-origin_ns))/1e6 if pred.get('start_ns') is not None else None,
            'generated_tokens_posthoc':metadata.get('worker_response',{}).get('generated_tokens'),
            'input_tokens_posthoc':metadata.get('worker_response',{}).get('input_tokens'),
            'forward_count_posthoc':len(metadata.get('worker_response',{}).get('semantic_forward_observations',{}).get('records',[]))})
        busy[alias] += actual
    return {'requests':request_rows,'phases':phase_rows,'request_groups':request_groups(request_rows),
        'phase_groups':phase_groups(phase_rows),'exposure_groups':dict(sorted(exposure_groups.items())),
        'interval_count':interval_count,'phase_reconstruction_coverage':{
            'requested':len(phase_rows),'known':sum(p['reconstructed_work'] is not None for p in phase_rows),
            'known_duration_fraction':sum(p['known_context_ms'] for p in phase_rows)/sum(p['actual_ms'] for p in phase_rows),
            'interpolated_duration_fraction':sum(p['interpolated_context_ms'] for p in phase_rows)/sum(p['actual_ms'] for p in phase_rows)},
        'interpretation':{
            'work':'Integral observed_dt / frozen_conditional_phase_service_ms; nominal complete phase work is 1. Above 1 means model would finish sooner on this observed path.',
            'context_proxy':'Observed phase duration divided by reconstructed work, summed per request. Uses future observed contexts; retrospective diagnostic only.',
            'exposure_groups':'Associates whole-phase work residual with active-count/age exposure duration. Does not observe instantaneous progress within a phase.',
            'causality':'Forecast-vs-observed-context differences are not a uniquely identified causal decomposition; the trajectories and phase positions interact.'}}


def reconstruct_work(traces, provider):
    """Pure ordered request/phase work vectors; never fit or relabel a split.

    W is a residual of frozen mixed-wall progress on observed contexts. It is
    not an independently identified physical work amount. Callers must retain
    every split and unknown flag before selecting training-only residuals.
    """
    vectors = []
    for trace in traces:
        aliases = {}
        for row in trace['requests']:
            alias = row['alias']
            if alias in aliases and aliases[alias]['family'] != row['family']:
                raise ValueError('Alias family changes inside one trace')
            aliases.setdefault(alias,{'family':row['family'],'supported_tasks':[]})
            if row['task'] not in aliases[alias]['supported_tasks']:
                aliases[alias]['supported_tasks'].append(row['task'])
        start = min(r['handler_start_ns'] for r in trace['requests'])
        end = max(r['handler_end_ns'] for r in trace['requests'])
        reconstructed = observed_reconstruction(trace,provider,aliases,origin_ns=start,horizon_ns=end-start)
        by_request = collections.defaultdict(list)
        for phase in reconstructed['phases']:
            by_request[phase['request_id']].append(phase)
        for row in trace['requests']:
            phases = sorted(by_request[row['request_id']],key=lambda s:s['start_ns'])
            known = all(p['reconstructed_work'] is not None for p in phases)
            vectors.append({'request_id':row['request_id'],'native_request_id':row.get('native_request_id',row['request_id']),
                'trace_id':trace.get('id'),'split':trace['split'],
                'not_for_training':trace['split']!='training' or trace.get('not_for_training') is True,
                'source_receipts':copy.deepcopy(trace.get('sources',[])),
                'family':row['family'],'alias':row['alias'],'task':row['task'],'pool_id':row['pool_id'],
                'phase_order':[p['phase'] for p in phases],
                'work_vector':[p['reconstructed_work'] for p in phases],
                'work_by_phase':{p['phase']:p['reconstructed_work'] for p in phases},
                'complete_known_vector':known,
                'phase_context_diagnostics':[{'phase':p['phase'],'direct_ms':p['direct_context_ms'],
                    'interpolated_ms':p['interpolated_context_ms'],'unknown_ms':p['unknown_context_ms'],
                    'unknown_reasons':p['unknown_reasons']} for p in phases]})
    if len({(r['trace_id'],r['request_id']) for r in vectors}) != len(vectors):
        raise ValueError('Duplicate trace/request residual vector identity')
    return {'schema':'gsdeploy.frozen_model_observed_joint_phase_work.v1','requests':vectors,
            'fitting_performed':False,'GPU_started':False,
            'training_eligible_requests':sum(not r['not_for_training'] and r['complete_known_vector'] for r in vectors),
            'definition':'W_phase = sum(observed_interval_ms / frozen_conditional_phase_service_ms); joint order/correlation retained per request',
            'limits':['Mixed wall residual, not physical GPU work','Observed future boundaries are retrospective evidence only',
                      'Validation vectors remain not_for_training; unknown contexts are not imputed']}


def replay(sealed,plan,candidate,provider):
    queues,future = {a:[] for a in plan['aliases']},[]
    for item in plan['trace']:
        row = {k:item[k] for k in ('request_id','task','pool_id')}
        row['arrival_ns'] = item['offset_ns']
        (queues[candidate['routes'][row['request_id']]] if row['arrival_ns']==0 else future).append(row)
    end = round(plan['window_seconds']*1e9)
    result = FullHorizonModel(plan['aliases'],provider,resources=candidate.get('resource_configuration'),
        max_requests=plan.get('max_forecast_requests',20000)).forecast(queues,0,end,future_arrivals=future,
        route_future=lambda row,pending,now:candidate['routes'][row['request_id']],require_energy=False,require_memory=False,
        drain_deadline_ns=end+round(plan['drain_seconds']*1e9))
    if result['incomplete_ids']:
        raise ValueError('Frozen replay is incomplete')
    frozen = {r['request_id']:r for r in sealed['requests']}
    if set(frozen) != set(result['finishes']):
        raise ValueError('Frozen replay request identity mismatch')
    forecast,phase_ms = {},collections.defaultdict(float)
    for rid,finish in result['finishes'].items():
        duration = (finish['finish_ns']-finish['start_ns'])/1e6
        if (finish['finish_ns'] != frozen[rid]['finish_ns'] or abs(duration-frozen[rid]['service_ms'])>1e-8):
            raise ValueError('Retrospective replay differs from original sealed forecast')
        forecast[rid] = {**frozen[rid],'start_ns':finish['start_ns']}
    for interval in result['transitions']:
        for alias,rid in interval['active_requests'].items():
            phase_ms[(rid,interval['segments'][alias])] += (interval['until_ns']-interval['time_ns'])/1e6
    return forecast,dict(phase_ms),result


def timeline_disagreement(trace,replayed,origin,end):
    actual = [(r['alias'],r.get('native_request_id',r['request_id']),s['phase'],s['start_ns']-origin,s['end_ns']-origin)
              for r in trace['requests'] for s in r['segments']]
    simulated = [(alias,rid,t['segments'][alias],t['time_ns'],t['until_ns'])
                 for t in replayed['transitions'] for alias,rid in t['active_requests'].items()]
    cuts = sorted({0,end,*[max(0,min(end,e)) for row in actual+simulated for e in row[3:]]})
    totals = collections.defaultdict(float)
    for left,right in zip(cuts,cuts[1:]):
        if left==right:
            continue
        a = {alias:(rid,phase) for alias,rid,phase,start,stop in actual if start<=left and right<=stop}
        s = {alias:(rid,phase) for alias,rid,phase,start,stop in simulated if start<=left and right<=stop}
        for alias in set(a)|set(s):
            totals['union_active_alias_ms'] += (right-left)/1e6
            if a.get(alias) != s.get(alias):
                totals['request_or_phase_disagreement_ms'] += (right-left)/1e6
            if alias in a and alias in s and a[alias][0]==s[alias][0] and a[alias][1]!=s[alias][1]:
                totals['same_request_different_phase_ms'] += (right-left)/1e6
    denominator = totals['union_active_alias_ms']
    return {**totals,'absolute_time_state_disagreement_fraction':totals['request_or_phase_disagreement_ms']/denominator if denominator else None,
        'scope':'Absolute measurement time. Includes queue/start drift; not an isolated phase-classification error.'}


def utility_attribution(original_report,weights):
    rows = original_report['candidates']
    baseline = rows[0]
    result = []
    for row in rows[1:]:
        terms = {}
        for term,weight in weights.items():
            predicted = -weight*(row['prediction_metrics']['terms'][term]-baseline['prediction_metrics']['terms'][term])
            observed = -weight*(row['observed_metrics']['terms'][term]-baseline['observed_metrics']['terms'][term])
            terms[term] = {'predicted_delta_contribution':predicted,'observed_delta_contribution':observed,
                           'predicted_minus_observed':predicted-observed}
        result.append({'candidate_id':row['candidate_id'],'baseline_candidate_id':baseline['candidate_id'],'terms':terms})
    return result


def diagnose(sealed_path,raw_paths,output,*,original_report_path=None):
    sealed = read(sealed_path)
    if sealed.get('schema') != 'gsdeploy.sealed_native_flow_predictions.v1':
        raise ValueError('Original frozen full-flow predictions required')
    for ref in [*sealed['sources'].values(),*sealed['predictor_sources'],sealed['tool_source']]:
        checked(ref)
    plan = read(checked(sealed['sources']['plan']))
    document = read(checked(sealed['sources']['model']))
    from .flow_profiles import map_flow_collection
    candidates = {c['id']:c for c in plan['candidates']}
    predictions = {c['candidate_id']:c for c in sealed['candidates']}
    results = []
    sources = [receipt(sealed_path),receipt(__file__),receipt(Path(__file__).with_name('flow_profiles.py'))]
    for raw_path in raw_paths:
        raw = read(raw_path)
        mapped = map_flow_collection(raw_path,allow_validation_diagnostics=True)
        traces = mapped['traces']
        if len(traces)!=1 or traces[0]['split']!='validation':
            raise ValueError('Read-only diagnostics require one explicitly validation flow cohort')
        trace = traces[0]
        cid = raw['candidate_id']
        if raw['plan'] != sealed['sources']['plan'] or cid not in candidates:
            raise ValueError('Heldout raw belongs to a different original sealed plan')
        provider = BackgroundStatisticalProvider(document,candidates[cid]['execution_context_sha256'])
        forecast,phase_ms,replayed = replay(predictions[cid],plan,candidates[cid],provider)
        origin = raw['phase_start_ns']
        result = observed_reconstruction(trace,provider,plan['aliases'],origin_ns=origin,
            horizon_ns=round(plan['window_seconds']*1e9),forecast=forecast,simulated_phase_ms=phase_ms,
            actual_metadata={r['request_id']:r for r in raw['request_records']})
        result.update(candidate_id=cid,mapping_report=mapped['mapping_report'],frozen_replay_matches_sealed=True,
            absolute_timeline_disagreement=timeline_disagreement(trace,replayed,origin,round(plan['window_seconds']*1e9)))
        results.append(result)
        sources.append(receipt(raw_path))
    attribution = None
    if original_report_path:
        original = read(original_report_path)
        attribution = utility_attribution(original,plan['weights'])
        sources.append(receipt(original_report_path))
    report = {'schema':SCHEMA,'kind':'post_validation_read_only_diagnostics','GPU_started':False,'fitting_performed':False,
        'validation_used_for_training':False,'original_gate_overridden':False,'source_receipts':sources,
        'candidates':results,'six_term_original_report_attribution':attribution,
        'limits':['One pilot trace per routing candidate; no causal control over thermal/cache/history state',
                  'Actual semantic intervals are mixed host/CUDA/sampling/synchronization wall time, not measured kernel work',
                  'Observed future boundaries are used only for diagnosis, never as predictor inputs or evaluation replacements',
                  'No coefficients, acceptance thresholds, original predictions or original gate are changed',
                  'Within-phase execution-age groups use exposure attribution; instantaneous progress remains unobserved',
                  'Unknown contexts remain UNKNOWN; no extrapolation or counterfactual validity is manufactured']}
    output = Path(output)
    with output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sealed',required=True)
    parser.add_argument('--raw',action='append',required=True)
    parser.add_argument('--original-report')
    parser.add_argument('--output',required=True)
    args = parser.parse_args(argv)
    result = diagnose(args.sealed,args.raw,args.output,original_report_path=args.original_report)
    print(json.dumps({'output':str(Path(args.output).resolve()),'candidates':len(result['candidates']),
                      'fitting_performed':False,'GPU_started':False}))


if __name__ == '__main__':
    main()
