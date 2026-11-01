"""Frozen-v8 diagnostics of new training flows; no predictive fitting or GPU.

All eight prospectively training-labelled flows are required. Validation input
is refused by the strict mapper. Correlations/centering here are descriptive;
they do not export correction coefficients or turn training errors into tests.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
from pathlib import Path
import statistics

from .background_statistical import BackgroundStatisticalProvider
from .flow_forensics import observed_reconstruction, reconstruct_work, stats
from .flow_profiles import map_flow_collection
from .mechanism_audit import receipt, checked


SCHEMA = 'gsdeploy.frozen_v8_continuous_training_state_diagnostics.v1'
ROUTES = ('shared_balanced','shared_skewed','shared_spill_low','shared_spill_high')


def read(path):
    return json.loads(Path(path).read_text())


def correlation(xs, ys):
    if len(xs)!=len(ys) or len(xs)<3:
        return None
    mx,my = statistics.mean(xs),statistics.mean(ys)
    xx,yy = sum((x-mx)**2 for x in xs),sum((y-my)**2 for y in ys)
    if xx<=0 or yy<=0:
        return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(xx*yy)


def grouped(rows, keys):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def stratum_summaries(rows, keys):
    result=[]
    for key,selected in sorted(grouped(rows,keys).items()):
        known = [r for r in selected if r['work'] is not None]
        result.append({**dict(zip(keys,key)),'requested':len(selected),'known':len(known),
            'work':stats(r['work'] for r in known),
            'log_work_standard_deviation':statistics.stdev(math.log(r['work']) for r in known) if len(known)>=2 else None,
            'actual_ms':stats(r['actual_ms'] for r in selected),
            'mean_absolute_work_residual':statistics.mean(abs(r['work']-1) for r in known) if known else None,
            'interpolated_duration_fraction':sum(r['interpolated_ms'] for r in selected)/sum(r['actual_ms'] for r in selected)})
    return result


def paired_seed_comparisons(rows):
    """Match exact alias/task/input/phase before comparing two run means."""
    output=[]
    keys=('alias','task','pool_id','phase')
    for route,selected in sorted(grouped(rows,('candidate_id',)).items()):
        seeds=sorted({r['seed'] for r in selected})
        if len(seeds)!=2:
            raise ValueError('Exactly two independent training seeds per route required')
        by_seed={seed:grouped([r for r in selected if r['seed']==seed],keys) for seed in seeds}
        common=set(by_seed[seeds[0]])&set(by_seed[seeds[1]])
        strata=[]
        for key in sorted(common):
            values=[[r['work'] for r in by_seed[seed][key] if r['work'] is not None] for seed in seeds]
            if not all(values):
                continue
            means=[statistics.mean(v) for v in values]
            logs=[[math.log(v) for v in group] for group in values]
            log_means=[statistics.mean(v) for v in logs]
            within=[value-statistics.mean(group) for group in logs for value in group]
            pooled_sd=math.sqrt(sum(v*v for v in within)/max(1,len(within)-2))
            strata.append({**dict(zip(keys,key)),'seeds':seeds,'counts':list(map(len,values)),
                'mean_work':means,'second_minus_first':means[1]-means[0],
                'second_over_first':means[1]/means[0],
                'mean_log_work':log_means,'pooled_within_run_log_sd':pooled_sd,
                'log_mean_difference_over_pooled_within_sd':(log_means[1]-log_means[0])/pooled_sd if pooled_sd else None})
        output.append({'candidate_id':route[0],'seeds':seeds,'common_strata':len(common),
            'only_first_seed_strata':[dict(zip(keys,key)) for key in sorted(set(by_seed[seeds[0]])-common)],
            'only_second_seed_strata':[dict(zip(keys,key)) for key in sorted(set(by_seed[seeds[1]])-common)],
            'strata':strata})
    return output


def history_diagnostics(rows):
    results=[]
    keys=('seed','candidate_id','alias','task','pool_id','phase')
    for key,selected in sorted(grouped(rows,keys).items()):
        known=sorted((r for r in selected if r['work'] is not None),key=lambda r:r['worker_request_ordinal'])
        if len(known)<3:
            continue
        logs=[math.log(r['work']) for r in known]
        results.append({**dict(zip(keys,key)),'observations':len(known),
            'log_work_vs_prior_worker_busy_correlation':correlation(logs,[r['prior_worker_busy_ms'] for r in known]),
            'log_work_vs_worker_ordinal_correlation':correlation(logs,[r['worker_request_ordinal'] for r in known]),
            'log_work_vs_measurement_time_correlation':correlation(logs,[r['measurement_age_ms'] for r in known]),
            'lag1_same_input_log_work_correlation':correlation(logs[:-1],logs[1:]),
            'mean_worker_request_gap_between_same_input_observations':statistics.mean(b['worker_request_ordinal']-a['worker_request_ordinal'] for a,b in zip(known,known[1:]))})
    return results


def joint_phase_correlations(vectors):
    output=[]
    keys=('seed','candidate_id','alias','task','pool_id')
    for key,selected in sorted(grouped(vectors,keys).items()):
        phases=sorted({phase for r in selected for phase in r['phase_order']})
        pairs=[]
        for first,second in itertools.combinations(phases,2):
            complete=[r for r in selected if r['work_by_phase'].get(first) is not None and r['work_by_phase'].get(second) is not None]
            pairs.append({'first':first,'second':second,'observations':len(complete),
                'log_work_correlation':correlation([math.log(r['work_by_phase'][first]) for r in complete],
                                                 [math.log(r['work_by_phase'][second]) for r in complete])})
        output.append({**dict(zip(keys,key)),'requests':len(selected),'phase_pairs':pairs})
    return output


def block_diagnostics(rows):
    """Descriptive within-run input centering, not learned predictor effects."""
    output=[]
    keys=('seed','candidate_id','alias','task','phase')
    for key,selected in sorted(grouped(rows,keys).items()):
        known=sorted((r for r in selected if r['work'] is not None),key=lambda r:r['worker_request_ordinal'])
        inputs=grouped(known,('pool_id',))
        centers={pool:statistics.mean(math.log(r['work']) for r in group) for (pool,),group in inputs.items()}
        residual=[math.log(r['work'])-centers[r['pool_id']] for r in known]
        blocks=[residual[i:i+5] for i in range(0,len(residual),5)]
        output.append({**dict(zip(keys,key)),'observations':len(known),
            'lag1_input_centered_log_work_correlation':correlation(residual[:-1],residual[1:]),
            'block_size_requests':5,'nonoverlapping_block_mean_log_residuals':[statistics.mean(b) for b in blocks],
            'block_lengths':list(map(len,blocks)),
            'centering_scope':'Within this observed run/input only, descriptive removal of input composition; no predictive coefficient or validation result'})
    return output


def expected_raws(training_root):
    root=Path(training_root).resolve()
    plans=sorted(root.glob('seed-*/plan.json'))
    if len(plans)!=2:
        raise ValueError('Exactly two prospectively training plans required')
    expected=[]
    for path in plans:
        plan=read(path)
        if plan.get('phase')!='predictor_training' or plan.get('split')!='training' or set(c['id'] for c in plan['candidates'])!=set(ROUTES):
            raise ValueError('Exact predeclared two-seed/four-route training domain required')
        expected.extend(path.parent/'runs'/route/'raw.json' for route in ROUTES)
    if any(not p.exists() for p in expected):
        raise ValueError('All eight training raw files must be complete before state diagnosis')
    return expected


def diagnose(training_root, model_path, output):
    raw_paths=expected_raws(training_root)
    model=read(model_path)
    for ref in model['source_receipts']:
        checked(ref)
    provider=BackgroundStatisticalProvider(model,model['execution_context_sha256'])
    observations,vectors,runs,handlers,sources=[],[],[],[],[receipt(model_path),receipt(__file__)]
    for path in raw_paths:
        mapped=map_flow_collection(path)
        trace=mapped['traces'][0]
        if trace['split']!='training' or trace.get('not_for_training') or trace['execution_context_sha256']!=provider.execution_context_sha256:
            raise ValueError('New matching training-only trace required; validation is never allowed')
        raw=read(path);plan=read(raw['plan']['path']);seed,cid=plan['seed'],raw['candidate_id']
        origin=raw['phase_start_ns']
        result=observed_reconstruction(trace,provider,plan['aliases'],origin_ns=origin,
            horizon_ns=round(plan['window_seconds']*1e9),actual_metadata={r['request_id']:r for r in raw['request_records']})
        by_request={r['request_id']:r for r in result['requests']}
        history_by_request={}
        for alias,history in grouped(mapped['resource_evidence']['stage_history'],('alias',)).items():
            busy,previous=0.,None
            for item in sorted(history,key=lambda r:r['handler_start_ns']):
                history_by_request[item['request_id']]={
                    'prior_owned_handler_busy_ms':busy,
                    'previous_handler_task':previous['task'] if previous else 'none',
                    'previous_handler_gap_ms':(item['handler_start_ns']-previous['handler_end_ns'])/1e6 if previous else None,
                    'deployment_age_ms':(item['handler_start_ns']-raw['startup_begin_ns'])/1e6}
                busy+=(item['handler_end_ns']-item['handler_start_ns'])/1e6
                previous=item
        joint=reconstruct_work([trace],provider)
        for vector in joint['requests']:
            if vector['not_for_training']:
                raise ValueError('Validation vector cannot enter training diagnostics')
            vectors.append({**vector,'seed':seed,'candidate_id':cid,'frozen_model_sha256':model['model_sha256']})
        for phase in result['phases']:
            request=by_request[phase['request_id']]
            history=history_by_request[phase['request_id']]
            observations.append({'seed':seed,'candidate_id':cid,'request_id':phase['request_id'],
                'alias':phase['alias'],'task':phase['task'],'pool_id':phase['pool_id'],'phase':phase['phase'],
                'work':phase['reconstructed_work'],'actual_ms':phase['actual_ms'],'interpolated_ms':phase['interpolated_context_ms'],
                'unknown_ms':phase['unknown_context_ms'],'period':phase['period'],
                'prior_worker_busy_ms':history['prior_owned_handler_busy_ms'],
                'prior_measurement_worker_busy_ms':request['prior_worker_busy_ms'],
                'worker_request_ordinal':request['worker_request_ordinal'],
                'measurement_age_ms':request['measurement_execution_age_ms'],**history})
        for request in result['requests']:
            handlers.append({**request,'seed':seed,'candidate_id':cid,**history_by_request[request['request_id']]})
        runs.append({'seed':seed,'candidate_id':cid,'raw':receipt(path),'mapping_report':mapped['mapping_report'],
            'phase_reconstruction_coverage':result['phase_reconstruction_coverage'],
            'request_groups':result['request_groups'],'exposure_groups':result['exposure_groups']})
        sources.append(receipt(path))
    source_names=('flow_forensics.py','flow_profiles.py','background_statistical.py','statistical_profiles.py','semantic_profiles.py')
    sources.extend(receipt(Path(__file__).with_name(name)) for name in source_names)
    report={'schema':SCHEMA,'GPU_started':False,'predictive_fitting_performed':False,'v9_read_or_used':False,
        'frozen_reference_model_sha256':model['model_sha256'],'source_receipts':sources,'runs':runs,
        'phase_strata':stratum_summaries(observations,('seed','candidate_id','alias','task','pool_id','phase')),
        'period_phase_strata':stratum_summaries(observations,('seed','candidate_id','alias','task','pool_id','phase','period')),
        'previous_task_phase_strata':stratum_summaries(observations,('seed','candidate_id','alias','task','pool_id','phase','previous_handler_task')),
        'paired_same_route_seed_comparisons':paired_seed_comparisons(observations),
        'within_input_history_diagnostics':history_diagnostics(observations),
        'joint_phase_correlations':joint_phase_correlations(vectors),'serial_block_diagnostics':block_diagnostics(observations),
        'handler_observations':handlers,'phase_observations':observations,'joint_phase_work_vectors':vectors,
        'interpretation':{
            'new_continuous_coefficients':'Systematic frozen-reference phase means away from 1, repeated across seeds/input-matched strata, support recalibrating conditional means on new training before adding random residuals.',
            'request_joint_residual':'Substantial remaining within-input phase variation and joint phase dependence motivate request-level vectors; this diagnoses the old frozen model, not residuals after a new mean fit.',
            'run_or_block_residual':'Within-input history association, serial centered residual correlation, or persistent block mean shifts motivate an additional block/run hypothesis; neither correlation nor two runs establishes a causal state model.',
            'required_recheck':'Reconstruct residuals again under any newly fitted continuous mean model using training only, before choosing/calibrating scenario distributions.',
            'independence':'Two seeds are two runs per route; route-dependent contexts and chronological execution confound run effects. Eight routes/runs are not eight exchangeable repeats of one condition.',
            'correlations':'Descriptive Pearson correlations of log W only, no significance claims, fitted correction, or independence assumptions.',
            'history':'Busy/ordinal/time correlations are partly redundant and can reflect context exposure; no temperature/cache cause is inferred.',
            'evaluation':'A new independent v11 must test full queue, per-task P95, delta-U sign and rank; these training diagnostics cannot serve as that validation.'}}
    with Path(output).open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('training-root','model','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    report=diagnose(args.training_root,args.model,args.output)
    print(json.dumps({'runs':len(report['runs']),'requests':len(report['handler_observations']),
        'phase_vectors':len(report['joint_phase_work_vectors']),'predictive_fitting_performed':False,'GPU_started':False}))


if __name__=='__main__':
    main()
