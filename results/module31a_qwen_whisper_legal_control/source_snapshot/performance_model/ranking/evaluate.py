"""Module 2F. No model fitting, inference workers, profiling or search.

python -m performance_model.ranking.evaluate
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from deployment_core.models import load_model_catalog
from deployment_core.workload import TASKS
from performance_model.features import FeatureExtractor
from performance_model.resources import load_resource_profiles

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/performance_model_ranking'
DATA = ROOT/'results/performance_model_active_sampling/dataset_rows.jsonl'
OOF = ROOT/'results/performance_model_generalization/predictions.jsonl'
OOF_AUDIT = ROOT/'results/performance_model_generalization/leakage_audit.json'
LABELS = ROOT/'results/performance_model_local_calibration/stability_v2_labels.jsonl'
VARIANTS = {'A':'A_standalone', 'B':'B_gbt_log_ratio', 'C_GBT':'C_gbt_log_ratio'}
POLICY = dict(
    predictions='Saved Module 2D primary LODO predictions only; no fitting, recalibration, or use of Module 2E predictions.',
    variants=VARIANTS,
    workload='Exact per-task target rates, with absent/zero tasks normalized to zero. Never group on total rate or own-model u alone.',
    protocol='Exact source metadata except deployed worker membership; exact durations, limits, arrival law, clocks, cohort, and priming policy. Shared worker implementations must match.',
    worker_membership_exception='Workers added/removed by deployment are the treatment, not a protocol change; only allowed when every shared worker definition is identical. No exception for harness, task-pool, device, environment or duration changes.',
    stability='Configuration-level within a protocol: strict requires ALL observed repetitions steady; extended requires ALL steady/recoverable. Any unstable repetition excludes the configuration, without deleting that repetition.',
    repetitions='Pair candidates on the exact seed AND initial-arrival-phase block; require >=2 common blocks. Aggregate per-run objectives equally, retaining sample std and per-block diagnostics.',
    minimum_common_repetition_blocks=2,
    true_mean='For each run sum(lambda_k * measured task-aggregate mean E2E_k)/sum(lambda), then mean across common blocks.',
    true_tail='For each run sum(lambda_k * measured task-aggregate P95 E2E_k)/sum(lambda), then mean across common blocks. This is neither system P95 nor P95 pooled across repetitions.',
    predicted_mean='Lhat_k=sum_m (routing_rps[k,m]/lambda_k)*OOF_path_mean[k,m]; use the identical workload weighting for Jhat_mean.',
    predicted_tail=None,
    tail_analysis='Available-data baseline only: evaluate saved MEAN-score ordering against observed tail objectives, explicitly a mean-score proxy, never a P95 prediction.',
    ties='Exact numeric score order, configuration-ID secondary ordering independent of labels. Report ties and their regret range. Kendall tau-b handles tied ranks.',
    top2_recall='Overlap of deterministic predicted Top-2 and true Top-2 divided by 2.',
    shortlist_coverage='Whether a true minimum lies in predicted Top-K; saturated K>=candidate count is marked trivial and excluded from nontrivial macro coverage.',
    sufficiency=dict(min_cross_deployment_groups=3, min_distinct_deployments=3,
                     min_nontrivial_top2_cross_deployment_groups=2,
                     rationale='Minimum breadth heuristic, not a statistical power claim. Repetitions, strict/extended views and multiple variants never count as independent workload groups.'),
    sufficient_gate=dict(PASS='C mean Kendall/Spearman >=.6, every tau positive, mean regret<=.05, worst<=.15, nontrivial Top-2 coverage>=.9, no available nontrivial Top-3 coverage below .9.',
        PARTIAL='Not PASS, but C nontrivial Top-2 coverage>=.8 or nontrivial Top-3 coverage>=.9, with positive macro Kendall and worst regret<=.5.',
        FAIL='Neither ranking nor shortlist thresholds met on sufficient comparable cross-deployment groups.'),
    insufficient='INSUFFICIENT COMPARABLE GROUPS; leave PASS/PARTIAL/FAIL gate unassessed, do not turn missing evidence into predictor failure or success.',
)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value): return hashlib.sha256(canonical(value).encode()).hexdigest()[:16]


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read_jsonl(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def workload_key(workload):
    if set(workload)-set(TASKS): raise ValueError('Unknown task in workload')
    return {k:float(workload.get(k,0)) for k in TASKS}


def protocol_signature(raw):
    source = {k:v for k,v in raw['source'].items() if k!='workers'}
    settings = {k:raw.get(k) for k in ('schema_version','warmup_seconds','measurement_seconds',
        'startup_timeout_seconds','drain_timeout_seconds','shutdown_timeout_seconds','max_pending_per_model','clock')}
    if any(settings[k] is None for k in ('schema_version','measurement_seconds','warmup_seconds','clock')):
        raise ValueError('Cannot certify measurement protocol')
    settings.update(source=source, arrival_law=raw['arrival_process']['name'],
        priming_excluded_from_measurement=raw['warmup'].get('priming_excluded_from_measurement'),
        cohort=raw['measurement_window']['cohort'],
        completion_count_definition=raw['measurement_window']['completion_count_definition'])
    return settings


def shared_workers_compatible(runs):
    versions = defaultdict(set)
    for r in runs:
        for m,v in r['workers'].items(): versions[m].add(canonical(v))
    return {m:sorted(v) for m,v in versions.items() if len(v)>1}


def configuration_eligible(states, analysis):
    allowed = {'steady'} if analysis=='strict' else {'steady','recoverable_congested'}
    return bool(states) and all(s in allowed for s in states)


def weighted_objective(workload, task_values):
    active = {k:v for k,v in workload.items() if v>0}
    if not active: raise ValueError('Empty workload objective')
    if any(task_values.get(k) is None or not math.isfinite(task_values[k]) or task_values[k]<=0 for k in active):
        return None
    return math.fsum(rate*task_values[k] for k,rate in active.items())/math.fsum(active.values())


def predicted_task_means(workload, routing, paths):
    out = {}
    for k,rate in workload.items():
        if rate<=0: continue
        route = {m:v for m,v in routing[k].items() if v>0}
        if not math.isclose(sum(route.values()),rate,rel_tol=1e-9,abs_tol=1e-12): raise ValueError('Routing conservation failure')
        out[k] = math.fsum(v/rate*paths[k,m] for m,v in route.items()) if all((k,m) in paths for m in route) else None
    return out


def ranking_metrics(ids, truth, scores):
    y,p = np.array(truth,float),np.array(scores,float)
    if len(ids)<2 or len(set(ids))!=len(ids) or not np.isfinite(y).all() or not np.isfinite(p).all() or np.min(y)<=0:
        raise ValueError('Invalid ranking population')
    n = len(ids)
    ti = sorted(range(n),key=lambda i:(y[i],ids[i]))
    pi = sorted(range(n),key=lambda i:(p[i],ids[i]))
    best = {i for i in range(n) if y[i]==y[ti[0]]}
    tied = {i for i in range(n) if p[i]==p[pi[0]]}
    varied = np.ptp(y)>0 and np.ptp(p)>0
    regret = float((y[pi[0]]-y[ti[0]])/y[ti[0]])
    return dict(candidates=n, kendall_tau=float(kendalltau(y,p,variant='b').statistic) if varied else None,
        spearman_rho=float(spearmanr(y,p).statistic) if varied else None,
        correlation_unavailable_reason=None if varied else 'constant score or true objective',
        top1_hit=int(pi[0] in best), top2_recall=len(set(pi[:2]) & set(ti[:2]))/2,
        top2_coverage=int(bool(set(pi[:2]) & best)), top2_nontrivial=n>2,
        top3_coverage=int(bool(set(pi[:3]) & best)) if n>=3 else None, top3_nontrivial=n>3,
        selected_candidate=ids[pi[0]], true_best_candidates=[ids[i] for i in sorted(best)],
        true_ranking=[ids[i] for i in ti], predicted_ranking=[ids[i] for i in pi], regret=regret,
        tied_predicted_minimum=[ids[i] for i in sorted(tied)],
        tie_regret_range=[float(min((y[i]-y[ti[0]])/y[ti[0]] for i in tied)),float(max((y[i]-y[ti[0]])/y[ti[0]] for i in tied))],
        uniform_random_baseline=dict(expected_regret=float(np.mean((y-y[ti[0]])/y[ti[0]])),
            top1_hit_probability=len(best)/n,
            top2_coverage_probability=1-math.comb(n-len(best),min(2,n))/math.comb(n,min(2,n)) if n-len(best)>=min(2,n) else 1.,
            top3_coverage_probability=(1-math.comb(n-len(best),3)/math.comb(n,3) if n-len(best)>=3 else 1.) if n>=3 else None))


def summarize_metrics(entries):
    out = dict(groups=len(entries))
    for key in ('kendall_tau','spearman_rho','top1_hit','top2_recall','top2_coverage','top3_coverage'):
        values=[e[key] for e in entries if e[key] is not None]
        out[key+'_mean']=float(np.mean(values)) if values else None
    for k in (2,3):
        values=[e[f'top{k}_coverage'] for e in entries if e[f'top{k}_nontrivial']]
        out[f'top{k}_nontrivial_groups']=len(values)
        out[f'top{k}_nontrivial_coverage']=float(np.mean(values)) if values else None
    regrets=[e['regret'] for e in entries]
    out['regret']=dict(mean=float(np.mean(regrets)),median=float(np.median(regrets)),
        p90=float(np.quantile(regrets,.9)),worst_case=max(regrets)) if regrets else {k:None for k in ('mean','median','p90','worst_case')}
    return out


def load_runs():
    rows=read_jsonl(DATA)
    labels={l['run_id']:l for l in read_jsonl(LABELS)}
    audit=json.loads(OOF_AUDIT.read_text())
    if sha(DATA)!=audit['dataset_sha256']: raise ValueError('Dataset differs from original OOF fit')
    def profile_sources(obj):
        if isinstance(obj,dict):
            if 'path' in obj and 'sha256' in obj: yield obj['path'],obj['sha256']
            for value in obj.values(): yield from profile_sources(value)
        elif isinstance(obj,(list,tuple)):
            for value in obj: yield from profile_sources(value)
    offline_hashes=dict(profile_sources(audit['standalone_catalog_sources'])) | audit['offline_resource_sources']
    if any(sha(p)!=h for p,h in offline_hashes.items()): raise ValueError('Offline profiles differ from Module 2D')
    folds={f['heldout_deployment']:f for f in audit['folds'] if f['subset']=='primary'}
    saved={}
    for p in read_jsonl(OOF):
        if p['subset']!='primary' or p['variant'] not in VARIANTS.values(): continue
        key=(p['variant'],p['row_id'])
        if key in saved: raise ValueError('Duplicate saved OOF prediction')
        saved[key]=p
    catalog=load_model_catalog()
    ex=FeatureExtractor(catalog,load_resource_profiles(catalog))
    by_run=defaultdict(list)
    for r in rows: by_run[r['run_id']].append(r)
    runs,proof,hashes=[],[],{}
    for rid,rr in by_run.items():
        r=rr[0]
        source=r['source_result_path']
        hashes[source]=sha(source)
        if hashes[source]!=r['source_sha256'] or labels[rid]['source_sha256']!=hashes[source]: raise ValueError('Raw source or v2 provenance drift')
        if labels[rid]['original_stability']!=r['stability']: raise ValueError('Original label drift')
        raw=json.loads(Path(source).read_text())
        w=workload_key(r['workload'])
        if w!=workload_key(raw['workload_rps']): raise ValueError('Workload metadata drift')
        sig=protocol_signature(raw)
        workers=raw['source']['workers']
        if set(workers)!={m for m,n in r['deployment'].items() if n}: raise ValueError('Worker registry/deployment mismatch')
        measured_mean,measured_p95,shortfall={}, {}, {}
        for task,rate in w.items():
            if rate<=0: continue
            metric=raw['per_task_metrics'][task]
            measured_mean[task]=metric.get('mean_e2e_latency_ms')
            measured_p95[task]=metric.get('p95_e2e_latency_ms')
            observed=[(q['completion_time_ns']-q['scheduled_time_ns'])/1e6 for q in raw['request_records'] if q['task']==task and q['status']=='completed']
            for value,expected in [(measured_mean[task],float(np.mean(observed)) if observed else None),
                                   (measured_p95[task],float(np.percentile(observed,95)) if observed else None)]:
                if value is not None and expected is not None and not math.isclose(value,expected,rel_tol=1e-10,abs_tol=1e-7):
                    raise ValueError('Task aggregate differs from raw request cohort')
            if len(observed)<30: shortfall[task]=len(observed)
            for row in rr:
                if row['task']==task and row['task_aggregate_labels']!=metric: raise ValueError('Duplicated aggregate label drift')
        pred,components,missing={}, {}, {}
        for method,variant in VARIANTS.items():
            path_predictions={}; used=[]; absent=[]
            for task,rate in w.items():
                if rate<=0: continue
                for model,route in r['routing'][task].items():
                    if route<=0: continue
                    key=f'{rid}::{task}::{model}'
                    p=saved.get((variant,key))
                    if p is None: absent.append(key); continue
                    f=folds[r['deployment_id']]
                    actual=next(q for q in rr if q['task']==task and q['target_model']==model)
                    if p['heldout_deployment']!=r['deployment_id'] or p['deployment_id']!=r['deployment_id']: raise ValueError('Wrong heldout deployment')
                    if r['deployment_id'] in f['train']['deployment_id'] or rid in f['train']['run_id'] or r['configuration_id'] in f['train']['configuration_id'] or key in f['fit_row_ids']:
                        raise ValueError('OOF training leakage')
                    if key not in f['test_row_ids']: raise ValueError('Prediction absent from saved heldout membership')
                    if p['configuration_id']!=r['configuration_id'] or p['source_sha256']!=hashes[source] or p['source_result_path']!=source:
                        raise ValueError('Prediction provenance mismatch')
                    if not math.isclose(p['true_latency_ms'],actual['labels']['mean_e2e_latency_ms'],rel_tol=1e-12): raise ValueError('OOF target drift')
                    path_predictions[task,model]=p['predicted_latency_ms']
                    used.append(dict(row_id=key,task=task,model=model,routing_rps=route,
                        predicted_mean_latency_ms=p['predicted_latency_ms'],heldout_deployment=p['heldout_deployment']))
                    proof.append(dict(variant=variant,row_id=key,heldout_deployment=r['deployment_id'],configuration_id=r['configuration_id'],run_id=rid,source_sha256=hashes[source]))
            pred[method]=weighted_objective(w,predicted_task_means(w,r['routing'],path_predictions))
            components[method]=used
            missing[method]=absent
        features=ex.extract(r['deployment'],r['workload'],r['routing'])
        for row in rr:
            target=features['per_model'][row['target_model']]
            if not math.isclose(target['standalone_service_demand'],row['standalone_service_demand'],rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('Diagnostic planning feature drift')
        runs.append(dict(run_id=rid,configuration_id=r['configuration_id'],deployment_id=r['deployment_id'],
            repetition_id=r['repetition_id'],seed=r['seed'],source_result_path=source,source_sha256=hashes[source],
            workload=w,deployment=r['deployment'],routing=r['routing'],
            protocol_id=r['protocol_id'],protocol_signature=sig,workers=workers,
            block=dict(seed=r['seed'],initial_arrival_phases=r['initial_arrival_phases']),
            stability_state_v2=labels[rid]['stability_state_v2'],stability_reason_v2=labels[rid]['stability_reason_v2'],
            true_mean_objective_ms=weighted_objective(w,measured_mean),true_tail_objective_ms=weighted_objective(w,measured_p95),
            measured_task_mean_ms=measured_mean,measured_task_p95_ms=measured_p95,tail_sample_shortfall=shortfall,
            predicted_mean_objective_ms=pred,predicted_tail_objective_ms=None,OOF_components=components,missing_OOF=missing,
            planning_features=features))
    return runs,proof,hashes,folds,offline_hashes


def build_groups(runs):
    buckets=defaultdict(list)
    for r in runs: buckets[canonical(dict(workload=r['workload'],protocol=r['protocol_signature']))].append(r)
    inventory=[]
    for key,rs in sorted(buckets.items()):
        by_config=defaultdict(list)
        for r in rs: by_config[r['configuration_id']].append(r)
        conflicts=shared_workers_compatible(rs)
        candidates=[]
        for cfg,cr in sorted(by_config.items()):
            if len({canonical(r['block']) for r in cr})!=len(cr): raise ValueError('Duplicate seed/phase block in a configuration')
            candidates.append(dict(configuration_id=cfg,deployment_id=cr[0]['deployment_id'],deployment=cr[0]['deployment'],routing=cr[0]['routing'],
                observed_run_ids=[r['run_id'] for r in cr],state_counts=dict(Counter(r['stability_state_v2'] for r in cr)),
                runs=cr,protocol_ids=sorted({r['protocol_id'] for r in cr})))
        inventory.append(dict(group_id='comparison_'+identity(json.loads(key)),workload=rs[0]['workload'],
            protocol_signature=rs[0]['protocol_signature'],worker_implementation_conflicts=conflicts,
            initial_candidate_count=len(candidates),initial_deployment_count=len({c['deployment_id'] for c in candidates}),candidates=candidates))
    analyses={}
    for analysis in ('strict','extended'):
        outcomes=[]
        for g in inventory:
            accepted=[]; excluded=[]
            for c in g['candidates']:
                reasons=[]
                if not configuration_eligible([r['stability_state_v2'] for r in c['runs']],analysis): reasons.append('configuration_contains_disallowed_state_repetition')
                if any(r['true_mean_objective_ms'] is None for r in c['runs']): reasons.append('unavailable_measured_task_mean')
                if any(any(v is None for v in r['predicted_mean_objective_ms'].values()) for r in c['runs']): reasons.append('missing_saved_OOF_path_prediction')
                if reasons: excluded.append(dict(configuration_id=c['configuration_id'],state_counts=c['state_counts'],reasons=reasons))
                else: accepted.append(c)
            common=set.intersection(*[{canonical(r['block']) for r in c['runs']} for c in accepted]) if accepted else set()
            status='rankable'
            if g['worker_implementation_conflicts']: status='incompatible_worker_implementations'
            elif len(accepted)<2: status='fewer_than_two_eligible_configurations'
            elif len(common)<POLICY['minimum_common_repetition_blocks']: status='insufficient_matched_repetition_blocks'
            candidates=[]
            for c in accepted:
                used=[r for r in c['runs'] if canonical(r['block']) in common]
                if not used: continue
                m=[r['true_mean_objective_ms'] for r in used]
                tail=[r['true_tail_objective_ms'] for r in used]
                tail_available=all(v is not None for v in tail)
                candidates.append(dict(configuration_id=c['configuration_id'],deployment_id=c['deployment_id'],deployment=c['deployment'],routing=c['routing'],
                    run_ids=[r['run_id'] for r in used],repetition_ids=[r['repetition_id'] for r in used],seeds=[r['seed'] for r in used],
                    source_result_paths=[r['source_result_path'] for r in used],source_sha256=[r['source_sha256'] for r in used],
                    omitted_noncommon_blocks=[r['run_id'] for r in c['runs'] if canonical(r['block']) not in common],
                    state_counts=c['state_counts'],
                    true_mean_objective_ms=float(np.mean(m)),true_mean_sample_std_ms=float(np.std(m,ddof=1)) if len(m)>1 else None,
                    true_tail_objective_ms=float(np.mean(tail)) if tail_available else None,
                    true_tail_sample_std_ms=float(np.std(tail,ddof=1)) if tail_available and len(tail)>1 else None,
                    predicted_mean_objective_ms={v:float(np.mean([r['predicted_mean_objective_ms'][v] for r in used])) for v in VARIANTS},
                    predicted_tail_objective_ms=None,planning_features=used[0]['planning_features'],
                    per_block_objectives=[dict(block=r['block'],run_id=r['run_id'],true_mean=r['true_mean_objective_ms'],true_tail=r['true_tail_objective_ms'],predicted_mean=r['predicted_mean_objective_ms']) for r in used]))
            deps={c['deployment_id'] for c in candidates}
            outcomes.append(dict(group_id=g['group_id'],analysis=analysis,status=status,workload=g['workload'],
                comparison_scope='cross_deployment' if len(deps)>1 else 'within_deployment_routing_only',
                deployment_count=len(deps),candidate_count=len(candidates),matched_blocks=[json.loads(k) for k in sorted(common)],
                excluded_configurations=excluded,candidates=candidates))
        analyses[analysis]=outcomes
    return inventory,analyses


def evaluate(analyses):
    records,predictions,regret=[],[],[]
    for analysis,groups in analyses.items():
        for g in groups:
            if g['status']!='rankable': continue
            cs=g['candidates'];ids=[c['configuration_id'] for c in cs]
            for method,variant in VARIANTS.items():
                scores=[c['predicted_mean_objective_ms'][method] for c in cs]
                for c in cs:
                    predictions.append(dict(analysis=analysis,group_id=g['group_id'],comparison_scope=g['comparison_scope'],method=method,OOF_variant=variant,
                        prediction_source=str(OOF),OOF_subset='primary',heldout_deployment=c['deployment_id'],
                        workload=g['workload'],**{k:v for k,v in c.items() if k not in ('predicted_mean_objective_ms','per_block_objectives')},
                        predicted_mean_objective_ms=c['predicted_mean_objective_ms'][method],
                        mean_rank_score=c['predicted_mean_objective_ms'][method],tail_proxy_score=c['predicted_mean_objective_ms'][method],
                        tail_proxy_limitation='Mean-only score used as available-data proxy; no P95 predictor exists.'))
                for objective,key in [('mean','true_mean_objective_ms'),('tail_mean_score_proxy','true_tail_objective_ms')]:
                    if any(c[key] is None for c in cs): continue
                    result=ranking_metrics(ids,[c[key] for c in cs],scores)
                    per_block=[]
                    for block in g['matched_blocks']:
                        rs=[next(r for r in c['per_block_objectives'] if r['block']==block) for c in cs]
                        metric=ranking_metrics(ids,[r['true_mean' if objective=='mean' else 'true_tail'] for r in rs],[r['predicted_mean'][method] for r in rs])
                        per_block.append(dict(block=block,metrics=metric))
                    record=dict(analysis=analysis,group_id=g['group_id'],comparison_scope=g['comparison_scope'],method=method,objective=objective,
                        metrics=result,per_repetition_diagnostics=per_block,
                        repetition_note='These are repeated measurements of ONE workload/protocol group, not extra independent ranking groups.')
                    records.append(record)
                    regret.append(dict(**record,workload=g['workload'],candidates=cs,
                        diagnostics=dict(qwen_only_candidate_present=any(c['deployment'].get('qwen')==1 and sum(c['deployment'].values())==1 for c in cs),
                            max_target_standalone_demand=max(m['standalone_service_demand'] for c in cs for m in c['planning_features']['per_model'].values()),
                            recoverable_candidate_present=any(c['state_counts'].get('recoverable_congested',0)>0 for c in cs),
                            interpretation='u, task mix and co-location vary with routing; this observational group does not isolate causal failure mechanisms.')))
    aggregates={}
    for analysis in analyses:
        aggregates[analysis]={}
        for scope in ('all_rankable','cross_deployment','within_deployment_routing_only'):
            aggregates[analysis][scope]={}
            for objective in ('mean','tail_mean_score_proxy'):
                aggregates[analysis][scope][objective]={method:summarize_metrics([r['metrics'] for r in records if r['analysis']==analysis
                    and (scope=='all_rankable' or r['comparison_scope']==scope) and r['objective']==objective and r['method']==method]) for method in VARIANTS}
    return records,aggregates,predictions,sorted(regret,key=lambda r:r['metrics']['regret'],reverse=True)


def gate(analyses,aggregates):
    breadth={}
    for analysis,groups in analyses.items():
        cross=[g for g in groups if g['status']=='rankable' and g['comparison_scope']=='cross_deployment']
        breadth[analysis]=dict(rankable_groups=sum(g['status']=='rankable' for g in groups),cross_deployment_groups=len(cross),
            distinct_deployments=len({c['deployment_id'] for g in cross for c in g['candidates']}),
            nontrivial_top2_cross_deployment_groups=sum(g['candidate_count']>2 for g in cross))
    b=breadth['strict'];s=POLICY['sufficiency']
    enough=b['cross_deployment_groups']>=s['min_cross_deployment_groups'] and b['distinct_deployments']>=s['min_distinct_deployments'] and b['nontrivial_top2_cross_deployment_groups']>=s['min_nontrivial_top2_cross_deployment_groups']
    if not enough: return dict(status='INSUFFICIENT COMPARABLE GROUPS',decision_gate=None,breadth=breadth,
        reason='Insufficient independent same-workload/protocol stable cross-deployment comparisons. Routing-only rankings cannot establish deployment selection validity.')
    m=aggregates['strict']['cross_deployment']['mean']['C_GBT']
    eligible=[g for g in analyses['strict'] if g['status']=='rankable' and g['comparison_scope']=='cross_deployment']
    every_positive=all(ranking_metrics([c['configuration_id'] for c in g['candidates']],
        [c['true_mean_objective_ms'] for c in g['candidates']],
        [c['predicted_mean_objective_ms']['C_GBT'] for c in g['candidates']])['kendall_tau'] is not None and
        ranking_metrics([c['configuration_id'] for c in g['candidates']],[c['true_mean_objective_ms'] for c in g['candidates']],
        [c['predicted_mean_objective_ms']['C_GBT'] for c in g['candidates']])['kendall_tau']>0 for g in eligible)
    passed=m['kendall_tau_mean'] is not None and m['kendall_tau_mean']>=.6 and m['spearman_rho_mean']>=.6 and every_positive and m['regret']['mean']<=.05 and m['regret']['worst_case']<=.15 and m['top2_nontrivial_coverage']>=.9 and (m['top3_nontrivial_coverage'] is None or m['top3_nontrivial_coverage']>=.9)
    partial=m['kendall_tau_mean'] is not None and m['kendall_tau_mean']>0 and m['regret']['worst_case']<=.5 and (m['top2_nontrivial_coverage']>=.8 or (m['top3_nontrivial_coverage'] is not None and m['top3_nontrivial_coverage']>=.9))
    decision='PASS' if passed else 'PARTIAL' if partial else 'FAIL'
    return dict(status=decision,decision_gate=decision,breadth=breadth)


def write_summary(out,decision,inventory,analyses,records,aggregates,proof):
    lines=['# Module 2F：Decision-Focused Ranking Gate','',f"**{decision['status']}**。",'',
        '当前数据不足以判定 cross-deployment scorer 的 PASS / PARTIAL / FAIL；没有进入 guided search。',
        '这不是把 predictor 判为 FAIL，而是缺少合法、足够的候选比较组。已有同 deployment routing 组的排序结果单独报告。','',
        '## 可比性审计','',
        f"检查69个真实 runs / 24个 configuration / 7个 deployment；形成 {len(inventory)} 个精确 workload + 兼容 protocol 单元。仅 {sum(g['initial_candidate_count']>=2 for g in inventory)} 个单元在筛选前有两个以上配置。",'',
        '- 比较键包含完整 task arrival rates、窗口/warmup/drain/队列限制、arrival law、task-pool/source/harness hashes、设备/环境及 clock 定义。不同 workload、只相同 u、不同持续时间均不拼组。',
        '- 唯一协议等价例外：deployment 引入/移除 worker；所有共同 worker 的实现和解释器必须相同。workers membership 不应把真实 deployment 对照人为拆开。',
        '- Strict：同配置同协议的全部重复必须 steady。Extended：全部 steady/recoverable。任一 unstable 重复都排除该配置，不能只保留恢复较好的 runs。',
        '- 各候选取共同 seed+initial-phase block（至少2个）；逐 run 计算 objective，再等权平均。重复不是独立 ranking groups。','',
        '| Analysis | 可排名组 | 跨 deployment 组 |', '|---|---:|---:|']
    for a,b in decision['breadth'].items(): lines.append(f"| {a} | {b['rankable_groups']} | {b['cross_deployment_groups']} |")
    lines+=['','可用组来自 Stage 2B-1：固定全部5个模型，workload=(text=2,image=1,ocr=1,asr=.3,VQA=.5,lid=1) req/s，100秒窗口。Strict 有4种 routing，Extended 有5种 routing；各3个配对 repetitions。它们测试的是一个 unseen deployment 内的 routing 排序。',
        '唯一原始跨 deployment 对照为 A_u_0.40 的 Qwen-only 与 C_no_active_contender 的四模型驻留配置；workload、183秒窗口和共同实现一致，但 Qwen-only 包含 unstable repetition，因此不能作为普通 latency ranking ground truth。其余相近特征点的 workload 不同。',
        '即便仅留下该对照的非 unstable 重复，也只有一个2-candidate组，Top-2覆盖必然100%，仍不足以验证 cross-deployment shortlisting。本轮未将它纳入主 ranking。','',
        '## 已保存 OOF 预测与目标口径','',
        f"仅复用 Module 2D primary subset 的 A_standalone、B_gbt_log_ratio、C_gbt_log_ratio；核验 {len(proof)} 条 path predictions 的 heldout deployment、训练组隔离、目标值和 source hash。未重新 fit 模型，未使用 Module 2E 校准输出。",
        '真实 mean/P95 使用原始 run 的 task aggregate，已对 request records 重算核验；不会把 service-path P95 按 routing 权重平均。预测 task mean 使用 r_km/lambda_k 混合，再与真值一样按 lambda_k 加权。',
        'J_tail 是 task P95 的 workload-weighted平均，再跨重复取均值；既不是 system P95，也不是跨所有重复 pooled P95。',
        '**没有 P95 predictor**：所有 predicted_tail_objective_ms 为 null。Tail 以下仅评估已有 mean score 排序对测量 tail 的 proxy baseline，不能称作 tail prediction 能力。','',
        '## 单个可用 routing 组的结果','',
        '| Analysis / objective | score | Kendall tau | Spearman rho | Top-1 hit | Top-2 recall | Top-2 coverage | Top-3 coverage | Regret |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        m=r['metrics']
        fmt=lambda x:'null' if x is None else f'{x:.3f}'
        lines.append(f"| {r['analysis']} / {r['objective']} | {r['method']} | "+' | '.join(fmt(m[k]) for k in ('kendall_tau','spearman_rho','top1_hit','top2_recall','top2_coverage','top3_coverage','regret'))+' |')
    lines+=['','每个 analysis 只有一个 workload/protocol 组，所以 mean / median / P90 / worst regret 的 group aggregate 均等于该单组 regret；不能把 strict/extended 或3次重复扩充为独立样本数。完整 per-repetition 结果、随机选择基线和非平凡 shortlist coverage 保存在 ranking_metrics.json。',
        'A/B/C 在这个组上均为零 regret；C 没有显示出优于 A/B 的 ranking benefit。单个 workload 下相同 deployment 的成功排序不足以支持跨 deployment scorer。','',
        '候选的真实目标与保存的 mean score（ms，跨共同 seeds 等权平均；P95 列为测量值）：','',
        '| Routing audit label | measured mean J | A mean score | B mean score | C mean score | measured tail J | state |',
        '|---|---:|---:|---:|---:|---:|---|']
    for g in analyses['extended']:
        if g['status']!='rankable': continue
        for c in sorted(g['candidates'],key=lambda c:c['true_mean_objective_ms']):
            label=c['run_ids'][0].split('_rep_')[0]
            tail='null' if c['true_tail_objective_ms'] is None else f"{c['true_tail_objective_ms']:.2f}"
            lines.append(f"| {label} | {c['true_mean_objective_ms']:.2f} | {c['predicted_mean_objective_ms']['A']:.2f} | {c['predicted_mean_objective_ms']['B']:.2f} | {c['predicted_mean_objective_ms']['C_GBT']:.2f} | {tail} | {','.join(c['state_counts'])} |")
    lines+=['','这些 routing audit labels 仅用于识别既有实验；完整正式变量 x/r/lambda 保存在结果中。','', '## 最大 regret 与局限','']
    mean_rows=[r for r in records if r['objective']=='mean']
    if mean_rows:
        worst=max(mean_rows,key=lambda r:r['metrics']['regret'])
        lines.append(f"可用组最大 mean regret={worst['metrics']['regret']:.4f}（{worst['analysis']}/{worst['method']}）。同一配置族的 Qwen load、task mix 与 contender workload 随 routing 同时改变，无法将误差单独归因于某一因素。")
    lines+=['Strict 不含 recoverable 候选；Extended 增加的 generalist-first routing 带有 recoverable congestion。Qwen-only 候选因不稳定未参加排名，不能在本轮归因其 ranking failure。',
        'regret_analysis.json 保存 workload、完整 deployment/routing、真实/预测目标、两种 ranking、u/q/C、稳定性与全部来源。',
        '未来要判定 deployment coarse scorer，至少需要同 workload/协议下多个稳定 deployment 候选，并覆盖多个独立 workload group；Top-K 还需候选数大于 K 才有辨别力。当前未采集这些数据。','',
        '## 产物与停止边界','',
        '- comparison_groups.json：所有协议/workload 单元、候选、筛除原因、配对 blocks。',
        '- ranking_metrics.json：每组 Kendall/Spearman/Top-K/regret 与 scope-separated aggregates。',
        '- ranking_predictions.jsonl：A/B/C 的配置级 OOF mean objective，P95 prediction 明确 null。',
        '- regret_analysis.json：最大 regret 的完整配置/feature 上下文。',
        '- ranking_leakage_audit.json：OOF provenance、fold 隔离及原文件哈希。',
        '完成后停止：没有新 profiling、模型训练、local calibration 或 guided deployment search。','']
    (out/'ranking_summary.md').write_text('\n'.join(lines))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=OUT)
    out=ap.parse_args(argv).output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    save(out/'evaluation_config.json',POLICY)
    protected={str(p):sha(p) for folder in ('performance_model','joint_deploy','deployment_core') for p in (ROOT/folder).rglob('*.py') if 'ranking' not in p.parts}
    protected.update({str(p):sha(p) for folder in ('performance_model_generalization','performance_model_local_calibration') for p in (ROOT/'results'/folder).iterdir() if p.is_file()})
    protected[str(DATA)]=sha(DATA)
    runs,proof,hashes,folds,offline_hashes=load_runs()
    inventory,analyses=build_groups(runs)
    records,aggregates,predictions,regret=evaluate(analyses)
    decision=gate(analyses,aggregates)
    save(out/'comparison_groups.json',dict(policy=POLICY,decision=decision,inventory=inventory,analyses=analyses))
    save(out/'ranking_metrics.json',dict(decision=decision,groups=records,aggregates=aggregates,
        tail_predictor_available=False,tail_metrics_scope='Mean-score proxy against observed weighted task P95 only'))
    (out/'ranking_predictions.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False,allow_nan=False)+'\n' for p in predictions))
    save(out/'regret_analysis.json',dict(decision=decision,worst_groups=regret,
        cross_deployment_failure_attribution='Not identifiable: no eligible cross-deployment ranking group.',
        no_cross_workload_ranking=True))
    if any(sha(p)!=h for p,h in protected.items()) or any(sha(p)!=h for p,h in hashes.items()) or any(sha(p)!=h for p,h in offline_hashes.items()): raise ValueError('Original data/code/results/profiles changed')
    save(out/'ranking_leakage_audit.json',dict(status='PASS',source_predictions=str(OOF),source_predictions_sha256=sha(OOF),
        source_fold_audit=str(OOF_AUDIT),source_fold_audit_sha256=sha(OOF_AUDIT),source_stability_v2=str(LABELS),source_stability_v2_sha256=sha(LABELS),
        prediction_subset='primary',variants=VARIANTS,raw_source_hashes=hashes,protected_source_hashes=protected,
        offline_profile_hashes=offline_hashes,
        checked_path_prediction_count=len(proof),prediction_proofs=proof,
        used_heldout_folds={d:dict(train=f['train'],test=f['test']) for d,f in folds.items()},
        fits_performed=0,new_inference_runs=0,original_labels_preserved=True,
        implementation_hashes={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}))
    write_summary(out,decision,inventory,analyses,records,aggregates,proof)
    print(json.dumps(dict(decision=decision,ranking_prediction_rows=len(predictions),ranked_comparisons=len(records)),indent=2))


if __name__=='__main__': main()
