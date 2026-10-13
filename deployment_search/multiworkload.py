"""Module 3C: frozen, bounded multi-workload real-runtime campaign.

Usage: python -m deployment_search.multiworkload --prepare-only
       python -m deployment_search.multiworkload
A rerun resumes only unfinished work and never refits a predictor.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

from deployment_core.models import load_hardware, load_model_catalog
from deployment_core.workload import Workload
from joint_deploy.measurement.schema import MeasurementConfig
from scipy.stats import spearmanr

from . import GuidedDeploymentSearch, SearchConfig
from .campaign_runtime import CampaignRuntimeEvaluator, execution_status
from .measurement_runtime import measurement_subprocess, summarize_repetition
from .pilot_provenance import source_hashes, verify_assets_unchanged
from .real_objective import NoiseAwareRuntimeObjective
from .real_pilot import SETTINGS as PILOT_SETTINGS, SEEDS as SEARCH_SEEDS
from .result import jsonable
from .runtime_cache import digest, file_hash, save_json
from .tests.fixtures import forbid_enumeration

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/deployment_search_multiworkload'
SEARCH=SearchConfig(top_b=2,max_iterations=3,no_improvement_patience=1,
    runtime_evaluation_budget=6,rho_max=.8,remove_pressure_threshold=.2,mean_tolerance_ms=1.)
SCALES={'W1_balanced':1.0,'W2_generalist_heavy':.7,'W3_specialist_friendly':.8}
POLICIES=('B0','B1_seed_1103','B1_seed_2207','B2','B3')
RANDOM_SEEDS={'B1_seed_1103':1103,'B1_seed_2207':2207}
ORDER_SEED=20260930
SHARED_UNIQUE_CAP=75
SEARCH_PHYSICAL_CAP=240
CONFIRM_VALID_PER_STATE=2
CONFIRM_PHYSICAL_CAP=70
CONFIRM_SEEDS=(8101,8202)


def utc():return datetime.now(timezone.utc).isoformat()


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=jsonable(value):
            raise RuntimeError('Frozen Module 3C input changed: '+str(path))
    else:save_json(path,value)


def gpu_meta():
    try:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used',
            '--format=csv,noheader'],text=True,timeout=10).strip()
    except Exception as exc:out=f'unavailable: {type(exc).__name__}: {exc}'
    return dict(timestamp=utc(),gpu=out,loadavg=os.getloadavg())


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    history=ROOT/'results/performance_model_comparable_groups/workload_groups.json'
    groups=json.loads(history.read_text())['groups']
    audit=ROOT/'results/data_quality_audit/audit_counts.json'
    audited=json.loads(audit.read_text())
    if audited['by_phase']['Module 2G']['completed_stability']!={'steady':30,'recoverable_congested':4,'unstable':11}:
        raise RuntimeError('Historical corrected Module 2G evidence changed')
    records=[]
    for name in SCALES:
        rates={k:float(v*SCALES[name]) for k,v in groups[name].items()}
        records.append(dict(name=name,arrival_rates_rps=rates,scale_from_historical=SCALES[name],
            historical_rates_rps=groups[name],measurement_seconds=210,warmup_seconds=10,
            routing_protocol='module3a_canonical_minmax_lp',
            workload_source=dict(path=str(history),sha256=file_hash(history)),
            correction_source=dict(path=str(audit),sha256=file_hash(audit)),
            rationale=('Existing W1 pilot workload, unchanged.' if name=='W1_balanced' else
                'Uniform one-time scale fixed before all Module 3C outcomes: corrected historical Module 2G '
                'shows genuine overloaded/incomplete repetitions (W2 4/15 unstable; W3 7/15 unstable). '
                'No result-dependent tuning or workload sweep.')))
    asset_file=ROOT/'results/deployment_search_real_pilot/asset_hashes.json'
    assets=json.loads(asset_file.read_text())
    verify_assets_unchanged(assets)
    sources=source_hashes(assets)
    sources['campaign_runtime_sha256']=file_hash(ROOT/'deployment_search/campaign_runtime.py')
    freeze(OUT/'source_model_hashes.json',sources)
    workload_doc=dict(schema_version=1,workloads=records,model_source_hashes_sha256=file_hash(OUT/'source_model_hashes.json'),
        hardware_identity=sources['gpu_identity'],asset_manifest_sha256=file_hash(asset_file),
        selected_before_algorithm_results=True)
    freeze(OUT/'benchmark_workloads.json',workload_doc)
    (OUT/'benchmark_workloads.sha256').write_text(file_hash(OUT/'benchmark_workloads.json')+'\n')
    config=dict(search=jsonable(SEARCH),predictor=None,objective='Module 3B NoiseAwareRuntimeObjective',
        min_relative_improvement=.05,search_repetitions_per_state=3,search_repetition_seeds=SEARCH_SEEDS,
        search_valid_unique_budget_per_policy=6,search_shared_exact_cache_unique_cap=SHARED_UNIQUE_CAP,
        valid_evaluation_budget='per policy includes seed, max 6; only three completed repetitions form a valid evaluation',
        physical_attempt_budget=dict(search=SEARCH_PHYSICAL_CAP,confirmation=CONFIRM_PHYSICAL_CAP),
        confirmation_valid_repetitions_per_seed_final=CONFIRM_VALID_PER_STATE,
        confirmation_repetition_seeds=CONFIRM_SEEDS,
        random_search_policy_seeds=RANDOM_SEEDS,execution_order_seed=ORDER_SEED,
        measurement_settings=PILOT_SETTINGS,
        run_status_policy='Data Quality Audit v1: completed genuine unstable valid; external/uncertain/harness/model interruption invalid for objective but physical attempt counted',
        b3_transition_criterion='positive canonical LP delta_z; runtime labels never read in transition comparison',
        cache_policy='exact full identity; read only on algorithm-selected state',
        historical_cache_import=False,real_exhaustive_optimum='unavailable')
    freeze(OUT/'benchmark_config.json',config)
    rng=random.Random(ORDER_SEED)
    order={}
    for name in SCALES:
        candidates=list(POLICIES)
        rng.shuffle(candidates)
        order[name]=candidates
    freeze(OUT/'execution_order.json',dict(seed=ORDER_SEED,blocked_order=order))
    freeze(OUT/'freeze_manifest.json',dict(files={name:file_hash(OUT/name) for name in
        ('benchmark_workloads.json','benchmark_config.json','execution_order.json','source_model_hashes.json')},
        implementation_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'deployment_search').glob('*.py'))},
        frozen_before_first_gpu_run=True))
    (OUT/'algorithm_runs').mkdir(exist_ok=True)
    (OUT/'interruption_archive').mkdir(exist_ok=True)
    return records,sources,order


def ledger():
    p=OUT/'shared_cache/runtime_cache_manifest.json'
    return json.loads(p.read_text()) if p.exists() else dict(configurations={},run_attempts=[])


def append_jsonl(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(jsonable(value),ensure_ascii=False,allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())


def policy_run(name,policy,workload,catalog,hardware,sources):
    run_id=f'{name}__{policy}'
    directory=OUT/'algorithm_runs'/run_id
    directory.mkdir(parents=True,exist_ok=True)
    final_path=directory/'run_result.json'
    if final_path.exists():
        return json.loads(final_path.read_text())
    evaluator=CampaignRuntimeEvaluator(OUT/'shared_cache',catalog,sources,PILOT_SETTINGS,
        run_id,SEARCH_SEEDS,SHARED_UNIQUE_CAP,SEARCH_PHYSICAL_CAP)
    start=time.time()
    start_meta=gpu_meta()
    save_json(directory/'run_start.json',dict(run_id=run_id,start_unix=start,**start_meta))
    config=SearchConfig(**dict(SEARCH.__dict__,max_iterations=0,runtime_evaluation_budget=1)) if policy=='B0' else SEARCH
    select='random' if policy.startswith('B1') else 'lp'
    transition='lp' if policy=='B3' else 'runtime'
    random_seed=RANDOM_SEEDS.get(policy)
    def checkpoint(result):
        save_json(directory/'search_trace.json',result.to_dict())
    with forbid_enumeration() as guards:
        result=GuidedDeploymentSearch(evaluator,predictor=None,
            runtime_objective=NoiseAwareRuntimeObjective(1.,.05),trace_callback=checkpoint,
            selection_policy=select,transition_policy=transition,random_seed=random_seed).search(
                workload,catalog,hardware,config)
    enum_calls=sum(g.call_count for g in guards)
    if enum_calls:raise RuntimeError('Forbidden deployment enumeration')
    after=ledger();end=time.time()
    physical=[a for a in after['run_attempts'] if a['algorithm_id']==run_id]
    if len(physical)>3*config.runtime_evaluation_budget+3:
        raise RuntimeError('Per-policy physical attempt excess requires review')
    trace=result.to_dict()
    audit=decision_audit(trace,policy,config)
    if not audit['passed']:
        raise RuntimeError('Decision audit failed for '+run_id+': '+str(audit['errors']))
    record=dict(run_id=run_id,workload=name,algorithm=policy,random_seed=random_seed,
        selection_policy=select,transition_policy=transition,search_config=config,
        search_status=result.status,seed_deployment=result.seed.deployment,final_deployment=result.deployment,
        seed_runtime=result.seed_runtime,final_runtime=result.runtime,
        valid_unique_runtime_evaluations=result.runtime_evaluation_count,
        physical_run_attempts=len(physical),physical_attempt_ids=[a['attempt_id'] for a in physical],
        lp_solve_count=result.lp_solve_count,milp_solve_count=result.milp_solve_count,
        visited_states=result.visited_deployment_count,accepted_mutations=sum(t['accepted'] is not None for t in result.iterations),
        rollback_count=sum(c['status']=='ROLLBACK' for t in trace['iterations'] for c in t['candidates']),
        enumeration_calls=enum_calls,planning_seconds=end-start-sum((a.get('process_outcome') or {}).get('elapsed_seconds',0) for a in physical),
        elapsed_seconds=end-start,start_unix=start,end_unix=end,start_environment=start_meta,end_environment=gpu_meta(),
        decision_audit=audit,trace_path=str(directory/'search_trace.json'))
    save_json(directory/'run_result.json',record)
    print(json.dumps(dict(event='policy_complete',run_id=run_id,status=result.status,
        final=result.deployment,valid=result.runtime_evaluation_count,attempts=len(physical))),flush=True)
    return jsonable(record)


def decision_audit(trace,policy,config):
    errors=[];decisions=[]
    if trace['runtime_evaluation_count']>config.runtime_evaluation_budget:errors.append('valid budget')
    if trace['milp_solve_count']!=1:errors.append('MILP count')
    for t in trace['iterations']:
        ranked=sorted((c for c in t['candidates'] if c['rank'] is not None),key=lambda c:c['rank'])
        selected=[c for c in ranked if c['selected_top_b']]
        if selected!=ranked[:config.top_b]:errors.append('Top-B mismatch')
        if policy=='B3':
            positive=[c for c in selected if c['runtime'] is not None and c['delta_z']>0]
            expected=max(positive,key=lambda c:c['delta_z']) if positive else None
            if (t['accepted'] or None)!=(sorted(m for m,n in expected['mutation']['deployment'].items() if n) if expected else None):
                errors.append('B3 transition diverges from LP-only criterion')
        if policy=='B2' or policy=='B3':
            if [c['delta_z'] for c in ranked]!=sorted((c['delta_z'] for c in ranked),reverse=True):
                errors.append('LP ranking mismatch')
        for c in selected:
            if c['runtime'] is None:continue
            decisions.append(dict(iteration=t['iteration'],deployment=c['mutation']['deployment'],
                delta_z=c['delta_z'],runtime_status=c['runtime']['stability_state_v2'],
                current_runtime=t['current_runtime'],candidate_runtime=c['runtime'],
                selected_rank=c['rank'],decision=c['status'],reason=c['reason'],
                transition_policy=('LP-only; runtime result excluded' if policy=='B3' else 'Module 3B RuntimeObjective')))
    return dict(passed=not errors,errors=errors,decisions=decisions,
                b3_runtime_result_influenced_transition=False if policy=='B3' else None)


def confirmation_attempts_count():
    p=OUT/'confirmation_attempts.jsonl'
    return len(p.read_text().splitlines()) if p.exists() else 0


def confirm_one(run,which,repetition,workload,catalog,sources):
    run_id=run['run_id'];state=run['seed_deployment'] if which=='seed' else run['final_deployment']
    target=OUT/'algorithm_runs'/run_id/'confirmation'/which/f'rep_{repetition}.json'
    if target.exists():
        data=json.loads(target.read_text())
        if data.get('run_status')=='completed' and file_hash(data['raw_result_path'])==data['raw_sha256']:
            return data
    directory=target.parent;directory.mkdir(parents=True,exist_ok=True)
    # The policy's final canonical routing is in its trace. Re-solve for seed
    # and final; canonical routing and full identity are checked before launch.
    from .routing import solve_canonical_routing
    canonical=solve_canonical_routing(state,workload,catalog,SEARCH.rho_max)
    if canonical is None:raise RuntimeError('Confirmation state not canonical feasible')
    nested={task:{m:float(canonical.routing.get((task,m),0.)) or 0. for m in sorted(catalog)}
            for task in sorted(workload.arrival_rate)}
    seed=CONFIRM_SEEDS[repetition-1]
    cfg=MeasurementConfig(state,dict(workload.arrival_rate),nested,random_seed=seed,**PILOT_SETTINGS)
    cfg.validate({m:p.supported_tasks for m,p in catalog.items()})
    identity=dict(deployment=state,workload=dict(workload.arrival_rate),routing=nested,
        measurement_settings=dict(PILOT_SETTINGS,random_seed=seed),source_model_hashes=sources)
    attempts=confirmation_attempts_count()
    for retry in range(1,4):
        if attempts>=CONFIRM_PHYSICAL_CAP:raise RuntimeError('Confirmation physical safety cap reached')
        stem=f'rep_{repetition}_attempt_{attempts+1:03d}'
        raw_path=directory/(stem+'.raw.json');log_path=directory/(stem+'.log');config_path=directory/(stem+'.config.json')
        save_json(config_path,cfg.as_dict())
        attempt=dict(attempt_id=attempts+1,run_id=run_id,which=which,repetition=repetition,
            post_search_confirmation=True,start_unix=time.time(),raw_result_path=str(raw_path),
            log_path=str(log_path),config_path=str(config_path),identity_sha256=digest(identity),
            environment_start=gpu_meta())
        append_jsonl(OUT/'confirmation_attempts.jsonl',dict(attempt,status='started'))
        outcome=measurement_subprocess(cfg.as_dict(),raw_path,log_path,config_path)
        raw=None
        if raw_path.exists():
            try:raw=json.loads(raw_path.read_text())
            except ValueError:pass
        status,reason=execution_status(raw,outcome,log_path)
        record=dict(attempt,run_status=status,reason=reason,outcome=outcome,end_unix=time.time(),
                    environment_end=gpu_meta(),raw_sha256=file_hash(raw_path) if raw_path.exists() else None)
        append_jsonl(OUT/'confirmation_outcomes.jsonl',record)
        attempts+=1
        if status!='completed':
            archive=OUT/'interruption_archive'/f'confirmation_attempt_{attempts:03d}'
            archive.mkdir(parents=True,exist_ok=True)
            for p in (raw_path,log_path,config_path,raw_path.with_suffix('.raw.process.json')):
                if p.exists():
                    import shutil;shutil.copy2(p,archive/p.name)
            continue
        rep=summarize_repetition(raw,workload,catalog,dict(returncode=0),raw_path)
        rep.update(run_status='completed',post_search_confirmation=True,repetition=repetition,
                   random_seed=seed,attempt_id=attempts,measurement_identity_sha256=digest(identity))
        save_json(target,rep)
        print(json.dumps(dict(event='confirmation_complete',run_id=run_id,which=which,
            repetition=repetition,stability=rep['stability_state_v2'])),flush=True)
        return rep
    raise RuntimeError('Three invalid confirmation attempts; review interruption archive')


def confirm_all(records,catalog,sources,order):
    runs={r['run_id']:r for r in records}
    for name,policies in order.items():
        w=Workload(next(x['arrival_rates_rps'] for x in json.loads((OUT/'benchmark_workloads.json').read_text())['workloads'] if x['name']==name))
        for policy in policies:
            r=runs[f'{name}__{policy}']
            # Interleave seed/final measurements within each blocked algorithm.
            for which,rep in [('seed',1),('final',1),('final',2),('seed',2)]:
                confirm_one(r,which,rep,w,catalog,sources)


def result_metrics(records,order):
    from .runtime import STABILITY
    data=[];eff=[];corr=[];ablation=[]
    objective=NoiseAwareRuntimeObjective(1.,.05)
    for r in records:
        directory=OUT/'algorithm_runs'/r['run_id']
        trace=json.loads((directory/'search_trace.json').read_text())
        report=dict(run_id=r['run_id'],workload=r['workload'],algorithm=r['algorithm'],
            seed_deployment=r['seed_deployment'],final_deployment=r['final_deployment'],
            search_final_stability=r['final_runtime']['stability_state_v2'],
            valid_unique_runtime_evaluations=r['valid_unique_runtime_evaluations'],
            physical_run_attempts=r['physical_run_attempts'],lp_solve_count=r['lp_solve_count'],
            visited_states=r['visited_states'],accepted_mutations=r['accepted_mutations'],
            rollback_count=r['rollback_count'],planning_seconds=r['planning_seconds'])
        for which in ('seed','final'):
            search_reps=r[which+'_runtime']['repetitions']
            conf=[json.loads((directory/'confirmation'/which/f'rep_{i}.json').read_text()) for i in (1,2)]
            reps=search_reps+conf
            means=[x['weighted_mean_e2e'] for x in reps]
            tails=[x['weighted_p95_e2e'] for x in reps]
            valid=[v for v in means if v is not None]
            state=min((x['stability_state_v2'] for x in reps),key=STABILITY.get)
            report[which]=dict(stability_state=state,
                steady_repetition_fraction=sum(x['stability_state_v2']=='steady' for x in reps)/len(reps),
                weighted_mean_e2e=dict(mean=statistics.mean(valid) if valid else None,
                    median=statistics.median(valid) if valid else None,
                    sample_std=statistics.stdev(valid) if len(valid)>1 else None,
                    values=means,observation_count=len(valid),total_repetitions=len(reps)),
                weighted_task_p95_mean=statistics.mean(v for v in tails if v is not None) if any(v is not None for v in tails) else None,
                per_task_latency=[x['per_task_mean'] for x in reps],
                queue_metrics=[x['queue_metrics'] for x in reps],
                completion_ratio=[x['completion_ratio'] for x in reps],
                confirmation_direction_consistent=None)
            if all(x['weighted_mean_e2e'] is not None for x in conf):
                report[which]['confirmation_mean_ms']=statistics.mean(x['weighted_mean_e2e'] for x in conf)
        s=report['seed']['weighted_mean_e2e']['mean'];f=report['final']['weighted_mean_e2e']['mean']
        report['improvement_vs_seed']=(s-f)/s if s and f is not None else None
        ss=r['seed_runtime']['weighted_mean_e2e'];ff=r['final_runtime']['weighted_mean_e2e']
        report['search_sample_improvement']=(ss-ff)/ss if ss and ff is not None else None
        cs=report['seed'].get('confirmation_mean_ms');cf=report['final'].get('confirmation_mean_ms')
        report['confirmation_direction_consistent']=((ss-ff)*(cs-cf)>=0 if None not in (ss,ff,cs,cf) else None)
        if r['seed_deployment']==r['final_deployment']:
            report['confirmation_direction_consistent']=True
        data.append(report)
        best_seen=[r['seed_runtime']]+[c['runtime'] for t in trace['iterations'] for c in t['candidates'] if c['runtime']]
        better=min(best_seen,key=lambda v:(-STABILITY[v['stability_state_v2']],v['weighted_mean_e2e'] if v['weighted_mean_e2e'] is not None else math.inf))
        seed_mean=r['seed_runtime']['weighted_mean_e2e'];best_mean=better['weighted_mean_e2e']
        first_steady=next((i+1 for i,v in enumerate(best_seen) if v['stability_state_v2']=='steady'),None)
        eff.append(dict(run_id=r['run_id'],runtime_evaluations=len(best_seen),
            efficiency_ms_per_evaluation=((seed_mean-best_mean)/len(best_seen) if seed_mean and best_mean is not None else None),
            first_steady_evaluation=first_steady,first_steady_elapsed_seconds=None,
            best_seen_objective=best_mean,best_seen_stability=better['stability_state_v2']))
        for t in trace['iterations']:
            cur=t['current_runtime']
            for c in t['candidates']:
                if not c['runtime']:continue
                cand=c['runtime'];delta_j=(cur['weighted_mean_e2e']-cand['weighted_mean_e2e']
                    if None not in (cur['weighted_mean_e2e'],cand['weighted_mean_e2e']) else None)
                corr.append(dict(run_id=r['run_id'],iteration=t['iteration'],deployment=c['mutation']['deployment'],
                    delta_z=c['delta_z'],delta_J_ms=delta_j,
                    candidate_stability=cand['stability_state_v2'],current_stability=cur['stability_state_v2']))
                if r['algorithm'] in ('B2','B3') and c['delta_z']>0:
                    ablation.append(dict(run_id=r['run_id'],algorithm=r['algorithm'],
                        delta_z=c['delta_z'],delta_J_ms=delta_j,decision=c['status'],
                        genuine_runtime_worse=(STABILITY[cand['stability_state_v2']]<STABILITY[cur['stability_state_v2']]
                            or (cand['stability_state_v2']==cur['stability_state_v2']=='steady' and
                                delta_j is not None and delta_j<0))))
    save_json(OUT/'per_workload_metrics.json',data)
    save_json(OUT/'search_efficiency.json',eff)
    by_workload={}
    for name in order:
        pairs=[x for x in corr if x['run_id'].startswith(name+'__') and x['delta_J_ms'] is not None]
        by_workload[name]=dict(n=len(pairs),spearman=(float(spearmanr([x['delta_z'] for x in pairs],
            [x['delta_J_ms'] for x in pairs]).statistic) if len(pairs)>=3 and
            len({x['delta_z'] for x in pairs})>1 and len({x['delta_J_ms'] for x in pairs})>1 else None))
    save_json(OUT/'lp_runtime_correlation.json',dict(transitions=corr,descriptive_by_workload=by_workload,causal_interpretation=False))
    save_json(OUT/'ablation_results.json',dict(transitions=ablation,
        lp_positive_runtime_worse=sum(x['genuine_runtime_worse'] for x in ablation),
        b2_rollbacks_on_runtime_worse=sum(x['genuine_runtime_worse'] and x['decision']=='ROLLBACK' for x in ablation if x['algorithm']=='B2'),
        b3_accepts_on_runtime_worse=sum(x['genuine_runtime_worse'] and x['decision']=='ACCEPTED' for x in ablation if x['algorithm']=='B3')))
    return data,eff,by_workload,ablation


def report(records,order):
    metrics,eff,corr,ablation=result_metrics(records,order)
    manifest=ledger()
    save_json(OUT/'shared_runtime_cache_manifest.json',manifest)
    access=[json.loads(s) for s in (OUT/'cache_access_audit.jsonl').read_text().splitlines()]
    if any(a['access_reason']!='algorithm_selected_state' for a in access):raise RuntimeError('Oracle access detected')
    if len(manifest['run_attempts'])>SEARCH_PHYSICAL_CAP:raise RuntimeError('Physical search cap')
    if any(r['enumeration_calls']!=0 for r in records):raise RuntimeError('Enumeration call')
    names={m['run_id']:m for m in metrics}
    lines=['# Module 3C：Multi-workload Equal-budget Evaluation','',
        'Workloads、预算与执行顺序均在首次真实测量前冻结。所有主策略 predictor=None。'
        'Data Quality Audit execution-status/stability 语义统一用于本轮：completed genuine unstable 计入有效 runtime objective；'
        '外部或不确定中断只计物理尝试，不进入性能标签。','',
        '## 核心对照','',
        '| Workload | Algorithm | Seed | Final | Final stability | Seed mean E2E ms | Final mean E2E ms | Improvement | Valid eval | Physical attempts | LP solves |',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for name,policies in order.items():
        for policy in policies:
            m=names[f'{name}__{policy}']
            dep=lambda d:'+ '.join(k for k,v in d.items() if v) or 'none'
            sf=lambda x:'N/A' if x is None else f'{x:.3f}'
            pct=lambda x:'N/A' if x is None else f'{x:.2%}'
            lines.append(f"| {name} | {policy} | {dep(m['seed_deployment'])} | {dep(m['final_deployment'])} | {m['final']['stability_state']} | {sf(m['seed']['weighted_mean_e2e']['mean'])} | {sf(m['final']['weighted_mean_e2e']['mean'])} | {pct(m['improvement_vs_seed'])} | {m['valid_unique_runtime_evaluations']} | {m['physical_run_attempts']} | {m['lp_solve_count']} |")
    proposed=[m for m in metrics if m['algorithm']=='B2']
    random_runs=[m for m in metrics if m['algorithm'].startswith('B1')]
    wins=sum(m['improvement_vs_seed'] is not None and m['improvement_vs_seed']>0 for m in proposed)
    vs_random=0
    for m in proposed:
        peers=[r for r in random_runs if r['workload']==m['workload']]
        value=m['final']['weighted_mean_e2e']['mean']
        if value is not None and len(peers)==2 and all(r['final']['weighted_mean_e2e']['mean'] is not None and
                value<r['final']['weighted_mean_e2e']['mean'] for r in peers):
            vs_random+=1
    b2_rollback=sum(x['genuine_runtime_worse'] and x['decision']=='ROLLBACK' for x in ablation if x['algorithm']=='B2')
    b3_wrong=sum(x['genuine_runtime_worse'] and x['decision']=='ACCEPTED' for x in ablation if x['algorithm']=='B3')
    support=('STRONG SUPPORT' if wins>=2 and vs_random>=2 and b2_rollback>0 else
             'NO SUPPORT' if wins==0 or vs_random==0 else 'PARTIAL SUPPORT')
    lines+=['', '## 十二项结论','',
        '1. 三个 workload 的 seed 见表及逐运行 trace。',
        '2. Proposed 的 final deployment 见表。',
        f'3. Proposed 对 seed 的正向改善出现在 {wins}/3 个 workload；绝对延迟见表。',
        f'4. Proposed 的 final mean E2E 优于同 workload 两个真实 random trials 的 workload 数：{vs_random}/3；稳定性与 best-seen 另见 JSON。',
        f'5. B2 对 LP 正 delta 但真实退化的回滚 {b2_rollback} 次；B3 接受这类状态 {b3_wrong} 次。',
        f'6. delta_z 与真实 ΔJ 的 Spearman 描述统计：{json.dumps(corr,ensure_ascii=False)}；不作因果解释。',
        f"7. 平均 real evaluations：{statistics.mean(m['valid_unique_runtime_evaluations'] for m in proposed):.2f}（B2，含 seed）。",
        f"8. 已测 unstable candidate 数：{sum(x['candidate_stability']=='unstable' for x in json.loads((OUT/'lp_runtime_correlation.json').read_text())['transitions'])}。",
        f"9. confirmation 保持搜索改善方向的 B2 workload 数：{sum(m['confirmation_direction_consistent'] is True for m in proposed)}/3。",
        '10. Module 3B 的 99.35% 为单次 sample-level 观察；逐 workload 的实际改善见表，不能视作稳定加速比。',
        '11. Deployment enumeration calls = 0；real exhaustive optimum = unavailable。',
        f'12. 当前证据：**{support}**。','',
        '随机 baseline 为每个 workload 两个固定种子的真实 runtime trial；没有把 offline replay 当成真实实验。'
        '每状态 3 次 search repetition 与独立 confirmation 的统计、P95、queue、completion、LP/物理预算详见 JSON。','']
    (OUT/'module3c_summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(event='campaign_complete',support=support,
                          attempts=len(manifest['run_attempts']))),flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        workloads,sources,order=prepare()
        catalog,hardware=load_model_catalog(),load_hardware()
        records=[]
        for w in workloads:
            name=w['name'];workload=Workload(w['arrival_rates_rps'])
            for policy in order[name]:
                records.append(policy_run(name,policy,workload,catalog,hardware,sources))
        confirm_all(records,catalog,sources,order)
        verify_assets_unchanged(json.loads((ROOT/'results/deployment_search_real_pilot/asset_hashes.json').read_text()))
        report(records,order)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:
        prepare();print('Module 3C workload and config frozen before runtime evaluation.',flush=True)
    else:run()
