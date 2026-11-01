"""Standalone training-fold-only arithmetic retransformation diagnostics.

Never modifies or activates the currently timed search oracle. Uses only the
19 audited development runs' derived strict request table, not new raw runs.
"""
from collections import defaultdict
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/static-spillover-study-20261022'
INVENTORY=BASE/'context-design-inventory-v5';DEVELOPMENT=BASE/'local-context-development-v1'
DESTINATION=BASE/'arithmetic-retransformation-development-v2'
CANDIDATES=['geometric','global_duan','family_duan_min3runs']
FORBIDDEN={108,61,62,71,72,73,81,82}
SPEC=importlib.util.spec_from_file_location('_original_local_service_fit',Path(__file__).with_name('fit_local_context.py'))
local=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(local)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def save(p,v):
    with Path(p).open('x') as stream:json.dump(v,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def pathkey(r):return r['family'],r['task'],r['n']
def groupkey(r):return (r['run_id'],*pathkey(r),r['sample_id'])


def prepare(output=DESTINATION):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    files=[INVENTORY/'manifest.json',INVENTORY/'source-ledger.json',INVENTORY/'requests.json',
        DEVELOPMENT/'manifest.json',DEVELOPMENT/'report.json',DEVELOPMENT/'grouped-training-data.json',
        Path(__file__).resolve(),Path(local.__file__),ROOT/'tests/test_arithmetic_retransformation.py']
    inv=json.loads((INVENTORY/'manifest.json').read_text());dev=json.loads((DEVELOPMENT/'manifest.json').read_text())
    if sha(INVENTORY/'manifest.json')!='81775f9116d666476a8b5a8ac86745a431bbf921f679072b2f2dffaa55b079e8':raise ValueError('Inventory changed')
    for p in files:
        if p.parent==INVENTORY and p.name!='manifest.json' and sha(p)!=inv['files'][p.name]:raise ValueError('Inventory content changed')
        if p.parent==DEVELOPMENT and p.name!='manifest.json' and sha(p)!=dev['files'][p.name]:raise ValueError('Development content changed')
    protocol=dict(schema='gsdeploy.arithmetic_retransformation.protocol.v1',frozen_unix_ns=time.time_ns(),
        status='DECLARED_BEFORE_RETRANSFORMATION_COMPARISON',candidates=CANDIDATES,service_candidate='family_self_other',
        data='Exactly original19 audited run_valid_for_fit runs; original included path/n selection; derived strict rows only; no new GPU/raw/current results.',
        folds=['leave_one_whole_run_out','leave_one_whole_deployment_out'],
        fit='Refit fixed original family_self_other columns in every training fold, original run/path/input balanced log-ratio objective and nonnegative pressure coefficients.',
        smearing='For each training run/path/input, average exp(log(S/b)-f(X)) over original response repeats; then average groups with equal run total, equal paths within run and equal inputs within path. Global factor equals weighted arithmetic residual exponential.',
        family_smearing='Same weights restricted to that family, normalized within family. Require >=3 distinct training runs in the family; otherwise UNKNOWN, no global fallback. This is a diagnostic heterogeneity comparator, not an activation choice.',
        target='Heldout run-path total=sum of actual worker service_ms across its retained requests; mean=total/retained request count. Predict sum b*exp(f(X))*training_factor on exactly the same request/input cohort; total/mean relative errors identical by construction.',
        metrics=['signed/absolute relative run-path total/mean errors','run-balanced mean of within-run path errors','worst path error and worst run mean error','matched-support candidate comparisons; UNKNOWN folds remain explicit'],
        independent_unit='Run, never request. Repetitions do not create independent folds. Deployment holdout groups all runs of the same resident composition.',
        limitations=['Duan factor is an empirical marginal training residual moment, not guaranteed conditional arithmetic expectation under context/input heteroskedasticity or domain shift.',
            'Targets are retained successful strictly audited requests, not unobserved/failed service or full infinite workload.',
            'Candidate comparison is development only, no threshold tuning, automatic winner,1SE variants,activation,rho/U/stability change or GPU execution.'],
        inputs=[ref(p) for p in files],new_raw_results_read=False,GPU_started=False,production_source_modified=False)
    output.mkdir(parents=True,exist_ok=False);save(output/'protocol.json',protocol)
    print(json.dumps(dict(output=str(output),protocol=ref(output/'protocol.json'))))


def smearing(raw,columns,model,family=None,minimum_runs=3):
    selected=[r for r in raw if family is None or r['family']==family]
    run_ids=sorted({r['run_id'] for r in selected})
    if not selected or family is not None and len(run_ids)<minimum_runs:
        return dict(value=None,status='UNKNOWN_INSUFFICIENT_TRAINING_FAMILY_RUNS',training_run_ids=run_ids)
    allgroups=defaultdict(list)
    for r in raw:
        fitted=math.fsum(r['features'][i]*b for i,b in zip(columns,model['coefficients']))
        allgroups[groupkey(r)].append(math.exp(r['log_service_ratio']-fitted))
    paths_by_run=defaultdict(set);inputs_by_path=defaultdict(int)
    for rid,f,t,n,sid in allgroups:
        paths_by_run[rid].add((f,t,n));inputs_by_path[rid,f,t,n]+=1
    selectedgroups={k:v for k,v in allgroups.items() if family is None or k[1]==family}
    weights={k:1/len(paths_by_run)/len(paths_by_run[k[0]])/inputs_by_path[k[:4]] for k in selectedgroups}
    mass=math.fsum(weights.values());perrun=defaultdict(lambda:dict(weighted_residual_moment=0.,weight_mass=0.))
    for key,values in selectedgroups.items():
        perrun[key[0]]['weighted_residual_moment']+=weights[key]*statistics.mean(values)
        perrun[key[0]]['weight_mass']+=weights[key]
    value=math.fsum(weights[k]*statistics.mean(v) for k,v in selectedgroups.items())/mass
    return dict(value=value,status='TRAINING_FOLD_ONLY_EMPIRICAL_RESIDUAL_MOMENT',
        training_run_ids=run_ids,training_groups=len(selectedgroups),training_requests=len(selected),
        original_balanced_weight_mass=mass,per_run_weighted_moments=dict(perrun))


def summarize(rows):
    good=[r for r in rows if r['absolute_relative_error'] is not None];byrun=defaultdict(list)
    for r in good:byrun[r['run_id']].append(r)
    means={rid:statistics.mean(r['absolute_relative_error'] for r in rr) for rid,rr in byrun.items()}
    signed={rid:statistics.mean(r['relative_error'] for r in rr) for rid,rr in byrun.items()}
    return dict(paths=len(rows),known_paths=len(good),unknown_paths=len(rows)-len(good),known_runs=len(byrun),
        run_balanced_mean_absolute_relative_error=statistics.mean(means.values()) if means else None,
        run_balanced_mean_signed_relative_error=statistics.mean(signed.values()) if signed else None,
        worst_path_absolute_relative_error=max((r['absolute_relative_error'] for r in good),default=None),
        worst_run_mean_absolute_relative_error=max(means.values(),default=None),
        per_run_mean_absolute_relative_error=means,
        worst_paths=sorted(good,key=lambda r:r['absolute_relative_error'],reverse=True)[:5])


def fold(raw,columns,heldout,mode,fold_id,deployment_by_run):
    train=[r for r in raw if r['run_id'] not in heldout];test=[r for r in raw if r['run_id'] in heldout]
    model=local.fit(local.aggregate(train),columns);factors={};rows=[]
    if model['success']:
        factors['global_duan']=smearing(train,columns,model)
        factors['family_duan_min3runs']={f:smearing(train,columns,model,f) for f in sorted({r['family'] for r in test})}
    grouped=defaultdict(list)
    for r in test:grouped[(r['run_id'],*pathkey(r))].append(r)
    for candidate in CANDIDATES:
        for (rid,f,t,n),rr in sorted(grouped.items()):
            actual=math.fsum(r['service_ms'] for r in rr);factor=None;reason=None
            if not model['success']:reason=model.get('reason','TRAINING_FIT_FAILED')
            elif candidate=='geometric':factor=1.
            elif candidate=='global_duan':factor=factors[candidate]['value']
            else:
                factor=factors[candidate][f]['value']
                if factor is None:reason=factors[candidate][f]['status']
            predicted=math.fsum(r['baseline_ms']*math.exp(math.fsum(r['features'][i]*b for i,b in zip(columns,model['coefficients'])))*factor for r in rr) if factor is not None else None
            relative=(predicted-actual)/actual if predicted is not None else None
            rows.append(dict(candidate=candidate,fold_mode=mode,fold_id=fold_id,run_id=rid,family=f,task=t,n=n,
                deployment=deployment_by_run[rid],requests=len(rr),unique_inputs=len({r['sample_id'] for r in rr}),
                actual_arithmetic_total_ms=actual,actual_arithmetic_mean_ms=actual/len(rr),
                predicted_arithmetic_total_ms=predicted,predicted_arithmetic_mean_ms=predicted/len(rr) if predicted is not None else None,
                relative_error=relative,absolute_relative_error=abs(relative) if relative is not None else None,
                training_smear_factor=factor,unknown_reason=reason))
    return dict(mode=mode,fold_id=fold_id,heldout_run_ids=sorted(heldout),training_run_ids=sorted({r['run_id'] for r in train}),
        fitted_model=model,factors=factors,rows=rows)


def run(output=DESTINATION):
    output=Path(output).resolve();protocol=json.loads((output/'protocol.json').read_text())
    if protocol['candidates']!=CANDIDATES:raise ValueError('Candidates changed')
    for p in protocol['inputs']:
        if ref(p['path'])!=p:raise ValueError('Frozen input changed')
    results=output/'results'
    if results.exists():raise FileExistsError(results)
    ledger=json.loads((INVENTORY/'source-ledger.json').read_text());qualified=[r for r in ledger if r['run_valid_for_fit']]
    if len(qualified)!=19 or any(r['seed'] in FORBIDDEN for r in qualified):raise ValueError('Exactly19 qualified development runs required')
    run_ids={r['run_id'] for r in qualified};deployment={r['run_id']:r['deployment'] for r in qualified}
    development=json.loads((DEVELOPMENT/'report.json').read_text());model=next(c for c in development['candidates'] if c['name']=='family_self_other')
    columns=model['columns'];domain={(d['family'],d['task'],d['n']) for d in development['supported_local_domain'] if d['included']}
    original=json.loads((INVENTORY/'requests.json').read_text())
    raw=[r for r in original if r['valid_for_fit'] and r['run_id'] in run_ids and pathkey(r) in domain]
    if {r['run_id'] for r in raw}!=run_ids or len(raw)!=development['fit_requests']:raise ValueError('Original service fitting cohort differs')
    if any(not r['success'] or not r['baseline_identity_verified'] or not math.isfinite(r['log_service_ratio']) for r in raw):raise ValueError('Strict audit missing')
    fullfit=local.fit(local.aggregate(raw),columns)
    if not np_close(fullfit['coefficients'],model['full_development_fit']['coefficients']):raise ValueError('Original fixed model could not be reproduced')
    folds=[fold(raw,columns,{rid},'leave_one_whole_run_out',rid,deployment) for rid in sorted(run_ids)]
    deployment_groups=defaultdict(set)
    for rid,x in deployment.items():deployment_groups[json.dumps(x,sort_keys=True,separators=(',',':'))].add(rid)
    folds.extend(fold(raw,columns,ids,'leave_one_whole_deployment_out',key,deployment) for key,ids in sorted(deployment_groups.items()))
    summary=[]
    for mode in protocol['folds']:
        allrows=[r for f in folds if f['mode']==mode for r in f['rows']]
        common={ (r['run_id'],r['family'],r['task'],r['n']) for r in allrows }
        for candidate in CANDIDATES:
            known={(r['run_id'],r['family'],r['task'],r['n']) for r in allrows if r['candidate']==candidate and r['relative_error'] is not None}
            common &= known
        summary.append(dict(mode=mode,candidates={candidate:dict(all_observed_support=summarize([r for r in allrows if r['candidate']==candidate]),
            common_support=summarize([r for r in allrows if r['candidate']==candidate and (r['run_id'],r['family'],r['task'],r['n']) in common])) for candidate in CANDIDATES},
            common_known_paths=len(common),total_paths=len(allrows)//len(CANDIDATES)))
    fullfactors=dict(global_duan=smearing(raw,columns,fullfit),family_duan_min3runs={f:smearing(raw,columns,fullfit,f) for f in sorted({r['family'] for r in raw})})
    report=dict(schema='gsdeploy.arithmetic_retransformation.development.v1',status='DEVELOPMENT_COMPARISON_ONLY_NOT_ACTIVATED',
        protocol=ref(output/'protocol.json'),runs=len(run_ids),deployments=len(deployment_groups),requests=len(raw),
        seed_counts=dict(CounterSafe(r['seed'] for r in qualified)),source_run_ids=sorted(run_ids),columns=columns,
        full_original_fit_reproduced=True,full_fit=fullfit,full_training_smearing=fullfactors,
        summary=summary,folds=folds,selected_candidate=None,predictor_activated=False,
        production_sources_modified=False,currently_timed_search_unchanged=True,new_raw_results_read=False,
        reads_108_or_reserved_results=False,GPU_started=False,U=None,formal_rho=None,
        limitations=protocol['limitations'])
    results.mkdir();save(results/'report.json',report)
    lines=['# 算术均值重变换：整run与整部署留出开发比较','',
        '固定family_self_other结构；每折重新训练原log-ratio模型，smear仅训练折估计。每run/path/input等权，重复响应内先平均exp残差。测试目标是保留请求的实际worker服务总量/均值；请求没有充当独立run。三个预声明候选全部保留、没有自动选型。','',
        '| 留出方式 | 候选 | 已知路径/全部 | run均衡平均绝对相对误差 | 最坏路径误差 | 最坏run均值误差 |',
        '|---|---|---:|---:|---:|---:|']
    for s in summary:
        for candidate,v in s['candidates'].items():
            m=v['all_observed_support'];fmt=lambda k:'UNKNOWN' if m[k] is None else f'{m[k]:.2%}'
            lines.append(f"| {s['mode']} | {candidate} | {m['known_paths']}/{m['paths']} | {fmt('run_balanced_mean_absolute_relative_error')} | {fmt('worst_path_absolute_relative_error')} | {fmt('worst_run_mean_absolute_relative_error')} |")
    lines+=['','共同已知路径上的公平比较、所有失败折、训练系数/因子、逐heldout run-path实际与预测total/mean及最差案例都保存在report.json。family候选训练族少于3run返回UNKNOWN，不用global填值。Duan是训练残差的经验边际矩，未证明异方差或新上下文下条件算术期望；无激活、GPU、U/rho/质量或稳定性修改。']
    (results/'README.zh.md').write_text('\n'.join(lines)+'\n')
    for p in protocol['inputs']:
        if ref(p['path'])!=p:raise ValueError('Frozen input changed during diagnostics')
    save(results/'manifest.json',dict(protocol=ref(output/'protocol.json'),files={p.name:sha(p) for p in results.iterdir() if p.is_file()},all_inputs_unchanged=True))
    print(json.dumps(dict(output=str(results),report=ref(results/'report.json'),summary=[dict(mode=s['mode'],candidates={c:{k:v['all_observed_support'][k] for k in ['known_paths','paths','run_balanced_mean_absolute_relative_error','worst_path_absolute_relative_error']} for c,v in s['candidates'].items()}) for s in summary])))


def np_close(a,b):return len(a)==len(b) and all(math.isclose(x,y,rel_tol=1e-8,abs_tol=1e-8) for x,y in zip(a,b))
def CounterSafe(values):
    from collections import Counter
    return Counter(map(str,values))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run']);p.add_argument('--output',type=Path,default=DESTINATION);a=p.parse_args()
    prepare(a.output) if a.mode=='prepare' else run(a.output)
