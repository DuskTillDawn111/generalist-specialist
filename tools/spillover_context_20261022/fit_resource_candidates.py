"""CPU development-only M/E candidates on exactly locked eligible run records.

No native execution, predictor activation, reserved-result discovery, request-row
regression, fit-as-physics memory certificate, or formal utility ranking.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import nnls,linprog

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
LOCK=ROOT/'artifacts/static-spillover-study-20261022/context-resource-candidates-dev-v1/input-lock.json'
sys.path.insert(0,str(SOURCE))
from gsdeploy.energy import energy_norm,integrate_power

FORBIDDEN_SEEDS={61,62,71,72,73,81,82,108}
METHOD='nonnegative_least_squares_equal_run_weight'
M_SPECS={'M_residency_only':['raw_resident_over_budget'],
    'M_intercept_residency':['intercept','raw_resident_over_budget'],
    'M_intercept_workers':['intercept','workers'],
    'M_full':['intercept','raw_resident_over_budget','workers']}
E_SPECS={'E_intercept':['intercept'],'E_intercept_demand':['intercept','sum_baseline_pressure'],
    'E_intercept_workers':['intercept','workers'],'E_full':['intercept','sum_baseline_pressure','workers']}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def close(a,b):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)


def rank_design(matrix):
    a=np.asarray(matrix,dtype=float);s=np.linalg.svd(a,compute_uv=False)
    rank=int(np.linalg.matrix_rank(a));scales=np.linalg.norm(a,axis=0);scaled=a/np.where(scales>0,scales,1)
    ss=np.linalg.svd(scaled,compute_uv=False);width=a.shape[1]
    return dict(rows=len(a),columns=width,rank=rank,full_column_rank=rank==width,
        singular_values=s.tolist(),condition_number=float(s[0]/s[-1]) if rank==width else None,
        scaled_singular_values=ss.tolist(),scaled_condition_number=float(ss[0]/ss[-1]) if rank==width else None)


def fit(matrix,target):
    x=np.asarray(matrix,dtype=float);y=np.asarray(target,dtype=float)
    if len(y)!=len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x<0) or np.any(y<0):
        raise ValueError('Nonnegative finite run-level data required')
    design=rank_design(x)
    if not design['full_column_rank']:
        return dict(status='UNKNOWN_RANK_DEFICIENT',design=design,coefficients=None,predictions=None)
    scales=np.linalg.norm(x,axis=0);scaled=x/scales
    beta,residual=nnls(scaled,y);beta=beta/scales
    return dict(status='FITTED_DEVELOPMENT_ONLY',design=design,coefficients=beta.tolist(),
        active_coefficients=int(sum(beta>1e-12)),predictions=(x@beta).tolist(),residual_l2=float(residual),
        nonnegative_function_on_nonnegative_features=True,safety_upper_bound=False)


def errors(actual,predicted):
    available=[(float(a),float(p)) for a,p in zip(actual,predicted) if p is not None]
    if not available:return dict(observations=0,mean_absolute_error=None,mean_absolute_relative_error=None,max_absolute_relative_error=None)
    diff=np.asarray([p-a for a,p in available]);relative=np.asarray([abs(p-a)/a for a,p in available if a>0])
    return dict(observations=len(available),mean_error=float(np.mean(diff)),mean_absolute_error=float(np.mean(abs(diff))),
        rmse=float(np.sqrt(np.mean(diff**2))),mean_absolute_relative_error=float(np.mean(relative)) if len(relative) else None,
        median_absolute_relative_error=float(np.median(relative)) if len(relative) else None,
        max_absolute_relative_error=float(np.max(relative)) if len(relative) else None,
        underpredictions=int(sum(diff<-1e-9)),overpredictions=int(sum(diff>1e-9)))


def hull_membership(train,test):
    # Development geometry flag only; convex support does not prove validity.
    a=np.asarray(train,dtype=float);point=np.asarray(test,dtype=float)
    solution=linprog(np.zeros(len(a)),A_eq=np.vstack([a.T,np.ones(len(a))]),
        b_eq=np.r_[point,1.],bounds=(0,None),method='highs')
    return bool(solution.success)


def split_diagnostics(rows,features,target,groups):
    x=np.asarray([[r['features'][f] for f in features] for r in rows]);y=np.asarray([r[target] for r in rows])
    predictions=[None]*len(rows);folds=[]
    for group in dict.fromkeys(groups):
        train=[i for i,g in enumerate(groups) if g!=group];test=[i for i,g in enumerate(groups) if g==group]
        result=fit(x[train],y[train]);beta=result['coefficients']
        held=[]
        for i in test:
            predicted=float(x[i]@np.asarray(beta)) if beta is not None else None;predictions[i]=predicted
            held.append(dict(run_id=rows[i]['run_id'],actual=float(y[i]),predicted=predicted,
                error=predicted-float(y[i]) if predicted is not None else None,
                relative_error=(predicted-y[i])/y[i] if predicted is not None and y[i]>0 else None,
                in_training_feature_convex_hull=hull_membership(x[train],x[i])))
        folds.append(dict(held_group=group,training_run_ids=[rows[i]['run_id'] for i in train],held_run_ids=[rows[i]['run_id'] for i in test],
            status=result['status'],design=result['design'],coefficients=beta,held_predictions=held))
    return dict(folds=folds,predictions=predictions,metrics=errors(y,predictions),
        no_held_run_used_for_fit=True,no_request_pseudoreplication=True,
        unknown_predictions=sum(p is None for p in predictions))


def summarize_deployments(rows,predictions,target):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['deployment_key']].append(i)
    return [dict(deployment=rows[ids[0]]['deployment'],runs=[rows[i]['run_id'] for i in ids],
        actual_min=min(rows[i][target] for i in ids),actual_max=max(rows[i][target] for i in ids),
        metrics=errors([rows[i][target] for i in ids],[predictions[i] for i in ids]),
        predictions=[dict(run_id=rows[i]['run_id'],actual=rows[i][target],predicted=predictions[i]) for i in ids]) for ids in groups.values()]


def candidate(rows,name,features,target):
    x=[[r['features'][f] for f in features] for r in rows];y=[r[target] for r in rows]
    full=fit(x,y);loo=split_diagnostics(rows,features,target,[r['run_id'] for r in rows])
    deployment=split_diagnostics(rows,features,target,[r['deployment_key'] for r in rows])
    return dict(name=name,target=target,features=features,method=METHOD,equal_run_weights=True,
        all_run_fit=full,all_run_metrics=errors(y,full['predictions']),leave_one_run_out=loo,
        leave_one_deployment_out=deployment,
        per_deployment_leave_one_run_out=summarize_deployments(rows,loo['predictions'],target),
        per_deployment_leave_one_deployment_out=summarize_deployments(rows,deployment['predictions'],target),
        activated=False,safety_upper_bound=False,independent_validation=False)


def memory_constraint_diagnostics(rows,predictions):
    counts=Counter();details=[]
    for row,predicted in zip(rows,predictions):
        actual=row['M_norm'];predicted_safe=predicted*row['safe_factor'] if predicted is not None else None
        actual_pass=actual<=1;prediction_pass=predicted_safe<=1 if predicted_safe is not None else None
        label='UNKNOWN' if prediction_pass is None else ('true_pass' if actual_pass and prediction_pass else
            'false_reject' if actual_pass else 'false_accept' if prediction_pass else 'true_reject')
        counts[label]+=1;details.append(dict(run_id=row['run_id'],actual_M_norm=actual,predicted_M_norm=predicted_safe,
            observed_budget_pass=actual_pass,predicted_budget_pass=prediction_pass,classification=label))
    infeasible=sum(r['M_norm']>1 for r in rows)
    return dict(counts=dict(counts),details=details,observed_feasible=len(rows)-infeasible,observed_infeasible=infeasible,
        false_accept_rate=None if not infeasible else counts['false_accept']/infeasible,
        false_accept_rate_status='UNKNOWN_NO_RESOURCE_INFEASIBLE_RUNS_IN_ELIGIBLE_ONLY_DATA' if not infeasible else 'EMPIRICAL_ONLY',
        safety_guarantee=False,rho_or_stability_constraints_replaced=False)


def load_locked(lockpath):
    lockpath=Path(lockpath).resolve();lock=json.loads(lockpath.read_text());opened={str(lockpath):sha(lockpath)}
    def read_bound(reference):
        path=Path(reference['path']).resolve()
        if sha(path)!=reference['sha256']:raise ValueError('Locked source changed: '+str(path))
        opened[str(path)]=reference['sha256'];return json.loads(path.read_text())
    manifest=read_bound(lock['inventory_manifest']);report=read_bound(lock['inventory_report']);ledger=read_bound(lock['inventory_ledger'])
    for key in ('inventory_report','inventory_ledger'):
        ref=lock[key]
        if manifest['files'][Path(ref['path']).name]!=ref['sha256']:raise ValueError('Manifest binding mismatch')
    selected=[r for r in ledger if r['run_valid_for_fit']]
    if report['run_valid_count']!=19 or len(selected)!=19 or {r['sha256'] for r in selected}!={r['sha256'] for r in lock['selected_runs']}:
        raise ValueError('Exactly19 already-qualified development runs required')
    rows=[];audits=[];unknown=[]
    for source in selected:
        if source['seed'] in FORBIDDEN_SEEDS:raise ValueError('Reserved or108 seed forbidden before raw read')
        path=Path(source['path']).resolve()
        if any(any(w in part.lower() for w in ('holdout','validation','confirmation')) for part in path.parts):
            raise ValueError('Reserved result path forbidden')
        raw=read_bound(source);seed=raw['transport_config']['random_seed']
        if seed!=source['seed'] or seed in FORBIDDEN_SEEDS:raise ValueError('Raw seed mismatch/forbidden')
        if raw['status']!='completed' or raw['evaluation']['utility']['eligible'] is not True:raise ValueError('Run eligibility changed')
        config=raw['config'];memory=raw['memory_evidence'];energy=raw['energy_evidence'];utility=raw['evaluation']['utility']
        safe=config['catalog']['safe_factor'];budget=config['gpu_memory_mib'];horizon=raw['transport_config']['measurement_seconds']
        if safe!=1.110854 or budget!=49140 or horizon!=600 or config['energy_reference_watts']!=450 or config['startup_amortization_seconds']!=3600:
            raise ValueError('Original resource/scoring contract differs')
        if (memory.get('measured') is not True or memory.get('resource_valid') is not True or energy.get('kind')!='measured'
            or energy.get('scope')!='whole_gpu_board' or energy.get('coverage_valid') is not True):raise ValueError('Measured resource evidence missing')
        samples=[*memory['samples'],*[s for s in (memory.get('before'),memory.get('after')) if s is not None]]
        if not samples:raise ValueError('No owned-memory peak evidence')
        peak=max(s['owned_used_mib'] for s in samples)
        if any(s['foreign_pids'] for s in samples) or not close(peak*safe,memory['safe_memory_mib']):raise ValueError('Owned safe peak identity mismatch')
        mnorm=peak*safe/budget;enorm=energy_norm(energy,horizon,450,3600)
        w=raw['measurement_window'];startup_begin=energy.get('startup_begin_time_ns')
        run_integral=integrate_power(energy['samples'],w['start_time_ns'],w['end_time_ns'])
        startup_integral=integrate_power(energy['samples'],startup_begin,w['start_time_ns']) if startup_begin is not None else None
        if run_integral is None or startup_integral is None or not close(run_integral,energy['run_joules']) or not close(startup_integral,energy['startup_joules']):
            raise ValueError('Frozen board energy integral reproduction mismatch')
        if enorm is None or not close(mnorm,utility['M_norm']) or not close(enorm,utility['E_norm']):raise ValueError('Original M/E utility fields mismatch')
        for term,value in [('M',mnorm),('E',enorm)]:
            if not close(-config['weights'][term]*value,utility[term+'_contribution']):raise ValueError('Original contribution identity mismatch')
        if raw['plan']['deployment']!=source['deployment']:raise ValueError('Deployment identity mismatch')
        residency=sum(config['catalog']['memory_table_mib'][m][str(n)] for m,n in source['deployment'].items())/budget
        workers=sum(source['deployment'].values());demand=sum(source['A_by_family'].values())
        if source['M_features']!=[1.,residency,workers] or source['E_features']!=[1.,demand,workers]:raise ValueError('Locked feature construction differs')
        row=dict(run_id=source['run_id'],source={'path':str(path),'sha256':source['sha256']},seed=seed,
            deployment=source['deployment'],deployment_key=json.dumps(source['deployment'],sort_keys=True,separators=(',',':')),
            routing=source['routing'],workload=source['workload'],stability=source['stability'],gpu_uuid=raw.get('preflight',{}).get('gpu',{}).get('uuid'),
            horizon_seconds=horizon,safe_factor=safe,budget_mib=budget,energy_reference_watts=450,startup_amortization_seconds=3600,
            features=dict(intercept=1.,raw_resident_over_budget=residency,sum_baseline_pressure=demand,workers=workers),
            M_raw_peak_over_budget=peak/budget,M_peak_owned_mib=peak,M_safe_memory_mib=peak*safe,M_norm=mnorm,
            E_window_watts=energy['run_joules']/horizon,E_window_joules=energy['run_joules'],E_startup_joules=energy['startup_joules'],E_norm=enorm,
            weights=config['weights'],M_contribution=utility['M_contribution'],E_contribution=utility['E_contribution'],
            eligible=True,formal_original_constraints_verified=utility.get('constraints_verified'))
        rows.append(row);audits.append(dict(run_id=row['run_id'],raw_peak_reproduced=True,M_norm_reproduced=True,M_contribution_reproduced=True,
            E_norm_reproduced=True,E_contribution_reproduced=True,window_integral_reproduced=True,startup_integral_reproduced=True,
            owned_peak_scope=memory.get('scope'),memory_samples=len(samples),power_samples=len(energy['samples']),
            measurement_interval_ns=[w['start_time_ns'],w['end_time_ns']],startup_interval_ns=[startup_begin,w['start_time_ns']],
            startup_includes_loading_priming_warmup=True,drain_in_E=False))
    return rows,audits,unknown,opened


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input-lock',type=Path,default=LOCK)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args(argv)
    rows,audits,unknown,opened=load_locked(args.input_lock)
    args.output.mkdir(parents=True,exist_ok=False)
    # Register candidate forms and split/error rules before producing fit output.
    protocol=dict(schema='gsdeploy.resource_candidates_development_protocol.v1',input_lock={'path':str(args.input_lock.resolve()),'sha256':sha(args.input_lock)},
        run_count=19,run_selection='Exactly inventory run_valid_for_fit records; no failed/paused or reserved results opened',
        method=METHOD,M_target='raw process-owned sampled peak /49140; apply original safe_factor1.110854 exactly once to recover original M_norm',
        E_targets=['formal600s whole-board window mean watts=run_joules/600','startup whole-board joules separately; load/prime/warmup through measurement start'],
        E_normalization='(predicted_window_watts + predicted_startup_joules/3600)/450; unchanged frozen energy_norm semantics',
        M_candidates=M_SPECS,E_candidates=E_SPECS,startup_additional_comparator=M_SPECS['M_full'],
        validation='Development LOO with all rows of held run removed; additional leave-whole-deployment-out diagnostic; equal run weights; no holdout result access',
        nonnegative='NNLS only ensures nonnegative function on declared nonnegative features, never safety upper bound, feasibility proof or independent validation',
        constraint_diagnostics='Only memory budget classification against measured safe peak. E has no original hard feasibility threshold; rho/stability not replaced. No infeasible qualified rows => false-accept rate UNKNOWN.',
        partial_utility='M/E-only contribution substitution error with measured other terms fixed; no full-U prediction or ranking',
        thresholds='Report inherited0.25 relative M/E errors descriptively; no development acceptance, threshold tuning or activation',
        GPU_started=False,predictor_activated=False,production_U=None,reads_108_or_reserved_results=False)
    save(args.output/'protocol.json',protocol);save(args.output/'run-fields.json',dict(rows=rows,audits=audits,unknown=unknown))
    models={}
    for name,features in M_SPECS.items():
        model=candidate(rows,name,features,'M_raw_peak_over_budget')
        for mode in ('leave_one_run_out','leave_one_deployment_out'):
            model[mode]['memory_constraint_diagnostics']=memory_constraint_diagnostics(rows,model[mode]['predictions'])
            model[mode]['M_norm_errors']=errors([r['M_norm'] for r in rows],[p*r['safe_factor'] if p is not None else None for r,p in zip(rows,model[mode]['predictions'])])
        models[name]=model
    for name,features in E_SPECS.items():
        models[name+'_window']=candidate(rows,name+'_window',features,'E_window_watts')
        models[name+'_startup']=candidate(rows,name+'_startup',features,'E_startup_joules')
    models['startup_residency_workers']=candidate(rows,'startup_residency_workers',M_SPECS['M_full'],'E_startup_joules')
    pipelines=[]
    for name in E_SPECS:
        for startup_name in (name+'_startup','startup_residency_workers'):
            pipeline=dict(name=name+'__'+startup_name,window_model=name+'_window',startup_model=startup_name,activated=False)
            for mode in ('leave_one_run_out','leave_one_deployment_out'):
                watts=models[name+'_window'][mode]['predictions'];startup=models[startup_name][mode]['predictions']
                values=[(p+q/3600)/450 if p is not None and q is not None else None for p,q in zip(watts,startup)]
                pipeline[mode]=dict(E_norm_predictions=values,E_norm_errors=errors([r['E_norm'] for r in rows],values),
                    per_deployment=summarize_deployments(rows,values,'E_norm'))
            pipelines.append(pipeline)
    # Fixed baseline full forms are reported jointly; selecting a winner on
    # these same folds would require a new independent validation set.
    baseline=next(p for p in pipelines if p['name']=='E_full__E_full_startup');partial=[]
    for mode in ('leave_one_run_out','leave_one_deployment_out'):
        values=[]
        for row,mp,ep in zip(rows,models['M_full'][mode]['predictions'],baseline[mode]['E_norm_predictions']):
            mn=mp*row['safe_factor'] if mp is not None else None
            delta=-(row['weights']['M']*(mn-row['M_norm'])+row['weights']['E']*(ep-row['E_norm'])) if mn is not None and ep is not None else None
            values.append(dict(run_id=row['run_id'],M_norm_predicted=mn,E_norm_predicted=ep,
                delta_U_from_M_E_only=delta,full_predicted_U=None,all_other_terms_held_at_measured_values=True))
        deltas=[abs(v['delta_U_from_M_E_only']) for v in values if v['delta_U_from_M_E_only'] is not None]
        partial.append(dict(mode=mode,rows=values,mean_absolute_partial_U_error=float(np.mean(deltas)),max_absolute_partial_U_error=max(deltas)))
    report=dict(schema='gsdeploy.resource_candidates_development_report.v1',status='DEVELOPMENT_CANDIDATES_NOT_VALIDATED',
        run_count=len(rows),deployment_count=len({r['deployment_key'] for r in rows}),models=models,E_pipelines=pipelines,
        baseline_full_form_M_E_only_utility_error=partial,source_field_audit_all_pass=True,
        domain=dict(observed_deployments=list({r['deployment_key']:r['deployment'] for r in rows}.values()),
            feature_ranges={key:[min(r['features'][key] for r in rows),max(r['features'][key] for r in rows)] for key in rows[0]['features']},
            GPU_UUIDs=sorted({r['gpu_uuid'] for r in rows}),horizon_seconds=600,
            applicability='Observed development domain only; feature hull is descriptive, unseen composition/input/task/activity or horizon remains UNKNOWN until independent evidence'),
        limitations=['All19 qualified runs pass actual memory budget; false feasible error rate is unidentifiable.',
            'Original memory samples start after READY; measured sampled peak can include priming, warmup, measurement and drain. It is not only idle residency nor a continuous/load-phase upper bound.',
            'Formal E window excludes drain. Startup includes load/prime/warmup; its internal subphase separation is UNKNOWN.',
            'Mean board watts is a finite-window average, not demonstrated stationary/steady power or model-only energy.',
            'NNLS does not provide a safe bound or resource certificate; do not prune from fitted memory.',
            'Repeated deployment runs can make LOO optimistic; whole-deployment-out is additional development diagnostic, not independent validation.',
            'M/E candidates alone do not validate Q/T/L/P or establish full U ranking; full predicted U remains null.'],
        GPU_started=False,predictor_activated=False,production_math_modified=False,production_U=None,independent_validation=False,
        reads_108_or_reserved_results=False)
    save(args.output/'report.json',report)
    lines=['# 局部 M/E 开发候选：19 条合格 run','',
        '仅使用锁定 inventory 的19条run资格有效开发记录，9种部署；未读108、验证、确认或不合格raw。未启动GPU、未激活模型。',
        '', 'M拟合目标为原实测owned采样峰值/49140，再乘原safe_factor1.110854一次得到M_norm；raw residency只是回归特征，不能替代峰值，更不是安全证据。',
        '', 'E分开拟合正式600秒整板积分/600（平均瓦）和startup焦耳；原E_norm=(window瓦+startup焦耳/3600)/450。startup含load/prime/warmup，正式窗口不含drain；不把平均功率称稳定功率。',
        '', '|候选|目标|列数/秩|缩放条件数|整run留一平均相对误差|最差相对误差|整deployment留出平均相对误差|',
        '|---|---|---|---|---|---|---|']
    for name,model in models.items():
        d=model['all_run_fit']['design'];a=model['leave_one_run_out']['metrics'];b=model['leave_one_deployment_out']['metrics']
        lines.append(f'|{name}|{model["target"]}|{d["columns"]}/{d["rank"]}|{d["scaled_condition_number"]:.3f}|{a["mean_absolute_relative_error"]:.4f}|{a["max_absolute_relative_error"]:.4f}|{b["mean_absolute_relative_error"]:.4f}|')
    lines+=['','|E组合|整run留一E_norm平均相对误差|最差相对误差|整deployment留出平均相对误差|','|---|---|---|---|']
    for p in pipelines:
        a=p['leave_one_run_out']['E_norm_errors'];b=p['leave_one_deployment_out']['E_norm_errors']
        lines.append(f'|{p["name"]}|{a["mean_absolute_relative_error"]:.4f}|{a["max_absolute_relative_error"]:.4f}|{b["mean_absolute_relative_error"]:.4f}|')
    lines+=['', '这些是同一开发集的候选比较，不按误差最小直接选型或宣称通过；详细系数、奇异值、每个留一fold及每种部署误差见report.json。正系数只保证非负预测，不能证明保守上界。',
        '', '全部训练run的实测memory都可行，故false-accept率UNKNOWN；误判表只能报告已观测可行run的false-reject。E原评分没有硬预算约束，不新增E准入。rho/稳定性保留原合同。',
        '', '完整U仍需独立合格的Q/T/L/P模型。本报告只在其他实测项固定时展示M/E替换的ΔU误差，不是预测完整U或最优证书。', '']
    with (args.output/'README.zh.md').open('x') as stream:stream.write('\n'.join(lines))
    opened[str(Path(__file__).resolve())]=sha(__file__)
    for file in ('energy.py','prediction.py','evaluation.py','runtime/runner.py'):
        p=SOURCE/'gsdeploy'/file;opened[str(p)]=sha(p)
    if any(sha(p)!=s for p,s in opened.items()):raise ValueError('Source changed during analysis')
    save(args.output/'manifest.json',dict(schema='gsdeploy.resource_development_candidates_manifest.v1',inputs=opened,
        files={p.name:sha(p) for p in args.output.iterdir() if p.is_file()},all_inputs_unchanged=True,
        selected_run_count=19,GPU_started=False,reads_108_or_reserved_results=False))
    print(json.dumps(dict(output=str(args.output),run_count=19,models=len(models),E_pipelines=len(pipelines),report_sha256=sha(args.output/'report.json'),
        baseline_full_M_LOO=models['M_full']['leave_one_run_out']['metrics'],baseline_full_E_LOO=baseline['leave_one_run_out']['E_norm_errors'],
        GPU_started=False,predictor_activated=False),ensure_ascii=False))


if __name__=='__main__':main()
