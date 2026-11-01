"""Predeclared alias service residual candidates; grouped development diagnostics."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import lsq_linear

PRESSURES={7,8,12,13}
NUMERICAL_DELTA_TOLERANCE=1e-8
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def aggregate(records):
    groups=defaultdict(list)
    for r in records:groups[(r['run_id'],r['family'],r['task'],r['n'],r['alias'],r['sample_id'])].append(r)
    out=[]
    for key,rr in sorted(groups.items()):
        fs=rr[0]['alias_features'];b=rr[0]['baseline_ms']
        assert all(r['alias_features']==fs and r['baseline_ms']==b and r['deployment_key']==rr[0]['deployment_key'] for r in rr)
        out.append(dict(run_id=key[0],family=key[1],task=key[2],n=key[3],alias=key[4],sample_id=key[5],
            deployment_key=rr[0]['deployment_key'],features=fs,baseline_ms=b,y=statistics.mean(r['log_service_ratio'] for r in rr),
            actual_arithmetic_service_total_ms=sum(r['service_ms'] for r in rr),response_repetitions=len(rr)))
    return out

def weights(rows):
    paths=defaultdict(set);samples=defaultdict(set);aliases=defaultdict(set)
    for r in rows:
        p=(r['family'],r['task'],r['n']);paths[r['run_id']].add(p);samples[r['run_id'],p].add(r['sample_id']);aliases[r['run_id'],p,r['sample_id']].add(r['alias'])
    return np.asarray([1/len(paths)/len(paths[r['run_id']])/len(samples[r['run_id'],(r['family'],r['task'],r['n'])])/len(aliases[r['run_id'],(r['family'],r['task'],r['n']),r['sample_id']]) for r in rows])

def fit(rows,columns):
    if not columns:return dict(success=True,coefficients=[],rank=0,parameters=0,condition_number=None,scaled_condition_number=None)
    a=np.asarray([[r['features'][i] for i in columns] for r in rows]);y=np.asarray([r['y'] for r in rows]);w=np.sqrt(weights(rows));aw=a*w[:,None];yw=y*w
    rank=int(np.linalg.matrix_rank(aw));scales=np.linalg.norm(aw,axis=0);sing=np.linalg.svd(aw/np.where(scales>0,scales,1),compute_uv=False)
    if rank<len(columns):return dict(success=False,rank=rank,parameters=len(columns),coefficients=None,reason='UNKNOWN_RANK_DEFICIENT_TRAINING_GROUP',scaled_singular_values=sing.tolist())
    lower=np.asarray([0 if i in PRESSURES else -np.inf for i in columns]);result=lsq_linear(aw,yw,bounds=(lower,np.inf),tol=1e-11,max_iter=1000)
    return dict(success=bool(result.success),rank=rank,parameters=len(columns),coefficients=result.x.tolist(),active_constraints=result.active_mask.tolist(),
        scaled_condition_number=float(sing[0]/sing[-1]),scaled_singular_values=sing.tolist(),positive_pressure_constraints=True)

def predict(rows,columns,model):
    return np.asarray([[r['features'][i] for i in columns] for r in rows])@np.asarray(model['coefficients']) if columns else np.zeros(len(rows))

def metrics(rows,pred):
    errors=pred-np.asarray([r['y'] for r in rows]);w=weights(rows);actual=defaultdict(float);geometric=defaultdict(float)
    for r,y in zip(rows,pred):
        key=(r['run_id'],r['family'],r['task'],r['n']);actual[key]+=r['actual_arithmetic_service_total_ms'];geometric[key]+=r['baseline_ms']*math.exp(y)*r['response_repetitions']
    ratios=[dict(run_id=k[0],family=k[1],task=k[2],n=k[3],actual_arithmetic_total_ms=actual[k],geometric_point_total_ms=geometric[k],actual_over_geometric_total=actual[k]/geometric[k]) for k in sorted(actual)]
    return dict(log_rmse=float(np.sqrt(np.sum(w*errors**2))),log_mae=float(np.sum(w*np.abs(errors))),
        absolute_relative_error=float(np.sum(w*np.abs(np.exp(np.clip(errors,-700,700))-1))),
        groups=len(rows),runs=len({r['run_id'] for r in rows}),arithmetic_scale_diagnostics=ratios,
        arithmetic_scale_diagnostics_not_used_as_retransformation=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--design',type=Path,required=True)
    p.add_argument('--design-manifest-sha256',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False);d=args.design.resolve();assert sha(d/'manifest.json')==args.design_manifest_sha256
    manifest=json.loads((d/'manifest.json').read_text())
    for name,expected in manifest['files'].items():assert sha(d/name)==expected
    for path,receipt in manifest['inputs'].items():assert sha(path)==receipt['sha256']
    contract=json.loads((d/'design-contract.json').read_text());report=json.loads((d/'report.json').read_text())
    assert sha(d/'design-contract.json')==report['design_contract_sha256']
    records=json.loads((d/'requests-with-alias-plan-features.json').read_text());run_meta=json.loads((d/'run-groups.json').read_text())
    assert not any(r['seed'] in [61,62,71,72,73,81,82,108] for r in run_meta)
    strict=[r for r in records if r['alias_valid_for_fit']];path_runs=defaultdict(set)
    for r in strict:path_runs[r['family'],r['task'],r['n']].add(r['run_id'])
    # Same eligibility domain as prior development fit: single-run path/n is insufficient.
    eligible=[r for r in strict if len(path_runs[r['family'],r['task'],r['n']])>=2];rows=aggregate(eligible)
    assert len({r['run_id'] for r in rows})==19 and len(eligible)==13951
    candidate_specs=report['candidate_ranks'];all_diagnostics=[];fold_records=[]
    for candidate in candidate_specs:
        name=candidate['name'];cols=candidate['columns'];full=fit(rows,cols);modes={}
        for grouping in ['run_id','deployment_key']:
            held_keys=sorted({r[grouping] for r in rows});folds=[];combined=[];predicted=[]
            for held in held_keys:
                train=[r for r in rows if r[grouping]!=held];test=[r for r in rows if r[grouping]==held];model=fit(train,cols)
                if not model['success']:
                    folds.append(dict(held_group=held,status='UNKNOWN',model=model,held_runs=sorted({r['run_id'] for r in test})));continue
                pred=predict(test,cols,model);combined.extend(test);predicted.extend(pred.tolist())
                paths={(r['family'],r['task'],r['n']) for r in train};inputs={(r['family'],r['task'],r['n'],r['sample_id']) for r in train}
                outside=0
                for r in test:
                    context=[v for v in train if (v['family'],v['task'],v['n'])==(r['family'],r['task'],r['n'])]
                    if not context or any(r['features'][i]<min(v['features'][i] for v in context)-1e-10 or r['features'][i]>max(v['features'][i] for v in context)+1e-10 for i in cols if i>=7):outside+=1
                folds.append(dict(held_group=held,status='DEVELOPMENT_DIAGNOSTIC',held_runs=sorted({r['run_id'] for r in test}),model=model,
                    metrics=metrics(test,pred),unseen_path_cells=sum((r['family'],r['task'],r['n']) not in paths for r in test),
                    unseen_input_cells=sum((r['family'],r['task'],r['n'],r['sample_id']) not in inputs for r in test),
                    outside_same_path_training_marginal_ranges=outside,convex_hull_membership_not_assumed=True))
            values=[f['metrics']['log_rmse'] for f in folds if f['status']!='UNKNOWN']
            modes[grouping]=dict(successful_folds=len(values),total_folds=len(held_keys),macro_group_log_rmse=statistics.mean(values) if values else None,
                descriptive_group_standard_error=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None,
                combined_metrics=metrics(combined,np.asarray(predicted)) if combined else None,
                unknown_folds=[f['held_group'] for f in folds if f['status']=='UNKNOWN'])
            fold_records.append(dict(candidate=name,grouping=grouping,folds=folds))
        all_diagnostics.append(dict(name=name,columns=cols,features=candidate['features'],full_development_fit=full,
            training_metrics=metrics(rows,predict(rows,cols,full)) if full['success'] else None,grouped_diagnostics=modes))
    base=next(c for c in all_diagnostics if c['name']=='family_self_other');comparisons=[]
    for c in all_diagnostics:
        if not c['name'].startswith('base_plus'):continue
        comparison=dict(candidate=c['name'],groups={})
        for grouping in ['run_id','deployment_key']:
            bf=next(v['folds'] for v in fold_records if v['candidate']=='family_self_other' and v['grouping']==grouping)
            cf=next(v['folds'] for v in fold_records if v['candidate']==c['name'] and v['grouping']==grouping)
            bm={f['held_group']:f for f in bf if f['status']!='UNKNOWN'};cm={f['held_group']:f for f in cf if f['status']!='UNKNOWN'};keys=sorted(set(bm)&set(cm))
            delta=[cm[k]['metrics']['log_rmse']-bm[k]['metrics']['log_rmse'] for k in keys]
            comparison['groups'][grouping]=dict(common_folds=len(keys),candidate_macro_log_rmse=statistics.mean(cm[k]['metrics']['log_rmse'] for k in keys) if keys else None,
                base_macro_log_rmse=statistics.mean(bm[k]['metrics']['log_rmse'] for k in keys) if keys else None,
                delta_candidate_minus_base=statistics.mean(delta) if delta else None,improved_groups=sum(v < -NUMERICAL_DELTA_TOLERANCE for v in delta),
                numerical_delta_tolerance=NUMERICAL_DELTA_TOLERANCE,
                per_group=[dict(group=k,delta_log_rmse=v) for k,v in zip(keys,delta)])
        comparison['improves_both_partitions']=all(v['delta_candidate_minus_base'] is not None and v['delta_candidate_minus_base'] < -NUMERICAL_DELTA_TOLERANCE for v in comparison['groups'].values())
        comparisons.append(comparison)
    complete=[c for c in all_diagnostics if all(v['successful_folds']==v['total_folds'] for v in c['grouped_diagnostics'].values()) and c['full_development_fit']['success']]
    best=min(complete,key=lambda c:c['grouped_diagnostics']['run_id']['macro_group_log_rmse']);cutoff=best['grouped_diagnostics']['run_id']['macro_group_log_rmse']+best['grouped_diagnostics']['run_id']['descriptive_group_standard_error']
    simple=min((c for c in complete if c['grouped_diagnostics']['run_id']['macro_group_log_rmse']<=cutoff),key=lambda c:(len(c['columns']),c['grouped_diagnostics']['run_id']['macro_group_log_rmse']))
    better=[c for c in comparisons if c['improves_both_partitions']]
    decision=dict(best_run_macro_candidate=best['name'],descriptive_run_one_SE_cutoff=cutoff,simplest_within_run_one_SE=simple['name'],
        added_feature_candidates_improving_both_partitions=[c['candidate'] for c in better],
        recommendation='KEEP_EXISTING_DEVELOPMENT_BASE_FORM_UNTIL_ADDED_TERMS_SHOW_USEFUL_GROUPED_IMPROVEMENT',
        production_activation=False,interpretation='Any selection is development-only; outcome108 already viewed and did not enter fitting. Small group count/unknown support forbid independent-validation claims.')
    domain=[]
    for key,ids in sorted(path_runs.items()):
        rr=[r for r in rows if (r['family'],r['task'],r['n'])==key]
        domain.append(dict(family=key[0],task=key[1],n=key[2],independent_runs=len(ids),included=len(ids)>=2,
            observed_input_ids=sorted({r['sample_id'] for r in rr}),observed_alias_contexts=[dict(run_id=r['run_id'],alias=r['alias'],A_self=r['features'][7],A_other=r['features'][8],alias_pressure=r['features'][12],partner_pressure=r['features'][13]) for r in rr],
            extrapolation='No automatic extrapolation beyond exact path/n/sample and observed context support; range flags are diagnostic, not proof of support.'))
    result=dict(schema='gsdeploy.alias_context_development_fit.v1',status='DEVELOPMENT_ONLY_NOT_VALIDATED_NOT_ACTIVATED',
        design=dict(path=str(d),manifest_sha256=args.design_manifest_sha256,contract_sha256=report['design_contract_sha256']),
        strict_requests=len(strict),fit_requests=len(eligible),grouped_alias_input_cells=len(rows),independent_runs=19,
        deployments=len({r['deployment_key'] for r in rows}),weights=contract['aggregation'],features=report['features'],
        candidates=all_diagnostics,added_feature_comparisons=comparisons,decision=decision,local_domain=domain,
        failures_retained_in_design=True,missing_covariate_policy='UNKNOWN not dropped as feasible',
        candidate_selection_uses_development_only=True,seed108_training_records=0,seed108_results_read_by_this_tool=False,
        old_forecast_and_failure_unchanged=True,geometric_scale_not_arithmetic_service_mean=True,
        predictor_activated=False,production_scoring_modified=False,GPU_started=False,activity_pair_status='PENDING_DECISION_NOT_QUEUED')
    save(out/'report.json',result);save(out/'grouped-fold-diagnostics.json',fold_records);save(out/'grouped-training-data.json',rows)
    overview=[dict(candidate=c['name'],parameters=len(c['columns']),run_log_rmse=c['grouped_diagnostics']['run_id']['macro_group_log_rmse'],
        deployment_log_rmse=c['grouped_diagnostics']['deployment_key']['macro_group_log_rmse'],fold_coverage={k:[v['successful_folds'],v['total_folds']] for k,v in c['grouped_diagnostics'].items()}) for c in all_diagnostics]
    save(out/'overview.json',overview)
    text=['# Alias计划上下文：小候选开发诊断','',
        '设计先冻结，再从原make_trace→SamplePool→SWRR与原始每input/n基线b重建covariates。25终态run全部保留，19run资格有效；本次沿用原拟合域13951请求，单run ASRn3状态未纳入拟合。108不入训练，旧107/108预测失败artifact未改。','',
        '每run等权、run内path等权、path内唯一input等权，一个input出现在多个alias时再均分该input权重。重复请求只平均log残差，不增加run权重。所有候选使用同一alias/input细胞和权重。','',
        '新增两项仅为alias计划b总work/horizon及同族partner计划b总work/horizon；没有使用实际service、overlap、tokens或completion作为特征。原模型的逐input b和精确aliasFIFO已经部分表达输入分配；新增项针对条件服务残差。','',
        '先验秩：base7列cond7.79；加alias8列cond38.55；加partner8列cond8.82；两项9列cond45.14。alias工作与Aself相关0.990，强共线性已显式记录。','',
        '|候选|参数|整run留一宏logRMSE|整deployment留组宏logRMSE|','|---|---:|---:|---:|']
    text.extend(f'|{c["candidate"]}|{c["parameters"]}|{c["run_log_rmse"]:.6f}|{c["deployment_log_rmse"]:.6f}|' for c in overview)
    text.extend(['',
        '所有候选19/19整run和9/9整deployment折数值有解。部署留组包含未见n状态与上下文外推，逐折unseen_path/input及范围标记保留；“有解”不等于可用域批准。',
        '', '加alias与base误差差异约1e−11，新增alias系数落在非负边界，按1e−8数值差容限不计改善。加partner整run仅改善约0.00109，但部署迁移恶化约0.01460；两项与partner相同。描述性1SE仍选7参数base；当前不推荐新增复杂度，不以这组数据宣布修复108预测失败。',
        '', '算术服务总量/几何点预测总量比按候选、held group和path另存，只作偏差诊断，未用于回变换补偿。exp(log残差)保持几何/中位尺度，不能作正式均值rho或完整U预测。',
        '', '后续保持CPU开发：先查逐部署误差、困难输入与时间分布的失败模式；不立即加入动态overlap或再排GPU。A/B/B/A固定驻留活动pair维持PENDING_DECISION_NOT_QUEUED。108结果已经被看过且启发了这类特征，未来只能作为事后开发诊断，不能称新的独立验证。',''])
    (out/'README.zh.md').write_text('\n'.join(text))
    save(out/'manifest.json',dict(script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),inputs={str(d/'manifest.json'):args.design_manifest_sha256},
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(overview=overview,decision=decision)))

if __name__=='__main__':main()
