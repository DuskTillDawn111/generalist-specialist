"""Positive service-only development models and whole-run LOO; never activation.

Reads a pinned, strictly audited context inventory, not result discovery. Repeated
requests share run/path/input weight. Candidate selection and LOO are development
diagnostics and provide neither independent validation nor a total-utility claim.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import lsq_linear

FEATURES=['alpha_generalist_1','alpha_generalist_2','alpha_generalist_3','alpha_text','alpha_image','alpha_ocr','alpha_asr',
    'A_self','A_other','M_resident','H_self','centered_log_b_x_A_other']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def aggregate(rows):
    """Group log residuals; effective independent unit remains an execution run."""
    groups=defaultdict(list)
    for r in rows:groups[(r['run_id'],r['family'],r['task'],r['n'],r['sample_id'])].append(r)
    out=[]
    for key,rr in sorted(groups.items()):
        assert all(r['features']==rr[0]['features'] and r['baseline_ms']==rr[0]['baseline_ms'] for r in rr)
        out.append(dict(run_id=key[0],family=key[1],task=key[2],n=key[3],sample_id=key[4],
            features=rr[0]['features'],baseline_ms=rr[0]['baseline_ms'],response_repetitions=len(rr),
            y=statistics.mean(r['log_service_ratio'] for r in rr)))
    return out

def balanced_weights(rows):
    paths=defaultdict(set);counts=Counter()
    for r in rows:
        p=(r['family'],r['task'],r['n']);paths[r['run_id']].add(p);counts[r['run_id'],p]+=1
    return np.asarray([1/len(paths)/len(paths[r['run_id']])/counts[r['run_id'],(r['family'],r['task'],r['n'])] for r in rows])

def fit(rows,columns):
    if not columns:return dict(coefficients=[],rank=0,parameters=0,condition_number=None,success=True,pressure_constraints='self/other >=0')
    a=np.asarray([[r['features'][i] for i in columns] for r in rows]);y=np.asarray([r['y'] for r in rows]);w=np.sqrt(balanced_weights(rows))
    aw=a*w[:,None];yw=y*w;s=np.linalg.svd(aw,compute_uv=False);rank=int(np.linalg.matrix_rank(aw))
    if rank<len(columns):return dict(success=False,reason='RANK_DEFICIENT_TRAINING_FOLD',rank=rank,parameters=len(columns),coefficients=None)
    lower=np.asarray([0 if i in (7,8) else -np.inf for i in columns]);upper=np.full(len(columns),np.inf)
    solution=lsq_linear(aw,yw,bounds=(lower,upper),tol=1e-11,max_iter=1000)
    return dict(coefficients=solution.x.tolist(),rank=rank,parameters=len(columns),condition_number=float(s[0]/s[-1]),
        success=bool(solution.success),pressure_constraints='self/other >=0',cost=float(solution.cost),active_constraints=solution.active_mask.tolist())

def predict(rows,columns,model):
    if not columns:return np.zeros(len(rows))
    return np.asarray([[r['features'][i] for i in columns] for r in rows])@np.asarray(model['coefficients'])

def scores(rows,pred):
    errors=pred-np.asarray([r['y'] for r in rows]);w=balanced_weights(rows)
    relative=np.abs(np.exp(np.clip(errors,-700,700))-1)
    return dict(balanced_log_rmse=float(np.sqrt(np.sum(w*errors**2))),balanced_log_mae=float(np.sum(w*np.abs(errors))),
        balanced_absolute_relative_error=float(np.sum(w*relative)),groups=len(rows),runs=len({r['run_id'] for r in rows}),
        interpretation='Errors of run/path/input mean log service ratio; no tail/queue/e2e/utility guarantee')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--inventory-manifest-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();inv=args.inventory.resolve();out=args.output;out.mkdir(parents=True,exist_ok=False)
    assert sha(inv/'manifest.json')==args.inventory_manifest_sha256
    manifest=json.loads((inv/'manifest.json').read_text())
    for name,expected in manifest['files'].items():assert sha(inv/name)==expected
    for path,expected in manifest['inputs'].items():assert sha(path)==expected
    assert sha(manifest['script']['path'])==manifest['script']['sha256']
    report=json.loads((inv/'report.json').read_text());runs=json.loads((inv/'source-ledger.json').read_text())
    assert report['identity_equivalence_receipt'] and not report['predictor_activated']
    assert not any(r['seed'] in [61,62,71,72,73,81,82,108] for r in runs)
    records=json.loads((inv/'requests.json').read_text());strict=[r for r in records if r['valid_for_fit']]
    assert all(r['baseline_identity_verified'] and r['success'] and math.isfinite(r['log_service_ratio']) for r in strict)
    path_runs=defaultdict(set)
    for r in strict:path_runs[r['family'],r['task'],r['n']].add(r['run_id'])
    # A single observed replica state cannot have an independent run-left-out diagnostic.
    eligible=[r for r in strict if len(path_runs[r['family'],r['task'],r['n']])>=2]
    rows=aggregate(eligible);run_ids=sorted({r['run_id'] for r in rows});columns=[i for i in range(7) if any(r['features'][i] for r in rows)]
    candidates=[('baseline_offset',[]),('family_only',columns),('family_self_other',columns+[7,8]),
        ('family_self_other_M',columns+[7,8,9]),('family_self_other_H',columns+[7,8,10]),
        ('family_self_other_shape',columns+[7,8,11]),('observed_full_context',columns+[7,8,9,10,11])]
    diagnostics=[];all_folds=[]
    for name,chosen in candidates:
        model=fit(rows,chosen);folds=[];all_predictions=[];all_test=[]
        for rid in run_ids:
            train=[r for r in rows if r['run_id']!=rid];test=[r for r in rows if r['run_id']==rid]
            trained=fit(train,chosen)
            if not trained['success']:
                folds.append(dict(run_id=rid,success=False,model=trained));continue
            pred=predict(test,chosen,trained);all_predictions.extend(pred.tolist());all_test.extend(test)
            seen={(r['family'],r['task'],r['n'],r['sample_id']) for r in train}
            seen_paths={(r['family'],r['task'],r['n']) for r in train}
            # LOO extrapolation flags are diagnostic, never reclassified as held-out validation.
            unseen_input=sum((r['family'],r['task'],r['n'],r['sample_id']) not in seen for r in test)
            out_of_range=0
            for r in test:
                context=[v for v in train if v['family']==r['family']]
                if not context or any(r['features'][i]<min(v['features'][i] for v in context)-1e-10 or r['features'][i]>max(v['features'][i] for v in context)+1e-10 for i in chosen if i>=7):out_of_range+=1
            folds.append(dict(run_id=rid,success=True,metrics=scores(test,pred),model=trained,
                unseen_training_input_groups=unseen_input,unseen_training_path_groups=sum((r['family'],r['task'],r['n']) not in seen_paths for r in test),
                outside_same_family_training_marginal_context_ranges=out_of_range,
                convex_hull_membership_not_assumed=True,development_diagnostic_only=True))
        values=[f['metrics']['balanced_log_rmse'] for f in folds if f['success']]
        summary=dict(name=name,columns=chosen,features=[FEATURES[i] for i in chosen],full_development_fit=model,
            training_metrics=scores(rows,predict(rows,chosen,model)) if model['success'] else None,
            loo_metrics=scores(all_test,np.asarray(all_predictions)) if all_test else None,
            successful_folds=len(values),total_folds=len(run_ids),
            macro_run_log_rmse=statistics.mean(values) if values else None,
            descriptive_run_standard_error=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None,
            all_predictions_positive_by_construction=True,development_only=True)
        diagnostics.append(summary);all_folds.append(dict(name=name,folds=folds))
    complete=[c for c in diagnostics if c['successful_folds']==c['total_folds'] and c['full_development_fit']['success']]
    best=min(complete,key=lambda c:c['macro_run_log_rmse']);cutoff=best['macro_run_log_rmse']+(best['descriptive_run_standard_error'] or 0)
    preferred=min((c for c in complete if c['macro_run_log_rmse']<=cutoff),key=lambda c:(len(c['columns']),c['macro_run_log_rmse']))
    domain=[]
    for key,ids in sorted(path_runs.items()):
        rr=[r for r in rows if (r['family'],r['task'],r['n'])==key]
        context={r['run_id']:dict(A_self=r['features'][7],A_other=r['features'][8],M_resident=r['features'][9],H_self=r['features'][10]) for r in rr}
        domain.append(dict(family=key[0],task=key[1],n=key[2],independent_runs=len(ids),
            included=len(ids)>=2,sample_ids=sorted({r['sample_id'] for r in rr}),contexts_by_run=context,
            weak_support=len(ids)<3 or (key[0]=='generalist_2' and key[1]=='image'),
            rule='Only listed sample IDs and exact n; only convex hull of listed A_self/A_other/M/H contexts; no extrapolation along absent axes. Domain must be checked before any future activation.'))
    forbidden=[dict(family=r['family'],task=r['task'],n=r['n'],reason='NO_TWO_INDEPENDENT_RUNS_AT_THIS_PATH_STATE') for r in report['coverage'] if r['fit_records'] and r['independent_runs']<2]
    result=dict(schema='gsdeploy.local_context_development.v1',status='DEVELOPMENT_ONLY_NOT_VALIDATED_NOT_ACTIVATED',
        inventory=dict(path=str(inv),manifest_sha256=args.inventory_manifest_sha256),source_requests=len(records),strict_requests=len(strict),
        fit_requests=len(eligible),grouped_run_path_inputs=len(rows),independent_runs=len(run_ids),
        target='log(service_ms / per_input_median_standalone_ms)',
        prediction='S = b * exp(X beta); median-scale service only, without a retransformation mean claim',
        weights='Each run equal total weight; within each run each family/task/n path equal; within each path each unique input equal; repeated request log residuals averaged.',
        feature_policy='Only plan-known b, baseline pressures, raw residency and planned traffic entropy; actual service/overlap/tokens do not enter features.',
        split='Leave one complete execution run out; all repeats from that raw remain together. Candidate selection and LOO are development diagnostics only.',
        candidates=diagnostics,selection=dict(best_macro_run_log_rmse=best['name'],descriptive_one_SE_cutoff=cutoff,
            simplest_within_one_SE=preferred['name'],rule='Fewest coefficients within one descriptive run SE of minimum; no statistical superiority claim.'),
        supported_local_domain=domain,forbidden_paths=forbidden,uncovered_families=['generalist_1','generalist_3'],
        full_12_column_rank_required=False,full_seven_pool_coverage=False,
        no_claims=['independent validation','global deployment optimum','queue/e2e prediction','energy/memory predictor calibration','total utility accuracy'],
        fitted=True,predictor_activated=False,production_scoring_modified=False,gpu_started=False,
        raw_results_opened_by_this_tool=False,heldout_results_opened=False)
    save(out/'report.json',result);save(out/'run-loo-diagnostics.json',all_folds);save(out/'grouped-training-data.json',rows)
    (out/'README.zh.md').write_text('# 局部正值服务模型：仅开发拟合与按 run 留一\n\n'+
        f'严格请求 {len(strict)}，剔除仅1个run的路径状态后拟合 {len(eligible)} 请求，合并为 {len(rows)} 个 run/path/input 单元，独立 run {len(run_ids)}。\n\n'+
        '每run等权，run内各path等权，path内唯一输入等权；重复请求仅平均log残差。所有LOO整run留出。self/other只由方案已知闭环基线压力构造，约束其系数非负；exp保证服务预测正值。\n\n'+
        f'开发诊断最小宏平均误差：{best["name"]}；描述性1SE内最简单候选：{preferred["name"]}。\n\n'+
        '逐候选系数、run留一误差及训练输入/上下文外推标记见JSON。只允许记录的family/task/n/输入和记录上下文凸包；G1/G3未覆盖，ASR n3单run排除，image仅2run及输入不全标弱支撑。没有以12列满秩为人为前置。此处没有独立验证、生产激活、总效用或最优组合证明。\n')
    save(out/'manifest.json',dict(script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        inputs={str(inv/'manifest.json'):sha(inv/'manifest.json')},files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(fit_requests=len(eligible),grouped=len(rows),runs=len(run_ids),selection=result['selection'],
        candidates=[dict(name=c['name'],rmse=c['macro_run_log_rmse'],folds=c['successful_folds']) for c in diagnostics])))

if __name__=='__main__':main()
