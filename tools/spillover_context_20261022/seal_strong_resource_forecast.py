"""Seal development M/E forecasts from frozen plans/models; never read run results."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/static-spillover-study-20261022'
MODEL_DIRECTORY=BASE/'context-resource-candidates-dev-v1/fit-v1'
PACKET=BASE/'text185-strong-controls-v1/experiment-packet.json'
INVENTORY=BASE/'context-design-inventory-v5'
DESTINATION=BASE/'strong-controls-resource-prequential-forecast-v1'
M_CONTEXT=['raw_resident_over_budget','workers']
E_CONTEXT=['sum_baseline_pressure','workers']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def hull_check(rows,features,names):
    missing=[k for k in names if features.get(k) is None or not isinstance(features[k],(int,float)) or not math.isfinite(features[k])]
    if missing:return dict(status='UNKNOWN_MISSING_INPUT',inside=None,missing_features=missing,features=names)
    if not rows:return dict(status='UNKNOWN_NO_TRAINING_CONTEXT',inside=None,features=names)
    if any(any(r.get('features',{}).get(k) is None for k in names) for r in rows):
        return dict(status='UNKNOWN_MISSING_TRAINING_FEATURE',inside=None,features=names)
    x=np.asarray([[r['features'][k] for k in names] for r in rows],dtype=float);point=np.asarray([features[k] for k in names])
    result=linprog(np.zeros(len(rows)),A_eq=np.vstack([x.T,np.ones(len(rows))]),b_eq=np.r_[point,1.],bounds=(0,None),method='highs')
    # Solver failure is different from its mathematical infeasible verdict.
    if result.status not in (0,2):return dict(status='UNKNOWN_SOLVER_STATUS',inside=None,features=names,solver_status=int(result.status))
    weights=result.x.tolist() if result.status==0 else None
    return dict(status='INSIDE_OBSERVED_FEATURE_HULL' if result.status==0 else 'OUTSIDE_OBSERVED_FEATURE_HULL',inside=result.status==0,
        features=names,point=point.tolist(),training_axis_ranges={k:[min(r['features'][k] for r in rows),max(r['features'][k] for r in rows)] for k in names},
        training_run_ids=[r['run_id'] for r in rows],convex_weights=weights,solver_status=int(result.status),
        scope='Descriptive feature support only; inside hull is not independent validation or composition coverage')


def derive_features(packet,plan,baselines):
    catalog=packet['config']['catalog'];deployment=plan['deployment'];budget=packet['config'].get('gpu_memory_mib')
    unknown=[];memory=[];pressure=[]
    for family,n in deployment.items():
        value=catalog.get('memory_table_mib',{}).get(family,{}).get(str(n))
        if value is None:unknown.append(dict(kind='MISSING_RAW_RESIDENCY',family=family,n=n))
        else:memory.append(dict(family=family,n=n,raw_mib=value))
    for task,row in plan['routing'].items():
        for family,rate in row.items():
            if rate<=0:continue
            n=deployment[family];matching=[b for b in baselines if (b['family'],b['task'],b['n'])==(family,task,n)]
            factor=catalog.get('capacity_factors',{}).get(family,{}).get(str(n))
            if len(matching)!=1 or matching[0].get('input_median_mean_ms') is None or matching[0].get('missing') or factor is None or factor<=0:
                unknown.append(dict(kind='MISSING_EXACT_BASELINE_OR_FACTOR',family=family,task=task,n=n));continue
            baseline=matching[0]
            pressure.append(dict(family=family,task=task,n=n,routing_rps=rate,mean_input_median_service_ms=baseline['input_median_mean_ms'],
                capacity_factor=factor,pressure=rate*baseline['input_median_mean_ms']/1000/factor,baseline_source=baseline['source']))
    features=dict(intercept=1.,workers=sum(deployment.values()),
        raw_resident_over_budget=sum(r['raw_mib'] for r in memory)/budget if len(memory)==len(deployment) and budget else None,
        sum_baseline_pressure=sum(r['pressure'] for r in pressure) if not any(r['kind']=='MISSING_EXACT_BASELINE_OR_FACTOR' for r in unknown) else None)
    return features,dict(raw_residency_terms=memory,baseline_pressure_terms=pressure,unknown=unknown,
        no_missing_value_fill=True,baseline_definition='Frozen inventory mean per-input median / original capacity_factor, not formal profile mean rho and not new native service')


def forecast_model(model,features,training_rows,deployment):
    names=model['features'];missing=[k for k in names if features.get(k) is None or not isinstance(features[k],(int,float)) or not math.isfinite(features[k])]
    target=model['target'];context=E_CONTEXT if target.startswith('E_') else M_CONTEXT
    # Complete context is descriptive. Each candidate's own feature hull is
    # recorded independently; no additional shared 3D applicability gate exists.
    full=hull_check(training_rows,features,context)
    projection=hull_check(training_rows,features,names)
    seen=any(r['deployment']==deployment for r in training_rows)
    reasons=[]
    if missing:reasons.append('MISSING_MODEL_INPUT')
    if projection.get('inside') is not True:reasons.append('OUTSIDE_OR_UNKNOWN_MODEL_FEATURE_HULL')
    if not seen:reasons.append('UNOBSERVED_DEPLOYMENT_COMPOSITION')
    beta=model['all_run_fit'].get('coefficients')
    if beta is None:reasons.append('MODEL_COEFFICIENTS_UNKNOWN')
    numeric_inputs=not missing and all(isinstance(features[k],(int,float)) and math.isfinite(features[k]) for k in names)
    numeric_beta=beta is not None and len(beta)==len(names) and all(math.isfinite(b) for b in beta)
    probe=float(np.dot(beta,[features[k] for k in names])) if numeric_inputs and numeric_beta else None
    value=probe if not reasons else None
    return dict(model_name=model['name'],target=target,features=names,coefficients=beta,
        status='UNKNOWN_UNSUPPORTED_CONTEXT' if reasons else 'DEVELOPMENT_FORECAST_NOT_VALIDATED',
        forecast=value,missing_inputs=missing,reasons=reasons,full_context_hull=full,model_feature_projection_hull=projection,
        deployment_observed=seen,formula_extrapolation_evaluated=probe is not None and bool(reasons),
        diagnostic_extrapolation_only=dict(value=probe,status='CPU_DEVELOPMENT_NUMERICAL_PROBE' if probe is not None else 'UNKNOWN_MISSING_INPUT_OR_COEFFICIENT',
            candidate_feature_hull_status=projection['status'],valid_prediction=False,formal_score=False,execution_admission=False,
            purpose='Pre-result numerical hypothesis for later transfer-error evaluation, including unsupported contexts; never activate or substitute into U'),
        nonnegative_is_not_safe_upper_bound=True,model_selection_performed=False,activated=False)


def prepare(destination=DESTINATION):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError(destination)
    opened={}
    def read(path,expected=None):
        path=Path(path).resolve()
        if expected is not None and sha(path)!=expected:raise ValueError('Frozen file changed: '+str(path))
        opened[str(path)]=sha(path);return json.loads(path.read_text())
    manifest=read(MODEL_DIRECTORY/'manifest.json')
    report=read(MODEL_DIRECTORY/'report.json',manifest['files']['report.json'])
    fields=read(MODEL_DIRECTORY/'run-fields.json',manifest['files']['run-fields.json'])
    protocol=read(MODEL_DIRECTORY/'protocol.json',manifest['files']['protocol.json'])
    inventory_manifest=read(INVENTORY/'manifest.json')
    baselines=read(INVENTORY/'baseline-inventory.json',inventory_manifest['files']['baseline-inventory.json'])
    if opened[str(INVENTORY/'manifest.json')]!=manifest['inputs'][str(INVENTORY/'manifest.json')]:raise ValueError('Training inventory changed')
    packet=read(PACKET)
    if report['run_count']!=19 or report['predictor_activated'] is not False or report['reads_108_or_reserved_results'] is not False:
        raise ValueError('Frozen candidate scope changed')
    jobs=packet['jobs'];plan=jobs[0]['plan']
    expected=dict(deployment={'text':2,'image':1,'ocr':1,'asr':1},routing={'text':{'text':1.85},'image':{'image':.01},'ocr':{'ocr':.01},'asr':{'asr':.01}})
    if plan!=expected or len(jobs)!=2 or [j['seed'] for j in jobs]!=[107,108] or any(j['plan']!=plan for j in jobs):raise ValueError('Expected frozen paired strong plan')
    if packet['config']['gpu_memory_mib']!=49140 or packet['config']['catalog']['safe_factor']!=1.110854:raise ValueError('Original memory contract changed')
    features,derivation=derive_features(packet,plan,baselines)
    rows=fields['rows']
    if len(rows)!=19 or any(r['seed'] in {108,61,62,71,72,73,81,82} for r in rows):raise ValueError('Forbidden training seed metadata')
    models={name:forecast_model(model,features,rows,plan['deployment']) for name,model in report['models'].items()}
    pipelines=[]
    for pipeline in report['E_pipelines']:
        window=models[pipeline['window_model']];startup=models[pipeline['startup_model']]
        w=window['forecast'];s=startup['forecast']
        pw=window['diagnostic_extrapolation_only']['value'];ps=startup['diagnostic_extrapolation_only']['value']
        pipelines.append(dict(name=pipeline['name'],window_model=window['model_name'],startup_model=startup['model_name'],
            status='UNKNOWN_COMPONENT_FORECAST' if w is None or s is None else 'DEVELOPMENT_FORECAST_NOT_VALIDATED',
            formal_window_watts=w,startup_joules=s,E_norm=(w+s/3600)/450 if w is not None and s is not None else None,
            E_window_joules=w*600 if w is not None else None,activated=False,
            diagnostic_extrapolation_only=dict(window_watts=pw,startup_joules=ps,
                E_norm=(pw+ps/3600)/450 if pw is not None and ps is not None else None,
                E_window_joules=pw*600 if pw is not None else None,
                valid_prediction=False,formal_score=False,execution_admission=False)))
    memory=[]
    for name in protocol['M_candidates']:
        value=models[name]['forecast']
        probe=models[name]['diagnostic_extrapolation_only']['value']
        memory.append(dict(model_name=name,status=models[name]['status'],raw_peak_over_budget=value,
            predicted_owned_peak_mib=value*49140 if value is not None else None,
            M_norm=value*1.110854 if value is not None else None,
            safe_memory_mib=value*49140*1.110854 if value is not None else None,
            physical_memory_feasibility='UNKNOWN',safe_factor=1.110854,
            diagnostic_extrapolation_only=dict(raw_peak_over_budget=probe,
                owned_sampled_peak_mib=probe*49140 if probe is not None else None,
                M_norm=probe*1.110854 if probe is not None else None,
                safe_memory_mib=probe*49140*1.110854 if probe is not None else None,
                valid_prediction=False,formal_score=False,execution_admission=False)))
    # Counts use declared workload/seed Poisson metadata only, not result rows.
    import sys
    source=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5';sys.path.insert(0,str(source))
    from gsdeploy.trace import make_trace,trace_digest
    from gsdeploy.runtime.workload import SamplePool,largest_inputs
    pool_path=Path(packet['inputs'])/'runtime-inputs.jsonl';opened[str(pool_path)]=sha(pool_path)
    startup_metadata=[]
    for job in jobs:
        pool=SamplePool(pool_path,list(packet['config']['workload']),job['seed']);priming=[]
        for family,n in sorted(plan['deployment'].items()):
            for replica in range(n):
                for task in packet['config']['catalog']['models'][family]['supported_tasks']:
                    if packet['config']['workload'].get(task,0)>0:
                        priming.extend(dict(alias=f'{family}#{replica}',task=task,pool_id=r['pool_id']) for r in largest_inputs(pool,task,3))
        trace=make_trace(packet['config']['workload'],30,job['seed'],'poisson')
        startup_metadata.append(dict(seed=job['seed'],priming_count=len(priming),priming_roster=priming,
            warmup_seconds=30,warmup_task_counts=dict(Counter(t for _,t in trace)),warmup_trace_sha256=trace_digest(trace),
            actual_loading_seconds=None,actual_priming_seconds=None,actual_warmup_drain_seconds=None,actual_startup_joules=None,
            reason='Plan/cohort counts only; no actual run results read'))
    forecast=dict(schema='gsdeploy.strong_resource_prequential_forecast.v1',status='SEALED_BEFORE_STRONG_RESULTS_READ',
        plan=plan,seeds=[107,108],same_plan_features_imply_same_static_candidate_forecast=True,features=features,feature_derivation=derivation,
        domain=dict(M_context=hull_check(rows,features,M_CONTEXT),E_context=hull_check(rows,features,E_CONTEXT),
            joint_context=hull_check(rows,features,['raw_resident_over_budget','sum_baseline_pressure','workers']),
            deployment_observed=any(r['deployment']==plan['deployment'] for r in rows),
            frozen_model_applicability=report['domain']['applicability'],
            gate='Each candidate records its actual feature hull separately; complete 2D/3D context hulls are descriptive, not shared admission gates. Frozen unseen-composition UNKNOWN scope retained. Numerical probes do not require observed composition and are never valid predictions or execution gates.'),
        models=models,memory_forecasts=memory,E_pipelines=pipelines,declared_startup_metadata=startup_metadata,
        limitations=['Formal forecasts remain UNKNOWN for unsupported composition or candidate feature support. Separate pre-result CPU numerical probes evaluate every complete frozen formula, including outside-domain cases, for later transfer-error measurement; no missing input is filled.',
            'Same plan gives equal static forecasts, but actual same-deployment startup energy need not match. It integrates serial load, priming and warmup/drain before measurement, affected by initialization/caches, input-dependent inference, power state and elapsed time.',
            'Priming largest3 inputs and their counts are identical across paired seeds. Thirty-second warmup uses separate seeded Poisson task/input trace, so actual warmup counts differ; counts alone cannot explain joules or identify loading/priming/warmup subphase causes.',
            'Even equal request counts and inputs cannot imply equal board energy; no startup time or per-stage power model was fitted.',
            'No comparison to strong queue/raw/live/actual results occurred. Later root-provided terminal receipts authorize a separate new validation artifact; this seal remains unchanged.',
            'Fitted memory is not physical evidence; original safe factor/score remain. No complete U calculation, ranking, model selection or activation.'],
        actual_results_read=[],strong_queue_read=False,reads_108_actual_results=False,GPU_started=False,gpu_queried=False,
        predictor_activated=False,full_predicted_U=None,production_U=None)
    destination.mkdir(parents=True,exist_ok=False)
    save(destination/'forecast.json',forecast)
    text='''# 强专 text2 部署 M/E 预序封存

只读取冻结的19-run候选、训练域字段、baseline inventory与strong计划；未读strong队列、raw、live或实际结果，没有GPU操作。保留全部13个模型及8种E组合，无新增结果选型。

计划是text2/image1/OCR1/ASR1，共5 workers，raw residency/49140=0.7765567765567766，baseline demand=0.77830474315875。M完整(residency,workers)、E完整(demand,workers)及联合3D上下文均在训练凸包外，各简化候选按实际使用的特征另记凸包，不把完整上下文凸包新增为共同门禁。冻结报告声明未见部署组合保持UNKNOWN；该部署未见于19-run训练集，因此正式forecast全部UNKNOWN/null。

另行冻结全部13个模型和8个E组合的diagnostic_extrapolation_only数值探针，完整输入时直接代入原系数，即使域外也保留数值供终态后检验迁移误差。它们是CPU开发假设，valid_prediction、formal_score、execution_admission均为false，不得作为正式U或执行准入，不填补缺失输入、不激活任何模型。

两个seed计划相同，静态候选应给相同预测资格；其实际startup焦耳不保证相同。原startup包含串行load、largest3 priming和30秒warmup直至measurement开始（含warmup drain），受初始化/缓存、实际耗时和整板功率影响。两seed largest3 priming同为15条且输入相同；seed107 warmup59条text，seed108为61条text＋1条ASR。请求数只能描述差别，不能解释startup焦耳、替代未测耗时/功率或拆分子阶段；即使同请求数同输入也不能推出同能耗。

完整forecast.json逐候选保存系数、缺字段检查、完整上下文与投影凸包检查、UNKNOWN原因和原safe-factor换算字段。manifest.json封存所有实际读取的来源SHA；不对未读raw做结果检查。未来仅在root给出授权终态收据后创建新验证记录，当前封存不回写。无正式激活、物理可行性证书或完整U。
'''
    with (destination/'README.zh.md').open('x') as stream:stream.write(text)
    opened[str(Path(__file__).resolve())]=sha(__file__)
    test=ROOT/'tests/test_strong_resource_forecast.py'
    if test.exists():opened[str(test)]=sha(test)
    for p in (source/'gsdeploy/trace.py',source/'gsdeploy/runtime/workload.py'):
        opened[str(p)]=sha(p)
    if any(sha(p)!=s for p,s in opened.items()):raise ValueError('Source changed while sealing')
    save(destination/'manifest.json',dict(schema='gsdeploy.prequential_resource_forecast_manifest.v1',inputs=opened,
        files={p.name:sha(p) for p in destination.iterdir() if p.is_file()},source_results_opened=[],strong_queue_or_raw_read=False,
        all_inputs_unchanged=True,GPU_started=False,gpu_queried=False,predictor_activated=False))
    print(json.dumps(dict(output=str(destination),forecast=ref(destination/'forecast.json'),manifest=ref(destination/'manifest.json'),
        models=len(models),all_model_predictions_unknown=all(v['forecast'] is None for v in models.values()),GPU_started=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DESTINATION)
    prepare(parser.parse_args().output)
