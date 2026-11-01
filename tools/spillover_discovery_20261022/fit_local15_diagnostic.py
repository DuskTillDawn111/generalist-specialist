#!/usr/bin/env python3
"""Fit condition-level local linear service ratios, with honest rank-aware LOCO."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_development_reversal import ROOT, STUDY, SOURCE, receipt, save, sha, stats, fifo, pathkey
from audit_five_development_points import diagnostic_terms, errors
from gsdeploy.calibration import _import_source
from gsdeploy.calibration_v2 import input_service_ms
from gsdeploy.config import SystemConfig
from gsdeploy.prediction import components


def features(condition):
    return [1.,condition['G2_ASR_fraction'],condition['asr_rps']-1.8]


def fit(conditions):
    paths=sorted({p for c in conditions for p in c['per_path']})
    models={}
    for path in paths:
        rows=[c for c in conditions if path in c['per_path']]
        design=np.asarray([features(c) for c in rows])
        target=np.asarray([c['per_path'][path]['actual_over_frozen_total'] for c in rows])
        rank=int(np.linalg.matrix_rank(design))
        model={'training_ids':[c['id'] for c in rows],'design':design.tolist(),'target_ratios':target.tolist(),
            'rank':rank,'parameter_count':3,'status':'IDENTIFIED' if rank==3 else 'UNKNOWN_RANK_DEFICIENT',
            'coefficients':None,'condition_number':float(np.linalg.cond(design)) if rank==3 else None}
        if rank==3:
            beta=np.linalg.lstsq(design,target,rcond=None)[0]
            residual=design@beta-target
            model.update(coefficients=beta.tolist(),training_ratio_residuals=residual.tolist(),
                training_ratio_RMSE=float(np.sqrt(np.mean(residual**2))))
        models[path]=model
    return models


def replay(condition,raw,models,spec):
    coefficients={}
    unknown=[]
    for path in condition['per_path']:
        model=models.get(path,{})
        if model.get('coefficients') is None:
            unknown.append(path)
            continue
        value=float(np.dot(features(condition),model['coefficients']))
        if value<=0:
            unknown.append(path+':NONPOSITIVE_RATIO')
        else:coefficients[path]=value
    path_errors={}
    services=[]
    for path,ratio in coefficients.items():
        actual=condition['per_path'][path]['actual_over_frozen_total']
        path_errors[path]={'predicted_ratio':ratio,'actual_ratio':actual,'ratio_error':ratio-actual,
            'service_total_relative_error':abs(ratio-actual)/actual}
    base={'status':'UNKNOWN' if unknown else 'DIAGNOSTIC_ONLY','unknown_paths':unknown,'per_path_errors':path_errors,
        'production_U':None,'production_eligible':False,'diagnostic_score':None}
    if unknown:return base
    for r in raw['request_records']:
        family=raw['replica_to_family'][r['model']]
        baseline=input_service_ms(spec,family,r['task'],raw['plan']['deployment'][family],r['sample_id'])
        services.append(baseline*coefficients[pathkey(raw,r)])
    local_raw={**raw,'evaluation':{**raw['evaluation'],'utility':condition['measured_six_term_utility']}}
    result=fifo(local_raw,SystemConfig(**raw['config']),services)
    base['FIFO_measured_QME']=diagnostic_terms(result['terms_with_measured_QME'])
    base['FIFO_measured_QME_errors']=errors(base['FIFO_measured_QME'],condition['measured_six_term_utility'])
    frozen=condition['frozen_zero_refit_diagnostic_prediction']
    config=SystemConfig(**raw['config'])
    arithmetic=components(result['per_task'],config.workload,config.latency_reference_ms,
        result['terms_with_measured_QME']['T_norm'],frozen['M_norm'],config.weights,frozen['E_norm'],frozen['Q_norm'])
    base.update(diagnostic_terms(arithmetic))
    base['errors']=errors(base,condition['measured_six_term_utility'])
    base['per_task']=result['per_task']
    base['service_total_relative_error']=abs(sum(services)-sum(r['worker_latency_ms'] for r in raw['request_records']))/sum(r['worker_latency_ms'] for r in raw['request_records'])
    base['request_service_RMSE_ms']=float(np.sqrt(np.mean([(p-r['worker_latency_ms'])**2 for p,r in zip(services,raw['request_records'])])))
    return base


def decomposition(conditions,raws):
    byid={c['id']:c for c in conditions}
    results={}
    for load in ('1p8','1p9'):
        a,b=(byid['asr'+load+'_G2share'+s] for s in ('0','0p125'))
        rates=a['workload'];total=sum(rates.values());raw=raws[a['id']]
        weights=raw['config']['weights'];refs=raw['config']['latency_reference_ms']
        tasks={}
        for task in rates:
            ta,tb=a['per_task'][task],b['per_task'][task]
            tasks[task]={'mean_before_ms':ta['mean_e2e_latency_ms'],'mean_after_ms':tb['mean_e2e_latency_ms'],
                'p95_before_ms':ta['p95_e2e_latency_ms'],'p95_after_ms':tb['p95_e2e_latency_ms'],
                'delta_L_contribution':-weights['L']*rates[task]/total/refs[task]*(tb['mean_e2e_latency_ms']-ta['mean_e2e_latency_ms']),
                'delta_P_contribution':-weights['P']*rates[task]/total/refs[task]*(tb['p95_e2e_latency_ms']-ta['p95_e2e_latency_ms'])}
        delta={k:b['measured_six_term_utility'][k]-a['measured_six_term_utility'][k] for k in ['U']+[t+'_contribution' for t in 'QTLPME']}
        results[load]={'a':a['id'],'b':b['id'],'direction':'12.5percent G2 minus no split','deltas':delta,'per_task_latency':tasks}
    a,b=byid['asr1p8_G2share0p125'],byid['asr1p9_G2share0p125']
    results['same_fraction_load_path_comparison']={p:{'ratio_at_1p8':a['per_path'][p]['actual_over_frozen_total'],
        'ratio_at_1p9':b['per_path'][p]['actual_over_frozen_total'],
        'ratio_delta':b['per_path'][p]['actual_over_frozen_total']-a['per_path'][p]['actual_over_frozen_total'],
        'ratio_of_ratios':b['per_path'][p]['actual_over_frozen_total']/a['per_path'][p]['actual_over_frozen_total'],
        'nine_parameter_service_relative_error_1p8':abs(a['per_path'][p]['empirical_service_ms']['mean']-a['per_path'][p]['actual_service_ms']['mean'])/a['per_path'][p]['actual_service_ms']['mean'],
        'nine_parameter_service_relative_error_1p9':abs(b['per_path'][p]['empirical_service_ms']['mean']-b['per_path'][p]['actual_service_ms']['mean'])/b['per_path'][p]['actual_service_ms']['mean']}
        for p in a['per_path']}
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,default=STUDY/'five-development-point-audit-v1/audit.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if any('validation' in p or 'confirmation' in p for p in args.audit.parts):raise ValueError('Development audit only')
    audit=json.loads(args.audit.read_text());assert audit['status']=='COMPLETE_DEVELOPMENT_AUDIT'
    args.output.mkdir(parents=True,exist_ok=False)
    before={str(p):sha(p) for p in SOURCE.rglob('*.py')}
    conditions=audit['conditions'];raws={};coverage={}
    for c in conditions:
        source=c['source'];packet=json.loads(Path(source['packet_path']).read_text())
        prepared_id=json.loads(Path(source['path']).read_text())['experiment_job_id']
        job=next(j for j in packet['jobs'] if j['id']==prepared_id)
        raw,utility,_=_import_source(source,{job['identity']['runtime_code_sha256']})
        assert utility==c['measured_six_term_utility']
        raws[c['id']]=raw
        coverage[c['id']]={p:{'requests':len(rs),'unique_inputs':len({r['sample_id'] for r in rs}),
            'sample_ids':sorted({r['sample_id'] for r in rs})} for p in c['per_path']
            for rs in [[r for r in raw['request_records'] if pathkey(raw,r)==p]]}
    basis=json.loads(Path(audit['basis']['path']).read_text());assert sha(audit['basis']['path'])==audit['basis']['sha256']
    spec=basis['catalog']['calibrated_replay_v2']
    model=fit(conditions)
    assert len(model)==5 and sum(m['parameter_count'] for m in model.values())==15
    training={c['id']:replay(c,raws[c['id']],model,spec) for c in conditions}
    loco={}
    for held in conditions:
        train=[c for c in conditions if c['id']!=held['id']]
        local=fit(train)
        pred=replay(held,raws[held['id']],local,spec)
        fold_coverage={}
        for path,held_coverage in coverage[held['id']].items():
            training_samples=set().union(*(set(coverage[c['id']].get(path,{}).get('sample_ids',[])) for c in train))
            held_samples=set(held_coverage['sample_ids'])
            fold_coverage[path]={'training_unique_inputs':len(training_samples),
                'held_unique_inputs':len(held_samples),'held_inputs_seen_in_training':len(held_samples&training_samples),
                'held_unseen_inputs':sorted(held_samples-training_samples),
                'scope':'Sample overlap only; condition-level ratio has no fitted per-input parameters'}
        nine_applicable=held['id'] not in audit['training_model']['training_ids']
        loco[held['id']]={'held_id':held['id'],'training_ids':[c['id'] for c in train],
            'models':local,'prediction':pred,'sample_coverage':fold_coverage,
            'fixed_9parameter_baseline':{'status':'DIAGNOSTIC_ONLY' if nine_applicable else 'UNKNOWN_TRAINING_ENDPOINT_HELD',
                'diagnostic_score':held['empirical_9parameter_full_external_diagnostic']['diagnostic_score'] if nine_applicable else None,
                'absolute_utility_error':held['empirical_9parameter_errors']['U']['absolute'] if nine_applicable else None,
                'scope':'9-parameter coefficients retain the original two endpoints, so this comparison is leak-free only for three new conditions'}}
    summary={}
    for name,rows in [('training15',training),('LOCO15',{k:v['prediction'] for k,v in loco.items()})]:
        valid=[r for r in rows.values() if r['diagnostic_score'] is not None]
        summary[name]={'complete_conditions':len(valid),'unknown_conditions':len(rows)-len(valid),
            'mean_absolute_U_error':float(np.mean([r['errors']['U']['absolute'] for r in valid])) if valid else None,
            'max_absolute_U_error':max((r['errors']['U']['absolute'] for r in valid),default=None)}
    report={'schema':'spillover_local15_development_diagnostic.v1','status':'DEVELOPMENT_FIT_DIAGNOSTIC_ONLY',
        'audit':receipt(args.audit),'models':model,'design':{'features':['intercept','G2_ASR_fraction','ASR_rps_minus_1p8'],
            'target':'Per-condition per-path sum(actual worker service)/sum(frozen per-input service)',
            'fit':'Unweighted OLS across conditions for each path; no regularization, no nonidentifiable pseudo-inverse prediction',
            'parameters':15,'single_seed':51,'background_each_rps':.05,'fixed_deployment':{'asr':2,'generalist_2':1,'ocr':1}},
        'input_coverage':coverage,'training_predictions':training,'leave_one_condition_out':loco,'summary':summary,
        'benefit_decomposition':decomposition(conditions,raws),'finer_per_input_model_fitted':False,
        'parameters_used_for_production':False,'production_U':None,'production_eligible':False,'gpu_started':False,
        'validation_results_read':False,'confirmation_results_read':False,'frozen_sources_unchanged':all(sha(p)==h for p,h in before.items()),
        'limitations':['Five conditions from one seed do not identify stochastic variability or causal workload dependence',
            'G2/asr has exactly three active conditions and no residual degrees of freedom; LOCO on an active condition has rank2 and must be UNKNOWN',
            'Condition-level means do not fit input-dependent variation, burst correlation or service tail shape',
            'LOCO metrics apply only to complete identifiable folds; do not treat two complete folds as all-five validation',
            'Production activation, global search and hardware optimality claims remain forbidden for this diagnostic']}
    save(args.output/'report.json',report)
    lines=['# 1.9微正收益与15参数局部模型诊断','', '仅五条seed51开发raw，均重新严格导入。未读取独立验证或确认，未启动GPU。','',
        '## 微正收益来源','']
    for load,item in report['benefit_decomposition'].items():
        if load=='same_fraction_load_path_comparison':continue
        lines += [f"ASR{load.replace('p','.')}：12.5%分流减不分流 ΔU={item['deltas']['U']:.9f}。",'',
            '| Q贡献差 | T贡献差 | L贡献差 | P贡献差 | M贡献差 | E贡献差 |','|---:|---:|---:|---:|---:|---:|',
            '| '+' | '.join(f"{item['deltas'][t+'_contribution']:.9f}" for t in 'QTLPME')+' |','',
            '| 任务 | ΔL效用贡献 | ΔP效用贡献 | p95前ms | p95后ms |','|---|---:|---:|---:|---:|']
        for t,p in item['per_task_latency'].items():lines.append(f"| {t} | {p['delta_L_contribution']:.9f} | {p['delta_P_contribution']:.9f} | {p['p95_before_ms']:.3f} | {p['p95_after_ms']:.3f} |")
        lines.append('')
    lines += ['1.9微正收益依赖背景 text/image 的p95改善抵消均值延迟恶化和其他成本；ASR自身p95未改善。净差约0.00013，远小于各项正负抵消量。单seed无法区分运行噪声、负载相关真实服务变化或输入/到达交互。重复seed52/53配对和强扩容控制有必要；不能称已证明分流获益。','',
        '同12.5%分流下，1.8与1.9的逐路径冻结服务比及9参残差如下。G2 text与ASR倍率升高、G2 image倍率下降，不支持一个全局服务系数；也不足以单凭五条观察确定因果负载依赖。','',
        '| 路径 | 1.8实际/冻结服务 | 1.9实际/冻结服务 | 1.8的9参服务相对误差 | 1.9的9参服务相对误差 |','|---|---:|---:|---:|---:|']
    for p,v in report['benefit_decomposition']['same_fraction_load_path_comparison'].items():
        lines.append(f"| {p} | {v['ratio_at_1p8']:.6f} | {v['ratio_at_1p9']:.6f} | {v['nine_parameter_service_relative_error_1p8']:.6f} | {v['nine_parameter_service_relative_error_1p9']:.6f} |")
    lines += ['',
        '## 15参数模型与LOCO','', '| 路径 | 训练条件数 | 秩 | 截距 | 分流系数 | 负载系数 |','|---|---:|---:|---:|---:|---:|']
    for p,m in model.items():lines.append(f"| {p} | {len(m['training_ids'])} | {m['rank']} | "+' | '.join(f'{v:.6f}' for v in m['coefficients'])+' |')
    lines += ['', '| 条件 | 15参训练U误差绝对值 | 15参LOCO状态 | LOCO U误差绝对值 | 9参可比误差 |','|---|---:|---|---:|---:|']
    fmt=lambda v:'UNKNOWN' if v is None else f'{v:.9f}'
    for c in conditions:
        key=c['id'];fold=loco[key];pred=fold['prediction']
        lines.append(f"| {key} | {training[key]['errors']['U']['absolute']:.9f} | {pred['status']} | {fmt(pred.get('errors',{}).get('U',{}).get('absolute'))} | {fmt(fold['fixed_9parameter_baseline']['absolute_utility_error'])} |")
    lines += ['', 'G2 ASR恰好三个活动条件确定三参数；留出任何一个活动条件后rank2，完整效用必须UNKNOWN，未用最小范数伪逆补造。其余两折只预测不分流条件，并不能验证分流性能。9参固定原两端点，因此留出原端点时不能继续把其系数当作无泄漏LOCO。','',
        '目前可实现的结论仅限开发诊断。15参数增加负载项但没有足够独立结构检验G2 ASR，且倍率模型仍未解释逐输入尾部。建议先等待已安排重复配对，不立即将新模型正式化；若重复结果稳定，再预声明训练集、条件覆盖、全新验证seed和可接受误差。无需为本诊断加入更细逐输入参数。所有正式U与eligible保持禁用。','']
    (args.output/'README.zh.md').write_text('\n'.join(lines))
    save(args.output/'manifest.json',{'script':receipt(__file__),'files':{p.name:sha(p) for p in args.output.iterdir() if p.is_file()},'gpu_started':False,'validation_results_read':False})
    print(json.dumps({'output':str(args.output),'summary':summary},ensure_ascii=False))


if __name__=='__main__':main()
