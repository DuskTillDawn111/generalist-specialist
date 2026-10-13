"""Final independent CPU recalculation and explicit evidence coverage labels.

No fitting, optimization, model selection or GPU execution in this step.
"""
from .common import *
from .replay import replay
from .calibration import ResidualModel
from module20.optimizer import predict
from collections import defaultdict
from scipy.stats import spearmanr

def within_workload_ranking(rows,refs,weights,final_model):
    out=[]
    for version in ['A','B','B_AFTER_DEVELOPMENT_PROBES']:
        groups=defaultdict(lambda:defaultdict(list))
        for r in rows:
            if r['U'] is None:continue
            if version!='B_AFTER_DEVELOPMENT_PROBES' and r['module']==24:continue
            if version=='B_AFTER_DEVELOPMENT_PROBES':
                prediction,_=replay(r['catalog'],r['deployment'],r['routing'],r['rates'],r['budget_mib'],refs,weights,final_model,detail=False,horizon=r['planned_seconds'])
            else:prediction=r.get('prediction_'+version,{})
            key=identity([r['split'],r['module'],r['rates'],r['budget_mib'],r['hardware_uuid']])
            cfg=identity([r['deployment'],r['routing']])
            groups[key][cfg].append((r['U'],prediction.get('U')))
        for group_id,configs in groups.items():
            if len(configs)<2:continue
            targets={c:float(np.mean([v[0] for v in vs])) for c,vs in configs.items()}
            predictions={c:float(np.mean([v[1] for v in vs if v[1] is not None])) for c,vs in configs.items() if any(v[1] is not None for v in vs)}
            top=max(predictions,key=predictions.get) if predictions else None
            actual_best=max(targets.values());eligible=list(predictions)
            actual=[targets[c] for c in eligible];predicted=[predictions[c] for c in eligible]
            parsed=json.loads(group_id)
            out.append(dict(split=parsed[0],module=parsed[1],algorithm=version,workload_group=group_id,
                configurations=len(configs),predictable_configurations=len(predictions),
                actual_best_U=actual_best,selected_actual_U=targets[top] if top else None,
                top1_regret_U=actual_best-targets[top] if top else None,
                top1_correct=abs(actual_best-targets[top])<1e-12 if top else None,
                within_workload_spearman=float(spearmanr(actual,predicted).statistic) if len(set(actual))>1 and len(set(predicted))>1 else None,
                repetitions_aggregated_before_ranking=True,independent_validation=parsed[0]=='RETROSPECTIVE_COMBINATION_HOLDOUT'))
    out.append(dict(split='RETROSPECTIVE_COMBINATION_HOLDOUT',algorithm='ALL',
        configurations=0,status='NO_SAME_WORKLOAD_ALTERNATIVE_DEPLOYMENTS',independent_ranking_groups=0))
    return out

def main():
    original=read(OUT/'original_optimizer_audit.json');optimized=read(OUT/'reoptimized_three_phase.json')
    gate=read(OUT/'module24a_gate.json');frozen=read(OUT/'residual_model_final_development.json')
    model=ResidualModel();model.exact=frozen['exact'];model.path=frozen['path'];model.training_groups=frozen['training_groups']
    d,refs,weights=inputs();rows=read(OUT/'historical_observations.json')+read(OUT/'probe_observations.json')
    checks=[]
    for st in RATES:
        for label,w in [('original',original['stages'][st]['winner']),('calibrated',optimized['stages'][st]['new_winner'])]:
            if not w:
                checks.append(dict(stage=st,algorithm=label,no_eligible_winner=True));continue
            x=w['deployment'];rr=w['routing'];rates=RATES[st]
            assert all(str(n) in d['capacity_factors'][m] and str(n) in d['memory_table_mib'][m] for m,n in x.items())
            assert all(abs(sum(rr[t].values())-q)<1e-6 for t,q in rates.items())
            assert all(m in x and t in d['models'][m]['supported_tasks'] for t,route_row in rr.items() for m,q in route_row.items() if q>1e-9)
            if label=='original':p=predict(d,x,rr,rates,BUDGET,refs,weights,phase=.37)
            else:p,_=replay(d,x,rr,rates,BUDGET,refs,weights,model)
            assert p['eligible']==w['prediction']['eligible']
            assert abs(p['U']-w['prediction']['U'])<1e-10
            for t in rates:
                assert abs(p['per_task'][t]['mean_ms']-w['prediction']['per_task'][t]['mean_ms'])<1e-7
                assert abs(p['per_task'][t]['p95_ms']-w['prediction']['per_task'][t]['p95_ms'])<1e-7
            checks.append(dict(stage=st,algorithm=label,U_recalculated=p['U'],U_saved=w['prediction']['U'],passed=True))
            if label=='calibrated':
                stage=optimized['stages'][st]
                stage['workload_group_observed_in_development']=group(x,rates) in model.training_groups
                stage['exact_replica_deployment_observed_in_GPU_history']=any(r['deployment']==x and r['measurement_available'] for r in rows)
                stage['exact_deployment_and_workload_observed']=any(r['deployment']==x and r['rates']==rates and r['measurement_available'] for r in rows)
                stage['unsupported_calibration_paths']=p['unsupported_paths']
                stage['evidence_scope']='Checkpoint combination, replica counts, workload and route must be assessed separately; an exact combination parameter alone does not establish workload transfer.'
        # Compare algorithm-selected deployments under the same predictor, in
        # addition to the previously reported route-only pure/mixed contrast.
        stage=optimized['stages'][st];old=stage['old_winner'];new=stage['new_winner']
        if new:
            ao=predict(d,old['deployment'],old['routing'],RATES[st],BUDGET,refs,weights,phase=.37)
            an=predict(d,new['deployment'],new['routing'],RATES[st],BUDGET,refs,weights,phase=.37)
            bo,_=replay(d,old['deployment'],old['routing'],RATES[st],BUDGET,refs,weights,model)
            bn,_=replay(d,new['deployment'],new['routing'],RATES[st],BUDGET,refs,weights,model)
            delta=bn['U']-bo['U'] if bn['U'] is not None and bo['U'] is not None else None
            env=stage['empirical_development_error_envelope_U']
            stage['algorithm_choice_comparison']=dict(old_choice_U_under_A=ao['U'],new_choice_U_under_A=an['U'],
                old_choice_U_under_B=bo['U'],new_choice_U_under_B=bn['U'],
                predicted_new_minus_old_delta_U_A=an['U']-ao['U'] if an['U'] is not None and ao['U'] is not None else None,
                predicted_new_minus_old_delta_U_B=delta,
                delta_U_empirical_stress_interval_NOT_CI=[delta-2*env,delta+2*env] if delta is not None and env is not None else None,
                actual_algorithm_selected_deployment_delta_U=None,independent_GPU_comparison_performed=False,
                limitation='Nominal advantage may reflect uncalibrated replica paths reverting to standalone values; unsupported deployment counts/workloads preclude a credible gain claim.')
    rawobs=[r for r in read(OUT/'probe_observations.json') if r['measurement_available']]
    signatures=[];releases=[];counts={}
    for ob in read(OUT/'probe_observations.json'):
        raw=read(ob['raw']['path']);counts[raw['slot']['arm']]=counts.get(raw['slot']['arm'],0)+1
        releases.append(raw['gpu_release']['verified'])
        if ob['measurement_available']:
            signatures.append([(r['task'],r['pool_id'],r['sample_id'],r['frozen_offset_seconds'],r['routing_uniform'],r['replica_uniform'])
                              for r in raw['phases']['measurement']['request_records']])
            assert len({v['pid'] for v in raw['worker_identities'].values()})==5
    assert all(n<=2 for n in counts.values())
    assert len(signatures)==2 and signatures[0]==signatures[1]
    assert all(releases)
    assert not gate['Module24B_executed'] and not gate['independent_ranking_supported']
    assert optimized['weights']==weights and optimized['rates']==RATES and optimized['budget_mib']==BUDGET
    held={r['combo'] for r in rows if r['split']=='RETROSPECTIVE_COMBINATION_HOLDOUT'}
    trained={r['combo'] for r in rows if r['split']=='DEVELOPMENT'}
    assert not held & trained
    before=read(OUT/'historical_metadata_before.json');after=metadata()
    assert before==after
    verdict=dict(CPU_recalculations=checks,matched_probe_offered_records_identical=True,
        per_route_GPU_attempts=counts,all_GPU_memory_releases_verified=True,holdout_combo_disjoint=True,
        historical_metadata_unchanged=True,Module24B_not_executed=True,all_checks_passed=True)
    save('final_verification.json',verdict);save('reoptimized_three_phase.json',optimized)
    gate['final_CPU_recalculation_passed']=True;gate['matched_probe_trace_verified']=True;gate['GPU_released']=True
    gate['per_route_GPU_attempts']=counts;save('module24a_gate.json',gate)
    byuid={r['uid']:r for r in rows};rawslots={r['uid']:read(r['raw']['path'])['slot'] for r in rows}
    utility=list(csv.DictReader((OUT/'utility_prediction_residuals.csv').open()))
    for row in utility:
        uid=row['uid'];logged=rawslots[uid].get('CPU_predicted_U')
        row['historical_logged_predicted_U']=logged
        row['historical_log_predictor']='Module21 Poisson/FFT' if byuid[uid]['module'] in [21,22] else 'Module24 diagnostic Module20 periodic' if byuid[uid]['module']==24 else 'Historical U prediction unavailable (J/capacity only)'
        u=byuid[uid]['U'];row['actual_minus_historical_logged_U']=u-logged if u is not None and logged is not None else None
    csvsave('utility_prediction_residuals.csv',utility)
    negatives=list(csv.DictReader((OUT/'negative_case_development_acceptance.csv').open()))
    for row in negatives:
        a=rawslots[row['proposed_uid']].get('CPU_predicted_U');b=rawslots[row['control_uid']].get('CPU_predicted_U')
        row['historical_logged_predicted_delta_U']=a-b if a is not None and b is not None else None
        row['historical_logged_predictor']='Module21 Poisson/FFT' if row['module']=='22' else 'Historical U unavailable'
    csvsave('negative_case_development_acceptance.csv',negatives)
    ranking=within_workload_ranking(rows,refs,weights,model);csvsave('within_workload_ranking_validation.csv',ranking)
    comparison=list(csv.DictReader((OUT/'prediction_model_comparison.csv').open()))
    for row in comparison:
        rr=[r for r in ranking if r.get('algorithm')==row['algorithm'] and r['split']==row['split'] and r.get('top1_correct') is not None]
        row['within_workload_ranking_groups']=len(rr)
        row['within_workload_top1_accuracy']=float(np.mean([r['top1_correct'] for r in rr])) if rr else None
        row['within_workload_mean_top1_regret_U']=float(np.mean([r['top1_regret_U'] for r in rr])) if rr else None
        row['within_workload_configs_abstained']=sum(r['configurations']-r['predictable_configurations'] for r in rr)
    csvsave('prediction_model_comparison.csv',comparison)
    summary=OUT/'module24a_summary.md';text=summary.read_text().split('\n## 最终候选的实测覆盖范围',1)[0]
    text+='\n## 最终候选的实测覆盖范围\n\n'
    text+='探针前冻结历史模型的排序留出见 ranking_holdout_validation.csv；开发探针更新后的分组排序另见 ranking_after_development_probes.csv，Module24 行仅为 DEVELOPMENT。\n\n'
    text+='| 阶段 | 新候选副本数 | workload 在开发中出现 | 精确部署+workload 已测 | 未获当前共置校准的路径数 |\n|---|---|---|---|---:|\n'
    for st,s in optimized['stages'].items():
        w=s['new_winner']
        if not w:text+=f'| {st} | 无 eligible 预测 | False | False | — |\n';continue
        names={G:'Qwen',S:'Whisper-large',TEXT:'RoBERTa',IMAGE:'ConvNeXt',OCR:'PARSeq'}
        deployment=', '.join(f'{names[m]}×{n}' for m,n in w['deployment'].items())
        text+=f"| {st} | {deployment} | {s['workload_group_observed_in_development']} | {s['exact_deployment_and_workload_observed']} | {len(s['unsupported_calibration_paths'])} |\n"
    text+='\n未覆盖 workload/副本的回退是带未知误差的名义预测；不能用缺少校准的副本看起来更快来认定收益。整组合留出、服务测量、路由配对和复制数支持分别报告。六个新旧阶段候选已从保存的路由独立重算 U/mean/P95；全部一致。全部 GPU 进程/显存已释放，原始历史文件未变。\n'
    text+='\n原算法/新算法部署差异还需在同一个预测器下比较；上表的 ΔU 是新选择与最佳纯路由的对比，以下另报告新选择−旧选择（实测 ΔU 未测）：\n\n| 阶段 | A 预测新−旧 ΔU | B 预测新−旧 ΔU | B 经验 stress 区间（非 CI） |\n|---|---:|---:|---|\n'
    for st,s in optimized['stages'].items():
        c=s.get('algorithm_choice_comparison',{})
        text+=f"| {st} | {c.get('predicted_new_minus_old_delta_U_A')} | {c.get('predicted_new_minus_old_delta_U_B')} | {c.get('delta_U_empirical_stress_interval_NOT_CI')} |\n"
    pair=read(OUT/'high_probe_paired_diagnostic.json')
    text+=f"\n单次开发诊断：GS18−S 的实际 ΔU={pair['actual_delta_U_GS18_minus_S']:.6f}，旧预测 ΔU={pair['old_predicted_delta_U']:.6f}。收益分解 mean={pair['component_deltas']['L_contribution']:.6f}、P95={pair['component_deltas']['P_contribution']:.6f}、throughput={pair['component_deltas']['T_contribution']:.6f}、memory={pair['component_deltas']['M_contribution']:.6f}；汇总 pooled mean/P95 几乎不变，不能替代按任务参考值和固定 offered 权重定义的 U。只有一次短配对，CI 未估计，独立验证=false。\n"
    text+=f"\n探针更新后的 B 对同驻留 GS18−S 的预测 ΔU={pair['updated_model_predicted_delta_U_development']:.6f}，仍与实际方向相反；路径中位 service/queue 残差不足以预测路由引起的全系统干扰变化。\n"
    text+='\nA 表示重新计算的固定 Module20A 周期预测器。Module22 原实验日志使用 Module21 的 Poisson/FFT 预测器，不能混作 A；两种原预测均保留在 utility_prediction_residuals.csv 和 negative_case_development_acceptance.csv 的 historical_logged_* 字段。\n'
    text+='\n同一 workload 内的方案排序另见 within_workload_ranking_validation.csv：先合并整次重复会话，再比较部署/路由配置的真实 U，报告 top-1 命中、收益后悔值、配置拒绝预测数和秩相关。留出依然没有可用排序组；开发排序不是独立泛化证据。\n'
    text+='\n| Module22 HIGH 负例 | 原实验日志预测 ΔU | 固定20A重算 ΔU | 实际 ΔU |\n|---|---:|---:|---:|\n'
    for row in negatives:
        if row['module']=='22' and row['case']=='HIGH':text+=f"| rep{row['repetition']} | {row['historical_logged_predicted_delta_U']} | {row['predicted_delta_U_A']} | {row['actual_delta_U']} |\n"
    summary.write_text(text)
    save('execution_source_manifest.json',[ref(p) for p in sorted((ROOT/'module24').glob('*.py'))])
    required=['original_optimizer_audit.json','service_time_residuals.csv','utility_prediction_residuals.csv',
        'prediction_model_comparison.csv','ranking_holdout_validation.csv','high_colocation_probe.csv',
        'reoptimized_three_phase.json','module24a_gate.json','module24a_summary.md']
    save('output_integrity.json',dict(required=[ref(OUT/n) for n in required],all_required_present=True,
        final_verification=ref(OUT/'final_verification.json')))
    print('FINAL_VERIFICATION_PASSED',json.dumps(verdict,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
