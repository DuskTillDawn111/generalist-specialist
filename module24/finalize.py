"""Reoptimize the identical pool; freeze evidence and stop before Module24B."""
from .common import *
from .replay import replay
from .calibration import ResidualModel,scalar_residuals,compare,pairs
from .audit import search
from module20.optimizer import predict

def enrich_original(d,refs,weights):
    original=read(OUT/'original_optimizer_audit.json');before=ref(OUT/'original_optimizer_audit.json')
    for stage,st in original['stages'].items():
        fit=st['winner'];x=fit['deployment'];rates=st['rates'];mode_detail={}
        z=fit['shares'];active_G_count=sum(p['service']['count'] for p in st['predicted_worker_paths'].values() if p['model']==G and p['task']=='asr')
        st['literal_float_selection']=st['choice'];st['predicted_G_ASR_requests']=active_G_count
        equivalents=[]
        for name,c in st['route_class_best'].items():
            if c and abs(c['prediction']['U']-fit['prediction']['U'])<1e-10:equivalents.append(name)
        st['numerically_equivalent_route_classes_tolerance_1e10']=equivalents
        if st['choice']=='G/S' and active_G_count==0:
            st['operational_interpretation']='S-only numerical tie; zero G-ASR simulated requests. Tiny positive G flow is retained, not credited as useful split.'
            st['reason']='Pure S and tiny G/S U differ only by floating roundoff; original free MILP selects S-only. The unchanged 120s periodic surrogate skips rare positive routes.'
        for name,f in [('G-only',1.),('S-only',0.),('GS18_diagnostic',.18)]:
            dx=dict(x)
            if f<1 and S not in dx:dx[S]=1
            zz=[z[0],z[1],f];rr=route(d,dx,rates,zz)
            p,ps=replay(d,dx,rr,rates,BUDGET,refs,weights)
            capacities={}
            for m in [G,S]:
                if m not in dx:continue
                n=dx[m];cg=d['capacity_factors'][m][str(n)]
                background=sum(rr[t].get(m,0)*d['models'][m]['service_time_ms'][t]/1000 for t in rates if t!='asr' and t in d['models'][m]['supported_tasks'])
                capacities[m]=max(0,cg-background)/(d['models'][m]['service_time_ms']['asr']/1000)
            mode_detail[name]=dict(deployment=dx,routing=rr,prediction=p,predicted_paths=ps,
                ASR_residual_capacity_QPS=capacities,diagnostic_only=name=='GS18_diagnostic',
                service_assumptions_ms={m:{t:s*dx[m]/d['capacity_factors'][m][str(dx[m])] for t,s in d['models'][m]['service_time_ms'].items()} for m in dx})
        st['same_background_route_diagnostics']=mode_detail
        for c in st['candidates']:
            x=c['deployment'];zz=[0. if sp in x else 1. for sp in [TEXT,IMAGE,S]]
            rr=route(d,x,rates,zz);p,ps=replay(d,x,rr,rates,BUDGET,refs,weights)
            c['canonical_endpoint_diagnostic']=dict(prediction=p,paths=ps,sortable=p['eligible'])
        st['finite_window_route_coverage_warning']='Positive rates below one request/window may have zero periodic requests; ASR LOW has ~2.4 expected Poisson arrivals. Finite surrogate ties do not certify benefit.'
    original['audit_before_diagnostic_enrichment']=before
    original['reported_numerical_tie_tolerance']=1e-10
    original['raw_optimizer_results_preserved']=True
    save('original_optimizer_audit.json',original)
    return original

def negative_acceptance(rows):
    lookup={r['uid']:r for r in rows};out=[]
    comparisons=list(csv.DictReader((ROOT/'results/module20_utility_optimizer/prediction_ranking_validation.csv').open()))
    for c in comparisons:
        if not c.get('GS_run') or not c.get('S_run'):continue
        a=lookup.get('M19:'+c['GS_run']);b=lookup.get('M19:'+c['S_run'])
        if not a or not b:continue
        out.append(dict(module=19,case=c['case_id'],repetition=a['repetition'],a=a,b=b))
    for st in ['MID','HIGH']:
        for rep in [1,2,3]:
            a=lookup.get(f'M22:{st.lower()}_proposed_rep{rep}');b=lookup.get(f'M22:{st.lower()}_control_rep{rep}')
            if a and b:out.append(dict(module=22,case=st,repetition=rep,a=a,b=b))
    result=[]
    for c in out:
        a=c.pop('a');b=c.pop('b');actual=a['U']-b['U'] if a['U'] is not None and b['U'] is not None else None
        p={**c,'split':'DEVELOPMENT_ACCEPTANCE_ONLY','proposed_uid':a['uid'],'control_uid':b['uid'],
           'actual_delta_U':actual,'actual_negative':actual<0 if actual is not None else None,
           'invalid_control':not b['eligible'],'invalid_proposed':not a['eligible']}
        for alg in ['A','B']:
            ap=a.get('prediction_'+alg,{}).get('U');bp=b.get('prediction_'+alg,{}).get('U')
            delta=ap-bp if ap is not None and bp is not None else None
            p['predicted_delta_U_'+alg]=delta;p['correct_direction_'+alg]=bool(delta*actual>0) if actual is not None and delta is not None else None
        result.append(p)
    return result

def main():
    assert (OUT/'offline_validation.json').exists()
    assert (OUT/'probe_campaign.json').exists(),'Diagnostic disposition must be saved'
    assert not (OUT/'module24a_gate.json').exists(),'Completed Module24A must not be silently rerun'
    d,refs,weights=inputs();original=enrich_original(d,refs,weights)
    history=read(OUT/'historical_observations.json');probes=read(OUT/'probe_observations.json') if (OUT/'probe_observations.json').exists() else []
    rows=history+probes;model=ResidualModel().fit(rows,refs,weights)
    save('residual_model_final_development.json',model.as_dict())
    timings=[]
    for stage,rates in RATES.items():
        for alg in ['A_MODULE20_NATIVE','B_CALIBRATED_FAST']:
            x={m:1 for m in MODELS};rr=route(d,x,rates,[0.,0.,.18]);elapsed=[]
            for repeat in range(21):
                started=time.perf_counter()
                if alg=='A_MODULE20_NATIVE':predict(d,x,rr,rates,BUDGET,refs,weights,phase=.37)
                else:replay(d,x,rr,rates,BUDGET,refs,weights,model,detail=False)
                if repeat:elapsed.append(time.perf_counter()-started)
            timings.append(dict(stage=stage,algorithm=alg,iterations=20,median_seconds=float(np.median(elapsed)),
                mean_seconds=float(np.mean(elapsed)),p95_seconds=float(np.percentile(elapsed,95)),
                scope='Same co-resident GS18 diagnostic configuration; first CPU call discarded; no GPU call'))
    csvsave('predictor_runtime_benchmark.csv',timings)
    development_combos={r['combo'] for r in rows if r['split']=='DEVELOPMENT'}
    assert not development_combos & {r['combo'] for r in rows if r['split']=='RETROSPECTIVE_COMBINATION_HOLDOUT'}
    # Final fit uses development probes only. Holdout sessions remain whole.
    for r in rows:
        if not r['measurement_available']:continue
        start=time.perf_counter();p,ps=replay(r['catalog'],r['deployment'],r['routing'],r['rates'],r['budget_mib'],refs,weights,model,horizon=r['planned_seconds'])
        r['prediction_B']=p;r['prediction_B_paths']=ps;r['prediction_B_seconds']=time.perf_counter()-start
    utility,service=scalar_residuals(rows)
    csvsave('service_time_residuals.csv',service);csvsave('utility_prediction_residuals.csv',utility)
    # Preserve the pre-probe validation and clearly label final development fit.
    prior=list(csv.DictReader((OUT/'prediction_model_comparison.csv').open()))
    final_pairs=pairs(rows)
    csvsave('ranking_after_development_probes.csv',[{**p,'model_version':'B_AFTER_DEVELOPMENT_PROBES'} for p in final_pairs],['split','module','contrast_group'])
    updated=compare(rows,final_pairs)
    updated=[{**r,'algorithm':'B_AFTER_DEVELOPMENT_PROBES','model_evaluation':'Retrospective unchanged archival holdout; probes never test targets'} for r in updated if r['algorithm']=='B']
    csvsave('prediction_model_comparison.csv',prior+updated)
    negatives=negative_acceptance(rows);csvsave('negative_case_development_acceptance.csv',negatives)
    calibrated=search(d,refs,weights,model,label='calibrated')
    offline=read(OUT/'offline_validation.json');campaign=read(OUT/'probe_campaign.json')
    # Grouped development errors are an empirical stress envelope, not a CI.
    group_errors={}
    for r in rows:
        pu=r.get('prediction_B',{}).get('U')
        if r['split']=='DEVELOPMENT' and pu is not None and r['U'] is not None:
            group_errors.setdefault(r['group'],[]).append(abs(pu-r['U']))
    envelope=max((max(v) for v in group_errors.values()),default=None)
    stages={}
    for stage,st in calibrated['stages'].items():
        old=original['stages'][stage];w=st['winner'];pure=[f for name,f in st['route_class_best'].items() if name!='G/S' and f]
        control=max(pure,key=lambda f:f['prediction']['U']) if pure else None
        delta=w['prediction']['U']-control['prediction']['U'] if w and control else None
        interval=[delta-2*envelope,delta+2*envelope] if delta is not None and envelope is not None else None
        endpoint={}
        if w:
            p,paths=replay(d,w['deployment'],w['routing'],st['rates'],BUDGET,refs,weights,model)
            w['calibrated_path_details']=paths;w['calibration_scope']=p['unsupported_paths']
            for name,f in [('S-only',0.),('G-only',1.),('GS18_diagnostic',.18)]:
                x=dict(w['deployment'])
                if f<1 and S not in x:x[S]=1
                rr=route(d,x,st['rates'],[w['shares'][0],w['shares'][1],f]);ep,ps=replay(d,x,rr,st['rates'],BUDGET,refs,weights,model)
                endpoint[name]=dict(prediction=ep,paths=ps,deployment=x,routing=rr)
        effective_mixed=False
        if w and st['choice']=='G/S':
            gp=w['calibrated_path_details'].get(identity([G,'asr']));sp=w['calibrated_path_details'].get(identity([S,'asr']))
            effective_mixed=bool(gp and sp and gp['service']['count']>0 and sp['service']['count']>0)
        stages[stage]=dict(rates=st['rates'],old_choice=old['choice'],old_operational_interpretation=old.get('operational_interpretation',old['choice']),
            new_choice=st['choice'],old_winner=old['winner'],new_winner=w,best_pure_route=control,
            predicted_delta_U_vs_best_pure=delta,empirical_development_error_envelope_U=envelope,
            predicted_delta_U_stress_interval_NOT_CI=interval,
            interval_scope='Conservative max observed grouped development absolute U residual for each arm; not a confidence bound and not guaranteed for unseen combinations',
            effective_mixed_requests_predicted=effective_mixed,same_background_endpoints=endpoint,
            new_prediction_supported_by_independent_ranking=False,credible_positive_margin=False,
            historical_scope='New pool was absent from historical co-location data. Two short probes, if available, are development only.')
    low=stages['LOW']['new_winner'];mid=stages['MID']['new_winner'];hi=stages['HIGH']['new_winner']
    raw_three=bool(low and mid and hi and S not in low['deployment'] and S in mid['deployment'] and stages['MID']['new_choice']=='S-only' and stages['HIGH']['new_choice']=='G/S')
    meaningful_three=raw_three and stages['HIGH']['effective_mixed_requests_predicted']
    result=dict(pool=MODELS,budget_mib=BUDGET,weights=weights,rates=RATES,stages=stages,
        old_optimizer='Unmodified Module20A periodic FIFO + all measured placement enumeration/free routing',
        new_optimizer='Same placement/routing competition and U weights, calibrated path service/queue predictor',
        raw_autonomous_three_phase=raw_three,meaningful_autonomous_three_phase=meaningful_three,
        no_forced_three_phase_constraints=True,independent_ranking_supported=False,
        HIGH_credible_positive_margin=False,new_independent_GPU_validation_conditions_met=False,
        Module24B_plan_proposed=False,Module24B_executed=False,optimization_performance=calibrated['performance'])
    save('reoptimized_three_phase.json',result)
    after=metadata();before=read(OUT/'historical_metadata_before.json')
    changed=[p for p in set(before)|set(after) if before.get(p)!=after.get(p)]
    input_checks=[]
    for source in read(OUT/'cpu_protocol.json')['inputs']:
        input_checks.append(dict(path=source['path'],unchanged=ref(source['path'])['sha256']==source['sha256']))
    raw_hash_checks=[dict(path=r['raw']['path'],unchanged=ref(r['raw']['path'])['sha256']==r['raw']['sha256']) for r in rows]
    integrity=dict(historical_files_checked=len(before),metadata_unchanged=not changed,changed_files=changed,
        scientific_input_hashes_unchanged=all(r['unchanged'] for r in input_checks),input_checks=input_checks,
        consumed_raw_hashes_unchanged=all(r['unchanged'] for r in raw_hash_checks),raw_hash_checks=raw_hash_checks,
        raw_sources_hashed=[r['raw'] for r in rows],scope='All preexisting results/module source file metadata; SHA256 for all consumed raw sessions and scientific inputs')
    save('historical_integrity.json',integrity)
    gate=dict(module='24A',status='PREDICTOR_RESEARCH_COMPLETE_FORMAL_VALIDATION_NOT_READY',CPU_verification_completed=True,
        CPU_search_original_all_legal_counts=True,CPU_route_global_certificate=False,
        original_HIGH_effective_selection=original['stages']['HIGH'].get('operational_interpretation',original['stages']['HIGH']['choice']),
        offline_validation_completed=True,independent_ranking_contrast_groups=offline['independent_ranking_contrast_groups'],
        independent_ranking_supported=False,cross_combination_generalization_established=False,
        GPU_diagnostic_attempts=campaign['attempts'],GPU_probes_development_only=True,
        meaningful_autonomous_three_phase=meaningful_three,HIGH_credible_positive_margin=False,
        Module24B_conditions_met=False,Module24B_plan_proposed=False,Module24B_executed=False,
        positive_utility_three_phase_established=False,historical_integrity_passed=not changed and integrity['scientific_input_hashes_unchanged'] and integrity['consumed_raw_hashes_unchanged'],
        reason='No independent matched combination/workload ranking evidence; large utility/queue holdout errors and false-stable predictions persist. Probes cannot establish formal gains.',
        required_next_evidence='Independent same-workload paired deployment contrasts across >=2 combinations (>=3 grouped contrasts), ranking agreement >=80%, no high-confidence reversals, calibrated uncertainty/stability validation. Current evidence does not justify a 24B execution plan.',
        stopped_after_Module24A=True)
    save('module24a_gate.json',gate)
    write_summary(original,result,offline,campaign,negatives,integrity)
    required=['original_optimizer_audit.json','service_time_residuals.csv','utility_prediction_residuals.csv',
        'prediction_model_comparison.csv','ranking_holdout_validation.csv','high_colocation_probe.csv',
        'reoptimized_three_phase.json','module24a_gate.json','module24a_summary.md']
    save('output_integrity.json',dict(required=[ref(OUT/n) for n in required],all_required_present=True))
    assert not changed,'Preexisting historical files changed; see historical_integrity.json'
    print('MODULE24A_COMPLETE',json.dumps(gate,ensure_ascii=False),flush=True)

def write_summary(old,new,offline,campaign,negatives,integrity):
    lines=['# Module 24A：Selected Pool Verification & Utility Prediction Error Calibration','',
        '本轮完成 CPU 优化器复核、历史分组校准与留出评估、HIGH 最小共置诊断和相同池重新优化。没有执行 Module 24B；不能宣称跨模型组合泛化或正效用三阶段成立。','',
        '## 原优化器复核','',
        '固定 Qwen2.5-Omni、Whisper-large-v3、RoBERTa-base-SST2、ConvNeXt-tiny、PARSeq；B4=34567.5534 MiB。Text=4、Image=2、OCR=5、LID=.03、VQA=.2、ASR=.02/.80/1.52 QPS。质量准入、服务画像、实测副本域和 Module20A 的 Q/T/L/P/M 权重 0/.25/.30/.30/.15 保持原值。完整枚举 288 个副本状态，71 个满足覆盖和预算的部署都自由竞争；Text/Image/ASR 全部允许自主选路。多起点、两组 DE、Powell、ASR 密网格、独立 top-five 加密检查，路由数值收敛但没有连续全局证书。','',
        '| 阶段 | 原始浮点选择 | G ASR 比例 | 预测 U | 数值/实际含义 |','|---|---|---:|---:|---|']
    for stage,st in old['stages'].items():
        w=st['winner'];lines.append(f"| {stage} | {st['choice']} | {w['ASR_G_fraction']:.8f} | {w['prediction']['U']:.8f} | {st.get('operational_interpretation',st['choice'])} |")
    h=old['stages']['HIGH'];lines+=['','HIGH 原自由 MILP 两个 rho cap 均选择 S-only。补充搜索的极小正 G 流在 120s 周期模拟中产生 0 个 G-ASR 请求；与 S-only 的 U 差只有浮点精度，保留原始路由但不记作有意义的混合收益。LOW/MID 均为 G-only，因此原算法没有自主形成要求的三阶段。低负载/极小正路由在有限窗口可能零请求，这是旧预测器的限制。','',
        '| HIGH 路由 | 预测 U | mean / P95 ms | max rho | ASR residual capacity QPS (G/S) |','|---|---:|---:|---:|---|']
    for name,c in h['same_background_route_diagnostics'].items():
        p=c['prediction'];lines.append(f"| {name} | {p['U']} | {p['pooled_request_mean_ms']:.3f} / {p['pooled_request_P95_ms']:.3f} | {p['max_rho']:.6f} | {c['ASR_residual_capacity_QPS']} |")
    lines+=['','G-only rho>=1 时 U=null；mean/P95 仅为不稳定诊断，不可排序。HIGH 每个合法部署的 mean/P95、路径排队、容量和服务假设在 original_optimizer_audit.json 中完整保留。原始 standalone ASR 为 Qwen=654.078 ms、Whisper=494.913 ms；常数服务、周期到达、family round-robin，不能认证 Poisson 尾部或共置干扰。','',
        '## 历史训练与留出','',
        f"读取 Module15–22 的全部 {offline['raw_sessions']} 个原始 GPU worker/task 会话、请求队列、E2E、显存和系统效用；Module20 没有原始 GPU 会话，Module21D 是 Module21 的 CPU 重分析，未重复当成新观测。开发 {offline['development_sessions']} 会话，留出 {offline['holdout_sessions']} 会话。Module16 的两个完整组合连同 Module17 中同组合的所有 workload/会话统一留出；训练/测试组合不相交。Module19/22 负例均是开发验收，不能充作留出成功。",'',
        'B 使用按模型组合/model/task/count 的中位 log 服务比、33 个经验服务分位、路径队列残差；先在整个 combo/workload/route 组内合并会话，再让每组等权。完整结束的 unstable 会话可提供未截尾 service 观测，但不能训练 stationary queue offset；不完整会话不参与正常延迟/U 拟合。未知组合只借用 model/task/count 参数并标为未验证迁移；未知模型路径沿用 standalone 并保留未知。不使用所有模型共享干扰系数。U 仍用原权重重新计算，实际 U 残差完整输出，不通过任意奖励项制造收益。','',
        '| 真正整组合历史留出 | A 原模型 | B 历史校准 |','|---|---:|---:|']
    comp={r['algorithm']:r for r in offline['comparison'] if r['split']=='RETROSPECTIVE_COMBINATION_HOLDOUT'}
    for metric in ['service_mean_ms_MAE','service_P95_ms_MAE','queue_mean_ms_MAE','system_mean_ms_MAE','system_P95_ms_MAE','U_MAE','false_predicted_stable','matched_contrast_groups','delta_direction_accuracy']:
        lines.append(f"| {metric} | {comp['A'].get(metric)} | {comp['B'].get(metric)} |")
    dev={r['algorithm']:r for r in offline['comparison'] if r['split']=='DEVELOPMENT'}
    lines+=['',f"开发集 ΔU 方向准确率：A={dev['A']['delta_direction_accuracy']:.3f} ({dev['A']['matched_contrast_groups']} 组)，B={dev['B']['delta_direction_accuracy']:.3f} ({dev['B']['matched_contrast_groups']} 可预测组)。B 对部分实际可用方案 abstain，分母不同，不能直接把条件 MAE 降低称作全面改进。排序相关、最高预测收益方案的真实收益/后悔值、服务/E2E/U 误差、假 stable 和计算耗时均见 prediction_model_comparison.csv。",'',
        '留出没有相同 workload 下的独立替代部署，独立排序样本=0：ΔU 方向、收益排序能力未获独立验证。服务时间误差有所下降，但队列/U 误差和假 stable 问题仍在；不能宣称新预测器具备跨组合泛化能力。','',
        '| 开发负例验收 | 实际 ΔU | A 预测 ΔU | B 预测 ΔU |','|---|---:|---:|---:|']
    for p in negatives:
        if p['actual_negative'] or (p['module']==19 and p['actual_delta_U'] is None):
            lines.append(f"| M{p['module']} {p['case']} rep{p['repetition']} | {p['actual_delta_U']} | {p['predicted_delta_U_A']} | {p['predicted_delta_U_B']} |")
    lines+=['','## HIGH 共置诊断（开发数据）','',
        f"状态 {campaign['status']}；GPU 启动尝试 {campaign['attempts']}（包含保留的启动失败），每种最多 2 次、正常请求测量每种最多 1 次。六任务率和相同五模型各 1 驻留固定，ASR S-only / 原探索性 G/S=18%；同一冻结 Poisson task/sample/uniform 轨迹，warmup=10s、measurement=120s、drain 上限60s。这两种是诊断配置，不代表优化器自主选择 18% G/S，也不构成 Module24B 成功证据。",'']
    pr=list(csv.DictReader((OUT/'high_colocation_probe.csv').open()))
    for r in pr:
        if r.get('kind')=='SYSTEM':lines.append(f"- {r['arm']}: eligible={r['eligible']}, unstable={r['hard_unstable']}, U={r['actual_U']}, 旧预测 U={r['old_predicted_U']}, 实际-旧预测={r['actual_minus_old_U']}, safe memory={r['safe_memory_mib']} MiB, mean/P95={r['system_mean_ms']}/{r['system_p95_ms']} ms。")
    lines+=['','ASR service 分布、逐 worker/task queue wait、busy、backlog、显存和完整请求/遥测/失败证据已保存。S-only 中闲置的 Qwen ASR 只报告同驻留 sequential path probe，不能冒充并发 HIGH 分布。原始完整计划请求保留；unstable/incomplete 的正常 U=null，完成子集只作明确诊断。两次短诊断不能评价稀疏 LID 的可靠 P95，不能形成正式收益置信区间或独立排序验证。','',
        '## 重新优化和后续 gate','',
        '| 阶段 | 旧选择 | 校准后选择 | 对最佳纯路由预测 ΔU | 不确定性（经验 stress，非 CI） |','|---|---|---|---:|---|']
    for st,s in new['stages'].items():lines.append(f"| {st} | {s['old_operational_interpretation']} | {s['new_choice']} | {s['predicted_delta_U_vs_best_pure']} | {s['predicted_delta_U_stress_interval_NOT_CI']} |")
    lines+=['',f"自主三阶段（排除零请求伪混合）={new['meaningful_autonomous_three_phase']}。HIGH 可信正收益余量=false；独立 GPU 正式验证条件=false。B 的开发探针参数及其历史留出复评已单独标记 B_AFTER_DEVELOPMENT_PROBES，探针始终不作为测试目标。",'',
        '不提出 Module24B 正式配对执行计划：需要另外收集/冻结独立的同 workload 部署配对排序证据，且跨至少两个模型组合、至少三组独立对比，并解决大队列误差和假 stable。当前结果仅支持继续预测器开发。Module24A 到此停止。','',
        '## 完整性与复现','',
        f"历史 {integrity['historical_files_checked']} 个文件 metadata 不变={integrity['metadata_unchanged']}，实际使用科学输入 SHA256 不变={integrity['scientific_input_hashes_unchanged']}；每个原始 GPU 来源有 SHA256。Module19/22 负结果、历史实验和原始 Module20 代码都保留。",'',
        '原 optimizer 的原有 CPU 测试、replay 一致性、数据组合隔离、未知路径回退和不稳定 U/null 检查见 verification.log。原始与校准预测器的实际 CPU 单次耗时见 predictor_runtime_benchmark.csv；comparison 表中的耗时包含诊断统计，优化 wall time 单独报告。CPU 搜索与拟合用 envs/data_env；GPU controller 用已有 gsdeploy 环境。首次离线报告生成的缺失字段错误、controller 缺少 psutil 的 prelaunch 日志、Whisper READY 未声明 ASR 的 startup 失败均保留。启动失败没有请求测量；只补齐 READY 能力元数据后在用户每路由两次上限内继续。保存路径错误仅移动本轮新增失败文件到 Module24 目录，未改动任何既有历史文件。',
        '复现步骤：python -m module24.audit → python -m module24.calibration → python -m module24.probe → python -m module24.finalize。CPU 步骤设置 CUDA_VISIBLE_DEVICES 空字符串、PYTHONDONTWRITEBYTECODE=1、OPENBLAS_NUM_THREADS=1。已完成步骤拒绝静默覆盖；不要重跑 GPU 诊断。']
    (OUT/'module24a_summary.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
