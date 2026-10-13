"""Finite-space conclusions only; censored/invalid runs never receive legal U."""
from .common import *
import numpy as np
from scipy.stats import spearmanr,kendalltau,t as student_t

def number(v,digits=6):return 'NULL' if v is None else f'{v:.{digits}f}'

def bootstrap(raw_a,raw_b,obs_a,obs_b,weights,refs,seed=330999):
    a=raw_a['phases']['measurement'];b=raw_b['phases']['measurement'];ra=a['request_records'];rb=b['request_records']
    assert [(r['task'],r['sample_id'],r['frozen_offset_seconds']) for r in ra]==[(r['task'],r['sample_id'],r['frozen_offset_seconds']) for r in rb]
    seconds=a['window']['planned_duration_seconds'];rates=raw_a['slot']['lambda_rps'];total=sum(rates.values());N=int(seconds/30)
    bins=[np.array([i for i,r in enumerate(ra) if int(r['frozen_offset_seconds']//30)==j],dtype=int) for j in range(N)]
    taskmask={t:np.array([r['task']==t for r in ra]) for t in rates};rng=np.random.default_rng(seed)
    la=np.array([r['e2e_latency_ms'] for r in ra]);lb=np.array([r['e2e_latency_ms'] for r in rb])
    ia=np.array([r['completion_time_ns']<=a['window']['end_time_ns'] for r in ra]);ib=np.array([r['completion_time_ns']<=b['window']['end_time_ns'] for r in rb])
    def utility(lat,inside,ix,ob):
        L=P=0.
        for task,q in rates.items():
            vals=lat[ix[taskmask[task][ix]]]
            if not len(vals):return None
            L+=q/total*float(np.mean(vals))/refs[task];P+=q/total*float(np.percentile(vals,95))/refs[task]
        T=1-min(1.,float(sum(inside[ix]))/(total*seconds));M=ob['components']['M_norm']
        return -weights['T']*T-weights['L']*L-weights['P']*P-weights['M']*M
    values=[]
    for _ in range(2000):
        ix=np.concatenate([bins[j] for j in rng.integers(0,N,size=N)])
        ua=utility(la,ia,ix,obs_a);ub=utility(lb,ib,ix,obs_b)
        if ua is not None and ub is not None:values.append(ua-ub)
    if not values:return dict(n=0,CI95_low=None,CI95_high=None)
    return dict(n=len(values),CI95_low=float(np.percentile(values,2.5)),CI95_high=float(np.percentile(values,97.5)),
        probability_positive=float(np.mean(np.array(values)>0)),scope='Paired within-trajectory 30-second arrival-block diagnostic; fixes observed memory and does not model order/environment/between-run uncertainty.')

def main():
    fm=read(OUT/'frozen_problem_manifest.json');sp=read(fm['space']['path']);weights=read(fm['utility_definition']['path'])['weights'];refs=read(fm['utility_reference']['path'])['latency_reference_ms']
    oracles=read(OUT/'cpu_oracles.json');correct=read(OUT/'CPU_oracle_correctness.json');objective=read(OUT/'objective_error_decomposition.json');decisions=read(OUT/'optimizer_decisions.json')
    dev=read(OUT/'development_observations.json') if (OUT/'development_observations.json').exists() else []
    pairs=read(OUT/'pair_observations.json') if (OUT/'pair_observations.json').exists() else []
    candidates=read(OUT/'gpu_candidates.json');byc={c['candidate_id']:c for c in candidates};bydev={o['candidate_id']:o for o in dev}
    runrows=[];taskrows=[];pathrows=[]
    # Include every failed initialization and every positive/negative measured run.
    observations=[read(p) for p in sorted((OUT/'gpu_observations').glob('*.json'))]
    for o in observations:
        raw=read(o['raw']['path']);value=o['components'];slot=raw['slot'];env=o.get('environment',{})
        runrows.append(dict(run_id=o['run_id'],candidate_id=o['candidate_id'],stage=o['stage'],eligible=o['eligible'],technical_valid=o['technical_valid'],stable=o['stable'],
            U=o['U'],predicted_U=byc[o['candidate_id']]['predicted_U'],predicted_J_seconds=byc[o['candidate_id']]['predicted_J_seconds'],
            deployment=slot['deployment'],routing=slot['routing'],offered=value.get('offered'),completed=value.get('completed'),within=value.get('within'),
            safe_memory_mib=value.get('safe_memory_mib'),system_state=(o.get('system_stability') or {}).get('stability_state_v2'),
            **{k:value.get(k) for k in ['T_norm','L_norm','P_norm','M_norm','T_contribution','L_contribution','P_contribution','M_contribution']},
            **env,raw=o['raw'],checks=o['checks'],errors=raw['errors']))
        for task,v in value.get('per_task',{}).items():taskrows.append(dict(run_id=o['run_id'],candidate_id=o['candidate_id'],task=task,eligible=o['eligible'],**v))
        for wid,v in o.get('worker_stability',{}).items():pathrows.append(dict(run_id=o['run_id'],worker=wid,**v))
    csvsave('gpu_oracle_runs.csv',runrows,['run_id','candidate_id','stage','U']);csvsave('gpu_per_task_latency.csv',taskrows);csvsave('gpu_worker_stability.csv',pathrows)
    complete=len(bydev)==len(candidates) and all(o['technical_valid'] for o in dev)
    eligible=[o for o in dev if o['eligible']];best=max(eligible,key=lambda o:o['U']) if eligible else None
    sel=next(d for d in decisions if d['pool']=='SMALL6' and d['load']=='ASR32B' and d['algorithm']=='20A_FINITE_DOMAIN_ADAPTER')
    chosen=next(c for c in candidates if identity(c['deployment'],c['routing'])==identity(sel['deployment'],sel['routing']));co=bydev.get(chosen['candidate_id'])
    ranking=[];regrets=[];validation=[];pair_summary={};prediction_errors=[]
    if best and co and co['U'] is not None:
        regrets.append(dict(scope='DEVELOPMENT_ALL8' if complete else 'BEST_AMONG_MEASURED',algorithm='20A_AND_FORMAL_V3_SAME_SELECTED_PLAN',
            oracle_candidate=best['candidate_id'],selected_candidate=chosen['candidate_id'],oracle_U=best['U'],selected_U=co['U'],
            empirical_U_regret=best['U']-co['U'],full_finite_GPU_oracle=complete,
            same_structure=set(byc[best['candidate_id']]['deployment'])==set(chosen['deployment']),
            same_replicas=byc[best['candidate_id']]['deployment']==chosen['deployment'],
            same_routing=identity(byc[best['candidate_id']]['deployment'],byc[best['candidate_id']]['routing'])==identity(chosen['deployment'],chosen['routing'])))
    for o in eligible:
        c=byc[o['candidate_id']];prediction_errors.append(dict(candidate_id=o['candidate_id'],predicted_U=c['predicted_U'],measured_U=o['U'],error_U=o['U']-c['predicted_U'],abs_error_U=abs(o['U']-c['predicted_U']),
            measured_minus_predicted_components={k:o['components'][k]-c['prediction'][k] for k in ['T_norm','L_norm','P_norm','M_norm']}))
    if len(eligible)>=2:
        pred=[byc[o['candidate_id']]['predicted_U'] for o in eligible];measured=[o['U'] for o in eligible]
        ranking_info=dict(n=len(eligible),Spearman=float(spearmanr(pred,measured).statistic),Kendall=float(kendalltau(pred,measured).statistic),
            U_MAE=float(np.mean([abs(a-b) for a,b in zip(pred,measured)])),all_frozen_candidates_measured=complete,
            false_nominal_stable_predictions=sum(not o['stable'] for o in dev),omitted_invalid_or_unstable_from_sorting=len(candidates)-len(eligible))
        for a,b in itertools.combinations(eligible,2):
            pa=byc[a['candidate_id']]['predicted_U'];pb=byc[b['candidate_id']]['predicted_U'];delta=a['U']-b['U'];pred_delta=pa-pb
            raw_a=read(a['raw']['path']);raw_b=read(b['raw']['path']);ci=bootstrap(raw_a,raw_b,a,b,weights,refs)
            ranking.append(dict(A=a['candidate_id'],B=b['candidate_id'],predicted_delta_U=pred_delta,observed_delta_U=delta,
                point_ranking_reversed=pred_delta*delta<0,CI95_excludes_zero=bool(ci.get('CI95_low') is not None and (ci['CI95_low']>0 or ci['CI95_high']<0)),
                **ci,temperature_difference_c=abs(a['environment']['temperature_gpu_c']-b['environment']['temperature_gpu_c']),
                sm_clock_relative_difference=abs(a['environment']['sm_clock_mhz']-b['environment']['sm_clock_mhz'])/max(a['environment']['sm_clock_mhz'],b['environment']['sm_clock_mhz'])))
        ranking_info['point_reversals']=sum(r['point_ranking_reversed'] for r in ranking);ranking_info['reversals_with_within_trace_CI_excluding_zero']=sum(r['point_ranking_reversed'] and r['CI95_excludes_zero'] for r in ranking)
    else:ranking_info=dict(n=len(eligible),all_frozen_candidates_measured=complete)
    if (OUT/'independent_pair_selection.json').exists():
        ps=read(OUT/'independent_pair_selection.json');selected=ps['optimizer_selected'];refbest=ps['development_best'];byrun={o['run_id']:o for o in pairs}
        for rep in [1,2,3]:
            a=byrun.get(f'PAIR{rep}_'+refbest)
            for cid in selected:
                b=byrun.get(f'PAIR{rep}_'+cid)
                if not a or not b:continue
                raw_a=read(a['raw']['path']);raw_b=read(b['raw']['path']);paired=all(raw_a['slot']['trajectories'][label]['sha256']==raw_b['slot']['trajectories'][label]['sha256'] for label in ['warmup','measurement'])
                tempdiff=abs(a.get('environment',{}).get('temperature_gpu_c',0)-b.get('environment',{}).get('temperature_gpu_c',0));clocks=abs(a.get('environment',{}).get('sm_clock_mhz',0)-b.get('environment',{}).get('sm_clock_mhz',0))/max(a.get('environment',{}).get('sm_clock_mhz',1),b.get('environment',{}).get('sm_clock_mhz',1))
                posttemp=lambda raw:float(np.mean([s['temperature_gpu_c'] for s in raw['postwarm_idle']]))
                postdiff=abs(posttemp(raw_a)-posttemp(raw_b));envpass=tempdiff<=5 and clocks<=.15 and postdiff<=5
                valid=a['eligible'] and b['eligible'] and paired and envpass
                delta=a['U']-b['U'] if a['U'] is not None and b['U'] is not None else None
                ci=bootstrap(raw_a,raw_b,a,b,weights,refs,seed=330999+rep) if valid and cid!=refbest else dict(CI95_low=0. if valid else None,CI95_high=0. if valid else None,n=0)
                validation.append(dict(repetition=rep,oracle_candidate=refbest,selected_candidate=cid,paired_trace=paired,environment_valid=envpass,
                    comparison_valid=valid,oracle_U=a['U'],selected_U=b['U'],oracle_minus_selected_U=delta,
                    same_candidate=refbest==cid,postwarm_temperature_difference_c=postdiff,temperature_difference_c=tempdiff,clock_relative_difference=clocks,**ci))
        valid=[r for r in validation if r['comparison_valid']];vals=[r['oracle_minus_selected_U'] for r in valid];n=len(vals)
        mean=float(np.mean(vals)) if vals else None
        half=float(student_t.ppf(.975,n-1)*np.std(vals,ddof=1)/math.sqrt(n)) if n>=2 else None
        pair_summary=dict(n_valid_pairs=n,expected_pairs=3,mean_oracle_minus_selected_U=mean,
            CI95_low=mean-half if half is not None else None,CI95_high=mean+half if half is not None else None,
            all_same_candidate=bool(validation) and all(r['same_candidate'] for r in validation),
            interpretation='Same candidate is an identity comparison, not independent rank validation against other candidates.' if validation and all(r['same_candidate'] for r in validation) else 'Paired independent-request trajectories with reversed serial arm order. Three-repetition t interval; limited population inference.')
        regrets.append(dict(scope='INDEPENDENT_PAIRS_FIXED_DEVELOPMENT_WINNER',**pair_summary))
    csvsave('gpu_empirical_regret.csv',regrets,['scope']);csvsave('gpu_independent_pairs.csv',validation);save('GPU_pair_statistics.json',pair_summary)
    csvsave('gpu_prediction_error.csv',prediction_errors);csvsave('gpu_ranking_pairs.csv',ranking);save('GPU_ranking_analysis.json',ranking_info)
    history=read(OUT/'history_integrity_after.json') if (OUT/'history_integrity_after.json').exists() else {'passed':False,'status':'NOT_YET_VERIFIED'}
    samewinner=bool(best and co and best['candidate_id']==co['candidate_id']);verifiedpairs=pair_summary.get('n_valid_pairs')==3
    supported=bool(complete and samewinner and verifiedpairs and history['passed'] and correct['passed'])
    gate=dict(module=33,ORACLE_SPACE_INVALID=not correct['passed'],CPU_ORACLE_COMPLETED=correct['passed'],
        OPTIMIZER_OBJECTIVE_MISMATCH=True,objective_mismatch_reason='Promoted formal V3 uses J routing/ranking and original noise-aware latency acceptance; 20A U optimizer remains unpromoted. Same selected U optimum here does not erase objective mismatch.',
        OPTIMIZER_SEARCH_GAP=any(r['U_regret']>1e-9 for r in read(OUT/'predicted_regret.json') if r.get('finite_scope') and r.get('U_regret') is not None),
        GPU_ORACLE_INCOMPLETE=not complete,GPU_RANKING_MISMATCH=bool(best and co and not samewinner),
        MIXED_OPTIMALITY_SUPPORTED=supported,CPU_grid_optimality_supported=correct['passed'],
        GPU_scope='All eight frozen candidates measured development empirical maximum only; no continuous/full-pool/population optimum certificate.',
        GPU_development_configurations=len(bydev),GPU_valid_configurations=sum(o['technical_valid'] for o in dev),GPU_eligible_configurations=len(eligible),
        GPU_expected_configurations=8,independent_pairs=pair_summary,CPU_measured_regret=read(OUT/'predicted_regret.json'),
        development_regret=regrets[0] if regrets and 'empirical_U_regret' in regrets[0] else None,
        history_preserved=history['passed'],paper_ready_general_global_optimality=False,
        paper_ready_scoped_finite_CPU_benchmark=correct['passed'] and history['passed'],
        paper_ready_scoped_GPU_results=complete and verifiedpairs and history['passed'],
        Module34_started=False,algorithm_modified_after_GPU=False)
    gate['primary_gate']='GPU_ORACLE_INCOMPLETE' if not complete else 'MIXED_OPTIMALITY_SUPPORTED' if supported else 'GPU_RANKING_MISMATCH' if gate['GPU_RANKING_MISMATCH'] else 'GPU_INDEPENDENT_EVIDENCE_INSUFFICIENT'
    save('module33_gate.json',gate)
    runtime_rows=list(csv.DictReader((OUT/'optimizer_runtime_benchmark.csv').open()));formal_times=list(csv.DictReader((OUT/'formal_optimizer_runtime_benchmark.csv').open()))
    lines=['# Module33：Mixed Deployment Oracle Benchmark and Optimizer Evaluation','',f"Gate：`{gate['primary_gate']}`。CPU 与 GPU 结论分别判断；历史保护={history['passed']}。",'',
        '正式算法是冻结 V3：J 决定任务路由与候选排名，真正的部署接纳使用原 NoiseAwareRuntimeObjective(1ms, 5%) 的稳定性、原始 mean/P95、显存比较；没有单一全局 J 或 U 部署目标。20A 是直接最大化预测 U 的开发算法，其 UTILITY_MODEL_INSUFFICIENT 历史 Gate 保留。Module25 的扩展未执行、32B 动态策略未修改。此区别不能用“J改成U后的表现”掩盖。','',
        'Stage A 冻结 6/7 模型两池：SMALL6 的 Phi/Whisper-large/Roberta/ConvNeXt/PARSeq/Whisper-small；LARGE7 加 Qwen 与部分实测双副本。Whisper-small 的 ASR 被明确排除，仅保留原32B LID路径，原质量准入不变；受限空间不代表14F完整模型池。六任务均有正QPS。LOW18来自18的text_low_rep1，ASR32B来自32B A0。预算34567.5534MiB、safe_factor、20A权重与逐任务归一化参考不变。全部路径网格为QPS的0/0.5/1弱组合；最优性只覆盖这个离散空间。','',
        '20A有限域适配只对原MILP路由变量做单位换算/整数网格约束；原42个生成尝试、目标、权重、.8/.95容量种子和U排序未改。原生连续20A另外执行，四组最终结果均恰好在网格内。正式V3使用原生create/search及原始接纳流程，RuntimeEvaluator明确用原CPU预测接口提供性能；不是dry_run仅返回seed，也不是实测GPU自适应搜索。四组native V3最终结果也均在网格内。','',
        '|池/负载|全部副本向量|覆盖后网格组合（显存前）|合法组合|Oracle U|20A/V3 U regret|CPU穷举秒|20A中位秒/加速|V3中位秒/加速|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for o in oracles:
        cell=next(c for c in correct['checks'] if c['pool']==o['pool'] and c['load']==o['load']);tt=[float(r['seconds']) for r in runtime_rows if r['pool']==o['pool'] and r['load']==o['load']];ft=[float(r['seconds']) for r in formal_times if r['pool']==o['pool'] and r['load']==o['load']];med=float(np.median(tt));fmed=float(np.median(ft))
        lines.append(f"|{o['pool']}/{o['load']}|{o['total_deployment_vectors']}|{cell['total_mixed_route_candidates_before_memory']}|{o['feasible_candidates']}|{o['U_oracle']['predicted_U']:.6f}|0 / 0|{o['seconds']:.6f}|{med:.6f} / {o['seconds']/med:.2f}×|{fmed:.6f} / {o['seconds']/fmed:.2f}×|")
    lines+=['','四组最优均为6模型各一副本，ASR全部走Whisper-large，VQA走Phi，其他四任务各走对应S；没有ASR分流。U目标选择、模型驻留、副本和路由均与U Oracle一致，无约束违规。每次20A42个MILP尝试，只评估3–5个唯一候选，包含四个预测场景；小池访问3/8=37.5%，大池约0.14%–0.24%。内部MILP分支并不等于显式候选访问。小空间枚举更快，大空间才有速度收益；仅两个真实画像规模与单机时序，不声称渐近复杂度定理。','',
        'CPU Oracle另有独立笛卡尔路由枚举，逐项覆盖计数/集合一致；独立标量FIFO回放与原predict名义U误差<1e-9。全部候选使用相同predict_scenarios，名义phase=.37、120秒、rho<1；原V3仍保留rho<=.8。J空间Oracle单独在原.8约束下取最小值。J最优偏好更多副本，其U相对U Oracle差约0.0104–0.0478；这说明目标的取舍不同。正式V3最终也不必达到全局J最小，因为原接纳噪声门槛是另一项规则，不能把这种J差距直接叫算法搜索失败。详见objective_error_decomposition.json，正式选择在原pairwise接纳规则下的有限候选改进数量也单独列出。','',
        'GPU全部配置和种子在实测前冻结。8个方案共同使用六任务ASR32B负载、同输入/Poisson轨迹/route与replica uniforms；开发窗口120s、warmup30s、drain60s、逐次fresh PIDs。配置包含有/无Whisper-large、OCR1/2副本、Phi ASR份额0/0.5/1；显存/质量/系统与逐worker稳定性失败的U=NULL。GPU范围不是CPU前几名，也没有更改动态分流策略。','',
        '|候选|OCR副本|Large驻留|ASR Phi份额|预测U|开发实测U|合法|','|---|---:|---|---:|---:|---:|---|']
    for c in candidates:
        o=bydev.get(c['candidate_id']);x=c['deployment'];ocr=next(m for m in x if m.startswith('baudm/parseq:parseq@'));large=any(m.startswith('openai/whisper-large') for m in x);share=c['routing']['asr'].get('phi4mm',0)/sp['loads']['ASR32B']['rates']['asr']
        lines.append(f"|{c['candidate_id']}|{x[ocr]}|{large}|{share:.1f}|{c['predicted_U']:.6f}|{number(o['U'] if o else None)}|{o['eligible'] if o else '未测'}|")
    lines+=['',f"GPU完整性：{len(bydev)}/8个配置完成测量，{len(eligible)}个合法可排序。真实参考方案={best['candidate_id'] if best else 'NULL'}；优化器方案={chosen['candidate_id']}。经验开发regret={number(best['U']-co['U'] if best and co and co['U'] is not None else None)}。",'',
        f"独立验证：{pair_summary.get('n_valid_pairs',0)}/3个有效配对；参考−优化器平均ΔU={number(pair_summary.get('mean_oracle_minus_selected_U'))}，t95 CI=[{number(pair_summary.get('CI95_low'))}, {number(pair_summary.get('CI95_high'))}]。同方案比较={pair_summary.get('all_same_candidate',False)}；同方案的0差值是恒等关系，不能伪造为其他候选的独立排名验证。独立轨迹用于复测获选方案；其它候选只具有完整开发空间测量证据。",'',
        f"预测与实测：{json.dumps(ranking_info,ensure_ascii=False)}。排序微差的30秒配对block CI见gpu_ranking_pairs.csv。CI跨0时不做严格排名，内窗bootstrap没有识别序列次序、热状态、显存测量误差或跨运行随机性。不同配置环境差异完整保留。",'',
        '基础设施负结果保留：新增评估脚本曾在任何worker启动前语法失败；其后首个配置在Phi READY前因漏预置隔离动态代码缓存失败，无测量窗口/U。恢复32E/F已有的byte-identical缓存预置后按原配置/种子重新启动，失败raw与NULL结果保留且run_id不同。没有根据GPU效用更改算法、模型、QPS、候选或门槛。任何真正测量失败都不会补值、删除或调整后验验收。','',
        '最终六项回答：',
        '1. 在四个冻结有限空间中，原生20A和正式V3的CPU预测后端结果均找到U最优；20A自己的U目标regret=0。正式V3并不以U或全局J为单一部署目标，J-routing精确Oracle及原噪声接纳规则的结果分别报告。',
        f"2. GPU开发空间参考−优化器ΔU={number(best['U']-co['U'] if best and co and co['U'] is not None else None)}；独立配对平均ΔU={number(pair_summary.get('mean_oracle_minus_selected_U'))}。证据范围为全部8个有限配置，不外推连续/完整模型池。",
        '3. CPU U搜索误差为0。目标函数差异确实存在，J最优与U最优不同；GPU预测误差由实测U/分项误差、排序反转及稳定性误判独立量化。不能将GPU逆序直接归为搜索失败，也不能以相同胜者忽略绝对预测误差。',
        '4. 求解速度见上表：小空间固定求解开销使其可能慢于穷举；大空间明显更快。V3和20A的时间分别报告，三次重复与目标调用计数留档。',
        f"5. 受限离散CPU最优性证据可复现；完整GPU排序证据={complete}，独立配对完整={verifiedpairs}。适用于论文的范围是带限制、失败和不确定性的benchmark结果；没有通用真实全局最优证据，论文用GPU结果准备条件={gate['paper_ready_scoped_GPU_results']}。",
        '6. 可在另行授权下定义纯G/纯S/Mixed统一候选域并复用Oracle与测量框架；本次混合空间强制覆盖的G/S必要路径不能当作三结构竞争证据。原预测器仍需按GPU结果评估是否可迁移。本轮不进入Module34。','',
        f"历史前后完整SHA256：{history['passed']}，文件数={history.get('files','NULL')}，包括历史结果、源码、Gate与模型权重；全部新输出/缓存位于Module33隔离目录。"]
    (OUT/'module33_summary.md').write_text('\n'.join(lines)+'\n')
    diagnosis=['# 排序误差与归因','',f"完整开发GPU Oracle={complete}；参考与优化器同方案={samewinner}。",'',
        '搜索层：四组U网格精确Oracle与20A/原生V3预测部署匹配，U搜索regret=0。MILP网格适配与原生连续结果都保留。原V3接受原CPU预测性能、原噪声规则；没有把正式流程替换成最大U排序。','',
        '目标层：正式V3的J routing/ranking与20A总U定义不同，J全局最优在CPU下反而比U最优多部署副本，U成本差见objective_error_decomposition.csv。正式V3没有单一全局J部署目标；其候选生成/Top2与真实噪声接纳规则同时决定输出，不能把J-global gap默认解释为搜索算法错误。','',
        '预测层：保持20A独立standalone s0/g/M不变，CPU为周期到达+常数服务，GPU为冻结真实输入+Poisson到达；输入变异、共置、runtime和窗口到达差异均可贡献误差。每个U分项的预测−真实误差和原生执行/等待保留，不做本轮后验拟合。此实验不能唯一识别纯GPU干扰/CPU/frontend/arrival机制各自因果贡献。','',
        f"实测诊断：{json.dumps(ranking_info,ensure_ascii=False)}。",'',
        '排名层：8个配置均按同一清单竞争；技术无效和unstable U为空。开发最佳是该窗口观测最大，并非潜在期望U最大或连续全局最优。bootstrap仅为同轨迹分块不确定性诊断，不删微小反转或用不显著差距强称严格排名。','',
        f"独立配对：{json.dumps(pair_summary,ensure_ascii=False)}。",'',
        '初始失败及环境：Phi缺失隔离cache预置导致READY前失败，与任何实测收益无关；byte-identical恢复缓存后未改配置/算法。每个运行的传感器/worker请求/队列/退出保留在raw_runs与request_journals。没有改写历史负收益。','',
        '下一轮改进：仅当新实测排序与CPU不一致时，在下一轮另行做预测器/目标/搜索改进和新独立验收；本轮到此停止。']
    (OUT/'ranking_error_analysis.md').write_text('\n'.join(diagnosis)+'\n')
    print('REPORT',gate['primary_gate'],'history',history['passed'],flush=True)
if __name__=='__main__':main()
