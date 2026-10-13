"""Freeze the failed predictive gate; do not search or run GPU past it."""
from .common import *
from .predictor import UtilityDifferencePredictor,original_J
from module20.optimizer import predict

def main():
    assert (OUT/'validation_completed.json').exists()
    assert not (OUT/'module25_gate.json').exists(),'Completed Module25 cannot be overwritten'
    report=read(OUT/'validation_report.json');attribution=read(OUT/'module24_prediction_error_attribution.json')
    protocol=read(OUT/'research_protocol.json');rows=list(csv.DictReader((OUT/'independent_ranking_validation.csv').open()))
    observations=read(OUT/'observations.json');byuid={o['uid']:o for o in observations};definition,refs=utility_inputs()
    finalmodel=UtilityDifferencePredictor.load(OUT/'colocation_prediction_model.json')
    grouped=defaultdict(list)
    for row in rows:
        if row['primary']=='True' and row['actual_delta_U']:grouped[row['contrast_group']].append(row)
    common=[]
    for group,values in grouped.items():
        if any(not r['predicted_delta_NEW'] or not r['predicted_delta_A20'] or not r['predicted_delta_J'] for r in values):continue
        actual=float(np.mean([float(r['actual_delta_U']) for r in values]))
        for algorithm in ['J','A20','NEW']:
            p=float(np.mean([float(r['predicted_delta_'+algorithm]) for r in values]))
            common.append(dict(contrast_group=group,case=values[0]['case'],algorithm=algorithm,
                actual_delta_U=actual,predicted_delta=p,direction_correct=p*actual>0,
                delta_U_error=p-actual if algorithm!='J' else None,
                wrong_sign_positive=actual<=0 and p>0,
                wrong_point_selection_above001=actual<=0 and p>.01 if algorithm!='J' else None,
                scalar_units='J_seconds_gain_NOT_U' if algorithm=='J' else 'delta_U',
                no_positive_examples_in_primary=True))
    common_metrics=[]
    for algorithm in ['J','A20','NEW']:
        subset=[r for r in common if r['algorithm']==algorithm]
        common_metrics.append(dict(algorithm=algorithm,scope='IDENTICAL_PRIMARY_COMMON_POINT_SUBSET',groups=len(subset),
            direction_accuracy=float(np.mean([r['direction_correct'] for r in subset])) if subset else None,
            delta_U_MAE=float(np.mean([abs(r['delta_U_error']) for r in subset])) if subset and algorithm!='J' else None,
            wrong_sign_positive=sum(r['wrong_sign_positive'] for r in subset),
            wrong_point_selection_above001=sum(r['wrong_point_selection_above001'] is True for r in subset) if algorithm!='J' else None))
    csvsave('common_subset_prediction_comparison.csv',common);csvsave('common_subset_metrics.csv',common_metrics)
    performance=[]
    # Actual wall time on identical frozen candidate pairs, warmed interpreters.
    for row in [r for r in rows if r['case']=='M24B_HIGH']:
        a,b=byuid[row['a_uid']],byuid[row['b_uid']]
        for algorithm in ['J','A20','NEW_FINAL_HELD_POOL']:
            times=[]
            for iteration in range(11):
                begin=time.perf_counter()
                for ob in [a,b]:
                    if algorithm=='J':original_J(ob['catalog'],ob['deployment'],ob['routing'],ob['rates'])
                    elif algorithm=='A20':predict(ob['catalog'],ob['deployment'],ob['routing'],ob['rates'],ob['budget_mib'],refs,definition['weights'],phase=.37)
                    else:finalmodel.predict(ob['catalog'],ob['deployment'],ob['routing'],ob['rates'],ob['budget_mib'],refs,definition['weights'],hardware_uuid=ob['hardware_uuid'])
                if iteration:times.append(time.perf_counter()-begin)
            performance.append(dict(algorithm=algorithm,paired_predictions=2,iterations=10,
                mean_seconds=float(np.mean(times)),median_seconds=float(np.median(times)),p95_seconds=float(np.percentile(times,95)),
                scope='Identical24B pair; final model excludes entire24 pool; first iteration discarded; scoring only, not full optimizer wall time'))
    csvsave('inference_overhead.csv',performance)
    catalog=read(SOURCE);checks=report['gate_checks']
    if not attribution['metric_flip_explained']:status='MODULE24_ERROR_ATTRIBUTION_INCOMPLETE'
    elif not report['passed']:status='INDEPENDENT_DELTA_SIGN_OR_UNCERTAINTY_NOT_VALIDATED'
    else:raise RuntimeError('Predictive gate passed: Stage D is required; do not declare completion here')
    search_policy=protocol['high_first_policy']
    # No post-gate point scoring of new pools; preserve explicitly unexecuted universe.
    csvsave('high_first_pool_search.csv',[dict(candidate_universe='ALL_EXISTING_ADMITTED_MODELS',
        status='NOT_EXECUTED_PREDICTIVE_GATE_FAILED',pool_model_count=len(catalog['models']),model_ids=sorted(catalog['models']),
        task_count=6,rates=RATES['HIGH'],budget_mib=BUDGET,autonomous_choice=None,predicted_delta_U=None,
        credible_positive_margin=None,legal_replica_constraints=catalog['capacity_factors'],
        reason=status,optimizer_implementation=str(ROOT/'module25/optimizer.py'),selection_policy=search_policy)])
    save('frozen_positive_candidates.json',dict(status='NO_CANDIDATES_FROZEN_STAGE_C_FAILED',frozen=False,candidates=[],
        new_pools_scored=0,not_a_claim_that_all_pools_lack_gain=True,gate=status,
        selection_rules_frozen_before_fit=True,selection_policy=search_policy,no_forced_A_or_B=True,
        Stage_D_executed=False,Stage_E_executed=False,formal_GPU_sessions=0))
    csvsave('gpu_paired_validation.csv',[dict(status='NOT_EXECUTED_PREDICTIVE_GATE_FAILED',formal_run_count=0,
        actual_U=None,delta_U=None,reason=status,exploratory_data_reused_as_formal=False)],['run_id','arm','status','actual_U','delta_U'])
    primary={m['algorithm']:m for m in report['metrics'] if m['scope']=='PRIMARY_GS_MINUS_S'}
    new=primary['NEW'];old=primary['A20'];j=primary['J'];chosen=[r for r in rows if r['primary']=='True' and r['confident_GS_selected']=='True']
    budget_gate=read(ROOT/'results/module24b_high_validation/module24b_gate.json')
    before=read(OUT/'historical_metadata_before.json');after=protected_metadata();changed=[p for p,v in before.items() if after.get(p)!=v]
    assert not changed,changed[:5];verify(read(OUT/'source_inventory.json'))
    gate=dict(module=25,status=status,Stage_A_metric_attribution_completed=attribution['metric_flip_explained'],
        unique_physical_causal_attribution_established=False,Stage_B_predictor_implemented=True,
        Stage_C_independent_prediction_passed=report['passed'],Stage_C_checks=checks,
        primary_metrics=primary,common_subset_metrics=common_metrics,
        false_sign_positive_predictions_NEW=new['wrong_point_GS_selections'],
        false_GS_point_choices_above_frozen001=sum(r['wrong_point_selection_above001'] is True for r in common if r['algorithm']=='NEW'),
        false_confident_GS_choices=0,confident_GS_choices=len(chosen),
        abstention_is_not_correct_direction=True,
        independent_positive_primary_GS_examples=0,development_positive_probe_counted_as_formal=False,
        interval_coverage_established=False,actual_unstable_identification=report['stability'],
        Stage_D_executed=False,high_positive_candidate_found=False,all_pools_exhaustively_searched=False,
        phase_A_or_B_established=False,Stage_E_executed=False,new_GPU_acquisition_sessions=0,new_formal_GPU_sessions=0,
        system_utility_positive_split_GPU_validated=False,
        original_Module24B_gate=ref(ROOT/'results/module24b_high_validation/module24b_gate.json'),
        original_Module24A_gate=ref(ROOT/'results/module24_utility_calibration/module24a_gate.json'),
        original_J_modified=False,original_algorithms_modified=False,history_unchanged=True,historical_file_count=len(before),
        output_schema_revision='New routing-conditioned predictor; numerical JSON artifact recovery only; no refit or tuning on holdout',
        final_model_fit_excludes_whole_Module24A_B_pool=True,
        retrospective_validation_not_blind_prospective=True,stopped_after_Module25=True,
        reason='Out-of-combination predictions fail sign coverage/reliability, positive test support and finite uncertainty criteria. User stop rule blocks new pool search and GPU efficacy experiments.')
    save('module25_gate.json',gate)
    save('historical_integrity.json',dict(metadata_unchanged=not changed,count=len(before),changed=changed,
        all_consumed_scientific_source_SHA256_unchanged=True,inventory=ref(OUT/'source_inventory.json')))
    (OUT/'utility_difference_predictor.py').write_bytes((ROOT/'module25/predictor.py').read_bytes())
    modeldoc=read(OUT/'colocation_prediction_model.json');modeldoc.update(utility_definition=protocol['utility_definition'],
        utility_reference=protocol['reference'],weights=definition['weights'],refs_ms=refs,validation_status=status,
        independent_ranking_validated=False,no_finite_calibrated_delta_interval=True,
        unknown_replica_or_combination_policy='Conditional point only, explicitly unsupported; no confident positive split recommendation',
        final_holdout_checkpoint_pool=protocol['final_holdout_checkpoint_pool'])
    save('colocation_prediction_model.json',modeldoc)
    pperf={r['algorithm']:r for r in performance};newtime=pperf['NEW_FINAL_HELD_POOL']['median_seconds'];oldtime=pperf['A20']['median_seconds'];jtime=pperf['J']['median_seconds']
    summary=['# Module25：系统效用差预测与正收益分流发现\n\n',
        f'最终Gate：**{status}**。完成Stage A/B/C后按用户停止规则停止。Stage D没有执行，Stage E没有执行，新GPU会话=0；没有声明任何新池或三阶段成功。\n\n',
        '## 错误复现与可核验归因\n\n',
        '24A开发探针实际ΔU=+0.476798，但20A预测=-0.029255、最终24A校准预测=-0.007317；双副本自主候选预测+0.031224才是24B验证的对象。24B实际ΔU=-0.106711。二者五个checkpoint和QPS相同，物理副本为5与8、计划ASR G份额18%与13.1833%、输入seed与cohort均不同；只能称不同配置下收益方向变化。各自配对内部匹配，原效用口径一致，没有截尾完成集效用。\n\n',
        '原J优化的是未归一化的平均拥塞surrogate，没有20A任务参考值、尾延迟和显存效用的完整目标。21的Poisson常数service及24的固定路径残差仍不能表达路由改变后的service动态；未知副本回退先验和共置画像失配使小正预测超过真实余量。20A本身在本轮5个合法负例组上均预测为负，不能混称所有旧算法都错误预测为正。\n\n',
        '24B的actual-minus-B24 ΔU误差中，mean项=-0.037677、P95项=-0.100411，吞吐+0.000163，memory约-0.000010；实际损失主要来自Text/OCR背景，而不只是ASR排队。将旧预测仅代入11.4943%实际份额，ΔU仍+0.003363，没有解释真实负收益。逐worker service/queue/P95、输入时长/类别差异、mean服务/等待/传输分解和观测service oracle均保存。P95没有错误地相加拆分。物理根因仍缺少同轨迹n1/n2×路由的因子实验和kernel/时钟/预处理证据，不作唯一GPU因果断言。\n\n',
        '## 新模型与支持范围\n\n',
        'Separate checkpoint/task/count的ridge拟合log服务均值和33个服务分位形状，输入包括own load、同worker其他任务load、foreign Generalist/Specialist load、驻留worker数量和idle-gap。同组合/workload内先合并整次重复会话，再按组平衡权重；4个以上组才估计route slopes，否则保留稀疏中位值并明确斜率未识别。Generalist背景与Specialist服务均允许随完整路由变化，无全模型统一干扰系数。\n\n',
        'Poisson队列由项目Module21 FIFO变换扩展到经验服务分布与随机副本thinning；周期实验使用原periodic_arrivals/Router和物理worker FIFO。所有多任务共享队列，E2E加实测transport残差。仍有stationary/独立service、未测GPU状态/输入混合的局限。实际有效容量的点预测由条件service产生，不把isolated g(n)当确定共置扩展比。缺失路径/副本/组合明确unsupported，区间无支持则unbounded。实际U始终用20A冻结公式和hard quality/stability/memory约束。\n\n',
        '全部125个原始会话纳入证据清单；20A无新GPU、21D重分析不重复计样本、23无GPU。最后模型完全排除24A/24B整池的6个会话（包括启动失败与preflight），预测验证每折排除比较两边的整个checkpoint组合，跨所有workload/副本/模块传播排除。完整deployment/count/QPS/hardware/request协议键用于inner calibration分组。没有请求随机训练/测试划分。已知历史结果的留出属于retrospective out-of-sample，不冒充盲测。\n\n',
        '## 严格独立排序结果\n\n',
        '| 打分器 | 合法独立对照组 | 可预测/拒绝 | 点方向正确率 | ΔU MAE | 错误正号 |\n|---|---:|---|---:|---:|---:|\n']
    for alg in ['J','A20','NEW']:
        m=primary[alg];summary.append(f'| {alg} | {m["independent_contrast_groups"]} | {m["point_available_groups"]}/{m["point_abstentions"]} | {m["point_direction_accuracy"]} | {m["delta_U_MAE"]} | {m["wrong_point_GS_selections"]} |\n')
    summary += ['\n5组均为实际负收益，来自4种checkpoint组合；重复配对先聚合为一个对照组。24A正开发探针不计独立正例，Module22 MID是部署策略差异，几乎零G流量，不能计作真实正GS/S对照。Module19原operationally-illegal Image/ASR比较器保留为辅助诊断，不扩大合法主验证分母。\n\n',
        '新模型仅3组有点预测，其中2组方向正确=66.67%；按全部5组计覆盖正确2/5，另外2组拒绝而不是“正确识别”。24B仍误判+0.002205（真实-0.106711），小于+0.01阈值但符号依然错误；不构成可推荐收益。整池留出的24A正开发样本仍被预测为负（-0.006831），说明没有证明识别正收益的能力。\n\n',
        '每折inner calibration仅0–1个独立ΔU对照组，95% split-conformal无法产生有限界；不把训练最小/最大或16次模拟变动当统计CI。有效有限区间=0，coverage未知。所有保守决策均ABSTAIN，不能把零错误推荐当预测已成功。2次实际unstable均仍被预测nominal stable，和20A一样；仅标uncertified不等于正确检出。\n\n',
        '| 同一3组可预测子集 | 方向准确率 | ΔU MAE |\n|---|---:|---:|\n']
    for m in common_metrics:summary.append(f'| {m["algorithm"]} | {m["direction_accuracy"]} | {m["delta_U_MAE"]} |\n')
    summary += ['\n必须比较相同分母，不能用新模型3组MAE低于20A5组MAE宣称改进。J没有系统U量纲，ΔU MAE不适用，未用经验换算系数。候选仅两个时，排序正确即方向正确；跨不同workload的增益排序是描述性Spearman，不冒充同一可执行候选列表的top-k搜索。\n\n',
        '## 计算开销与后续搜索\n\n',
        f'同一24B候选对，热态10次、首轮丢弃，median：J={jtime:.6f}s，20A={oldtime:.6f}s，新模型={newtime:.6f}s；新/20A={newtime/oldtime:.2f}倍，新/J={newtime/jtime:.2f}倍。新模型没有推理速度改善；扩展服务分布/队列有额外成本。整池优化搜索因gate失败未执行，不能给出搜索加速或新算法优化收益。原Module24B旧搜索耗时只能作背景，不是本轮等条件对照。\n\n',
        'HIGH-first筛选规则已在拟合前冻结：全部现有质量准入模型及已测副本参与，B4预算、固定背景4/2/5/.03/.2 QPS，HIGH ASR=1.52、降低到.02/.80；使用真实20A Generator及全合法部署/自由路由的评分扩展，竞争纯G、纯S和混合。不强制A/B结构。module25/optimizer.py提供guarded adapter，当前gate会在求解开始前拒绝执行。high_first_pool_search.csv明确NOT_EXECUTED而非伪造失败候选；frozen_positive_candidates.json为空且frozen=false。\n\n',
        '## 六个问题\n\n',
        '1. 旧预测为什么误判：J目标不同、standalone/共置service不匹配、固定残差不随路由变、周期/Poisson与副本thinning差异，背景mean/P95损失遗漏；并非归一化算错或简单份额偏差。\n',
        '2. 新预测器能否独立判断：未通过，3个可预测合法组正确2个、拒绝2组，无独立正例/有限误差区间，unstable漏检仍有2次。\n',
        '3. 找到值得分流的新组合：没有形成可信候选；未执行新池搜索，不能断言所有池无收益。\n',
        '4. 同一六任务系统形成A或B：未建立，未强制结构，也没有输出未验证成功。\n',
        '5. GPU证明真实总效用优势：本轮GPU=0，既有24B负结果保留；24A单个开发正探针不是正式优势证明。\n',
        '6. 相比J的准确率/开销：同一3组负例J0/3、新2/3、20A3/3，不能推广到正例；推理更慢，完整搜索速度改进未测。\n\n',
        f'历史{len(before)}个文件metadata不变，所有读取的科学输入SHA256不变，24A/24B gate原样保留。首轮大CSV字段限制及NumPy布尔JSON序列化失败日志保留；仅补齐缺失诊断artifact并独立确认点预测未变，没有重拟合或结果调参。所有科学输出、失败候选、原始负实验都保留。完成后停止。\n']
    (OUT/'module25_summary.md').write_text(''.join(summary));print('MODULE25_STOP',status,'NO_GPU_NO_SEARCH',flush=True)

if __name__=='__main__':main()
