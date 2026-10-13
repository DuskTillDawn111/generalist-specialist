"""Finalize gates and six user questions without touching historical records."""
from .common import *

def main():
    assert not (OUT/'module26_gate.json').exists(),'No completed gate rewrite'
    audit=read(OUT/'positive_probe_audit.json');report=read(OUT/'replication_result.json') if (OUT/'replication_result.json').exists() else None
    auto=read(OUT/'autonomous_optimizer_check.json') if (OUT/'autonomous_optimizer_check.json').exists() else None
    if not audit['passed']:status='PROBE_INVALID'
    elif report is None:raise RuntimeError('Passed audit requires completed independent replication before finalization')
    elif report['Stage_B_replicated']:
        assert auto,'Stage C required after repeated positive evidence'
        status='POSITIVE_AND_AUTONOMOUSLY_SELECTED' if auto['autonomous_replicated_candidate_selected'] else 'POSITIVE_CONFIGURATION_REPLICATED'
    else:status=report['status']
    before=read(OUT/'historical_metadata_before.json');after=protected_metadata();changed=[p for p,v in before.items() if after.get(p)!=v];assert not changed,changed[:5]
    verify(read(OUT/'read_only_input_inventory.json'));verify(read(OUT/'read_only_implementation_inventory.json'))
    gate=dict(module=26,at_utc=now(),status=status,Stage_A_passed=audit['passed'],
        original_24A_is_legal_development_positive=audit['passed'],original_24A_used_as_independent=False,
        Stage_B_executed=report is not None,independent_pairs=report['independent_pairs'] if report else 0,
        independently_positive_configuration_replicated=bool(report and report['Stage_B_replicated']),
        independent_delta_U=report['independent_delta_U'] if report else [],paired_effect_interval=report['paired_effect_interval'] if report else None,
        Stage_C_executed=auto is not None,autonomously_selects_replicated_configuration=bool(auto and auto['autonomous_replicated_candidate_selected']),
        existence_of_positive_configuration_distinct_from_algorithm_discovery=True,
        independent_low_mid_validation_has_basis=status=='POSITIVE_AND_AUTONOMOUSLY_SELECTED',
        further_LOW_MID_research_has_positive_HIGH_basis=status in ['POSITIVE_CONFIGURATION_REPLICATED','POSITIVE_AND_AUTONOMOUSLY_SELECTED'],
        further_fixed_pool_HIGH_algorithm_diagnosis_has_basis=status=='POSITIVE_CONFIGURATION_REPLICATED',
        original_predictors_refitted=False,new_predictor_developed=False,new_pool_search=False,
        original_weights_QPS_quality_refs_replica_route_changed=False,
        longer_window_user_authorized=True,foreground_infrastructure_abort_retained=(OUT/'replication_campaign_interrupted_attempt1.json').exists(),
        historical_file_count=len(before),historical_metadata_unchanged=True,all_consumed_scientific_sources_SHA256_unchanged=True,
        next_Module_automatically_started=False,completed_and_stopped=True)
    save('module26_gate.json',gate);save('historical_integrity.json',dict(metadata_unchanged=True,scientific_hashes_unchanged=True,count=len(before),changed=[]))
    parts=list(csv.DictReader((OUT/'historical_utility_decomposition.csv').open()));historical={}
    for r in parts:historical.setdefault(r['module'],{})[r['component']]=float(r['delta_contribution'])
    lines=['# Module26：正开发探针审计与独立复现\n\n',f'最终Gate：**{status}**。不开发预测器，不搜索新池，不自动进入后续Module。\n\n',
        '## 历史开发正例审计\n\n',
        '24A +0.476798为完整1500/1500 GPU配对的系统U，S-only=-2.256672、GS18=-1.779874。原配置是五个模型各1副本，固定六任务负载，ASR18% G分流；原对照仅将ASR改为Whisper。质量路径、safe/peak显存、完整到达输入匹配及原队列规则通过，G/S有峰值拥塞告警但恢复，无硬unstable。合法开发正例，不是独立验证。\n\n',
        f'24A分项ΔU：mean={historical["24A"]["L"]:+.6f}，P95={historical["24A"]["P"]:+.6f}，吞吐={historical["24A"]["T"]:+.6f}，memory={historical["24A"]["M"]:+.6f}。收益主要来自Text/OCR背景延迟改善。\n\n',
        '24B负值来自另一配置：Text/Image/OCR各2副本、共8worker、ASR13.1833%且Text/Image有极小G流量，seed/输入cohort/顺序也不同。checkpoint、GPU、QPS、效用参考相同，但不是原18%单副本配置的独立重复。历史温度/时钟不足，不能唯一归因到硬件共置、热状态、输入或顺序。逐任务、逐worker及完整重算见positive_probe_audit.json和historical_*.csv。\n\n',
        '原窗口仅120s，LID2、VQA32，尾延迟不确定；warmup与测量窗口分离且完全drain，未完成请求/窗口截尾/归一化重算错误均未发现。启动READY修复前没有测量结果，原失败保留。\n\n']
    if report:
        lines+=['## 独立复现\n\n','用户选择每臂2400s；原模型、副本、18%路由、六任务QPS、输入池、质量和U/refs全部冻结。三组新Poisson轨迹，顺序control→proposed/proposed→control/control→proposed。冷启动温度≤38℃、连续3样本，原prime/10s warmup/2s idle；温度、时钟、功率及队列原始数据均保留。\n\n',
            '首次前台控制进程在约200s测量后消失，无完整raw请求记录，不能计算合法U；进度和日志作为不完整基础设施失败保留，不算正或负收益。根据继续指令，仅一次基础设施修复为独立后台进程+10s增量请求journal，同一冻结配置和种子，全部负结果保留，无按收益重试。源manifest及统计措辞修正均原样可审计。\n\n',
            '| 配对 | G/S U | S-only U | ΔU | 可比 |\n|---|---:|---:|---:|---|\n']
        for p in report['pairs']:lines.append(f'| {p["repetition"]} | {p["proposed_U"]} | {p["control_U"]} | {p["delta_U"]} | {p["both_legal_stable"] and p["same_all_inputs_and_trace"] and p["thermal_comparable"]} |\n')
        lines+=[f'\n独立配对t95区间：{report["paired_effect_interval"]}。n=3强依赖效应近似正态；三次同正的单侧精确符号p最小0.125，不能声称分布无关5%显著。逐组60s block bootstrap只是条件敏感性，不冒充独立重复CI。LID的P95 order-statistic区间可能无有限上界；59样本支持最大值的一侧95%覆盖，72才支持一侧97.5%，统计说明单独保留。\n\n',
            '每任务mean/P95、服务/等待、原始质量指标见per_task_latency.csv；pooled mean/P95、offered/completed/within、safe/peak显存和实际G份额见gpu_paired_runs.csv；各分项和worker busy/backlog/slope见utility_decomposition.csv及worker_stability.csv。不稳定/不完整U为null，完成子集统计仅诊断。\n\n']
    if auto:
        lines+=['## 正式优化器能力\n\n',f'原20A自主选择：{auto["original20A"]["stages"]["HIGH"]["choice"]}；冻结25评分自主选择：{auto["frozen25"]["stages"]["HIGH"]["choice"]}。是否精确选中已复现配置：20A={auto["original20A_selects_replicated_candidate"]}，25={auto["frozen25_selects_replicated_candidate"]}。\n\n',
            '完全相同五模型池、预算、合法副本和六任务负载，所有合法驻留和自由路由竞争；没有强制18%或三阶段结构。其他高预测U候选尚未测，不能算实测更优。附近16/17/19/20%只做CPU敏感性，没有GPU验证。预测器原生120s吞吐窗口与本轮2400s不同，明确保留这一限制，未开发新预测器修正它。\n\n']
    else:lines+=['\nStage B没有通过可重复正收益gate，Stage C未启动，没有新模型池搜索或附近路由GPU重试。\n\n']
    lines+=['## 六个问题\n\n',
        '1. 为什么24A与24B不同：副本数、路由、输入、运行顺序不同；效用口径相同，背景mean/P95贡献翻转，尚无唯一物理因果分离。\n',
        '2. 24A是否合法稳定真实正例：是，按原稳定性规则recoverable且完整，是开发正例，独立性不能升级。\n',
        f'3. 独立GPU是否复现正收益：{bool(report and report["Stage_B_replicated"])}；ΔU={report["independent_delta_U"] if report else []}，Gate={status}。\n',
        '4. 是否来自系统mean/P95：历史24A主要来自背景mean/P95；本轮以system_utility_comparison.csv逐组分项和per_task_latency.csv为准，不把目标吞吐或pooled P95代替系统U。\n',
        f'5. 正式优化器能否选中：{bool(auto and auto["autonomous_replicated_candidate_selected"])}；未执行Stage C不能据此断言其算法必定失败。若有正配置但未选中，报告为“存在正收益配置，但算法未成功发现”。\n',
        f'6. 继续LOW/MID研究的HIGH正例依据：{gate["further_LOW_MID_research_has_positive_HIGH_basis"]}；自主LOW/MID正式验证准备条件：{gate["independent_low_mid_validation_has_basis"]}。仅HIGH正配置但未自主发现时，先研究固定池算法排序，不能宣称已有三阶段收益。\n\n',
        f'历史{len(before)}个文件metadata和全部读取科学输入SHA256不变，24A/24B/25原gate未修改。所有失败、负结果和中断证据保留。完成后停止。\n']
    (OUT/'module26_summary.md').write_text(''.join(lines));print('MODULE26_FINAL',status,flush=True)

if __name__=='__main__':main()
