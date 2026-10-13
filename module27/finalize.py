"""Full evidence tables, gate, history integrity; never starts a next Module."""
from .common import *
from .analyze import write_run_tables

def main():
 f=read(OUT/'discovery_protocol.json');campaign=read(OUT/'campaign.json');b=read(OUT/'exploratory_result.json')
 c=read(OUT/'independent_result.json') if (OUT/'independent_result.json').exists() else None
 a=read(OUT/'optimizer_autonomous_decision.json') if (OUT/'optimizer_autonomous_decision.json').exists() else None
 if c and c['Stage_D_permitted']:assert a is not None,'CPU autonomy must finish before finalization'
 status=a['status'] if a else c['status'] if c else b['status']
 runs=write_run_tables(campaign['executed_B']+campaign['executed_C'])
 before=read(OUT/'historical_metadata_before.json');after=protected_metadata();changed=[p for p,v in before.items() if after.get(p)!=v]
 assert not changed,changed;verify(read(OUT/'consumed_historical_sources.json'));verify(f['implementation'])
 integrity=dict(metadata_unchanged=True,historical_files=len(before),consumed_scientific_SHA256_unchanged=True,changed=[]);save('historical_integrity.json',integrity)
 gate=dict(module=27,at_utc=now(),status=status,Stage_A_passed=True,Stage_B_GPU_attempts=campaign['B_attempts'],Stage_B_budget=6,
  Stage_B_user_maximum=12,Stage_C_GPU_attempts=campaign['C_attempts'],positive_exploratory_candidate=b['selected'],
  independently_replicated=bool(c and c['independent_positive']),autonomously_selected=bool(a and a['exact_replicated_configuration_selected']),
  all_gpu_runs_preserved=True,no_outcome_retry=True,no_budget_expansion=True,original_utility_quality_references_unchanged=True,
  historical_integrity=integrity,remaining_work=False,next_Module_automatic=False,completed_and_stopped=True,
  failure=campaign['fatal'],queue_failure=campaign['queue_stop'])
 save('module27_gate.json',gate)
 selected=b['selected'];point=next((p for p in f['points'] if selected and p['point_id']==selected['point_id']),None)
 summary=['# Module27：GPU实测正分流发现',f'\nGate：**{status}**。停止；不自动进入后续Module。',
  '\n固定一个五模型池：Qwen2.5-Omni、Whisper-medium、RoBERTa-base-SST2、ConvNeXt-tiny、PARSeq，均1副本。B4=34567.5534MiB。',
  '\n目标LID=10.5/11.5QPS，Text/Image/OCR/ASR/VQA各0.2QPS；ASR与LID共享Whisper-medium，VQA走Qwen。只改变LID的0/10/20%G份额。',
  f'\n预注册StageB最多6次（用户上限12），每次600s；实际启动{campaign["B_attempts"]}次。完整搜索空间、失败/跳过及理由见discovery_protocol.json、candidate_screening.csv、stage_b_campaign.json。',
  '\n效用严格沿用20A的offered-QPS加权任务mean/P95、窗口吞吐损失及safe memory；pooled P95单报，不能代替任务P95。原始质量准入为硬约束，未补齐缺失质量分数。',
  '\n所有native service、queue/E2E、worker busy/backlog、实际route、热状态/时钟/功率、READY/模型身份、退出及释放均保留。缺乏精确worker receipt ACK，host send-start不冒充接受时刻；显存峰值为0.5s NVML采样高水位。',
  '\n探索比较：','\n|负载|G份额|ΔU|合法稳定|温度可比|进入独立复现|','|---|---:|---:|---|---|---|']
 for p in b['pairs']:summary.append(f'|{p["point_id"]}|{p["G_fraction"]}|{p["delta_U"]}|{p["both_legal_stable"]}|{p["thermal_comparable"]}|{p["qualifies_for_independent_replication"]}|')
 summary+=['\n600s开发窗口的P95采用逐任务实际n报告条件order-statistic区间；至少72个样本才有有限equal-tail95上界，仍依赖iid，队列时间相关不保证。60s block区间仅单次配对的条件敏感性，开发正例不能算独立验证。']
 if c:
  summary+=['\n独立测量每臂2400s，背景每任务期望480请求，种子/顺序在首次探索GPU前已经预注册；开发数据排除。',
   f'\n三组独立ΔU：{c["independent_delta_U"]}；近似配对t95：{c["paired_effect_interval"]}。',
   '\n只有三组效应都>+0.01、全部合法稳定/同驻留同输入同轨迹/热状态可比且t95下界>0，才进入自主优化。n=3效应正态假设强；三次同正的精确单侧符号p最小0.125，不作分布无关5%显著或跨模型泛化宣称。']
 else:summary.append('\n没有合格的开发正候选，StageC独立GPU未执行。')
 if a:summary.append(f'\n正式20A联合部署/副本/路由自由竞争结果：{a["status"]}。预测更优的其他副本/比例没有新的GPU证据；不把邻近分流比例等同已验证配置。')
 else:summary.append('\n独立正收益门槛未通过，StageD正式优化器检查未执行；不能由未执行断言算法必定失败。')
 summary+=['\n六个问题：',f'\n1. 真实正分流：开发候选={bool(selected)}，独立复现={bool(c and c["independent_positive"])}。',
  f'2. 胜出配置：{point["rates"] if point else None}；G份额={selected["G_fraction"] if selected else None}，模型/副本同上述冻结池。',
  '3. 原因必须以匹配请求的S/G服务、共享队列和全部六任务U分项为依据；见exploratory_utility_comparison.csv/independent_paired_utility.csv的matched_path_service_effects、per_task_latency.csv及utility_decomposition.csv。不能把名义容量或单目标吞吐当真实U收益。',
  f'4. 独立GPU复现：{c["status"] if c else "NOT_EXECUTED"}。',f'5. 正式优化器：{a["status"] if a else "NOT_EXECUTED"}。',
  '6. 若未找到，仍缺少在同驻留、稳定可比且采样充分的独立配对上同时改善S队列并限制G慢路径/背景干扰的正效用证据；不得靠改效用、重试种子或扩充搜索制造结果。',
  f'\n历史{len(before)}文件metadata与读取科学来源SHA256不变，14F–26原实验/gate保持原样。所有负结果及不稳定的U=null证据保留。']
 (OUT/'module27_summary.md').write_text('\n'.join(summary)+'\n')
 print('MODULE27_FINAL',status,flush=True)

if __name__=='__main__':main()
