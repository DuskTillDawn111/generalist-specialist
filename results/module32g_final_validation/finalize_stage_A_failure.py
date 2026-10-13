"""Report the original control gate failure; no policy repair, rerun or GPU."""
from m32g.isolation import install
install('supervisor')
from m32g.common import *
import subprocess,ast
from datetime import datetime
from zoneinfo import ZoneInfo

def main():
 prior=ROOT/'results/module32f_cuda_recovery';f=read(prior/'frozen_validation_manifest_v2.json')
 assert read(prior/'module32f_gate.json')['gate']=='GPU_PREFLIGHT_RECOVERED'
 failure=read(OUT/'cpu_regression/CPU_validation_failure.json');partial=read(OUT/'cpu_regression/persisted_partial_benchmark_summary.json');eq=read(OUT/'cpu_regression/routing_equivalence_report.json')
 assert 'CONTROL_CHECK_OVER_10MS' in failure['error']
 assert not (OUT/'frozen_validation_manifest.json').exists()
 assert not list((OUT/'run_attempts').glob('*.json'))
 assert not list((OUT/'raw_runs').glob('*.json'))
 hist=read(OUT/'history_hashes/before_regression_vs_after_campaign.json')
 immutable=[x['path'] for x in f['sources']+f['frozen_inputs'] if ref(x['path'])['sha256']!=x['sha256']]
 identical={n:ref(OUT/n)['sha256']==ref(prior/n)['sha256'] for n in ['router_optimized.py','vqa_aware_dynamic_router.py','frozen_routing_parameters.json','audio_metadata_cache.json','selected_pool_catalog.json','frozen_utility_context.json']}
 env={}
 for py,old in f['environment'].items():
  now=json.loads(subprocess.check_output([py,'-c','import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'],text=True));env[py]=dict(unchanged=now==old,current=now)
 source=[ref(p) for p in sorted(OUT.glob('*.py'))]+[ref(p) for p in sorted((OUT/'m32g').glob('*.py'))]
 for x in source:ast.parse(Path(x['path']).read_text())
 save('stage_A_source_snapshot.json',dict(scope='Audit snapshot after StageA failure; not a formal GPU experiment freeze',sources=source,dependencies=env,Module32F_frozen_source_mismatches=immutable,byte_identical_policy_runtime_inputs=identical,only_namespace_port_of_runtime_CUDA_and_cache=True,original_Module32F=ref(prior/'frozen_validation_manifest_v2.json')))
 order=['P1_S','P1_D','P2_D','P2_S','P3_S','P3_D'];seeds=[320701,320702,320703]
 slots=[]
 for pair,seed in enumerate(seeds,1):
  for suffix in ['S','D']:
   base=copy.deepcopy(next(s for s in f['slots'] if s['arm']==('s_only' if suffix=='S' else 'dynamic')))
   base.update(run_id=f'P{pair}_{suffix}',pair=pair,seed=seed,measurement_seconds=2400.,stage='Module32G_PLANNED_NOT_EXECUTED',status='NOT_EXECUTED_STAGE_A_CONTROL_FAILURE',trajectories=None,frozen_offered_counts=None,U=None)
   base.pop('actual_planned_G_asr',None);base.pop('actual_planned_G_fraction',None);slots.append(base)
 save('planned_validation_manifest.json',dict(formal_frozen=False,reason='StageA CPU regression failed before formal configuration/trajectory freeze; no formal trajectory generated or used',model_pool_source=ref(prior/'frozen_validation_manifest_v2.json'),unchanged_pool_weights_quality_rates_budget_replicas=True,planned_order=order,seeds_chosen_once=seeds,slots=slots,router_P95_max_ms=1.,control_max_ms=10.,original_monitor_abort_at_ge10ms=True,conditioning_seconds=300.,warmup_seconds=30.,measurement_seconds=2400.,drain_seconds=60.,thermal_protocol=f['thermal_protocol'],environment_control=f['environment_control'],failure_policy='STOP_NO_RETRY; no strategy repair or formal GPU launch after StageA failure',retry_count=0,new_seeds_screened=False,seeds_replaced=False,original_utility_context=ref(OUT/'frozen_utility_context.json')))
 save('stage_A_failure.json',dict(stage='CPU_concurrent_regression',gate='VALIDATION_INVALID',measured_control_wall_ms=partial['failure_wall_ms'],original_threshold_ms=10.,history_multiplier=8,arrival_time_scale=50,logical_cohort_seconds=2400,completed1x=partial,failure=failure,full_failed_probe_thread_CPU_lock_wait_hold_unavailable=True,diagnosis='Wall-time control violation established. Failure exception/stack retained; no complete failed8x per-probe CSV. Scheduling/GIL/shared-lock/IO contributions cannot uniquely be attributed from saved evidence.',is_GPU_control_failure=False,is_CUDA_304_recurrence=False,formal_runs=0,formal_U=None,paired_delta_U=None,rerun=False,policy_adjusted=False))
 pairs=[dict(pair=i,seed=seeds[i-1],S_U=None,Dynamic_U=None,delta_U=None,matched_inputs_arrivals=None,G_ASR=None,S_ASR=None,real_G_and_S=None,status='NOT_EXECUTED_STAGE_A_CONTROL_FAILURE',sensitivity_without_Text_P95=None) for i in [1,2,3]]
 csvsave('utility_comparison.csv',pairs)
 csvsave('gpu_paired_runs.csv',[dict(run_id=rid,seed=seeds[int(rid[1])-1],measurement_seconds=2400.,status='NOT_EXECUTED',reason='StageA control probe15.338ms exceeds original10ms',U=None) for rid in order])
 save('pair_level_statistics.json',dict(n_complete_independent_pairs=0,planned_pairs=3,mean_delta_U=None,CI95=None,df_if_complete=2,t_critical_if_complete=4.302652729911275,historical_U_used=False,reason='No independent GPU pair executed'))
 audit=dict(passed=hist['passed'] and not immutable and all(identical.values()) and all(e['unchanged'] for e in env.values()),history=hist,frozen32F_source_mismatches=immutable,identical=identical,packages=env,original_history_Gates_not_modified=True,old32D_non_byte_exact_progress_exception_still_documented=True)
 save('final_data_acceptance.json',audit)
 final=dict(module='32G',gate='VALIDATION_INVALID',stopped_stage='A_CPU_ENGINEERING_REGRESSION',control_probe_wall_ms=partial['failure_wall_ms'],control_threshold_ms=10.,router_1x_P95_ms=partial['P95_ms'],router_8x_full_acceptance=None,routing_equivalence_exact=eq['passed'],historical_decisions=eq['decisions'],router_policy_and_thresholds_unchanged=True,formal_runs=0,planned_runs=6,three_paired_deltas=[None,None,None],CI95=None,U=None,utility_validation_not_started=True,history_preserved=hist['passed'],final_integrity_passed=audit['passed'],all_failures_retained=True,per_probe_evidence_limit='failed8x detailed probe CSV not persisted; exception value+stack+partial journal retained',seeds_replaced=False,retry_count=0,optimizer_modified=False,optimizer_integration_supported=False,Module32F_gate_unchanged=True,next_stage_started=False,at_local=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
 save('module32g_gate.json',final)
 (OUT/'module32g_summary.md').write_text(f'''# Module32G最终独立效用验证

Gate：`VALIDATION_INVALID`。StageA CPU工程回归触发原控制检查门槛，按预先声明的技术失败STOP_NO_RETRY停止。正式GPU运行0/6，独立配对0/3，U、三个ΔU、均值和CI95均NULL。**本轮没有验证Dynamic的效用优势，也没有证据将它判为效用负收益。**

## 实际通过与失败

原32F Gate仍为GPU_PREFLIGHT_RECOVERED；本轮不修改该历史Gate。原六任务池、权重、质量准入、QPS、34567.5534MiB预算、safe factor和单副本未改。原32B路由源码/阈值/背压及32F增量runtime均byte-identical；CUDA隔离和Triton cache provider只做Module32G目录/包名移植，没有算子、precision或generation修改。

本轮1381次历史同状态决策逐字段等价，容差0。历史写入保护、phase transition与CPU local Phi cache解析通过。1x历史的完整2400秒规模/50x加速真实并发CPU回放产生1070个ASR路由计时、3419个control样本，路由P95={partial['P95_ms']:.6f}ms、P99={partial['P99_ms']:.6f}ms、max={partial['max_ms']:.6f}ms；control max={partial['control_max_wall_ms']:.6f}ms。全部长尾保留，不以max路由替代原P95≤1ms验收。

随后8x历史段在m32g.monitor.IncrementalMonitor.check触发CONTROL_CHECK_OVER_10MS：原perf_counter墙钟15.338264ms>10ms。原检查包含state_lock等待和原异常谓词；未放宽门槛、删样本或重跑。CPU_validation_failure.json与cpu_validation.log保存完整异常栈，CPU_CONCURRENT_8.jsonl保存已写出的部分cohort。该段在失败前尚未写出raw_control_probes_8.csv及最终CPU_runtime_acceptance.json；thread_cpu/state_wait/state_hold等该次字段无法恢复，不能声称完整逐探针归因，也不能仅凭15.338ms断定GIL、锁或操作系统调度哪一项负责。此为加速CPU压力测试，不是GPU控制检查失败，不是CUDA304复发，更不是2400秒墙钟GPU验证。

## 预定配对的处理

已在生成前声明新种子320701/320702/320703和顺序P1_S→P1_D→P2_D→P2_S→P3_S→P3_D，没有换种子或机会筛选。StageA失败后未生成正式新轨迹、未调用正式freeze、未启动任何worker GPU测量。planned_validation_manifest.json仅保存计划与未执行原因，stage_A_source_snapshot.json仅是审计源码/环境快照，均不伪称正式GPU配置验收成功。未执行脚手架源代码保留但不会自动启动。

|组|S-only U|Dynamic U|ΔU|状态|
|---|---|---|---|---|
|1|NULL|NULL|NULL|NOT_EXECUTED|
|2|NULL|NULL|NULL|NOT_EXECUTED|
|3|NULL|NULL|NULL|NOT_EXECUTED|

没有新真实ASR分流比例、Whisper等待改善、VQA保护效果、Text偶然尾延迟或额外执行/共置成本的三组证据。去除Text P95单项的诊断敏感度也为NULL，不把32B探索性ΔU或32C无效配对替代独立结果。Module20A原六任务效用和归一化参考保持，未计算合法U；所有原始CPU数据及失败仍保留。

历史SHA/size/mtime/symlink前后核验={hist['passed']}，冻结32F源码/输入及依赖核验={audit['passed']}。历史32D旧progress不可逐字节恢复例外仅继续记录，不伪称恢复。GPU最终空闲1MiB、无计算进程，见nvidia_smi_after_stage_A.txt。没有修改正式优化器或任何历史Gate。

不进入优化器集成：路由真实GPU优势及优化器自主启用均未由本轮验证。本Module在工程失败Gate处停止，不开发新策略、不进行故障修复后重试或自动启动下一阶段。
''')
 print('STAGE_A_FINAL',final,flush=True)
if __name__=='__main__':main()
