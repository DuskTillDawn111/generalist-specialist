"""Final artifact ledger, explicit 11 required answers, automatic continuation status."""
from .common import *
def main():
 h=read(OUT/'heavy_empirical_replica_table.json');cmp=read(OUT/'structural_transition_comparison.json');gate=read(OUT/'module14d_gate.json');x3=read(OUT/'x3_profiling_decision.json');p=h['task_path_capacity_records'];v=ROOT/'results/module14e_heavy_gpu_validation/module14e_gate.json';s=ROOT/'results/module14f_heavy_stable_overflow/module14f_gate.json'
 downstream={'GPU_validation_entered':v.exists(),'GPU_validation':read(v) if v.exists() else None,'stable_overflow_search_entered':s.exists(),'stable_overflow_search':read(s) if s.exists() else None}
 if downstream['GPU_validation']:
  if downstream['GPU_validation']['continue_stable_overflow_search']:
   assert s.exists(),'Authorized automatic overflow search unfinished'
   downstream['stop_reason']=downstream['stable_overflow_search']['gate']
  else:downstream['stop_reason']=downstream['GPU_validation']['gate']
 else:downstream['stop_reason']='REVALIDATION_GATE_NOT_SUPPORTED'
 gate.update(automatic_continuation_status='COMPLETE_WITH_MANDATORY_STOP' if gate['automatic_GPU_validation_required'] else 'STOPPED_BY_REVALIDATION_GATE',automatic_GPU_validation_entered=downstream['GPU_validation_entered'],downstream_stop_reason=downstream['stop_reason'],downstream_result_directories=[str(q.parent) for q in [v,s] if q.exists()]);save('module14d_gate.json',gate);save('automatic_continuation.json',downstream)
 verify(read(OUT/'protected_before.json')['files']);verify(read(OUT/'existing_replica_integrity.json')['files']);ev=cmp['event_counts']
 text=f"""# Module 14D：Minimal Heavy Replica Profiling + Empirical Structural Revalidation

Revised gate：**{gate['gate']}**。27 个原 development contexts、E/H pool、quality/capability SLO、任务映射、memory budgets、Module9、J-routing、rho≤0.8 均保持冻结。

## 必须回答的 11 项

1. 成功测得 {sum(r['valid_all_three'] for r in p)}/7 条 task-path g(2)、M(2)，共6个checkpoint，每条x=2均3次独立repetition。原g(1)=1与M(1)保留；Turbo、RoBERTa-base、ConvNeXt-tiny既有副本表和raw证据完整性通过，未重测。
2. 真实不可行/失败 replica states：{len(h['failed_replica_states'])}。错误完整保留，不填估计值。首次STRHub启动的READY元数据兼容性错误单独保存在infrastructure_archive，未进入capacity计时，不解释为模型OOM或不可行；修复只补READY.tasks，不改handler、checkpoint、预处理、计时或并发。
3. 14C 的14个N_S reduction剩 **{cmp['persistent_original_NS_reduction_count']}/14**；消失 **{cmp['missing_replica_artifact_count']}**。本轮总N_S reduction {ev['NS_reduction']}；排除近零flow驻留后 {ev['effective_positive_flow_NS_reduction']}。
4. Generalist routing share increase：**{ev['G_share_increase']}** 个context。
5. 新G/S split：**{ev['new_GS_split']}** 个context。
6. N_G increase：**{ev['NG_increase']}** 个context。
7. 新second G：**{ev['new_second_G']}** 个context。
8. 原8个Heavy infeasible：仍不可行 {len(set(cmp['original_H_infeasible'])&set(cmp['current_H_infeasible']))}；恢复 {cmp['resolved_H_infeasible']}；新增 {cmp['new_H_infeasible']}。E可行 {cmp['E_feasible']}/27；H可行 {cmp['H_feasible']}/27。每个不可行context保留最低实测task-cover memory下界。
9. x=3 profiling decision：**{x3['decision']}**。只有completed x2 replay后，依据最优解使用已测x=2、仍并行使用同任务其它队列以及实测x1→x2正capacity增量生成需求；逐模型context witness见x3_profiling_decision.json。未运行任何新x≥3，未生成未测g/M。
10. Revised gate：**{gate['gate']}**。多个context持久N_S reduction符合用户gate；它本身不证明Generalist-sharing gain。
11. 自动进入正式bounded real-GPU Efficiency-vs-Heavy validation：**{'YES' if downstream['GPU_validation_entered'] else 'NO'}**。后续结果独立保存在14E/14F目录。最终后续停止原因：**{downstream['stop_reason']}**。

## 新实测点

|checkpoint/task path|g(2)|M(2) idle MiB|M(2) safe MiB|x2 repetitions|
|---|---:|---:|---:|---:|
"""
 for r in p:text+=f"|{r['model_id']} / {r['task_path']}|{r['g2']:.9f}|{r['M2_idle_mib']:.6f}|{r['M2_safe_mib']:.6f}|3|\n" if r['valid_all_three'] else f"|{r['model_id']} / {r['task_path']}|FAILED|unknown|unknown|3 preserved attempts|\n"
 text+="\n## 协议与解释边界\n\nCapacity沿用11B closed loop：5s warmup、30s timed capacity、30s独立lower-load stability validation，三组seed7101/7202/7303，fresh process，每worker1个outstanding。g(2)=原g(1)×三次paired capacity ratio中位数，计时窗口只统计成功完成响应，drain不计throughput。辅助x=1 reference只形成真实paired denominator，不覆盖冻结g(1)/M(1)。M(2)为postwarm idle实测均值，safe factor沿用原冻结值，另保留GPU high-water与完整request/telemetry。periodic offered-rate只用于独立稳定性，不用作service capacity。\n\nWhisper-medium的ASR和LID分别实测，task-path gain均完整保留；同checkpoint独立worker共享模型身份与residency语义。原优化器是family scalar g(x)：本轮采用两个实测gain较小值、两个path实测residency较大值，checkpoint每副本只计一次驻留。它是保守的family transfer，不是未测mixed-task capacity证书；Module9、J和任务原s0不改，独立GPU验证承担实际混合并发核验。\n\n27行context比较、54行CPU结果均有完整witness。原E的27个J*复现，逐行独立重算flow conservation、safe memory、rho与J通过，所有可行CPU结果global gap≤1e-8秒。未经测量的count保持unknown，不线性外推、不把empirical count上界当真实Specialist saturation。\n"
 text+='\n## 消除的missing-replica artifacts\n\n'+('\n'.join('- '+cid for cid in cmp['missing_replica_artifact_contexts']) or '没有原NS reduction消失。')+'\n'
 text+='\n## 后续执行与强制停止\n\n'+json.dumps(downstream,indent=2,ensure_ascii=False)+'\n'
 (OUT/'module14d_summary.md').write_text(text)
 # Retain frozen manifest as an immutable provenance file; final required
 # manifest links both initial freeze and completed observations.
 manifest=read(OUT/'replica_measurement_manifest.json');save('replica_measurement_manifest_frozen.json',manifest)
 manifest.update(status='COMPLETE',completed_at_utc=now(),frozen_manifest=ref(OUT/'replica_measurement_manifest_frozen.json'),completed_measurements=ref(OUT/'replica_measurements.csv'),successful_path_count=sum(r['valid_all_three'] for r in p),failed_replica_states=h['failed_replica_states'],task_path_results=[{k:r[k] for k in ['planning_id','model_id','task_path','g1','g2','M1_idle_mib','M2_idle_mib','M2_safe_mib','state_status']} for r in p],automatic_continuation=ref(OUT/'automatic_continuation.json'))
 # The table's original manifest reference is the frozen provenance copy,
 # with identical content hash; do not change any empirical point.
 save('replica_measurement_completion.json',manifest)
 save('final_verification.json',{'at_utc':now(),'historical_files_unchanged':True,'protected_file_count':len(read(OUT/'protected_before.json')['files']),'existing_replica_integrity_passed':True,'required_outputs_exist':all((OUT/n).exists() for n in ['replica_measurement_manifest.json','replica_measurements.csv','heavy_empirical_replica_table.json','structural_replay_empirical.csv','structural_transition_comparison.json','x3_profiling_decision.json','module14d_gate.json','module14d_summary.md']),'completed_authorized_pipeline_or_mandatory_stop':True})
 save('module14d_artifact_manifest.json',{'at_utc':now(),'files':[ref(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='module14d_artifact_manifest.json'],'new_module_sources':[ref(p) for p in sorted((ROOT/'module14d').glob('*.py'))]})
 print('CLOSEOUT',gate['gate'],downstream['stop_reason'],flush=True)
if __name__=='__main__':main()
