"""Final evidence report; no optimizer integration and no formal hotspot runs."""
from .common import *
from .analyze_costs import main as analyze_costs
from .plots import main as plots
from .replay import gate

def main():
 analyze_costs();plots();g=gate()
 scaling=read(OUT/'replica_scaling.json');cost=read(OUT/'fixed_cost_characterization.json');windows=read(OUT/'updated_windows.json')['rows'];lat=read(OUT/'updated_latency_witnesses.json')['rows']
 required=read(OUT/'measurement_plan.json')['required_initial_frontier'];terminal=scaling['terminal_frontiers']
 positive1=[w for w in windows if w['W_1G'] and w['W_1G']>0];positive2=[w for w in windows if w['SECOND_GENERALIST_CAPACITY_NECESSARY']]
 primary=read(OUT/'replica_evidence_gate.json')['primary_candidate'];candidate=next(w for w in windows if w['budget']=='B6' and w['set_id']=='asr+image+lid')
 delta=[r for r in lat if r['DeltaJ_reference_seconds'] is not None and r['DeltaJ_reference_seconds']>1e-8]
 frozen_positive=[r for r in lat if r['DeltaJ_frozen_seconds'] is not None and r['DeltaJ_frozen_seconds']>1e-8]
 frozen_certified=[]
 for r in delta:
  good=True
  for key in ['frozen_Module9_1G','frozen_Module9_2G']:
   f=r[key];good=good and bool(f and f['converged'] and f.get('operational_certificate') and f['operational_certificate']['passed'])
  if good:frozen_certified.append(r)
 summary={'replica_gate':g,'W1_positive_contexts':len(positive1),'W2_positive_certified_contexts':len(positive2),
  'max_W1':max((w['W_1G'] for w in positive1),default=0.),'max_W2':max((w['W_2G'] for w in positive2),default=0.),
  'diagnostic_B6_ASR_Image_LID':candidate,'primary_candidate':primary,
  'mathematical_J_positive_comparisons':len(delta),'both_frozen_routes_operationally_accepted_positive_reference_comparisons':len(frozen_certified),
  'original_frozen_solver_positive_comparisons':len(frozen_positive),'fixed_cost_stage_complete':cost['stage_complete'],'formal_hotspot_GPU_runs':0,'Task_Rich_started':False,
  'Module9_unchanged':True,'new_calibration_in_separate_artifact':True}
 save('module11b_summary.json',summary)
 lines=['# Module 11B：replica evidence 与固定成本', '',f'Gate：`{g}`。正式 multi-hotspot GPU 和 Task-Rich 均未启动。Module 9、worker/runtime、旧 measured g/M、11A 和旧 hotspot 结果均保留。',
 '', '## Replica evidence', '', 'Closed-loop 只测 service capacity；periodic 只提供真实 arrival、E2E、latency、stability evidence。相关 worker 同步处理单请求，无跨请求 batching；每 replica 最多一个 outstanding。5 秒 warmup、30 秒窗口、3 independent repetitions，drain 不计吞吐。新 g(x) 沿历史锚点乘以三组相邻 count 实测 throughput ratio 的 median，历史 s0/g/M 不改。代表任务 scalar 向其它 task/混合 runtime 迁移仍有局限。',
 '', '| Model | 原缺失 count | 最终实测 bound | 实测终止 |', '|---|---:|---:|---|']
 for r in required:
  m=r['model'];t=terminal.get(m);lines.append(f"| {m} | {r['required_x']} | {scaling['evidence_bounds'][m]} | {t['reason']+' @ '+str(t['x']) if t else 'Generalist：已补齐≤2G所需count；未扩展无关count'} |")
 lines+=['',f"1–2. 原缺失的 counts 和直接涉及的 contexts/boundaries 见 `replica_evidence_required.csv`；上述 Specialist 的原 frontier 限制 capacity breakpoints 或 conditional 2G window，Phi/MiniCPM 双 replica 涉及 2G 结构。只沿这些 Specialist 的下一 count 逐级补测。",
 f"3–4. 补测后 W1>0 的 contexts={len(positive1)}，max W1={summary['max_W1']:.12g}；证据闭合且 W2>0 的 contexts={len(positive2)}，max W2={summary['max_W2']:.12g}。原 normalized headrooms 未重算。",
 f"5. {'存在' if positive2 else '未确认'}真正 1G infeasible / 2G feasible 区间。每个正 W2 的 midpoint 位于全局 MILP alpha1 上界之上，并有 2G fixed-LP witness；没有用未测 count 外推。",
 f"6. B6 / ASR+Image+LID：alphaS={candidate['alpha_S']:.12g}，alpha1={candidate['alpha_1G']:.12g}，alpha2={candidate['alpha_2G']:.12g}，W2={candidate['W_2G']:.12g}。{'保留为可考虑的后续 GPU context；正式运行仍未启动。' if candidate['SECOND_GENERALIST_CAPACITY_NECESSARY'] else '不能沿用旧 conditional W2 作为正式 2G primary。应按更新 evidence gate/CPU latency evidence 决定。'}",
 '', 'CPU J 数值对照使用同一 J objective 和 rho≤0.8，切平面 MILP dual lower bound 与 feasible convex routes upper bound 给出 10 ns global gap certificate。它是离线数学对照，未替换 frozen Module 9 router。实际 J1G/J2G 只计原 frozen solver 的 operationally accepted returns；按 deployment identity 调用原 solver，用数学 lower bounds 剪枝剩余 states，未返回/不合法 routes 原样记录，reference route 不作修复或替代。',
 f"可分辨（DeltaJ>既有 10 ns 数值 tolerance）的数学 J 改善 comparison={len(delta)}；实际 frozen solver 全局最佳 J 改善 comparison={len(frozen_positive)}。完整记录见 `updated_oneG_vs_twoG_j.csv`。",
 '', '## Specialist 固定系统成本', '', '以下为相同 G ladder 相邻 S 的三次独立、配对 post-warm idle 增量；RSS 为 process-tree RSS sum，可能重复计算共享页；CPU 100%=一个 core。精确 CUDA context count 不可观测，compute PID 仅为 proxy。',
 '', '| 新增 S / resident set | GPU MiB（mean Δ）| Host GiB（mean Δ）| CPU core %（mean Δ）| Process Δ | Compute PID Δ |', '|---|---:|---:|---:|---:|---:|']
 for r in cost['resource_increments']:
  d=r['paired_resource_increments']
  if d['owned_used_mib']['n']==3:
   lines.append(f"| {','.join(r['added_specialist'])} / {r['set_id']} | {d['owned_used_mib']['mean']:.3f} | {d['host_RSS_bytes']['mean']/2**30:.3f} | {d['worker_CPU_core_percent']['mean']:.5f} | {d['owned_process_tree_count']['mean']:.3f} | {d['compute_PID_count']['mean']:.3f} |")
 supported=[r for r in cost['low_load_effects'] if r['paired_differences']['G_mean_E2E_ms'].get('repeatable_positive_supported')]
 supported_by_mode={mode:sum(r['mode']==mode for r in supported) for mode in ['G_ONLY','G_PLUS_LOW_S','RARE_ACTIVATION_MINUS_IDLE_RESIDENCY']}
 lines+=['', '7–9. 上表与 `fixed_cost_measurements.csv` 保留总量、per-PID VRAM/RSS、CPU、startup、process/compute-PID 原始测量。CPU/latency 不设人为性能阈值；三重复的不确定性见 JSON。',
 f"10. 在可比较 steady low-load phases 中，三组 latency 差值同向且描述性 paired 95% t interval 全为正的 mean slowdown comparisons 分开为 {supported_by_mode}。G_ONLY 才是 idle residency 对照；rare-active 包含实际 S service work，不能解释成纯 idle/context 固定 penalty。这只是所测 resident sets / visual_qa 的局部证据，不是普遍 context-interference 定律；完整 mean/P95/throughput 与 CI 在 `fixed_cost_characterization.json`。",
 '11. DeltaD=lambda*(sG−sS) 随 lambda→0 趋零；resident GPU/host memory、worker/process residency 保留。曲线分别使用各资源单位，不合并为 scalar。',
 '12–13. 固定资源成本与趋零收益支持审视低 QPS Specialist 的机会成本；不能据此普遍宣称低 QPS 一定不值得部署。是否值得取决于实际资源约束、替代 resident state 和质量要求。没有人为 Specialist/context penalty，也没有 equivalent memory。',
 '', '![Separate service benefit and fixed resource curves](/root/autodl-tmp/generalist-specialist/results/module11_replica_and_cost/low_qps_separate_resource_curves.svg)',
 '', '## 下一步范围', '',
 f"14. 当前 6-task pool 的 planning evidence gate 是 `{g}`。{'已有允许考虑正式 multi-hotspot GPU 的 2G capacity context，但混合 runtime 稳定性需未来验证。' if positive2 else '未保留证据闭合的 2G capacity window；不能基于旧 11A conditional window 启动正式 2G capacity campaign。'} 本模块没有正式 GPU hotspot result。",
 '15–16. 本轮不修改资源 formulation。GPU VRAM 是已冻结的约束；新测 host RSS/startup/process 数据用于资源 characterization。只有 host RAM 或 CPU 的实际 budget/bottleneck 得到证实时才考虑纳入约束；不可直接加入 CPU/context 数量 penalty。compute-PID proxy 不是可加权的物理资源量，低 QPS benefit 也不等同于 negative interference。',
 '', '全部原始 observation、独立 repetitions、协议和 source refs 保存在本目录；旧 periodic raw 不覆盖，作废 hotspot 归档不覆盖。达到真实终止条件后更高 Specialist counts 不再被视为 artificial missing evidence，也未加入 extrapolated g/M。']
 (OUT/'module11b_summary.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
