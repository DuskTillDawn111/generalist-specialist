# Module 8F-1B — Workload Intensity Breakpoint Audit + Final Resource Grid Freeze

完成纯 CPU planning 审计。RESOURCE_STATIC_GPU_CAMPAIGN_READY=True。本阶段 GPU runs=0、Proposed search calls=0；static v2/canonical routing/quality/pool/s0/g/bounds/epsilon/memory 均未修改。

1. 固定 mixture composition，lambda_base 对应旧冻结 Qwen-reference L=1。旧 .4/.72 intensity 只保存 provenance。alpha_max 随显存预算如下（最终预算，MiB）：

| Memory MiB | mix_01 alpha_max | mix_02 alpha_max | mix_03 alpha_max |
|---:|---:|---:|---:|
| 17007.174054952 | 6.574338168 | 5.172416482 | 16.227618902 |
| 27980.189424960 | 21.505756393 | 5.731116398 | 23.469896720 |
| 32019.254406267 | 21.505756393 | 5.731116398 | 23.469896720 |
| 34567.553379621 | 21.824457072 | 5.731116398 | 23.469896720 |
| 43736.541926295 | 21.824457072 | 10.903532880 | 29.532530884 |
| 49140.000000000 | 23.141751300 | 10.903532880 | 29.668595474 |

2. 所有 deployment 的 capacity breakpoints、边界 vector、bottleneck models/tasks 已保存。每个 deployment 在 alpha_limit 本身仍可行，紧随其后失效；intensity_breakpoint_curve.csv 同时保存 inclusive/right-limit count。质量 admission 下 Generalist-only 和 Specialist-only 均不可覆盖全部六任务，不能伪造 regime transition。
3. x=2 memory-feasible 不依赖 alpha；capacity-useful 为同一向量 x=1 capacity boundary 之后、x=2 boundary 以内的 (alpha_1,alpha_2] 区间。全预算空间是否必须复制另存 global necessity；没有 runtime 复制收益推断。
4. Replica activation infimum（开放下界）；逐预算/mixture 完整 intervals 及 global necessity 见 replica_activation_thresholds.json：

| Mixture | Memory MiB | Qwen | Whisper-small | ResNet50 |
|---|---:|---:|---:|---:|
| mix_01 | 17007.174054952 | 无 memory-feasible/useful x2 | 无 memory-feasible/useful x2 | 无 memory-feasible/useful x2 |
| mix_01 | 27980.189424960 | 无 memory-feasible/useful x2 | 6.574338168 | 无 memory-feasible/useful x2 |
| mix_01 | 32019.254406267 | 无 memory-feasible/useful x2 | 6.574338168 | 无 memory-feasible/useful x2 |
| mix_01 | 34567.553379621 | 无 memory-feasible/useful x2 | 6.574338168 | 无 memory-feasible/useful x2 |
| mix_01 | 43736.541926295 | 1.123877837 | 6.574338168 | 无 memory-feasible/useful x2 |
| mix_01 | 49140.000000000 | 1.123877837 | 5.927930738 | 无 memory-feasible/useful x2 |
| mix_02 | 17007.174054952 | 无 memory-feasible/useful x2 | 无 memory-feasible/useful x2 | 无 memory-feasible/useful x2 |
| mix_02 | 27980.189424960 | 无 memory-feasible/useful x2 | 无 memory-feasible/useful x2 | 无 memory-feasible/useful x2 |
| mix_02 | 32019.254406267 | 无 memory-feasible/useful x2 | 无 memory-feasible/useful x2 | 无 memory-feasible/useful x2 |
| mix_02 | 34567.553379621 | 无 memory-feasible/useful x2 | 无 memory-feasible/useful x2 | 无 memory-feasible/useful x2 |
| mix_02 | 43736.541926295 | 0.808796726 | 无 memory-feasible/useful x2 | 无 memory-feasible/useful x2 |
| mix_02 | 49140.000000000 | 0.808796726 | 无 memory-feasible/useful x2 | 无 memory-feasible/useful x2 |
| mix_03 | 17007.174054952 | 无 memory-feasible/useful x2 | 无 memory-feasible/useful x2 | 无 memory-feasible/useful x2 |
| mix_03 | 27980.189424960 | 无 memory-feasible/useful x2 | 15.717838910 | 无 memory-feasible/useful x2 |
| mix_03 | 32019.254406267 | 无 memory-feasible/useful x2 | 15.717838910 | 无 memory-feasible/useful x2 |
| mix_03 | 34567.553379621 | 无 memory-feasible/useful x2 | 15.717838910 | 无 memory-feasible/useful x2 |
| mix_03 | 43736.541926295 | 1.405594806 | 15.717838910 | 无 memory-feasible/useful x2 |
| mix_03 | 49140.000000000 | 1.405594806 | 15.717838910 | 无 memory-feasible/useful x2 |

5. 最终 intensity 使用 beta=alpha/alpha_max(w,49140)，每个 beta 都来自真实 capacity breakpoint。LOW/MODERATE 保留宽 feasibility 且全部预算至少一解；PRESSURED 保留至少 1/3 多数预算空间；HIGH 取多数预算仍至少两个解的最大断点。规则在 LP 结果前保存，不针对特定模型调整。

| Level | beta | mix_01 alpha | mix_02 alpha | mix_03 alpha |
|---|---:|---:|---:|---:|
| I1 LOW_MODERATE | 0.103780728139 | 2.401667800 | 1.131576582 | 3.079028441 |
| I2 PRESSURED | 0.401404561960 | 9.289204543 | 4.376727840 | 11.909109570 |
| I3 HIGH | 0.525620132611 | 12.163770387 | 5.731116398 | 15.594411087 |

6. B2/B3 按 5 percentage-point information separation 机械去重；保留 one-third B2，将 shifted B3 移到下一个对两侧 surviving neighbors 都分离至少 5pp 的 empirical breakpoint。minimum/SG/physical anchors 全保留。
7–8. 最终 memory budgets/resource regimes：

| ID | MiB | GiB | 8F1A fraction | SG memory feasible | x2 memory-feasible | Regime |
|---|---:|---:|---:|---|---|---|
| M1 | 17007.174054952 | 16.608568413 | 0.054526% | False | none | minimum_feasible_anchor |
| M2 | 27980.189424960 | 27.324403735 | 19.083969% | True | whisper-small, resnet50 | frozen_SG_anchor |
| M3 | 32019.254406267 | 31.268803131 | 33.369684% | True | whisper-small, resnet50 | one_third_space |
| M4 | 34567.553379621 | 33.757376347 | 38.386041% | True | whisper-small, resnet50 | separated_middle_space |
| M5 | 43736.541926295 | 42.711466725 | 66.684842% | True | qwen, whisper-small, resnet50 | two_thirds_space |
| M6 | 49140.000000000 | 47.988281250 | 100.000000% | True | qwen, whisper-small, resnet50 | physical_endpoint |

物理端点 49140 MiB 明确保留 empirical-breakpoint exception；所有其他点均为 empirical sum breakpoint。上表 fraction 使用冻结 8F1A 口径，新的 intensity-conditioned fractions/counts 完整保存在 final_memory_budgets.json。
HIGH 在部分预算存在单模型增加可解决容量失败的 trade-off，但没有 paired x2 capacity-useful 状态；ResNet50 在全部审计 mixture/budget 内没有 paired capacity gain。这些都是保留的 planning 结果，不调整 intensity 来强制复制。
9. RC-SG：遍历单 Generalist base，以最小 s0 Specialist 修复 uncovered tasks；在预算可行 bases 中按总 lambda×s0、safe memory、Generalist ID、vector 确定基底。每步添加能放入预算、对当前 assignment 有正改善的 Specialist，按总节省 worker-demand / 新增 safe memory 排序，ties 按更小 memory/model ID。不复制、不驱逐、不使用 g 或 runtime 做 selection；构造后调用 unchanged canonical routing，容量不满足只标记不可行、不修补。Generalist-centric 保留旧 B1，越预算时不运行。
10. Memory Sweep: 6 cells，composition medoid=mix_03，固定 I3，扫描所有 6 budgets。
11. Intensity Sweep: 9 cells，同一 mixture，3 levels × tight/intermediate/loose；与 A 重叠 3 cells，去重后总 12 cells ×3 methods。
12. GPU budget：未扣结构不可行的 worst-case logical evaluations/physical states=96/96，timed runs=384。CPU preflight 后 method cells ready: v2=12, Generalist-centric=0, RC-SG=12；结构不可行跳过 12 个 method cells。保守 estimated new runs=324，physical states≤84；timed-window hours≈18.900（未含启动/释放 overhead）。没有历史 exact-lambda reuse，未来 search reuse 不预先打折。
13. 全部 mixture/memory/intensity/grid/baseline/hypotheses/预算选择均在任何新 runtime 前冻结；没有查看 v2/B2 performance。
14. RESOURCE_STATIC_GPU_CAMPAIGN_READY=True。仅设计与 CPU preflight ready；本阶段执行授权为 false，GPU runs=0。

可复现：`/root/miniconda3/envs/gsdeploy/bin/python results/module8f1b_intensity_grid_freeze/run_audit.py`。

完成后停止；不启动 resource sweep、不修改 method/routing、不进入 dynamic、不使用第二硬件。

独立验证 PASS：原有 canonical pressure LP 复核全部 5772 limits，最大相对差 4.237038865532085e-16；947 curve rows、复制 intervals、RC-SG 每步 gain/tie-break、所有 ready routing/memory 与冷 seed 最小显存均已验证。选定 intensity 满足规则且下一个 capacity breakpoint 不满足，B3 replacement 为首个合格 empirical point。

已知 seed/RC-SG exact lambda/deployment/canonical-route signature 重复组=6，潜在 search raw-run savings 上界=54；仍需完整 context/phase identity 校验，保守 324-run 估计不预先打折。见 reuse_opportunities.json。

完整重建与验证顺序：run_audit.py → verify_audit.py → finalize_freeze.py（均使用 /root/miniconda3/envs/gsdeploy/bin/python）。结果数值不从日志或 Markdown 的显示精度重建。
