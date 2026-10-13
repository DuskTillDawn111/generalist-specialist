# Module 9A-0 / 9A-1 — Task-Conditioned Interference Pilot

状态：COMPLETE；gate：**GATE_PARTIAL**。8F-2 仅用于 development/failure evidence，永久禁止作为 v3 held-out final validation。

## 12 项最终回答

1. Generalists：['phi4mm', 'qwen']；12 个 v2 final occurrence：{'phi4mm': 10, 'qwen': 2}；替换 accepted：{'qwen': 2}，评估计数：{'qwen': 15, 'minicpm': 1}。
2. 全部任务：{'phi4mm': ['asr', 'visual_qa'], 'qwen': ['asr', 'lid', 'visual_qa']}；来自所有冻结实测 states 的 active/admitted/nonzero routes，包括候选 LID，没有删任务。
3. Co-models：['whisper-small', 'paddleocr']；12 个 v2 final routes 累计 r*s0 排名：[{'model': 'whisper-small', 'cumulative_worker_demand': 6.211105529252385}, {'model': 'paddleocr', 'cumulative_worker_demand': 3.8262012375477887}, {'model': 'resnet50', 'cumulative_worker_demand': 1.6021801401768825}, {'model': 'distil_whisper', 'cumulative_worker_demand': 0.5201572529978105}, {'model': 'distilbert', 'cumulative_worker_demand': 0.1828820220237435}]。
4. Historical noise floor：**1.179717279783796** absolute Delta-S。最大同 state 重复 run-means null-ratio bootstrap Q95，阈值未根据新结果修改。
5. Stable kernel signature differences：True；详见 task_path_kernel_distance.csv。
6. Fixed co-model task-conditioned slowdown detected：False；8 个 task-pair/co-model comparisons 的独立重复 bootstrap 95% CI 和噪声门槛全部保存。
7. 最大 absolute Delta-S：**0.9447368005267134**；不是人工追加模型/负载后选出的结果。
8. Kernel distance vs interference：INSUFFICIENT_FOR_CORRELATION；descriptive Spearman rho=0.0，仅 2 个 Generalist clusters，不能把共享任务和重复 co-model 当独立数据。
9. Profiler timing distortion diagnostic ratios：{'phi4mm/asr': 1.2904347928348405, 'phi4mm/visual_qa': 1.0470702799722975, 'qwen/asr': 1.2453139701260854, 'qwen/lid': 1.0908718545832619, 'qwen/visual_qa': 1.092089449413078}。profiler timing 从未作为 service ground truth 或 slowdown 分母；catalog s0 未修改。
10. Roomie primitive 可用性见 profiler_capability_audit.json：只审计 kernel resource metadata 和官方 occupancy calculator；不宣称完整 Roomie predictor 已验证。计数器权限受限不通过更换 CUDA/driver 绕过。
11. Gate：**GATE_PARTIAL**。
12. 是否建议完整 Module9A：False。完成 pilot 后停止；完整 v3、路由/排序修订均未开始。

运行统计：{'planned_profiled_runs': 15, 'planned_unprofiled_runs': 45, 'completed_profiled_runs': 15, 'completed_unprofiled_runs': 45, 'new_attempts': 60, 'reuse': 0, 'retries': 0, 'complete_deployment_searches': 0, 'CPU_prelaunch_failures': 1}。

## Task-conditioned slowdown

| Generalist | Task | Co-model | Standalone ms | Pair ms | S_real | gamma |
|---|---|---|---|---|---|---|
| phi4mm | asr | whisper-small | 1033.138105 | 1669.480829 | 1.615932 | 0.615932 |
| phi4mm | asr | paddleocr | 1033.138105 | 1175.162507 | 1.137469 | 0.137469 |
| phi4mm | visual_qa | whisper-small | 177.857513 | 264.099251 | 1.484892 | 0.484892 |
| phi4mm | visual_qa | paddleocr | 177.857513 | 206.363135 | 1.160272 | 0.160272 |
| qwen | asr | whisper-small | 708.613130 | 1007.635872 | 1.421983 | 0.421983 |
| qwen | asr | paddleocr | 708.613130 | 842.014772 | 1.188257 | 0.188257 |
| qwen | lid | whisper-small | 270.077045 | 388.687800 | 1.439174 | 0.439174 |
| qwen | lid | paddleocr | 270.077045 | 284.009414 | 1.051587 | 0.051587 |
| qwen | visual_qa | whisper-small | 151.735538 | 359.115512 | 2.366720 | 1.366720 |
| qwen | visual_qa | paddleocr | 151.735538 | 203.436596 | 1.340731 | 0.340731 |

## Contrast tests

| Generalist | Co | Tasks | Delta S | 95% CI | > historical floor | Detected |
|---|---|---|---|---|---|---|
| phi4mm | whisper-small | asr / visual_qa | 0.131040 | [0.09311968993434454, 0.1801602778305446] | False | False |
| phi4mm | paddleocr | asr / visual_qa | -0.022803 | [-0.04819980702435115, -0.0009826041641984426] | False | False |
| qwen | whisper-small | asr / lid | -0.017191 | [-0.1267220851290622, 0.09735706211429673] | False | False |
| qwen | whisper-small | asr / visual_qa | -0.944737 | [-1.0831056038893092, -0.7539808289297739] | False | False |
| qwen | whisper-small | lid / visual_qa | -0.927546 | [-1.155660275752664, -0.7106216051082551] | False | False |
| qwen | paddleocr | asr / lid | 0.136671 | [0.06123956395682778, 0.22471756828302064] | False | False |
| qwen | paddleocr | asr / visual_qa | -0.152474 | [-0.24265053949095983, -0.07565401144383163] | False | False |
| qwen | paddleocr | lid / visual_qa | -0.289145 | [-0.3434005916918083, -0.22145962155256185] | False | False |

## Interpretation limits

This pilot is task-path isolation under a fixed active-pair characterization protocol, not a new benchmark mixture or memory/intensity resource point. Saturated co-backlog is identical across target tasks. Handler-time interval overlap is not proof of simultaneous SM execution; process time-slicing, CPU preprocessing and context effects may contribute. Kernel-signature/slowdown coexistence supports a hypothesis, not a causal attribution or validated analytical predictor.

Roomie kernel/occupancy/pairwise ideas, USHER/iGniter interference-aware resources, and Cascadia/RouterWise deployment-routing coupling remain prior art. The proposed chain remains HYPOTHESIZED GAP; no first/novel claim.

Historical noise is transferred conservatively from multi-model steady runs. Limited repetitions/input coverage and profiler overhead constrain conclusions. All tasks and failed/negative contrasts are retained.

## PARTIAL gate 原因审计与最终核验

4/4 个 kernel contrasts 有稳定结构差异；7/8 个 Delta-S 的 95% CI 不跨 0；0/8 个 absolute Delta-S 超过 frozen historical noise floor=1.179717279784。最大 absolute Delta-S=0.944736800527，仍低于门槛；不改阈值，不追加 GPU。
历史门槛由 phi4mm/asr 的同状态 steady 重复驱动；CV=0.369429，run means=[952.100066, 1076.962238, 1830.6321345000001] ms。这是跨路径/多模型历史噪声的保守迁移，不是本 controlled pair 的局部噪声估计。五个 pilot paths 均有历史 steady 重复证据；判定仍按预先冻结的全局门槛执行。
Profiler 计时有可见差异，但 profiling 的三次请求连续执行，unprofiled 对照每 7 秒到达；比值包含 profiler、到达节奏、缓存/时钟与 run variability，不能当作纯 profiler overhead。Profile timing 未用于 s0、S_real 或 gate。
CPU closeout PASS：hash groups={'8F2_artifacts': 1314, 'pilot_core': 9, 'pilot_execution_sources': 8, 'prior_protected_inputs_and_sources': 181, 'selected_model_assets': 50}；60/60 runs 的输入身份、样本数、warmup、计时窗口、pair overlap、正常退出/GPU 释放等检查通过。最大 peak=21135.0 MiB；minimum pair overlap=0.987956019。
决定：停止 pilot，审计原因，不自动进入完整 Module 9A。8F-2 继续永久冻结并仅作 development/failure evidence。
