# Module 7C-1：Real Multi-replica Scaling Validation

**Decision gate: SUPPORTED。** Clear repeated capacity-pressure throughput gain exists for qwen, whisper-small, resnet50.

SUPPORTED 表示预冻结阈值下的可用 capacity-pressure throughput 增益方向得到支持。2× 线性幅度的证据按模型分开判断：Qwen / Whisper-small 被高估，ResNet50 近似成立。

固定计划完成 54 个真实 runs，其中 54 个完整有效执行；每个 condition 的重复/保护停止情况见 measurement_validation.json。
GPU: NVIDIA GeForce RTX 4090, usable ledger capacity 49140 MiB, UUID `GPU-cb0ecffa-6991-af78-e574-f2c4090b5987`。
计划 SHA-256: `8e5db7afc59ec792c19c595d7032b13d5bc0f3c81937b7bfe3f132fcc88fe308`；首次 worker 启动之前冻结，运行中没有调整负载、任务或预算。

Qwen 固定 ASR 单任务，Whisper-small 固定 ASR，ResNet50 固定 Imagenette closed-set image。使用原 Stage-1 各任务前 30 个样本，保留 frozen native adapter 的输入预处理、prompt、输出和 generation limits，按 seed shuffle 后循环。
低/中/高 offered load = 0.45/0.95/2.2 × (1000 / frozen standalone service-demand ms)。每次新建 1 或 2 个独立 CUDA worker；priming 每 replica 一条，固定 low load warmup 5 s，measurement 30 s，bounded drain 60 s，然后卸载并验证释放。两副本 family stream 均匀确定性 round-robin。

下表 throughput/idle 是三轮均值，gain/efficiency/ratio 是 paired ratios 的均值，峰值列是三轮最大 sampled peak。独立重复值、sample std、min/max 保存在 JSON。Throughput 为 measurement window 内成功完成请求/s；cohort 全部 drain 完成不等于可持续容量。Mean/P95 E2E 和 queue 是成功 cohort 请求（包括 drain 完成）的统计，始终附带 completion ratio。低负载吞吐受到达量约束，不能用低负载约 1× gain 判断最大容量。

| Model/task | Load | Offered/s | T1 | T2 | Gain mean±sample std | Efficiency | E2E mean 1→2 ms | Queue mean 1→2 ms | Completion 1/2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen/asr | low | 0.688 | 0.689 | 0.689 | 1.000 ± 0.000 | 0.500 | 776.1 → 755.7 | 11.5 → 0.5 | 1.000/1.000 |
| qwen/asr | medium | 1.452 | 1.244 | 1.389 | 1.117 ± 0.020 | 0.558 | 3172.2 → 1336.0 | 2403.9 → 311.4 | 1.000/1.000 |
| qwen/asr | high | 3.364 | 1.322 | 1.600 | 1.216 ± 0.094 | 0.608 | 23353.5 → 16254.8 | 22610.1 → 15066.4 | 1.000/1.000 |
| whisper-small/asr | low | 2.366 | 2.356 | 2.356 | 1.000 ± 0.000 | 0.500 | 200.3 → 202.6 | 2.0 → 0.4 | 1.000/1.000 |
| whisper-small/asr | medium | 4.996 | 4.922 | 4.967 | 1.009 ± 0.016 | 0.505 | 426.2 → 288.2 | 235.4 → 35.5 | 1.000/1.000 |
| whisper-small/asr | high | 11.569 | 5.156 | 6.378 | 1.238 ± 0.034 | 0.619 | 18701.8 → 12311.8 | 11970.8 → 11954.1 | 1.000/1.000 |
| resnet50/image | low | 81.851 | 81.833 | 81.833 | 1.000 ± 0.000 | 0.500 | 5.8 → 8.5 | 0.1 → 0.1 | 1.000/1.000 |
| resnet50/image | medium | 172.796 | 172.767 | 172.767 | 1.000 ± 0.000 | 0.500 | 5.8 → 5.7 | 0.3 → 0.1 | 1.000/1.000 |
| resnet50/image | high | 400.158 | 185.267 | 366.478 | 1.978 ± 0.046 | 0.989 | 17345.4 → 1405.4 | 710.4 → 639.2 | 1.000/1.000 |

| Model | Load | P95 E2E 1→2 ms | Worker mean 1→2 ms | Worker interference ratio | Stability 1 / 2 | Valid repeats 1 / 2 |
|---|---|---:|---:|---:|---|---:|
| qwen | low | 1351.4 → 1327.4 | 764.2 → 754.7 | 0.987 | steady×3 / steady×3 | 3/3 |
| qwen | medium | 5099.8 → 2622.7 | 767.9 → 1024.2 | 1.333 | recoverable_congested×3 / steady×2, recoverable_congested×1 | 3/3 |
| qwen | high | 43243.4 → 30308.1 | 743.0 → 1188.0 | 1.600 | unstable×3 / unstable×3 | 3/3 |
| whisper-small | low | 387.0 → 389.2 | 197.9 → 201.8 | 1.019 | steady×3 / steady×3 | 3/3 |
| whisper-small | medium | 779.1 → 580.7 | 190.4 → 252.4 | 1.325 | steady×3 / steady×3 | 3/3 |
| whisper-small | high | 35086.4 → 23721.0 | 192.3 → 305.5 | 1.589 | recoverable_congested×3 / unstable×3 | 3/3 |
| resnet50 | low | 6.7 → 10.5 | 5.5 → 8.2 | 1.492 | steady×3 / steady×3 | 3/3 |
| resnet50 | medium | 7.0 → 6.6 | 5.3 → 5.4 | 1.018 | steady×3 / steady×3 | 3/3 |
| resnet50 | high | 32814.7 → 2617.0 | 5.3 → 5.4 | 1.016 | recoverable_congested×3 / recoverable_congested×3 | 3/3 |

中/高负载三轮均值 throughput 下降：本次未观察到。

同卡两个 CUDA 进程的 worker latency 上升是 interference 的直接服务时间证据；还包含输入读取、preprocessing、generation 与输出处理，不能解释为纯 GPU kernel 时间。GPU utilization、memory bandwidth utilization、power 和 per-replica request/throughput 的独立重复值及 std 完整保存在 JSON/raw 中。

| Model | Load | Post-load idle 1→2 MiB | Post-warm idle 1→2 MiB | Idle ratio | Idle increment MiB | Worst sampled workload/drain peak 1→2 MiB | Additive frozen estimate for 2 MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen | low | 17446 → 34892 | 17909 → 35670 | 1.992 | 17761 | 18186 → 36018 | 35956 |
| qwen | medium | 17446 → 34892 | 17909 → 35670 | 1.992 | 17761 | 18390 → 36198 | 35956 |
| qwen | high | 17446 → 34892 | 17909 → 35670 | 1.992 | 17761 | 18390 → 36198 | 35956 |
| whisper-small | low | 1400 → 2800 | 1621 → 3240 | 1.999 | 1619 | 1622 → 3244 | 3232 |
| whisper-small | medium | 1400 → 2800 | 1621 → 3240 | 1.999 | 1619 | 1622 → 3244 | 3232 |
| whisper-small | high | 1400 → 2800 | 1621 → 3240 | 1.999 | 1619 | 1622 → 3244 | 3232 |
| resnet50 | low | 504 → 1008 | 604 → 1208 | 2.000 | 604 | 604 → 1208 | 1208 |
| resnet50 | medium | 504 → 1008 | 604 → 1208 | 2.000 | 604 | 604 → 1208 | 1208 |
| resnet50 | high | 504 → 1008 | 604 → 1208 | 2.000 | 604 | 604 → 1208 | 1208 |

与 additive estimate 的差额（使用 high condition，MiB）：

- qwen: approximately additive (within 5%); actual idle V2−2V1=-147; actual idle V2−frozen 2Vm=-286; worst observed idle V2−frozen 2Vm=6; worst observed peak V2−frozen idle 2Vm=242.
- whisper-small: approximately additive (within 5%); actual idle V2−2V1=-1; actual idle V2−frozen 2Vm=8; worst observed idle V2−frozen 2Vm=12; worst observed peak V2−frozen idle 2Vm=12.
- resnet50: approximately additive (within 5%); actual idle V2−2V1=0; actual idle V2−frozen 2Vm=0; worst observed idle V2−frozen 2Vm=0; worst observed peak V2−frozen idle 2Vm=0.

内存数值是 NVML owned CUDA PID residency 总和；device-wide 值同时保存。Peak 是 0.5 s 采样 high-water mark，可能遗漏更短暂峰值。Idle ratio 小于 2 不自动证明共享模型权重或库，allocator cache / workspace / input history 也会影响；大于 frozen idle additive estimate 的差额也不是更改 7B ledger 的理由。本阶段未修改 planning memory surrogate。

High-load backlog / host backpressure evidence（保持原 DQA label）：

| Model | Replicas | Submitted backlog at end | Offered backlog at end | Host schedule delay mean ms | Drain mean s |
|---|---:|---:|---:|---:|---:|
| qwen | 1 | 61.3 | 61.3 | 0.2 | 45.2 |
| qwen | 2 | 53.0 | 53.0 | 0.2 | 31.6 |
| whisper-small | 1 | 78.3 | 192.7 | 6538.5 | 36.9 |
| whisper-small | 2 | 151.0 | 156.0 | 52.1 | 24.7 |
| resnet50 | 1 | 139.3 | 6447.0 | 16629.5 | 34.5 |
| resnet50 | 2 | 267.3 | 1010.7 | 760.7 | 2.8 |

| Model | 7B linear surrogate assessment | Exploratory g(2) | gamma=g(2)/2 | Usable gain observed |
|---|---|---:|---:|---|
| qwen | over-estimates real scaling | 1.216 | 0.608 | True: medium, high |
| whisper-small | over-estimates real scaling | 1.238 | 0.619 | True: high |
| resnet50 | approximately supported | 1.978 | 0.989 | True: high |

g(2)/gamma 是固定高负载测量窗口的探索性 throughput characterization，不是精确容量估计。判定约 2× 的预冻结区间为 1.8–2.2；clear usable gain 需三个可比完整 pairs 全部 >1.10，并观察到单实例 queue/capacity pressure。保护停止、执行失败的 ratio 仅描述，不用于容量证明。

Observed sustainable region 仅列实际测试点；不声称精确最大 capacity，也不证明长时间稳定性。Recovered congestion 只证明有限 burst 能 drain，不能证明该 offered rate 可持续；stdin backpressure 可能使 submitted backlog 的 tail 变平，offered backlog / host schedule delay 仍在增加。DQA labels 完全沿用，两个 backlog 定义分别报告：

- qwen ×1: steady tested req/s = 0.688; recovered congestion = 1.452; unstable = 3.364; incomplete/failed = none.
- qwen ×2: steady tested req/s = 0.688; recovered congestion = 1.452; unstable = 3.364; incomplete/failed = none.
- whisper-small ×1: steady tested req/s = 2.366, 4.996; recovered congestion = 11.569; unstable = none; incomplete/failed = none.
- whisper-small ×2: steady tested req/s = 2.366, 4.996; recovered congestion = none; unstable = 11.569; incomplete/failed = none.
- resnet50 ×1: steady tested req/s = 81.851, 172.796; recovered congestion = 400.158; unstable = none; incomplete/failed = none.
- resnet50 ×2: steady tested req/s = 81.851, 172.796; recovered congestion = 400.158; unstable = none; incomplete/failed = none.

验证：passed；full SHA checks 732 文件，旧 7B、candidate pool、真实 service-demand catalog、worker/model/input bytes 保持不变；environment matches frozen v2 = True；CPU regression 132 passed = True。保留四个未重跑的原 synthetic/scalability wrapper 证据，没有重跑旧 campaigns。
未进行 12-model integer deployment search、Generalist-vs-Specialist allocation optimization、routing search、dynamic adaptation、predictor retraining 或 3+ replicas。Module 7B 的 D_m ≤ rho_max x_m 仍是 analytical planning surrogate，未据此次 characterization 自动改写。完成此报告后停止。

Artifacts: replica_scaling_plan.json / plan_freeze.json / raw_runs / campaign_ledger.json / replica_scaling_metrics.json / memory_scaling.json / capacity_observations.json / planning_surrogate_validation.json / measurement_validation.json / source_hashes*.json / environment_manifest.json / cpu_regression*.json|log / qwen_replica_analysis.md。
