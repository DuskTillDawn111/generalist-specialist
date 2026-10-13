# Module 7C-2：Empirical Replica-capacity Calibration

完成 CPU planning calibration；未新增 GPU measurement、未运行真实 integer deployment search、未修改 frozen candidate pool 或旧测量结果。
全部 148 项 CPU tests 通过，旧 binary traces 保持兼容。离散 MILP 通过 3-model、R≤2、27-vector 验证专用 oracle；生产 search 不枚举 replica vectors。

| Model | g(2)，完整精度 | 第二副本 Δg | M(1) MiB | M(2) MiB | M(2)−2M(1) |
|---|---:|---:|---:|---:|---:|
| qwen | 1.2159252553989397 | 0.2159252553989397 | 17978.0 | 35670.0 | -286.0 |
| whisper-small | 1.237806637806638 | 0.2378066378066379 | 1616.0 | 3240.0 | 8.0 |
| resnet50 | 1.978474738000047 | 0.9784747380000469 | 604.0 | 1208.0 | 0.0 |

g(2) 从冻结 7C-1 high-load 三轮 paired throughput-gain mean 读取完整精度并验证 SHA256，未从摘要表抄写。M(1) 保留 v2 单实例实测 idle VRAM，确保 binary 和 x≤1 行为一致；M(2) 使用 7C-1 双实例 post-warm idle mean。7C-1 同期单实例与 peak 数据另存 memory table。

Linear baseline 保留 D≤ρx。Empirical mode 使用 D≤ρg(x)，canonical LP 使用 D≤zg(x)。MILP 对每个 model/count 使用 one-hot z[m,n]，x=Σnz、y=1−z[m,0]；coverage/routing linkage 仍使用 presence。memory 使用 ΣM(x)；目标仍是最小 memory 后最小 replica count。没有新增 memory/latency 加权和。

Qwen、Whisper-small、ResNet50 的 evidence bound=2，其余 9 types=1 且 scaling pending。R_mem 与 R_evidence 分开，R 为 memory/process/evidence/显式更小界的最小值。这不是对其他模型物理能力的判定。完整真实 v2 integer search 输入默认 empirical；旧 binary/synthetic 默认 linear。

## W1/W2/W3：同 bounds、同 memory 的容量对照

| Workload | Seed | N_G / N_S | Memory MiB | Linear Top-B | Empirical Top-B |
|---|---|---:|---:|---|---|
| W1_balanced | {'qwen': 1} | 1 / 0 | 17978.0 | [['qwen', 2], ['whisper-small', 1]] | [['whisper-small', 1], ['whisper_turbo', 1]] |
| W2_generalist_heavy | {'qwen': 1} | 1 / 0 | 17978.0 | [['qwen', 2], ['whisper-small', 1]] | [['whisper-small', 1], ['whisper_turbo', 1]] |
| W3_specialist_friendly | {'qwen': 1} | 1 / 0 | 17978.0 | [['qwen', 2], ['minicpm', 1]] | [['minicpm', 1], ['whisper-small', 1]] |

三个 cold dry-runs 的全候选排序和 Top-B 均改变。Qwen 1→2 的 rank 均从 1 降至 8；memory、bounds、workload、service demand、Top-B policy 完全相同，所以变化来自 capacity calibration。另存 linear/additive-memory baseline，未把 memory policy 改变混入以上容量对照。

Qwen seed presence 由唯一 verified visual_qa coverage 强制，不能描述成性能主动选择。每个 JSON 保存 seed、routing、effective pressure、所有生成的一跳候选（包括拒绝原因）、完整精度 Δg / memory increment / Δz / ranking、Top-B、N_G/N_S 和 trace。cold state 仅 Qwen=1，没有已驻留 Specialist，所以 cold Specialist 1→2 候选不存在。

每个 workload 额外保存固定 CPU diagnostic state：Qwen=1、Whisper-small=1、ResNet50=1。除原 guided neighborhood 外，另存该固定状态全部受 bounds 允许的 x±1 直接 mutations，无状态转移/真实 serving。它直接并列 G 1→2、S 0→1、Whisper/ResNet 1→2；该状态不是搜索得到的性能最优 allocation。Whisper/ResNet second replica 对此状态 Δz=0，原 pressure-task guidance 正确剪枝这些无当前 bottleneck relief 的扩容；不能将其称作有收益候选。

| Workload | Cold generated / feasible | Warm diagnostic generated / feasible | Warm ordering changed |
|---|---:|---:|---|
| W1_balanced | 13 / 12 | 10 / 9 | True |
| W2_generalist_heavy | 13 / 12 | 10 / 9 | True |
| W3_specialist_friendly | 13 / 12 | 10 / 9 | True |

## 直接 replica choice

| Workload / state | Candidate | Δg (empirical) | Memory increment MiB | Δz | Feasible | Rank |
|---|---|---:|---:|---:|---|---:|
| W1_balanced / cold | qwen 1→2 | 0.215925255 | 17692.0 | 0.055474991 | True | 8 |
| W1_balanced / cold | resnet50 0→1 | 1.000000000 | 604.0 | 0.055845276 | True | 7 |
| W1_balanced / cold | whisper-small 0→1 | 1.000000000 | 1616.0 | 0.154075542 | True | 1 |
| W1_balanced / cold | whisper_turbo 0→1 | 1.000000000 | 3882.0 | 0.154075542 | True | 2 |
| W1_balanced / fixed warm | qwen 1→2 | 0.215925255 | 17692.0 | 0.018197037 | True | 7 |
| W1_balanced / fixed warm | resnet50 1→2 | 0.978474738 | 604.0 | 0.000000000 | True | 10 |
| W1_balanced / fixed warm | whisper-small 1→2 | 0.237806638 | 1624.0 | 0.000000000 | True | 11 |
| W1_balanced / fixed warm | whisper_turbo 0→1 | 1.000000000 | 3882.0 | 0.000000000 | True | 12 |
| W2_generalist_heavy / cold | qwen 1→2 | 0.215925255 | 17692.0 | 0.067495424 | True | 8 |
| W2_generalist_heavy / cold | resnet50 0→1 | 1.000000000 | 604.0 | 0.068410463 | True | 7 |
| W2_generalist_heavy / cold | whisper-small 0→1 | 1.000000000 | 1616.0 | 0.183019353 | True | 1 |
| W2_generalist_heavy / cold | whisper_turbo 0→1 | 1.000000000 | 3882.0 | 0.183019353 | True | 2 |
| W2_generalist_heavy / fixed warm | qwen 1→2 | 0.215925255 | 17692.0 | 0.022846259 | True | 7 |
| W2_generalist_heavy / fixed warm | resnet50 1→2 | 0.978474738 | 604.0 | 0.000000000 | True | 10 |
| W2_generalist_heavy / fixed warm | whisper-small 1→2 | 0.237806638 | 1624.0 | 0.000000000 | True | 11 |
| W2_generalist_heavy / fixed warm | whisper_turbo 0→1 | 1.000000000 | 3882.0 | 0.000000000 | True | 12 |
| W3_specialist_friendly / cold | qwen 1→2 | 0.215925255 | 17692.0 | 0.046988721 | True | 8 |
| W3_specialist_friendly / cold | resnet50 0→1 | 1.000000000 | 604.0 | 0.055845276 | True | 7 |
| W3_specialist_friendly / cold | whisper-small 0→1 | 1.000000000 | 1616.0 | 0.112067679 | True | 2 |
| W3_specialist_friendly / cold | whisper_turbo 0→1 | 1.000000000 | 3882.0 | 0.112067679 | True | 3 |
| W3_specialist_friendly / fixed warm | qwen 1→2 | 0.215925255 | 17692.0 | 0.017170566 | True | 7 |
| W3_specialist_friendly / fixed warm | resnet50 1→2 | 0.978474738 | 604.0 | 0.000000000 | True | 10 |
| W3_specialist_friendly / fixed warm | whisper-small 1→2 | 0.237806638 | 1624.0 | 0.000000000 | True | 11 |
| W3_specialist_friendly / fixed warm | whisper_turbo 0→1 | 1.000000000 | 3882.0 | 0.000000000 | True | 12 |

## 决策

1. **Linear 对 Qwen 与 Whisper-small 明显高估**：2 相对真实 g(2) 分别高估约 64.5% 与 61.6%；ResNet50 约 1.1%，接近 linear。第二副本边际增益不再统一为 +1。
2. **Calibration 实际改变 W1/W2/W3 候选 ordering 和 Top-B**，详见上表。synthetic competition 还验证 winner 从 Qwen second replica 转为 ResNet second replica；不硬编码 Specialist 优先。
3. **三个 workload 均有 Qwen=2 可行 planning 候选**；固定 warm 全部直接邻居 diagnostic 还有 Whisper-small=2 / ResNet50=2（Δz=0，非 guided Top-B）。这些都是 CPU planning 状态，未启动副本。
4. **G 与 S 在同一 Δz ranking 和 49140 MiB 硬约束下直接竞争**。Qwen 1→2 增量 memory=17692 MiB、边际容量≈0.216；新增 Specialist 首副本容量=1。并非每对候选都互斥：Qwen=2 + Whisper-small=1 可以放入 idle 账本，但 Qwen=2 + MiniCPM=1 为 54062 MiB，超过容量。当前阶段没有求解最终 G/S allocation。
5. **Planning 已具备进入 evidence-bounded 真实 GPU validation 的基础；通用 real integer search 尚不可直接启动。** 通用 DeploymentRuntime 仍保留 count>1 guard，7C-1 独立 family#replica launcher 尚未接入通用 family routing / RuntimeEvaluator。下一阶段需先接入 counts→独立 worker IDs，并冻结小预算混合 workload validation plan。本阶段没有实现或运行这些 serving 改动。
6. **首轮小预算 validation 不必先测齐另 9 个模型**：保持三个已测 types≤2、其余≤1 即可。只有要允许未测模型第二副本时，才需先补其 scaling evidence。Qwen/Whisper 因素来自 ASR 单 task；用于其他 tasks/mix 属于 transfer surrogate，尚需真实 mixed-workload validation。异构共驻留 interference 和 peak memory 也未获校准，M(x) 是 idle-memory planning surrogate，不能保证真实 peak 可行性。

## 验证与 provenance

CPU tests: 148 passed；Case A：x≤1 的 seed、canonical LP、search serialization 完全一致（仅忽略 solver wall time）；Case B：Qwen marginal g(2)−1；Case C：ResNet 接近 linear；Case D：G/S ranking reversal。验证专用小 oracle 比较 min-memory / min-count lexicographic 最优值，涵盖可行、不可行、零 workload。

source_hashes.json 验证全部 before-manifest 冻结输入未改变，记录源代码改动及 7C-1/Stage-1/v2 hashes；模型、adapter、environment hashes 保存在 factors evidence。未安装/升级 dependencies。磁盘恢复仅将一份 pip HTTP 缓存和 metadata 移至 /tmp/gs_7c2_pip_cache_backup，未删除模型/环境/结果。

所有 dry-run runtime evaluations=0、GPU measurements=0、workers started=0。完成后停止，未进入 real search 或 dynamic adaptation。
