# Module 7C-4 Multi-workload Real Integer Deployment Validation

Decision gate：**SUPPORT**。固定 W1/W2/W3，未修改 rates、mix、10 s warmup / 210 s window 或 duration。
W2 直接引用 Module 7C-3 search、confirmation、raw traces；新 GPU runs=0。W1/W3 独立 MILP seed；empirical capacity/memory、predictor=None、Top-B=2、最多 3 rounds / 6 unique states / 3 repetitions、patience=1、5% gate。
接受规则与 7C-3 同一 implementation：stability 优先；same-steady mean→weighted task P95→memory；same-nonsteady 保留 incumbent。

| Workload | Seed | Final | N_G | N_S | x=2 accepted? | Mean ms seed→final | Mean gain | Weighted task P95 ms seed→final | P95 gain | Stability | Runtime states | Enum |
|---|---|---|---:|---:|---|---|---:|---|---:|---|---:|---:|
| W1_balanced | qwen=1 | qwen=1, whisper-small=1 | 1 | 1 | False | 276.677→114.824 | 58.50% | 733.842→210.507 | 71.31% | steady→steady | 5/6 | 0 |
| W2_generalist_heavy | qwen=1 | qwen=1, whisper-small=1 | 1 | 1 | False | 260.237→105.740 | 59.37% | 711.755→200.185 | 71.87% | steady→steady | 5/6 | 0 |
| W3_specialist_friendly | qwen=1 | qwen=1, whisper-small=1 | 1 | 1 | False | 222.622→105.840 | 52.46% | 544.773→171.293 | 68.56% | steady→steady | 5/6 | 0 |

完整 12-type replica vectors（包括所有 zeros）、active types、replica counts、memory、mandatory coverage 在 JSON/CSV 中保存。

## Cross-workload observations

1. Final deployment 是否不同：False。
2. N_G 是否不同：False；观测值=[1, 1, 1]。
3. N_S 是否不同：False；观测值=[1, 1, 1]。
4. Workload-dependent G/S allocation：False；只描述本 frozen space 的观测结果。
5. 是否选择 replica=2：False；全部 x=2 proposal 的 rank、evaluated/accepted 状态见 replica_activation.json。
6. 在当前三个 workload 和单 GPU 资源条件下，搜索倾向于增加 model diversity，而不是复制同模型。
7. Empirical calibration 改变 Top-B：{'W1_balanced': True, 'W2_generalist_heavy': True, 'W3_specialist_friendly': True}；CPU counterfactual 匹配实际 incumbent/visited history/evidence/memory，没有 linear GPU campaign。
8. Runtime feedback rollback analytical Top-B：9 次；包括不改善 incumbent，以及其他 Top-B candidate 更好两种原因，全部保留。
9. 同一 evaluated deployment 跨 workload groups=4；接受/回滚 decision 不同的 groups=0。不能把仅 mean/P95 数字不同当作相反的接受方向；每组 workload/runtime/outcome 完整列在 summary JSON。
10. Memory constraint 实际排除 candidate：0 次；只统计 memory rejection，tabu/coverage 不混入。

## Independent confirmation

| Workload | Seed mean ms | Final mean ms | Mean gain | Objective direction |
|---|---:|---:|---:|---|
| W1_balanced | 253.105 | 100.088 | 60.46% | True |
| W2_generalist_heavy | 263.220 | 96.122 | 63.48% | True |
| W3_specialist_friendly | 223.038 | 103.522 | 53.59% | True |

每个 workload 的 seed/final 各 2 次独立 confirmation，不影响搜索决策；W2 复用原有 2+2。

新 search physical attempts=30；新 confirmation attempts=8；合计上限=44。Enumeration calls=0；audit errors=[]。
Qwen presence 由唯一 verified VQA capability 强制，不描述成算法基于性能主动选择。Qwen/Whisper 双副本因子来自 ASR；mixed-task transfer 和异构 GPU interference 仍是 planning surrogate，runtime feedback 用于检验候选。
没有 global optimum / universal G/S count / 2 replicas always worse / Specialists always better 的结论。只报告 frozen W1/W2/W3、candidate pool v2、当前 RTX 4090、evidence-bounded replica space、Top-B 和预算下的观测 allocation。
Evidence bounds：qwen / whisper-small / resnet50≤2，其余≤1。没有 new workloads、dynamic adaptation、new scaling campaign 或 final paper benchmark。完成后停止。
