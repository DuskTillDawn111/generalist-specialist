# Module 8F-2 Resource-Constrained Static Real GPU Campaign

状态：**COMPLETE**；CPU 完成审计：**PASS**。12 个冻结单元和 3 个方法全部保留。缺失测量为缺失值，不填造延迟。

GPU UUID 差异按用户明确授权放行；GPU 型号、驱动、49140 MiB 物理容量及冻结环境已经核验。原始 8F-1A/1B 输入与 static v2 源码未修改；旧 UUID 的历史测量未复用。

## 16 项最终回答

1. 新增 timed runs：**192**；有效完成 192；物理启动 192；独立确认 72；重试 0。
2. 实际复用：**36** 次搜索重复测量引用；历史复用 0，确认复用 0。54 个提前标记机会均已在 GPU 前审计，45 个跨预算引用禁止、9 个同单元引用条件允许；后续相同候选共享也必须满足同一冻结身份规则。
3. Memory Sweep 的 v2 vs RC-SG：**{'TIE': 6}**。这是 6 个 HIGH 单元的比较，不混入 LOW 单元胜负。
4. Feasibility win budgets：[]。结构不可行没有虚构延迟。
5. M1→M6 的部署组成见下表，所有数值来自独立 final 确认；未完成的点显式保留。
6. Generalist-heavy→Hybrid→Specialist-heavy 链是否出现：**False**（None 表示实验未齐）。M1→M6 实际显示类别：['Specialist-heavy', 'Specialist-heavy', 'Specialist-heavy', 'Specialist-heavy', 'Specialist-heavy', 'Specialist-heavy']。Frozen predicates 有重叠：Hybrid = N_G>0 且 N_S>0；Specialist-heavy = N_S>N_G；原始两个标记全部保留，不重新定义阈值。
7. M6 是否接近 RC-SG：{'C10': True, 'C11': False, 'C12': False}；按冻结 same-stability/max(1ms,5%) 规则判断，latency gap、部署 L1 distance 与 safe memory gap 的数值附于下文。
8. Tight-resource 正式结果：{'M1': 'TIE', 'M2': 'TIE', 'M3': 'TIE'}；Generalist task sharing：{'M1': {'phi4mm': ['asr', 'visual_qa']}, 'M2': {'phi4mm': ['asr', 'visual_qa']}, 'M3': {'phi4mm': ['asr', 'visual_qa']}}。只有正式 WIN/FEASIBILITY_WIN 才列为 objective 优势；相同部署的测量 WIN 不证明部署选择优势，双方不稳定时 TIE 也不意味着均值相等。完整 mean/P95/内存差见 tight_resource_value.json 和 Table A。
9. Offered load 是否改变部署：{'M1': False, 'M4': True, 'M6': True}（None 表示未齐）。下文展示各固定预算 I1→I2→I3 的实际向量；向量不变时如实记录延迟/稳定性变化。
10. x=2 evaluated / accepted 候选次数：{'qwen': {'evaluated': 0, 'accepted': 0}, 'whisper-small': {'evaluated': 3, 'accepted': 0}, 'resnet50': {'evaluated': 0, 'accepted': 0}}。这是候选 occurrence，不是物理重复次数。
11. 未接受原因：{'qwen': {'PRUNED_TOP_B': 3, 'canonical_capacity_lp': 2}, 'whisper-small': {'memory': 3, 'PRUNED_TOP_B': 11, 'mean_worse': 1, 'stability_worse': 2, 'runtime budget exhausted': 1}, 'resnet50': {}}。未生成、显存不符、Top-B 剪枝与 runtime 回滚分开记录；不得增加强度强行激活。
12. Planning-safe 但 actual peak 超预算：**0**。每份真实测量的 plan/safe/idle/peak/B 在 physical_memory_audit.csv；任何违规均保留为 safety-model failure，预算不扩大。
13. 有效运行稳定性计数：{'steady': 71, 'recoverable_congested': 22, 'unstable': 99}；执行事件分类：{'recoverable_congested': 22, 'unstable': 99}。campaign 停机错误：无。OOM、外部中断和真实 unstable 不合并。
14. H1–H5：**{'H1': 'NOT_SUPPORTED', 'H2': 'PARTIAL', 'H3': 'PARTIAL', 'H4': 'PARTIAL', 'H5': 'INSUFFICIENT'}**；证据与限制见下文。假设不要求全部成立。
15. 联合部署的必要性与 v2 的优势分别判断：已冻结 8F-1B 规划曲线中，这 12 个点的纯 Generalist / 纯 Specialist 可行数均为 0；runtime 结果进一步检验具体联合部署的稳定性和表现。此结构证据不证明 v2 全局最优或普遍优于 RC-SG。
16. 预定 Static main GPU 数据采集是否完成：**True**。本冻结 campaign 可结束；论文性能主张必须服从本次正面和负面证据。

## Table A — Memory Sweep

| Budget / MiB | v2 deployment | RC-SG deployment | v2 / RC mean ms | v2 / RC P95 ms | v2 / RC safe MiB | Outcome |
|---|---|---|---|---|---|---|
| M1 / 17007.174054952 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 7768.688 / 7781.490 | 13742.745 / 13936.883 | 17007.174 / 17007.174 | TIE |
| M2 / 27980.189424960 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | distil_whisper×1, distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 7570.974 / 3368.446 | 13593.190 / 5752.200 | 17007.174 / 21086.230 | TIE |
| M3 / 32019.254406267 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | distil_whisper×1, distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 7682.876 / 3304.604 | 13725.282 / 5668.171 | 17007.174 / 21086.230 | TIE |
| M4 / 34567.553379621 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | distil_whisper×1, distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 7646.776 / 3251.769 | 13704.342 / 5574.452 | 17007.174 / 21086.230 | TIE |
| M5 / 43736.541926295 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | distil_whisper×1, distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 7757.194 / 2863.797 | 13960.482 / 4852.315 | 17007.174 / 21086.230 | TIE |
| M6 / 49140.000000000 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | distil_whisper×1, distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 7566.623 / 3019.880 | 13458.956 / 5140.497 | 17007.174 / 21086.230 | TIE |

## Table B — Intensity Sweep

| Budget / Intensity | v2 deployment | RC-SG deployment | v2 / RC mean ms | v2 / RC stability | Outcome |
|---|---|---|---|---|---|
| M1 / I1 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 28.264 / 30.072 | steady / steady | WIN |
| M1 / I2 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 2043.256 / 2126.843 | unstable / unstable | TIE |
| M1 / I3 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 7768.688 / 7781.490 | unstable / unstable | TIE |
| M4 / I1 | distil_whisper×1, distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | distil_whisper×1, distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 26.771 / 26.108 | steady / steady | TIE |
| M4 / I2 | distil_whisper×1, distilbert×1, paddleocr×1, qwen×1, resnet50×1, whisper-small×1 | distil_whisper×1, distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 34.488 / 41.700 | steady / steady | WIN |
| M4 / I3 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | distil_whisper×1, distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 7646.776 / 3251.769 | unstable / unstable | TIE |
| M6 / I1 | distil_whisper×1, distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | distil_whisper×1, distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 27.319 / 27.284 | steady / steady | TIE |
| M6 / I2 | distil_whisper×1, distilbert×1, paddleocr×1, qwen×1, resnet50×1, whisper-small×1 | distil_whisper×1, distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 37.245 / 41.107 | steady / steady | WIN |
| M6 / I3 | distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | distil_whisper×1, distilbert×1, paddleocr×1, phi4mm×1, resnet50×1, whisper-small×1 | 7566.623 / 3019.880 | unstable / unstable | TIE |

## Table C — Deployment Composition

| Budget | N_G | N_S | Total replicas | Additional replicas | Frozen regime |
|---|---|---|---|---|---|
| M1 | 1 | 4 | 5 | 0 | Specialist-heavy; hybrid=True, specialist_heavy=True |
| M2 | 1 | 4 | 5 | 0 | Specialist-heavy; hybrid=True, specialist_heavy=True |
| M3 | 1 | 4 | 5 | 0 | Specialist-heavy; hybrid=True, specialist_heavy=True |
| M4 | 1 | 4 | 5 | 0 | Specialist-heavy; hybrid=True, specialist_heavy=True |
| M5 | 1 | 4 | 5 | 0 | Specialist-heavy; hybrid=True, specialist_heavy=True |
| M6 | 1 | 4 | 5 | 0 | Specialist-heavy; hybrid=True, specialist_heavy=True |

## Table D — Feasibility

| Cell | v2 | RC-SG | Generalist-centric |
|---|---|---|---|
| C01 | COMPLETE | COMPLETE | MEMORY_INFEASIBLE |
| C02 | COMPLETE | COMPLETE | MEMORY_INFEASIBLE |
| C03 | COMPLETE | COMPLETE | MEMORY_INFEASIBLE |
| C04 | COMPLETE | COMPLETE | CAPACITY_INFEASIBLE |
| C05 | COMPLETE | COMPLETE | CAPACITY_INFEASIBLE |
| C06 | COMPLETE | COMPLETE | CAPACITY_INFEASIBLE |
| C07 | COMPLETE | COMPLETE | CAPACITY_INFEASIBLE |
| C08 | COMPLETE | COMPLETE | CAPACITY_INFEASIBLE |
| C09 | COMPLETE | COMPLETE | CAPACITY_INFEASIBLE |
| C10 | COMPLETE | COMPLETE | CAPACITY_INFEASIBLE |
| C11 | COMPLETE | COMPLETE | CAPACITY_INFEASIBLE |
| C12 | COMPLETE | COMPLETE | CAPACITY_INFEASIBLE |

## Table E — Replica Audit

| Model | Generated | Memory feasible | Capacity useful | Top-B | Evaluated | Accepted | Rollback |
|---|---|---|---|---|---|---|---|
| qwen | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| whisper-small | 18 | 15 | 0 | 4 | 3 | 0 | 3 |
| resnet50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Memory-feasible x=2 只表示 M_safe(x2)<=B。Capacity-useful 是相同 λ、其余部署不变时 x2 的 canonical capacity gate 可行而配对 x1 不可行；两者不等同，runtime 是否改善另行判断。

每个实测 x=2 的 paired planning interval、frozen g(2)、代表测量任务、task-transfer 限制以及实际 throughput/queue 见 replica_empirical_consistency_audit.json。未饱和的 offered-load throughput 比值不能作为 capacity gain 估计；不假设线性翻倍。

主表 bottleneck_models 是冻结 canonical D/g 的规划压力；实测排队最长的 worker 与加权延迟贡献最大的 task 另见 runtime_bottlenecks.csv / runtime_bottleneck_audit.json。这是描述性运行压力，不用来修改路由或选择。规划可行不等于实际 steady。

## Loose-resource and Intensity Evidence

```json
{
  "cells": [
    {
      "cell_id": "C10",
      "outcome": "TIE",
      "forward_reason": "p95_worse",
      "reverse_reason": "latency_gain_below_min_relative_improvement",
      "mean_difference_ms": 0.03492277433004887,
      "p95_difference_ms": 0.03538680551882578,
      "safe_memory_difference_mib": 0.0,
      "deployment_L1_distance": 0,
      "v2_feasibility_status": "COMPLETE",
      "RC_SG_feasibility_status": "COMPLETE",
      "latency_imputed": false,
      "close_under_frozen_rule": true
    },
    {
      "cell_id": "C11",
      "outcome": "WIN",
      "forward_reason": "mean_improved",
      "reverse_reason": "mean_worse",
      "mean_difference_ms": -3.862172615297929,
      "p95_difference_ms": 3.700790155973337,
      "safe_memory_difference_mib": 6893.959646311789,
      "deployment_L1_distance": 2,
      "v2_feasibility_status": "COMPLETE",
      "RC_SG_feasibility_status": "COMPLETE",
      "latency_imputed": false,
      "close_under_frozen_rule": false
    },
    {
      "cell_id": "C12",
      "outcome": "TIE",
      "forward_reason": "same_nonsteady_stability",
      "reverse_reason": "same_nonsteady_stability",
      "mean_difference_ms": 4546.742533301909,
      "p95_difference_ms": 8318.459733689286,
      "safe_memory_difference_mib": -4079.055723695925,
      "deployment_L1_distance": 1,
      "v2_feasibility_status": "COMPLETE",
      "RC_SG_feasibility_status": "COMPLETE",
      "latency_imputed": false,
      "close_under_frozen_rule": false
    }
  ],
  "old_B2_runtime_used_for_matched_comparison": false
}
```

```json
[
  {
    "memory_budget_id": "M1",
    "complete": true,
    "distinct_deployment_count": 1,
    "composition_changed": false
  },
  {
    "memory_budget_id": "M4",
    "complete": true,
    "distinct_deployment_count": 3,
    "composition_changed": true
  },
  {
    "memory_budget_id": "M6",
    "complete": true,
    "distinct_deployment_count": 3,
    "composition_changed": true
  }
]
```

相同部署出现正式 WIN 的单元：['C01']。这些测量差异不能归因为不同部署选择。

## Frozen H1–H5

| Hypothesis | Status | Evidence |
|---|---|---|
| H1 | NOT_SUPPORTED | Frozen memory sweep observed 1 distinct final vectors and 1 distinct (N_G,N_S); monotone towards specialists=True |
| H2 | PARTIAL | 1/3 endpoint cells close under frozen same-stability max(1 ms, 5%) rule |
| H3 | PARTIAL | Old unconstrained SG violates the M1 memory budget; matched RC-SG/v2 evidence does not show a v2 advantage under the frozen objective |
| H4 | PARTIAL | Composition/replica vector changes in 2/3 fixed-budget intensity sweeps |
| H5 | INSUFFICIENT | No accepted x2 activation; conditional empirical-scaling hypothesis was not observed; no extra load added |

Logical evaluations 52 / 96（结构调整后 84）；physical states 40 / 96（调整后 84）；new timed 192 / 384（调整后启动/计时上限 324）。

全部原始结果、trace、独立确认、复用核查、预算检查和可直接绘图的 CSV 保留。没有新资源点、新强度、方法/路由修订、dynamic、第二硬件对比或额外 GPU 运行。
