# Module 2G：Comparable-Group Profiling

45 次计划运行已结束：15 个配置 × 3 次重复。原始结果全部保留，没有补跑或替换失败配置。
运行状态：{'completed': 36, 'failed': 9}；三态标签：{'steady': 30, 'recoverable_congested': 4, 'unstable': 11}。

**结论：INSUFFICIENT COMPARABLE GROUPS。本轮不能授权 global scorer 进入 guided search，也不能据此认定 predictor 已证伪。**
沿用 Module 2F 的最低覆盖要求：至少 3 个独立 workload 比较组，其中至少 2 组具有非平凡 Top-2 比较。

| Workload | 计划候选 | Strict 合法候选 | Extended 合法候选 |
|---|---:|---:|---:|
| W1_balanced | 5 | 5 | 5 |
| W2_generalist_heavy | 5 | 0 | 2 |
| W3_specialist_friendly | 5 | 1 | 1 |

Strict 要求配置的全部 3 次重复均 steady；Extended 允许 recoverable_congested，但任一 unstable 即排除整个配置。
每组 workload、measurement protocol、3 个 seed 与初始 arrival phases 完全匹配；不同 workload 之间没有排名。

| 分析 / workload | 候选 | Kendall τ | Spearman ρ | Top-1 | Top-2 | Top-3 | Regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict / W1_balanced | 5 | 0.800 | 0.900 | 1 | 1 | 1 | 0.00% |
| extended / W1_balanced | 5 | 0.800 | 0.900 | 1 | 1 | 1 | 0.00% |
| extended / W2_generalist_heavy | 2 | 1.000 | 1.000 | 1 | 1 (平凡) | None (平凡/不可用) | 0.00% |

Macro / regret（仅对合法组，不能外推到被排除配置）：
```json
{
  "strict": {
    "result": "INSUFFICIENT COMPARABLE GROUPS",
    "assessed": false,
    "reason": "Need >=3 independent workload groups and >=2 nontrivial Top-2 groups.",
    "macro": {
      "groups": 1,
      "kendall_tau_mean": 0.7999999999999999,
      "spearman_rho_mean": 0.8999999999999998,
      "top1_hit_mean": 1.0,
      "top2_recall_mean": 0.5,
      "top2_coverage_mean": 1.0,
      "top3_coverage_mean": 1.0,
      "top2_nontrivial_groups": 1,
      "top2_nontrivial_coverage": 1.0,
      "top3_nontrivial_groups": 1,
      "top3_nontrivial_coverage": 1.0,
      "regret": {
        "mean": 0.0,
        "median": 0.0,
        "p90": 0.0,
        "worst_case": 0.0
      }
    }
  },
  "extended": {
    "result": "INSUFFICIENT COMPARABLE GROUPS",
    "assessed": false,
    "reason": "Need >=3 independent workload groups and >=2 nontrivial Top-2 groups.",
    "macro": {
      "groups": 2,
      "kendall_tau_mean": 0.8999999999999999,
      "spearman_rho_mean": 0.9499999999999998,
      "top1_hit_mean": 1.0,
      "top2_recall_mean": 0.75,
      "top2_coverage_mean": 1.0,
      "top3_coverage_mean": 1.0,
      "top2_nontrivial_groups": 1,
      "top2_nontrivial_coverage": 1.0,
      "top3_nontrivial_groups": 1,
      "top3_nontrivial_coverage": 1.0,
      "regret": {
        "mean": 0.0,
        "median": 0.0,
        "p90": 0.0,
        "worst_case": 0.0
      }
    }
  }
}
```

## 审计与解释边界

- 每组仅考虑 MILP seed 加 5 个一跳 mutation：5 个可行候选全部进入预算，空 deployment 不可行。没有枚举全空间。
- Frozen policy 写了 farthest-point，但本轮全部可行候选均在预算内；实际按 configuration ID 排序，距离仅作审计，没有进行 farthest-point 淘汰。
- Canonical routing 为 min-max standalone demand LP；它保证 surrogate 可行，不保证真实稳定。
- 规划负载并不充分保守：Qwen-only 的 W1/W2/W3 u 分别约 0.312/0.543/0.331。尤其 W2 高于既有 Qwen backlog 区域；冻结说明中 W1 “below prior overload region” 不准确。不同 task mix 不能保证沿用单一边界，必须承认该设计限制，不能删除失败结果来修复。
- 5 个 deployment context 都在历史数据出现过；没有新增历史未见 context。但每个冻结 predictor 的训练集完整排除了自身待评估 deployment，因此预测属于 deployment-held-out。
- 预测冻结于本轮测量之前；本次只验证哈希并复现推理，没有重新 fit。参数固定的 C GBT 采用历史 v2 steady、无样本权重，不能声称与 2D 加权拟合的数值完全等价。
- E2E mean 按固定 workload arrival rates 加权，重复间等权并保存 sample std。真实 tail 是按 arrival rates 加权的 per-task P95，不是系统 P95。没有 P95 predictor，未生成预测 tail。
- 失败/不稳定主要证据见 regret_analysis.json：incomplete、drain timeout、worker/run error 或持续尾段 backlog 增长。后期较低负载也发生失败，不能仅归因于 Qwen u；顺序运行与环境漂移未随机化，因果解释受限。
- 有效组太少；一个完整组的 Top-K 命中不能替代跨三个 workload 的可靠性证据。

- 排序指标以事后稳定性筛选为条件，并不代表部署前能安全选点。W2 完整候选池中冻结 scorer 的 Top-1 是 Qwen+Whisper，但该配置有 1 次 unstable；排除它后得到的零 regret 不能证明完整候选池决策可靠。
- W1 的主要排序错误是 Qwen+ResNet 与 Qwen+DistilBERT 对调；Top-1 正确，但 Top-2 recall 为 0.5。完整 u/q/C、真实和预测排序保存在 comparison_groups.json 与 regret_analysis.json。

全部输出已保存。本阶段停止；未进入 Module 3、routing optimization 或新增 profiling。
