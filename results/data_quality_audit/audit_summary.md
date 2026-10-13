# Data Quality Audit：External Interruption Reclassification

本次只读取已有原始测量与历史分析文件，新增纠错层和 CPU 重新分析输出。原始 raw JSON、request records、旧标签及旧报告的 547 个文件 SHA-256 在分析前后完全一致；原始文件修改数 **0**。新增 GPU profiling **0**，deployment enumeration 调用 **0**。

## 物理 run 总账

| 项目 | 数量 |
|---|---:|
| Total runs | 130 |
| completed | 129 |
| external_interruption | 1 |
| harness_failure | 0 |
| model_failure | 0 |
| uncertain_interruption | 0 |
| Completed: steady | 100 |
| Completed: recoverable_congested | 12 |
| Completed: unstable | 17 |

旧 unstable 改为 external interruption：**1** 次，只有 Module 3B 已归档的 host power-off；旧 unstable 改为 uncertain：0 次。External interruption 的 corrected stability 为 null，所有 performance 有效性均为 false，仅 physical attempt accounting 为 true。

2G 的 45 次都有完整计时 measurement window、arrival loop 和 drain 记录，也都有 harness 结果页脚。旧 11 次 unstable 中，9 次在完整 drain 后仍有 incomplete requests，2 次记录了持续尾段 backlog 增长。raw status 为 `failed` 的 9 次表示服务未能在 drain 截止前完成请求；它们并非 host power-off。一个运行的 shutdown broken pipe 出现在完整测量和 drain 之后，保留其真实 backlog/incomplete 证据。没有依据将任何 2G 运行改为 external。
Module 2B/2C profiling 的 3 次 raw `failed` 同样有完整测量与 drain，未发现外部中断。Module 3B 的 15 次有效重复仍为 completed；归档的 1 次 adapter failure envelope 无 harness 原始结果，与既有关机纠正记录一致。

## Module 2E stability 纠正统计

69 次 completed：**57 steady / 6 recoverable / 6 genuine unstable / 0 external**。本次仍等于旧统计，原因是 2E 的全部 69 次都已完成测量。

## Module 2D LODO（固定原模型配置与 random seed）

| v2 subset | path rows | C GBT macro MAE (ms) | C GBT macro sMAPE | GBT gate |
|---|---:|---:|---:|---|
| strict | 391 | 56.964 | 26.718% | PASS |
| extended | 427 | 465.814 | 33.315% | not PASS |

原门槛同时要求 strict 和 extended、Ridge 和 GBT 通过；纠正后整体 **PARTIAL**。LODO 按原配置在 CPU 重新拟合每折估计器以评估纠正样本，没有调整超参数，也没有导出新的线上 predictor。

## Module 2E bias-only calibration

| K | Strict global MAE | Strict bias-only MAE | Extended global MAE | Extended bias-only MAE |
|---:|---:|---:|---:|---:|
| 0 | 57.763 | 57.763 | 828.058 | 828.058 |
| 1 | 57.763 | 71.197 | 828.058 | 836.046 |
| 2 | 58.554 | 75.662 | 1213.996 | 1221.123 |
| 3 | 77.561 | 98.178 | 77.561 | 98.178 |

固定 K=0/1/2/3 和原 selection/bias-only 策略重算，gate **FAIL**（旧结果 FAIL）。这些是匹配测试子集的 macro MAE；K 间测试集可能不同，不应把行间变化解释为同一样本上的单调收益。

## Module 2F / 2G ranking

Module 2F 使用 corrected strict LODO OOF 预测及 corrected completed labels 重新建组：**INSUFFICIENT COMPARABLE GROUPS**；strict/extended cross-deployment rankable groups 均为 0。

| Workload | Strict 合法配置 | Extended 合法配置 | Extended Kendall τ | Extended Spearman ρ | Top-1 | Top-2 coverage | Top-3 coverage | Regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1_balanced | 5 | 5 | 0.800 | 0.900 | 1.000 | 1.000 | 1.000 | 0.000 |
| W2_generalist_heavy | 0 | 2 | 1.000 | 1.000 | 1.000 | 1.000 | N/A | 0.000 |
| W3_specialist_friendly | 1 | 1 | N/A | N/A | N/A | N/A | N/A | N/A |

2G strict 可排序 workload 组数 1；extended 为 2，低于预先固定的 3 组、其中至少 2 组非平凡 Top-2 要求。两视图 gate 均为 **INSUFFICIENT COMPARABLE GROUPS**，与旧结论相同。相关性/Top-K/regret 只描述已有合法组，不可替代缺失的跨 workload 覆盖。

## 决策

**Case A：Module 2G corrected result remains INSUFFICIENT.** Module 2 继续为 optional scorer；后续 Module 3 主线保持 `predictor=None`。本次停在 data quality audit，不进入 Module 3C。

执行状态与 serving stability 独立；只对 completed run 计算 steady / recoverable / unstable。所有纠正记录保留绝对源路径和 SHA-256；原始文件未覆盖。
