# Module 3C：Multi-workload Equal-budget Evaluation

Data Quality Audit 语义：仅 `completed` run 有 stability_v2；completed 的 genuine unstable 保留在 runtime objective。外部/不确定/基础设施中断不计有效性能评估，但计入物理尝试。

Workloads 在首个 GPU run 前冻结：W1 原历史速率、W2 历史速率统一乘 0.70、W3 统一乘 0.80（依据纠错后的历史负载证据）；每次 measurement 210 s、warmup 10 s，predictor=None。`benchmark_workloads.json` SHA-256：`4b2cc94257870199d36fca03cea0b39028e05f20a5c37eabe55c5a7df9a6fbed`。

| Workload | B2 seed → final | Final stability | Seed E2E mean ms | Final E2E mean ms | Improvement | B1 two final means ms | B2 / B1 evals |
|---|---|---|---:|---:|---:|---|---|
| W1_balanced | qwen → paddleocr+qwen+resnet50+whisper | steady | 285.397 | 57.532 | 79.84% | 66.620, 77.506 | 6 / [6, 6] |
| W2_generalist_heavy | qwen → distilbert+qwen+resnet50+whisper | steady | 282.335 | 56.077 | 80.14% | 56.170, 75.654 | 6 / [6, 6] |
| W3_specialist_friendly | qwen → paddleocr+qwen+resnet50+whisper | steady | 225.541 | 53.233 | 76.40% | 54.874, 74.464 | 6 / [6, 6] |

## 全部策略指标

| Workload | Policy | Final | Stability / steady fraction | E2E mean / median / sample std (ms) | Task P95 (ms) | Improvement | Valid evals | Physical search / total | LP solves | Plan s | Visited | Accept | Rollback-labelled |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1_balanced | B1_seed_1103 | distilbert+paddleocr+qwen+whisper | steady / 1.00 | 66.620 / 66.355 / 6.388 | 102.739 | 76.72% | 6 | 18 / 22 | 10 | 4.59 | 6 | 3 | 2 |
| W1_balanced | B0 | qwen | steady / 1.00 | 278.767 / 263.602 / 39.018 | 705.376 | 0.00% | 1 | 0 / 4 | 1 | 0.23 | 1 | 0 | 0 |
| W1_balanced | B1_seed_2207 | qwen+resnet50+whisper | steady / 1.00 | 77.506 / 74.438 / 9.694 | 128.993 | 72.56% | 6 | 6 / 10 | 10 | 3.95 | 6 | 2 | 3 |
| W1_balanced | B3 | paddleocr+qwen+resnet50+whisper | steady / 1.00 | 57.893 / 57.851 / 3.673 | 92.641 | 79.52% | 6 | 6 / 10 | 10 | 4.79 | 6 | 3 | 0 |
| W1_balanced | B2 | paddleocr+qwen+resnet50+whisper | steady / 1.00 | 57.532 / 56.031 / 3.764 | 91.734 | 79.84% | 6 | 0 / 4 | 10 | 2.04 | 6 | 3 | 2 |
| W2_generalist_heavy | B1_seed_1103 | distilbert+qwen+resnet50+whisper | steady / 1.00 | 56.170 / 54.756 / 4.082 | 92.346 | 79.80% | 6 | 18 / 22 | 10 | 14.58 | 6 | 3 | 2 |
| W2_generalist_heavy | B3 | distilbert+qwen+resnet50+whisper | steady / 1.00 | 56.122 / 54.756 / 3.879 | 91.932 | 79.97% | 6 | 0 / 4 | 10 | 2.85 | 6 | 3 | 0 |
| W2_generalist_heavy | B0 | qwen | steady / 1.00 | 275.735 / 283.112 / 33.964 | 753.713 | 0.00% | 1 | 0 / 4 | 1 | 0.43 | 1 | 0 | 0 |
| W2_generalist_heavy | B2 | distilbert+qwen+resnet50+whisper | steady / 1.00 | 56.077 / 54.756 / 3.797 | 92.673 | 80.14% | 6 | 0 / 4 | 10 | 2.84 | 6 | 3 | 2 |
| W2_generalist_heavy | B1_seed_2207 | qwen+resnet50+whisper | steady / 1.00 | 75.654 / 74.407 / 4.921 | 126.286 | 73.48% | 6 | 6 / 10 | 10 | 8.15 | 6 | 2 | 3 |
| W3_specialist_friendly | B0 | qwen | steady / 1.00 | 236.590 / 246.132 / 19.162 | 644.932 | 0.00% | 1 | 3 / 7 | 1 | 3.44 | 1 | 0 | 0 |
| W3_specialist_friendly | B2 | paddleocr+qwen+resnet50+whisper | steady / 1.00 | 53.233 / 53.372 / 2.575 | 82.117 | 76.40% | 6 | 15 / 19 | 10 | 20.15 | 6 | 3 | 2 |
| W3_specialist_friendly | B1_seed_1103 | distilbert+qwen+resnet50+whisper | steady / 1.00 | 54.874 / 55.648 / 2.272 | 88.830 | 76.19% | 6 | 6 / 10 | 10 | 11.51 | 6 | 3 | 2 |
| W3_specialist_friendly | B1_seed_2207 | qwen+resnet50+whisper | steady / 1.00 | 74.464 / 71.392 / 7.580 | 122.994 | 68.02% | 6 | 6 / 10 | 10 | 12.37 | 6 | 2 | 3 |
| W3_specialist_friendly | B3 | paddleocr+qwen+resnet50+whisper | steady / 1.00 | 52.814 / 53.372 / 2.407 | 80.302 | 77.15% | 6 | 0 / 4 | 10 | 4.16 | 6 | 3 | 0 |

## 结论

1. 三种 workload 的 MILP seed 均为 `qwen`。
2. B2 的三个最终 deployment 见首表。
3. B2 依据冻结 RuntimeObjective 改善 seed：3/3；绝对延迟及各自 improvement 见首表。MILP seed 优化的是最小可行内存，不是 runtime 性能。
4. 与相同 seed、neighborhood、budget、accept/rollback 的两个真实 B1 trial 比，B2 在 3/3 个 workload 至少优于一个 trial 或样本效率中位数，严格目标比较胜出 4/6 个真实 random trials：W1 2/2、W2 1/2、W3 1/2。W2 有一个 trial 与 B2 相同；W3 有一个接近。
5. LP 正 delta 但真实退化的候选 0 次；B2 运行时回滚避免 0 次；B3 误接受 0 次。三个 workload 中 B2/B3 的 final 均相同，本轮未显示 runtime feedback 纠正 LP 错判。
6. delta_z 与真实性能 ΔJ 的 Spearman（policy transition，可能复用缓存）：{"W1_balanced": 0.784290527959965, "W2_generalist_heavy": 0.8720282996993475, "W3_specialist_friendly": 0.7163616936964159}；去重 transition 敏感性结果：{"W1_balanced": 0.6435682390352331, "W2_generalist_heavy": 0.7797730291639321, "W3_specialist_friendly": 0.6210110403690234}。均仅作描述，不作因果推断。
7. B2 平均使用 6.00 次真实有效 state evaluation（含 seed）；搜索物理尝试 84 次，confirmation 物理尝试 60 次。
8. 已测 unstable candidate 数：0；所有 completed unstable 均会保留。
9. 独立 confirmation 与搜索方向一致的 B2 workload 数：3/3；逐次 latency/stability、P95、queue 与 completion 见 per_workload_metrics.json。
10. Module 3B 的 99.35% 是单次样本观察；按本轮搜索加独立 confirmation 的延迟统计，达到或超过该幅度的 workload 为 0/3。绝对延迟见首表，不宣称稳定加速比。
11. Deployment enumeration calls = 0；real exhaustive optimum = unavailable。
12. 冻结门槛下当前证据：**PARTIAL SUPPORT**。

每个 workload 两个 B1 random seeds 均为真实 runtime trial，没有把 offline replay 当作 GPU 证据。表中 Rollback-labelled 包含优于 incumbent 却未成为同轮最优的候选；真正“LP 看好而 runtime 变差”的回滚单独见第 5 项。`search_efficiency.json` 给出 best-seen、首次 steady 的评价次数和耗时；共享缓存命中时耗时从该策略开始计，不能视为独立 GPU 测量时间。执行顺序、环境元数据及缓存审计见同目录。
