# Module 3C：Multi-workload Equal-budget Evaluation

Workloads、预算与执行顺序均在首次真实测量前冻结。所有主策略 predictor=None。Data Quality Audit execution-status/stability 语义统一用于本轮：completed genuine unstable 计入有效 runtime objective；外部或不确定中断只计物理尝试，不进入性能标签。

## 核心对照

| Workload | Algorithm | Seed | Final | Final stability | Seed mean E2E ms | Final mean E2E ms | Improvement | Valid eval | Physical attempts | LP solves |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| W1_balanced | B1_seed_1103 | qwen | distilbert+ paddleocr+ qwen+ whisper | steady | 286.120 | 66.620 | 76.72% | 6 | 18 | 10 |
| W1_balanced | B0 | qwen | qwen | steady | 281.055 | 290.106 | -3.22% | 1 | 0 | 1 |
| W1_balanced | B1_seed_2207 | qwen | qwen+ resnet50+ whisper | steady | 282.418 | 77.506 | 72.56% | 6 | 6 | 10 |
| W1_balanced | B3 | qwen | paddleocr+ qwen+ resnet50+ whisper | steady | 282.684 | 57.893 | 79.52% | 6 | 6 | 10 |
| W1_balanced | B2 | qwen | paddleocr+ qwen+ resnet50+ whisper | steady | 285.397 | 57.532 | 79.84% | 6 | 0 | 10 |
| W2_generalist_heavy | B1_seed_1103 | qwen | distilbert+ qwen+ resnet50+ whisper | steady | 278.022 | 56.170 | 79.80% | 6 | 18 | 10 |
| W2_generalist_heavy | B3 | qwen | distilbert+ qwen+ resnet50+ whisper | steady | 280.240 | 56.122 | 79.97% | 6 | 0 | 10 |
| W2_generalist_heavy | B0 | qwen | qwen | steady | 276.426 | 284.051 | -2.76% | 1 | 0 | 1 |
| W2_generalist_heavy | B2 | qwen | distilbert+ qwen+ resnet50+ whisper | steady | 282.335 | 56.077 | 80.14% | 6 | 0 | 10 |
| W2_generalist_heavy | B1_seed_2207 | qwen | qwen+ resnet50+ whisper | steady | 285.247 | 75.654 | 73.48% | 6 | 6 | 10 |
| W3_specialist_friendly | B0 | qwen | qwen | steady | 231.409 | 231.752 | -0.15% | 1 | 3 | 1 |
| W3_specialist_friendly | B2 | qwen | paddleocr+ qwen+ resnet50+ whisper | steady | 225.541 | 53.233 | 76.40% | 6 | 15 | 10 |
| W3_specialist_friendly | B1_seed_1103 | qwen | distilbert+ qwen+ resnet50+ whisper | steady | 230.515 | 54.874 | 76.19% | 6 | 6 | 10 |
| W3_specialist_friendly | B1_seed_2207 | qwen | qwen+ resnet50+ whisper | steady | 232.874 | 74.464 | 68.02% | 6 | 6 | 10 |
| W3_specialist_friendly | B3 | qwen | paddleocr+ qwen+ resnet50+ whisper | steady | 231.161 | 52.814 | 77.15% | 6 | 0 | 10 |

## 十二项结论

1. 三个 workload 的 seed 见表及逐运行 trace。
2. Proposed 的 final deployment 见表。
3. Proposed 对 seed 的正向改善出现在 3/3 个 workload；绝对延迟见表。
4. Proposed 的 final mean E2E 优于同 workload 两个真实 random trials 的 workload 数：3/3；稳定性与 best-seen 另见 JSON。
5. B2 对 LP 正 delta 但真实退化的回滚 0 次；B3 接受这类状态 0 次。
6. delta_z 与真实 ΔJ 的 Spearman 描述统计：{"W1_balanced": {"n": 20, "spearman": 0.784290527959965}, "W2_generalist_heavy": {"n": 20, "spearman": 0.8720282996993475}, "W3_specialist_friendly": {"n": 20, "spearman": 0.7163616936964159}}；不作因果解释。
7. 平均 real evaluations：6.00（B2，含 seed）。
8. 已测 unstable candidate 数：0。
9. confirmation 保持搜索改善方向的 B2 workload 数：3/3。
10. Module 3B 的 99.35% 为单次 sample-level 观察；逐 workload 的实际改善见表，不能视作稳定加速比。
11. Deployment enumeration calls = 0；real exhaustive optimum = unavailable。
12. 当前证据：**PARTIAL SUPPORT**。

随机 baseline 为每个 workload 两个固定种子的真实 runtime trial；没有把 offline replay 当成真实实验。每状态 3 次 search repetition 与独立 confirmation 的统计、P95、queue、completion、LP/物理预算详见 JSON。
