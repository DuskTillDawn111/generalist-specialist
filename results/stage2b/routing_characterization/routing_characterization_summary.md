# Stage 2B-1 Routing Characterization

已保存 15/15 轮。固定五模型 deployment；workload 总负载 5.8 req/s。
每轮 measurement=100 s，warmup=10 s，另有逐路径 priming；每轮重新启动全部 worker。
三组 seed=101/202/303；同一重复内各 share 采用相同 seed，运行顺序已打乱并保存在 plan.json。
表格为三个 run-level 指标的 mean ± sample std，P95/P99 不做请求池混合。

## System measurements

| specialist_share | reps | cohort completion rps | within-window rps | completion ratio | unstable runs |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.00 | 3 | 5.800 ± 0.000 | 5.763 ± 0.035 | 1.000 ± 0.000 | none |
| 0.25 | 3 | 5.800 ± 0.000 | 5.797 ± 0.006 | 1.000 ± 0.000 | none |
| 0.50 | 3 | 5.800 ± 0.000 | 5.797 ± 0.006 | 1.000 ± 0.000 | none |
| 0.75 | 3 | 5.800 ± 0.000 | 5.797 ± 0.006 | 1.000 ± 0.000 | none |
| 1.00 | 3 | 5.800 ± 0.000 | 5.797 ± 0.006 | 1.000 ± 0.000 | none |

## Routing matrix: specialist_share=0.00

| task | qwen | distilbert | resnet50 | paddleocr | whisper |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 2 | 0 | 0 | 0 | 0 |
| image | 1 | 0 | 0 | 0 | 0 |
| ocr | 1 | 0 | 0 | 0 | 0 |
| asr | 0.3 | 0 | 0 | 0 | 0 |
| visual_qa | 0.5 | 0 | 0 | 0 | 0 |
| lid | 1 | 0 | 0 | 0 | 0 |

单位 req/s；specialist_share 仅为生成矩阵的实验标签。

| task | E2E mean ms | E2E P50 ms | E2E P95 ms | E2E P99 ms | worker mean ms | schedule mean ms | runtime−worker mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 734.946 ± 88.765 | 568.143 ± 81.594 | 2056.679 ± 508.723 | 2426.720 ± 549.659 | 44.693 ± 0.622 | 0.146 ± 0.010 | 690.107 ± 89.190 |
| image | 803.457 ± 15.333 | 630.514 ± 143.023 | 2063.907 ± 433.582 | 2465.022 ± 544.589 | 141.751 ± 3.990 | 0.134 ± 0.014 | 661.573 ± 12.772 |
| ocr | 660.197 ± 148.021 | 507.203 ± 73.656 | 1964.874 ± 650.475 | 2270.423 ± 702.858 | 102.796 ± 3.606 | 0.162 ± 0.022 | 557.238 ± 147.922 |
| asr | 1132.642 ± 97.828 | 983.608 ± 18.257 | 2306.804 ± 426.378 | 2578.467 ± 576.551 | 794.881 ± 26.517 | 0.146 ± 0.008 | 337.614 ± 74.339 |
| visual_qa | 869.993 ± 37.812 | 650.024 ± 82.351 | 2113.959 ± 427.786 | 2444.276 ± 540.247 | 144.436 ± 7.139 | 0.142 ± 0.010 | 725.415 ± 31.225 |
| lid | 759.039 ± 106.781 | 554.588 ± 164.063 | 2017.003 ± 514.635 | 2406.065 ± 546.685 | 264.539 ± 22.404 | 0.159 ± 0.016 | 494.341 ± 91.608 |

## Routing matrix: specialist_share=0.25

| task | qwen | distilbert | resnet50 | paddleocr | whisper |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 1.5 | 0.5 | 0 | 0 | 0 |
| image | 0.75 | 0 | 0.25 | 0 | 0 |
| ocr | 0.75 | 0 | 0 | 0.25 | 0 |
| asr | 0.225 | 0 | 0 | 0 | 0.075 |
| visual_qa | 0.5 | 0 | 0 | 0 | 0 |
| lid | 0.75 | 0 | 0 | 0 | 0.25 |

单位 req/s；specialist_share 仅为生成矩阵的实验标签。

| task | E2E mean ms | E2E P50 ms | E2E P95 ms | E2E P99 ms | worker mean ms | schedule mean ms | runtime−worker mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 242.581 ± 31.337 | 147.207 ± 92.552 | 862.495 ± 54.879 | 1228.933 ± 149.073 | 36.351 ± 0.136 | 0.143 ± 0.009 | 206.086 ± 31.316 |
| image | 305.965 ± 80.816 | 278.976 ± 105.380 | 832.569 ± 203.977 | 1180.447 ± 194.904 | 109.179 ± 1.118 | 0.131 ± 0.011 | 196.655 ± 80.214 |
| ocr | 206.926 ± 40.366 | 97.048 ± 11.774 | 781.943 ± 120.630 | 1215.244 ± 337.555 | 83.049 ± 1.106 | 0.163 ± 0.016 | 123.714 ± 40.843 |
| asr | 682.267 ± 21.578 | 657.704 ± 59.437 | 1356.282 ± 219.114 | 1534.410 ± 223.889 | 609.451 ± 29.486 | 0.142 ± 0.007 | 72.673 ± 16.341 |
| visual_qa | 462.426 ± 94.542 | 388.267 ± 80.982 | 936.395 ± 131.603 | 1318.495 ± 88.719 | 145.170 ± 6.583 | 0.135 ± 0.010 | 317.122 ± 100.582 |
| lid | 290.754 ± 37.181 | 242.903 ± 13.619 | 829.493 ± 193.359 | 1156.226 ± 230.912 | 194.155 ± 7.448 | 0.155 ± 0.017 | 96.444 ± 30.884 |

## Routing matrix: specialist_share=0.50

| task | qwen | distilbert | resnet50 | paddleocr | whisper |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 1 | 1 | 0 | 0 | 0 |
| image | 0.5 | 0 | 0.5 | 0 | 0 |
| ocr | 0.5 | 0 | 0 | 0.5 | 0 |
| asr | 0.15 | 0 | 0 | 0 | 0.15 |
| visual_qa | 0.5 | 0 | 0 | 0 | 0 |
| lid | 0.5 | 0 | 0 | 0 | 0.5 |

单位 req/s；specialist_share 仅为生成矩阵的实验标签。

| task | E2E mean ms | E2E P50 ms | E2E P95 ms | E2E P99 ms | worker mean ms | schedule mean ms | runtime−worker mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 103.745 ± 34.428 | 35.494 ± 6.117 | 454.934 ± 140.832 | 917.116 ± 124.476 | 27.741 ± 0.476 | 0.156 ± 0.013 | 75.848 ± 34.735 |
| image | 189.249 ± 53.602 | 126.048 ± 57.786 | 607.011 ± 21.438 | 1121.323 ± 228.433 | 80.892 ± 2.616 | 0.141 ± 0.010 | 108.216 ± 52.266 |
| ocr | 110.064 ± 11.241 | 54.524 ± 4.039 | 486.449 ± 31.108 | 936.068 ± 90.003 | 61.189 ± 7.124 | 0.180 ± 0.013 | 48.695 ± 9.014 |
| asr | 530.471 ± 57.572 | 451.532 ± 72.633 | 1188.312 ± 128.691 | 1342.622 ± 199.068 | 511.012 ± 41.004 | 0.148 ± 0.002 | 19.311 ± 17.569 |
| visual_qa | 290.841 ± 142.696 | 232.500 ± 135.541 | 693.649 ± 161.288 | 1028.388 ± 287.710 | 148.065 ± 10.057 | 0.146 ± 0.018 | 142.630 ± 133.917 |
| lid | 191.288 ± 10.318 | 168.723 ± 38.099 | 535.775 ± 71.882 | 1042.254 ± 284.994 | 153.617 ± 3.653 | 0.168 ± 0.018 | 37.503 ± 11.387 |

## Routing matrix: specialist_share=0.75

| task | qwen | distilbert | resnet50 | paddleocr | whisper |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 0.5 | 1.5 | 0 | 0 | 0 |
| image | 0.25 | 0 | 0.75 | 0 | 0 |
| ocr | 0.25 | 0 | 0 | 0.75 | 0 |
| asr | 0.075 | 0 | 0 | 0 | 0.225 |
| visual_qa | 0.5 | 0 | 0 | 0 | 0 |
| lid | 0.25 | 0 | 0 | 0 | 0.75 |

单位 req/s；specialist_share 仅为生成矩阵的实验标签。

| task | E2E mean ms | E2E P50 ms | E2E P95 ms | E2E P99 ms | worker mean ms | schedule mean ms | runtime−worker mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 43.480 ± 10.259 | 10.613 ± 0.465 | 179.169 ± 124.049 | 398.738 ± 144.226 | 19.262 ± 0.573 | 0.159 ± 0.011 | 24.060 ± 10.099 |
| image | 89.151 ± 33.459 | 20.232 ± 1.240 | 325.156 ± 137.444 | 418.030 ± 221.395 | 50.125 ± 0.220 | 0.153 ± 0.018 | 38.873 ± 33.307 |
| ocr | 56.757 ± 9.631 | 18.363 ± 0.275 | 157.010 ± 6.664 | 639.328 ± 334.942 | 42.507 ± 3.744 | 0.175 ± 0.018 | 14.075 ± 13.209 |
| asr | 407.413 ± 30.027 | 266.016 ± 58.419 | 1185.256 ± 481.527 | 1592.072 ± 676.337 | 403.263 ± 28.481 | 0.152 ± 0.006 | 3.997 ± 5.551 |
| visual_qa | 274.601 ± 80.560 | 260.977 ± 82.518 | 518.583 ± 161.174 | 971.719 ± 573.698 | 147.308 ± 9.168 | 0.158 ± 0.021 | 127.135 ± 89.728 |
| lid | 109.513 ± 14.556 | 40.378 ± 2.399 | 303.450 ± 33.993 | 613.668 ± 320.328 | 94.345 ± 2.750 | 0.167 ± 0.020 | 15.001 ± 11.834 |

## Routing matrix: specialist_share=1.00

| task | qwen | distilbert | resnet50 | paddleocr | whisper |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 0 | 2 | 0 | 0 | 0 |
| image | 0 | 0 | 1 | 0 | 0 |
| ocr | 0 | 0 | 0 | 1 | 0 |
| asr | 0 | 0 | 0 | 0 | 0.3 |
| visual_qa | 0.5 | 0 | 0 | 0 | 0 |
| lid | 0 | 0 | 0 | 0 | 1 |

单位 req/s；specialist_share 仅为生成矩阵的实验标签。

| task | E2E mean ms | E2E P50 ms | E2E P95 ms | E2E P99 ms | worker mean ms | schedule mean ms | runtime−worker mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 10.412 ± 0.426 | 9.814 ± 0.438 | 16.435 ± 0.895 | 19.071 ± 0.613 | 9.796 ± 0.412 | 0.156 ± 0.013 | 0.459 ± 0.015 |
| image | 20.397 ± 0.348 | 21.069 ± 0.313 | 24.216 ± 0.798 | 27.102 ± 2.265 | 19.786 ± 0.370 | 0.143 ± 0.010 | 0.468 ± 0.023 |
| ocr | 17.731 ± 0.711 | 17.322 ± 0.665 | 26.044 ± 0.257 | 44.360 ± 1.968 | 17.102 ± 0.681 | 0.165 ± 0.021 | 0.464 ± 0.018 |
| asr | 220.711 ± 6.090 | 205.769 ± 10.073 | 447.128 ± 88.708 | 570.432 ± 238.541 | 219.906 ± 6.024 | 0.158 ± 0.010 | 0.647 ± 0.096 |
| visual_qa | 155.113 ± 9.780 | 143.757 ± 13.639 | 307.796 ± 37.991 | 354.808 ± 30.804 | 154.407 ± 9.771 | 0.150 ± 0.004 | 0.556 ± 0.022 |
| lid | 48.625 ± 8.141 | 37.828 ± 2.685 | 98.919 ± 93.748 | 242.420 ± 116.941 | 36.908 ± 2.054 | 0.162 ± 0.009 | 11.555 ± 7.358 |

## Scope and interpretation

- 全部 raw result 保存在 raw/，独立 configuration 在 configs/，启动/执行日志在 logs/。
- summary JSON 保留每个指标 mean/std/min/max/values，以及每轮验证、实际 offered/routing counts。
- 完成吞吐按 Stage 2A 的 cohort completions / 100 s；另展示窗口内完成吞吐，避免 drain 掩盖积压。
- worker latency 是 handler 墙钟时间；runtime−worker 包含 worker queue、传输和返回开销，不等于纯 queue time。
- ASR 每轮只有 30 个样本，P95/P99 对少数请求敏感；三轮 std 是重复间离散程度，不是置信区间。
- 出现不完整点时保留并标记 overloaded_or_unstable，不自动归因、删除或用成功重跑替换。
- 本实验为固定 deployment/workload 的描述性测量，不给出最优 routing 或性能预测。

<!-- ROUTING_FINDINGS -->

---

# Routing characterization: research observations

以下均为当前 x_full、固定 5.8 req/s workload、periodic arrivals 和 deterministic routing 下的观测。
三次重复间 std 描述离散程度，不是置信区间；不作统计显著性或系统最优性声明。

## 1. 仅改变 routing 是否改变 latency / throughput？

各 routing 的 cohort completion throughput 均值范围为 5.8000–5.8000 req/s；窗口内完成吞吐均值范围为 5.7633–5.7967 req/s。
当请求全部在 drain 期限前完成时，固定 offered load 会使 cohort throughput 相同，不能据此推断各 routing 的最大承载能力相同。

| Task | Generalist-first E2E mean ± std ms | Specialist-first E2E mean ± std ms | Endpoint reduction | 3 paired seeds with lower E2E |
| --- | ---: | ---: | ---: | ---: |
| text | 734.95 ± 88.77 | 10.41 ± 0.43 | 98.6% | 3/3 |
| image | 803.46 ± 15.33 | 20.40 ± 0.35 | 97.5% | 3/3 |
| ocr | 660.20 ± 148.02 | 17.73 ± 0.71 | 97.3% | 3/3 |
| asr | 1132.64 ± 97.83 | 220.71 ± 6.09 | 80.5% | 3/3 |
| visual_qa | 869.99 ± 37.81 | 155.11 ± 9.78 | 82.2% | 3/3 |
| lid | 759.04 ± 106.78 | 48.63 ± 8.14 | 93.6% | 3/3 |

## 2. 各 task 是否一样敏感？

| Task | Across-routing max/min E2E mean factor | Generalist-first P95 mean ± std ms | Specialist-first P95 mean ± std ms |
| --- | ---: | ---: | ---: |
| text | 70.59× | 2056.68 ± 508.72 | 16.44 ± 0.89 |
| image | 39.39× | 2063.91 ± 433.58 | 24.22 ± 0.80 |
| ocr | 37.23× | 1964.87 ± 650.48 | 26.04 ± 0.26 |
| asr | 5.13× | 2306.80 ± 426.38 | 447.13 ± 88.71 |
| visual_qa | 5.61× | 2113.96 ± 427.79 | 307.80 ± 37.99 |
| lid | 15.61× | 2017.00 ± 514.63 | 98.92 ± 93.75 |

## 3. Specialist-first 是否一定更好？

上述表格仅比较本次测量的两个端点，并保留三个 paired-seed 的方向。即使 specialist-first 在某指标上为本实验 measured lowest，也不能称为全系统最优 routing。吞吐、均值、tail 也不能自动视为同一个目标。

## 4. 哪个 latency 分量变化最大？

下面是 specialist-first − generalist-first 的均值差；负值表示下降。runtime−worker 包含 pipe/worker queue、传输与返回开销，原始 worker_queue_wait_ms 可交叉核对。

| Task | E2E Δ ms | Worker Δ ms | Schedule Δ ms | Runtime−worker Δ ms | Largest absolute component change |
| --- | ---: | ---: | ---: | ---: | --- |
| text | -724.53 | -34.90 | 0.010 | -689.65 | runtime−worker |
| image | -783.06 | -121.97 | 0.009 | -661.10 | runtime−worker |
| ocr | -642.47 | -85.69 | 0.003 | -556.77 | runtime−worker |
| asr | -911.93 | -574.98 | 0.012 | -336.97 | worker |
| visual_qa | -714.88 | 9.97 | 0.008 | -724.86 | runtime−worker |
| lid | -710.41 | -227.63 | 0.003 | -482.79 | runtime−worker |

## 5. 非单调、tail 与跨任务影响

对相邻 specialist_share 的实测均值作直接比较，以下条目出现了上升（不拟合函数）：

- 本次各相邻点均未观察到 E2E mean 上升；这不证明任意 workload 下单调。

三次平均可能掩盖 seed 间不同方向。VQA 的 share 0.50 → 0.75 配对结果：

| Seed | E2E mean at 0.50 (ms) | E2E mean at 0.75 (ms) | Delta (ms) |
| --- | ---: | ---: | ---: |
| 101 | 221.07 | 328.08 | +107.01 |
| 202 | 196.45 | 313.78 | +117.33 |
| 303 | 455.00 | 181.95 | -273.05 |

变化幅度并不均匀：不能把这五个测量点解释为恒定比例的 latency 改善。generalist-first 各任务 P95 约 2 s，明显高于均值；specialist-first 的 LID P95 跨重复 std 较大，tail 仍依赖请求相位和样本。具体数值见上述表和完整汇总。

VQA 始终全部交给 Qwen，所以其曲线变化不能由“VQA 改用了 specialist”解释。确定性周期到达、路由节奏、分配给 Qwen 的 ASR 样本长度等都可能影响它遇到的排队。条件 service-path 样本数/均值及 Qwen handler 总时间另存 findings JSON。

原始时间线中的具体例子（VQA 提交时，其前序 Qwen handler 仍在执行）：

- share_0.75_rep_1 / measurement:113: 前序 asr handler=1832.5 ms；VQA queue wait=1705.6 ms，handler=72.6 ms，E2E=1778.5 ms。
- share_0.50_rep_1 / measurement:55: 前序 asr handler=889.2 ms；VQA queue wait=762.5 ms，handler=326.9 ms，E2E=1089.7 ms。
- share_0.75_rep_3 / measurement:29: 前序 asr handler=930.1 ms；VQA queue wait=369.4 ms，handler=93.8 ms，E2E=463.6 ms。

这些时间线支持同一 worker 内任务之间排队影响的解释，不足以识别 GPU 微观干扰机制。ASR 每轮 n=30、VQA n=50，P99 对极少数请求敏感；完整 P50/P95/P99、std/min/max 均保留，本次结果不外推到 Poisson arrivals、其他 workload 或其他 deployment。

![Measured latency](routing_latency.png)

![Measured decomposition](routing_latency_components.png)
