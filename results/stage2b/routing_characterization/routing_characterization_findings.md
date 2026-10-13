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
