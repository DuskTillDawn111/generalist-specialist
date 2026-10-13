# Module 7A-2 Standalone Service-Demand Completion

完整性：25/25 verified task-model service paths，新增 14 条，已有 11 条原值保留，pending=0。

## Protocol and provenance

直接运行未修改的 `joint_deploy/run_active_single_profile.py` main。只扩展运行时 worker 注册，保留原始选择、warmup、submit/collect、timer、raw format 和 summary 实现；外层增加日志、GPU 独占核验和 unload 等待。Module 7B 算法源文件没有修改。

每条路径三轮独立 fresh worker；每轮先对第一个 target warmup 一次，再 enqueue 同一 task 的前 30 个冻结 mixed-pool rows。warmup 排除，31 次真实调用/轮；42 轮产生 1260 个 measured requests 和 42 个 warmups。每次只有一个 model type、一个 worker、一个 replica。

s0 的单位是 ms/request；统计量为 handler 内 `worker_latency_ms` 的算术均值，跨三轮用 `sum(mean*count)/sum(count)` 聚合。不使用 median、IPC queue wait 或 submit-to-completion E2E；包括 frozen handler 内的数据解码、预处理、CUDA 传输、推理、同步和输出处理。

输入逐项核对历史前 30 个请求的 sample payload/pool IDs/arrival index。ASR 使用原 service pool 的不同长度音频，不替换成最小加载/quality 的 4–6 秒子集。

沿用 frozen v2 native adapters：相同任务 dataset/label semantics、generation limits 和 output settings。Generalists 读取冻结 task prompts；模型各自 tokenizer/chat template、官方 weight transform 保持原样；PaddleOCR-VL 保持已冻结 element-level `OCR:` prompt 和 32-token recognized-text 输出，不运行 document parsing。

冻结环境的 Python/package/executable hash 前后完全一致；未升级任何依赖。checkpoint、tokenizer、custom model code、adapter、prompt 和 historical inputs 的 hash 保持不变。source revision 和每个模型 weight/file hash 记录于 pending manifest、before/source hashes。

NVML 记录证明每轮 GPU 上只有该 worker 的 PID，shutdown 后显存释放再启动下一轮。raw summary 和 handler ns timers 可独立复算；任何 load/inference/isolation/identity/timer failure 均不能填零、无穷或 synthetic fallback。冻结 capability matrix 未更改。

## New real standalone demands

| Model | Task | Repetition means (ms) | Aggregate (ms) | Sample std of 90 requests (ms) | Count |
|---|---|---|---:|---:|---:|
| convnext_tiny | image | 5.420581, 7.537057, 4.585570 | 5.847736 | 1.527399 | 90 |
| distil_whisper | asr | 145.126564, 144.731756, 148.812679 | 146.223666 | 34.575424 | 90 |
| minicpm | asr | 780.586542, 775.299321, 775.891085 | 777.258982 | 452.815703 | 90 |
| minicpm | image | 221.902170, 221.251262, 221.960278 | 221.704570 | 118.625816 | 90 |
| minicpm | ocr | 160.193840, 159.796887, 161.123728 | 160.371485 | 52.400486 | 90 |
| minicpm | text | 59.117668, 58.887415, 59.101225 | 59.035436 | 3.438113 | 90 |
| paddleocr_vl | ocr | 84.486200, 84.958716, 87.152304 | 85.532406 | 31.030277 | 90 |
| phi4mm | asr | 1107.548889, 1151.802625, 1129.415231 | 1129.588915 | 649.281100 | 90 |
| phi4mm | image | 232.666596, 231.512962, 226.165916 | 230.115158 | 39.406863 | 90 |
| phi4mm | ocr | 204.151278, 204.364925, 205.615802 | 204.710669 | 51.386496 | 90 |
| phi4mm | text | 67.199666, 68.161396, 69.006908 | 68.122657 | 1.129246 | 90 |
| roberta_sst2 | text | 5.630630, 5.465311, 5.423300 | 5.506414 | 8.560355 | 90 |
| whisper_turbo | asr | 169.346700, 170.023910, 172.196265 | 170.522292 | 49.061079 | 90 |
| whisper_turbo | lid | 82.955877, 83.592454, 82.987366 | 83.178565 | 0.977632 | 90 |

sample std 是所有 measured request 的样本标准差（ddof=1），包括不同输入样本的变化。每轮 raw latencies、每轮 sample std、三轮 mean 的 sample std 均另存于 results JSON，不把 request variability 当作纯 repetition uncertainty。

## Restored W1/W2/W3 integer planning

只有 completeness=100%、still_pending_paths=0 时才执行以下 dry-runs。输入为冻结 v2 measured idle VRAM + 新冻结真实 s0；不重测旧 service paths，不运行 runtime evaluator，不启动候选 replica workers。

| Workload | Integer seed (active entries) | Memory MiB | N_G | N_S | Active G/S types | z | Candidates / feasible |
|---|---|---:|---:|---:|---|---:|---|
| W1_balanced | qwen=1 | 17978 | 1 | 0 | 1/0 | 0.31239256 | 13/12 |
| W2_generalist_heavy | qwen=1 | 17978 | 1 | 0 | 1/0 | 0.38008241 | 13/12 |
| W3_specialist_friendly | qwen=1 | 17978 | 1 | 0 | 1/0 | 0.26460440 | 13/12 |

### W1_balanced

Bottleneck families: qwen.
Top-B: qwen 1→2, memory cost=17978 MiB, delta_z=0.15619628; whisper-small 0→1, memory cost=1616 MiB, delta_z=0.15407554.
Scale-down: qwen 1→0: REJECT_INFEASIBLE (coverage), saving=17978 MiB.

Pressure task **asr** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.15619628 | True | 1 |
| generalist | minicpm 0→1 | 18392 | 0.13952315 | True | 4 |
| generalist | phi4mm 0→1 | 11772 | 0.11879949 | True | 5 |
| specialist | distil_whisper 0→1 | 3672 | 0.09811177 | True | 6 |
| specialist | whisper-small 0→1 | 1616 | 0.15407554 | True | 2 |
| specialist | whisper_turbo 0→1 | 3882 | 0.15407554 | True | 3 |

Pressure task **lid** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.15619628 | True | 1 |
| specialist | whisper-small 0→1 | 1616 | 0.15407554 | True | 2 |
| specialist | whisper_turbo 0→1 | 3882 | 0.15407554 | True | 3 |

Pressure task **image** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.15619628 | True | 1 |
| generalist | minicpm 0→1 | 18392 | 0.13952315 | True | 4 |
| generalist | phi4mm 0→1 | 11772 | 0.11879949 | True | 5 |
| specialist | convnext_tiny 0→1 | 610 | 0.05584528 | True | 7 |
| specialist | resnet50 0→1 | 604 | 0.05584528 | True | 8 |

Pressure task **ocr** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.15619628 | True | 1 |
| generalist | minicpm 0→1 | 18392 | 0.13952315 | True | 4 |
| generalist | phi4mm 0→1 | 11772 | 0.11879949 | True | 5 |
| specialist | paddleocr 0→1 | 568 | 0.03785928 | True | 9 |
| specialist | paddleocr_vl 0→1 | 2308 | 0.03785928 | True | 10 |

Pressure task **text** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.15619628 | True | 1 |
| generalist | minicpm 0→1 | 18392 | 0.13952315 | True | 4 |
| generalist | phi4mm 0→1 | 11772 | 0.11879949 | True | 5 |
| specialist | distilbert 0→1 | 750 | 0.03571097 | True | 11 |
| specialist | roberta_sst2 0→1 | 988 | 0.03571097 | True | 12 |

Pressure task **visual_qa** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.15619628 | True | 1 |

该 task 没有两类可同时比较的候选；保留能力结构约束。


### W2_generalist_heavy

Bottleneck families: qwen.
Top-B: qwen 1→2, memory cost=17978 MiB, delta_z=0.19004120; whisper-small 0→1, memory cost=1616 MiB, delta_z=0.18301935.
Scale-down: qwen 1→0: REJECT_INFEASIBLE (coverage), saving=17978 MiB.

Pressure task **asr** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.19004120 | True | 1 |
| generalist | minicpm 0→1 | 18392 | 0.16934851 | True | 4 |
| generalist | phi4mm 0→1 | 11772 | 0.14480670 | True | 5 |
| specialist | distil_whisper 0→1 | 3672 | 0.11446373 | True | 6 |
| specialist | whisper-small 0→1 | 1616 | 0.18301935 | True | 2 |
| specialist | whisper_turbo 0→1 | 3882 | 0.18301935 | True | 3 |

Pressure task **lid** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.19004120 | True | 1 |
| specialist | whisper-small 0→1 | 1616 | 0.18301935 | True | 2 |
| specialist | whisper_turbo 0→1 | 3882 | 0.18301935 | True | 3 |

Pressure task **image** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.19004120 | True | 1 |
| generalist | minicpm 0→1 | 18392 | 0.16934851 | True | 4 |
| generalist | phi4mm 0→1 | 11772 | 0.14480670 | True | 5 |
| specialist | convnext_tiny 0→1 | 610 | 0.06841046 | True | 7 |
| specialist | resnet50 0→1 | 604 | 0.06841046 | True | 8 |

Pressure task **text** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.19004120 | True | 1 |
| generalist | minicpm 0→1 | 18392 | 0.16934851 | True | 4 |
| generalist | phi4mm 0→1 | 11772 | 0.14480670 | True | 5 |
| specialist | distilbert 0→1 | 750 | 0.04687065 | True | 9 |
| specialist | roberta_sst2 0→1 | 988 | 0.04687065 | True | 10 |

Pressure task **ocr** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.19004120 | True | 1 |
| generalist | minicpm 0→1 | 18392 | 0.16934851 | True | 4 |
| generalist | phi4mm 0→1 | 11772 | 0.14480670 | True | 5 |
| specialist | paddleocr 0→1 | 568 | 0.04637762 | True | 11 |
| specialist | paddleocr_vl 0→1 | 2308 | 0.04637762 | True | 12 |

Pressure task **visual_qa** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.19004120 | True | 1 |

该 task 没有两类可同时比较的候选；保留能力结构约束。


### W3_specialist_friendly

Bottleneck families: qwen.
Top-B: qwen 1→2, memory cost=17978 MiB, delta_z=0.13230220; minicpm 0→1, memory cost=18392 MiB, delta_z=0.11802752.
Scale-down: qwen 1→0: REJECT_INFEASIBLE (coverage), saving=17978 MiB.

Pressure task **asr** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.13230220 | True | 1 |
| generalist | minicpm 0→1 | 18392 | 0.11802752 | True | 2 |
| generalist | phi4mm 0→1 | 11772 | 0.10127536 | True | 5 |
| specialist | distil_whisper 0→1 | 3672 | 0.07848941 | True | 6 |
| specialist | whisper-small 0→1 | 1616 | 0.11206768 | True | 3 |
| specialist | whisper_turbo 0→1 | 3882 | 0.11206768 | True | 4 |

Pressure task **image** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.13230220 | True | 1 |
| generalist | minicpm 0→1 | 18392 | 0.11802752 | True | 2 |
| generalist | phi4mm 0→1 | 11772 | 0.10127536 | True | 5 |
| specialist | convnext_tiny 0→1 | 610 | 0.05584528 | True | 7 |
| specialist | resnet50 0→1 | 604 | 0.05584528 | True | 8 |

Pressure task **ocr** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.13230220 | True | 1 |
| generalist | minicpm 0→1 | 18392 | 0.11802752 | True | 2 |
| generalist | phi4mm 0→1 | 11772 | 0.10127536 | True | 5 |
| specialist | paddleocr 0→1 | 568 | 0.03785928 | True | 9 |
| specialist | paddleocr_vl 0→1 | 2308 | 0.03785928 | True | 10 |

Pressure task **text** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.13230220 | True | 1 |
| generalist | minicpm 0→1 | 18392 | 0.11802752 | True | 2 |
| generalist | phi4mm 0→1 | 11772 | 0.10127536 | True | 5 |
| specialist | distilbert 0→1 | 750 | 0.03571097 | True | 11 |
| specialist | roberta_sst2 0→1 | 988 | 0.03571097 | True | 12 |

Pressure task **lid** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.13230220 | True | 1 |
| specialist | whisper-small 0→1 | 1616 | 0.11206768 | True | 3 |
| specialist | whisper_turbo 0→1 | 3882 | 0.11206768 | True | 4 |

Pressure task **visual_qa** — Generalist/Specialist +1 comparison:

| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |
|---|---|---:|---:|---|---:|
| generalist | qwen 1→2 | 17978 | 0.13230220 | True | 1 |

该 task 没有两类可同时比较的候选；保留能力结构约束。


三个 workload 的 visual_qa rate 均为正，唯一 verified capable model 是 Qwen，因此 **mandatory model induced by capability coverage: y_qwen=1**。不能把这一 presence 写成基于性能的主动选择；当前 Generalist composition 的自由度受 VQA capability asymmetry 限制。

完整 integer vectors（含零项）、canonical family-level routing、per-replica pressure、task/model total demand、所有生成的 x±1 candidates、拒绝原因、delta_z、ranking、Top-B 和 N_G/N_S 均保存在三个 dry-run JSON。

Memory 仍为 **planning memory surrogate**（additive measured idle VRAM）；capacity 仍为 **analytical planning surrogate, not a claim of linear real-GPU scaling**。这些结果没有建立真实多副本性能或最优 latency allocation 的结论。

## CPU regression and stopping

118 项当前 CPU regressions/demand-gate tests 全部通过。四个会重新运行 A–E 或 20/30 synthetic scalability campaign 的 test wrappers 按用户要求未重跑；其既有通过记录保留，相关 55 个源文件 hash 未变。原 113 项测试的验证由本次重新运行的 109 项与这 4 项不变性核验共同保留。

未运行 final deployment benchmark、joint/co-location measurement、dynamic multi-replica search、forecasting、predictor retraining 或 Module 7C。真实 profiling 仅用于补齐 14 条 single-worker service paths；real-pool integer dry-run 的 runtime/GPU measurement 次数为 0。到此停止。

## Artifact entry points

- `pending_service_demand_paths.json`: 精确 14 条 verified+missing paths 与环境/adapter/model provenance。
- `historical_protocol_audit.json`: 原协议、样本选择、11 条已有 s0 与三轮对照。
- `raw_measurements/r1,r2,r3/`: 独立 raw JSONL、Stage-1 summaries、worker/profile logs、isolation/requests/timer audit。
- `standalone_service_demand_results.json`: 每轮 values/mean/std、90 次聚合与 failure state。
- `updated_service_demand_catalog.json`: 新冻结 planning input；原 catalog 没有覆盖。
- `measurement_validation.json`, `cpu_regression_validation.json`, `source_hashes.json`: gates 和不变性证据。
- `integer_dry_runs/w1,w2,w3_integer_dry_run.json`: 与原 7B 输出目录同步的已解锁 records。
