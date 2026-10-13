# Performance Model Foundation / Module 2

真实结果：18 runs，8966 measurement requests，148 task-model rows；3 个实际观测 deployment。
包含 Stage 2A 两个 smoke、Stage 2B-1 十五轮正式结果和一个 full-deployment smoke gate。未导入旧 closed-loop pair experiments 作为 E2E labels。
每条 row 保留完整 x/r/lambda、source result 绝对路径、哈希、run ID、seed 和协议。latency 按真实 task-model 请求重新聚合，不复制整个 task 均值。
mean/P95 单位 ms；throughput 为 cohort completions/window seconds，同时保留 within-window throughput、drain count 和 incomplete。
标签只基于成功请求；incomplete/error 与观测数单独保留。跨 run 的 P95 平均不是 pooled P95。

| Observed deployment | Count vector |
| --- | --- |
| observed 1 (audit label only) | {'distilbert': 1, 'paddleocr': 0, 'qwen': 1, 'resnet50': 0, 'whisper': 0} |
| observed 2 (audit label only) | {'distilbert': 0, 'paddleocr': 0, 'qwen': 1, 'resnet50': 0, 'whisper': 1} |
| observed 3 (audit label only) | {'distilbert': 1, 'paddleocr': 1, 'qwen': 1, 'resnet50': 1, 'whisper': 1} |

正式 predictor 不接收 deployment_id，不调用枚举，也不要求全组合 profile。以上列表只统计已观测覆盖。

| Model | observed u min | observed u max | distinct u |
| --- | ---: | ---: | ---: |
| distilbert | 0.00000000 | 0.01130793 | 6 |
| paddleocr | 0.00000000 | 0.00606752 | 5 |
| qwen | 0.04463871 | 0.73192551 | 7 |
| resnet50 | 0.00000000 | 0.00549782 | 5 |
| whisper | 0.00000000 | 0.19275890 | 6 |

u=Σ r*s0，为 standalone_service_demand，不能称为真实 GPU utilization。任务顺序固定为 text, image, ocr, asr, visual_qa, lid。
上表包含 deployed-but-idle 模型的 u=0；task-mix 范围仅对 u>0 定义，完整 min/max/distinct 见 feature_coverage.json。

## 1. 是否足够建立 u → L piecewise curve？

受控 context 内至少两个 distinct u 的组数：0；正式数据是否足够：False。
当前不生成 LoadPoint。五个 routing 点虽然带来多个 u，却同时改变 Qwen task mix 和其他模型负载，不能直接串成负载曲线。
PiecewiseLoadLatencyModel 默认返回 insufficient_profile_data；未来只在同 feature context 的实测范围内插值，不外推。多路径任务不能平均 P95，缺少分布时返回 null。

## 2. 相同/相近 u、不同 task mix 是否出现 latency 差异？

Same model/task, |delta u|<=0.05 and relative delta<=20%, mix L1>=0.1; exploratory comparison only
- qwen/asr: u=0.27302 vs 0.23717，mix L1=0.762；mean=745.88 vs 912.26 ms，平均 run P95=1282.38 vs 1562.39 ms（1 vs 3 runs）。
- qwen/visual_qa: u=0.27302 vs 0.23717，mix L1=0.762；mean=370.77 vs 274.60 ms，平均 run P95=1044.21 vs 518.58 ms（1 vs 3 runs）。
- qwen/lid: u=0.27302 vs 0.23717，mix L1=0.762；mean=441.13 vs 296.60 ms，平均 run P95=1097.35 vs 593.47 ms（1 vs 3 runs）。

这些是描述性差异。smoke 与正式结果的 deployment、warmup、窗口、seed/sample/phase 也不同；不能把差异单独归因于 mix，更不宣称统计显著性。完整成对来源在 feature_coverage.json。

## 3. 已有 deployment/resource features

Stage 1 catalog：五模型实测驻留显存、11 条路径各90样本的 standalone service time。
NVML 离线设备级均值：GPU/memory utilization、power、SM clock、device VRAM、energy/request；Nsight 既有派生表：kernel time/request 和按名称分类的时间比例。
按字段核对已有 NVML summary 和 service identity；Nsight 分类复用现成 CSV 并保留原 report/派生脚本路径，不运行 nsys。

| Resource field | Available paths | Missing paths | Complete co-location rows |
| --- | ---: | --- | ---: |
| kernel_fraction_attn | 11/11 | none in current 11 paths | 148/148 |
| kernel_fraction_conv | 11/11 | none in current 11 paths | 148/148 |
| kernel_fraction_elemcopy | 11/11 | none in current 11 paths | 148/148 |
| kernel_fraction_gemm | 11/11 | none in current 11 paths | 148/148 |
| kernel_fraction_gemv | 11/11 | none in current 11 paths | 148/148 |
| kernel_fraction_other | 11/11 | none in current 11 paths | 148/148 |
| kernel_fraction_rednorm | 11/11 | none in current 11 paths | 148/148 |
| kernel_time_ms_per_request | 11/11 | none in current 11 paths | 148/148 |
| offline_device_vram_used_mib | 11/11 | none in current 11 paths | 148/148 |
| offline_energy_j_per_request | 11/11 | none in current 11 paths | 148/148 |
| offline_gpu_util_pct | 11/11 | none in current 11 paths | 148/148 |
| offline_memory_util_pct | 11/11 | none in current 11 paths | 148/148 |
| offline_power_w | 11/11 | none in current 11 paths | 148/148 |
| offline_sm_clock_mhz | 11/11 | none in current 11 paths | 148/148 |

现有路径字段完整不等于物理资源画像完整：NVML 约100ms采样、部分窗口只有约1–3秒，设备级值不是进程级；kernel 分类只是名称启发式。
co-location pressure 对每个动态 feature 计算 Σ r*p；单位是对应 fingerprint 单位乘 req/s，不是实际 GPU 利用率或经验证的物理压力。
任一正负载贡献路径缺值时总量为 null，另保留 known_partial_sum 与 missing_paths；零负载贡献为零。

## 4. 跨 deployment predictor 还缺什么？

- No matched load-intensity sweep at fixed routing proportions/task mix and deployment.
- Only one formal deployment; two additional deployments are short Stage 2A smoke runs.
- No controlled equal-u/different-mix experiments; no held-out deployment evaluation.
- No sustained overload/saturation labels, input length/duration distribution features or latency distribution model.
- No per-process resource attribution / SM occupancy / hardware memory-bandwidth counters; NVML and kernel categories do not provide these.
- No multi-replica calibration; planning baseline explicitly declines count>1.

Planning predictor 只用 x/r/lambda/offline profiles。实时 queue length、实时 GPU utilization、recent latency 归未来独立 runtime correction，本轮不实现。

## 5. 下一轮最少采哪些点？

建议先采一个可解释的最小设计（只是采样建议，本轮不执行）：

1. x_full 下固定 R0/R50/R100 各自 routing proportions，先取 scale=0.25/0.50/0.75/1.00，各3 seeds；沿用至少约30正式请求/task、有界 drain 和 adaptive overload stop。这是最多12配置/36轮的首轮，不保证所有 routing 都到1.0。
2. 未出现上升区间的 routing 再加1.25/1.50/2.00；发生 overload 即停止更高点。knee 由真机数据观察，不以 rho=0.8 定义。
3. 从稳定低/中负载选至少两个固定 u 的点，构造长 ASR 占比高与短 text/image 占比高的 mix，各3 seeds；保持相同 deployment、相同其他模型负载与测量协议。VQA probe rate 固定，避免混淆自身负载。
4. 至少两个额外可行 deployment，各取相同 workload 和两档稳定负载、3 seeds；例如现有 MILP seed（不含 DistilBERT）和 x_full 配对，其中一个完整 deployment 保留作 holdout。先校验可行性，不枚举全组合，也不按性能选优。
5. 每个采样保留输入长度/音频时长分布、sample IDs、初始 phases、完整请求记录及窗口内吞吐。已有资源 fingerprint 可先复用；新增 service path 时补缺失 offline profile，不猜值。

这只是启动跨 deployment 建模的数据维度下界，不代表这些点已足够得到可靠 predictor。训练、accuracy evaluation、guided search、routing optimization 均未执行。

验证：16/16 foundation tests 通过；106 个既有 deployment/MILP/measurement 代码与 Stage 2A/B JSON 文件 SHA256 未改变。
