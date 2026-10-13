# Module 9A-2 — Protocol-Matched Repeatability and Noise-Source Audit

执行完成；follow-up gate：**REPEATABILITY_STRONG_SUPPORT**。9A-1 gate 永久保持 **GATE_PARTIAL**。

## Anchors

- A1: ['qwen__asr__whisper-small', 'qwen__visual_qa__whisper-small']; original Delta-S=-0.9447368005267134
- A2: ['qwen__asr__paddleocr', 'qwen__lid__paddleocr']; original Delta-S=0.13667072019433113
- A3: ['qwen__asr__whisper-small', 'qwen__lid__whisper-small']; original Delta-S=-0.01719070866976402
- A4: ['historical_phi_asr_full']; historical full replica deployment, raw_0070/71/72; fixed replay source raw_0070

## Exact-state run noise

| State | Mean S | Std | CV | MAD |
|---|---:|---:|---:|---:|
| qwen__asr__whisper-small | 1.417887575 | 0.051729253 | 0.036483325 | 0.036804604 |
| qwen__visual_qa__whisper-small | 2.275807804 | 0.102012643 | 0.044824806 | 0.048660761 |
| qwen__asr__paddleocr | 1.169485079 | 0.087406814 | 0.074739572 | 0.002179412 |
| qwen__lid__paddleocr | 1.091831858 | 0.111045170 | 0.101705377 | 0.048869713 |
| qwen__lid__whisper-small | 1.293609685 | 0.136857898 | 0.105795356 | 0.116474746 |
| historical_phi_asr_full | 1.106607407 | 0.098122815 | 0.088669942 | 0.036953982 |

## Task effect repeatability

| Anchor | Mean Delta S | Median | 95% CI | Sign consistency | SNR | Variance ratio |
|---|---:|---:|---|---:|---:|---:|
| A1 | -0.857920229 | -0.921041783 | [-0.984654005, -0.730349474] | 1.000 | 10.607587 | 56.260455 |
| A2 | 0.077653220 | 0.039165124 | [-0.018461721, 0.199821870] | 0.600 | 0.777096 | 0.301939 |
| A3 | 0.124277890 | 0.092567675 | [-0.005389355, 0.257298676] | 0.200 | 1.201271 | 0.721526 |

## Interpretation and protocol decision

Historical floor=1.179717279783796; current active-pair sigma summary={'median': 0.10201264339439409, 'p95': 0.13169535219429768, 'max': 0.13685789765181672}; historical full-state fixed-input sigma=0.09812281496590526. These are different statistics; homologous null Q95 also saved.
Historical noisy source: B_INPUT_HETEROGENEITY_SUPPORTED_WITH_INSUFFICIENT_CAUSAL_DECOMPOSITION. Each run had two Phi-ASR requests with six distinct IDs. Audio duration and generation-word proxies, queue/arrival timing, overlap and available telemetry are saved; missing historical clocks/thermal/token counters remain missing.
FULL_9A_GROUND_TRUTH_PROTOCOL=AGGREGATE_PROTOCOL_RETAINED; paired bootstrap variance-difference CI=[-0.0012937202981522092, -2.0082176415919147e-05]. Decision uses variance only, never favorable task significance.
FULL_MODULE_9A_RECOMMENDED=True. Full Module9A, Roomie predictor, v3 search, routing/ranking changes and held-out validation were not started.

All 50 timed runs cold-start independent processes, replay fixed sample IDs and exact warmup counts. A1/A2/A3 share identical standalone/pair states rather than duplicate their measurements. A4 preserves the original multi-model deployment, existing budget, lambda and per-request routing; it is reported separately and is not mislabeled as the current saturated active-pair protocol.

No new benchmark workload/memory point, no automatic runtime retry, and no reclassification of the frozen 9A-1 gate.

## Final requested answers（CPU closeout）

1. A1：Qwen + Whisper-small，ASR/VQA，原最大 |Delta-S|；A2：Qwen + PaddleOCR，ASR/LID，8 个比较的 lower observed median（排序第 4 个）；A3：Qwen + Whisper-small，ASR/LID，原最小 |Delta-S|；A4：历史 Phi-ASR full deployment，原 raw_0070/71/72，固定回放 raw_0070。A4 未与前三锚点重合。
2. 六个 exact slowdown state 的 std/CV/MAD 见上表及 noise_decomposition.json。十个物理测量条件的 latency std/CV/MAD 另存 physical_condition_repeatability.csv；standalone 本身没有独立 slowdown 定义，未伪造 S=1。
3. 当前 active-pair sigma 中位数/p95/max = {'median': 0.10201264339439409, 'p95': 0.13169535219429768, 'max': 0.13685789765181672}。A4 fixed-input sigma=0.098122814966。历史 floor=1.179717279783796 是最大 bootstrap null Q95，不是 sigma；同公式 fixed-input null Q95=0.370443627968，较历史值下降 68.599%。
4. 最直接支持的历史来源是 input heterogeneity：每轮仅 2 个 Phi-ASR 请求，三轮 6 个 IDs 全不相同，音频 4.015–13.115 s，预测词数 13–34；duration/latency Spearman=0.885714，词数/latency=0.971008（仅 6 点，描述性）。queue wait 最大 0.341713 ms、arrival delay 最大 0.152147 ms，不能直接解释秒级差异。历史 overlap 在约 0.368–0.436 间变化，背景共驻不持续；原 trace 缺 clocks/thermal/精确生成 token。B_INPUT_HETEROGENEITY_SUPPORTED_WITH_INSUFFICIENT_CAUSAL_DECOMPOSITION；无法证明其占主导的因果方差比例，也不能排除其他因素。
5. matched-input mean(S_i) 未显著降低 variance。六状态等权 pooled variance：aggregate=0.010235274506，matched=0.011047740536；aggregate-minus-matched 95% CI=[-0.0012937202981522092, -2.0082176415919147e-05]。冻结建议 AGGREGATE_PROTOCOL_RETAINED。两种估计都使用精确相同输入；保留固定 sample IDs/order，不把 AGGREGATE_PROTOCOL_RETAINED 理解为允许输入漂移。
6. A1 独立复现：mean Delta-S=-0.857920228627，median=-0.921041783132，5/5 同方向。
7. A2 中位 effect 与 A3 最小 effect 未可靠复现；各自 CI 均跨 0，A3 的 mean direction 还反转。不能将主效应支持外推到所有 task pairs。
8. A1 hierarchical 95% CI=[-0.984654004972, -0.730349474406]，仍不跨 0；abs(effect)-pooled std 95% CI=[0.6719770528498202, 0.9526768113346834]，下界为正。
9. Sign consistency：A1=100%（5/5），A2=60%（3/5），A3=20%（1/5）。
10. Between-task/within-state variances 及 variance ratio：

| Anchor | Between-task variance | Within-state variance | Ratio |
|---|---:|---:|---:|
| A1 | 0.368013559343 | 0.006541247513 | 56.260455 |
| A2 | 0.003015011317 | 0.009985490523 | 0.301939 |
| A3 | 0.007722497006 | 0.010702999882 | 0.721526 |

11. SNR_effect：A1=10.607587，A2=0.777096，A3=1.201271。只报告数值，无 magic cutoff。
12. 9A-1 GATE_PARTIAL 保持不变；1745 项冻结引用及 runtime environment 核验 PASS。
13. 9A-2 follow-up gate=REPEATABILITY_STRONG_SUPPORT，依据是预先冻结的 A1 aggregate estimator 条件。
14. FULL_MODULE_9A_RECOMMENDED=true。
15. 建议下一阶段 Task-Path Kernel Interference Model，保留固定输入、独立进程重复和 unprofiled aggregate slowdown ground truth。本模块未进入完整 9A，未实现 predictor/search，也未修改任何 routing/ranking/method/workload/memory point；完成后停止。

## Execution and interpretation limits

50/50 new timed runs；100 个不同 worker PID；1220 条 target request；760 条 exact matched-input ratios；GPU reuse=0、GPU retries=0。current pair 最小 overlap=0.989496187152；peak VRAM=21135 MiB；所有 worker 正常退出，GPU 已释放。未发生 OOM、强制终止或外部中断。
GPU 全部完成后，CPU 汇总一次因历史 overlap helper 的 sibling import path 缺失而退出。cpu_analysis_failure.json 和原 execution.log 保留错误；CPU-only entrypoint 补齐既有路径后完成汇总。冻结 execution sources、统计规则、协议和 GPU 原始数据未修改，未重新执行 GPU。
A4 是原多模型历史背景回放，与当前 backlog=64 的 active-pair 家族不同。新 clock/thermal ranges 包含 startup/warmup/measurement，不可据此归因历史噪声。生成词数只是 length proxy。固定输入回放并非随机化的输入因素因果实验。
本阶段支持 A1 的 task-conditioned repeatability，尚未证明特定 kernel/SM/DRAM/L2 机制的因果解释，未验证 interference predictor，也未创建未来 held-out validation。
