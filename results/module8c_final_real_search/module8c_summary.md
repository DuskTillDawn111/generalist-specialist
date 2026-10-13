# Module 8C — Final Real Multi-workload Search

**PASS，六组搜索与 independent confirmation 完成。** Search physical attempts=72；confirmation attempts=24；另保留两次 premeasurement failure；enumeration=0。完成后停止。

第一轮在 PaddleOCR 启动阶段因 libnvrtc.so.13 搜索路径失败，measurement 未开始、request_records=0、valid states=0、GPU 已释放。完整失败目录保存于 results/module8c_final_real_search_startup_failure_001/。修复仅复用历史 OCR LD_LIBRARY_PATH；新的执行环境及恢复原因在 startup_recovery.json 中冻结。六组有效搜索的 states/repetitions/confirmation 预算未增加，没有丢弃已测结果或更改 frozen 8B inputs。

严格使用 8B 冻结 pool v2.1、21 条 admitted routing edges、27/27 demand catalog 和六组 λ。PaddleOCR-VL 固定 x=0；JSON/CSV 保存包括此零项的全部 12-type integer vector。未修改 quality/capability/λ/算法。Predictor=None、Top-B=2、每组≤6 unique runtime states，每 state 三 repetitions；seed/final 各二独立 confirmation，不影响搜索决策。沿用 max_iterations=3、patience=1、NoiseAwareRuntimeObjective(1ms,5%)、210s measurement/10s warmup、empirical g/M 与 evidence bounds。

|Workload|Final deployment|N_G/N_S|Mean E2E ms (seed→final)|Mean gain|Weighted task P95 ms (seed→final)|P95 gain|Stability|States|
|---|---|---|---|---|---|---|---|---|
|demand_balanced|distilbert=1, paddleocr=1, phi4mm=1, resnet50=1, whisper-small=1|1/4|44.702→44.702|0.00%|68.537→68.537|0.00%|steady|3/6|
|speech_heavy|distil_whisper=1, distilbert=1, paddleocr=1, phi4mm=1, resnet50=1, whisper-small=1|1/5|79.678→55.685|30.11%|213.474→82.475|61.37%|steady|5/6|
|vision_heavy|distilbert=1, paddleocr=1, phi4mm=1, resnet50=1, whisper-small=1|1/4|32.325→32.325|0.00%|45.561→45.561|0.00%|steady|3/6|
|text_ocr_heavy|distilbert=1, paddleocr=1, phi4mm=1, resnet50=1, whisper-small=1|1/4|21.319→21.319|0.00%|30.396→30.396|0.00%|steady|3/6|
|reasoning_vqa_heavy|distilbert=1, paddleocr=1, phi4mm=1, qwen=1, resnet50=1, whisper-small=1|2/4|103.243→95.491|7.51%|156.144→157.304|-0.74%|steady|5/6|
|no_vqa_phase|distil_whisper=1, distilbert=1, paddleocr=1, resnet50=1, whisper-small=1|0/5|22.931→21.355|6.87%|34.505→27.889|19.17%|steady|5/6|

## Final deployment diversity

实际 distinct final deployments=4；N_G values=[0, 1, 2]；N_S values=[4, 5]。
Planning DIVERSIFIED 与实际 final diversity 分别记录。所有重复状态如实保留，不通过修改 workload 或扩大预算追求差异。
- demand_balanced, vision_heavy, text_ocr_heavy: distilbert=1, paddleocr=1, phi4mm=1, resnet50=1, whisper-small=1
- speech_heavy: distil_whisper=1, distilbert=1, paddleocr=1, phi4mm=1, resnet50=1, whisper-small=1
- reasoning_vqa_heavy: distilbert=1, paddleocr=1, phi4mm=1, qwen=1, resnet50=1, whisper-small=1
- no_vqa_phase: distil_whisper=1, distilbert=1, paddleocr=1, resnet50=1, whisper-small=1

## Reasoning/VQA-heavy Generalist competition

Final Generalists=['phi4mm', 'qwen']；搜索 seed 为 Phi + 四 Specialists。以下是实际 mutation evidence，新增 Generalist 可能与 Phi 同时驻留，不能等价为单 Generalist 替换胜负。
- qwen: seed=False，final=True，containing runtime states=3，accepted mutations=1，rollback mutations=0。
  - iteration=0，x 0→1，analytical rank=1，mean=95.491ms，weighted P95=157.304ms，stability=steady，ACCEPTED (mean_improved)。
- phi4mm: seed=True，final=True，containing runtime states=5，accepted mutations=0，rollback mutations=0。
- minicpm: seed=False，final=False，containing runtime states=1，accepted mutations=0，rollback mutations=1。
  - iteration=0，x 0→1，analytical rank=2，mean=107.236ms，weighted P95=158.606ms，stability=steady，ROLLBACK (mean_worse)。

## No-VQA and replicas

No-VQA: seed N_G=0，final N_G=0；N_G=0 preserved=True。Final=distil_whisper=1, distilbert=1, paddleocr=1, resnet50=1, whisper-small=1；search stability=steady，confirmation stability=steady。
Final x=2 activation by workload={"demand_balanced": [], "speech_heavy": [], "vision_heavy": [], "text_ocr_heavy": [], "reasoning_vqa_heavy": [], "no_vqa_phase": []}。所有 x=2 proposals 的 generated/pruned/evaluated/accepted/rollback 证据见 replica_activation.json，未测候选不推断真实优劣。

## Runtime feedback and independent confirmation

Analytical positive delta_z 却不改善实际 incumbent 的 reversal=12；实际 Top-B 第二名被 runtime objective 优先的次数=0。另一个 Top-B 更好的 rollback 与不改善 incumbent 分开记录。

|Workload|Confirmation mean seed→final ms|Mean gain|Confirmation P95 seed→final ms|P95 gain|Objective direction|
|---|---|---|---|---|---|
|demand_balanced|42.892→43.123|-0.54%|64.540→64.085|0.70%|False (latency_gain_below_min_relative_improvement)|
|speech_heavy|66.534→52.421|21.21%|125.242→76.741|38.73%|True (mean_improved)|
|vision_heavy|31.219→31.418|-0.64%|41.822→41.012|1.94%|False (latency_gain_below_min_relative_improvement)|
|text_ocr_heavy|20.832→20.726|0.51%|28.803→28.438|1.27%|False (latency_gain_below_min_relative_improvement)|
|reasoning_vqa_heavy|103.625→94.771|8.54%|155.275→154.826|0.29%|True (mean_improved)|
|no_vqa_phase|22.289→20.725|7.02%|29.639→27.808|6.18%|True (mean_improved)|

Confirmation 即使不维持 objective gain，也如实保留，不回改已冻结 final。seed=final 时仍独立执行 2+2，不缓存复用。

Memory、stability、per-task 与 per-replica metrics、CUDA worker identities 和 telemetry 在 raw 与 memory_validation.json 中保留；sampled peak 不是精确瞬时峰值。VQA/LID 使用已授权语义准入，数值 quality threshold 仍为 UNSPECIFIED。
未启动 baseline、ablation、dynamic、第二硬件实验。结论只覆盖冻结 workload、单平台、证据 bounds 和已访问状态，不声称全局最优或某 Generalist 普遍占优。

## Zero-rate transport recovery

No-VQA 首次调用在 0.36 秒的 transport 配置校验中退出：没有 workers、没有测量数据。原 transport 仅根据驻留模型的 capability 校验全六维 workload，因 VQA=0 且无 VQA worker 判为未知任务。该次完整 cache/log/process/config 和 campaign failure 状态保留于 no_vqa_pretransport_failure_001/。CPU regression checks 通过后，仅本 phase 采用 module8c_runtime_compat 外部兼容层；原算法、evaluator、harness 文件、handlers、capability、quality、λ、计时、aggregation、bounds 均未改。内部 transport 仅省略无驻留 capability 且到达率/路由均为零的任务；search/cache/raw 仍保留原六维 λ。非零 unsupported 路由仍拒绝。runner AST 除模块入口外与原实现相同，前五组没有重跑。两次 premeasurement failures 单独计数，不隐藏、不替代已测数据；每个有效 state 仍恰为三次，每组有效 states≤6，confirmations=2+2。恢复 provenance 在 zero_rate_transport_recovery.json 冻结，每个 no-VQA raw 保存 compatibility source hash。

## Memory surrogate diagnostic

Five phases contain at least one post-warmup idle observation beyond the historical 5% diagnostic tolerance; maximum deviation is 7.756%. All actual sampled device peaks remain within the frozen 49140 MiB hard budget (campaign maximum 40356 MiB). The tolerance is reported as a surrogate-accuracy diagnostic and is not silently converted into an acceptance gate. Frozen empirical memory tables and final decisions were not changed. Per-phase/all-run details are retained in memory_surrogate_diagnostic.json and memory_validation.json.

Final verification: 164 original native model/adapter files rehashed unchanged; 24 official runtime state means independently reconstructed from task contributions; 96 valid physical measurements authenticated. Two premeasurement failures are separately preserved, making 98 total physical attempts. Campaign processes exited and GPU workers released. No further experiments launched.
