# Module 8A quality and Pareto audit

Within each task: identical 200 sample IDs, dataset/split/task/prompt/output metric, existing model-specific preprocessing/decoding retained. Across tasks: metrics are different and cannot be averaged into universal quality.

s0 from frozen standalone catalog, often historical 30-row service pool / ASR 4–6s subset; audit quality over frozen 200 evaluation IDs. Pareto is an exploratory cross-artifact pairing, not a new co-measured latency-quality benchmark.

Quality 是同 task 的 task-specific matrix；不使用 accuracy 与 WER 的跨任务混合分数。CER/WER 使用 corpus error totals / corpus reference units，保留 OCR exact match。现有实现没有 F1/macro-F1，故未替换 metric。

|Task|Model|N|Accuracy / top1|CER|OCR exact|WER|s0 ms|Pareto|
|---|---|---:|---:|---:|---:|---:|---:|---|
|text|qwen|200|0.945000|—|—|—|44.639|False|
|image|qwen|200|1.000000|—|—|—|139.613|False|
|ocr|qwen|200|—|0.102490|0.940000|—|94.648|False|
|asr|qwen|200|—|—|—|0.041732|654.078|False|
|visual_qa|qwen|200|0.900000|—|—|—|144.507|True|
|lid|qwen|200|0.860000|—|—|—|139.909|False|
|text|phi4mm|200|0.905000|—|—|—|68.123|False|
|image|phi4mm|200|0.920000|—|—|—|230.115|False|
|ocr|phi4mm|200|—|0.051724|0.930000|—|204.711|False|
|asr|phi4mm|200|—|—|—|0.019520|1129.589|False|
|text|minicpm|200|0.920000|—|—|—|59.035|False|
|image|minicpm|200|1.000000|—|—|—|221.705|False|
|ocr|minicpm|200|—|0.012452|0.970000|—|160.371|True|
|asr|minicpm|200|—|—|—|0.019295|777.259|True|
|text|distilbert|200|0.925000|—|—|—|3.769|True|
|image|resnet50|200|1.000000|—|—|—|5.498|True|
|image|convnext_tiny|200|1.000000|—|—|—|5.848|False|
|ocr|paddleocr|200|—|0.008621|0.960000|—|6.068|True|
|ocr|paddleocr_vl|200|—|0.026820|0.950000|—|85.532|False|
|asr|whisper-small|200|—|—|—|0.042181|190.166|False|
|lid|whisper-small|200|0.980000|—|—|—|27.084|True|
|asr|whisper_turbo|200|—|—|—|0.035001|170.522|True|
|lid|whisper_turbo|200|0.995000|—|—|—|83.179|True|
|asr|distil_whisper|200|—|—|—|0.037694|146.224|True|
|text|roberta_sst2|200|0.955000|—|—|—|5.506|True|

Pareto：最小化 s0；accuracy 最大化；ASR WER 最小化；OCR 同时最小化 CER、最大化 exact。支配结果没有正式 quality gate 的含义。单 task 的 Pareto 未包含整模型共享加载、VRAM 或跨 task 覆盖，不能据此剔除整模型；例如 Whisper-small 的 ASR 点被支配，但其 LID 点在 frontier 上。

|Task|Specialist|Metric|Qwen|Specialist|Δ quality (S−Q)|s0 speedup|Faster but worse?|
|---|---|---|---:|---:|---:|---:|---|
|text|distilbert|accuracy|0.945000|0.925000|-0.020000|11.84|True|
|text|roberta_sst2|accuracy|0.945000|0.955000|+0.010000|8.11|False|
|image|resnet50|top1_accuracy|1.000000|1.000000|+0.000000|25.39|False|
|image|convnext_tiny|top1_accuracy|1.000000|1.000000|+0.000000|23.87|False|
|ocr|paddleocr|cer|0.102490|0.008621|-0.093870|15.60|False|
|ocr|paddleocr_vl|cer|0.102490|0.026820|-0.075670|1.11|False|
|asr|whisper-small|wer|0.041732|0.042181|+0.000449|3.44|True|
|asr|whisper_turbo|wer|0.041732|0.035001|-0.006731|3.84|False|
|asr|distil_whisper|wer|0.041732|0.037694|-0.004039|4.47|False|
|lid|whisper-small|accuracy|0.860000|0.980000|+0.120000|5.17|False|
|lid|whisper_turbo|accuracy|0.860000|0.995000|+0.135000|1.68|False|

Legacy quality presets 在 benchmark/profiles/placement_scenario.json 中早已存在（none / specialist_baseline / best_observed），其中后两者覆盖 text/image/OCR/ASR；完整数值在 quality_evaluation_manifest.json 保留。它们是历史四任务 placement 条件，没有自动推广到当前六任务 v2 optimizer；visual_qa/LID 未定义 threshold。

Current six-task v2 optimizer quality_threshold_status = UNSPECIFIED。未在看结果后创建阈值；低质量不会被此 audit 偷偷标为 optimizer 可用。正式淘汰/接受留待冻结政策。

Closed-set contract diagnostics: qwen/lid invalid=14/200；phi4mm/image invalid=12/200；whisper-small/lid invalid=4/200。

各 task 的最弱 primary quality 点估计（排序，不是新 threshold）：text: phi4mm accuracy=0.905000；image: phi4mm top1_accuracy=0.920000；ocr: qwen cer=0.102490；asr: whisper-small wer=0.042181；visual_qa: qwen accuracy=0.900000；lid: qwen accuracy=0.860000。

每条结果保留 source revision、manifest hash、adapter hash、raw response/IDs 和模型资产哈希。历史质量结果无法绑定全部五项复用条件，故统一重新评估当前 handlers；没有将历史 summary 混入当前 matrix。

Generalist VQA 的两条新测 path 单独放在 generalist_vqa_revalidation.json，尚不属于 frozen v2 的 25 条 matrix，也没有伪造其 s0。
