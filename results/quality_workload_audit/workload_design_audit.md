# Demand-normalized workload design audit

Target shares 冻结后 λ 由解析公式唯一确定。各 regime 的总 Qwen reference demand 相同，等于旧三组总 demand 中位数；没有依据 deployment/planning outcome 反向调参。没有运行 canonical planning 或 real search。

|Proposal|Target demand shares (text,image,ocr,asr,VQA,lid)|λ vector rps|Total rps|Dominant task|
|---|---|---|---:|---|
|demand_balanced|0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667|1.166374, 0.372926, 0.550094, 0.079601, 0.360296, 0.372137|2.901428|text, image, ocr, asr, visual_qa, lid|
|speech_heavy|0.0750, 0.0750, 0.0750, 0.4500, 0.0750, 0.2500|0.524868, 0.167817, 0.247542, 0.214923, 0.162133, 0.558205|1.875489|asr|
|vision_heavy|0.0800, 0.6000, 0.0800, 0.0800, 0.0800, 0.0800|0.559859, 1.342535, 0.264045, 0.038209, 0.172942, 0.178626|2.556215|image|
|text_ocr_heavy|0.3500, 0.0750, 0.3500, 0.0750, 0.0750, 0.0750|2.449385, 0.167817, 1.155198, 0.035821, 0.162133, 0.167461|4.137815|text, ocr|
|reasoning_vqa_heavy|0.0800, 0.0800, 0.0800, 0.0800, 0.6000, 0.0800|0.559859, 0.179005, 0.264045, 0.038209, 1.297065, 0.178626|2.516808|visual_qa|
|no_vqa_phase|0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.2000|1.399649, 0.447512, 0.660113, 0.095521, 0.000000, 0.446564|3.049359|text, image, ocr, asr, lid|

Common reference demand=0.312392558 GPU-reference s/s。Raw share 与实际反算 demand share 全量见 JSON；浮点误差范围内完全匹配 target。相同 reference load 不代表相同真实 E2E 或不同 deployment。

Vision regime 以 Imagenette image 为主；reasoning/VQA 以当前 ScienceQA-IMG multiple choice 为主，不能描述为覆盖 generic reasoning。Speech 同时包含转写与独立语言识别。每个 regime 保留原 dataset/split/prompt/normalization。

No-VQA phase 语义合理：真实流量阶段可以没有 ScienceQA 请求；既不是删去其他任务的视觉语义，也没有改变能力/评分。它移除唯一 VQA coverage 约束，可研究 Generalist choice 是否改变，但不预言一定切换到 Phi/MiniCPM；LID 仍由 Qwen/Whisper 覆盖。现有 Specialists 共同覆盖其余五类，所以该 phase 连 Generalist presence 都不是结构性必需，N_G=0 也具备 capability coverage；是否实际采用仍未测。

当前冻结的是可复现 proposals 和 hash。最终 workload matrix 暂不能作为最终 benchmark 冻结：需先冻结六任务质量政策、确认 candidate_pool_v2.1 的能力/adapter/service-demand 注册策略；未来执行协议可延用现有 10s warmup / 210s window、deterministic periodic、三次重复，但本阶段不启动。建议保留这六种 regime；禁止依据未来 planning/search 结果改变 target shares。
