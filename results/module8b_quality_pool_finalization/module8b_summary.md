# Module 8B — Quality Admission + Candidate Pool v2.1 Finalization

**PASS；FINAL_BENCHMARK_MATRIX_READY=true；完成后停止。** 本阶段仅有 6 次新增 VQA standalone physical attempts（2 paths × 3 repetitions，每次一次 warmup + 30 measured requests）。Real deployment search=0，RuntimeEvaluator=0，baseline/ablation=0，算法修改=0，enumeration=0。

## Frozen policy and formal exclusions

用户明确选择：历史 SLA + VQA/LID 语义准入。四任务完全复用已经用于 placement_specialist_baseline_g1 的原阈值；保留 best_observed 为历史记录，未事后选择另一 preset。VQA/LID 的正式政策是 SEMANTIC_VALIDITY_ONLY：数值门槛仍为 UNSPECIFIED，但在此已授权政策下 verified+measured semantic paths 可以准入。单样本错答/无效标签保留并计错，不添加临时“零错误”门槛。

|Task|Metric|Direction / threshold|Policy status|
|---|---|---|---|
|text|accuracy|>= / 0.925|HISTORICAL_THRESHOLD|
|image|accuracy|>= / 1.0|HISTORICAL_THRESHOLD|
|ocr|exact_match|>= / 0.96|HISTORICAL_THRESHOLD|
|asr|wer|<= / 0.042181|HISTORICAL_THRESHOLD|
|visual_qa|accuracy|higher_is_better_report_only / None|SEMANTIC_VALIDITY_ONLY|
|lid|accuracy|higher_is_better_report_only / None|SEMANTIC_VALIDITY_ONLY|

27 verified+measured paths 中，21 admitted；以下 6 paths 被已有数值 gate 正式排除。排除的是 routing edge，模型 capability inventory 不降级：

|Model / task|Measured quality|Historical threshold|
|---|---:|---:|
|qwen / ocr|0.940000|0.960000|
|phi4mm / text|0.905000|0.925000|
|phi4mm / image|0.920000|1.000000|
|phi4mm / ocr|0.930000|0.960000|
|minicpm / text|0.920000|0.925000|
|paddleocr_vl / ocr|0.950000|0.960000|

PaddleOCR-VL 仅有的 OCR edge 未过 gate，故在本 planning 中固定 x=0；其模型身份和 verified capability 仍保留于 v2.1。Phi 的 text/image/OCR 被屏蔽，但 ASR/VQA 可用；MiniCPM text 被屏蔽；Qwen OCR 被屏蔽。所有状态与理由保存于 quality_admission_matrix.json。

## VQA capability and standalone evidence

v2→v2.1 只有 Phi/MiniCPM visual_qa pending→verified 两项 capability 更新，模型 IDs/revisions/weights/precision/roles/VRAM/其他能力不变。原 v2 未覆盖。正式 versioned adapter 的推理/prompt/归一化/评分 body 与 8A revalidation AST 等价，只替换版本 metadata。模型规范 ID 为 phi4mm（Phi）。

VQA s0：Phi=160.116457611 ms；MiniCPM=195.690576922 ms。均执行原 run_active_single_profile.main 和 7A-2 的隔离、计时、原始 summary、卸载校验；相同前 30 个 ScienceQA service-pool rows、greedy max_new_tokens=1、相同一次 warmup、三独立 repetitions、90 samples weighted mean。原有 25 条值及 provenance 完全不变，completeness=27/27。

新 VQA capable/admitted Generalists = Qwen/Phi/MiniCPM。λVQA>0 时仅要求至少一个 admitted VQA model，已不能推出 qwen≥1；VQA 阈值仍未数值化，这一点不会被“READY”隐藏。

## Frozen six-workload planning outcomes

全部 λ 向量逐项与 Module 8A proposal 相等、原 JSON SHA-256 未变。只调用原 MILP seed、canonical LP、bottleneck_feedback、x±1 generation 和 exact empirical delta_z ranking；不执行 search loop，不实例化/调用 RuntimeEvaluator，不模拟 runtime feedback。质量作为 routing-edge mask 传入原 solver；现有 empirical g、M(x)、replica evidence bounds 均保留。

A = Phi + DistilBERT + ResNet50 + PaddleOCR + Whisper-small，各 x=1，15,310 MiB。B = A 去掉 Phi，3,538 MiB。Top-B 表示由 seed 新增的 model family，不是实测 final deployment。

|Workload|Seed|N_G / N_S|Bottleneck model(s)|Dominant bottleneck task|Top-1 addition|Top-2 addition|
|---|---|---|---|---|---|---|
|demand_balanced|A|1 / 4|phi4mm|visual_qa|qwen|minicpm|
|speech_heavy|A|1 / 4|phi4mm,whisper-small|asr|distil_whisper|whisper_turbo|
|vision_heavy|A|1 / 4|phi4mm|visual_qa|qwen|minicpm|
|text_ocr_heavy|A|1 / 4|phi4mm|visual_qa|qwen|minicpm|
|reasoning_vqa_heavy|A|1 / 4|phi4mm|visual_qa|qwen|minicpm|
|no_vqa_phase|B|0 / 4|whisper-small|asr|distil_whisper|whisper_turbo|

五个含 VQA workload 的 minimum-memory seed 相同；no-VQA 的 seed 为 N_G=0。原先 Qwen monopoly 消失，但 minimum-memory seed 不等于性能最优，不能把 Phi seed presence 当作已经赢得 runtime benchmark。

Reasoning/VQA-heavy：Phi 在 seed，Qwen/MiniCPM 都成为合法新增 Generalist candidate 并占 Top-B；三种 Generalist 的单独替代 witnesses 都通过 admitted-edge routing、capacity、memory 验证。新增 Qwen/MiniCPM 是 one-hop additions；单 Generalist替代 states 是独立 feasibility diagnostics，未伪装成同一跳 swap。

## Diversity gate and practical limits

Suite status=DIVERSIFIED。Seed signatures=2；Top-B family sets=2；bottleneck model sets=3；dominant tasks 为 ASR/VQA。四组 Top-B 是 Qwen/MiniCPM（new_generalist），speech-heavy/no-VQA 是 distil-whisper/Whisper-turbo（new_specialist）。

Diversity rule 在规划结果出现前冻结：不同 dominant bottleneck tasks、不同 Top-B family sets，以及 Generalist/Specialist mutation patterns 均出现则 DIVERSIFIED。结果满足这一有限定义，不代表六组都独立：demand-balanced、vision-heavy、text/OCR-heavy、reasoning/VQA-heavy 都仍由 Phi VQA 主导，四组相同 Top-B；image 与 OCR 没有成为 seed bottleneck。没有为改变此结果修改 workload。

Generalist additions 没有被 memory 或 VQA quality/capability gate 阻止。Speech/no-VQA 中 Qwen/Phi/MiniCPM 仍有合法 witnesses，但部分 Generalist proposals 被 rank 剪枝。Phi scale-down 在含 VQA seed 中因 coverage 被拒绝，这是删除唯一已部署 VQA model 的约束，不是对 Phi 本身不支持 VQA 的判定。

## Final benchmark matrix gate

五 gate={"quality_policy_frozen": true, "candidate_pool_v2_1_frozen": true, "service_demand_completeness_100_percent": true, "generalist_VQA_monopoly_resolved_or_explained": true, "workload_planning_audit_DIVERSIFIED": true}；remaining blockers=[]。

已冻结 final_benchmark_workload_matrix.json 和 SHA-256，原 proposal λ 不变；绑定本次 policy/admission、v2.1、真实 service catalog 和 planning evidence。READY 适用于用户选择的语义准入政策，不声称满足尚未定义的 VQA/LID accuracy SLO，也不证明规划排名会成为真实部署结果。下一阶段可以执行冻结矩阵上的真实 multi-workload benchmark；本阶段没有启动。

## Final fourteen answers

1. 六任务 policy：text/image/OCR/ASR=HISTORICAL_THRESHOLD；VQA/LID=SEMANTIC_VALIDITY_ONLY，numeric threshold=UNSPECIFIED。
2. 正式排除 6 条 paths：Phi text/image/OCR、MiniCPM text、Qwen OCR、PaddleOCR-VL OCR；未按相对排名或 Pareto 自动淘汰。
3. Phi/MiniCPM VQA 已正式进入独立 v2.1，原 v2 未改。
4. 新 VQA s0：Phi 160.116458 ms；MiniCPM 195.690577 ms。
5. 达到 27/27；原 25 条没有重测或修改。
6. Qwen 不再 mandatory；五个含 VQA 的 seed 都是 Phi + 四 Specialists。
7. Reasoning/VQA-heavy 出现 planning Generalist competition：Phi seed，Qwen/MiniCPM Top-B，三者独立替代 states 都合法；没有 runtime 竞争结论。
8. No-VQA 允许且实际 seed 为 N_G=0，四 Specialists 完整覆盖。
9. Seed 有两种，五组相同，no-VQA 不同；不是六个独立 seeds。
10. Bottleneck model sets 有三种：Phi；Phi+Whisper-small；Whisper-small。Dominant tasks 有 VQA/ASR 两种。
11. Top-B 有两种明显类别：四组 Qwen/MiniCPM；两组 distil-whisper/Whisper-turbo。
12. DIVERSIFIED，按预先冻结的有限 diversity 定义；四组重复已披露。
13. FINAL_BENCHMARK_MATRIX_READY=true；final matrix 与 hash 已冻结。
14. 下一步可进入真实 multi-workload benchmark；本阶段完成并停止。

## Protocol clarification

Module 8A report characterized catalog ASR s0 as a 4–6s subset. The actual Stage-1 protocol selects the first 30 service-pool task rows without duration filtering. Historical values/inputs are not changed.
实际前 30 条 ASR 音频 min/mean/max=2.695/8.687/24.450 s。此说明只纠正上阶段文字，不改变历史 catalog 或其 demand-share 数字。

产物：results/module8b_quality_pool_finalization/；正式 pool/catalog/raw measurements 同时保存在 results/candidate_pool_v2_1/。audit_validation.json、artifact_manifest.json 保留 immutable-input 和输出 hash 核验。
