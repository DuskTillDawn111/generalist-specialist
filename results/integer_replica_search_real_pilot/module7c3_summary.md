# Module 7C-3：Replica-aware Runtime Adapter + Real Integer Search Pilot

Frozen workload：Module 3C W2_generalist_heavy，rates 未调整；capacity_mode=empirical，predictor=None，Top-B=2。
Family routing 使用原 smooth weighted round-robin，再对该 family 的独立 model#i processes 做 deterministic round-robin。模型 handler、precision、input pool 和 measurement semantics 保持冻结版本。
Evidence bounds：Qwen、Whisper-small、ResNet50≤2，其余 9 types≤1；不外推，不枚举 replica vectors。

Adapter smoke=passed；search valid unique evaluations=5/6；search physical repetitions=15/18。
Audit errors：[]。

1. Integer MILP seed：{'convnext_tiny': 0, 'distil_whisper': 0, 'distilbert': 0, 'minicpm': 0, 'paddleocr': 0, 'paddleocr_vl': 0, 'phi4mm': 0, 'qwen': 1, 'resnet50': 0, 'roberta_sst2': 0, 'whisper-small': 0, 'whisper_turbo': 0}。
2. 最终 replica vector：{'convnext_tiny': 0, 'distil_whisper': 0, 'distilbert': 0, 'minicpm': 0, 'paddleocr': 0, 'paddleocr_vl': 0, 'phi4mm': 0, 'qwen': 1, 'resnet50': 0, 'roberta_sst2': 0, 'whisper-small': 1, 'whisper_turbo': 0}。
3. N_G=1，active Generalist types=1。
4. N_S=1，active Specialist types=1。
5. Seed/accepted state 是否有 x=2：False；search 中实际 evaluated x=2 states：[]。smoke 的 x=2 只证明 adapter correctness，不能算作算法接受。
6. Generalist replica-up candidates：[(0, 'qwen', 1, 2, 8, 'PRUNED_TOP_B'), (1, 'qwen', 1, 2, 9, 'PRUNED_TOP_B')]。
7. Specialist candidates：['convnext_tiny', 'distil_whisper', 'distilbert', 'paddleocr', 'paddleocr_vl', 'resnet50', 'roberta_sst2', 'whisper-small', 'whisper_turbo']；逐轮完整 table 见下文和 integer_replica_decisions.json。
8. Calibration 改变 Top-B：[True, True]（按轮）。CPU linear counterfactual 使用相同实际 incumbent、memory/evidence bounds 和 prior visited history；见 linear_counterfactual_audit.json。没有 linear GPU campaign，不宣称其完整运行路径或性能。
9. Weighted mean E2E：260.23710722992075→105.73971586164912 ms，relative gain=0.5936793296421499；weighted task P95：711.7553996873019→200.1848579134921 ms，relative gain=0.7187448693730457。这不是 pooled system P95。
10. Search/confirmation 的 19 次 post-warmup idle 在冻结 5% tolerance 内：True；最大相对偏差=4.494382%。所有 runs 最大 sampled device peak=39796.0 MiB；peaks≤49140：True。3 次 correctness smoke 无 warmup，cold idle 不与 post-warmup 表混比；原始 cold 数值及全样本诊断保留在 memory_validation.json。
11. OOM/runtime resource infeasible：0；GPU release 全部 verified=True。
12. Enumeration calls=0，正式 search 没有调用 oracle。
13. Runtime evaluations=5/6；valid unique=5；confirmation/smoke 独立列账不影响搜索预算或决策。
14. 独立 2+2 confirmation complete=True；mean direction=True，frozen objective direction=True。若 final==seed 则解释为 no deployment change，不把两组噪声当成改善。
15. Adapter、预算和 decision replay 可用于下一阶段受限 W1/W2/W3 integer validation；仅一个 W2 pilot 不能证明多 workload 稳定性或最优性。独立 confirmation 保持改善方向。

Search stability：steady→steady；stop=NO_RUNTIME_IMPROVEMENT。
同 stability 的 mean/P95/memory 判定严格复用原 Module 3 implementation：same-steady 才进行 latency 比较，same-nonsteady 保留 incumbent，latency acceptance 需要 ≥5% mean 改善。失败/不稳定值不被删除。

## 每轮 Generalist/Specialist choice

| Iter | Type | Model / counts | Memory Δ MiB | Empirical Δg | Δz | Rank | Runtime | Decision |
|---|---|---|---:|---:|---:|---:|---|---|
| 0 | generalist_replica_up | qwen 1→2 | 17692.0 | 0.215925255 | 0.06749542421837451 | 8 | not evaluated | PRUNED_TOP_B () |
| 0 | scale_down | qwen 1→0 | -17978.0 | -1.000000000 | None | None | not evaluated | REJECT_INFEASIBLE (coverage) |
| 0 | new_specialist_type | convnext_tiny 0→1 | 610.0 | 1.000000000 | 0.06841046286588892 | 6 | not evaluated | PRUNED_TOP_B () |
| 0 | new_specialist_type | distil_whisper 0→1 | 3672.0 | 1.000000000 | 0.11446373001583332 | 5 | not evaluated | PRUNED_TOP_B () |
| 0 | new_specialist_type | distilbert 0→1 | 750.0 | 1.000000000 | 0.046870649816666665 | 9 | not evaluated | PRUNED_TOP_B () |
| 0 | new_generalist_type | minicpm 0→1 | 18392.0 | 1.000000000 | 0.16934850878417432 | 3 | not evaluated | PRUNED_TOP_B () |
| 0 | new_specialist_type | paddleocr 0→1 | 568.0 | 1.000000000 | 0.04637761728133333 | 11 | not evaluated | PRUNED_TOP_B () |
| 0 | new_specialist_type | paddleocr_vl 0→1 | 2308.0 | 1.000000000 | 0.04637761728133333 | 12 | not evaluated | PRUNED_TOP_B () |
| 0 | new_generalist_type | phi4mm 0→1 | 11772.0 | 1.000000000 | 0.14480670292773484 | 4 | not evaluated | PRUNED_TOP_B () |
| 0 | new_specialist_type | resnet50 0→1 | 604.0 | 1.000000000 | 0.06841046286588892 | 7 | not evaluated | PRUNED_TOP_B () |
| 0 | new_specialist_type | roberta_sst2 0→1 | 988.0 | 1.000000000 | 0.046870649816666665 | 10 | not evaluated | PRUNED_TOP_B () |
| 0 | new_specialist_type | whisper-small 0→1 | 1616.0 | 1.000000000 | 0.18301935288816665 | 1 | steady; mean=105.73971586164912 | ACCEPTED (mean_improved) |
| 0 | new_specialist_type | whisper_turbo 0→1 | 3882.0 | 1.000000000 | 0.18301935288816665 | 2 | steady; mean=119.38031416277741 | ROLLBACK (another_top_b_candidate_preferred) |
| 1 | generalist_replica_up | qwen 1→2 | 17692.0 | 0.215925255 | 0.034994659603233585 | 9 | not evaluated | PRUNED_TOP_B () |
| 1 | scale_down | qwen 1→0 | -17978.0 | -1.000000000 | None | None | not evaluated | REJECT_INFEASIBLE (coverage) |
| 1 | new_specialist_type | convnext_tiny 0→1 | 610.0 | 1.000000000 | 0.0684104628658889 | 3 | not evaluated | PRUNED_TOP_B () |
| 1 | new_specialist_type | distilbert 0→1 | 750.0 | 1.000000000 | 0.046870649816666665 | 5 | not evaluated | PRUNED_TOP_B () |
| 1 | new_generalist_type | minicpm 0→1 | 18392.0 | 1.000000000 | 0.0809531398221616 | 1 | steady; mean=133.14568574188087 | ROLLBACK (mean_worse) |
| 1 | new_specialist_type | paddleocr 0→1 | 568.0 | 1.000000000 | 0.04637761728133333 | 7 | not evaluated | PRUNED_TOP_B () |
| 1 | new_specialist_type | paddleocr_vl 0→1 | 2308.0 | 1.000000000 | 0.04637761728133333 | 8 | not evaluated | PRUNED_TOP_B () |
| 1 | new_generalist_type | phi4mm 0→1 | 11772.0 | 1.000000000 | 0.07657483206339782 | 2 | steady; mean=124.81797054451548 | ROLLBACK (mean_worse) |
| 1 | new_specialist_type | resnet50 0→1 | 604.0 | 1.000000000 | 0.0684104628658889 | 4 | not evaluated | PRUNED_TOP_B () |
| 1 | new_specialist_type | roberta_sst2 0→1 | 988.0 | 1.000000000 | 0.046870649816666665 | 6 | not evaluated | PRUNED_TOP_B () |
| 1 | scale_down | whisper-small 1→0 | -1616.0 | -1.000000000 | None | None | not evaluated | SKIP_TABU (already visited or rejected) |

## Independent confirmation

{"complete": true, "influences_search": false, "relative_mean_gain": 0.6348206339877065, "mean_direction_maintained": true, "objective_direction_maintained": true, "objective_reason": "mean_improved", "search_final_equals_seed": false, "physical_attempts": 4}


Raw repetitions、PID/CUDA process identity、model/environment/adapter hashes、family/replica metrics、实际 idle/peak memory、shutdown/release evidence 均保留。
Qwen presence 由唯一 VQA capability 强制；Qwen/Whisper g(2) 来自 ASR，mixed-task 和异构 interference 的泛化仍是 surrogate。
没有 new scaling profiling，没有 replica>evidence bound，没有 dynamic adaptation、full multi-workload integer benchmark 或 final paper benchmark。完成后停止。

Offline report clarification：memory 按 cold/post-warmup phase 分层；linear counterfactual 匹配 visited history。原 search decision、raw metrics、pilot protocol、calibration 和 source freeze 均未改动，没有新增 GPU run。

## 已观察的 search runtime states

| Deployment（未列 types=0） | Repetitions | Stability | Mean E2E ms | Weighted task P95 ms | Planning idle MiB | Actual idle range MiB | Sampled device peak MiB |
|---|---:|---|---:|---:|---:|---|---:|
| qwen=1 | 3 | steady | 260.237 | 711.755 | 17978 | 18138–18786 | 19606 |
| qwen=1, whisper-small=1 | 3 | steady | 105.740 | 200.185 | 19594 | 19796–20448 | 21405 |
| qwen=1, whisper_turbo=1 | 3 | steady | 119.380 | 217.947 | 21860 | 22058–22710 | 23667 |
| minicpm=1, qwen=1, whisper-small=1 | 3 | steady | 133.146 | 275.316 | 37986 | 38172–38738 | 39796 |
| phi4mm=1, qwen=1, whisper-small=1 | 3 | steady | 124.818 | 218.356 | 31366 | 32022–32500 | 33718 |

没有对被 Top-B 裁剪的 Qwen=2 state 做 W2 runtime measurement，因此本 pilot 不能给出复制 Qwen 与新增 Specialist 的 W2 因果性能比较。三次双副本 smoke 只验证 adapter correctness；最终选多个 model types 仍是合法整数搜索结果，不视为 replica-count 支持失败。
