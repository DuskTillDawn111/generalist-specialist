# Module 4B：Real-runtime Routing Search Pilot

**判定：NEUTRAL**。This pilot did not establish a confirmed routing gain; canonical routing may already be sufficient for this workload/deployment.

仅 W1_balanced，固定 Module 3C B2 deployment：paddleocr + qwen + resnet50 + whisper。predictor=None。
到此停止；未进入 Module 4C 或任何 deployment-routing、replica、dynamic、forecasting、paper benchmark。

## Runtime metrics

| Metric | Canonical | Final |
|---|---:|---:|
| Weighted mean E2E (ms) | 59.215912 | 59.215912 |
| Weighted task P95 (ms) | 95.686276 | 95.686276 |
| Stability | steady | steady |
| Compatible repetitions | 3 | 3 |

Improvement = (J(r0) - J(rfinal)) / J(r0) = 0.000000（相对 weighted mean E2E）；绝对差 0.000000 ms。
Weighted task P95 是 workload arrival-rate 加权各 task P95，再取 repetition 均值；不是 pooled global P95。

## 必答 14 项

1. Canonical routing：text、visual_qa 全部 qwen；image 全部 resnet50；OCR 全部 paddleocr；ASR、LID 全部 whisper。完整 rps 及 fraction 见下表。
2. Routing freedom：image（qwen/resnet50）、ocr（paddleocr/qwen）、asr 和 lid（qwen/whisper）。text、visual_qa 无 routing freedom。
3. 初始 model pressure：`{"paddleocr": 0.00242700996, "qwen": 0.06461246086888889, "resnet50": 0.0021991287155555556, "whisper": 0.03935853461944445}`；z=0.064612461。
4. 每轮 Top-B（按照 delta_z 从大到小）：
   - 第 1 轮：ocr: paddleocr → qwen, delta=0.040000 rps, delta_z=-0.003785928, ROLLBACK (mean_worse); image: resnet50 → qwen, delta=0.040000 rps, delta_z=-0.005584528, ROLLBACK (mean_worse)
5. 被真实运行的 states：canonical r0；r0 加 10% OCR paddleocr → qwen；r0 加 10% image resnet50 → qwen。三者均在本 pilot 新测 3 次，逐项完整 routing/identity/raw paths 见 runtime_evaluations.json。
6. analytical/runtime mismatch：false；runtime feedback correction：false。只基于同轮已测 Top-B 记录。
7. 真实 runtime rollback：2；原因逐项见 decision_audit.json 和下方 trace。
8. 最终 routing 与 canonical 完全相同：text/visual_qa 全部 qwen，image 全部 resnet50，OCR 全部 paddleocr，ASR/LID 全部 whisper。
9. Weighted mean E2E：59.215912 → 59.215912 ms；改善方向=False。
10. Weighted task P95：95.686276 → 95.686276 ms；改善方向=False。
11. Routing fraction 均未改变；最大变化 0 percentage points。
12. 是否改变 deployment：**no**。
13. Routing grid enumeration calls：**0**。
14. Module 4B 判定：**NEUTRAL**。

## Per-task routing and latency

| Task | Before fraction | After fraction | Mean before/after ms | P95 before/after ms |
|---|---|---|---:|---:|
| text | qwen=100.00% (0.800000 rps) | qwen=100.00% (0.800000 rps) | 55.121769 / 55.121769 | 90.351779 / 90.351779 |
| image | qwen=0.00% (0.000000 rps), resnet50=100.00% (0.400000 rps) | qwen=0.00% (0.000000 rps), resnet50=100.00% (0.400000 rps) | 19.414490 / 19.414490 | 24.327159 / 24.327159 |
| ocr | qwen=0.00% (0.000000 rps), paddleocr=100.00% (0.400000 rps) | qwen=0.00% (0.000000 rps), paddleocr=100.00% (0.400000 rps) | 16.631346 / 16.631346 | 25.643912 / 25.643912 |
| asr | qwen=0.00% (0.000000 rps), whisper=100.00% (0.150000 rps) | qwen=0.00% (0.000000 rps), whisper=100.00% (0.150000 rps) | 205.922053 / 205.922053 | 372.835941 / 372.835941 |
| visual_qa | qwen=100.00% (0.200000 rps) | qwen=100.00% (0.200000 rps) | 168.241741 / 168.241741 | 295.150476 / 295.150476 |
| lid | qwen=0.00% (0.000000 rps), whisper=100.00% (0.400000 rps) | qwen=0.00% (0.000000 rps), whisper=100.00% (0.400000 rps) | 40.262465 / 40.262465 | 44.093525 / 44.093525 |

## Model pressure

| Model | Before | After |
|---|---:|---:|
| paddleocr | 0.002427010 | 0.002427010 |
| qwen | 0.064612461 | 0.064612461 |
| resnet50 | 0.002199129 | 0.002199129 |
| whisper | 0.039358535 | 0.039358535 |

## Budget, cache and validity

Search runtime evaluations=3/6（canonical 包含在内），每 state 3 valid repetitions；新 physical search attempts=9/18；新 valid search runs=9。
Confirmation physical attempts=0/8；新增总 physical attempts=9。历史 cache 运行不重复计为本 pilot physical attempts。
Accepted mutations=0，rollbacks=2；termination=NO_RUNTIME_IMPROVEMENT；decision audit passed=True。
Cache full identity 包括 deployment、冻结 workload、完整 explicit routing、measurement settings/seeds、model/source/input/interpreter hashes、hardware UUID/driver 和审计策略。相同 deployment 不同 routing 是不同 key。
Execution status 与 serving stability 分开：只有 completed repetition 才计 latency，completed genuine unstable 保留；external/uncertain/harness/model interruption 不作为 serving label，仅计 physical attempts。
Worst repetition stability 决定 aggregate class；同 steady 至少 5% weighted mean improvement 才能因 latency 接受，stability improvement 优先。
Confirmation 与 search decision 隔离；若 final==canonical，无额外 confirmation。PASS 表示 pilot 真实收益证据，不表示统计显著性或多 workload 结论。

## 实际运行的三个 routing states

| State | Qwen OCR / image fraction | Stability | Mean ms | Weighted task P95 ms | Mean sample std ms | Repetition means ms | Decision |
|---|---|---|---:|---:|---:|---|---|
| Canonical r0 | 0% / 0% | steady | 59.215912 | 95.686276 | 4.443934 | 63.122089, 54.381004, 60.144641 | baseline / final |
| r0 + OCR 10% → qwen | 10% / 0% | steady | 61.408380 | 113.932222 | 3.902249 | 64.046214, 56.925775, 63.253152 | ROLLBACK: mean_worse |
| r0 + image 10% → qwen | 0% / 10% | steady | 63.166133 | 119.931572 | 4.824154 | 68.185026, 58.563724, 62.749650 | ROLLBACK: mean_worse |

OCR candidate 比 canonical mean 高 3.7025%；image candidate 高 6.6709%。两者均未获得 mean 改善，也未改善 stability，因此均真实 rollback。
Analytical 与 runtime 排序一致：OCR 的 delta_z 较大，且 measured mean/P95 均低于 image；不存在 analytical_runtime_mismatch，也不存在 reject A / accept B 的 correction 案例。
首轮无改善，按 no_improvement_patience=1 停止。LID/ASR 经 Top-B pruning 未运行。本 pilot 不能证明 routing optimization 无价值，也不证明 canonical 是全局 routing optimum。

## Cache 与执行记录

Module 3C 相同 canonical 的近似 identity 只有 hardware UUID 不匹配：历史 GPU-3da0d2e4-b490-e938-703a-87e9ead891cc；当前 GPU-011dad14-9680-4374-06d9-c6186fa3fa47。因此未导入历史结果，canonical 在当前 GPU 新测。详细差异见 cache_compatibility_audit.json。
Canonical 测量完成后、首个 candidate 测量之前，pilot 记录层曾因 Workload 的 MappingProxyType 无法 JSON 序列化而停止；修复仅将记录中的 workload 转为普通 dict。原始 runner、freeze 和失败日志保留，完整修复证据见 implementation_repair_audit.json。冻结 workload/deployment/settings/objective、worker 和 runtime cache identity 均未改变。
恢复时本 pilot canonical exact cache 命中 1 次，因此没有重复 GPU 测量；physical attempts 仍为 9，valid unique state evaluations 仍为 3。
所有 9 次 GPU run 的 execution status=completed、serving stability=steady；invalid/interrupted physical attempts=0。
Decision audit 通过；post_run_verification.json 从原始记录复算 mean/P95、核验实际 routing、完整 cache identity、冻结输入和修复后的 source hash；post_run_provenance.json 再核验 model/input/source/interpreter/hardware 未发生变化。
离线验证通过：18 项既有 routing/runtime/Module 3C 测试及 3 项 Module 4B evidence/gate/serialization 测试。测试用 synthetic evidence 未混入本 pilot GPU 结果。
Final==canonical，所以 confirmation_results.json 记录 required=false；没有额外 confirmation，也未扩大搜索预算。
