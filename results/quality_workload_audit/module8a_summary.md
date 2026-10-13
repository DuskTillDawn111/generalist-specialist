# Module 8A — Quality + Workload Demand Audit

已完成审计并停止。无新的 deployment search、baseline/ablation、predictor、新模型或 7B/7C 算法修改。质量评估使用现有六类各 200 条冻结 manifest；25 条 verified path 及 Phi/MiniCPM VQA revalidation 分开。

## 1. W1/W2/W3 在 raw request space 中有多不同？

Pair 顺序为 W1/W2、W1/W3、W2/W3：L1=0.033435, 0.147732, 0.128482；cosine=0.999360, 0.988994, 0.991033。差异有限，W1/W2 近似同方向；workload 名称夸大了 regime 差别。

## 2. 在 Qwen service-demand space 中有多不同？

同样顺序：L1=0.025821, 0.149637, 0.127530；cosine=0.999428, 0.987341, 0.988554。W3 相对 W1 的 demand L1 略大于 raw L1，因此不能笼统说 demand 空间总是比 request 更相似。绝对总 demand 单独报告，归一化没有覆盖 load-level 差异。

## 3. ASR 是否主导三个 workload？

ASR 是三者最大单项：W1_balanced=31.41%, W2_generalist_heavy=30.12%, W3_specialist_friendly=29.66%；但都不到 50%。ASR+LID：49.32%, 48.15%, 42.35%。应表述为最大单项 speech-related cost，而非 ASR 绝对多数。

## 4. 为什么 Whisper-small 三组均高价值？

同时覆盖 ASR/LID，small LID s0=27.084 ms，比 Qwen 139.909 ms、turbo 83.179 ms 更低；ASR s0=190.166 ms 对 Qwen 654.078 ms 有下降。三组 mix 接近，Qwen mandatory VQA 相同。冻结 trace 的 W1/W2 首轮 Top-B 是 small/turbo，W3 是 MiniCPM/small；三组都由 runtime 选择 small。第二轮 Top-B Generalist additions 未改善，便停止。empirical g(2) 为小于线性的代表任务测量，不能代替 runtime。完整路径与压力见 convergence report。

## 5. 25 条 verified path 的质量是否可比较？

25/25 完成。同 task 下可比较：相同 frozen IDs/dataset/split/task semantics，复用各模型既有 preprocessing 和 serving decode；不跨 task 混合质量。各模型原生官方图像 transform/API 有合理差异且已记录。质量200样本与 s030-row/短ASR subset 的范围差异限制 Pareto 的 latency 外推。

## 6. 是否存在 latency-quality tradeoff？

实际数据：text/distilbert 更快但 accuracy 更差，Δ=-0.020000; asr/whisper-small 更快但 wer 更差，Δ=+0.000449。这些是描述性点估计，不代表统计显著性。

当前最终 speech offload：Qwen→Whisper-small ASR WER 0.041732→0.042181；LID accuracy 0.860000→0.980000。不能只报告 latency 改善而省略此质量变化。

## 7. 是否有质量太差而不适合直接进入最终 optimizer 的模型？

测量诊断：qwen/lid invalid=14/200；phi4mm/image invalid=12/200；whisper-small/lid invalid=4/200。各 task 最弱点估计：text: phi4mm accuracy=0.905000；image: phi4mm top1_accuracy=0.920000；ocr: qwen cer=0.102490；asr: whisper-small wer=0.042181；visual_qa: qwen accuracy=0.900000；lid: qwen accuracy=0.860000。需要逐 task 审查低质量及被支配 path；不能把某 path 的低分或被支配直接变成整模型淘汰，Phi 的 VQA 与 text/image 表现就不同。当前 v2 verified 只表示 capability，不是质量准入。没有正式阈值便不能按临时规则宣告某模型通过/淘汰。quality matrix 的所有分数、invalid counts 和 Pareto 都已披露；最终 optimizer 不能在未冻结质量政策时默认这些 path 全部合格。

## 8. 是否已有正式 quality threshold？

存在历史四任务 placement presets，完整值已保存；当前六任务 v2 optimizer 的统一 threshold 未规定：UNSPECIFIED。VQA/LID 尤其没有历史门槛。没有新增或事后选择 threshold。

## 9. Phi 是否可承担统一 VQA？

VERIFIED。All 200 same-semantic requests completed, valid option outputs observed; quality measured without an accuracy gate。Quality={"n": 200, "accuracy": 0.965, "top1_accuracy": null, "exact_match": null, "cer": null, "wer": null, "invalid_output_count": 0, "errors": null, "reference_units": null}。

## 10. MiniCPM 是否可承担统一 VQA？

VERIFIED。All 200 same-semantic requests completed, valid option outputs observed; quality measured without an accuracy gate。Quality={"n": 200, "accuracy": 0.96, "top1_accuracy": null, "exact_match": null, "cer": null, "wer": null, "invalid_output_count": 0, "errors": null, "reference_units": null}。

## 11. Qwen 是否仍然是唯一 mandatory VQA Generalist？

在 unchanged frozen v2 中是，λVQA>0 时仍强制存在。本次相同语义验证已发现其他 capable Generalists，结构性唯一性可在 v2.1 审批/版本冻结后解除；这不自动说明其他模型符合质量政策。

## 12. 是否需要 candidate_pool_v2.1？

建议建立 v2.1：新增 capability proposals 为 phi4mm/visual_qa, minicpm/visual_qa。只生成 proposal，未改 v2；新 path 的 s0 尚缺、正式质量政策尚未冻结，不能直接注册进完整 optimizer。

## 13. 下一版 suite 应包含哪些 workload？

demand-balanced、speech-heavy、vision-heavy、text/OCR-heavy、reasoning/VQA-heavy、no-VQA phase。所有 λ 从固定 Qwen-reference target demand shares 解析生成，common D 取旧三组总 demand 中位数；JSON/hash 已冻结。没有根据 planning/search result 调参。

## 14. no-VQA phase 语义合理吗？

合理，研究有五类现有任务且没有 ScienceQA 请求的流量阶段。λVQA=0 只改变流量覆盖约束，其他 task semantics 不变；它检验而不保证 Generalist choice 变化；现有 Specialists 已能覆盖全部五个非 VQA task，因此 N_G=0 也无结构性 coverage 障碍。不能将此 phase 当成全六任务 coverage。

## 15. 可以冻结最终 workload matrix 吗？

现在可冻结需求原则、proposal λ 向量与 hash；不能把它们称为最终完整 benchmark matrix。需要先决定 v2.1、新 VQA path standalone demands 和六任务质量政策；正式执行协议可延用现有 10s warmup / 210s window、deterministic periodic、三次重复。本阶段没有启动。后续不能为得到不同 deployment 改 workload。

所有要求产物位于 results/quality_workload_audit/。质量测量与原 deployment search 独立；完成后停止，不进入禁止的后续阶段。
