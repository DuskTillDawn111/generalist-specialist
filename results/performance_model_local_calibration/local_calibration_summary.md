# Module 2E：Stability Semantics + Few-shot Local Calibration

**Decision Gate：FAIL**。

当前 bias-only 校准没有稳定收益；不继续增加 calibration model 复杂度，可考虑 setup-local model + guided profiling，本轮不执行。

## A. 三态语义与时间线

保持全部原始 result、stable 字段及 v1 stability 对象不变。新增 run-level stability_state_v2 / stability_reason_v2；dataset_rows_v2.jsonl 同时保留原始 row 与新增标签。
B(t) 使用 actual_submit_time_ns（host send-start）减成功完成，不等价于 worker ACK。W(t) 对每个已提交未完成请求求完整 standalone s0 之和，单位 standalone service-seconds，不是剩余 GPU 工作量。
同一单调时钟上的逐请求事件精确重建峰值及 bin 时间积分；输出窗口末值与 drain 后最终值。scheduled-arrival backlog 单列，不能与提交 backlog 混用。
硬不稳定：incomplete、drain timeout、completion<0.98、worker/run error 或有非零末积压的持续 tail 增长。即便最后 drain 完成，持续增长的 tail 仍可 unstable。
recoverable_congested：完整恢复但达到 backlog>=30、W>=5 s、queue wait>=1 s 的请求数达到 max(10,5% offered)、或 drain>=5 s 中任一警戒条件。steady 为其余完整完成且未触警戒的运行。
tail 使用最后 25% 的 20-bin 时间线；同时要求 slope>=.02 requests/s、growth>=5、>=60% 相邻 bin 增长、end backlog>=5。没有单一 E2E latency 阈值。

| Warning 倍率 | steady runs | recoverable runs | unstable runs | steady rows | extended rows |
|---|---:|---:|---:|---:|---:|
| 0.75 | 53 | 10 | 6 | 356 | 427 |
| 1.00 | 57 | 6 | 6 | 391 | 427 |
| 1.25 | 57 | 6 | 6 | 391 | 427 |

仅缩放 congestion warning 的四个幅度阈值；hard instability 条件固定。每个 warning 设置都重做相同 LODO/校准流程，未调参。
先前异常 B_text_dominant_rep_2 现在为 recoverable_congested：实际提交 backlog 峰值 158；旧诊断 >600 是 scheduled backlog 的 bin mean。两者定义不同，原始数据未被修改。

## B. 校准选择与防泄漏

- 外层完整 deployment LODO，global 使用其他部署的 v2 steady rows；固定 Module 2D GBT 参数、41 个 planning features 和 log(L/s0) target。strict 与 extended 共享同一个 global fit，差异只在评估区域。
- preprocessing、distance normalization 仅 fit 在 training deployment 数据。新 deployment 的任何标签都不进入 global training。
- planning safety screen：单 resident 每模型 u<=.20，多 resident 每模型 u<=.40；只是保守筛查，不能证明 steady。排序用 planning load、global prediction 和 feature distance，不读 test labels。
- 最多三个不同起始 anchor，每个做 farthest-first nested selection。选中 probe 才揭示标签；若该配置任一 repetition 非 steady，拒绝其 residual，但仍计入已花费 K，不用 test labels 找替补。
- primary correction 只有 b=median(log(L/s0)-global prediction)。使用所选配置的全部现有 repetitions/path rows；K 是配置数，真实测量成本还包含重复次数和窗口秒数。无 affine fitting。
- 每个 trial 预留整个最大 calibration chain；余下 configuration 固定作为 test。因此同 trial 的 K=0/1/2/3 使用完全相同 test rows，未用到的 reserve 也不能混入 test。
- 一个配置的所有 repetitions/tasks 始终在同一侧。trial 之间可能复用配置，因此不是独立实验，也不报告统计显著性。

| Deployment | total configs | planning-safe configs | max K with independent test | trials |
|---|---:|---:|---:|---:|
| qwen+whisper | 2 | 2 | 1 | 2 |
| qwen+resnet50 | 1 | 1 | 0 | 0 |
| qwen | 7 | 2 | 2 | 2 |
| distilbert+paddleocr+qwen+resnet50+whisper | 5 | 2 | 2 | 2 |
| paddleocr+qwen | 2 | 2 | 1 | 2 |
| distilbert+qwen | 3 | 3 | 2 | 3 |
| distilbert+paddleocr+qwen+whisper | 4 | 4 | 3 | 3 |

Qwen+ResNet50 只有一个配置，所有 K>0 不可独立评估；绝不拆 repetitions 凑出 calibration/test。K=3 的支持范围很窄，不能把不同 fold 组成的 macro 差异当作预算收益。

## C. 配对评估结果

下表为 nominal thresholds 下、同一 test 配置上的 deployment-macro MAE。每个 deployment 先对所有 deterministic trials 取平均，再等权 macro。

| Evaluation region | K | supported deployments | matched Global MAE ms | Bias-only MAE ms |
|---|---:|---:|---:|---:|
| strict | 0 | 6 | 57.76 | 57.76 |
| strict | 1 | 6 | 57.76 | 71.20 |
| strict | 2 | 4 | 58.55 | 75.66 |
| strict | 3 | 1 | 77.56 | 98.18 |
| extended | 0 | 6 | 828.06 | 828.06 |
| extended | 1 | 6 | 828.06 | 836.05 |
| extended | 2 | 4 | 1214.00 | 1221.12 |
| extended | 3 | 1 | 77.56 | 98.18 |

不预留 calibration 的全数据 K=0 macro 单独保存为 global_K0_all_macro；不能拿它与上表改变了 test 支持集的 K>0 直接比较。

本轮也不能与 Module 2D 的 449.8 ms 直接比较：v2 改变了健康区标签、global training subset 和本地 calibration/test 支持集；数值变小并不是重新调优模型的证据。

每 deployment 的 strict MAE（mean ± trial sample std；括号为 worst trial）：

| Deployment | K | matched Global mean | local mean ± std (worst) | result |
|---|---:|---:|---:|---|
| qwen+whisper | 1 | 56.82 | 58.81 ± 42.07 (88.56) | mixed/small change |
| qwen | 1 | 69.35 | 83.97 ± 20.66 (98.58) | degrades |
| distilbert+paddleocr+qwen+resnet50+whisper | 1 | 57.80 | 81.59 ± 12.16 (90.18) | degrades |
| paddleocr+qwen | 1 | 55.54 | 76.82 ± 49.58 (111.87) | degrades |
| distilbert+qwen | 1 | 29.50 | 29.65 ± 19.80 (52.17) | mixed/small change |
| distilbert+paddleocr+qwen+whisper | 1 | 77.56 | 96.36 ± 2.54 (97.84) | degrades |
| qwen | 2 | 69.35 | 98.58 ± 0.00 (98.58) | degrades |
| distilbert+paddleocr+qwen+resnet50+whisper | 2 | 57.80 | 74.98 ± 0.00 (74.98) | degrades |
| distilbert+qwen | 2 | 29.50 | 31.90 ± 17.55 (52.17) | mixed/small change |
| distilbert+paddleocr+qwen+whisper | 2 | 77.56 | 97.19 ± 0.03 (97.22) | degrades |
| distilbert+paddleocr+qwen+whisper | 3 | 77.56 | 98.18 ± 1.18 (99.54) | degrades |

MAE、median AE、sMAPE、P90 AE、Spearman 的每 trial、每 deployment、macro、sample std 和 worst-case 全部保留在 calibration_fold_metrics.json。以下是相同 trial/test 支持下的边际 MAE 减少，正数才是收益：

- strict K=0→1：-13.43 ms，6 deployments。
- strict K=1→2：-2.77 ms，4 deployments。
- strict K=2→3：-0.99 ms，1 deployments。
- extended K=0→1：-7.99 ms，6 deployments。
- extended K=1→2：-0.96 ms，4 deployments。
- extended K=2→3：-0.99 ms，1 deployments。

## D. 原失败 context、退化与 decision gate

原来 GBT 的两个非改善 context 是：qwen+resnet50、qwen。这不同于 Module 2D 的 Ridge collapse contexts。
其中单配置 context 的修复能力无法验证。PASS 必须修复两个原失败 context、覆盖至少五个 strict folds、macro MAE 至少下降10%、sMAPE 不升、至少80% folds无明显退化且无最坏 trial collapse；extended不能显著退化，三个阈值设置一致。
PARTIAL 要求 nominal 某个 K 在 strict 或 extended 至少两个 folds 明显改善；否则 FAIL。阈值是预先固定的工程 gate，不是统计显著性检验。

- warning_075, K=1：strict 改善 0/6；明显退化 5；最坏 trial collapse 0；原失败修复 0/2；extended 改善 0。
- warning_075, K=2：strict 改善 0/4；明显退化 3；最坏 trial collapse 0；原失败修复 0/2；extended 改善 0。
- warning_075, K=3：strict 改善 0/1；明显退化 1；最坏 trial collapse 0；原失败修复 0/2；extended 改善 0。
- nominal, K=1：strict 改善 0/6；明显退化 4；最坏 trial collapse 0；原失败修复 0/2；extended 改善 0。
- nominal, K=2：strict 改善 0/4；明显退化 3；最坏 trial collapse 0；原失败修复 0/2；extended 改善 0。
- nominal, K=3：strict 改善 0/1；明显退化 1；最坏 trial collapse 0；原失败修复 0/2；extended 改善 0。
- warning_125, K=1：strict 改善 0/6；明显退化 4；最坏 trial collapse 0；原失败修复 0/2；extended 改善 0。
- warning_125, K=2：strict 改善 0/4；明显退化 3；最坏 trial collapse 0；原失败修复 0/2；extended 改善 0。
- warning_125, K=3：strict 改善 0/1；明显退化 1；最坏 trial collapse 0；原失败修复 0/2；extended 改善 0。

## E. 结论与限制

当前数据下 K=0 更合算：1–3 shot 未在至少三个 contexts 上提供正 macro MAE 收益，还增加配置测量成本。没有证据支持支付 bias-only local calibration 的成本。
Strict 衡量健康稳定区；Extended 加入可恢复拥塞且仍排除 unstable。Global 仅训练 steady，所以 extended 也检验健康区模型向拥塞区外推，不能把失败自动归因于校准算法。
宏观结论不能用最好的 anchor 掩盖最坏 trial。部分 probe 会在揭示三态标签后被拒绝，仍计入成本；该机制的收益/损失也计入结果。
现有数据覆盖稀疏、每个 deployment 的配置数不等，并有不同 measurement protocols。此为回顾性 few-shot simulation，尚无新部署的前瞻性验证。
告警阈值敏感性只覆盖0.75–1.25倍的局部范围；不表示阈值是客观稳定性真理。依据 request records 的 operational state 也不是严格排队稳定性的数学证明。

## 文件与边界

- stability_v2_labels.jsonl / dataset_rows_v2.jsonl：新增标签、原始 stable、source hashes 与 group IDs。
- stability_threshold_sensitivity.json：完整阈值、三态分布与 deployment 分布。
- calibration_fold_metrics.json：外层 folds、预算/试验、完整五项指标与敏感性 gate。
- calibration_predictions.jsonl：逐测试 row、global/local 配对预测与四级分组。
- calibration_selection_audit.json：选择特征/距离/安全筛查、global fit 组、预处理、probe reveal 与 test 隔离。
仅 CPU 读取旧结果；没有运行 worker、新增 profiling、deployment search、routing optimizer、最终 predictor training 或 forecasting。完成后停止。
