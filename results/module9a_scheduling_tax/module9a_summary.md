# Module 9A 第一版完成：选择 I0

保留全部 21 个 admitted task-path 的 offline profiling；Roomie spatial occupancy predictor 已退出当前路线。未启用 MPS，未修改 native worker/runtime、v2、RC-SG 或冻结结果。

模型为 `T_hat[p <- q] = T_p + n_p * delta_q`。`delta_q` 属于 co-resident model-task path；对多个 calibration targets 的归一化额外延迟取中位数，没有 target fitted tax、pair-specific coefficient 或 target susceptibility。

6 个 calibration identities、6 个 validation identities，各 3 次独立进程重复；8 个 matched standalone paths 各 3 次。每个 co-path 的 calibration/validation target paths 完全分离。所有 repetitions 留在同一个 split；旧的混合 co-task 测量不作为纯 co-path 数据复用。

## 参数与验证

| co-path | delta (ms / target kernel) | process-refit 95% CI |
|---|---:|---|
| whisper-small__asr | 0.021495395 | [0.012601336342194009, 0.03003638316635556] |
| whisper-small__lid | 0.0270987702 | [0.014915281967057952, 0.04185963330434817] |
| qwen__visual_qa | 0.0240374607 | [0.003383291126814251, 0.043413267568092016] |

| Model | MAE (slowdown) | RMSE | Spearman | Kendall | latency MAE (ms) |
|---|---:|---:|---:|---:|---:|
| I0 | 0.1468 | 0.19911 | None | None | 31.7776 |
| I1 | 0.180202 | 0.229324 | -0.08827348295047496 | -0.07161148740394328 | 54.1134 |
| I2 | 0.15964 | 0.220309 | -0.028571428571428574 | 0.06666666666666665 | 48.1037 |

冻结规则选中 **I0**；eligible models = ['I0']。所有更复杂模型必须相对每个更简单模型具有严格负的 paired error-difference 95% CI，才能参与最低 MAE 选择。没有百分比 improvement threshold，没有结果出来后的调参、追加 identities 或特征。

- I1-I0: mean paired MAE difference = 0.0334018, 95% CI = [-0.1563579579034397, 0.24279691746972096].
- I2-I0: mean paired MAE difference = 0.0128394, 95% CI = [-0.18812758574467195, 0.2344627192218287].
- I2-I1: mean paired MAE difference = -0.0205624, 95% CI = [-0.049517344857040256, 0.0019894392149211737].

## 范围与限制

I0 同时具有最低整体 MAE 和 RMSE；I1/I2 相对 I0 的 paired CI 均跨零，I2 相对 I1 的 CI 也跨零。结论是未获得整体迁移改善支持，不是证明所有 co-pressure 恒为零。I2 的 Generalist-target MAE 为 0.109066（I0 为 0.185089），Specialist-target MAE 为 0.260787（I0 为 0.0702235）。Whisper-turbo ASR/LID 在同一 Qwen VQA co-path 下的 ordering 被 I1/I2 反向预测。不得用 subgroup 的较低误差宣称整体模型成功。

统计比较按 pair identity bootstrap，条件于已拟合的 calibration。多个 identities 共享 target/co-path，不能声称独立 co-model population 的显著性。`uncertainty_sensitivity.json` 另报进程重复/重拟合与 target、co-path cluster 检查；这些审计不重新选择模型。

第一版只校准 3/21 co-paths（2 个 co-models），验证 4 个独立于 calibration 的 target paths。21/21 profiling coverage 不代表 21/21 co-pressure calibration coverage。未校准 coefficient 使用明确标记的 zero fallback，不 hard-prune candidate。I1 对未测 co-task 的 model-level extrapolation 也不等于该 co-task 已被验证。

delta 是观察到的 handler-latency residual proxy，可能混合 CPU、时钟和 temporal scheduling 影响；不是因果分离的 context-switch duration，不假定固定时间片、FCFS、round-robin 或 kernel alternation。每个 n_p 只来自三个 offline samples。

routing evaluator 使用 baseline-demand-weighted co-task mixture、co-model busy proxy 和不同 co-model 的 additive contributions。Own replicas 仅由 empirical g 表示。busy probability、3+ models 的 additive aggregation 和多 replica co-pressure 尚无 GPU validation；本轮不追加 campaign。

CPU pair median = 0.0007 ms；21-path/11-model evaluate median = 0.395749 ms，p95 = 0.419727 ms。驻留 estimator memory = 25363 bytes。每次评价均不启动 GPU/profiler。

借用 iGniter 的 kernel-count × extra-scheduling-delay decomposition（Sec. III-A Eq. 5–6），没有复现其 MPS spatial provisioning。co-path pressure 与 routing-conditioned aggregation 是项目 adaptation/extension。来源：[iGniter](https://arxiv.org/html/2211.01713v1)。

9A 在模型选择后结束，不继续增加特征。按最新用户指示随后进入固定 deployment/workload 的 9B bounded routing 阶段；不进入 9C，不把 CPU surrogate 改善声称为 serving/E2E 改善。
