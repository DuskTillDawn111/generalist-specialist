# Module 2F：Decision-Focused Ranking Gate

**INSUFFICIENT COMPARABLE GROUPS**。

当前数据不足以判定 cross-deployment scorer 的 PASS / PARTIAL / FAIL；没有进入 guided search。
这不是把 predictor 判为 FAIL，而是缺少合法、足够的候选比较组。已有同 deployment routing 组的排序结果单独报告。

## 可比性审计

检查69个真实 runs / 24个 configuration / 7个 deployment；形成 20 个精确 workload + 兼容 protocol 单元。仅 2 个单元在筛选前有两个以上配置。

- 比较键包含完整 task arrival rates、窗口/warmup/drain/队列限制、arrival law、task-pool/source/harness hashes、设备/环境及 clock 定义。不同 workload、只相同 u、不同持续时间均不拼组。
- 唯一协议等价例外：deployment 引入/移除 worker；所有共同 worker 的实现和解释器必须相同。workers membership 不应把真实 deployment 对照人为拆开。
- Strict：同配置同协议的全部重复必须 steady。Extended：全部 steady/recoverable。任一 unstable 重复都排除该配置，不能只保留恢复较好的 runs。
- 各候选取共同 seed+initial-phase block（至少2个）；逐 run 计算 objective，再等权平均。重复不是独立 ranking groups。

| Analysis | 可排名组 | 跨 deployment 组 |
|---|---:|---:|
| strict | 1 | 0 |
| extended | 1 | 0 |

可用组来自 Stage 2B-1：固定全部5个模型，workload=(text=2,image=1,ocr=1,asr=.3,VQA=.5,lid=1) req/s，100秒窗口。Strict 有4种 routing，Extended 有5种 routing；各3个配对 repetitions。它们测试的是一个 unseen deployment 内的 routing 排序。
唯一原始跨 deployment 对照为 A_u_0.40 的 Qwen-only 与 C_no_active_contender 的四模型驻留配置；workload、183秒窗口和共同实现一致，但 Qwen-only 包含 unstable repetition，因此不能作为普通 latency ranking ground truth。其余相近特征点的 workload 不同。
即便仅留下该对照的非 unstable 重复，也只有一个2-candidate组，Top-2覆盖必然100%，仍不足以验证 cross-deployment shortlisting。本轮未将它纳入主 ranking。

## 已保存 OOF 预测与目标口径

仅复用 Module 2D primary subset 的 A_standalone、B_gbt_log_ratio、C_gbt_log_ratio；核验 1281 条 path predictions 的 heldout deployment、训练组隔离、目标值和 source hash。未重新 fit 模型，未使用 Module 2E 校准输出。
真实 mean/P95 使用原始 run 的 task aggregate，已对 request records 重算核验；不会把 service-path P95 按 routing 权重平均。预测 task mean 使用 r_km/lambda_k 混合，再与真值一样按 lambda_k 加权。
J_tail 是 task P95 的 workload-weighted平均，再跨重复取均值；既不是 system P95，也不是跨所有重复 pooled P95。
**没有 P95 predictor**：所有 predicted_tail_objective_ms 为 null。Tail 以下仅评估已有 mean score 排序对测量 tail 的 proxy baseline，不能称作 tail prediction 能力。

## 单个可用 routing 组的结果

| Analysis / objective | score | Kendall tau | Spearman rho | Top-1 hit | Top-2 recall | Top-2 coverage | Top-3 coverage | Regret |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| strict / mean | A | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| strict / tail_mean_score_proxy | A | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| strict / mean | B | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| strict / tail_mean_score_proxy | B | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| strict / mean | C_GBT | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| strict / tail_mean_score_proxy | C_GBT | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| extended / mean | A | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| extended / tail_mean_score_proxy | A | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| extended / mean | B | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| extended / tail_mean_score_proxy | B | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| extended / mean | C_GBT | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| extended / tail_mean_score_proxy | C_GBT | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

每个 analysis 只有一个 workload/protocol 组，所以 mean / median / P90 / worst regret 的 group aggregate 均等于该单组 regret；不能把 strict/extended 或3次重复扩充为独立样本数。完整 per-repetition 结果、随机选择基线和非平凡 shortlist coverage 保存在 ranking_metrics.json。
A/B/C 在这个组上均为零 regret；C 没有显示出优于 A/B 的 ranking benefit。单个 workload 下相同 deployment 的成功排序不足以支持跨 deployment scorer。

候选的真实目标与保存的 mean score（ms，跨共同 seeds 等权平均；P95 列为测量值）：

| Routing audit label | measured mean J | A mean score | B mean score | C mean score | measured tail J | state |
|---|---:|---:|---:|---:|---:|---|
| share_1.00 | 43.34 | 30.26 | 45.91 | 42.60 | 81.05 | steady |
| share_0.75 | 103.78 | 54.24 | 105.25 | 100.18 | 303.25 | steady |
| share_0.50 | 172.87 | 78.23 | 173.73 | 161.92 | 559.04 | steady |
| share_0.25 | 297.36 | 102.21 | 235.36 | 225.31 | 869.67 | steady |
| share_0.00 | 770.24 | 126.19 | 295.19 | 287.36 | 2053.13 | recoverable_congested |

这些 routing audit labels 仅用于识别既有实验；完整正式变量 x/r/lambda 保存在结果中。

## 最大 regret 与局限

可用组最大 mean regret=0.0000（strict/A）。同一配置族的 Qwen load、task mix 与 contender workload 随 routing 同时改变，无法将误差单独归因于某一因素。
Strict 不含 recoverable 候选；Extended 增加的 generalist-first routing 带有 recoverable congestion。Qwen-only 候选因不稳定未参加排名，不能在本轮归因其 ranking failure。
regret_analysis.json 保存 workload、完整 deployment/routing、真实/预测目标、两种 ranking、u/q/C、稳定性与全部来源。
未来要判定 deployment coarse scorer，至少需要同 workload/协议下多个稳定 deployment 候选，并覆盖多个独立 workload group；Top-K 还需候选数大于 K 才有辨别力。当前未采集这些数据。

## 产物与停止边界

- comparison_groups.json：所有协议/workload 单元、候选、筛除原因、配对 blocks。
- ranking_metrics.json：每组 Kendall/Spearman/Top-K/regret 与 scope-separated aggregates。
- ranking_predictions.jsonl：A/B/C 的配置级 OOF mean objective，P95 prediction 明确 null。
- regret_analysis.json：最大 regret 的完整配置/feature 上下文。
- ranking_leakage_audit.json：OOF provenance、fold 隔离及原文件哈希。
完成后停止：没有新 profiling、模型训练、local calibration 或 guided deployment search。
