# Module 2D：Cross-deployment Generalization Gate

**判定：PASS**。

cross-deployment feature representation appears viable；仅支持进入正式 predictor development，不表示已完成正式 predictor。

数据：69 个真实 runs、463 个 path rows、24 个 configuration、7 个 deployment contexts。
主评估：391 rows / 57 runs，按已有 stable==true；严格敏感性：333 rows / 54 runs。
行不是独立 deployment samples；七个完整 deployment folds 才是跨 deployment 检查单位。未采集新数据，未训练或导出正式 predictor。

## 设计与泄漏审计

- LODO：held-out deployment 的全部 configuration/run/repetition/task 被排除；训练、测试 group 交集全部为空。repetition 数字/seed 跨配置相同不构成 run 重叠。
- 仅用 shared FeatureExtractor 从 x/r/lambda 与独立 offline profiles 重算特征。无 exact deployment ID one-hot，无实时 queue/latency/completion/GPU 输入。
- B：u + 六维 q。C：B + C resource pressure + standalone path/resource fingerprints + numeric model/deployment memory/count。
- Standalone s0 是独立 offline 校准，允许所有 folds 使用；joint held-out labels 不参与校准。每个 fold 单独 fit median/missing indicators、常数过滤和标准化，记录全部参数。
- Ridge alpha=10；GBT 50 trees、depth=2、min_samples_leaf=10、learning_rate=.05、Huber loss，固定 seed。直接 ms 与 log(L/s0) 各比较一次，无调参、无模型家族择优。
- 训练权重依次按 configuration / run / path 平衡。报告每 fold 的 row metrics 与 configuration-macro metrics；所有 deployment 等权 macro。
- 直接预测若为负仅按预先规定的 1e-6 ms floor；次数写入 fold metrics。normalized 输出按 s0*exp(yhat) 恢复，不按 held-out labels 限幅。
- A 为 standalone s0；另运行 training-only controlled piecewise-u，缺失曲线时明确 fallback 到 s0。未导入全数据构建的 load curve。

## 每 deployment 的主结果

下表为 MAE ms（normalized target）；median AE、sMAPE、P90 AE、Spearman、direct-target variants 和严格敏感性完整保存在 fold_metrics.json。

| Held-out deployment | rows | A s0 | B Ridge | C Ridge | B GBT | C GBT |
|---|---:|---:|---:|---:|---:|---:|
| qwen+whisper | 20 | 61.3 | 86.5 | 49.7 | 61.0 | 37.0 |
| qwen+resnet50 | 21 | 68.8 | 111.2 | 65.1 | 116.8 | 74.2 |
| qwen | 72 | 141.8 | 129.6 | 103.5 | 131.4 | 73.3 |
| distilbert+paddleocr+qwen+resnet50+whisper | 123 | 92.3 | 84.5 | 53.5 | 84.7 | 46.2 |
| paddleocr+qwen | 36 | 99.5 | 108.6 | 58.3 | 104.6 | 57.9 |
| distilbert+qwen | 38 | 66.0 | 72.2 | 33.7 | 59.3 | 30.1 |
| distilbert+paddleocr+qwen+whisper | 81 | 162.6 | 147.2 | 84.9 | 152.6 | 80.1 |

| Variant | macro MAE ms | macro median AE ms | macro sMAPE % | macro P90 AE ms | macro Spearman |
|---|---:|---:|---:|---:|---:|
| A_standalone | 98.899 | 65.931 | 55.800 | 211.059 | 0.821 |
| A_piecewise_or_standalone | 98.899 | 65.931 | 55.800 | 211.059 | 0.821 |
| B_ridge_direct_ms | 159.222 | 95.450 | 63.633 | 438.498 | 0.485 |
| B_ridge_log_ratio | 105.670 | 64.247 | 38.725 | 274.488 | 0.832 |
| B_gbt_direct_ms | 123.473 | 69.903 | 53.136 | 299.240 | 0.639 |
| B_gbt_log_ratio | 101.487 | 47.836 | 32.060 | 282.529 | 0.859 |
| C_ridge_direct_ms | 49.932 | 38.806 | 31.010 | 106.344 | 0.886 |
| C_ridge_log_ratio | 64.097 | 45.833 | 30.171 | 135.673 | 0.904 |
| C_gbt_direct_ms | 57.232 | 41.835 | 37.940 | 115.890 | 0.900 |
| C_gbt_log_ratio | 56.964 | 44.057 | 26.718 | 131.714 | 0.891 |

## Decision gate 的逐 fold 证据

预先固定实质改善为 C 的 MAE 同时低于 A/B 至少 10%，且 sMAPE 不高于两者。Collapse 为 C MAE 超过较低 comparator 的 2 倍且增加 >50 ms。
PASS 要求两种 normalized 家族在主评估与严格敏感性均多数 folds 改善、无 collapse、macro 改善。PARTIAL 要求至少一个家族/子集在至少两个 folds 实质改善；否则 FAIL。规则是描述性工程检查，不是统计显著性检验。

- primary/ridge：实质改善 5/7；collapse 0/7；macro 条件=True。
- primary/gbt：实质改善 6/7；collapse 0/7；macro 条件=True。
- strict/ridge：实质改善 4/7；collapse 0/7；macro 条件=True。
- strict/gbt：实质改善 5/7；collapse 0/7；macro 条件=True。

直接 target 与 normalized target 都已报告；gate 使用预先指定的 normalized target，两种模型家族共同检查，不能用较好的 pooled average 掩盖失败 fold。

现有 controlled piecewise-u 在本次 held-out contexts 的覆盖为 0：主评估 427/427 行均 insufficient_profile_data，明确回退 s0。它在这里没有已测曲线外的泛化证据，不能冒充独立有效的插值比较。
严格敏感性的 macro MAE：A=94.4 ms；B/C Ridge=89.6/65.2 ms；B/C GBT=76.7/53.0 ms。

## 稳定性分布与适用边界

| Deployment | stable runs | boundary runs | unstable runs |
|---|---:|---:|---:|
| qwen+whisper | 4 | 0 | 0 |
| qwen+resnet50 | 3 | 0 | 0 |
| qwen | 12 | 0 | 9 |
| distilbert+paddleocr+qwen+resnet50+whisper | 13 | 0 | 3 |
| paddleocr+qwen | 6 | 0 | 0 |
| distilbert+qwen | 7 | 0 | 0 |
| distilbert+paddleocr+qwen+whisper | 12 | 0 | 0 |

现有 stable boolean 包含 boundary 类别；严格敏感性进一步排除 boundary、smoke/smoke_gate 和 path completed<30。未按 latency 大小删除样本。
标签审计另发现 0 个 stable=true run 在窗口中途曾有大幅 backlog，随后恢复；见 stability_distribution.json。此诊断不参与特征或重新筛选。
稳定标签是有限观察窗口内的 operational label，不保证稳态排队或低 latency。不同协议的窗口长短、任务池和运行环境差异可能混杂结果；本轮不把协议或运行后指标加入输入修补。
unstable 仅见于 Qwen-only context；因此还不足以验证跨 deployment stability classifier。吞吐受 offered workload 限制，且没有全面受控 saturation 数据，不能据此宣称 throughput predictor 已就绪。

## 错误诊断

- C_ridge_log_ratio 最大错误：qwen，asr/qwen，u=0.1000；真实 1290.6 ms，预测 578.4 ms，s0=654.1 ms。run=A_u_0.10_rep_2。
  own_load_band 中平均绝对误差最大组：u>.4，MAE=106.1 ms，n=36。
  task_mix 中平均绝对误差最大组：mixed，MAE=86.3 ms，n=279。
  active_contenders 中平均绝对误差最大组：none，MAE=103.3 ms，n=90。
  stability_category 中平均绝对误差最大组：stable，MAE=68.1 ms，n=391。
- C_gbt_log_ratio 最大错误：qwen，asr/qwen，u=0.1000；真实 1290.6 ms，预测 816.0 ms，s0=654.1 ms。run=A_u_0.10_rep_2。
  own_load_band 中平均绝对误差最大组：u>.4，MAE=92.3 ms，n=36。
  task_mix 中平均绝对误差最大组：mixed，MAE=74.5 ms，n=279。
  active_contenders 中平均绝对误差最大组：resnet50，MAE=86.2 ms，n=18。
  stability_category 中平均绝对误差最大组：stable，MAE=58.7 ms，n=391。
- 相同 planning configuration 重复测量仍可能差异很大：cfg_5b8ca666a062c60a / asr/qwen，3 runs，范围 813.1–1290.6 ms，sample std=250.6 ms。协议和 run source 保留在 error_analysis.json。
- 最大错误 run 的已有 stable label 需要特别解释：中途最高 bin mean backlog=0.6，窗口结束 backlog=0，drain=0.001 s，completion=1.000。标签捕捉到最终恢复，却未排除瞬时严重排队。严格子集仍保留该 run，因为没有事后按 latency 剔除。

这些 breakdown 的样本数和 task/model 组成不匹配，只描述关联，不能将某个 contender 或 resource dimension 解释为已证明原因。每个最坏 row 的 u/q/C、真值、预测、s0 与原始 source 路径完整保留。
所以 PARTIAL 是在现有标签定义下的 gate 结论；不能把 Qwen-only 失败全部归因于 feature representation。正式稳定区建模前，需先统一对窗口中途排队的标签语义。这一轮仅记录问题，没有更改历史标签。
七个 contexts 都包含 Qwen，多个 context 只有 1–2 个配置；task-mix、load、co-location 与 deployment 并未充分交叉。LODO 的实际含义是这些已观察 context 的留出检验，不是全 model space 泛化证明。

## 产物与停止边界

- fold_metrics.json：每 fold / 每模型 / 两个 subsets 的五项误差、configuration macro、部署 macro 与 gate。
- predictions.jsonl：逐 held-out row 的预测及四级 grouping IDs、seed、source hash/path。
- leakage_audit.json：训练/留出组、预处理 fit 参数、source hashes、特征 schema、排除原因、原数据完整性。
- error_analysis.json：最大错误、分层诊断和同配置重复差异。
- evaluation_config.json / stability_distribution.json：固定方案与全部 69 runs 的标签统计。
本阶段完成后停止：未启动 profiling、guided search、routing optimization、最终 predictor、stability/throughput training 或 forecasting。
