# whole57 实际可见队列到恢复风险训练数据

`tools/predictor_validation_20261027/whole57_visible_risk_adapter_v30.py` 接通已通过审核的 whole57 原始观察和风险训练样本。它不拟合模型，不运行 GPU，不修改冻结 collector/forecast。推荐实际结果在 `artifacts/predictor-whole57-visible-risk-v30-20261028-training02-light-balanced/` 和 `artifacts/predictor-whole57-visible-risk-v30-20261028-training03-natural-balanced/`；training01 及其 implementation snapshot 原样保留。

运行方式：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m tools.predictor_validation_20261027.whole57_visible_risk_adapter_v30 \
  --review artifacts/predictor-whole57-joint-training-v23-20261028-reviewed01/whole57-joint-light-v23/balanced/report.json \
  --output <新目录>
```

必须是 PASS、training-only 的 whole57 review。adapter 重新 streaming SHA 核验 review/raw/plan 和观察实现，检查全部 57 身份、公开到达、原输入、预声明路由和原 cohort。natural/balanced 的 PASS review 已用同一入口独立提取，没有用 light 的结束标签补数据。

## 特征只来自已收到的观察和完整公开计划

使用 `gsdeploy.forecasting.observed_state` 的 `feedback_from_runtime_records`、`new_observed_state`、`advance_observed_state`。START/PHASE、响应和 next-ready 的可见时刻分别取实际 `execution_events[].received_time_ns`、`response_reader_event_enqueue_time_ns`、`previous_handler_next_request_ready.received_time_ns`。worker perf 时间早于 cutoff 但 reader 尚未收到时，不得提前更新。

每个 cutoff 保存完整 57 条预声明到达、路由、原输入 size/hash、全部未来到达及 1/5/10 秒公开请求数 envelope；这些均来自 plan。当前状态保存 waiting IDs/count、oldest age、按 owner 等待/handler-active/等待 next-ready 的请求、当前收到的 phase 与边界。semantic token length/batch 等只有收到相应 phase 事件后才出现，不能从最终 worker response 中提前读取。

training02 还保留 cutoff 前的父进程 enqueue/dispatch/pipe-write-completed 边界，区分尚未入队、入队未 dispatch、正在 pipe write、write 已完成。write 尚未返回时只给已等待年龄，不读取未来返回时间或最终写耗时。父进程时间边界代表记录的主机事实，不认证在线控制器确实在每个离线 cutoff 执行过一次决策。

waiting 的精确定义是“公开计划已到达，但没有 reader-visible START”；它可能包含尚未实际 enqueue 或 START 通知尚未到达的请求。因此它是 offered/visible backlog，不等同于物理 worker queue。RESULT 已收到而 READY 尚未收到时保留 handoff，不能认为 owner 自动立即可接下一请求。

GPU cursor、DAG ready、执行资源份额均保持 UNKNOWN。缺少 source-bound solo-work provider 时，queue reference work 也保持 UNKNOWN；原输入 size 不是服务毫秒或吞吐。该输出不是 DAG checkpoint，也没有执行 rolling forecast。

## 未来标签隔离且保留右删失

特征写入 `prefix-features.jsonl.gz`，标签单独写入 `offline-risk-labels.jsonl.gz`，用样本 ID 连接。未来完整 raw 仅进入 `OfflineQueueLabels`，不会传回特征。

每个样本产生 1/5/10 秒三个窗口标签，按 reader receipt 的完整队列轨迹统计 waiting peak、queue area、outstanding peak、窗口观察终点。恢复标签是从 cutoff 起第一次 visible waiting queue 为空的时刻；如果当前已空则为零。未来公开到达参与恢复过程，同刻到达与 START 原子处理，不能制造瞬时假清空。尚未恢复时保留 elapsed-or-censor 和 `event_observed=false`，而不是记为“不发生风险”。这既包括窗口截止，也包括采集截止；恢复不表示之后持续稳定。

tail cohort 包含 cutoff 已到达未收到结果的请求，以及未来窗口内预声明到达。每条保存 arrival→reader receipt 或右删失 followup；观察结束前尚未到达的请求另标 `NOT_YET_ARRIVED_AT_OBSERVATION_END`。只有 cohort 全部结果已观察才给精确 P95，空 cohort 不造 P95。whole57 plan 没有预声明 latency violation 阈值，所以 violation 保持 UNKNOWN，不根据新标签事后调阈值。

## 第一组真实结果与验证

light/balanced 的 57 个 measured 请求、1179 个 START/PHASE 事件生成 **1385 个 prefix、4155 个风险窗口标签**。最大 visible waiting 为 **10**；3636 个窗口覆盖完整未来时限，519 个窗口受观察末端截断。2747 个窗口内没有观察到首次恢复，作为 right-censored recovery 保留。4146 个非空 tail cohort 完整，9 个为空；本组请求均完成，因此没有实际未完成的 tail cohort，不能据此宣称 tail censoring 已经实测验证。

natural/balanced 独立产出 **1382 个 prefix、4146 个风险窗口标签**，最大 visible waiting 同为 **10**；3538 个完整未来窗口，608 个末端截断窗口，2503 个恢复右删失窗口。两组都使用同一 v30 schema，并保留各自 source SHA/context。两组的 elapsed、queue 或 tail 不能因 schema 相同便视为同一物理执行域。

九个 adapter 测试覆盖 late receipt、未来 completion 和未收到 token metadata 改变不影响 prefix、公开完整未来与原输入、跨未来到达恢复、同刻原子事件、截断标签、RESULT/READY 分离，以及未来 pipe-write 返回不可见。加公共 observed_state/recovery_risk 回归共 **28 tests PASS**。

另在实际第一个请求上检查：worker START 是 6100114575913876，cutoff 设为 6100114575913877，reader receipt 是 6100114576086472；该请求仍在 waiting。改变未来 worker END/response receipt 后，该 cutoff 的整个 prefix 完全相等。遍历实际产物确认 1385 个 prefix 均保留完整 57 公共请求和 UNKNOWN GPUcursor，4155 个标签的 waiting peak 均不少于其当前 visible waiting。

`split-manifest.json` 绑定 raw SHA 的 whole run_id，group 绑定公开 recipe/input family；禁止按请求或相关窗口随机训练/留出。当前有 light、natural 两个完整 run，recipe/input group 相同，execution context 不同，每个 context 仅一个 run，无法支撑独立风险拟合。旧 `recovery_risk.v2` 还要求至少 3 个训练 run、2 个 group；本次 v30 扩展特征 schema 也需先冻结显式投影及目标协议，不能直接绕过旧 fitter 的严格契约。最终交付的 `whole-run-dataset-manifest.json` 保存两组独立来源与不足原因。后续按独立整 run/输入家族留出验证，再考虑模型拟合与 rolling-forecast 接入，不把这次数据链完成当作预测准确率或分流收益。
