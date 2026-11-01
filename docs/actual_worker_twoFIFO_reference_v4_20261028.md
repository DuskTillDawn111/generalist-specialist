# Actual worker two-FIFO reference v4（CPU）

新增 `tools/predictor_validation_20261027/actual_worker_twoFIFO_reference_v4.py`，
从 training05 冻结的 `macro_protocol_timing_reference_v3.py` 复制隔离实现。
不修改 v3、worker、collector、模型、原 scorer 或既有报告。
冻结 v3 report：`artifacts/predictor-joint-request-protocol-transport-v3-natural-20261028-training05/report.json`，
SHA256 `8aa40984ad9fe2698e3f415a851981bdeacd70cba2b8aacd4cdc09c6676bf520`。

v4 在完整原到达窗口内接收所有 public requests，以 `(owner,task,pool_id,arrival ordinal within key)`
关联显式联合 response plan 和 TransportDraw。sender 按实际 enqueue 顺序发送；同 submit 时间的请求保留 sender FIFO。
worker 串行执行 stdin FIFO，在前一实际 READY occurrence 后，按 `max(READY,line_available)+worker_pre_START` 启动。
父端 RESULT reader、共享 collector processing 和 phase visibility 独立推进，不再占用 worker 至 READY processing。
worker 的实际 macro/request/phase/template identity 与父端可见证据分开保存。

`simulate` 保留 v3 的完整 arrival、原测量窗口、固定 drain、cutoff、RESULT reader E2E endpoint、
实际 macro exposure、待处理通知和 phase visibility。新增完整 public request/draw 输出、sender schedule、
pending enqueue/sender、worker FIFO/reserved START、actual worker release clocks。
`maximum_parallel_handlers` 统计实际 active handlers；reserved-or-active 的计数单列。
实际 exposure 补充 phase、task、pool 和 public-key ordinal，可用于全局 resource timeline。

caller 必须显式给定：

- `transport_draws[(owner,task,pool_id,ordinal)] = TransportDraw(...)`，每个 offered public-key ordinal 恰好一个。
- 每个 offered owner 的 `initial_sender_available_ns`、`initial_worker_read_release_ns`。
- `initial_protocol_state`：`reader_free`、`collector_free`、`last_collector_arrival`、`pending_messages=()`。
  目前只支持 caller 显式指定无既有 pending message 的世界；这不是历史初始状态认证。
- 每个 RequestPlan/mark 的联合 protocol station slacks，以及明确的正 response rates。

缺传输或初始状态会报 UNKNOWN。每 owner 必须具有一个跨请求一致、与其他 owner 区别的 stdout producer；
所有 mark/READY 使用该 producer。v3 的旧 `dispatch_to_worker_start_ns` 是 idle composite，
v4 拒绝非零值，避免与 supplied TransportDraw 双计。
调用 v3 `to_reference` 后需显式 `replace(plan,dispatch_to_worker_start_ns=0)`，
并将同一份 slacks 绑定到这个新 plan；此操作只保留 response/protocol 部分，没有自动学习 transport 分解。

`resource_timeline_view(reference)` 是显式 schema adapter，供 root 的 `macro_resource_timeline_v1.timeline` 使用。
它保留 `origin_schema=v4`、所有资格标志，并明确 `schema_adapter_confers_native_qualification=False`。
`macro_six_term_bridge_v1.whole_score` 本身可读取 v4 相同的完整 cohort、窗口和 RESULT reader 字段；
没有修改六等权、3600s amortization 或任何 scorer 算术。资源和质量值仍必须由 caller 独立供应。
`dispatch_contract_matches_original_sender=True` 只说明此显式世界使用 source 支持的两个 FIFO 顺序，
不认证 transport 分布、initial state、原生数值或预测准确性。

CPU 报告：`artifacts/predictor-actual-worker-twoFIFO-reference-v4-20261028-cpu01/report.json`，
SHA256 `bb79563e140e4935c9e95a5f100bd8fdb63a9b36aa90ff0f7b9253e767ec13af`，616 个 source receipts 全部复核。
`actual_worker_twoFIFO_review_v4.py` 可以用 exclusive output 重建该报告。

报告包含三种基本反例：首请求 START=1/READY=11，次请求 START=12，首 RESULT reader=19、processing=119；
sender write record=20 时次 START=21；cutoff=5 时完整保留未完成请求与各 pending 队列。
额外测试对同一 notification station world 比较冻结 v3 的替代 policy：
两者首 START 都为 1；v3 下一 START=120，v4 下一 START=12。

源绑定的敏感性世界选取 capture01 `profile:34` 的 idle donor pair：
`d=2417825ns,w=2217173ns`。以同一个 pair 的 `x=0` 和 `x=min(d,w)` 两端分解，
固定三个 synthetic handlers 各 10ms、相同 public arrivals，首次 START 均为 2417825ns；
后两次 START 分别为 `[14835650,27253475]ns` 与 `[12618477,22819129]ns`。
差值为 2217173ns 和 4434346ns。复用 idle d/w 是报告内显式假设，busy sender 回压仍未识别；
donor request ID 只绑定训练 pair 的来源，不用于查询 target request 的预测答案。

合计 22 项 FIFO、partial-identification、v4 tests 通过，包括精确有理数、同 submit tie、
enqueue 顺序变化、实际 release、独立 sender 阻塞、pending cutoff、v3 对照和 resource adapter。
全部仅 CPU，无验证标签、GPU 初始化、新采集或 Nsight export。
response rates 使用 parent-visible covariates，不能称为实际 SM 工作率；stdout 回压机制仍未知。
native CUDA API 数值认证保持 **UNQUALIFIED**，main queue provider 与独立预测准确性资格保持 false。
