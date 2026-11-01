# v3：带完整通知链的联合协议时序候选

v2 的固定通知延迟迁移在 600 个 offered 场景请求中留下 430 个整 world FIFO UNKNOWN。这个失败分母保留在 v2 training03，不能通过删除场景或修补旧 delay 隐藏。v3 是独立的新时序模型和新产物，解决重新生成 wall time 后的通知队列可组合性。

`protocol_timing_slack_v3.py` 对每个训练 capture 的全部 measured START、PHASE、RESULT、WORKER_READY 分解四时钟。三个 capture 各 1538 条消息，共 4614 条；START 是原 execution_events 中没有 event 字段的真实消息，放在 entry work=0 的 diagnostic mark，不能因为它不改变 phase 而漏出队列。每请求的全部消息数独立从 raw execution_events+RESULT+READY 重推，并逐项核对 vector 完整性。

模型设每 worker 一个 FIFO reader，以及一个公共串行 collector。原时钟分解为：

`sR_i = R_i - max(O_i, R_previous_same_producer)`

`sD_i = D_i - max(R_i, P_previous_shared_collector)`

`sP_i = P_i - D_i`。

所有 slack 必须非负。原数据的 reader 归属按 worker，公共 collector 顺序按 reader availability 排序；三个 capture 没有违反该顺序的 dequeue/processing 逆序，所有 source 时钟都被独立重建。新 world 按同一递推生成新 R/D/P，公共 collector 的繁忙时间会影响其他 owner 的通知；phase 仍只在 P 时可见，E2E RESULT 在 R 时完成。query 不改变宏 phase 或释放 handler。

这不是把旧延迟 clamp 到 FIFO 下界。slack 是新 joint vector 的一部分，新 occurrence 和前站排队共同生成新 availability；原 v2 延迟和 UNKNOWN 证据完整保留。slack 也不是已识别的 CPU service demand：缺完整 parent 入队序号、未采集的控制/priming 消息、父进程背景工作，可能都被吸收到经验时序 slack 中。按 reader availability 决定公共入队顺序是明确的模型假设。

`joint_request_protocol_transport_candidate_v3.py` 在 v2 完整 request work vector 上增加这些成对时序 slack。实际训练 query 的 source forward 和 occurrence macro 继续分开；final query 保留在 native-END→READY tail。每个请求与 provenance 按原完整迭代顺序配对，合法相同局部 vector hash 不会覆盖另一个请求的来源或分母。

隔离执行器 `macro_protocol_timing_reference_v3.py` 用独立 reader heap 和公共 collector station；cutoff 保存尚未到 reader、尚未处理的通知以及实际已完成但未通知的宏。站接口 `TimingStations.issue` 和 `accept_reader` 接收实际 occurrence，不依赖 parent 的派发策略，可供真实 sender/worker two-FIFO 引擎组合。当前 forward 场景沿用 after-READY-processing 的显式替代策略，不是原 runtime 的队列资格。

最终产物在 `artifacts/predictor-joint-request-protocol-transport-v3-natural-20261028-training05/`；training04 是 provenance 配对与 source freeze guard 补强前的产物，保留并被 training05 替代。三个 whole-training-capture 留出、固定四组 draw 的 600 个 offered 请求均完成，FIFO UNKNOWN 为 0；reader E2E MAE 160.04ms、RMSE 286.56ms。该完整分母结果不能与 v2 的成功子集 MAE 直接比较，也不证明预测准确或已取得校准资格。

v3 仍须检验 response-clock transport、共享 capture latent 和请求 residual 依赖、公共 collector 背景负载迁移及原 sender/worker 队列合同。没有新增 GPU 工作、validation 标签、资源／质量／能耗 provider、native assimilation、P95 或 ΔU 结论。
