# 原生忙态站点适配 v1

新增 `tools/predictor_validation_20261027/twoFIFO_native_busy_state_v1.py`，不改正在运行的冻结采集闭包，不启动 GPU。它把真实 v3 `history_snapshot` 的字段适配为 collector 已处理的可见前缀，保留完整公开到达日历、原生历史请求和四个站点时钟。这里没有把现有 CPU 模拟 checkpoint 当成原生忙态推断结果。

```python
state = from_snapshot(snapshot, public_cohort=requests, cutoff_ns=cut,
    owner_producers=owners, execution_context_sha256=context,
    forecast_origin_ns=origin, previous=previous_state)
result = forecast(state, requests=requests, plans=plans, rates=rates,
    window_start_ns=origin, window_end_ns=end, drain_end_ns=drain,
    protocol_slacks=slacks, transport_draws=transport)
```

`forecast` 有实际调用 `actual_worker_twoFIFO_reference_v5_2.simulate`。当前可调用域是已有证据能认证其结构的 settled native seed：历史请求完整 START/RESULT/READY 与 sender write 均处理完毕、所有驻留 owner 有可用历史、当前完整 cohort 与历史 ID 不重叠。CPU 测试实际跑通该接口，并验证后续引擎 JSON checkpoint/resume 与连续执行一致。输出仍声明 `native_source_authenticated=False`：纯 snapshot 适配不代替训练收据的独立重导审查。

忙态不进入数值续算，而是保留可行状态集合并返回明确阻断原因。START 已处理而 RESULT 未处理时，状态可以是 GPU/host 仍活动，也可以是 handler 已结束但通知仍未处理；缺失 START 也不能证明 waiting。`exact_remaining_work` 一律为空，不用 host elapsed 当已完成 GPU 工作。device query 若已处理单独记为 QUERY，不作为 PHASE 边界或新的 START，不将它解释成完成全部 GPU 工作、精确 kernel cursor 或剩余百分比。

真实 `DeviceProgressObserver._observation` 带 `event=observation`、`kind=device_event_query`，但不带 PID；semantic PHASE 同样没有 PID。这两类消息按 runtime 记录的 owner/stdout transport 和 snapshot worker lifetime 绑定，若显式提供 PID 则必须一致。START 必须符合显式原生 START 身份并带匹配 PID；未识别的无 event payload 不能自动成为 START。测试直接调用真实 `_observation` emitter 的纯 CPU 路径，保留两次 query 为独立 QUERY，并确保没有生成第二个 START。

前缀可见性按 `collector_processing_time_ns <= cutoff`，不会因 reader 已收到而提前开放正文。先只读取 processing 标量，未来事件和 RESULT 正文不会被复制、哈希或读取。可见事件保存 worker occurrence、reader receipt、collector dequeue 和 processing 四个独立时钟。`previous=` 校验完整公开 cohort、进程身份和已处理前缀，拒绝删除、改写、插入旧事实或倒退 cutoff。完整 future cohort 一直保留，没有按 observed 或已完成请求缩分母。

历史重放还会过滤截止点后才注册的 native 行：只有截止点前的真实 sender enqueue/submit/write 或 collector 已处理消息，才能证明该行当时已进入可见原生状态。公开 scheduled 时间本身不能证明已注册，哪怕 scheduled 已经过期。新增 future registry 行或其尚未可读的事件，不能改变已有截止点的输出状态与 SHA；实际已观测 enqueue 则会保留，但不证明 worker 尚未 START。

`pending_request_ids` 在真实 runtime 是记录注册表，不能当 GPU 排队数。`queued_runtime_events` 只保存当前快照的 envelope 数量，正文保持不透明；历史重放不使用最终快照的队列数量。空 queue 也不证明 protocol 空闲：`DeploymentRuntime._collect` 可以已从 queue 取走消息、等待 `runtime.lock`，而该消息既不在 queue，也未写入记录。适配器明确保留这种 in-flight 不确定性。

可复现阻断反例见作者 CPU 产物目录中的 `counterexample.json`。两个合法的 synthetic native 未来都在截止点 5 前仅处理同一 START（worker=1、reader=2、dequeue=3、processing=4），sender 和全部公开到达完全相同；未来 handler END 分别为 10 和 100，读者看到 RESULT 的时刻分别为 11 和 101。适配后状态完全相等；任何只靠此已处理前缀给出唯一真实剩余量的算法会对至少一个未来判断错误。测试确认两个输入都不能调用忙态数值引擎。这是不可识别的证据，不是预测误差报告。

要超出该受限入口，最少还需以下采集事实与对应续算支持：

- 原生 active request/profile-node/stream 的预绑定身份，非阻塞 device event query 及 host query bracket、reader/collector 时钟；跨 stream 依赖也须有界。节点未完成时保留 remaining-work 支持集，不能从 elapsed 推出点值。动态 decode 的未来 EOS 和 token 上界单独保留。
- 在同一因果切面可核验的 sender FIFO、write 正在进行状态、worker queue/start 原子 acknowledgement。仅有 parent write record 不识别 stdin line availability；仅缺 START 不识别尚未开始。
- 持久的 producer/reader/collector 消息身份、完整前缀/水位和 dequeued-but-unprocessed envelope。队列复制和 parent 记录锁不能补出 collector 已取走但未处理的消息。
- 将这些支持集恢复到 v5_2 所需 active macro/virtual residual、pending transport/notice 和完整 station heaps 的推断接口；存在多个可行隐状态时应传播分支或已证明外包络，不能挑一个 seed 冒充真实续算。

CPU 验证：

```bash
python -m unittest tests.test_twoFIFO_native_busy_state_v1 tests.test_twoFIFO_settled_history_initial_v1 tests.test_actual_worker_twoFIFO_reference_v5_2 -v
```

作者 25 项相关 CPU 测试通过（本模块 12 项），另有 10 项独立 CPU 测试。覆盖真实无 PID 的 semantic PHASE/QUERY、未来正文毒化对象、历史 registry 存在性不泄漏、reader-visible/collector-invisible、四站点、完整 future cohort、前缀连续推进与篡改、忙态两未来反例、实际 settled 引擎调用及 JSON 续算。没有读取 validation 标签，没有新增 GPU 数值或最终分流收益结论。此模块为新适配边界，尚未接入冻结 whole-block runner；忙态闭环仍明确未完成。
