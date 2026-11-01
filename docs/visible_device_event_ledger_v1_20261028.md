# 可见设备事件账本 v1

`tools/predictor_validation_20261027/visible_device_event_ledger_v1.py`
将设备查询证据与原 host START/PHASE/RESULT/READY 状态分开保存。
旧适配器会把 `event=observation` 当成 PHASE 边界并报错；新接口只把
真正的阶段边界交给原 observer，将查询放入独立账本。

```python
state = new_state(aliases, execution_context_sha256)
state = advance(state, complete_persistent_runtime_registry, decision_time_ns)
```

调用者必须提供从开始保留的同一请求注册表和完整执行身份。
可见性使用父进程 reader receipt；尚未收到的事件只读取 receipt，
不读取事件正文。RESULT 正文也只在 receipt 可见后取 handler END，
不把模型输出、答案或未来 token 数传入状态。
后续调用必须保留已消费的 host/device 前缀；删除、改写、插入旧事件均拒绝。

设备查询必须对应已收到的同一请求、owner、forward 边界，具有连续的请求内
查询序号、稳定的 event/device/stream/index 身份和正确的 host-query/reader
时间关系。多请求各自的查询序号与原 host 全局反馈序号分离。
false 后可再查询 true；true 后重查、未完成却报告 elapsed、跨设备改身份，
或宣称全 GPU 完成／剩余百分比／物理 SM 服务都会拒绝。

返回的 `kernel_cursor` 始终为 `None`，
`main_lifecycle_resume_eligible` 始终为 `False`。
query=true 只说明记录 stream 的前缀在该次查询时已完成；query=false
只说明没有在那次查询中确认完成。二者都不表示决策时刻的精确剩余工作。
该接口没有补上跨 stream 依赖、动态 EOS、原生 sender 队列原子快照，
也没有实现主生命周期引擎 checkpoint/resume 或在线滚动分流。

作者 7 项和独立 11 项测试全部通过。独审使用真实 natural smoke 在
104 个 reader-receipt cutoff 逐前缀重放，保留 30 条查询、4 个 host 请求，
不产生 cursor 或主引擎续算资格。报告为
`artifacts/predictor-visible-device-event-ledger-v1-independent-review-20261028-cpu01/report.json`，
SHA `929a3654e99827e1aa974baa86bafcb5b450f356f8a26cb9157261f72420e383`。
该 CPU 账本审查不认证真实 CUDA API 调用、GPU 预测准确率或分流收益。
