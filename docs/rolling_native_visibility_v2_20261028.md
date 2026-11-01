# 原训练采集的滚动可见性审计 v2

`rolling_native_visibility_v2.py` 已在 CPU 上重放原四路共驻训练采集，每路保留原 57 个测量请求、
12 个 priming 身份、完整 30 秒公开到达与原 120 秒 drain，不修改原数据或启动 GPU。
切点固定为 1、5、10、15、20、25、30 秒。每个切点先依据父进程 reader receipt 选事件，
再读取相应 worker 内容，复用原 `observed_state v2` 和 `rolling_visible_queue_v1` 的因果校验。

实际产物为 `artifacts/predictor-rolling-native-visibility-v2-20261028-cpu01/report.json`。
28 个快照共 1596 个“请求 × 切点”状态如下；这些是重复观测计数，不能当成独立请求样本数。

| 可见状态 | 请求 × 切点数 | 预测可用范围 |
|---|---:|---|
| 已可见 END 和 READY | 634 | 保留实际 START、END、父进程 receipt 及 READY 的不同边界 |
| 已知未来公开到达 | 788 | 保留全部后续到达，不以当前积压代替全过程 |
| 可见 START，但无当前设备进度 | 40 | 有 handler 状态；当前 GPU/segment 剩余量未知 |
| 已到达，但尚不可见 START | 134 | 不能判断仍在队列还是 START 延迟可见；无精确 waiting 证明 |

所有选中事件均保留 worker 时间与 reader 可见时间；最大可见延迟为 4,442,303 ns。
因此旧 v1 的 `observed_at == cutoff` 且 `received_time <= cutoff` 精确游标契约只适合明确的 CPU fixture，
真实生产者无法默认提供零延迟瞬时测量。v2 不修改旧契约，也不把过期游标改写为当前值。

## 接入真实进度前必须补的事实

当前 native worker 的 START/PHASE 是 host 边界。forward 返回、NVTX 范围结束或 host phase end
不自动表示对应 CUDA 操作已完成；更不能表示尚未结束 kernel 内部的精确进度。
当前实现没有直接设备进度生产器；`runtime_direct_observation` 字符串和文件 SHA 本身不能证明传感器语义。

将来采集应使用预先绑定的 request/profile-node/stream/device-event 身份，以非阻塞 event query
确认已完成节点，记录 host query 的前后括号、父进程 reader receipt、稳定序号和拥有进程 PID。
禁止为获得进度强制同步 GPU，否则会改变待预测干扰。
这些事实只能给出最后确认完成的节点；未完成节点的 fractional progress 仍未知。
query 到父进程 receipt 之间可能继续推进，因此应保留延迟与剩余工作边界，不提供未经证实的当前点值。

动态 decode 还需独立处理未来 EOS/长度不确定性；公开 `max_new_tokens` 是上界，不是观测到的剩余 token 数。
真正未开始状态需要 native 队列与 start 的原子确认；缺失 START 或 host 入队并不能替代它。
这些要求均写在报告 `required_device_evidence` 中，目前没有宣称采集已实现。

## 主引擎续算仍缺的部分

本审计不恢复 `LifecycleDagPhaseModel`，不复制旧模拟 checkpoint。
主引擎仍需导出/恢复所有请求 FIFO、节点残差、事件 heap、receipt/READY、物理持有者、pending switch、
restore 起点、累计 SAVE budget、mark clock 和完整未来公开到达，并把观测事实与预测隐状态分开。
这些条件没有满足时，主路径数值续算、完整六项 utility、GPU 准确度均保持未知。

验证命令：

```bash
python -m unittest tests.test_rolling_native_visibility_v2 tests.test_rolling_visible_queue_v1 -v
```

10 项测试通过，包含延迟 START 不能证明 waiting、host forward end 不能充当设备完成、
未来 response body 不访问、READY 延迟、训练角色先于 raw 读取、全未来 cohort 与实际通用模拟器的两次续算。
本次真实 raw 重放是训练数据可观测性审计，不能计作独立预测验证。
