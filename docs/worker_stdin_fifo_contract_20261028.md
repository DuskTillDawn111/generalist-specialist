# Worker stdin FIFO 与 idle donor 的部分识别（CPU，2026-10-28）

`worker_stdin_fifo_contract_v1.py` 审核三份自然训练 raw、macro/review/source closure，输出
`artifacts/predictor-worker-stdin-fifo-contract-v1-natural-20261028-review01/report.json`。
150 个 measured donor 都绑定同 owner 的前一实际 READY occurrence（必要时使用 priming predecessor）；
busy-at-submit 为 0。三份 capture 的最小 READY→submit 间隔为 860562、2432416、3008761 ns。
submit→START 范围分别为 126493–2417825、119294–1415221、144927–1361916 ns。
46/150 个 START 早于 parent write-completed record；该 record 不能成为 START gate。

冻结的 active deployment `_send` 只在 send 返回并记录完成后继续本 owner FIFO，不等 RESULT 或 READY。
transport 将 JSON 写入 stdin 并 flush。实际 semantic worker 的 `for line in sys.stdin` 串行执行 handler。
前一 READY 的 perf clock 在 READY emit/flush 之前采样；它是前一 handler 的实际边界，
但尚未证明 worker 已回到 stdin 阻塞读取。下一 START 前仍有 READY emit、read、parse、任务检查及准备。
parent READY reader/collector processing 并不阻止已排入 stdin 的下一请求执行。

两个 FIFO 的显式 CPU 接口为：

```
enqueue = arrival + supplied_enqueue_extra
submit = max(enqueue, previous_parent_write_record) + supplied_sender_extra
line_available = submit + supplied_line_extra
START = max(previous_actual_READY_occurrence, line_available) + supplied_worker_pre_START
```

每项 TransportDraw 必须显式给定、非负、精确有理数，line_available 不晚于 write record。
worker START 不等待 write record、RESULT reader 或 READY processing。初始 sender/worker release 也必须显式给定。
`compose_explicit_world` 的 handler durations 只定义反例世界，没有拟合 response model。
stdout RESULT/READY 及跨请求 FIFO 矛盾会报 UNKNOWN，不截断、不补零。

`worker_stdin_fifo_partial_identification_v1.py` 保留每个 donor 的联合 pair：
`d=START-submit`, `w=write_record-submit`, `u=min(w,d)`。
在 prior READY≤submit 的这些样本中，只能得到 `x=line_extra∈[0,u]`，
以及同一次 draw 的 `worker_pre_START=d-x`。不得将两个边际独立抽样。
若 caller **显式假设**此 idle pair 可迁移到新 submit `S`、前一实际 READY `R`，则可行区间为：

```
START ∈ [max(R,S+u)+d-u, max(R,S)+d]
```

区间端点均可达到，CPU 穷举和精确有理数测试验证了区间覆盖与端点。
不传 `assume_idle_pair_transfers=True` 会报 UNKNOWN。该接口以 supplied submit 为条件，
没有预测新 schedule 下的 sender submit，也没有迁移 idle `w` 为 busy pipe backpressure。

`partial-identification.json` 包含 150 个绑定 raw identity 的 pair、源 receipts 及以下合成反例：

- positive worker continuation：首请求 START=1、实际 READY=11；次请求 submit=1、START=12。
  首请求 RESULT reader=19、READY reader=21、READY processing=121。次 START 早于这些父端事件，且所有 producer FIFO 成立。
- sender 阻塞：首请求 write record=20、START=1、READY=11；次 submit=20、START=21。
- 同 idle `d=w=5` 可由 `(line,pre)=(5,0)` 或 `(0,5)` 解释；新 `R=10,S=0` 时 START 分别是 10 与 15。
- `d=w=2ms` 的两个一致潜在世界，经两个排队且实际 handler 各 10ms 的请求，
  START 序列分别为 `[10,20]ms` 与 `[12,24]ms`；第二 START 的不确定宽度增为 4ms。
  这只是显式 transfer assumption 下的不确定传播，不能解释为已测到的 busy donor。

未知项包括 sender 调度/序列化、pipe 回压、READY emit、read/parse/pre-START 的分解。
既未测当前/历史 worker pipe occupancy，也没有证明发送永不阻塞。transport 使用有限容量 OS pipe；
即使单 payload 很小，积压和 worker 停止读取也可阻塞 flush。当前新建 pipe 容量无法认证历史容量或未来占用。
现有 capture 无 request-admission resource binding，不新增 admission cap。

13 项隔离 FIFO/partial-identification 测试通过。全部仅 CPU，无新采集、验证标签、GPU 初始化或 Nsight export。
main queue provider 资格仍为 false；native CUDA API 数值认证仍为 **UNQUALIFIED**。
