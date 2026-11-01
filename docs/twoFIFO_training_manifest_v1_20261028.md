# 原 sender/worker FIFO 补采准备（CPU，未执行）

已落地 `artifacts/predictor-twoFIFO-training-design-v1-20261028-cpu01/manifest.json`，
SHA256 `4e7793e1b349dc3305cad61a61f9933e562a81316d8871833afbc2ba0680918f`。
330 个源 receipts、冻结训练输入内容/大小、每个有序 arrival cohort 的 hash 均已复核。
`runtime-inputs.jsonl` 只含选定的六个原训练输入；没有改内容、填充 prompt 或读取验证标签。
`runtime-case-configs.json` 包含 48 份通过现有 `MeasurementConfig.validate` 的具体配置。

三份计划 training blocks 使用相同 16-case 矩阵、不同的固定 case 顺序，每 block 100 offered，共 300 条。
每 case 固定 measurement=10s、drain=60s；deadline=`epoch+70s`，不从最后到达或最后完成重新计时。
它们是新训练计划，尚无采集数据；三个 block 是计划复制单位，不能把内部 case 当作独立 capture 重复。

| 每 block 的 case 组 | 数量 | 每 case offered | 刺激与可证限制 |
| --- | ---: | ---: | --- |
| 四 owner 各短/中/长 spaced | 4 | 3 | 0/3/6s 到达，目标提供稀疏参考；不保证每次已空闲 |
| 四 owner 各长→短与短→长 HOL | 4 | 6 | 前三条间隔1ms，3s 后反序；保留连续同 owner 队列历史 |
| text#0、G2 的 long→short→short | 2 | 3 | 0/50/100ms，目标 worker 忙但 sender 已完成前次发送 |
| text#0、G2 连续大 payload burst | 2 | 18 | 16条 long 同刻，随后 short/medium；目标 sender/pipe pressure |
| mixed owner overlaps | 4 | 4/4/6/8 | 1/2/3 个 peer，0/50/300ms 偏移，同 owner 各两条请求 |

三个 text 输入长度为 163/538/19027 个 context+question 字符；保持冻结 tokenizer 和生成配置，
不把字符数叫作执行 token 数。OCR 输入文件为 90343/230532/1794212 bytes，不能当作 pipe payload 大小。
实际送入 pipe 的 OCR sample 是路径/元数据，约945–947 UTF-8 JSON bytes；三档 text sample 约632/1029/19799 bytes，
还需加实际 request envelope。未来必须记录实际 text encoding、完整序列化 byte length/hash。
大 payload 只是压力刺激；有限 pipe 是否阻塞仍取决于容量、占用、读取、buffering 和调度。

`twoFIFO_training_manifest_v1.py` 的 CLI 只有 `prepare` 和 `check`，没有 execute、worker startup 或 GPU 初始化。
现有冻结 `profile_owned_collect_v3` 的 job 以 owner 为字典键，不能表达同 job 重复 owner，不能直接塞入旧格式。
新增非冻结 `dispatch_case` 接口给未来独立 owned supervisor 使用：它接收已启动、已 primed 的原
`DeploymentRuntime`，逐 arrival 调用原 `runtime.enqueue`，最后以固定剩余时间调用原 `runtime.drain`。
不会等待单请求 RESULT/READY 才 enqueue 下一条，也不改变 sender、worker、collector 源。

所有 offered rows 在首次 enqueue 前构造并 checkpoint。enqueue 失败、超时、晚于 fixed drain 未提交、
返回晚于 deadline 的请求都保留；late RESULT 另标 `RESULT_reader_within_fixed_drain=False`。
即使提前完成，也等到固定 cutoff，留足资源观测范围。OS/polling overshoot 单列，不能延长 scoring 窗口。
原配置 rps 只供能力/路由检查；实际到达来自 manifest，不能再调用周期/Poisson generator 替换它。
未来 supervisor 仍须承接原 lease、所有权、startup/priming、monitor/checkpoint、cleanup；本次未编写或触发新的 GPU 执行命令。
若前 case 不完整，之后未启动的 case 记录 NOT_STARTED；不可删除已启动 case 的任何 offered。

采集后按真实时钟分层，不能仅按 case 名称贴标签：

- worker 忙：新 submit 位于前一实际 START→READY occurrence 内；前 READY perf 之后尚有 emit/read/preparation，不自动等于 stdin idle。
- sender FIFO 忙：新 enqueue 早于前一次 send/write/flush 返回。旧 write-record stamp 还含 record-lock 等待，不能直接叫 pipe 阻塞。
- worker 忙而 sender 已完成、sender 与 worker 同忙、两者都未命中分别保留；缺少自然 sender-busy/worker-idle 支持时记 UNKNOWN。
- pending owner depth、同 owner arrival ordinal、前一 actual READY/request identity 全部保留，用于审计 HOL 和连续 burst。

另有独立 CPU-only `sender-probe-manifest.json`，只表示显式20ms sender hold 的机制测试设计，
没有安装 proxy、启动 runtime 或执行 probe。它改变 timing law，18 条 planned probe rows 不计入自然300条，
绝不进入自然 response/transport fit。原始 `TransportDraw` CPU 队列反例已经能检验 FIFO 拓扑；
自然训练无需为保证标签而实施 hold。若将来单独授权此类机制实验，仍须用实际 line/read markers 验证目标层，而非凭设计声称命中。

`collection-fields.json` 给出具体字段合同。冻结原时钟必须全部保留，包括全部 START/PHASE/query、RESULT、READY
occurrence/reader/dequeue/processing。新的独立 instrumented variant 才可增补：

| 位置 | 必需的新时间/身份 | 能区分什么 |
| --- | --- | --- |
| parent sender | dequeue；serialize begin/end；write begin/return；flush begin/return；record lock acquired；完整 line bytes/hash/encoding | host queue、serialization、write/flush wall interval、record-lock 等待 |
| worker stdin | read ordinal、read enter、完整 line return；parse begin/end、task checks、peak reset begin/end；START | read 等待、parse/pre-START composite 与前 handler 分开 |
| worker READY | 原 READY occurrence；READY emit begin 与 flush return | 实际边界至 READY emit 完成、下一 read 的间隔 |
| parent notifications | stdout line return、JSON parse end、event put enter/return、collector dequeue/lock | reader、入队、collector 的不同时间腿 |

read-enter 当时尚无 request ID，必须绑定 worker PID/start identity 与 read ordinal，在 parse 后再绑定 request ID。
可先记录局部内存 clock，并在后续已有消息 piggyback，避免为“记录 flush 完成”再引入同步 stdout flush；
最后一条的 post-flush marker 需保留到退出日志。任何新增 CPU/IO 开销均属于新 observer domain，
需要与原标记版本作独立对照，不能覆盖旧采集或声称零扰动。
只可选地读取普通 pipe 容量元数据 `F_GETPIPE_SZ`；不假设 CUPTI/perf/eBPF、硬件 counter 或额外权限。

即便这些字段齐全，**kernel 中完整 line 的首次可用时间仍未必可识别**：
worker 忙时数据可早已到 pipe/userspace buffer，之后 read return 只给上界；write/flush 的 Python buffering
和线程抢占也不能仅凭 wall stamps 分解为纯 pipe 传输。read enter/return 能约束 wait，不能回溯早已排队数据的到达时刻。
READY flush marker 能去掉已知 continuation 段，但不自动识别 read scheduling/preparation 的 CPU service demand。
未知 busy backpressure law、调度因果贡献及未采样组件必须保留为 UNKNOWN/联合区间。

响应矩阵重点覆盖 OCR inference、G2 prefill、text#0/#1 prefill 和 G2 末 decode，保留所有 entry/READY tail。
启动偏移只尝试改变同/异 phase exposure，实际 achieved phase coverage 必须由全部生成与处理的 phase 事件审核。
当前只有一个 OCR owner，不能声称补齐不存在的 OCR same-phase peer；READY tail 已清除 visible phase，
也不能用偏移制造 same-visible-tail 列。22维设计的结构性零列仍失败，不能删节点或按 job ID 硬拟合。
此最小16-case设计不声称穷尽所有 phase×input×peer strata；需要额外支持的格子在后续矩阵中明确缺失。

资源侧保留四个相同 resident，不要求穷举144个 exact phase/input fullstates。
原域只有48个 power、58个 memory states 获得至少两 capture 支持，短 entry/tail 常小于 NVML 周期；未采到不能补0。
可检验的较粗假设是：按完整 resident roster 与实际 START→READY owner/input multiset，在一个全局 board clock 上 pool。
`one-peer-simultaneous` 与 `one-peer-staggered` 保持相同 resident 和 offered owner/input 多重集，仅偏移0↔300ms，
用于检查 phase composition 改变后，该粗 pooling 假设是否仍有系统残差。不要先假设可 pool；按 whole block leaveout 诊断，
缺少 state/sample 或跨 capture 支持仍是 UNKNOWN。实际 handler phase 组成、query duration、sample brackets 和 pending host 队列一起保留。

资源 clock 必须记录 startup→priming→epoch→measurement end→fixed drain→最后 owned sample→worker close。
board energy 仅积分固定10s measurement，drain不混入；startup energy 单列，不在采集里提前做3600s amortization。
owned sampled VRAM peak 包括 startup、inactive residents、measurement 与 drain，worker close 前结束；
它不是 instantaneous peak。不能求和 owner power/exposure 代替全局能量，也不能按稀疏 phase sample 伪造资源值。

7项 injected CPU 测试通过；全部源、配置与 manifest 已校验。本次没有 GPU、模型、collector、counter 或 Nsight 执行，
未改变冻结文件；native CUDA API 数值认证与主 provider 资格保持 UNQUALIFIED/false。
