# 原公开输入匹配的补充 macro 训练准备

本补充只准备 CPU 计划，不执行 GPU。原 150 cases / 546 measured requests、
原 exact12 producer 和 181-request、90+120 s 验证参照全部保留。

## 输入与执行兼容性

prospective cpu02 的 16 个 required `(owner,task,pool_id)` 中有 12 个缺少
measured 模板。原 source-rebound context / runtime pool 有四任务各四个输入，
但原 measured cases 仅含 text/OCR。缺口中的 G2 image 一种、ASR 一种已在
context/pool 中，priming 不算 measured support；另六种原公开输入不在此域：
两种 text、两种 image、一种 ASR、一种 OCR。新 pool 是 old16 与原公开10
输入的精确 union22；pool IDs、text context/question、媒体 bytes 都不变。

`semantic_progress_worker_v1` 已支持 G2 text/image/ASR，原 context 的模型、
spec、command、allocator、threads 和 worker bytes 可以作为低层执行来源。
但 union 输入声明会形成新的候选上下文；相同 worker bytes 不认证 statistical
support、generation shape 或物理 transfer。新 context 保留 parent receipt，
仅扩展 declared_inputs，明确 `context_transfer_qualified=false`。

原 `twoFIFO_source_rebound_execute_v1.collect` 先调用硬绑定 exact12 的
audit_plan；review/intake 又固定原 campaign、18 priming、4 shards、150 cases/
546 rows。不能把新计划标成旧 producer 或猴子补丁绕过审计。可以复用已审的
initial_rows、dispatch_case、history snapshot、activity guard、owned monitor/
cleanup 和 semantic worker 的冻结 closure；仍需一个显式的新 owned executor
和 review/intake 接口。本准备器没有 execute 子命令，entrypoint 为 null。

## 合并后的自然活动日历

required keys 按 owner 分组：text#0 三个、text#1 三个、G2 八个、OCR 两个。
保留四个已有 measured keys 作桥接 control，覆盖全部16 keys。

1. 每 key 一个 solo case，共16。solo 中同 owner 两请求在0和1ms到达，既提供
   无其他 owner 的活动意图，也保留 sender/worker FIFO burst 诊断。
2. 对每个 unordered owner pair，按两个 key 列表循环，生成 max(nA,nB) 个
   case。每个 case 同时为两边 key 提供另一个 owner 活动的覆盖，因此六个
   pair 合计33 cases。三block的启动差预声明为1ms、20ms、300ms；两条到达
   的先后按block与pair内序号交替，所有block保留同样的key组合。
3. 全四 owner 按各 key 列表循环，生成8 cases；三block分别按1ms、20ms、50ms
   步进错开四条到达，owner 顺序按公开block与case序号旋转并交替反转。每个
   key 都在每block至少一个 all4 日历中出现。

每 block **57 cases、130 measured requests**。这是全部有向覆盖证明：每 key
均出现 solo、分别与另外三个 owner、all4 五类意图。三 block 重复完整日历并
旋转 case 次序，分为每 shard <=13 cases，共15独立 startup attempts、171
cases、390 measured requests。每 attempt 沿用原18条 priming作为历史，共270
priming，不能计入 measured support 或当成另270独立样本。三 whole blocks
才是 leaveout unit；不能按 raw SHA、case 或 shard 计独立 replication。

每 case 沿用训练10 s measurement /60 s drain，原保护区间无完整snapshot写入
合同保持；每 attempt 最大固定case时间910 s。原2400 s supervisor和18GiB
preflight不降低；输入/模型首次运行可能超时，CPU准备不宣称预算一定满足。

该合并相对逐target重复5模式的80 cases/block减少23个固定drain，但不删除
任何 key/mode。没有通过未来实测结果选择case；没有全局 cap1/cap2，仍是
natural shared/no batching/no weight reuse/all arrivals enqueue。

## 可识别性与资源范围

本补充是 **完整 request macro/template 训练**，不是 kernel 资源 fit，不是
独立验证。pair/all4 只证明公开活动意图；实际 overlap、parent-visible phases、
OCR inference n0/npositive、same-phase 变化和12列设计的残差秩必须由native
数据确认。不得把 intent peer_count 填成真实 covariate，也不计算虚构设计秩。
full fit 和每 whole-block leaveout 都需保留所有macro、检查nuisance support、
残差Jacobian满秩、正定局部根等原门槛；不足保持UNKNOWN，不删列或补经验系数。

资源预测仍缺完整历史状态/三owner状态/相位组成和90+120 s窗口的支持认证。
本计划不承诺资源表覆盖，不把macro共驻日历当M/E训练资格。后续资源拟合需独立
明确原measurement E、startup/3600一次、startup→drain owned sampled M及一次
安全因子。当前不封预测、不选候选、不输出 ΔU 或最终分流收益。

训练与未来验证共用 **exact input contents** 是known-input条件验证范围；
training block/attempt/execution/request IDs 必须与历史以及未来验证完全分离。
不能重命名输入绕支持，不能把本补充训练heldout标签称作独立validation labels。

## CPU 独审后的启动差修订

固定300ms对短image/ASR可能错过重叠；三block改为上述多尺度启动差，增加公开
候选覆盖，但仍不保证真实overlap。`case-definitions.json`现在保存三block各自
完整calendar；`coverage.json`含每block的全有向mode证明；audit按block重建
实际arrival/tie order，因此不能同步改receipt绕过启动差声明。

cpu01-prepared与cpu01-author完整保留；cpu01-author/source-at-author-review.py
保存当时源码bytes，匹配原author记录的SHA。修订后的准备与作者收据写入
`artifacts/predictor-twoFIFO-matching-macro-supplement-v1-20261029-cpu02-prepared/`
及对应`cpu02-author/`。没有改冻结GPU closure，也没有执行这些训练计划。
