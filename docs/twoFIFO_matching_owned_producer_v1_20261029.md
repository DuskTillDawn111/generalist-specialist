# Matching supplement 的显式 owned producer

输入是已独审 `matching-macro-supplement cpu02` 的准备清单。新 producer
`twoFIFO_matching_execute_v1` 生成自己的 `matching_owned_training.v1.attempt`
与 capture schema；不替换原 exact12 collector、audit、review、intake，
不把 union22、新日历或新上下文冒充旧 producer 的证据。

## 可复用边界与身份

直接调用既有 `twoFIFO_owned_collect_v3` 的 ActivityLedger、initial_rows、
dispatch_case、history_snapshot（由 dispatch 调用）、audit_activity 和
finish_collection。原 sender FIFO、worker FIFO、历史快照、固定窗口和
monitor-before-worker-close 顺序保留。StartupBoardMonitor / live ownership、
SourceChangeGuard、supervisor lease / flock / watchdog / descendants cleanup
也直接复用；旧 collect / audit 入口不调用、不 monkeypatch。

新 producer 将原 CPU02 的全部 15 attempt / 3 block / 171 cases / 390
measured requests 显式转换为新合同；保留 seed plan receipt 和精确日历。
每 attempt 的 18 条 priming 获得预声明 `attempt_id:priming:index` 身份，
共 270 条，与 measured IDs 分离。collector 在启动 worker 前就登记全部
measured 和 priming 行；启动或 priming 失败也保留尚未开始的行，禁止选择
成功子集。每个完整 case 的 prior history 包含全部18条 priming及之前所有
offered rows，而非在 case 边界清空状态。

priming 的物理内容轴与来源身份分开。`planned_priming_semantics` 投影有序
`(model,task,pool_id,exact_input_sha256)`，不包含每次启动不同的 request ID；
ID 仍留在计划、原始行、history、native review 和 intake 中强校验。公开
计划只声明 `actual_native_history_qualified=false`。只有完整 native review
通过后，才调用既有 `legacy.priming_identity(raw,context)` 对真实18条成功
priming重建内容身份，与计划投影比对。intake同时要求270个来源ID正确和
15个 attempt 的有序物理语义一致。没有修改旧 resource_v3 的投影。

## 完整采集、审查与 intake

新 collector body 保留已审 source-rebound 的执行顺序，但使用自己的
schema/audit。原10s measurement与60s drain、每attempt最多13cases、
2400s supervisor / 18GiB preflight均不改变。所有到达自然入队；没有全局
cap1/cap2，没有共同batch、cache或权重复用。

exclusive claim 在调用 supervisor 前用 `open('x')` 写入；claim receipt
加入 inotify guard 与 execution binding。collection 已存在、claim 已存在、
失败、未完成、旧 schema、缺失 owned cleanup 或最终 source audit 失败均
拒绝重试。supervisor 使用现有全局 GPU lease，执行与 cleanup 期间保持
独占；CPU 准备不会获取该 GPU lease。准备文件不构成 GPU 启动授权。

新 review 显式重用纯 clock、FIFO、history、checkpoint、协议、telemetry
检查，不依赖旧入口接受新 schema。原始输出必须重新解析；CPU 注入运行
明确标 `CPU_TEST_ONLY`，不能成为 native 训练证据。新 macro adapter 先
重建 review，再用完整 attempt events 生成每个 measured request 的 macro；
priming进入历史，不能计入训练标签。

每 attempt 的 startup energy 只记一次，包含启动至首个 case 前的 priming。
case energy 只覆盖10s measurement；60s drain 的能耗仅作诊断。显存使用
startup 至该 drain 的 owned sampled peak，保留 inactive residents 与历史，
不声称连续峰值、不提前加安全系数。startup和每case能耗缺少原始power
bracket时拒绝；measurement/drain时长不能修改。每个资源sidecar都留receipt。

continuation 必须逐个完成 owned execution → native review → resources →
macro，任一失败停止后续 attempt。新 intake 对15份 review、macro与资源
重建，并核全部3个完整block、390 measured +270 history、15次startup和
171个资源窗口。leaveout单位是完整block的全部5个shards；case、raw SHA、
priming或startup不能伪称独立replication。intake不fit。

## 当前范围与后续门槛

本实现仅做CPU契约验证与新producer准备，不启动GPU，也不做真实fit。
GPU执行需根对话独立审查新producer及其冻结收据，当前exact12基线仍为
唯一运行中的campaign。已有失败记录、旧CPU01/02准备和冻结执行源码保留。

日历覆盖不能证明真实overlap、OCR路径对照或12特征残差秩；完整采集也
不能自动认证拟合根、leaveout支持或独立预测准确度。macro训练与10+60s
观测不等于kernel/resource fit，也不认证90+120s窗口、startup/3600、完整
历史资源域、USHER/Roomie物理机制、P95、候选排序或分流收益。

CPU测试分别覆盖真实public preparation/audit和fake runtime控制路径。
完整intake合成测试明确mock上游native/protocol重建，实际执行文件收据、
claim核验、资源计算和390/270/171/15分母；它不提供native训练资格证据。
