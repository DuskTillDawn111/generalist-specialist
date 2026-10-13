# Module 31：预算与停止规则

**状态：待批准；本阶段授权 GPU 会话=0，实际启动=0。** 下列未来预算固定为审阅提案，不能视为执行许可。已核对的原窗口/超时/显存规则继续生效；新增规模和 deadline 明示来源。

## 预算表

|阶段|计算规则|最多会话/attempt|formal 秒|状态|
|---|---|---:|---:|---|
|粗扫|3 level×2 profile×2 seed×2 arm|24|14400|新两轴组合，待批准|
|细扫|每 profile 唯一 midpoint×2 seed×2 arm|8|4800|条件性，不回收未用额度|
|探索复核|每 profile 唯一 level×3 seed×2 arm|12|7200|条件性，不回收未用额度|
|探索总计|24+8+12|44|26400|7 小时 20 分 formal；待批准|
|确认性三臂|固定 0.80 steady×3 seed×3 arm|9|5400|1 小时 30 分 formal；另行批准|
|未来总计|仅双方预算均另行批准时|53|31800|8 小时 50 分 formal；本阶段不可执行|

每个 attempt 在 prelaunch 检查开始前登记；prelaunch 失败、加载失败、prime/warmup 中断、formal 早停都消耗其唯一 slot 和 attempt，不重试、不重命名、更不能把失败占用移给另 seed。预算 ledger 同时记录实际 elapsed、formal 实际观测秒及冻结的 planned 秒；截断/失败仍保留 600 秒计划分母，不能延长窗口补满。

沿用每 slot formal 600 秒、warmup 30 秒，每 phase drain≤60 秒，prime 每支持路径 3 次，idle 各 2 秒；prelaunch cooling≤300 秒、resident cooling≤180 秒，单 worker READY≤300 秒、shutdown 配置 15 秒。原五个 worker 的启动 timeout 是逐 worker，最坏不能当作总 300 秒。显存预算 34567.5534 MiB、安全因子 1.110853955254881 不能提高。原 release 核验 timeout 15 秒、相对 baseline 显存容差 64 MiB 保留。

**新增 wall 提案（待批准 P02）**：prime 总 wall≤300 秒；每 attempt（从 prelaunch 到日志关闭及释放）wall≤3600 秒；探索累计 wall≤158400 秒=44 小时，确认性累计 wall≤32400 秒=9 小时，双阶段合计≤190800 秒=53 小时。包括 cooldown、模型启动、监测、失败和清理，不因 formal 未使用而放宽。执行前新 harness 必须实现包含清理预留的监督截止，不能等到 wall 到期才开始无限等待；具体终止/清理实现及 hash 需 P04 CPU 审核。登记下一 slot 前保证其最多预算可容纳，否则 stop，未运行标 `NOT_EXECUTED_BUDGET`。GPU 占用时间、wall 时间和付费金额不是同义；货币预算/设备可用时段 UNKNOWN 待批准 P02，不自行给出单价。

没有本阶段额外 capacity pilot、harness GPU 烟测、质量 GPU 校准、失败重试或增加 seed 额度。未来若另行请求 GPU 功能验证，必须先在独立协议中预算，不借这 44/9 次名额隐性追加。

## 规则优先级

1. **当前阶段结束规则**：七份文件验证并提交后停止，等待审核，不执行任何未来 slot。
2. **全局安全/基础设施停止**：任一 OOM、显存或 safe-memory 超限、foreign PID、worker crash、request error、错 task/path、语义执行配置变化、必要日志缺失/重复/不一致、错误时间戳、journal write failure、warmup未完成、drain timeout、无法清洁关闭/释放、原热态条件/同组≤5°C匹配失败、hash变化或 deadline 到达，停止整个已批准 campaign，保存已完成/中断/未执行清单。仅清理本 campaign 的 owned worker，不杀外部进程。没有自动恢复重跑。
3. **正常科学 unstable**：完整 600 秒、clean drain/关闭、日志/资源都合格，但原尾部 AND 或其他真实科学证据触发 unstable：保留 U=NULL，不自动跳过同条件安全可运行的已注册另一 arm 或第二 seed。不能以科学失败规避过载救援配对。
4. **更高粗网格停止**：同一 profile、同一 coarse level 的两个 seed、两个 arm 共四会话全部原 hard-unstable，完成该 level 的安全组后跳过该 profile 更高 coarse level，标 `NOT_EXECUTED_PROFILE_UPPER_STOP`。另一个 profile 保持原顺序，除非全局异常。仅一 arm unstable 或混合 seed 不触发该停止；不能基于收益跳过高点。
5. **唯一细扫**：各 profile 至多一次。仅最低相邻已完成 coarse 对中同一 arm 的低点两 seed 非 unstable、高点至少一次 unstable触发；中点仅 0.70/0.90。基础设施失败触发全局 stop，不能用来定边界。没有相邻转换就不追加；停止高点后仍可按已完成相邻点执行该次已注册的更低中点。
6. **唯一探索复核**：各 profile 最高已完成共同合法通过点，3 枚新固定 seed；不存在点就跳过。复核不通过不向下重复、不新加 seed，也不更改参数；正常科学失败仍完成安全可执行的已注册复核组。
7. **阶段间停止**：探索无论成功失败，输出全部表/ledger/gate 后停止等待审核。确认性阶段不自动从 execution_order 最后一个 block 继续，必须有单独授权。
8. **确认性停止**：固定 0.80 steady、三臂三组；正常科学 unstable、零分流、样本不足或 ΔU 阴性都保留，若安全仍完成已注册组，以免选择性删掉负结果。任一全局安全异常立即停止；任何缺组/违法/不匹配使正式证明门槛不通过。不依据效果正负追加或提前宣布成功。

原 plan 的 protective_stop 含 outstanding 10000/service-demand 60 秒/oldest 20 秒，但 Module28/29 主 phase **没有调用该 guard**。本协议不把这些字段伪装成已经实际使用的在线停止阈值，也不擅自启用新科学截断标准。host queue max_pending_per_model=10000 是原传输容量限制，queue full仍是 request error。需要新的保护阈值须结果前独立审核修订，不能由本次结果决定。

## 异常账本与统计处理

每 slot 必须保存 planned 全 cohort、sent、success in-window、success during-drain、error、incomplete、unsent、shutdown/release 状态；记录停止时点、触发原规则、实际与planned时长。没有正常 formal 全窗不能标为完整 stable/steady。未运行保留 slot/seed/条件和客观理由，不伪造请求结果。

时间戳不跨时钟相减；B 使用原 host submit perf_counter，native START/END 与 Unix映射只用于离线时序诊断。缺 worker ACK 或 prediction 字段标 UNKNOWN。迟到响应不改变已冻结的 drain timeout/incomplete。数据错误可做只读追查，不能覆盖 raw、重放 GPU、替换 seed 或修改原 classifier。

failed/unstable U 与合法 ΔU 均 NULL；仅完成子集延迟标 CENSORED。样本数不够不能补请求、延长600秒、合并不同 seeds凑72，或按任务成绩删样本。分类 raw、警告和实际恢复逐项报告；发生非单调分类不事后强加单调拟合。

所有跳过、阴性、零 G 分流与异常公开保留；预算未用满不构成追加理由。任何后续修订须结果前新版本、标出差异并重新审核，不能追溯修改 Module28–30 或本冻结版。
