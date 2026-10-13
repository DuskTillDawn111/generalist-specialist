# Module 29：Independent GPU Replication

**最终 Gate：REPLICATION_INVALID。实验已按冻结技术故障规则停止，GPU 已释放。**

预注册 3 个新种子 291001/291002/291003，9 个 2400 秒正式窗口。实际尝试 5 次：4 次完整合法、稳态；第 5 次 G2 S-only 在 1997.196239 秒因遥测并发错误中止；其余 4 次未运行。只有第 1 组完成匹配三方案，独立配对 n=1，开发数据未计入统计。

|顺序|组|方案|状态|合法 U|G / S 目标请求|
|---:|---:|---|---|---:|---:|
|1|1|s_only|COMPLETED|-3.9145228074403065|0 / 25214|
|2|1|fixed10|COMPLETED|-4.710230534707492|2465 / 22749|
|3|1|dynamic|COMPLETED|-4.121575926832398|59 / 25155|
|4|2|dynamic|COMPLETED|-4.118393078621379|54 / 24921|
|5|2|s_only|FAILED|NULL|0 / 20803|
|6|2|fixed10|NOT_EXECUTED|NULL|— / —|
|7|3|fixed10|NOT_EXECUTED|NULL|— / —|
|8|3|dynamic|NOT_EXECUTED|NULL|— / —|
|9|3|s_only|NOT_EXECUTED|NULL|— / —|

第 5 次的请求数来自中止前缀及清洁关闭；其延迟仅诊断，不参与 U 或配对。NOT_EXECUTED 不是失败或负收益样本。

## 六个结论

1. **没有验证三组均正收益。**仅第 1 组完整合法，Dynamic − S-only = **−0.207053119**，Dynamic − Fixed10 = **+0.588654608**。该组本身未满足两项均 >+0.01 的门槛。
2. **未验证同时持续优于 S-only 和 Fixed10。**第 1 组相对 S-only 退化、相对 Fixed10 改善；第 2/3 组不构成完整效用比较。
3. **三组配对均值、样本标准差和 95% 区间不可用。**完整独立配对 n=1；只有上述单组点估计。paired_statistics 中三组统计保持 NULL，不能将两个 Dynamic 单臂窗口、时间块或开发窗口补成配对样本。即使完成 n=3，t95 区间也依赖效应近似正态，最小单侧符号检验 p=0.125。
4. **两个已完成 Dynamic 窗口均实际分流，但不足三组。**G1：G 59 / S 25155，G 占 0.233997%；G2：G 54 / S 24921，G 占 0.216216%。第三组未运行，无法认定三组分流行为及收益稳定复现。
5. **已完成配对的主要变化来自 LID。**相对 S-only，LID 的 mean/P95 效用分项分别 −0.044625 / −0.163059；共享 S 输入服务增加 1.024407 ms、真实 native 排队增加 12.677250 ms。全系统 pooled mean/P95 增加 11.392754 / 39.565405 ms。背景 Text/Image/OCR/VQA 并非本组退化主因。相对 Fixed10，LID 分项合计 +0.559406，VQA 合计 +0.024855；VQA 排队减少 170.815088 ms。服务和排队变化是匹配输入跨运行的关联，未唯一归因于物理共置干扰或运行波动。
6. **当前证据不支持据此集成正式部署优化器。**未得到可信独立正收益复现，Stage E 未触发。保持策略、参数、模型池及优化器不变，停止，不补 GPU 运行或种子。

## 原始效用、稳定性与技术停止

四个完整窗口均按 Module 20A 原始六任务效用、原系统/worker 稳定性规则计算；全 cohort 延迟包含正式窗口后的有界排空，吞吐仍采用计划 2400 秒分母。稳定性不是由最终完成自动推定。四个完整窗口均为 steady；第 5 次为 UNASSESSABLE_TECHNICAL_ABORT，U=NULL，不声称其物理容量不稳。

效用分解（第 1 组，Dynamic − 对照）：{"s_only": {"T": 0.0, "L": -0.04388159514212686, "P": -0.16309439847071827, "M": -7.712577924627595e-05}, "fixed10": {"T": 0.0, "L": 0.15643339012866764, "P": 0.4322212177464255, "M": 0.0}}。Q_norm 保持 NULL，wQ=0，原质量准入仅为硬约束。

遥测错误为 `dictionary changed size during iteration`。完整注册队列 27395；中止前计划到达 22816，接受 22815，清洁关闭后完成 22815。未来计划槽 4579，异常时未接受 1 个请求；总计 4580 个注册请求未完成，全部留在原始记录。

五个 worker 均收到 BYE、exit=0、未强杀；GPU 释放验证通过。遥测采集器只保存异常 repr，没有具体栈帧。DurableMonitor 未加 runtime.lock 的 live 请求 deepcopy 与该错误一致；这是源码支持的候选原因，未通过新 GPU 实验验证，也未修补冻结实现。见 technical_failure_analysis.md。

failure_policy 在新 GPU 前冻结为 ZERO GPU retries/replacements，并在技术故障时停止整轮；按该规则执行，未因首组负收益重跑、换种子、调 QPS/阈值、改变模型或窗口。CPU 收尾只补充诊断与完整性审计。

## 记录、口径与完整性

frozen_replication_manifest.json 的 independent_slots / replication_protocol 为本轮权威协议；复制的祖先 independent_protocol / exploration_seconds 仅为开发来源元数据。actual_window_accounting.csv 逐次核对完整窗口、运输配置均为 2400 秒，warmup 为 30 秒。冻结 manifest 字节未改。

环境与功率状态、频率、温度、NVML 0.5 秒采样高水位、原生 allocator MEMORY_END 输出均保留。采样峰值不等于不可观测的瞬时全 GPU 峰值；部分 worker 不提供 allocator 峰值，不填补缺失数据。峰值与容量合法性沿用原判据。

完整窗口曲线见 completed_backlogs.png/pdf；中止前真实 backlog 与未观测剩余区间见 aborted_backlog.png/pdf。aborted_task_diagnostics.csv、aborted_backlog_timeseries.csv 与已标记的主路由/worker 表保留中止数据，禁止作为合法效用比较。所有均值/P95/有效 n、质量诊断、执行/排队、效用分项、尾部与时间块敏感性均留档；敏感性不更改 Gate、U 或独立样本量。

历史 16831 个文件元数据、冻结源码、模型文件和完整 28R 输出 SHA256 均通过；无变化、无新增历史文件。Module 14F–28R 保持不变。正式优化器未改，未开始新 Module 或模型池搜索。
