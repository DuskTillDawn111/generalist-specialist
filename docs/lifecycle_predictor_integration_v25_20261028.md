# 生命周期条件与全过程队列预测接线

v25 将现有模型中“驻留上下文、活跃请求、CUDA 工作已经提交、GPU 节点已就绪”分开。原 v16 会在 balanced 公共队列没有 G2 请求时反复选择 G2，并把专用模型的就绪工作置为零速率；完整四候选均未在原排空窗口内完成。这个问题不能通过限制全局并发、缩短队列或改评分门槛修复。

## 已接入的计算路径

`learned_lifecycle_forecast_v25.py` 使用原来源验证过的 DAG、孤立参考工作、host release、语义 phase、parent receipt/worker ready、板级资源和 startup。它新增以下组合：

1. `owner_lifecycle_engine_v21.py` 在同一时刻完成 arrival/handler/节点/receipt/ready 闭包后，给 scheduler 传入 handler、已到达 FIFO 与三值 lifecycle。尚未释放的 DAG 节点不能直接算成已提交 CUDA 工作；没有对应证据时保持 unknown。
2. `owner_lifecycle_mark_v21.py` 以同一状态格式训练/推理。保留全部物理上下文类别与 715 个非 ready-frontier 目标；活跃参考服务目标单独报告。每个新引擎重新构造 mark，避免继承上一候选的观察时钟。
3. `owner_lifecycle_hazard_v24_fit.py` 与 validated query 重建完整配对上下文风险时间，包含空 ready 集合与无活跃竞争者区间。最终风险集为 559130 段、22383203487ns 暴露、35799 个 SAVE_END。原 35737 个有 mark 的事件身份全部保留，另 62 个只有合法 SAVE 端点但缺 next-owner 监督。旧 hazard 遗漏的 133 个空 ready SAVE 已恢复。
4. `cumulative_lifecycle_hazard_v24.py` 按旧 lifecycle/旧 frontier 积分到观察时刻，然后更新未来状态。每个已知存活 RESTORE 只抽一次条件风险预算；保留原 RESTORE_START 年龄，不补算进入前未知历史。空 ready 区间调用已训练的完整风险模型，不再暂停风险累计。整数 crossing、age bin、补偿求和和固定大小状态均保留。
5. `learned_lifecycle_oracle_v25.py` 在 frontier 或 lifecycle 改变时更新累计模型；mark 只在预测 SAVE 时查询。初始联合 prior、handoff 候选和二值参考推进假设继续明确标注。same-holder reacquisition 仍为 UNKNOWN，不虚构 grant 或延长 HOLD。
6. 全公共队列诊断进入实际 engine 调用：完整未来 arrival/input 的 offered-work envelope，与当前已到达 FIFO、剩余参考工作、oldest age、恢复时间分开。原评分拒绝时，v20 wrapper 保留真正返回的未评分 forecast，仅作为诊断。

六项原评分入口仍是 `AvailabilityForecast.execute` 与 `score_complete_forecast`。没有改变路由、请求分母、P95、startup 能耗摊销、显存安全系数、rho、g(n) 或稳定性。当前没有滚动反馈纠偏或 learned scheduler checkpoint 支持；这两个缺口不因存在生命周期观察接口而自动消失。

## 训练与 CPU 证据的适用范围

推荐 mark 是 `predictor-owner-lifecycle-mark-v21-20261028-training03/model.json`；hazard 是 `predictor-owner-lifecycle-hazard-v24-20261028-training03/model.json`。最终实现修正了基于未来 worker END 不确定性提前改变当前状态的问题：状态只由已发生前缀事件改变，时钟误差单独记录；frontier 和 gpu_ready 必须一致。

mark 两个 capture holdout 的物理标签正确率约 99.949%/99.939%，非 frontier 分别正确 357/364 和 340/351。hazard 三次最终拟合均收敛；完整训练在无活跃竞争者的 8.849674422s、0 个观测 SAVE 上仍预测约 2.272 次事件，保留有限非零风险。旧正则失败模型在对应旧分母上预测约 7891 次。两者不能单靠总事件数比较：最终空 ready 的 holdout 仍有 57.24/71 和 84.81/62 的条件误差。它们是训练 capture 留出诊断，不是新的独立 GPU 预测准确率。

root 累计控制器、oracle 和 factory 小图共 13 项 CPU 测试通过。独立审阅覆盖额外解析 crossing、条件进入、fresh mark 与因果/source 边界。source guard 直接读取实际文件 SHA；当前文件系统上，文件同长度变更可能连 mtime/ctime 都不变，因此不能只信 stat cache。旧失败报告和原执行源码快照保留。

`predictor-reference-feasibility-v22-20261028-cpu01/report.json` 在去掉切换开销、每个活动 GPU 节点独立参考速率一的诊断中，四组均完成全部 57 个请求，最后 parent receipt 为约 29.765s/33.083s/29.765s/29.765s。这个放松模型没有物理容量保证，也没有 M/E 和六项分数；它只说明冻结参考工作/host/FIFO 本身并未使原 150s 排空期限必然不可达。

## 当前执行顺序

先在 `predictor-lifecycle-v25-20261028-preflight01` 跑 balanced 原 57 请求、原 30s+120s 全过程。完整结果用 gzip 无损保存并逐字节解压 SHA 校验，避免再次产生数 GB 重复诊断文本。若仍 UNKNOWN，保留原理由和实际引擎证据继续定位；不扩大 drain、不删请求、不强制 sole-ready。

完整四候选数值预检及独审通过后，才冻结源码并重新 fresh 计算预测，随后进行独立 GPU 预测验证。联合 host/lifecycle 训练采集属于训练补证，不是冻结后的验证；新 v23 入口使用新训练请求身份及训练输入，独立审阅和存储条件就绪后按先训练后验证顺序执行。之后才是 batch/admission 等机制消融与公平四组分流比较。USHER/Roomie 的物理资源、真实公共权重、batch/cache 收益仍需各自后端与画像支持，不能由这一轮队列训练结果代替。

## 实际全队列预检及后续修复

v25 balanced 已运行退出，完整保留 57 个公共输入，但在 10.748970283s 返回 `ENGINE_SKIPPED_PREDICTED_SAVE_BOUNDARY_NO_REWIND`，没有得到可评分预测。诊断发现：oracle 在 lifecycle 改变时更新 hazard，而继承的 scheduler 只在 frontier 改变或旧 deadline 到达时采纳新 review。因此同 frontier 的 host/lifecycle 变化可提前 SAVE 却未通知实际事件时钟。

`learned_lifecycle_scheduler_v27.py` 显式通过原 `_accept` 接纳更早 review，并保留真正错过边界时的 UNKNOWN。小图复现了 v25 的 693148ns 旧边界与 159315ns 新候选差异，v27 在新边界触发真实 switch；空 frontier、pending handoff 不可变、预算不重抽均覆盖。当前只有 CPU 证据，不代表物理推进假设已经正确。

独审另发现 v25 声明来源列表漏了 `learned_availability_forecast_v20.py` 与 `idle_owner_selection_audit_v20.py` 两个传递依赖。旧源码与执行快照保留，原报告旁新增 `source-claim-correction.json` 限定来源检查范围。v27 factory 改用以自身入口为根的静态 AST 导入闭包，包含这两个文件；不把旧检查追认成新完整闭包。

v27 完整 balanced 重跑已结束：通过原边界错误，在 14.819039758s 以 `UNSUPPORTED_SAME_CONTEXT_REACQUISITION_PREDICTION` 返回 UNKNOWN，当前 holder 为 OCR。完整证据保存在 `predictor-lifecycle-v27-20261028-preflight01`；没有分数或验证资格。191 个完整静态闭包来源核对通过，独审共 22 项 CPU 检查通过，但数值/物理问题仍在。后续核查 OCR 训练类别覆盖与真实 same-owner 转换支持，不通过删 guard 或强制其他 owner 凑出 numeric。
