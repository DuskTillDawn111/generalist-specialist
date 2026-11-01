# 实际 DAG 路径的队列工作量与整数事件时钟

2026-10-28。新的 `cuda_dag_simulator_v6.py` 保留 v4 原文件，修复大绝对时间戳和微小浮点剩余工作相加时的零时间推进。每次先对相对剩余服务时间向上取整到整数纳秒，再加整数时钟；完成判断使用同一个相对边界。每次取整增量小于 1ns，不代表整请求累计误差小于 1ns。事件预算和缺证据处理不变。

独立 CPU 反例保存在 `artifacts/predictor-dag-large-clock-independent-counterexample-20261028-final02/report.json`：在 30s 时钟上，10000.0000018ns 活动会让旧引擎反复保留约 0.0000018ns 工作而不推进。独立 v6 检查 `artifacts/predictor-dag-integer-tick-independent-review-v6-20261028-final02/report.json` 对六种场景、四种时钟平移核对通过，并绑定最终含 queue hooks 的引擎源码；它不是 GPU 准确率验证。final01 绑定较早源码，原报告保留。

真实 campaign 的同一故障也已复现：`artifacts/predictor-dag-actual-campaign-clock-review-20261028-final01/report.json` 在第 827078 事件、clock=17179970365ns 捕获 text#0/gpu:19566 剩余 1.3822377695760224e-6ns；此处浮点 ULP 为 3.814697265625e-6ns，下一完成时间等于当前时钟，只有 30/57 请求完成。原 v4 源码未改，sealed02 的 5M 事件耗尽结果完整保留。

`dag_queue_diagnostics_v6.py` 已通过 `forecast_gate_v3.py` 注入真正运行的 host-supply DAG 分支。它在真实仿真 arrival、START 和每次工作消耗前更新，包含：

- 已到达请求尚未完成的 GPU、copy、memset 工作，保留部分执行进度；新请求加入完整工作，活动只消耗本次实际进度。
- 所有公开未来请求的 0.1/1/10s 多尺度到达数与工作量 envelope，保留完整 offered cohort；缺工作不能以零替代。
- 每 owner FIFO 的最大等待数、最老等待年龄、队首阻塞时间和全部等待请求的时间积分，分别记录 handler 活动与 handoff/not-ready 阻塞。
- 等待队列出现积压至清空的恢复片段及未清空积压起点。

这里的工作量是拟合独占活动耗时之和，允许相互重叠，不是请求墙钟、SM 指令数或 DRAM 字节量。SM/DRAM/L2 加权工作只接受另外提供的归一化需求；当前 gate 没有这种硬件证据，继续输出 null，不能从 occupancy 制造吞吐。固定路由 owner FIFO 与全局准入 HOL 分开；未增加全局 cap 或绕行行为。

模板数值只按 `(alias, task, pool_id)` 缓存一次，再由执行和诊断共用。适用于当前固定生成工作、固定场景的 exact-content 模板；不能直接套到同一输入具有不同生成工作或上下文相关时长的模型。baseline/copy coverage 的 query 数量明确标为 unique-used-template nodes；请求执行实例数量另列，不能互换分母。

七项新增诊断测试包含可手算的长请求阻塞短请求（最大剩余工作 125ms、等待/恢复 75ms），以及双 owner、资源分时和独立 handoff 的全请求核验：FIFO 等待积分等于逐请求 queue wait 总和，开启诊断不改变任何 START/finish。包含新引擎、gate、未来工作量和原六项评分的九模块回归共 110 项通过。

未完成项仍包括实际可见状态的 DAG continuation/滚动校正、恢复风险的物理训练与接线、真实 batch/padding 路径的物理验证，以及独立 GPU 预测门槛。此次接线不改变原六项评分、rho/g(n) 或稳定性规则，也不构成物理共驻互补、缓存收益或最终分流收益声明。
