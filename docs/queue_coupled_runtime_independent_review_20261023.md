# 新受控耦合运行链路的独立审查

2026-10-23，Asia/Shanghai。只读审查 `tools/queue_coupled_utility_20261023/{protocol,runtime_window,evaluation,native_runner,study}.py` 和其 policy/model/resources 接口；没有修改这些实现或冻结的 controlled_calibration，没有启动 GPU。

## 因果与评价链路

新 window 把完整到达 trace 留在 runtime，本次 dispatcher 仅收到当前 ID/scheduled time、同 phase host 队列快照和已经返回的 completion allowlist。没有把未来 trace、reference、answer、生成 token 或真实未完成 service 提供给策略。

新 `evaluation.audit_phase` 先逐条要求反馈内容严格等于既往 raw row 的 `feedback` 白名单，再用旧 `audit_phase` 检查原 cohort、SamplePool、trace/decision digest、时间单调、completion 在 now 之前且不重复、准确队列快照，并通过 ReplayDispatcher 重放显式 feedback/current-arrival。这种组合检查能够排除“仅因 raw 终局包含真实 service 就提前向策略暴露”的漏洞。

原有 warmup 与 measurement 分别构造新 dispatcher；背景任务有固定路由并登记 admission，因此之后完成反馈可加入同一个观察前缀 cohort。legacy wrapper 保留原始路由实现，仍可经新反馈 window 运行及 replay。

实测 evaluator 保持冻结原始 Q/T/L/P/M/E arithmetic、输出质量测量、原 recoverable_congested/steady 稳定性准入、原容量/显存门禁、窗口完成率与能耗完整性门禁。缺少某个 task 的观测时完整 U 应 UNKNOWN；不能将该失败归结成不存在因果审计。root 已修改为保留已经完成的 audit，同时另列 metric_evaluation_errors。

## 已报告并修复的问题

- native execute 的 checkpoint binding 或初始化错误原本落在 run 的 try 外，无法保留 terminal raw。root 增加了 failed run.json 保留。
- 新 native registry 最初没有显式固定 OMP/MKL/OPENBLAS 为1。root 在 workers 和 registry 层明确设置。
- contract 验证最初只核 resource 文件 receipt，错误 deployment/provider/params 要等 GPU 加载后才暴露。root 增加了纯 CPU provider/Policy 构造预检。
- 缺 task 的 metric 异常原本覆盖已完成的 dynamic audit。root 保留原 audit，并添加 metric 错误字段。

这些修复均由 root 完成，本次审查没有并发修改它们。

## 仍须明确的范围

观察前缀策略并未预测真实未来到达，physical M 的路由差和绝对 past/startup E 仍未知。因此其条件 score 不能称完整物理六项 U，也不能直接保证全窗口正收益。最终必须由新的完整 native 配对验证。

受控资源 provider 的 mean/p90 包含两轮输入独占基线分位，但 progress 系数没有完整估计不确定区间，p90 不是整个外部代价的90%覆盖保证。同族专用副本交换性也是明确假设，S1 没有独立 solo 校准；per-alias residual 只能诊断，不能证明交换性。

受控资源 fit 入口检查 raw SHA 与 seed/状态，但这不替代完整 batch packet 审计。实际交付仍应另外验证 source_packet_digest、320 batch/576 request 的精确声明、源码/input receipts、每个 batch 隔离、worker UNIX/monotonic 两时轴转换和整卡 power coverage。

决策在 runtime.lock 内模拟 pending，需要预先冻结确定性计算守卫或工作量预算并记录耗时。单纯 max_events=10000 不保证低运行开销；非确定性 wall-clock 截止会破坏 replay。超过支持域/预算时应完整专用回退，不能删减 pending 后称完整 cohort 预测。

## 独立 CPU 验证

新增 `tests/test_queue_coupled_independent_runtime_review.py`。测试使用真实 CPU worker subprocess、原 transport、完整新 window/Dispatcher/Policy，注入一个简单 provider 与非 native 评价，以隔离运行及 replay 而不是伪造物理资源证明。

验证内容：

- 混合四任务 measurement 与 warmup 的完整 causal replay，phase 首条反馈为空。
- 修改已绑定反馈 service 值或注入 private answer 字段，均被 allowlist 阶段拒绝。
- 去掉 coupled 合同后 legacy Dispatcher 仍按固定背景路由运行并通过完整 replay。

三项独立测试通过。另与现有三项 feedback 单元测试联合运行时通过。测试没有 GPU/system query，没有产生物理效用或独立效益验证结论。

## 实际受控校准独立审计

新增只读 `experiment_audit.py::audit_controlled(packet_path, raw_path)`，实际运行输出 `artifacts/queue-coupled-study-20261023/controlled-calibration-independent-audit-v1/audit.json`。5095项检查通过，没有错误。

审计绑定 raw SHA `f2864c82a5f144aebd841961646c2112abc24cbabb4a8d8cc0feea1983c4e4fb`，审计产物 SHA `ff2f19035f0f42591a607c15f10d05a610a788e3dfd31c9fa8df0878222dcbff`。

- 精确320批次/576文本请求，5种context各64批次，301/302各160批次；随机条件/alias顺序、pool_id、batch/response身份和逐批drain隔离全部一致。
- 原largest-three全路径priming、线程数1、原生预检、显存预算和全部worker正常关闭均通过。
- worker response end减实际service与raw记录的UNIX/monotonic interval逐项一致；host clock anchor offset跨度3064ns。
- 实际校准跨度272.710012336秒，整卡84399.08059690433J；服务区间完整覆盖，仍采用原2秒最大采样间隙规则，没有补值。
- safe memory峰值38177.830272MiB。该值是本次受控校准观测，不是以后路由不变的物理上界。

新增 `tests/test_queue_coupled_experiment_audit.py` 六项测试通过，覆盖完整自包含合成协议、source/packet篡改、batch顺序与隔离、service时轴和response身份、功率缺口不插补、正式U声明拒绝，以及opaque worker答案不影响审计也不出现在报告中。审计仅证明本次证据与协议一致，不证明外部代价因果识别、全任务模型覆盖或分流效用正收益。
