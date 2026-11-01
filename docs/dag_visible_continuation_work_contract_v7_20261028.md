# 实际 DAG 可见状态续算与滚动校正：下一阶段实现契约

2026-10-28。本文起草时为待实现契约；随后 sealed03 四候选 GPU 验证已完成，科学门槛 FAIL、源码冻结检查 PASS，才解除源码冻结。候选、情景、阈值、评分及旧产物不改。下方状态更新区分已完成的数值续算与仍待实现的实际反馈校正，不是预测精度改善报告。

## 当前实施状态

新增 `cuda_dag_simulator_v7.py` 和 `dag_checkpoint_v7.py`，保留 v6 原文件。`forecast(..., pause_after_event_count=N)` 在第 N 个自然原子事件边界导出显式 simulated checkpoint；后续使用相同的原请求、原 now/deadline/drain 和 provider 绑定，调用 `forecast(..., checkpoint=...)` 续算。没有新增人为时间分割点，避免改变整数 ceil 的数值路径。checkpoint 保存当前事件预算、所有未完成节点/队列/receipt/ready 堆、累计资源和完整队列诊断。显式原文件 receipt 和 provider 实现源码摘要发生变化时拒绝续算。provider 须为确定性函数，其完整模型证据由调用者声明；接口检查声明不等于证明声明的完备性。

相关三模块 34 项 CPU 检查通过，包括所有自然暂停边界、反复 JSON 序列化续算、同时间事件、部分执行、双 owner、多 stream/copy、receipt/ready 顺序、measurement/drain 截止和大时钟。实际原 OCR 训练 DAG 的 7915 个节点、12357 个事件在事件 1、100、3000 暂停并重载后，与完整 v6 数值结果逐字段相同；资源累计核验初版的实际 DAG 报告为 `artifacts/predictor-real-dag-checkpoint-v7-20261028-final03/report.json`。它使用明确声明的 profiled replay 作数值检验，不是训练服务预测、自然时延或 GPU 精度验证。原 final01/02 与全部反例作为此前版本的结果保留。

独立子对话复验了 18 个 seed、477 个自然边界，逐字段等于不暂停结果和原 v6，见 `artifacts/predictor-dag-checkpoint-independent-v7-20261028-final03/report.json`。独审此前发现的 provider 身份/配置和内部状态反例均保留；最终重验八个反例全部拒绝，见 `artifacts/predictor-dag-checkpoint-counterexamples-recheck-v7-20261028-final01/report.json`。绑定现包含选中 callable 的源码/实现、闭包/default/partial 参数和可编码的实例配置；诊断器递归绑定其真正的 provider。恢复时另核原 cohort、arrival/FIFO/依赖与生命周期分区，并从 fresh provider 重推诊断缓存。校验运行前后生产源码一致；这仍以确定性 provider 和调用者声明的完整不可变证据为前提，未证明任意隐藏全局状态的完备性。

后续资源核验发现并修复两类真实反例：重签累计 energy 可与保存历史不符；伪造连续 resource transition 可跨过已保存的 handler START。最新实现从 START/handler END 重建每段 owner 状态，以 fresh 绑定 provider 核 watt/memory，要求历史连续、不跨 handler 或 measurement 边界，独立重算 measurement 能量、through-drain 峰值及 known flags。未观测资源不能变成零。最终独审 `artifacts/predictor-dag-checkpoint-independent-v7-20261028-final05/report.json` 记录 477 个原正常边界及 198 个补充边界（unknown/None/known 资源、零时长 handler、大时钟）；`artifacts/predictor-dag-checkpoint-counterexamples-recheck-v7-20261028-final02/report.json` 记录全部 17 个原反例被拒。原失败和测试构造失败报告保留。该资源核验不改变 v6 正常数值与六项效用。

真实 visible-host reconciliation、不可见 GPU 进度的条件残余识别、冻结的在线纠偏策略、滚动独立验证和风险校准仍未完成；生产 gate 当前也未启用该续算模式。不得把 checkpoint 的预测剩余工作称为实际观测进度。

## 当前执行路径与先后关系

实际调用路径是 known-input forecast gate → forecast_gate_v3 → cuda_dag_simulator_v6。该路径已接入 DagQueueDiagnostics：完整公开未来到达、部分执行后的剩余活动工作、多尺度 envelope、owner FIFO 等待与队首阻塞、积压清空片段。SM/DRAM/L2 需求未知时保持 null；当前没有可见状态续算或校准后的长期风险。

先完成当前四候选封存、精确路由与57请求核验、顺序采集、实际生成工作核验及原门槛评估。若门槛失败，先分解服务、主机供给、handoff、生成工作或资源误差，再决定模型改动。不得在看到当前结果后换 primary host scenario、删请求、改阈值或用同批标签拟合后重报独立通过。

## 第一步：实现可复现的仿真 checkpoint/continuation

在新版本引擎中增加显式状态导出与导入，保留 v6 原文件。状态至少绑定引擎和全部 provider 的源码/模型身份、执行身份、原完整 cohort、原测量与 drain 边界、整数时钟、事件序号及事件预算消耗量；包含未来到达堆、各 owner FIFO、running 请求与原 start、节点依赖计数和已完成集合、active 节点剩余工作、release/receipt/ready 堆、已知 START/finish、累计资源及诊断状态。导入不重置预算、不重算完整活动工作、不把活跃请求重新排队或重发，也不改变同时间事件顺序。

checkpoint 必须声明来源 simulated_checkpoint。它只证明同一数值模型可续算，不是观测到的 GPU 进度。只在预声明的原子事件处理边界导出；若将来允许同时间事件处理中途暂停，必须另存处理 phase/游标，单凭 clock 和 sequence 不足以避免重复 START/receipt。完整性摘要之外，还需验证状态内部关系：一请求仅有一个一致的生命周期位置；active、ready、done 不冲突；依赖和剩余工作有效；queued/active/finished 与全 cohort 一一对应；已弹出的事件不能复活；所有 alias/task/pool 与原路由相同。

必要 CPU 验证：连续执行与在 arrival/release/部分 activity 完成/handler end/parent receipt/worker ready/measurement deadline 分别暂停再续算逐项一致；覆盖同时间事件、双 owner、多 stream、copy、drain 截断和大绝对时钟。检查 START、finish、六项输入及剩余工作守恒，而非只比较请求数。损坏状态、重复事件、跨执行身份和总预算超限必须拒绝。

## 第二步：把真实可见 host 事实与预测的设备进度分开

沿用 observed_state.v2 的 reader-receipt 可见性：只有收到时间不晚于 cutoff 的 START、PHASE、HANDLER_END、RESULT、WORKER_READY 才能用于该 cutoff 的状态。worker payload 中较早的 perf 时间只说明已收到事件的历史发生时间，不能提前看见事件。公开全过程 schedule 可以在开始时已知，但实际输出、生成 token/forward 工作和未来反馈不可提前读取。

必须分清三类输入：真实可见 host 事实；仿真 checkpoint 的设备进度假设；额外经独立核实的真实 GPU cursor。没有第三类证据时，kernel cursor 仍 UNKNOWN，不能用 elapsed/原时长比例或 occupancy 宣称已完成多少设备工作。当前 collector 的 RESULT 和 WORKER_READY 也不能互相替代。

对于已 START、未收到结束的请求，续算需要冻结的服务/设备状态假设及其训练域支持。若旧预测已结束而真实结束尚不可见，不得把剩余工作设零、伪造完成或无限延长最后一个节点。在 cutoff 通常无法区分“已结束但通知未到”和“仍在执行”；除非独立证据排除其一，应保留二者的可行状态集合。cutoff 之后收到的事件不能回填为当时可见事实。条件残余模型须同时绑定服务状态与 reader 延迟/丢失的观察模型；未经校准的情景不自动成为概率。训练支持的条件残余分布或明确声明的可行状态情景可用于预测，否则返回该请求/情景 UNKNOWN，并说明不能生成完整精度结论。仿真续算接口本身不能填补这个识别缺口。

## 第三步：接入冻结的滚动预测策略

预先声明 cutoff 网格或触发事件、允许使用的反馈、情景及组合规则。每次输出绑定 cutoff、已消费反馈序号、可见状态摘要、原完整 cohort 和原评分窗口；不重新生成或重路由已经出现的请求。旧预测、校正预测和实际观测分别保留。原本已知的未来到达必须继续全部使用，而非只预测当前积压。

评分仍使用原 Q/T/L/P/M/E：arrival→parent receipt，完整 offered cohort，原 P95 算法；T 固定原 measurement deadline；M 为 startup-through-drain 的 sampled owned peak，仅乘一次安全因子；E 为原 measurement board energy 加 loading/priming 按3600秒摊销并排除drain。滚动时 E 必须来自同一原测量窗口的已观测 prefix 加预测 suffix，startup 摊销仅一次；M 为原 startup 至 cutoff 的已观测峰值与预测 suffix 峰值取 max 后仅乘一次安全因子；P95 从原请求明细重算，质量从原 numerator/denominator 重算，不能平均各段 P95 或质量比率。cutoff 不能成为新的 phase_start。cutoff 之后的预测不得读取未来样本；缺失 prefix 保持 UNKNOWN，未观测过去量不补零。滚动风险输出不改变原 rho/g(n) 或效用权重。

独立验证同时比较原冻结预测与滚动预测，在相同 cutoff/请求子集上报告服务、剩余完成时间、队列深度/等待、恢复、P95 和 ΔU/排序；不能按事后效果挑 cutoff 或只报告有利子集。当前 known-input pilot 保持一次性预封预测，不追认成滚动验证。

## 第四步：恢复风险与真实批处理分别补证

Sinan 启发的风险模型已有库实现，实际 DAG 目前只提供积压与恢复诊断。应先从训练 raw 构造上述 cutoff 可见特征，再用 cutoff 之后固定 horizon 的真实恢复/违约作为标签。run/group 分离，冻结后独立评估校准与覆盖；短期事件预测、长期风险和原稳定性规则分别报告。四候选短 pilot 不足以自动构成风险训练集。

batch 等待、padding、microbatch、公共权重/缓存成本需要与真实执行 backend 及相应画像绑定，之后再接同一 DAG 事件路径。不能把已有 CPU batch selector、共同子矩阵实现或活动耗时缓存称为 GPU 上已验证的 batching/cache 收益；当前 duration 模板缓存只是 CPU 预测加速。

这些工作完成并经独立验证后，才扩展机制消融及原/优化执行方式×禁/允分流四组对照。主要收益仍是优化执行方式内，允许分流相对最优禁分流的增益。
