# Predictor v2 独立 CPU 复核

日期：2026-10-27。复核对象为新增/修改的 `forecasting/workload.py`、`queue_diagnostics.py`、`simulator.py`、`__main__.py`、`runtime/batching.py`，并追到 `metrics.py` 的完整评分输入边界。这里的“独立”指第二代理独立阅读、构造反例和运行测试，不是独立模型实测验证。没有启动 GPU、扩展采集、修改冻结实验或继承旧收益结论。

结论：本轮识别并修复了停滞小 cursor 提前完成、batch 漏掉提前就绪事件、诊断二次重算、物理证据标签和评分输入边界问题。CPU 反例通过后可以继续集成；这不是完整工作完成证明，也不是预测器已达到物理精度或改善效用的证据。

## 1. 已发现并复测的问题

|问题|反例/证据|处理及复测|
|---|---|---|
|以 remaining_fraction≤1e-10 判完成会吞掉未消耗工作|cursor=1e-12 的 s 在 g 活动时显式 stalled；测量/排空均早于 g 完成，s 本应保持 incomplete|主代理按实际有限速率 completion boundary 判完成；独立 `test_tiny_stalled_cursor_cannot_complete_without_progress` 通过|
|batch 的 WAIT 只返回 head timeout|两行 ready=0/10，max_batch=2、timeout=100；now=0 若返回100会错过10的满 batch|主代理令 next_event=min(timeout,next future ready)；独立事件反例通过|
|诊断逐事件重新扫描全部 pending 并重复计算 segment work|单 alias、每请求1ms、N=100/200/400，事件101/201/401；`remaining_work` 调用10100/40200/160400，精确 N(N+1)|主代理新增 register/cache/增量总 work，事件只读队列长度/等待头及 active 消耗；相同规模调用均1，N=1000 也为1；独立调用次数测试通过|
|CLI 仅看 service provider 的 physical_evidence|measured service + cpu_fixture lifecycle 可能被标物理证据充分|主代理合并生命周期证据标识；独立 mixed-evidence CLI 测试返回False|
|CLI contract 标签与实际 lifecycle 不一致|提供 lifecycle 但省略 contract 时曾标 legacy、实际却应用延迟|现在按存在性默认 strict；显式 legacy+profiles 拒绝，独立测试通过|
|oldest-waiting 只看 FIFO 头会漏掉非到达排序的旧请求|FIFO 中等待请求 arrival=90ms/10ms，now=100ms，drain=150ms|主代理用 register 入堆、mark_started 删除等待身份，不依赖arrival顺序；独立反例返回140ms|
|评分允许测量窗外请求/完成公共身份错配|arrival=-1 或 deadline、历史 arrival<phase_start、finish arrival 不同于 cohort、finish float 或<now 均曾被接受|独立复核代理修 `metrics.py`，新增四组边界反例，全部通过|

诊断计时只作 CPU 开销量级定位，不是稳定性能基准：原 N=100/200/400 为约 .097/.371/1.498s；修后同规模约 .007/.022/.026s，N=1000 约 .064s。前后均带同一 Python mock 计数包装，硬件及系统负载未控制；核心可重复证据是调用次数从 N(N+1) 变为每配置一次。该修复不保证整个 gated admission 算法线性：许可刷新、上下文查询、完整 trace 留存及多候选多场景预测仍有成本。

## 2. Gamma 与全时段 piecewise

实现以每任务 CV 定义 Gamma shape=1/CV²、scale=CV²，使 integrated-rate 间隔均值为1。窗口跨零速率时保留同一 renewal clock；并非每个窗口重新启动 burst。中途预测要求可见的每任务最后到达时刻，或明确该任务截至当前尚未到达，按该可见年龄条件化 residual。采样调用的是独立 forecast namespace，不读取实测 future RNG 或 future output。

独立精确反例固定首个 uniform 为 .25：窗口速率2→0→4，last=.5s、now=2s，integrated age=1。预期下一间隔使用 `gamma.isf(.75*gamma.sf(1,shape=4,scale=.25))`，再从2s按速率4逆映射；代码返回的 ns 与该值一致。既有统计测试另检查 CV=.5/2 的样本间隔、条件 residual 不会重启、零速率窗口无到达。

这是一种明确的 time-changed Gamma renewal 模型；改变负载窗口时累计的是 integrated rate 年龄。它不是任意非平稳业务到达过程的无偏预测保证。尾部数值 underflow、极小间隔无法用浮点/ns 分辨时保持 UNKNOWN；未来到达预算耗尽也 UNKNOWN，未截断成成功。静态 rates 仍是评分任务权重，piecewise windows 是到达分布，调用方须声明二者的实验含义。

## 3. Lifecycle、许可、ready 与 stalled

实际评分终点为 parent receipt。handler end、permit release、worker next-ready 均保留独立事件，不用 handler end 替代前者。已有测试验证：handler100ms、receipt+20、release+30、ready+10，在 gated 模式下同 alias 下一请求130ms启动；shared 模式可110ms启动而上一请求120ms才被父进程接收。receipt 跨测量截止且超出排空时保持 incomplete。功耗只积分剩余测量窗口，不随排空延长。

显式 `known/context_supported=True, stalled=True, service_ms=None` 仅表示假设模型的当前进度0；内部下一完成时间为无穷，输出 JSON 的 service_ms 为 null，未输出 Infinity。其他 kernel/到达/生命周期边界发生时重算。所有工作永久停滞且无可解事件时仍 UNKNOWN，不凭空完成。极小正 cursor 不再通过完成容差被吞掉。

严格 lifecycle 需要训练绑定、相同 execution context、每个必要边界的证据。缺 worker-ready 不会变成0或 parent receipt。初始观测恰处在 post-handler/in-flight receipt/held-permit 区间的完整 snapshot 协议尚未实现；当前明确拒绝无法表示的初始 handoff 状态，不声称覆盖任意在线切入时刻。CLI 的 strict contract 缺 lifecycle profiles 时显式拒绝。

## 4. 诊断与评分语义

remaining solo work 保留当前 segment cursor，和真实共享完成时间不是同一量；normalized-resource×ms 不是 bytes、GPU 指令量或吞吐。资源未知时对应资源 work 不会补零。offer envelope 是当前完整场景的经验窗口最大值，窗口为 `(t-window,t]`，不构成 deterministic burst bound。HOL 只累计 strict FIFO 的首票不可准入、但后票可准入的时间，不允许后票绕过。backlog recovery 指队列清空，初始已有 backlog 标为 left-censored；这不是稳态证明或额外效用惩罚。

优化后的 oldest-waiting 用到达最小堆与等待身份集合维护，不依赖 FIFO 列表按 arrival 排序；每个请求最多一次入堆/懒删除，增加 O(N log N) 总开销，避免每事件扫描全部等待请求。未把诊断输出用于改变原 rho 稳定性规则。

评分现在严格要求 `phase_start<=arrival<deadline`，包括完成历史；预测完成必须和 cohort 的 request_id/task/alias/pool_id/arrival_ns 一致，映射key也须一致，完成时间必须为整数且不早于 visible now/arrival。parent receipt 截止判断保留原 `finish<=deadline` 约定；没有擅自改变原窗口计分定义。排空未完成时 L/P 保持 UNKNOWN，T 仍可计算；Q 使用冻结 family/task 聚合质量，没有读取请求 label/输出。历史 memory peak 通过 max 与预测未来 peak 合并，不能当公共加法常数取消；缺公共历史 energy 可以做同分母比较，但绝对 U 仍未知。

旧 `test_missing_quality` 曾只改 cohort alias，未同步完成 alias。该 fixture 已同步两处以保持公共身份一致，仍验证未知质量不能补零；没有放宽或回避新校验。

## 5. 仍未完成的范围

- Kernel static occupancy→progress 仅显式 opt-in 假设，物理吞吐、DRAM/L2竞争与误差界未被本轮测试验证。
- batch helper 的形成/选择正确不等于任意 native 后端已执行共同 batch；真实 backend、padding峰值、生命周期和许可合规必须有独立执行证据。主代理仍在集成 batch provider/simulator，本复核不将后续未检查改动提前视为完成。
- 只复核当前代码和小型 CPU 场景，未证明任意规模/所有 alias 拓扑无死锁，未给全候选域最优证书，未改变部署策略物理资格。
- strict lifecycle 的任意初始 handoff snapshot、实际资源隔离和实证增益均未完成。

## 6. 可重复验证

独立新增 `tests/test_predictor_v2_independent_review.py` 共12项，全部通过。连同既有 full horizon、queue prediction、lifecycle、kernel adapter 测试，运行：

```sh
python -m unittest tests.test_predictor_v2_independent_review tests.test_full_horizon_forecast tests.test_queue_prediction_v2 tests.test_forecast_lifecycle_v2 tests.test_kernel_provider -q
```

本轮以上联合命令在新增最后两项反例前84项通过，新增后的独立模块12项通过。最终联合测试数量与后续 batch 集成结果应以主代理合并后的复测为准，不能从本报告外推。

## 7. 只读能力审计入口与真实审计结果

新增 `tools/predictor_validation_20261027/capability_audit_v2.py`，仅重解析既有 SQLite、读取 `/proc/driver/nvidia/params`、本进程 CapEff 等字段及 proc GPU metadata。脚本不导入 torch/CUDA，不调用子进程、nsys/ncu/nvidia-smi，不改权限，不加载模型。`--output` 必须为不存在的新目录，也拒绝将新输出放在旧证据目录下。输出含原文件 before/after SHA256 清单、历史 receipt hash 对照、当前源代码 hashes 和新产物 manifest。

运行：

```sh
python -m tools.predictor_validation_20261027.capability_audit_v2 \
  --output artifacts/predictor-capability-audit-v2-20261027
```

新产物：`artifacts/predictor-capability-audit-v2-20261027/audit.json`、`reparsed-kernels.json`、历史 CUDA device metadata、当前只读 proc captures 与 `output-manifest.json`。结果为 `CPU_READ_ONLY_AUDIT_COMPLETE`，`legacy_files_unchanged=True`，历史 receipt hash 对照为 MATCH。

审计确认：

- 旧 Nsight Systems 返回0，SQLite重新解析得到6个实际 CUPTI kernel，6个均带 block/grid；这不是新采集或目标模型测试。
- 旧 stdout 明确设备是 **NVIDIA GeForce RTX 4090**，不是 H600；历史设备表只含 ordinal/pid/uuid/SM数等，其中 uuid 为 null、SM数128，不能凭当前 minor/名称自动绑定历史设备。
- 旧 NCU 返回1并明确 `ERR_NVGPUCTRPERM`；当前 `/proc` 为 `RmProfilingAdminOnly:1`、`CapEff=00000000a80425fb`、CAP_SYS_ADMIN(bit21)未置位。UID0不代表具有该 capability。这些元数据与旧拒绝一致，但没有重新测试当前 counter access，状态仍为 `UNKNOWN_NOT_REPROBED`。
- 6个 kernel 均缺架构取整 allocation；完整 residency limits 也缺证据，occupancy保持 UNKNOWN。
- 没有传入训练绑定的目标模型 unprofiled-isolated 有序 kernel 基线；不能把 tiny synthetic probe 迁移成模型预测证据，compute/DRAM互补能力保持 UNKNOWN。

入口新增6项CPU测试，覆盖不执行进程、旧文件字节不变、新目录拒覆盖、geometry保留/架构分配未知、proc缺证据、capability存在也不冒称访问成功及产物manifest。加入之前所有针对性模块后运行125项通过；`compileall` 与 `git diff --check` 通过。审计完成不改变前文未完成范围。
