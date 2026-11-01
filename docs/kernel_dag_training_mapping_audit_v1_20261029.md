真实八请求的 canonical kernel↔CUDA DAG 映射已经通过 CPU 重建校验。结果为 `PASS_CPU_MAPPING_ONLY`；这不是 native 动态预测资格。证据在 `artifacts/predictor-kernel-dag-training-mapping-v1-20261029-cpu01/report.json`。审计器为 `tools/predictor_validation_20261027/kernel_dag_training_mapping_audit_v1.py`，测试为 `tests/test_kernel_dag_training_mapping_audit_v1.py`。没有改动已审 serial response v1、features v2、现有 DAG 引擎或运行中的采集/等待闭包，也没有启动 GPU、profiler 或拟合。

输入是 strict review PASS 的原始 light-kernel training capture 和保存的 `request-cuda-dag-v2-20261028-training03/profiles.json`。raw SHA 为 `87a3f416d952aa15fb967ba3393c0d4110457b74b7c20180132f79f3fe40c11b`，execution context SHA 为 `fc20ee9a472633fcc3d0f67dc706585cf8b7ae0f230ae249634817cfc57a7388`。四个 owner 都从 SQLite 只读重新执行现有 `audit_sqlite`，再执行现有 `request_plan`，逐请求完整内容与保存的 DAG 相等。canonical kernel 从 NVTX ranges 加 unattributed once 读取，没有重复拼接 phases。

| 请求 | owner | canonical kernels | DAG nodes | DAG edges | kernel streams |
|---|---|---:|---:|---:|---|
| profile:0 | text#0 | 19,423 | 19,937 | 19,983 | 7 |
| profile:4 | text#0 | 19,423 | 19,877 | 19,927 | 7 |
| profile:1 | text#1 | 19,423 | 19,937 | 19,983 | 7 |
| profile:5 | text#1 | 19,423 | 19,877 | 19,927 | 7 |
| profile:2 | generalist_2#0 | 11,552 | 13,438 | 15,078 | 7 |
| profile:6 | generalist_2#0 | 20,104 | 38,990 | 57,738 | 7 |
| profile:3 | ocr#0 | 5,493 | 7,915 | 11,881 | 7, 13 |
| profile:7 | ocr#0 | 14,704 | 22,013 | 35,344 | 7, 13, 14 |

总计 129,545 kernels 全部归属并精确映射，unassigned=0，八请求完整保留。四份 whole-trace parser 均 missing_runtime=0、unresolved=0、cyclic=false；这次单独检查了 whole-trace 结果，未将 request_plan 的局部 unresolved 空列表当成事件完整性的替代。OCR 两请求的 event_record_to_wait 分别为 1,129/3,282，均保留；不存在将多个 stream 串成一条序列的转换。

身份检查同时覆盖 SQLite SHA、start/end、PID、device、context、stream、correlation、READY owner PID、request/task/pool/training role。实际 PID 分别为 text#0 151823、text#1 152489、generalist_2#0 151734、ocr#0 151774；本 capture 的 device=0/context=1 是各自 profiler/process 的局部标号，不能把不同 PID 的 context=1 当成同一 CUDA context。event 依赖沿用精确 `(PID,context,eventId,eventSyncId)`；没有把事件 timestamp=0 当作完成，也没有按跨流 END 排序发明依赖。

附加只读 `static-coverage.json` 使用现有 `AuditedOccupancyResolver` 和 `predictor-occupancy-declaration-light-training-v3-20261028/audit.json`：129,545/129,545 kernel 在精确 launch 源键上有 known allocation，包括全部 OCR。resolver 重新检查 SQLite、hardware source、实现和已编译 occupancy 证据；硬件 UUID 唯一绑定为 `GPU-533a287b-0e15-a7c8-8e58-2b21bdcbb13a`。静态 per-block allocation 已可复用，不需要再通过 GPU 采集补这一项。它不代表 achieved occupancy、同时驻留、SM 吞吐、DRAM 流量、cache hit 或响应系数。

本次实际运行 v9 full-horizon engine，八请求每 owner 按 0/50 ms 的明确诊断到达进入 FIFO，完成 161,984/161,984 DAG nodes、143,835 events；OCR 最大同时活跃 activity nodes=3，其余 owner 为 1。这个指标包含 activity kinds，不能改称同时运行三个 GPU kernels 或已证实同 SM。所有请求完成，第二请求仍等待同 owner 第一请求的结束。v9 沿用 profiled kernel duration、API issue release、handler-end anchor，并明确打开 equal-context time-slicing、independent copy/memset、无隐式 legacy-default-stream 边假设；request start 为 arrival/owner-ready，handoff 为 instant diagnostic。energy/memory 未识别。没有把这组结果当作独立留出、自然无 profiler 时间、物理精度或真实 response fit。

已经消除的局部 failure 是 OCR `NON_SERIAL_STREAM_DAG_REQUIRED` 的拓扑入口问题。仍然开放的资格边界是 default-stream mode、pre-request predecessors 的完成前提（第二请求分别 1/1/1/9 条；其余为 0）、host launch supply、request start/handoff、copy/DRAM 干扰、跨 context 服务、自然请求端点迁移及真实共跑响应拟合。当前记录没有未解析 event/runtime edge；这不证明上述未测机制不存在。已保存 handler-end/release 是 replay 标签，不能进入 prospective host supply 或用未来 END 修正 active set。

最小新接口应放在新模块，不修改 serial v1/v2。现有 `FullHorizonModel`/serial feature adapter 的 active 字典以 owner 为键，每个 owner 只容一个 segment，且把同 owner 排除于 peers。它不能表达 OCR。应复用 v9 的 DAG/FIFO/未来到达与 checkpoint 结构，以 `(owner,node_id)` 为 active 键；v9 的 duration_provider 只注入初始 work，硬编码 equal-context rate 仍须通过一个独立审查的新 engine/scheduler 边界替换。现有 `dag_phase_scheduler_v8` 已有近似所需的 `ActiveGpuNode/SchedulingView/SchedulingDecision`，可复用这一接口思路，而不能声称 v9 已有动态 kernel-rate injection。

最小拟议 contract 如下（仅接口计划，尚未实现或连接）：

```python
NodeBinding = (owner, node_id, request_id, profile_id, kernel_identity,
               sqlite_sha256, hardware_uuid, device, context_id, stream_id,
               execution_context_sha256, profiler_configuration_id)

reference_work_ns(binding, scenario) -> finite_positive_ns
features_for(target_node_key, complete_active_gpu_nodes, immutable_bindings)
decide(SchedulingView(clock_ns, active_gpu_nodes)) -> SchedulingDecision(
    gpu_rates={node_key: finite_nonnegative_reference_work_rate},
    next_transition_ns=None_or_strictly_future_integer)
```

reference work 使用已有 lognormal reference functional：mean=`1e6*exp(mu+sigma²/2)` ns，预声明 quantile=`1e6*exp(mu+sigma*Phi_inverse(q))` ns。每个边界针对所有活动 GPU nodes 重新计算同一静态 `competition_features(target, peers, hardware)`，peer 是其他 node key，**同 owner 的不同活动 kernel 也必须计入竞争**；owner、context 和 stream 分开存储，不能仅按 owner 聚合。response rate 为 `exp(-beta*x)`，对剩余 reference work 积分。active-set、依赖解锁、未来到达和训练过的 supply 产生边界；不读未来真实 kernel END。host/copy/memset/sync 用其显式 law，缺 law 为 UNKNOWN，不自动设为单位速度。

现有 v8 `validate_decision` 限定 rate≤1；负 beta 允许模型产生 rate>1，所以该 contract 也不能直接无改动套用。新独立 engine 应允许有限非负 rate，保留零进度必须有未来 transition 的规则；不能 clip、强行总和归一化或借 occupancy 给速度。checkpoint 需绑定 active node identities、剩余 reference work、完整 source/context/profile 和 provider state。任何旧 serial `service(row, active_by_owner, ...)` 不能直接充当这一 scheduler。

首批后续 GPU kernel training 只形成 CPU campaign 计划。`ordered-kernel-profiles training01/02` 同 raw，不是两次独立 launch。建议初始 pilot 为六个 fresh process/native collector launch；每个均完整提供原八请求（每 owner 两个、含两个 OCR），统一 profiler 配置、execution source/config、hardware UUID 和输入/generation 参数，原始 launch/run id 独立留存。不能将旧 capture 的复制/重导出计作重复。

| fresh launch | 完整八请求的供给安排 | 目的 |
|---|---|---|
| L1 | 四 owner 依次执行，各自两请求 FIFO | 跨 owner solo 基线；OCR 内部多流仍自然保留 |
| L2 | text0+text1 配对，再 generalist2+OCR 配对 | 同 kernel 名称 peer 与异构 peer 对照 |
| L3 | text0+generalist2 配对，再 text1+OCR 配对 | 不同静态 register/shared/warp 特征组合 |
| L4 | text0+OCR 配对，再 text1+generalist2 配对 | 轮换 peer 与完整 owner 覆盖 |
| L5 | 四 owner 同时供给；第二请求保持 FIFO | 多 peer 数、packing 指标和重叠变化 |
| L6 | 预声明错峰的四 owner 供给，完整八请求 | 整个 fresh launch 留出，检验变化 active-set 响应 |

这六次是起始实验，不保证六个斜率可识别。选择固定 kernel baselines 后，必须从**实际** native kernel start/end 全分母构造分段 peer 特征矩阵，再检查 baseline nuisance 加全部六特征的 rank，检查 likelihood root/Hessian/有限方差等数值资格，并按完整 launch 留出。计划中的重叠不算实际活动；同名 peer、不同 register/shared/warp、packing indicator 都必须有实际可区分变化。若 deficient，保留六列和所有 offered rows，在新 CPU 设计中补更有区分的 native peer/错峰配置；不能删 OCR/困难 kernel、删列或虚构系数。若真实 backend launch sequence 改变，需全 cohort 重认证 profile/config mapping，不能仅用名称或 ordinal 强行复用 baseline。重启后的 PID/context/correlation 是新 run 身份，静态 baseline 可迁移必须另外证明同 native signature、输入、配置与执行源；run id 不能混为 profiler configuration id，也不能假定跨重启原 PID 相同。

每个 fresh capture 应由 native intake 认证 READY owner PID、实际 UUID↔profiler device/context、request/pool/profile↔canonical kernel、runtime correlation、stream/event 和完整 offered/accepted denominator；统一 timeline 最简单。若多 profiler session，则必须测量 session/host clock mapping 与误差边界，误差内的活动关系标 UNKNOWN，不能比较裸 timestamps。完整训练 compiler 是新 node-keyed intake；现有 serial capture compiler 明确不支持多流，不能绕过其拒绝后直接宣称已支持 native。

时间线/CUPTI kernels/runtime/copy/memset/event 数据已有可读来源。现存只读 capability 报告 `predictor-gpu-metrics-capability-20261028-readonly01/report.json` 为 `UNAVAILABLE_INSUFFICIENT_PRIVILEGE`（ERR_NVGPUCTRPERM）；本轮未重新运行 profiler 或权限探针。因此可以计划 timeline lifetime response 训练，SM/DRAM/L2 throughput/cache 实证标签仍缺。需要这些标签时另补授权 counters 采集；静态 allocation、NVML 功率/利用率、宏阶段 lifetime 均不能代替。整个 campaign 的目标域先明确为 CUPTI profiled kernel lifetime，再以独立自然请求运行验证端点迁移；它不自动识别硬件 service。

CPU 验证共 18 项通过：新增映射测试 8 项，加 canonical sequence 和 request DAG intake 既有 10 项。覆盖缺失/重复 kernel、完整请求分母、request/owner/profile/role、device/context/stream/runtime 身份、SQLite capture 混用、未解析依赖/环以及同 owner 多流节点保留。没有新 fit。可复现命令为：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/bin/python -m unittest tests.test_kernel_dag_training_mapping_audit_v1 tests.test_request_cuda_dag_v5 tests.test_kernel_sequence_audit_v2

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/bin/python -m tools.predictor_validation_20261027.kernel_dag_training_mapping_audit_v1 --review artifacts/predictor-gpu-20261027/kernel-training-bridge-v3-20261028-prepared00/light-kernel/strict-training-review-v3/report.json --profiles artifacts/predictor-gpu-20261027/request-cuda-dag-v2-20261028-training03/profiles.json --output NEW_UNUSED_ARTIFACT_DIRECTORY --replay-v9
```
