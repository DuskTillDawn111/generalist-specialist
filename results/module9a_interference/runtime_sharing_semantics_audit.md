# Runtime Sharing Semantics Audit

**结论：当前 runtime 是 independent-process / independent-CUDA-context / no-MPS 的 temporal sharing。Roomie 的跨 context spatial occupancy-based pair dynamics 不能直接用于该 runtime。未选择 pair predictor 或 scheduler policy；按用户指示在本检查点停止。**

## 实测与证据边界

- RTX 4090，driver 595.71.05，实际显存 49,140 MiB，128 SM，compute capability 8.9。现有 toolkit 为 CUDA 11.8；driver 支持的 CUDA API version 是 13.2，不能把它误当所有 worker 使用的 runtime version。未安装或切换任何 stack。
- 从冻结 quality/capability matrix 重建 21 admitted paths：11 条 Generalist paths、10 条 Specialist paths，分属 3G / 8S model identities。每条路径 1 个独立 standalone profiling process，6 次固定 warmup，随后 3 个固定 sample IDs/order 的 inference。共 63 captured inferences、628,174 个 ordered kernel launches；原始 Nsight reports / SQLite 均保留。
- 21/21 trace 只有一个 worker CUDA context。7 个早期 paths 使用 Nsight context metadata；后续 14 个还在 profiling wrapper 中读取了 `cuCtxGetCurrent` / `cuCtxGetId` / default stream priority，原 worker 文件未改动。不同进程/trace 的相同数字 context ID 不代表共享 context；身份必须结合 PID / report。
- 执行模型 kernels 的 stream priorities 全为 0。20 paths 使用 default stream；PaddleOCR 使用 priority 0 的非 default stream。Nsight 创建的其他高 priority stream 没有执行模型 kernels，不能据此推断 worker priority policy。
- 审计时未见 MPS server/control process、标准 MPS socket directories 或 `CUDA_MPS_*` 环境设置；现有 worker 没有显式跨模型共享 context、SM partition 或 stream-priority scheduling。该结论来自当前可见 process/namespace 与 trace evidence，不是改变系统后测得的行为。
- 既有 9A-2 pair records 显示 target/co-worker 是不同 PID，handler wall-clock intervals 高度重叠。**request interval overlap 不等于同时在 SM 上执行 kernels。** 未启动新的 pair / concurrency experiment。现有 profiler traces 均为 standalone 或只覆盖一个进程，没有联合双进程 GPU timeline。

## 能确认的调度语义，及不能确认的细节

CUDA 的公开执行约束：不同 context 的 compute kernels 不并发执行，GPU 可以 time-slice；本机只读查询同时得到 `concurrent_kernels=1` 和 `compute_preemption_supported=1`。前者不绕过 context 边界；后者不测量实际抢占频率。[CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#concurrent-kernel-execution)

MPS 提供跨 process kernel concurrency；本模块没有启用它。[NVIDIA MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)

本审计未识别 context quantum、fairness/time share、FCFS/RR 顺序、实际 preemption frequency 或 switch overhead。不能从 no-MPS 推导“kernel 跑到结束才换 context”，也不能把逐 kernel alternation 当作实测 scheduler。Stream priority 是 stream 属性，不能证明跨 process 的固定优先策略；context 的 CPU wait flags 也不等于 GPU 的 context scheduler。[Context API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__CTX.html), [Stream API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__STREAM.html)

## Roomie 机制适用性

| 机制 | 当前 runtime 下的判断 |
|---|---|
| per-kernel profiling / ordered path signature | 适用：offline trace metadata |
| theoretical occupancy `o=Wb*b/W` | 适用：standalone launch resource capacity；不是 achieved occupancy |
| 扣除 co-kernel 同时占用的 SM registers/shared/warps，求 adjusted occupancy | 不适用跨 context pair：没有该 simultaneous co-residency phase |
| `d_e=d*o/o_tilde` 预测 pair latency | 没有物理依据：context 等待/抢占不是同一 SM 的 co-resident occupancy 减少 |
| 短 kernel 完成后 survivor 恢复 full occupancy 的 two-phase | 不适合作为现有跨 context pair 执行模型 |
| temporal kernel alignment | 可保留为 trace descriptor；不能直接模拟不可用的 spatial overlap |
| pairwise approximation | 可复用的建模原则，须先有适用于 temporal sharing 的 event model；尚未验证 |
| robust median alignment aggregation | 可复用的统计聚合，须先定义物理有效的 scenarios；尚未验证 |
| `gamma=max(q_p/q_q,1)` | Roomie heuristic，不能代替实测 context time share / quantum |
| equal spatial partition | 排除，不符合当前 sharing mode |
| priority-based / FCFS / whole-kernel alternation | 均未得到实际跨 context scheduler 的证据；不选策略 |

Roomie 公式与伪代码差异见 [formula audit](roomie_formula_audit.md)。本表判断是将 Roomie 的机制前提与本项目 runtime evidence 对照后的推断，**不是宣称 Roomie 在所有平台都无效**。

## Offline profile 可用性与缺失信息

21/21 paths 的所需 launch fields 完整：name/order、registers/thread、static/dynamic shared memory、threads/warps per block、grid/block dimensions。Standalone max active blocks/SM 由已安装 NVIDIA occupancy API calculator 重建；theoretical occupancy 按精确 block count 和 `Wb*b/W` 保存。

重建仍假设未知 function max-thread limit 使用 device maximum、barrier count=1、默认 shared-memory carveout/opt-in 设置。这些 function/runtime 参数没有直接测得；因此 **launch metadata 完整不等于 occupancy 的所有隐含属性都直接观测完整**。Achieved occupancy、hardware counters、unprofiled isolated kernel time、CPU/launch-gap 分解也未获得。Counters 不可用并非本检查点失败原因；更关键的是跨 context sharing premise。

Profile durations 仅保留为 trace/relative information。另保存工程诊断 `s0*profile_duration/sum(profile_duration)`：63/63 inference 的 corrected duration sum 精确等于冻结 unprofiled s0；它把 CPU/launch/transfer 时间也按比例分配，**不是 Roomie isolated-kernel timing reconstruction**。没有用新 serving outcomes 调权重，也没有进行 raw/corrected pair-ranking sensitivity，因为用户要求在任何 pair predictor execution 前停止。

## 停止边界

- 新 pair-validation GPU runs：**0**；新 multi-model GPU runs：**0**；pair prediction tables：**未生成**。
- 在用户新指示之前写入且仅做过 synthetic CPU tests 的原型已移入 `module9a_full/deferred_prototype/`，退出有效 `deployment_core/interference/` package；没有把它当作获验证的 9A estimator。原型存在的事实和 hashes 见 [hold record](predictor_hold.json)。其 FCFS/whole-kernel alternation 提议已撤回，不作为当前 scheduler 结论。
- `deployment_core/interference/` 当前仅保留 offline profile schema/duration diagnostic。没有有效 pair predictor / routing evaluator，也未进入 9B。
- Frozen modules、worker/runtime、driver/CUDA、MPS、routing、model pool、workload、memory budgets 均未改动。保护文件 hash 与环境校验结果见 [validation](validation_results.json)。

**检查点状态：OFFLINE_PROFILING_AND_RUNTIME_AUDIT_COMPLETE。Full 9A predictor / validation / gate 尚未执行；下一步取决于用户在本审计之后的决定。**
