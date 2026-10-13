# Module 9A checkpoint — offline profiling + runtime semantics audit

用户在进行中将范围收敛为：完成全部 task-path offline profiling；新增 Runtime Sharing Semantics Audit；在继续实现或运行任何 occupancy-based pair predictor、新 pair-validation GPU runs 之前停止。

## 已完成

- 冻结质量矩阵重建：**21 admitted paths，3 个 Generalist / 8 个 Specialist models**。
- **21/21 standalone profiles，63 captured inferences，628,174 ordered kernel launches**。Required launch fields 完整；occupancy API 重建的未知 function defaults 另行披露。
- Deterministic sample IDs/order、6 次 warmup 与 capture 分开；profiler timing 未替代冻结 unprofiled s0。
- Prior-art boundary、Roomie formula discrepancy、hardware metric audit、duration diagnostic audit 和 runtime sharing audit 均已保存。
- 原始 Nsight reports、SQLite、path JSON 与 complete kernel catalog 均保留。3 次 adapter 启动失败也保留：2 次模型加载前的 import/name-shadow 问题，1 次 Paddle 环境没有 torch 的 profiling-wrapper 问题；均未产生 target inference measurements。已完成 paths 没有因结果而重测。

## 实际 runtime 结论

Independent processes / independent CUDA contexts / no MPS；GPU compute 在这些 contexts 间 temporal sharing。21 traces 各一个 context，active model kernel stream priority 全为 0；device 支持 compute preemption。当前材料无法识别实际 context-switch quantum、公平性、抢占频率或 FCFS/RR policy。

Roomie profiling、standalone theoretical occupancy 和 task-path signatures 可以保留；跨 context 同时扣除 SM 资源的 adjusted occupancy、由此产生的 occupancy slowdown 和短 kernel 结束后恢复 occupancy 的 two-phase 不能直接作为该 runtime 的 pair execution model。也不能把 whole-kernel alternation 当实测 scheduler。

详细证据、适用机制列表和出处见 [Runtime Sharing Semantics Audit](runtime_sharing_semantics_audit.md) 及 [JSON](runtime_sharing_semantics_audit.json)。

## 明确未执行

- Pair prediction tables / calibration-validation split / new validation selection：未执行。
- 新 pair / multi-model GPU measurements：0。
- Held-out MAE/RMSE/Spearman/Kendall、baseline bootstrap comparison、task ordering accuracy、multi-model analytical accuracy：没有估计值，不能伪造为通过或失败。
- Candidate evaluator overhead：未测；无有效 evaluator。
- Full 9A gate：**NOT_EVALUATED_AT_USER_REQUESTED_CHECKPOINT**。没有根据 sharing mismatch 伪造 held-out accuracy gate。
- Module 9B / 9C：未进入。

用户 steering 前已写入并进行 synthetic CPU checks 的草案原型已隔离为 deferred evidence，未产出实际 pair predictions；hold record 已完整披露。此检查点停止，等待用户决定后续 predictor 的物理建模范围。
