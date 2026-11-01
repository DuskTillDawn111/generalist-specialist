# USHER / Roomie 实际接线审计（CPU 只读，2026-10-28）

当前主路径已有完整到达序列和 DAG 推进，但论文机制没有自动进入该主路径。不能把库里实现的功能、CPU 测试和 GPU 干扰收益等同。以下是本次源码快照；runtime / held-context 并行开发可能继续推进。

| 机制 | 已有可复用入口 | 主预测 / 搜索 / 执行现状 | 最小缺口和证据限制 |
|---|---|---|---|
| 计算/显存互补 | `gsdeploy/forecasting/roofline.py:classify_operator_work, complementary_operator_candidates`；`kernel_resources.py:complementarity_candidate, static_occupancy, check_resident_blocks` | 分类和保留候选的 helper 有 CPU 测试；主 v11 factory、静态 route 搜索、实际 runtime 没有调用 | 可先用于候选排序，不可剪枝或给服务时间打折。算术强度使用声明字节；占用不是吞吐。实际 compute/DRAM counters 未取得，物理互补仍未验证。 |
| Roomie 式时序/占用 | `kernel_resources.py:predict_sequence_alignment`；`kernel_provider.py:KernelResourceProvider` | package simulator 可显式选择静态资源假设；主路径 `availability_forecast_v11.py:AvailabilityForecast.engine` 注入的是经验 `TrainedDagAvailabilityScheduler` | sequence alignment 是显式流体假设，不是 Roomie Algorithm 1 复现，不包含真实 DRAM/cache 干扰；不能把 context envelope 解释为 SM 服务。 |
| 同模型共同 batch | `runtime/batching.py:select_batch`；`backends/shared_text_batch.py:SharedTextBatchExecutor`；`runtime/batch_owner.py:run_retained_arrivals` | 真正共享一个模型并调用一次 generate 的执行代码已写；CPU fake / frozen adapter 合约测试存在。`forecasting/simulator.py:FullHorizonModel` 支持相同 select_batch，`batch_profiles.py:BatchedProfileProvider` 读取精确画像 | 该 package 分支没有注入主 v11 DAG；主候选仍是四条静态 route。需要 batch 配置成为执行候选，且重新绑定物理 owner、startup、容量/显存身份，独立画像后才能比较收益。 |
| 带背景的 batch 采集 | `tools/predictor_validation_20261027/batch_profile_collect_v2.py:prepare, collect` | 可真实保留 G2/OCR 常驻，但只 idle resident；目前只看到 CPU demo / plan，没有 qualified GPU batch capture | 首先补齐 active G2/OCR 按公共 arrival 派发；共享 text 后是三个物理 owner、两个 text 逻辑 alias，不能继承原双专用 g(n)/startup/显存。 |
| 分 batch / prefill 微批 | `runtime/batching.py:select_batch` | 当前是同 owner/key/bucket 的 whole-request FIFO batch，超大队首明确返回 `PREFILL_MICROBATCH_EXECUTOR_REQUIRED` | prefill chunk、连续 decode、KV/游标迁移均需要真实执行器，不能用布尔开关声称完成。 |
| 最大公共权重子矩阵 | `backends/shared_linear.py:find_max_common_contiguous, prepare, SharedLinearPlan.execute` | NumPy float32/64 字节严格相同矩形提取、合并 GEMM、残余计算和开销审计已实现并有 CPU 测试；只被测试导入 | 未接 HF/量化/LoRA 层、模型 runtime 或候选搜索；MAC 总量不减少。先检查真实模型张量和量化算子兼容，再做层适配及独立开销画像。严格字节相等比论文约 1e-7 条件更强。 |
| prefix/cache 复用 | SharedTextBatchExecutor 明确报告 `prefix_cache_reuse=False, common_submatrix_merge=False` | 共享模型权重驻留已经实现；公共子矩阵执行、跨请求 prefix/KV cache 尚未接入 | 权重共享、子矩阵合并和 prefix cache 是三个不同机制，不能混用一个折扣。 |
| 整请求准入 | `gsdeploy/runtime/admission.py`；`resources.py:prepare_registry` | 真正 host 整请求 gate 已有；`known_input_native_flow_v11.py` 可选择 request admission | 这是当前最近的可执行干扰机制对照。shared 与 admission 必须各自有匹配画像/预测/资源身份；MPS/green/static SM 仍未核实硬件控制生效。 |

主预测入口是 `tools/predictor_validation_20261027/availability_forecast_v11.py:AvailabilityForecast.execute`，绑定全部公共请求、startup/generation、原六项评分；`native_queue_phase_busy_v10.py:NativeQueuePhaseFactory` 使用不可变 solo work / host / phase DAG。它们没有 batch/shared-linear/互补 helper 的调用。`trained_dag_availability_v8.py:TrainedDagAvailabilityScheduler` 是 empirical frontier 的支持模型；新 held-context/survival 文件当前主要用于 source replay / 训练诊断，不能因此声称已经接进主 factory。

`tools/resource_control_20261024/forecast_policy.py:FullHorizonPolicy` 已提供包含 batch_configurations 的 package 全过程预测接口和六项评分，是后续可复用接口。`admission_search_check_v1.py:search` 明确只核验有限静态 route 域；deployment、replica、share、batch/cache、adaptive 均不在已搜索域。

建议先完成主路径 held-context / SAVE / next-owner / host 的可训练预测和独立全 cohort 诊断，再做 shared versus 整请求 admission。batch 重用现有执行器、补 active background 和新物理部署画像后，再接 candidate 维度和 DAG。互补 helper 可先排序完整候选域，不能替代物理干扰证据。公共子矩阵先做真实模型/量化后端接线，再测缓存、拼接、配批等待和显存总开销。

本次未重跑已有测试、未启动 GPU、未拟合独立验证数据。源文件收据见 `artifacts/usher-roomie-wiring-audit-20261028-cpu01/report.json`。历史 readiness 文档中的 “NOT_IMPLEMENTED” 只反映当时快照，不能覆盖后来已经实现的 CPU batch / shared-linear 后端。
