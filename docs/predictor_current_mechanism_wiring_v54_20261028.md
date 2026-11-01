# 当前预测器机制接线状态：v54

**状态快照：2026-10-28 03:30:24 UTC，工作仍在进行。** 当前主候选已经使用全部预声明到达、逐事件 DAG、owner lifecycle 和完整六项评分；尚未通过主预测器的独立服务时间、全过程队列/P95、ΔU 符号及排序门禁。v54 四候选 CPU 预检仍在运行，不能将实现、训练拟合或数值跑通写成物理精度或分流收益。本文只读既有代码和产物，没有启动 GPU、拟合模型或改写历史证据。

这里区分三个层次：“主 v54 调用”表示实际 factory 的调用链；“独立诊断/库分支”表示已有可执行软件但没有注入该主链；“物理证据”只说明相应数据实际观测过什么。较早文档描述的 v11 empirical scheduler 不是当前 v54 lifecycle scheduler 的完整状态。

## 主链实际执行什么

```text
原 v11 plan + 四份固定 routes + 完整公开57请求/到达时间
  → prospective_flow_contract_v11.bind_public
  → projected_handler_forecast_v54.ObservableContextForecast.execute
  → v16不可变compiled DAG / reference work / host release / phase
  → v21 LifecycleDagPhaseModel + v49 LearnedLifecycleScheduler
       ├─ initial prior / handoff
       ├─ v42 projected-handler SAVE hazard与physical next-owner mark
       ├─ v53只替换严格all-idle状态的SAVE rate
       └─ v44 physical-owner → active-reference-service桥（未独立验证）
  → v19完整公开队列sidecar（解释，不改变调度/评分）
  → 原六项Q/T/L/P/M/E、parent receipt端点、startup/priming摊销
```

主入口是 [projected_handler_forecast_v54.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/projected_handler_forecast_v54.py) 的 `ObservableContextForecast.__init__/engine/execute`。`engine` 实际传入 compiled duration/release/phase、lifecycle、power、memory provider，并实例化 v49 scheduler；它没有调用 package 的 batch、roofline、Roomie sequence-alignment 或 Sinan risk provider。

| 机制/论文借鉴 | 主 v54 实际接到哪里 | 已有但不在主链的部分，以及证据边界 |
|---|---|---|
| **InferLine：全过程到达＋事件仿真** | [prospective_flow_contract_v11.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/prospective_flow_contract_v11.py) 的 `bind_public` 核验完整57个 request/input/arrival/route、完整未来 suffix、原30秒窗口＋120秒drain；[availability_forecast_v11.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/availability_forecast_v11.py) 的 `AvailabilityForecast.execute` 经主v54执行全过程。未来请求会在各自到达时进入FIFO；不只依据当前是否积压。 | 这是已声明固定计划的预测，不是知道线上尚未发生的随机实到达。库 [workload.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/workload.py) 的 `FutureWorkload.suffix` 另支持分布情景。主计划见 [plan.json](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/known-input-availability-v11-20261028-prepared01/plan.json)。 |
| **Clockwork：动作、DAG和执行状态分开** | [learned_availability_forecast_v16.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/learned_availability_forecast_v16.py) 的 `LearnedAvailabilityForecast.__init__` 编译完整原图；[owner_lifecycle_engine_v21.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/owner_lifecycle_engine_v21.py) 的 `LifecycleDagPhaseModel.forecast` 推进依赖、host release、执行与完成；[learned_lifecycle_scheduler_v49.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/learned_lifecycle_scheduler_v49.py) 的 `decide` 将已经存在但过期的lifecycle review重新发布到引擎事件表，保留no-rewind与same-holder UNKNOWN。 | 这落实了显式状态/事件的一部分，不是Clockwork运行时复现。主引擎仍没有从真实中途状态恢复完整DAG checkpoint并在线重算的闭环；不能把一个预测内部的逐事件更新称为在线rolling纠偏。 |
| **AlpaServe：rate/CV/scenario** | 当前主 `bind_public` 明确要求 `process='declared_schedule'`，优先使用已知完整公开到达。四候选共享相同到达与输入，只改变原固定route。 | `FutureWorkload.suffix` 已实现 `gamma_renewal`：rate积分跨完整分段窗口、间隔CV、可见last-arrival age条件采样；另有Poisson/periodic。主v54本轮没有执行CV情景集合、随机情景鲁棒排序或在线burst估计；当前数值不是置信界。 |
| **Sinan：短期详细预测与1/5/10秒积压风险分开** | 主v54使用事件模型给出队列与完成时间，**没有把风险分数接成第七项、准入或route控制**。 | [horizon_recovery_risk_v31.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/horizon_recovery_risk_v31.py) 的 `vector/binary_target`、[independent_risk_validation_v39.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/independent_risk_validation_v39.py) 的 `export_features/seal/export_labels/evaluate` 已完成独立host-visible风险数据链。目标是当前waiting>0时“第一次清空晚于H”，不是未来任意新积压、SM进度或完整P95。模型为项目的boosted-stump适配，不是Sinan CNN复现。实际结果见下文v39。 |
| **剩余work，而非只看队列长度** | v54调用 [learned_availability_forecast_v19.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/learned_availability_forecast_v19.py) 的 `QueueDiagnosticEngine` 和 `FullPublicDagQueueDiagnostics`；继承 [dag_queue_diagnostics_v6.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/dag_queue_diagnostics_v6.py) 的 `register/observe/result`，按仿真节点进度扣减gpu/copy/memset reference activity work。 | 这是冻结单路参考活动时长之和，不是request墙钟、真实剩余GPU指令、容量或吞吐。主sidecar未传normalized SM/DRAM/L2 demand provider，因此相应resource-work保持UNKNOWN。它是解释量，不额外改变六项评分或scheduler rate。 |
| **多尺度到达work envelope** | `FullPublicDagQueueDiagnostics.__init__` 先缓存**全部**公开输入的参考成本；`result` 调用 [queue_diagnostics.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/queue_diagnostics.py) 的 `workload_envelopes`，涵盖100ms/1s/10s。 | 未到达请求参与offered envelope，但不register成当前waiting或当前remaining work；未知profile保留UNKNOWN，不当成零或删掉。经验窗口最大值不是确定性burst bound或稳定性证明。 |
| **attained age/条件剩余时间** | [cumulative_lifecycle_hazard_v24.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/cumulative_lifecycle_hazard_v24.py) 的 `CumulativeLifecycleHazard` 从原RESTORE_START累计age与hazard budget；[projected_handler_bridge_v44.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/projected_handler_bridge_v44.py) 的 `feature_projection` 传holder、context age与四owner handler状态；不按每次query重抽budget。 | context age不是当前request已经执行的SM工作。另有 [reader_completion_hazard_v46.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/reader_completion_hazard_v46.py) 的 `ReaderCompletionCandidate`，用reader START以来age、input和peer mask预测reader RESULT条件剩余时间；[v46训练报告](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-reader-completion-hazard-v46-20261028-training01/report.json) 明示training diagnostic未qualify。v54没有导入这个v46完成模型，不能把它称为已替换主服务模型。 |
| **FIFO HOL/convoy** | `DagQueueDiagnostics.observe/result` 输出owner FIFO head blocked时间、逐请求等待及blocking causes；v21引擎保留每owner队列顺序。 | 这是按仿真host/handler/lifecycle状态解释owner队首阻塞，没有实现绕过队首或新调度动作。`global_admission_hol_applicable=False`；主四owner shared路径没有全局整请求cap。v39使用的visible HOL又是reader可见口径，不能混成物理GPU队列。 |
| **backlog recovery** | 同一sidecar记录 `queue_recovery_episodes`、`unrecovered_backlog_since_ns` 及初始backlog left-censoring。 | 它是预测轨迹里的等待队列清空片段；v39则用独立reader事件重放验证固定horizon风险。两者都不自动证明原rho规则已满足，也不把“清空一次”解释成稳定性。 |
| **padding与配批等待** | **主v54没有共同batch执行/预测分支**，仍是原四物理owner、单请求handler。 | [batching.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/runtime/batching.py) 的 `select_batch`、[batch_owner.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/runtime/batch_owner.py) 的 `run_arrivals/run_retained_arrivals` 已记录串行prepare、formation wait、padding、batch dispatch和整批交付；[simulator.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/simulator.py) 的 `FullHorizonModel`＋[batch_profiles.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/batch_profiles.py) 的 `BatchedProfileProvider` 是另一个可配置分支。没有匹配真实batch画像时不能给v54套batch速度折扣。 |
| **rolling feedback** | v54结果明确 `rolling_feedback_correction_applied=False`。主CPU预检从同一原始公开阶段开始，各候选独立运行；同一预测内部重新计算状态不等于接入真实在线反馈。 | 可见事件重放/observer、库的forecast cursor和v46 reader残余模型都不是完整DAG checkpoint。尚未实现并验证“真实进度→恢复全部host/DAG/lifecycle→重跑全过程→纠偏”的主闭环。 |

## v54当前特别补了什么，仍不能解释成什么

v50修复的是**过期review没有重新发布导致预测事件被跳过**，没有修改原SAVE预算或mark目标。其 [四候选CPU报告](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-projected-handler-v50-20261028-preflight01/report.json) 实际为 balanced、skewed、spill_low 三组 `COMPLETE_UNVALIDATED_PROSPECTIVE_FORECAST`，各57 START/57 receipt；spill_high仍为 `UNKNOWN: NO_VISIBLE_ACTIVE_REFERENCE_TARGET`。这修复了数值执行路径的一处漏洞，尚不能判定高分流会更好。

v54的 [配置](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-projected-handler-v54-20261028-configuration01/configuration.json) 在原v42 projected-handler模型外加v53 idle SAVE模型。[idle_physical_save_hazard_v53.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/idle_physical_save_hazard_v53.py) 的 `IdlePhysicalSaveHazard.query` 只在四owner的 `handler_active/issued_outstanding/gpu_ready` 全部严格为False且frontier为空时取新rate，其他状态返回原fallback。目标仍是physical context SAVE_END，不是SM服务；mark仍完全保留原 `ObservableReferenceMark.predict_given_save_now`。没有active reference target时仍允许UNKNOWN，没有强行挑目标或把SAVE跳过去。

[v53训练报告](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-idle-physical-save-v53-20261028-training01/report.json) 的关键数据是91个合格idle风险段、14.697318765秒曝光、**0次SAVE**；OCR没有曝光。正rate来自预先固定Jeffreys prior的Gamma posterior mean，不能写成“实测观察到正的idle切换率”。handler-only训练观察向完整12字段runtime idle域的迁移、median-clock投影、常率和active-reference桥均未独立验证。[v54预检intent](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-projected-handler-v54-20261028-preflight01/intent.json) 只证明四个独立CPU候选已启动，`GPU_started=False, seal_authorized=False`；在本快照时尚无最终四组报告；balanced/skewed各57 START/57 receipt完整，spill_high已返回 `UNSUPPORTED_SAME_CONTEXT_REACQUISITION_PREDICTION`，spill_low仍运行。idle-rate替换没有消除原same-holder支持缺口，也没有伪造RESTORE。

## 已有真实数据分别支持到哪里

| 实际产物 | 已得到的事实 | 不能据此声称 |
|---|---|---|
| [v23八次完整训练采集inventory](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-whole57-training-eight-run-actual-inventory-v23-20261028-cpu03/report.json) | light/native trace与natural各四route，8×57测量请求、8×12 priming，共456＋96；严格冻结和独审通过。原四owner、完整到达/阶段/资源数据为主训练链提供真实来源。 | 训练数据重构好不等于新的独立共驻减速、P95或ΔU排序准确；light插桩与natural端点口径仍分开。 |
| [v39四次fresh独立风险结果](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-independent-risk-validation-v39-20261028-delivery01/report.json) | 5,528个相关prefix，其中4,388个当前positive-backlog、1,140个not-at-risk；25个UNKNOWN（skewed超训练support）。每个1/5/10秒horizon实评分母为4,363/4,388。raw Brier分别0.036949/0.077647/0.081940，同支持训练常数baseline分别0.091614/0.188215/0.248801。 | 这是相同recipe/content的新执行、causal receipt replay。相关prefix不是5,528个独立run；raw horizon单调违例381条，预先固定PAVA诊断也保留，不能看验证标签后挑赢家。未证明概率校准、新输入/计划泛化、在线rolling、GPU服务或主P95/ΔU门禁，更不是分流增益。 |
| [v39来源与数字独立复核](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-independent-risk-validation-v39-delivery-independent-metadata-review-20261028-cpu01/report.json) | 所列receipt字节匹配、route数值与冻结evaluation一致、汇总分母与加权Brier重算一致。 | 该次只解码报告，没有重做raw/label抽取，也没有重新证明causal replay。不能扩大其review范围。 |
| [v50全cohort CPU预检](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-projected-handler-v50-20261028-preflight01/report.json) | 三组完整57请求数值，high一组UNKNOWN，source-before/after保留。 | CPU COMPLETE不是物理预测精度PASS，也不足以封存四候选完整主gate。 |
| [v54预检启动intent](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-projected-handler-v54-20261028-preflight01/intent.json) 及 [high进度报告](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-projected-handler-v54-20261028-preflight01/shared_spill_high-progress-report.json) | 新idle-rate候选仍在运行；balanced/skewed各57＋57完整，high为 `UNSUPPORTED_SAME_CONTEXT_REACQUISITION_PREDICTION`，low尚未交付，最终report未产生。 | 不能把intent、单个progress或factory CPU测试写成四组完成；已有UNKNOWN也不能被省略或算作主gate通过。 |

## USHER / Roomie执行机制的实际边界

这里的“内存型”须分清**显存容量**与**DRAM带宽需求**。大权重/高驻留显存不自动代表带宽瓶颈；低occupancy不自动代表可互补。当前physical context envelope也不是SM正在执行的时间或独占执行许可。

| 用户关注的机制 | 已有明确实现/证据入口 | 当前接线与真实执行边界 |
|---|---|---|
| **计算/访存互补放置** | [roofline.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/roofline.py) 的 `classify_operator_work/complementary_operator_candidates`；[kernel_resources.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/kernel_resources.py) 的 `complementarity_candidate/static_occupancy/check_resident_blocks`。 | 结构代理和有严格counter证据时的候选检查已实现，**未进入v54候选排序或物理调度**。SM/DRAM/L2有效counter与互补收益未验证；只能保留待验证候选，不能剪枝或直接折扣服务时间。原能力审计见 [predictor执行计划中的实测范围](/root/autodl-tmp/generalist-specialist-project/docs/predictor_v2_execution_plan_20261027.md)。 |
| **Roomie有序kernel/阶段对齐** | `kernel_resources.predict_sequence_alignment`、[kernel_provider.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/kernel_provider.py) 的 `KernelResourceProvider`；主v54另有真实画像派生的冻结DAG/phase。 | 库的alignment是显式流体/占用假设，不是Roomie Algorithm1完整复现；主v54实际rate由lifecycle/reference桥给出，**没有注入该occupancy provider**。有序DAG不等于已解释DRAM/cache干扰，也不等于运行时可暂停、重排或抢占kernel。 |
| **同权重共同batch** | [shared_text_batch.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/backends/shared_text_batch.py) 的 `SharedTextBatchExecutor.execute_prepared` 是单owner、一次HF `generate` 的真实执行代码；`batch_owner.run_retained_arrivals`负责到达/准备/配批。 | 已实现且CPU合同验证，尚无本轮合格GPU batch画像；未进入v54。两个text合成一个权重owner后是**shared_text＋G2＋OCR三个物理owner**，不能继承原四owner startup、容量、M/E或干扰模型。 |
| **活跃背景共同batch采集** | [batch_profile_collect_v3.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/batch_profile_collect_v3.py) 的 `prepare/collect/BackgroundDispatcher/supervise_process` 已有真实执行入口，包含主动G2/OCR到达、独立背景时钟、checkpoint和外部进程组守护。 | [交付报告](/root/autodl-tmp/generalist-specialist-project/artifacts/batch-active-background-v3-20261028-delivery01/report.json) 明示 `CPU_CONTRACT_AND_INDEPENDENT_REVIEW_PASS_GPU_UNVERIFIED`、`fit_eligible=False`。较早“仅idle background”的v2限制已由v3软件补齐，但尚不能写成GPU实验完成。 |
| **分batch/microbatch** | `batching.select_batch` 当前支持同owner/兼容key/bucket的whole-request FIFO batch，显式max_wait/padding。 | prefill chunk、continuous decode、KV迁移/游标仍无该主执行接口；超大队首要求 `PREFILL_MICROBATCH_EXECUTOR_REQUIRED`。论文的跨GPU workload division也未在单卡上实现，不能把增加同卡副本当作算力增加。 |
| **最大公共权重子矩阵＋残余GEMM** | [shared_linear.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/backends/shared_linear.py) 的 `find_max_common_contiguous/prepare/SharedLinearPlan.execute` 为严格字节相等、NumPy CPU原型；含合并输入、残余贡献与开销。 | **未接HF量化/LoRA层或主runtime/search**。原型找的是其限定域内的连续矩形，不能称为已找到真实模型的全域最大共享块。[真实模型审计](/root/autodl-tmp/generalist-specialist-project/artifacts/weight-sharing-feasibility-v18-20261028-cpu01/report.json) 尚无认证跨text/G2大线性块；ChatQA INT8/BF16＋adapter与Qwen3系GPTQ4/FP16不构成同基座/同算子合同。 |
| **cache复用** | shared_text代码共享一个模型权重owner；shared_linear原型可以复用一次共同权重读取的计算形态。 | shared_text明确 `prefix_cache_reuse=False, common_submatrix_merge=False`；新批次fresh generate，不跨请求复用KV。权重只存一份、公共乘积、prefix/KV cache、硬件L2命中是不同机制；当前没有L2命中改善或cache净收益实测，不能合成一个折扣。 |
| **降低干扰的可执行准入/资源份额** | [admission.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/runtime/admission.py)＋[known_input_native_flow_v11.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/known_input_native_flow_v11.py) 的 `resource_configuration/execute` 可执行整请求准入；当前主计划四candidate仍是shared静态route。 | 准入与batch是不同执行候选，要各自画像、冻结和公平对照。MPS/green/static SM份额没有已核实生效的本轮执行证据，不能把配置字段当作SM隔离。 |

这些机制尚不能共用一张“已验证收益”标签。实际同text权重共享是明确的软件路径；当前跨text/G2公共子矩阵与cache收益则仍需真实块身份证据、执行层适配及匹配画像。详细量化/LoRA与KV边界见 [实际权重共享核查](/root/autodl-tmp/generalist-specialist-project/docs/actual_weight_sharing_feasibility_v18_20261028.md)。

## 当前门禁与下一步位置

六项效用仍为 `U=-(Q+T+L+P+M+E)/6`，P95取完整offered cohort的parent receipt延迟；原稳定性规则不变。主v54目前只比较四份固定route，没有完成部署、副本、资源份额、batch/cache联合搜索。独立风险诊断没有替代主gate，也没有成为额外效用项。

下一处可审查结果是v54四组完整CPU预检及其首个UNKNOWN/完整metrics。只有四组完整数值、独立重投影和source/START一致性审阅都通过，才可使用新的 [v55 artifact-binding producer](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/projected_handler_campaign_seal_v55.py) 封存主预测，随后由已审阅的 [v2 supervisor](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/main_prediction_gate_supervisor_v2.py) 执行匹配的独立GPU验证。这里是待办门槛，不表示已封存、已运行或已通过。之后才是机制消融和“原/优化执行方式×禁/允分流”四组净收益对照。

论文来源沿用现有核对记录：[InferLine原附件](</root/.codex/attachments/6df614a6-078f-4f3b-9237-f38ad23d9c5f/InferLine ML Prediction Pipeline Provisioning and.pdf>)、[USHER原附件](</root/.codex/attachments/0f495237-c590-4303-b0b7-efc9c15f0925/Holistic Interference Avoidance for Resource Optimized ML Inference.pdf>)、[Roomie v1已保存来源](https://arxiv.org/html/2607.16784v1)、[Sinan既有DOI](https://doi.org/10.1145/3445814.3446693)。Clockwork/AlpaServe的迁移范围沿用 [原机制清单](/root/autodl-tmp/generalist-specialist-project/docs/predictor_v2_execution_plan_20261027.md)；本次没有新增或虚构论文链接，也不把“受启发”写成完整复现。
