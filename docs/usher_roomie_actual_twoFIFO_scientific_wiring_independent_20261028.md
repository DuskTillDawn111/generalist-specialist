# USHER / Roomie：新两级 FIFO 预测的科学接线独审

**下一步应补可识别的自然画像与严格输入接线。当前没有已验证的物理资源模型可以直接替换 v4 的响应率；继续增加自由斜率不能补齐它。** 本次只读作者之外的机制源码和既有训练产物；不改冻结源、不重跑测试、不启动 GPU，也未使用独立验证标签。

## 真入口与可用输入

新原策略执行核心为 `actual_worker_twoFIFO_reference_v4.simulate`。其真实输入是完整公开请求/到达、按公开键与到达 ordinal 选择的 `RequestPlan`、`rates[(phase, visible_count, same_phase_count)]`、协议 slacks、`TransportDraw` 和初始状态。`rate_for` 读取的是父进程已处理的可见宏证据，**没有调用 roofline、occupancy、batch 或 cache provider**。宏参考坐标也不是 FLOP、SM 服务或独占墙钟。

最新训练入口 `twoFIFO_composable_macros_v2.extract(training_review)` 接受 `gsdeploy.original_twoFIFO_owned_training_review.v2`，输出 `gsdeploy.twoFIFO_composable_response_macro_training.v2`：整个 attempt（含 priming 和重复 owner）参与可见历史，priming 不作训练标签，leaveout 单位为整个 capture block 的全部 shard。它修复历史边界，但没有增加资源物理模型。旧 `macro_joint_response_fit_v1.verified_document` 严格接受 v1 schema，不能给 v2 文档改个 schema 字符串就继承旧资格；应新增完整 attempt 的训练适配及来源闭包。

| 用户要求的机制 | 已有真实软件/数据 | 能否直接进入新 v4，以及具体边界 |
|---|---|---|
| Roomie 的 phase、顺序和启动差 | 宏、真实 BEGIN、通知处理时钟；公开到达/offset/order | phase/通知机制已进入 v4。未来实际 shape、query outcome、完成时刻必须由合格计划生成，不能从目标执行回放。看到宏不等于看到 SM 工作。 |
| USHER 的算子需求与计算／访存互补 | `static_language_work_v1.build_binding/shape_work/features_from_visible_begin`；源绑定七类 decoder projection。真实 affine observer/CUPTI 匹配也已有结构 smoke | 目前可作结构诊断和完整候选域的优先级；未进入 v4 rate。`gsdeploy.static_language_projection_work.v1` 是名义 FLOP／逻辑字节，attention/KV、实际量化／融合代价和 DRAM/cache 仍未知。BEGIN shape 只在父进程处理后可见；更早路由须有合格未来 shape 生成契约。 |
| Roomie occupancy／kernel alignment | `kernel_resources.static_occupancy/predict_sequence_alignment`、`kernel_provider.KernelResourceProvider` | 是另一个 package 分支的 opt-in 静态流体假设，不是已验证吞吐规律，也非 Algorithm 1 完整复现。缺实际分配粒度、隔离基线或并发证据时 UNKNOWN；即使静态 ceiling 已知也不能将其倒数塞入 v4 rate。 |
| 同模型共同 batch | `SharedTextBatchExecutor.execute_prepared` 真正一次 HF generate；`select_batch`、`batch_owner.run_retained_arrivals` 和 active-background collector v3 已有执行代码／CPU 合约 | 尚无合格 GPU 画像，未接 v4。将双 text 物理 owner 改成一个会使四 owner 变三 owner；现 v4 一个请求一个 handler／RESULT 无法表示整批交付。应新建 batch-aware 合同，复用同一配批规则；不能复制 batch1 计划冒充 batch2。 |
| 分 batch／阶段调度 | whole-request FIFO 配批已实现 | prefill chunk、连续 decode、每请求 KV/cursor 尚缺执行器。预测器开关不能改变完整 generate 的运行语义。 |
| 最大公共权重子矩阵 | `shared_linear.find_max_common_contiguous/prepare/SharedLinearPlan.execute`：NumPy float32/64、严格字节相等的最大连续矩形及残余计算 | 是 CPU 原型，没有 HF／INT8／GPTQ 层后端，未接 v4。不是任意行列子矩阵搜索，不减少总 MAC；拼接和残余成本不能忽略。两个 text 副本同基座同 adapter，可研究同权重共享；ChatQA 与 G2 未认证任何可用公共 dense 块，且有限检查不能证明处处不存在。 |
| cache 复用 | shared-text 已共享一份权重驻留 | 后端显式 `common_submatrix_merge=False, prefix_cache_reuse=False`。权重驻留共享、共同 GEMM、prefix/KV cache 和 L2 命中是不同事实；后三者没有合格收益输入。 |

## 计数器权限阻断什么

既有 NCU 收据明确为 `ERR_NVGPUCTRPERM`；本次未重新探测，故**当前权限为未知**，不能宣称仍被拒或已有权限。SM／DRAM throughput、L2 hit rate 没有成功来源；NVML 显存容量、功率、粗利用率，以及 kernel lifetime overlap 均不能替代这些量。静态精度峰值也不能替代实际量化 kernel 的资源需求。

这阻断直接的计算／DRAM 分类和 cache 归因，**不阻断有控制的自然共跑响应与净收益验证**。即使以后获得计数器，仍需同输入的单路／配对、时序和执行身份对照，不能把计数器相关性自动当作因果干扰模型。原算子结构 smoke 的自然／插桩 handler 比值随 owner 约为 1.09～2.13，只有每 owner 一个短请求；不能拿统一插桩倍率补自然时间。

本次独立逐片读取原三 capture、150 请求、1002 宏，确认：留出 `96f5…` 后，剩余两 capture 的 8 条 OCR entry 路径 peer count 严格全零；四类 same-phase 列严格全零，OCR entry 的 count 与 same 严格相同。12／22 维候选的失败有真实 support 原因。加优化重启或更多斜率不能识别这些列；结构不可能状态也不应当作“缺样本”盲采。

## 最小可执行后续动作与门槛

1. **现在做 CPU 接线。** 以新 v2 完整 attempt schema 新增适配，明确 source、capture/attempt、真实 shape 的可见时刻和 `UNKNOWN`。七投影特征先挂训练诊断 sidecar／候选排序，不新增服务倍率。保留所有 entry、OCR、尾段；已知／未知 peer 分母完整。另把 busy sender／stdin 的观测缺口写成字段契约，不能把未测的分解项置零。
2. **随后补最小 natural 训练支持。** 每个独立 capture 内，让同一 OCR 输入的 entry 重复出现在可见 peer count 0／1／2；交叉 own phase、peer phase、count 与启动顺序，保留未达到目标重叠的全部请求。加入同 owner 连续突发，记录 submit／write／START、实际 READY、RESULT occurrence／reader／processing，以及可实施的 worker 读入／续接观测。至少三个独立重启 capture 才能在 leaveout 中保留支持；先检查完整 attempt 的秩与可达性，再拟合固定的少量校正。已有 `phase_response_identifiability_audit_v1.targeted_training_plan` 可作清单，不能自动视为采集执行计划已获资格。
3. **资源归因是硬件分支。** 若必须产出测量的 compute／DRAM／L2 标签，先取得并验证计数器权限和相同指标／精度／归一化范围，再做有限配置画像。schema 可复用 `gsdeploy.kernel_baseline_observations.v2` 与 `gsdeploy.kernel_resource_profiles.v2`，但必须补 isolation、operator→kernel 关联、配置/硬件身份和对应时间域；旧无绑定 trace 审计的 6 行均 UNKNOWN，不是现成训练集。没有权限则保留 structural proxy，不给“互补奖励”。
4. **batch／公共权重执行单列。** batch 先以新三 owner 的 batch1 作为控制，再测 batch2 的配批等待、padding、shared tail、质量、startup、容量、能耗和显存；不能继承四 owner 的画像。公共子矩阵先有真实张量／量化语义一致的正候选，再实现新层后端并量净开销；没有候选就保留不适用。prefix cache 则另需 key、容量／驱逐、构建和 KV 状态合同。

新资料封存后仍须独立检验全过程队列、每任务 P95、ΔU 正负及候选排序，才进入同到达／输入的四组分流收益对照。此审查没有给予 native/provider/physical/risk/stability 或最终收益资格。

证据包：`artifacts/usher-roomie-actual-twoFIFO-scientific-wiring-independent-20261028-review01/`。
主报告 SHA256：`d652aaf1f72cc9f222ff5f36339af48c6e565e72c4cd6cca1b16ccdd934be909`。
最新 v2 入口补充收据见 `latest-entrypoint-addendum.json`；21 份机制源码快照及 14 份证据收据均在包中。未重跑已有测试。
