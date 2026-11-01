# 有序 kernel 到资源竞争特征：实际训练与预测接口

新增 `tools/predictor_validation_20261027/kernel_competition_features_v1.py`，复用现有 `KernelResourceProvider` 的训练域、输入、ordered segment 与 active row 校验。新增接口把有序 kernel 和当前预测活动集合连接到同一个资源竞争特征函数，再把训练时的完整 kernel 路径编译为既有 response likelihood 可直接消费的数组。没有修改冻结的 kernel provider、采集闭包、原有 12 列响应模型或已预声明源，也没有运行 GPU 工具。

```python
provider = KernelCompetitionFeatureProvider(existing_kernel_provider)
covariates = provider.features(target_active_row, all_active_rows, aliases, scenario)
prepared = provider.training_rows(training_captures, allow_cpu_fixture=False)
info, beta, mu, variance = fit_response(prepared)
```

六个固定特征为 peer kernel 数、同名 peer 数、peer 单 block 寄存器/SM 寄存器容量、peer 单 block shared/SM shared 容量、peer 单 block warp/SM warp 容量，以及所有活动 kernel 各取一个 block 时静态必要约束是否冲突。它们表达 launch 资源压力，**不能把模型分成已经测量的计算型/访存型**。`SM_throughput`、`DRAM_bytes`、`L2_hit_rate`、`service_rate_multiplier` 明确为空。

静态属性来自已有训练 catalog；真实接入时，可由 `AuditedOccupancyResolver` 对 source-bound CUPTI launch 提供已审的架构取整 patch。不能从 `registers_per_thread * threads` 猜分配粒度，也不能把 driver shared carveout 构造器默认值当真实配置。`check_resident_blocks` 检查寄存器、shared、warp、threads、block slots 等必要约束：不满足可解释为什么“同一 SM 上各放一个 block”的方案不成立；满足不证明实际跨 context 共驻，也不证明吞吐不会下降。

训练输入为 `gsdeploy.kernel_competition_training_capture.v1`，显式训练角色先于 kernel 正文读取。字段包括 capture/context/profiler、完整 offered request IDs、owner PIDs，以及原 canonical kernel 事件；每条事件显式绑定 profile/segment 和已解析 static kernel。canonical identity 保留 nsys START/END、PID、stream、correlation、device、context。训练适配器只在这个明确训练域使用 kernel END，按所有 overlap 断点生成时间有序特征路径，保留全过程。预测时只读取冻结 catalog 和生成世界的 active kernel 身份，完全不读取目标未来 native END，也不根据目标真实 overlap 填特征。

测试的具体贯通例子：目标 kernel 需要 40,000 bytes shared；另一个 owner 的第一 kernel 也需 40,000 bytes，在 65,536-byte fixture SM 上两者各一 block 的方案冲突。该 owner 顺序进入 1,024-byte shared 的第二 kernel 后，冲突特征从 1 变成 0。真实训练路径断点和预测 active row 切换调用同一个函数，输出完全一致。这里没有乘 occupancy 比值或自动生成减速倍率。

`training_rows` 输出的 durations/matrix/ordered_durations/ordered_matrix 直接匹配已有 `macro_projection_contrast_training_v1.fit` 数学入口；`fit_response` 调用该未修改求解器。这里拟合的是 **profiled kernel lifetime 的统计响应规律**，不冒充自然时长或有效 SM service。先检查每个 baseline group 与全部六列的结构秩；如果某些特征对比可完全被 baseline 吸收，返回 `STRUCTURAL_KERNEL_COVARIATE_RANK_FAILURE`，不删列、不正则化造出唯一系数。通过结构秩还需原求解器的三起点、梯度 root、曲率和 nuisance Jacobian 资格。

本轮 CPU fixture 刻意缺少六个独立资源对比，因此实际 response likelihood 的目标和梯度可消费这些训练数组，但 fit adapter 在结构门禁明确拒绝；未取得任何真实新拟合结果。所有受影响 request/kernel 保留在 offered 分母。未知 static、缺 kernel trace、同 owner 多 stream 或重叠等当前单 active-segment forecaster 不可表达的情况，整 capture 拒绝并逐行记录原因；不能只挑有正例的 kernel 拟合。

这个接口还没有进入冻结 twoFIFO runner；v5_2 当前只有宏响应而没有逐 kernel active 状态。因此不能直接把此六列塞入原宏 rate table。它也不解决原生忙态进度：真实当前 active kernel 集合不可识别时，必须保留 UNKNOWN。训练输入的收据、隔离/共跑条件和 operator→kernel 关联仍需独立 native intake 验证；本模块明确 `native_sources_authenticated=False`，不会因为传入一个 `measured` 字符串而给予物理资格。

## 论文与硬件事实

只读复核了本地 Roomie 原文 `artifacts/roomie-resource-design-20261024/roomie.html` 的 §3.2–3.3，以及 USHER 附件 *Holistic Interference Avoidance for Resource Optimized ML Inference* 的 PDF 8、9、11 页。USHER 将 operator 映射为 kernel 序列，按配置估计资源需求；实现段明确使用 `achieved_occupancy` 和 `dram_utilization`。Roomie 用 kernel sequence alignment 和 adjusted occupancy 估计干扰，执行时间反比于 adjusted occupancy 是其建模假设，不能直接继承为本机吞吐规律。本实现借鉴 kernel 顺序和不同资源约束的对比，尚未宣称复现 Roomie Algorithm 1 或 USHER 资源分配。

现有成功的 nsys/CUPTI artifact 已有 kernel 时间区间、CUDA runtime correlation、NVTX、PID/context/stream、grid/block、register/shared 等 launch 属性。这些 activity 事件在历史环境已经采得，不依赖本项目取得 NCU 性能计数器权限。其时间区间是 kernel lifetime，overlap 不证明两个 kernel 同时在同一个 SM 执行；静态 occupancy ceiling 也不是 achieved occupancy。

本次只读 `/proc` 文件：`RmProfilingAdminOnly=1`，`CapEff=CapBnd=00000000a80425fb`，两者均不含 `CAP_SYS_ADMIN`。历史 NCU 收据是 `ERR_NVGPUCTRPERM`；没有重跑 NCU、nsys、CUDA API 或权限探测。当前权限配置仍限制管理员性能计数器接口，但本次不声称刚执行过一次新计数器失败。SM throughput/achieved occupancy、DRAM throughput/transactions、L2 hit/miss/bytes 没有成功测量证据；nsys metrics 的 CLI 选项存在不能绕过该权限。NVML power/显存/粗利用率不能补成这些指标。

## 最小下一轮采集协议

先等待当前自然训练采集结束，再用独立 profiler campaign 采集。保持四物理 owner 与执行/量化/输入/硬件身份明确绑定；训练和 validation 角色隔离。本轮没有启动该采集。

1. **关联与顺序。** 预先绑定 request、输入配置、operator/forward、kernel launch identity、PID/context/stream，保留所有 canonical launch 和未归因数量；明确 stream 依赖。能表达单串行 stream 的域先进入本接口，多 stream 需要完整事件依赖后另扩 DAG。
2. **独占与配对。** 同配置测单路、两路、三路、带背景；覆盖两种顺序与启动差。不能只保留实际达成重叠的成功请求。kernel launch 元数据及架构 static patch、profiler 身份和时间域同封存。与独立 natural 请求/阶段计时配对估计插桩扰动，不能把整请求校正比直接施加到每个 kernel。
3. **识别资源对比。** 对固定目标 kernel 配置，让 peer 数、同名性、register/shared/warp 单 block 需求和 mixed-block 冲突有独立变化；先检查全部预声明六列加 baseline 的结构秩，再考虑拟合。当前不完整 fixture 的 rank failure 提供可复现的缺证据输入/输出。
4. **物理分类分支。** 若要叫“已测量的计算型/访存型”，必须另有经核实可用、同硬件/精度/归一化 scope 的 SM/DRAM counters，或经独立校准的 resource-service-demand 识别实验。当前 static pressure 只能作为 covariates。若计数器不可用，可在清楚界定的负载对比上验证响应预测，仍不能宣称 DRAM/L2 归因或 cache 复用收益。
5. **预测链。** 冻结训练拟合及支持域后，由逐 kernel 事件引擎生成 active set，用同一映射计算特征并更新统计响应模型；此前不能用目标原生未来 END。独立检验 kernel/phase 减速、全到达队列与 P95、ΔU 正负与候选排序，最后才对照分流收益。

CPU 命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest tests.test_kernel_competition_features_v1 tests.test_kernel_provider tests.test_kernel_sequence_audit_v2 -v
```

26 项通过，其中新增 9 项；不构成 GPU 预测准确度或最终效用验证。
