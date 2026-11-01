# Kernel 静态资源、序列对齐与服务 adapter v2

日期：2026-10-27。本轮只读既有资料和 SQLite，新增 CPU 实现与测试；没有启动 GPU、采集新模型数据、修改冻结实验或执行资源隔离。已实现静态约束、显式假设下的序列进度，以及可接 `FullHorizonModel` 的 `plan/service` adapter。它们均不提供物理吞吐证据或部署资格。

## 1. 实现与接口

`gsdeploy/forecasting/kernel_resources.py` 提供：

|接口|输入与输出|证据边界|
|---|---|---|
|`static_occupancy(kernel, hardware)`|声明的 block/grid、寄存器、共享内存和硬件上限；输出每 SM block 静态上界及理论 warp 占比|缺几何、设备身份、资源上限或架构取整分配证据时 `UNKNOWN`|
|`check_resident_blocks(kernels, block_counts, hardware)`|检查给定每 SM 整数 block 配置的 block/thread/warp/register/shared 约束|通过只说明声明的必要约束满足，不证明真实 CUDA 共驻留|
|`allocate_block_progress(kernels, hardware, policy=...)`|同一整数分配器生成 block 配额及 `allocated_blocks / solo_blocks_ceiling`|返回 `ANALYTICAL_HYPOTHESIS`；这个比例不是实测吞吐|
|`predict_sequence_alignment(sequences, hardware, ...)`|保留顺序、偏移及固定非 GPU gap，按下一个完成/就绪事件重定价|默认拒绝把 occupancy 转为进度；须显式 opt-in 与并发能力证据|
|`complementarity_candidate(a, b, ...)`|按配置绑定的 compute/DRAM 吞吐指标，1.2 比值区分互补标签|仅候选启发式，不是域剪枝、干扰消除或资源隔离证明|

缺证据返回带原因的 `known=False, status=UNKNOWN`；维度为零、负资源、分配量小于原始需求等畸形输入抛出 `ValueError`。输出均不将 `physically_validated` 或 `deployment_qualified` 设为真。

硬件需要非空 `hardware_id/evidence` 和正整数 `warp_size, sm_count, max_threads_per_block, max_threads_per_sm, max_warps_per_sm, max_blocks_per_sm, registers_per_sm, shared_bytes_per_sm`。kernel 需要相同设备身份、`static_evidence`、三维 `block/grid`、`registers_per_thread, static_shared_bytes, dynamic_shared_bytes`。非零资源还需要 `allocated_registers_per_block, allocated_shared_bytes, allocation_evidence`，来自对应架构的 API/计算器；`registers_per_thread × threads` 只作为未取整算术上界，不能替代架构分配粒度。声明 cluster 产品大于 1 或 cooperative launch 的情况为 `UNKNOWN`。

即使字段齐全，该模型仍只检查显式聚合约束；未建模寄存器分区、架构调度分区、launch bounds、线程维度分别上限、特殊保留量等约束。上界不能被称为实际 achieved occupancy，也不能保证可达到该上界。每 SM 均匀分配假设仅接受 `grid_blocks >= sm_count × solo_blocks_ceiling` 的大 grid；满足这个必要条件也不保证真实均匀调度。

## 2. FullHorizonModel 接入

`gsdeploy/forecasting/kernel_provider.py` 新增：

```python
KernelResourceProvider(
    document, execution_context_sha256,
    allow_cpu_fixture=False,
    allow_analytical_hypothesis=False,
    power_provider=None, memory_provider=None,
)
```

document schema 为 `gsdeploy.kernel_resource_profiles.v2`。顶层字段为 `execution_context_sha256`、`provenance`、`hardware`、`profiles`，并声明 `policy` 和 `concurrency_evidence`。`provenance` 必须有 `kind: measured|cpu_fixture`、`split: training` 及非空 `sources`；CPU fixture 需额外开关。构造器无论原始静态证据是否 measured，始终令 `physical_evidence=False, physically_validated=False, deployment_qualified=False`，因为 occupancy→吞吐关系尚未独立验证。

profile 绑定 `id/family/task/pool_id/execution`，execution 明确包含 `batch_size/backend/weight_reuse`。默认可用执行形态是 batch 1、独立权重；adapter 不实现共同 batch 或权重合并，FullHorizonModel 对未支持形态保留原有拒绝。segment 按顺序声明：

```json
{
  "id": "kernel-0",
  "kind": "gpu",
  "duration_ms": {"mean": 1.0},
  "kernel": {"hardware_id": "device", "static_evidence": "source"},
  "solo_timing_basis": "unprofiled_isolated",
  "timing_evidence": "training-isolated-baseline-source"
}
```

示例中的 kernel 字典仅展示结构，实际缺资源/几何会保持 `UNKNOWN`，不会被补默认值。`duration_ms` 为命名场景下的训练独占基线；不能直接搬用 profiler 中受扰动/共跑的 kernel 时长。`synthetic` timing 只接受已显式打开的 CPU fixture。非 GPU segment 使用 `host/sync/cleanup`，只需训练绑定的正时长场景；它们独立以进度 1 前进，不推断为共享 GPU 需求。无法归因的 NVTX 剩余时间不能直接转成 host segment。

`plan(row, aliases, scenario)` 保持有序 segment，标识与 `ProfileProvider` 一致，GPU 的 `demand.sm/dram/l2` 为 `None`，不伪造吞吐需求。`service(row, active, aliases, scenario)` 在每个 active 边界重新读取当前 GPU segment，复用静态 block 分配器。策略支持 `equal_block_round_robin` 与 `fcfs_blocks`；adapter 明确按 alias 字典序打破同一边界的分配顺序，这不是 CUDA 优先级或公平性证据。

分配到零 block 的目标返回 `known=True, context_supported=True, stalled=True, service_ms=None`，使支持此契约的 FullHorizonModel 保留剩余工作、在其他事件发生后重定价；它不是完成或零时长。正分配返回 `baseline_ms / assumed_relative_progress`，并标记 `status=ANALYTICAL_HYPOTHESIS` 与 `model_assumption`。没有 opt-in、并发能力、静态资源或适用基线时返回 `UNKNOWN`。此处能表达资源冲突导致的等待；它不证明驱动会按该策略共跑或串行。

板级功耗与内存容量分别调用独立 `power_provider.power` / `memory_provider.memory`。未提供时返回 UNKNOWN，绝不从 block 配额、occupancy 或静态张量大小制造 watts/peak-memory 数值。调用方也可向 FullHorizonModel 单独传入这两类 provider；若使用 `ProfileProvider`，其 context 的 profile/segment 标识必须与本 adapter 保持一致。

## 3. 序列进度与 Roomie 原文歧义

来源为本地 `artifacts/roomie-resource-design-20261024/roomie.html` §3.1–3.3；USHER 对照来源为 `docs/usher_methods_and_project_adaptation_20261024.md`。当前实现明确不是完整 Roomie 论文复现。

|原文|本轮处理|
|---|---|
|Eq.1 `o = W_b × b / W`|作为静态理论 occupancy 定义；b 来自显式约束上界，不能当实际吞吐|
|Eq.2 `Σφ > Φ`|原文 φ 的多资源聚合语义不足，当前按 block/thread/warp/register/shared 各维检查，不把不同单位相加|
|Eq.3 `d_tilde^e = d × o / o_tilde`|只在显式假设开关下使用进度比；不是无需验证的硬件规律|
|Eq.4 第二分支打印为 `Δ+(d_tilde−Δ)×(o_tilde/o)`|存在自指；正文描述的是估计剩余时长 `d_tilde^e−Δ`。当前按工作守恒更新 `remaining_solo_work = max(0, d−Δ×o_tilde/o)`，下一边界重新计价，不宣称逐字实现原式|
|Eq.5 `(T_tilde−T)/T` 与 Algorithm 1 第 21 行 `(T_tilde−T)/T_tilde`|同时输出两个不同名称：`relative_increase_over_solo` 与 `fraction_of_colocated_latency`，不混为同一 μ|
|Algorithm 1 使用候选起始对、双方 kernel 索引同步递增、启发式 delay、median、gamma 和 pairwise 累加|当前按最早完成推进异步 kernel 序列，输出 `roomie_algorithm_1_reproduction=False`；不声称 median 是上界或 pairwise 累加是精确 N-way 预测|

序列 helper 的独占工作量不会因另一个 kernel 提前结束而重置。例如同时启动 100/200 单位的两个相同资源 kernel，假设各得到一半进度，分别在 200/300 完成；第二个在前 200 时间已完成 100 单位工作，之后恢复进度 1。固定 host gap 不占 GPU block。事件预算耗尽返回 UNKNOWN，不输出截断成功结果。

## 4. USHER 互补候选的适配边界

USHER 的估计与配置有关，不能把模型名称/参数量当 compute/DRAM 标签。`complementarity_candidate` 要求每个配置有 `config_id=counter_config_id`、同一 `hardware_id`、`counter_evidence`、`counter_metric_signature`、规范到 `[0,1]` 的 `compute_fraction/dram_fraction`，且 basis 明确为 `measured_compute_throughput/measured_dram_throughput`。签名必须绑定相同指标定义、归一化峰值及采集范围；混用 Tensor/FP32 峰值或不同窗口将 UNKNOWN。

1.2 的标签比值与吞吐分数相加仅生成待验证候选。即使同一签名，Nsight SM throughput 可能包含多个管线指标的聚合，归一化分数并不必然线性可加；通过这个筛选不说明干扰小。`resident_bytes` 与 `memory_budget_bytes` 单独做声明容量检查，既不代表 DRAM 带宽，也不证明实测峰值。缺容量证据时 memory_capacity 为 UNKNOWN；候选仍只能标为启发式。`candidate_domain_pruning_allowed=False`，不可据此删去全域搜索候选。这里没有实现 USHER MPS 份额隔离、跨卡 workload division 或 OG-Merger。

## 5. 当前 hardware-probe 证据审计

以下均来自既有 `artifacts/predictor-gpu-20261027/hardware-probe/`，本轮只读：

|证据|已知内容|仍未知|
|---|---|---|
|`receipt.json`|Nsight Systems 返回 0；Nsight Compute 返回 1|成功获得 CUDA 时间线不代表成功获得硬件计数器|
|`tiny-counters.csv`|明确 `ERR_NVGPUCTRPERM`|compute/DRAM/L2 指标不可用，不能产出实证互补标签|
|旧 `parsed-kernels.json`|真实 kernel 时间、stream/device、寄存器/thread、静态/动态 shared 等|旧 JSON 未导出 block/grid，不能直接计算静态 occupancy|
|原始 `tiny-cuda-nvtx.sqlite` CUPTI kernel 表|确有 blockX/Y/Z、gridX/Y/Z、context/green-context、cluster、local-memory；可 CPU 重导出|架构取整后 register/shared 分配证据仍缺失|
|同 SQLite `TARGET_INFO_CUDA_DEVICE`|字段只有 gpuId/cudaId/pid/uuid/numMultiprocessors；本次记录 SM 数 128|缺 warp/block/thread/register/shared 硬件上限，不能靠型号猜值|

导出器增量保留可选 `block/grid/cluster_raw/context_id/green_context_id/local_memory_per_thread_bytes`。原始 cluster `[0,0,0]` 以 `cluster_raw` 保留，并令规范化 `cluster=None`；不把零维度当作合法 cluster 几何。架构取整字段显式为 `None`，旧 SQLite schema 缺字段时仍能解析。没有重写冻结 JSON 或重新采集。

tiny probe 也不是目标模型的训练绑定、未被 profiler 扰动的独占有序 kernel 基线；缺并发能力与真实调度策略证据。因此当前实际模型的占用、共跑吞吐、配置互补候选与部署资格均不能从这些 probe 推导为已知。

### 2026-10-28：UUID与静态上界审计更正

早期审计只查看 `TARGET_INFO_CUDA_DEVICE.uuid`（为NULL），漏看了同一旧tiny
SQLite的 `TARGET_INFO_GPU.uuid`。后者实际为唯一完整
`533a287b-0e15-a7c8-8e58-2b21bdcbb13a`，与只读设备查询相同；旧trace无法绑定
物理GPU的结论已被更正，旧审计产物和trace不改写。新证据为
`artifacts/predictor-occupancy-declaration-old-probe-correction-v2-20261028/audit.json`。

新增独立arch-aware resolver利用真实launch DEFAULT/OPTIN、driver选定共享容量，
以及官方header的不适用分支/成功launch下界等价证明，对旧6个launch和prepared01
smoke的55,891个launch均生成静态上界。真实funcMaxThreads、maxDynamic和请求
carveout仍未测量；等价代表值只适用于确切已完成的观测launch，不能冒充函数属性
或任意未来配置。47个CPU分支敏感性检查及API见
`docs/occupancy_declaration_resolution_v2_20261028.md`，最终smoke新产物为
`artifacts/predictor-occupancy-declaration-audit-v2-20261028-final/audit.json`。
smoke仍不进入baseline训练，实际目标训练行仍为0；这些静态计算不提供achieved
occupancy、counter互补标签、自然时序精度、真实driver调度或部署资格。

## 2026-10-28：声明 profiling 域的额外入口

本次新增 `allow_profiled_baselines=False` 参数，CLI 和实际 gate producer 可通过
`--allow-profiled-baselines` 显式打开。它允许离线真实 `profiled_isolated`
baseline 作为待独立验证的预测假设，而不是要求每个 kernel 都有不存在的“零插桩”
计时。每个相关 GPU segment 必须有 `profiler_id`、`profiler_perturbed: true`、
非空 `perturbation_flags` 及原来的 isolated timing evidence。缺声明或默认关闭
时 UNKNOWN；旧 `profiled_kernel` 不明来源标签仍被拒绝。

service 返回逐 alias `timing_domains`，始终 `natural_timing_validated: false`。
计时域不会被重命名成自然时序，也不会获得 physical_evidence。结构钩子、轻量
内核画像和无 profiler 的请求端点须分别采集；自然时序独立 gate 决定这种假设是否
可用。不能把整请求／阶段的倍率乘给所有 kernel 来制造自然 baseline。

## 6. CPU 验证

运行：

```sh
python -m unittest tests.test_kernel_resources tests.test_kernel_provider tests.test_nsys_trace_export -v
```

38 个测试通过：包含 warp 取整、共享内存冲突、分配粒度缺证据、硬件/config/指标身份错配、不同 kernel 完成边界、offset/gap、输入不变、事件预算完整性、默认 opt-in 禁止、真实 FullHorizonModel 的 200/300 进度守恒及 100/300 零 block 等待恢复、独立功耗/内存委托，以及新旧 SQLite schema 的可选元数据。测试数据均为 synthetic CPU fixture，不是 GPU 吞吐精度验证。
