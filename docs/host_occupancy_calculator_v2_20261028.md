# 只读设备属性与纯 host occupancy 计算

本轮新增可执行路径，解决“硬件上限和架构取整是否只能等待新 GPU 实验”的问题：设备上限通过只读 Driver API 查询，寄存器/共享内存取整和架构 occupancy 上限通过 CUDA 自带纯 host header 计算。没有创建/保留 CUDA context、分配 GPU 内存、加载模型或发射 kernel。这里得到的是静态计算结果，不是实际 achieved occupancy 或吞吐验证。

## 1. 设备查询

入口 `tools/predictor_validation_20261027/device_attributes_v2.py` 的白名单为 `cuInit, cuDriverGetVersion, cuDeviceGetCount, cuDeviceGet, cuDeviceGetName, cuDeviceGetUuid, cuDeviceGetAttribute`。`cuInit` 的 driver 初始化明确记录为真，不将其伪装成完全未接触 CUDA；context/kernel/分配请求均为0。每次 API 回码、属性 enum、header/实际加载 libcuda/driver proc 版本的来源和 SHA256 均写入结果。

新输出为 `artifacts/predictor-device-attributes-v2-20261027-extended/device-attributes.json`。19次调用全部成功，当前可见设备：

|属性|只读结果|
|---|---:|
|设备|NVIDIA GeForce RTX 4090，CC8.9|
|UUID|GPU-533a287b-0e15-a7c8-8e58-2b21bdcbb13a|
|SM数 / warp线程数|128 / 32|
|最大线程/block / SM|1024 / 1536|
|最大warp/SM|48（1536/32整除）|
|最大block/SM|24|
|寄存器/block / SM|65536 / 65536（32-bit register数）|
|共享内存/block默认 / opt-in|49152 / 101376 bytes|
|共享内存/SM|102400 bytes|
|driver reserved shared/block|1024 bytes|

旧 trace 的 `TARGET_INFO_CUDA_DEVICE.uuid` 为 null；早期查询因此记录了 `UNKNOWN_NO_UNIQUE_EXACT_UUID_MATCH`。**2026-10-28 更正：**旧 tiny probe 的 `TARGET_INFO_GPU.uuid` 实际存在，且与只读设备查询完整唯一 UUID 相同。该早期 identity-only UNKNOWN 结论已由新 `artifacts/predictor-occupancy-declaration-old-probe-correction-v2-20261028/audit.json` 取代；没有以 ordinal、型号或 SM 数猜测映射，旧查询产物和 trace 均未改写。设备上限证据可作为 `static_occupancy` 的 hardware 输入；真实函数最大值仍没有被测量。

## 2. Host header 计算器 API

新增 `tools/predictor_validation_20261027/occupancy_calculator_v2.cpp/.py`：

```python
calculator = build_calculator(new_output_directory)
result = calculate_occupancy(
    kernel, hardware, function_attributes, device_state,
    calculator=calculator,
    assumed_fields=(), assumption_evidence=None, allow_assumptions=False,
)
```

`build_calculator` 用 g++ 编译纯 host C++，只包含 `/usr/local/cuda/include/cuda_occupancy.h`，不链接 CUDA runtime/driver。记录编译命令、回码/stdout/stderr，以及 compiler/header/source/wrapper/binary SHA256。header明确提示不保证新架构 forward compatibility；当前 CUDA11.8 header 对CC8.9有显式minor分支，最大block/SM=24，且与只读设备查询相符。返回header拒绝或header常量与观测设备block上限不符时UNKNOWN。

输入字段：

- `kernel`：相同的 `hardware_id`、`static_evidence`、三维 `block`、`registers_per_thread`、`static_shared_bytes`、`dynamic_shared_bytes`。cluster/cooperative launch不在该wrapper建模范围。
- `hardware`：非空设备身份/证据，`compute_capability_major/minor, max_threads_per_block, max_threads_per_sm, registers_per_block, registers_per_sm, warp_size, shared_bytes_per_block, shared_bytes_per_sm, sm_count, shared_bytes_per_block_optin, reserved_shared_bytes_per_block, max_blocks_per_sm`。extended device query已导出这些字段。
- `function_attributes`：`max_threads_per_block, partitioned_gc_config (OFF|ON|ON_STRICT), shmem_limit_config (DEFAULT|OPTIN), max_dynamic_shared_bytes, num_block_barriers, evidence`。
- `device_state`：`cache_config (NONE|SHARED|L1|EQUAL), carveout_config (-1或0..100), evidence`。

后两组必须有明确来源，不能从C++构造器的默认值推断实际launch设置。旧 CUPTI 的 registers/static/dynamic shared 不能提供函数最大线程数、opt-in上限/模式、partitionGC和carveout。若缺任何字段直接UNKNOWN，不启动host helper；如果只想研究一种假设，必须完整填值、列出具体 `assumed_fields`（如 `device_state.carveout_config`）、给出说明并打开 `allow_assumptions=True`。此时返回 `ASSUMED_STATIC_HYPOTHESIS`，`kernel_patch=None`，不能将假设伪装成正常allocation证据。

无假设且声明完整时返回 `HOST_CALCULATED_STATIC_CEILING`，包含header原始输出：`active_blocks_per_sm, allocated_registers_per_block, allocated_shared_bytes, block_limits, limiting_factors_mask, partitioned_gc_effective`。`kernel_patch`供调用方合入原kernel，包含两项取整分配、allocation evidence digest和 `architecture_blocks_per_sm_ceiling`。

`gsdeploy/forecasting/kernel_resources.py` 已将最后一项纳入静态约束。不能只使用取整后的总寄存器数相除，否则会再次丢失header检查的SM寄存器子分区约束。例如96threads、100regs/thread的CC8.9 fixture，header为5blocks/SM，按总65536/9984取整会误给6。该证据不证明不同kernel混合时的子分区装箱、跨context共驻留或driver调度；原输出的物理验证/部署资格仍为False。

CLI支持 `--request <json> --output <new-directory>`，请求字段为 `kernel/hardware/function_attributes/device_state`，可加显式假设说明。已有目录拒覆盖。

## 3. CPU结果与剩余缺口

`artifacts/predictor-occupancy-host-v2-20261028/validation.json` 保存5种CPU fixture结果和6个旧kernel的UNKNOWN审计；build回执和host binary另存于其build子目录。fixture包含：

- 33threads、33regs/thread、1byte静态共享内存，加driver保留量后header取整到2560 registers/block及1152 shared bytes/block。
- 静态/动态共享内存均0的fixture仍有1024 bytes driver保留量，不能将它静默归零。
- 上述寄存器子分区5vs6反例；默认/opt-in允许的动态共享内存不同；carveout也会改变静态上限。
- 显式假设输出不带可直接合入的正常kernel patch。

上述原始 wrapper 在当时完整声明接口下将6个旧trace kernel保留UNKNOWN，旧产物维持原样。**2026-10-28 更新：**新增独立 `occupancy_declaration_v2.py` resolver，使用真实 `TARGET_INFO_GPU` UUID、每 launch DEFAULT/OPTIN及driver选择共享内存，加官方header分支与成功launch的严格下界等价证明，已对旧6个launch及prepared01 smoke全部55,891个launch计算静态上界；真实函数maxThreads/maxDynamic及请求carveout仍未测得，不能转成任意未来配置的已知属性。完整证明和47个CPU分支敏感性检查见 `docs/occupancy_declaration_resolution_v2_20261028.md`。没有把fixture常量冒充实测设置，baseline训练仍为0；目标独占有序基线、计数器及吞吐假设独立验证不由此产生。

设备查询6项fake-CDLL测试、host计算器8项CPU测试通过。新增resolver的12项CPU测试另见更新文档。测试仅验证输入/计算器集成、架构约束和证据门控，不能称GPU吞吐精度验证。baseline回归器现已另行实现，见 `docs/kernel_baseline_estimator_v2_20261028.md`；当前实际目标模型训练行仍为0。
