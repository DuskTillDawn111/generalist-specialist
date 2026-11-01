# H600 后续执行机制：当前可做范围

2026-10-27，只读代码、历史采集证据及设备信息；未启动 CUDA workload、MPS daemon、驱动配置或新实验，未改当前预测/评分/训练来源与主进度。固定 H600 训练已完成，机制实验仍排在当前冻结预测器的独立验证之后。

当前最可执行的路线是：**先比较 shared 与整请求准入；随后在新后端内验证同一专用模型的分 batch 与权重共享；MPS 先做平台能力验证；跨异构模型最大公共权重子矩阵保持待证。** 现有参数接口不等于机制已经生效。

## 硬件与可观测性

本次只读 `nvidia-smi` 返回 `NVIDIA GeForce RTX 4090`、`49140 MiB`、driver `595.71.05`、compute mode `Default`；UUID 为 `GPU-533a287b-0e15-a7c8-8e58-2b21bdcbb13a`。以实际报告容量为准，不由商品型号推定标准显存。`/usr/bin/nvidia-cuda-mps-control` 存在，进程表未见 MPS daemon/server；本次未启动它。默认 `/usr/local/cuda/include/cuda.h` 的 `CUDA_VERSION=11080`，其中未找到 green-context API；这只说明默认 SDK 缺入口，不代表已检验所有 Python 环境或硬件支持。

历史 [hardware-probe/receipt.json](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/hardware-probe/receipt.json) 中 Nsight Systems CUDA/NVTX 采集成功，而 [tiny-counters.csv](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/hardware-probe/tiny-counters.csv) 明确为 `ERR_NVGPUCTRPERM`。因此可以观察 kernel 时间/重叠、真实 worker 阶段、排队、NVML 板级功率和 owned sampled memory；**目前不能用 SM/DRAM/L2 利用率证明“计算型/显存带宽型互补”或 cache 命中率提高**。经验服务干扰可验证，资源归因仍 UNKNOWN。显存容量占用与 DRAM 带宽瓶颈也须分别判断。

## 机制状态与下一步

| 机制 | 当前真实支持状态 | 可执行的后续路线与验收限制 |
|---|---|---|
| 全请求准入 | 运行时已有 host permit、FIFO/每 alias 排队及事件审计，不需要新 GPU 分区能力。H600 shared 训练不能直接给准入模式合格结论。 | 首先前瞻固定同一新 trace/route，比较 shared、global max1、max2；记录准入等待与真实激活。按原六项效用及稳定性核算，保留完整等待和 drain；新 controller 身份、模式专属资源/容量及 mapper 缺口见下述已有审阅。 |
| MPS active-thread cap | 二进制存在；实际本机 MPS 支持、启动成功、四个真实 worker 加入以及 cap 生效均未验证。`resources.py` 对该模式仍阻止 launch。 | 先另建版本化平台适配器，使用独立 pipe/log 目录并验证 daemon/server、每 worker 的真实 server 归属及上下文可用 SM 属性，再验证不同 cap 的实际执行行为。不能仅设置环境变量、删除 blocker 或据新驱动号宣称支持。若消费级/容器环境拒绝，保留 UNSUPPORTED/UNKNOWN，继续准入与 batch 路线。 |
| 固定 SM 份额/green context | `mps_static_sm`、`green_context` 只有声明与校验，没有创建资源上下文的适配器；默认 CUDA 11.8 headers 也无 green API。未验证本机支持、SM 粒度或实际映射。 | 后续须先核对相应平台 API/SDK/设备条件，再在模型加载前创建正确上下文并证明实际资源归属；普通 PyTorch context 不会因填写 `sm_ids` 自动绑定。当前不能安全当作已可用的实验后端。 |
| 同模型共同 batch / 分 batch | 硬件可执行张量 batch；当前请求执行器按单样本 infer、单 alias FIFO 运行，`sequence_ids` 明确只支持 batch1，模拟器遇 batch>1 直接 UNKNOWN。 | 优先为两路相同 text 专用模型设计一个新 owner/执行后端，按输入长度和输出步数配批、可分 prefill microbatch 与 decode batch；记录配批窗口、逐请求完成、KV 峰、padding/分批成本。先同任务同模型，不能把四个异构 worker 的请求直接拼成一次 generate。必须重新核对 mask/position/EOS/随机性与质量分母，不复用 batch1 服务画像。 |
| 同模型公共权重与缓存复用 | 两专用 alias 使用同一 text worker/spec/checkpoint/adapter，是最强的精确共享候选；当前四独立 PID 分别拥有模型，未实现共享 tensor owner、跨请求 KV/prefix cache 或联合 GEMM。 | 先做 CPU tensor/量化/adapter 身份与映射审计，再由单 owner 多队列或受控共享存储执行两专用请求。区分“只载一次权重省显存”和“共同计算或 cache 命中省时间”；包括 IPC、调度及额外 KV 成本。单 owner 改变原两 replica 执行语义，须新身份与原定义的容量对齐，不能原地更名继承 g(n) 数值。 |
| 异构最大公共权重子矩阵 | text 为 ChatQA/Llama3，G2 为 MiniCPM-o-4.5（观测语言核心 Qwen3），量化分别含 bitsandbytes int8 与 GPTQ int4；OCR 是另一原生后端。没有证据表明这些异构模型存在可直接复用的非平凡相同权重子矩阵。 | 先 CPU 检查层语义、tensor 维度、精确内容/量化元数据与实际可共享块，再决定后端。相同维度、相似数值、同为 Transformer 或零块均不证明可合并算子。没有精确共享块就保留该组合不适用；近似匹配/改权重属于新模型，须另做质量验证。当前不能给预测器填公共权重折扣。 |

MPS 的 active-thread percentage 是**每客户端上限**，不是相加后互斥的 SM 分区；它不保证调度公平、DRAM 带宽份额、L2 隔离或显存上限。即使 MPS 以后可用，也不能把 50% 解释成“独占一半 SM”或把 service 简单乘二。MPS 本身也不会自动共享两个模型进程的权重张量。

模型的常规 KV cache、图像内部 microbatch、ExLlama linear-row microbatch 和 `torch.cuda.empty_cache()` 都已有各自用途，但不等于跨请求共同 batch、跨模型公共权重缓存或共享 L2 命中。L2 命中计数器权限目前缺失；将来即便可测执行时间和板级能耗收益，仍应把机制归因与净收益验证分开。

## 机制实验的最小后续次序

1. 完成当前独立 H600 四路预测验证，保留失败与 UNKNOWN；本文件不授权提前改模型或套用机制折扣。
2. 先做同 route 的 shared/准入机制试点。引用现有 [最小准入方案审阅](/root/autodl-tmp/generalist-specialist-project/docs/h600_admission_minimum_protocol_review_20261027.md)，尤其补 context/controller 身份、准入真实 acquire/release 顺序和模式专属资源证据。
3. CPU 先落实 text×2 精确共享边界和 batch 接口；新版本单独做最小真实正确性/显存预检，然后采训练画像、封预测、独立验证。MPS 平台验证可作为另一条有限试点，未支持即不扩大。
4. 每个生效后端均重建其对应 service、完整队列、启动焦耳、owned peak、功率与容量来源。净效用必须含配批等待、IPC/同步、cache 构建/驱逐和显存成本。达到原门槛后再做优化后端内部禁/允许分流比较，最终相对其最佳禁分流方案评价六项收益。

新增后端不能编辑当前冻结源码、计划或来源收据；使用新的执行身份、训练/独立请求划分和 exclusive artifact 目录。计算/带宽分类缺乏计数器时可保持 UNKNOWN，不能从阶段墙钟或显存占用冒充硬件需求证据。两篇论文的资源互补、分 batch 与复用启发可以进入后续设计，但这些具体能力须由实际后端与实测证据支撑。

## 代码与证据依据

- [resources.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/resources.py)：MPS ceiling 语义、平台 blocker、未实现的 SM/green-context adapter、显式 shared 清除继承的 MPS 设置。
- [simulator.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/simulator.py)：`JOINT_BATCH_OR_MERGED_WEIGHT_EXECUTOR_REQUIRED`；[compact_native.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/backends/compact_native.py) 的 batch1 输出及 generate 路径；[compact_memory.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/backends/compact_memory.py) 的内部 microbatch/cache 回收实现。
- [background-training-v7/context.json](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/background-training-v7/context.json)：当前四驻留模型、worker/checkpoint/量化绑定；当前 H600 沿用该不可变 base context。
- [mechanism-audit-cpu/receipt.json](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/mechanism-audit-cpu/receipt.json)：已有 CPU 审计范围；CPU 测试不提供 MPS/硬件机制生效证据。

另核查了已准备残差工具的 OCR 词表逻辑：它读取 mapper 的 `segments[].phase` 和 provider plan 的 `segments[].id`，没有用 raw 内部 fine events 硬判不匹配。`semantic_profiles.parse_semantic_request` 与 `duration_variant_profiles` 的 OCR handler-tied profile 均使用 `ocr_native`。当前源下应匹配；仅真实不一致才输出 `UNKNOWN_W_VOCABULARY`，本审阅不声称现有 OCR 发生 UNKNOWN，也未执行残差重算。
