# 四驻留模型的 batch、shared owner 与公共权重后端能力审计 v2

2026-10-27。仅 CPU 读取历史探针、真实执行源码、worker spec、checkpoint 元数据与 safetensors JSON header；未加载模型、未启动 CUDA workload、未运行新 GPU 实验、未改驱动或重启。未编辑冻结训练、验证、执行协议及其来源。

**可落地的第一条 batch/公共权重路线是 text×2 使用同一模型 owner，先实现正确的同模型共同 batch，再研究分 batch 和公共权重算子复用。现有四进程后端并未实现这些能力。** G2 和 OCR 应先保留独立 owner；异构模型的精确公共权重子矩阵须先有 CPU 内容与算子语义证据，再决定是否建设合并后端。硬件资源归因和最终效用收益仍须单独验证。

本文件补充 [执行机制审阅](/root/autodl-tmp/generalist-specialist-project/docs/h600_execution_mechanism_readiness_20261027.md)，不把历史画像、CPU 可读性或接口参数升级为新后端已经通过 GPU 验证。

## 实际执行链与四个 owner

[background-training-v7/context.json](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/background-training-v7/context.json) 的驻留为 `text#0,text#1,generalist_2#0,ocr#0`，执行身份为 `batch_size=1,weight_reuse=false,backend=retained-native-observed-v1`。采集器为每个 alias 深拷贝对应 family 的 worker 配置；[transport.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/runtime/transport.py) 对每个 alias 调用一次 `subprocess.Popen`。两个 text alias 复用的是相同配置和磁盘 checkpoint，并非同一 CUDA tensor owner。

历史 raw 的 READY 给出四个不同 PID：text#0=27302、text#1=27337、G2=27272、OCR=27284。raw 状态 completed，errors/telemetry_errors 为空。此处 PID 是历史记录，不能当作当前仍存活进程。

真实模型推理来自 frozen `artifacts/retained-replica-profiles-20261021/source-qa/gsdeploy/backends/compact_native.py`，由 [semantic_worker.py](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/semantic_worker.py) 调用；通常运行入口是 [static_study_worker.py](/root/autodl-tmp/generalist-specialist-project/scripts/static_study_worker.py)。不能只检查当前工作区同名 backend，遗漏 frozen source-root。

| alias / 模型 | 真实执行配置 | 当前 batch / owner 边界 |
|---|---|---|
| text#0、text#1：Llama3-ChatQA-1.5-8B | 相同 worker/spec/checkpoint revision；运行时 bitsandbytes int8，`dtype=bfloat16`，`lm_head` 排除量化，SDPA，同一已训练 QA PEFT adapter | 每进程各加载一份模型和 adapter。每次单样本 `infer`/`generate`，`sequence_ids` 明确拒绝多行 batch。两 alias 是最强的完整公共权重候选。 |
| G2：MiniCPM-o-4_5-GPTQ | GPTQ int4、group_size128、desc_act=false、ExLlama v1、float16、SDPA；支持 text/image/OCR/ASR；内部 vision_microbatch_size2 | 单 owner 内按任务更换 executor.candidate，再串行处理单请求。已启用内部视觉分块不代表跨请求共同 batch。 |
| OCR：PP-OCRv6-medium-det+rec | 独立 torch2.8/cu128 环境；两个 FP32 det/rec 模型；[retained adapter](/root/autodl-tmp/generalist-specialist-project/scripts/retained_ppocrv6_profile_adapter.py) | 每页检测一次，按阅读顺序对裁剪逐个调用 rec，`images=[image]`。另一个 standalone 原始脚本支持 recognition batch1/8，但当前 retained adapter 没有传入或实现该选项。 |

text/G2 历史 READY 均为 torch2.6/cu124；OCR 使用另一环境。把三套后端导入同一个进程不是无条件可行的 owner 合并方式，依赖/量化 kernel/API 兼容性仍须核对。优先只合并两个相同 text owner。

当前 executor 的 `candidate`、生成捕获 hook、临时 logits hook 和部分媒体状态会被请求修改；不能直接用两条 Python 线程并发调用同一个 legacy executor。新 owner 要显式管理请求局部状态与生命周期，或先用受控单调用 batch 接口，避免可重入性问题。

## 本次 CPU 确认的 checkpoint 边界

三个 Python 可执行文件、worker spec 和六个 safetensors 文件当前均存在。读取 header 得到：

| checkpoint | 文件字节数 | header 信息 |
|---|---|---|
| text shard1/2 | 9,976,525,376 / 6,084,042,232 | 209 / 114 tensors，F16/F32；config=LlamaForCausalLM，hidden_size4096。磁盘 FP16 checkpoint 在加载时量化为 int8，不能直接以磁盘字节代替运行时量化存储证明。 |
| G2 shard1/2 | 4,273,353,840 / 4,192,786,348 | 731 / 1,439 tensors，F16/I32；含 GPTQ qweight/qzeros/scales/g_idx；config=MiniCPMO，hidden_size4096。 |
| OCR det / rec | 88,020,412 / 76,741,720 | 350 / 269 tensors，F32；含卷积/归一化等参数。 |

另按 frozen `load_native` 的 CPU stat 判据核对 text 的10个和 G2 的21个 checkpoint 资源文件，inode/device/size/mtime_ns/ctime_ns 均匹配原 spec；OCR adapter 源码 SHA 匹配 spec。本次没有重新完整 hash 所有权重，没有执行 GPU model load，因此只能报告 `CPU_STATIC_BINDINGS_MATCH; GPU_LOAD_NOT_RUN`，不能报告新一轮 GPU_READY。如果后续加载报错，保留原始异常和失败状态，不能切 mock、替换模型或隐式修正 spec。

两个 text alias 本来就绑定同一份输入文件/adapter；单 owner 可以引用这一份已加载权重，无须先在两个不同文件之间寻找相似子矩阵。但共享后的 CUDA 对象、量化状态、PEFT active adapter、精度、tokenizer/生成协议仍须明确绑定并核验。两次独立量化后的存储是否逐字节相同，本次未加载验证。

text 与 G2 虽然 hidden_size 都是4096，但模型语义、层结构、词表、量化格式和 kernel 不同。header 形状相同不证明权重内容相同；小型零值/常量/归一化块相同也不自动构成可盈利的公共 GEMM。未做全权重内容比较，故当前结论为 `NO_EVIDENCE_OF_NONTRIVIAL_EXACT_COMMON_COMPUTE_BLOCK`，不是已经证明不存在任何相同元素。

## 机制可行路径与验收边界

| 机制 | CPU 先实现/核查的最小内容 | 后续必须补的真实证据 |
|---|---|---|
| text 单 owner、双逻辑队列 | 新执行身份；保留逐请求 route、输入、adapter/协议；单次加载一份权重。明确原两 replica 到新执行单元的映射和 backpressure。 | 确认 owned PID 与模型数、参数/量化 storage、启动成本、峰值显存；单 owner 减少驻留内存的真实幅度。它本身不证明吞吐提高。 |
| 同模型共同 batch | 新 `infer_batch`，按长度/生成政策配批；batch mask/position/padding、每行 EOS/停止和输出恢复；不调用仅支持 batch1 的后处理。先 batch2 同 text。 | 同一请求质量、输出/停止语义；逐请求 submit/batch_admit/prefill/decode/finish；配批窗口、padding、KV/临时张量、吞吐和 P95。量化 kernel 接受 batch2 不等于整条 adapter 已正确。 |
| 分 batch / 阶段安排 | 将 prefill microbatch 和 decode batch 显式化；保存每请求 KV/cursor、原输入与全部计算；先定义确定的有限调度规则。 | 真实阶段/kernel 先后及等待；分块开销、额外 launch/sync、KV 峰与竞争。legacy 整请求 `generate` 无法仅靠预测器开关变成可抢占的阶段后端。 |
| 同权重 GEMM/cache-aware reuse | 精确共享权重块 W 的算子映射；相容输入可按数学等价的方式合并为 `[X1;X2]W`，保留两行独立结果；量化 packing、scale/zero/g_idx、bias/adapter 和 kernel 约束全部一致。 | 共同算子/权重加载组织确实生效的 timeline、正确性和成本；节省是否覆盖 gather/padding/切分/同步。只共享存储不等价于共同 GEMM。L2 命中率当前仍无法验证。 |
| 异构最大公共权重子矩阵 | 先 CPU 从 tensor/operator 图筛选候选：层语义、精确值、坐标/转置/布局、量化元数据；再验证切分与重组代数等价。输出可共享块清单和剩余独立部分；不能按形状或相似度给折扣。 | 有非平凡候选才实现版本化联合算子；验证所有残余运算、adapter、精度/质量及额外显存/调度。无候选应明确该组合不适用，不引入近似权重修改。 |
| 跨请求 prefix/KV cache | cache key 至少绑定模型/量化/adapter/tokenizer、精确 token 前缀、mask/position/模板、模态状态；定义容量、驱逐、拷贝和失效。 | 独立输入的真实命中率、prefill 节省、构建/驱逐时间和峰显存。不同输入只可共享其精确相同前缀，不能混用不同模型 KV。此机制与公共权重缓存分开记录。 |
| OCR recognition batch | 新 adapter 将同页按阅读顺序的裁剪按有限 batch 打包，逐块还原；先采用原脚本已有 batch8 思路，协议保持完整检测/识别。 | 输出顺序/CER、padding宽度、检测/裁剪CPU等待、识别吞吐/峰显存。它是OCR内部优化，不自动合并四个模型。 |

单 owner batch 的服务能力不能继承原独立双进程的 `g(n)/n`、共驻模型或容量数字。应按新的执行单元与实际 batch/owner 状态建立新画像及容量映射，并保留原 `g(n)/n` 定义，不能重新命名旧数字来绕过门槛。评分需要纳入配批/准入等待、残余 drain、cache/IPC/同步成本、startup energy、owned sampled memory；Q/T/L/P/M/E 六项与原稳定性规则不变。

## 采集能力与尚缺证据

[hardware-probe/receipt.json](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/hardware-probe/receipt.json) 记载 nsys CUDA/NVTX 成功(return0)，ncu 失败(return1)；[tiny-counters.csv](/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/hardware-probe/tiny-counters.csv) 原因为 `ERR_NVGPUCTRPERM`。当前能够依据已有链路采 worker 阶段、关联 kernel timeline、准入/排队、NVML 板级功率与 owned PID sampled memory；本次未复跑探针。sampling peak 仍可能漏掉两采样点间的短暂峰值，板级功率也不能精确分摊到各模型。

SM throughput、DRAM throughput、L2 hit rate 尚无成功计数器证据。显存容量占用较大不等于带宽受限，墙钟减速也不足以区分 SM/DRAM/L2 冲突。USHER 的计算/访存互补可以作为待验证假设，Roomie 的阶段序列可指导解释实际重叠；在成功资源画像前，分类/归因保持 UNKNOWN，不填入经验性的“互补折扣”。

[resources.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/resources.py) 对 shared/request_admission 有执行路径，对 MPS/SM/green context 保留平台/adapter blocker。二进制存在和环境变量不证明隔离已生效。MPS cap 是每客户端上限，不是互斥 SM 分区，也不共享模型权重。当前不绕过 blocker。

[simulator.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/simulator.py) 发现 batch>1 或 weight_reuse 时明确抛出 `JOINT_BATCH_OR_MERGED_WEIGHT_EXECUTOR_REQUIRED`；这正确反映当前缺口。新 batch 后端须有匹配的 batch 入队、配批完成、逐请求进度/KV/cache 状态及服务模型后才能接入全过程预测，不能直接删除拒绝条件并套 batch1 画像。

## 顺序与交付要求

1. CPU 阶段先落实 text 同 owner/batch2 的版本化接口、精确共享边界、正确性用例和事件字段；本次仅完成能力审计，未更改执行代码。
2. 在既定共驻预测独立验证后，再按授权运行有限正确性/显存预检和相应新训练画像；机制采集要与独立验证输入分开。shared/整请求准入已有路径优先，batch 采用独立身份。
3. 封存新预测后，独立报告共驻减速、全过程队列/各任务 P95、ΔU 正负和候选排序；保留 FAIL/UNKNOWN。
4. 机制生效及排序可靠后，做同到达/输入的四组比较；核心仍是优化执行内部允许分流相对最优禁分流方案的原六项净收益。有限候选中的最优与全部署最优须区分。

## 本次读取的关键来源 SHA256

以下为本次 CPU 读取时快照，不是新 GPU 运行收据：

| 来源 | SHA256 |
|---|---|
| hardware-probe/receipt.json | `2f3b1f7413660c01ca8adb631ae175a1ad91d90a5b870813ec74fe1077c01cee` |
| hardware-probe/tiny-counters.csv | `9c06175fcca212bfb29a8c58bc0b5ce4e3b1eeba3c313f0bbf6779cc57b52ef2` |
| background-training-v7/context.json | `e9004d0f16279fc6e14c7b5daf6e246e23d4b35554bb88c8ec9bc4f9fddc781f` |
| frozen source-qa/gsdeploy/backends/compact_native.py | `8d0641b07d7587fe90b281214d0676c801ceaa2bf53af96d1b335a23caae7126` |
| semantic_worker.py | `ba2a5e4034dff4fcda37cc26b27bb7e9d74df24ea475ff0cd28c999a5dbf6b07` |
| static_study_worker.py | `b8e264dc90a1a43028f7777323418a1668491768bd816b17ed40e07ba6323f73` |
| retained_ppocrv6_profile_adapter.py | `5d638aed7d2beb4859c954bd4740922f562f6d8ce18a09bec7932c0ad7c7a8d3` |
| runtime/transport.py | `92f7fc1264df0becbc19617d9b9159afbc20f59651ad90dca25dc3d277eec506` |
| resources.py | `2098f812d8d61b4ace5753a8d8e5ad41bd037f81998d9b51673cd3cd950aa485` |
| forecasting/simulator.py | `e3bb7aeffdbdcb16b73e9fa746f8172e45ccb948d573597a928611e1cb215a8c` |
