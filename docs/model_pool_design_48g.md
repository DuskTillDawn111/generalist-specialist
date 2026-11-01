# 四任务小显存模型池

当前主实验为上下文问答、图像分类、整页 OCR、ASR；移除 LID 和 VQA。
通用与专用都改用较小驻留显存的配置。候选身份、版本、精度配置、状态和筛选要求
统一存于 `configs/study.json`，不把候选元数据当作已有测量或准入证据。

## GPU 后的显存纠正（2026-10-17）

显存优化已实现并准备独立配置，具体修改、CPU 验证与后续测量顺序见
[显存优化与待测入口](compact_memory_optimizations.md)。尚无优化后 GPU 测量结论。

首批三种通用配置未达到“与专用显存相近”的要求，不能进入系统实验。
v5 冒烟回放中的 12 个专用/通用驻留显存比均不满足原定 [0.5,2]。
旧配置与 GPU 记录保留为失败候选证据，不将其正式加入运行 catalog。

原始记录显示，预热后的存活 PyTorch 张量约为 Qwen 8.77、MiniCPM 8.31、Phi 6.68 GiB，
而整卡驻留分别为 23.88、34.27、14.65 GiB。两者之间包含分配器闲置预留和 CUDA 等占用；
这些字节同样阻碍同卡共驻，不能从验收值扣掉。生成式分类的一万到一万三千 token 完整目录
也产生了较大的瞬时分配，必须检查原生 attention、输出 logits 和缓存复用。

按当前四个专用的预热后驻留反推，通用的共同预算区间约为 3.54–8.46 GiB；
下一轮优先以 **实际驻留 4–8 GiB** 为规划目标。这个区间是预算推导，不是新配置已达标的实测。
专用配置或分配器策略变化后需重新计算全部逐对比例，完整输入峰值与共驻仍须另测。

队列现在会在全部 smoke 后执行 `gsdeploy.profiling.memory_gate.screen_pool_memory`，
任一逐对显存比不符就停止进入独立质量评价。通过这项早期筛查仍不替代完整校准输入压力测试、
正式匹配负载显存/能耗对照和质量准入。缩减不用的语音输出模块、量化和内存优化属于新候选配置，
必须重新冻结和验证；无法达到预算的候选需要更换，不放宽比例、不增加专用的无用占用。

诊断与下一轮 CPU 方案在 `artifacts/compact-gpu-results/memory-review-20261017/`。
当前会话没有 GPU 设备，尚未开展任何修正配置的 GPU 重测。

## 保留的要求

- 每个通用单独覆盖全部四任务并通过最低服务质量。
- 每个专用在对应任务的精度严格高于三个通用，CER/WER 则严格更低。
- 每个专用在匹配输入、请求率、窗口、启动摊销和硬件条件下的整卡能耗更高。
- 专用/通用显存比、平均延时比对每个通用分别在 [0.5,2]。
- T/L/P/M/E 各 0.2，质量仍为硬准入，不给更高精度另加效用分。
- 不要求任何方案必须保留通用；低负载通用、分流、高负载全专用及多副本是实测目标。

## 先按整套部署分配预算

当前 GPU 预算为 49,140 MiB = 47.988 GiB，安全系数 1.110854。
原始驻留量的规划上限约 43.199 GiB，运行峰值需另外验证。

通用优先调查每个约 6–9 GiB 的配置。专用优先调查约 4.5–5 GiB 的配置；
四个专用单副本合计以不超过 20 GiB 为设计目标，可按任务重新分配，
不能为凑单模型下限增加无用占用。真实峰值、质量或能耗不符则候选不能通过。

以 G=8 GiB、各 S=5 GiB 的设计值举例：

| 状态 | 原始驻留 GiB | 安全修正 GiB |
|---|---:|---:|
| 一个通用 | 8 | 8.89 |
| 通用加一个专用 | 13 | 14.44 |
| 通用加两个专用实例 | 18 | 20.00 |
| 四个专用各一个实例 | 20 | 22.22 |
| 四个专用全部双副本 | 40 | 44.43 |

最后一行保留约 3.55 GiB 安全修正后的预算空间；它不是峰值或容量认证。
各任务实际驻留不必完全相等，应共同满足整套预算与逐通用比例要求。

## 现已核实版本的首批候选

| 类型 | 候选 | 紧凑配置方向 | 状态 |
|---|---|---|---|
| 通用 1 | Qwen2.5-Omni-3B | 保留文本/视觉/音频理解，关闭不用的 Talker；必要时验证 INT8 | 元数据已核实，实际占用与质量未知 |
| 通用 2 | MiniCPM-o-2.6 INT4 | 官方 GPTQ W4A16，关闭不用的语音输出 | 元数据已核实，实际占用与质量未知 |
| 通用 3 | Phi-4-multimodal | 量化语言层，保留必要的视觉/音频组件 | 检查点版本已核实，量化兼容性未测 |
| QA 专用 | UnifiedQA-v2 T5 3B | INT8 候选，BF16 参考；沿用真正 QA 训练版本 | 原生适配与量化质量待测 |
| 图像专用 | SigLIP 2 Giant OPT 384 | BF16，正常缓存类别文本特征 | 总参数元数据约 1.87B，实际占用与能耗未知 |
| OCR 专用 | DeepSeek-OCR | 约 3.34B 总参数，INT8 候选、BF16 参考 | 原生实现、量化与评分协议待核验 |
| ASR 专用 | Whisper Large-v3 | FP16 或质量通过的 INT8，复用已有权重 | 旧 BF16 驻留约 7.14 GiB；不冒充紧凑配置新测量 |

“量化方向”不代表对应模型一定支持该加载方式，也不代表其精度已经达标。
各精度配置有独立证据身份，不能继承 BF16 的精度记录。
候选若不能满足预算或高能耗/高精度条件，应更换，不能先把未知指标填入运行 catalog。

当前不再以 UnifiedQA 11B、EVA-CLIP 8B、olmOCR-2 8B、VibeVoice 9B 组成主实验池。
已有文件与原始实验保留作为历史对照，不删除权重或改写记录。

## 四任务入口与数据边界

```bash
python -m gsdeploy.profiling.study --config configs/study.json \
  --source EXISTING_CALIBRATION_DIRECTORY --output NEW_STUDY_DIRECTORY
```

入口只核验并冻结四个任务的输入，ASR 的 speech.jsonl 不再含 LID；
输出 PREPARED_NOT_ADMITTED、native_execution_ready=false 和 U=null。
它不会启动 GPU，也不会生成虚假的可运行 catalog。

当前保留的新任务校准输入为 SQuAD2、ImageNet-A、SROIE 整页、LibriSpeech test-other。
OCR 完整参考、结构文本转换、独立质量集和完整段落 QA 已在 CPU 冻结，见[准备记录](cpu_preparation.md)。
ImageNet-A 仍只有 200 个来源类；没有把它冒充全 1000 类验证集。正式质量仍需 GPU 输出，CPU 准备不能宣称验收通过。
`static_replicas.json` 也已移除两个任务的负载，但其旧 catalog 仅作历史软件回归，
不能把旧 SST-2 或单词 OCR 的参数拿来代替新协议。

## 必须验证的真实收益

精度是硬准入，专用的精度优势本身不直接增加 U。
高能耗也不会自动保证低负载只选通用：低流量时推理能耗差很小，
需要真实的共享显存、启动和常驻成本超过低负载速度收益。
待机功率、整套共驻处理能力和启动成本必须实测，不假设新增模型固定增加多少 W。

同一张卡上复制模型未必增容。因此提前测 G+S、专用双副本与真实分流，
只把有实测容量收益的副本配置登记为相应实验目标的可用实现。
通用也允许使用所有可行的副本和正常优化，不能限制通用制造专用胜出。
全高负载是仍存在可行部署的负载区间，不是无限增加的需求。

先导配对检查只用于排除明显不符候选，不能代替对三个通用的正式比较。
执行次序：四任务协议及适配核验 → 小规模配对测量 → 独立质量/成本准入 →
共驻、副本与固定请求率扫描 → 独立输入/种子的搜索验收。

## 官方元数据来源

- [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)
- [MiniCPM-o-2.6 INT4](https://huggingface.co/openbmb/MiniCPM-o-2_6-int4)
- [Phi-4-multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
- [UnifiedQA-v2 T5 3B](https://huggingface.co/allenai/unifiedqa-v2-t5-3b-1251000)
- [SigLIP 2 Giant OPT 384](https://huggingface.co/google/siglip2-giant-opt-patch16-384)
- [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [Whisper Large-v3](https://huggingface.co/openai/whisper-large-v3)
