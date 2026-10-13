# Module 7A-1 Expanded Candidate Pool Audit

结论：**frozen**；本阶段仅做兼容性、静态 metadata、单模型最小加载预审计。

## 十二项审计结论

1. Verified local Generalist：**3**（Qwen、Phi、MiniCPM）。指定 Qwen/Phi/Gemma 三者中 verified 为 2；Gemma pending。
2. Verified Specialist：**9**。
3. 正式 pool：**12 model types**；共审查 15 条 candidate 记录，pending/rejected 不计入正式 pool。
4. 六个 task 的 capable models 数量见下表；仅计 verified。
5. 仅一个 specialist 的 task：无；零 specialist 的 task：['visual_qa']。visual_qa 目前仅 Qwen capable。
6. SiEBERT 被排除：其 15-dataset 通用 sentiment checkpoint 不满足本阶段严格 SST-2 fine-tuned dataset 条件，尽管 binary label semantics 相同。Gemma 与 GOT 是 pending，不能误写为语义不支持。
7. 每个 model 的实测 idle VRAM 见模型表；未测模型为 null，绝不由文件大小估算。
8. 正式 pool idle VRAM 总量：**63140.0 MiB**。
9. 相对实际容量 49,140 MiB 超出 **14000.0 MiB**；严格大于条件：True。
10. 在审计的进程 idle 资源模型下，full-pool 不可同时驻留：True。这是单模型实测之和的结论，未执行 full-pool 联合加载。
11. 真实资源竞争成立：**True**；没有降低 GPU capacity 或扩大非任务能力。
12. 是否可进入 Module 7B：**True**。此处只交付能力/资源前置条件；后续性能建模与 replica 优化未执行。

## 六任务覆盖

| Task | Capable models | Generalists | Specialists |
|---|---:|---:|---:|
| text | 5 | 3 | 2 |
| image | 5 | 3 | 2 |
| ocr | 5 | 3 | 2 |
| asr | 6 | 3 | 3 |
| visual_qa | 1 | 1 | 0 |
| lid | 3 | 1 | 2 |

## 模型、资源与接口状态

| Model | Role | Verified tasks | Idle VRAM MiB | Primary weights GB | Pool |
|---|---|---|---:|---:|---|
| qwen | generalist | text, image, ocr, asr, visual_qa, lid | 17978.0 | 22.3629 | formal |
| phi4mm | generalist | text, image, ocr, asr | 11772.0 | 12.8102 | formal |
| gemma3n | generalist | — | null | 15.7002 | pending |
| minicpm | generalist | text, image, ocr, asr | 18392.0 | 18.7437 | formal |
| distilbert | specialist | text | 750.0 | 0.2678 | formal |
| resnet50 | specialist | image | 604.0 | 0.1025 | formal |
| convnext_tiny | specialist | image | 610.0 | 0.1144 | formal |
| paddleocr | specialist | ocr | 568.0 | 0.0765 | formal |
| paddleocr_vl | specialist | ocr | 2308.0 | 1.9173 | formal |
| got_ocr | specialist | — | null | 1.4321 | pending |
| whisper-small | specialist | asr, lid | 1616.0 | 0.967 | formal |
| whisper_turbo | specialist | asr, lid | 3882.0 | 1.6178 | formal |
| distil_whisper | specialist | asr | 3672.0 | 1.5129 | formal |
| roberta_sst2 | specialist | text | 988.0 | 0.501 | formal |
| siebert_rejected | specialist | — | null | 1.4216 | unsupported |

## Generalist-only competition

Qwen：17,978 MiB；Phi：11,772 MiB；Gemma：null（HTTP 403，未获本机访问授权）。因此 Qwen/Phi/Gemma 全部共驻留结论为 **unknown**。

所有 verified Generalists（Qwen/Phi/MiniCPM）合计 **48142.0 MiB**，按本次 idle accounting 小于 49,140 MiB；算术上能够容纳，但只剩 998.0 MiB。**未做联合驻留实验**，不声明任意 task inference peak 也能同时容纳。

MiniCPM 是项目现有本地候选，有历史同任务 adapter 实验依据，本阶段新增单模型预审计。纳入它形成有意义的重叠选择；没有把它冒充为 Gemma。

## 兼容性依据与限制

- `verified` 依赖实际当前 task adapter inference；既有模型引用历史正式 summary 与本次 sanity 证据，新模型只在本次实际 inference 通过后置 true。官方 modality 支持只能进入 pending。
- ConvNeXt 使用 torchvision 官方 IMAGENET1K_V1 transforms 与权重；保持相同 RGB 输入、Imagenette 数据集和十类闭集 label space。Transforms 可以按各 pretrained weight 的官方预处理不同。
- RoBERTa 使用 TextAttack 已 fine-tuned 的 SST-2 checkpoint；config 与官方 model zoo 共同确认 task，正负样本验证 ID 映射。实测 runtime classifier 为 124,647,170 参数，相对 DistilBERT 66,955,010 为 1.86 倍；stored checkpoint 的 125,237,762 tensor elements 包含未用于 classifier 的 pooler。FP32 CUDA worker 的 idle footprint 也不同。没有创建、训练或虚构 checkpoint。
- PaddleOCR-VL 只使用元素级 `OCR:` plain-text recognition，沿用 IIIT5K crop 和冻结 normalization。无 document parsing/layout/formula task。
- Whisper Turbo 的 ASR 与 LID 独立检查：LID 调用声学 detect_language，取消 forced English，保存 token map，并验证中文/土耳其语样本；这只是最小接口验证，不是全语言质量 benchmark。Distil Whisper 为 ASR-only，LID=false。
- Phi/MiniCPM 的 visual_qa 与 lid 未经当前 adapter 实验验证，保持 pending；Gemma 的六项 task 均 pending。GOT 为可选 fallback，尚无本地适配实验，OCR pending。
- 新模型的质量没有正式测量；历史 latency 不迁移到新模型，未输出 expanded-pool latency profile。视觉 QA 单模型覆盖仍是当前资源/能力结构的一部分。

## 单模型加载与资源测量

每个实际测试模型只加载一次；一个最小 warmup、三次 sanity inference，随后 graceful BYE/进程退出，三次 GPU release 检查。Generalist 未在三次 sanity 内覆盖的 task 使用已有冻结 formal inference 证据。NVML idle memory 在每个 phase 采样三次，使用最大观测值，包含各进程 CUDA context 与缓存；不记录 inference peak 为 idle。单模型脚本串行，不存在 co-resident profiling。

成功模型数：12；显存释放：True。原始协议中的逐请求时间仅作为 trace 保留，没有用于 latency benchmark、sweep 或搜索。

所有 source_revision、选用权重文件、SHA-256、environment package/version/hash 见 JSON。复用现有兼容环境，没有 pip install、更新 Transformers 或改动冻结算法/worker。下载直连只作用于新的 requests session；未改全局代理。文件大小采用 decimal GB，显存采用 NVML MiB，两者明确区分。

## Freeze gates

| Gate | Pass |
|---|---|
| model_type_count_10_to_15 | True |
| at_least_two_verified_local_generalists | True |
| three_main_tasks_with_two_specialists | True |
| all_admitted_idle_vram_measured | True |
| pool_idle_vram_strictly_exceeds_actual_gpu_capacity | True |
| verified_capabilities_match_matrix | True |
| source_and_environment_integrity | True |

## 官方来源

- [Qwen](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)、[Phi](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)、[Gemma](https://huggingface.co/google/gemma-3n-E4B-it)、[MiniCPM](https://huggingface.co/openbmb/MiniCPM-o-4_5)。
- [Torchvision ConvNeXt Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html)。
- [PaddleOCR-VL 1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)、[GOT plain OCR API](https://huggingface.co/stepfun-ai/GOT-OCR2_0)。
- [Whisper Turbo](https://huggingface.co/openai/whisper-large-v3-turbo)、[Distil Whisper](https://huggingface.co/distil-whisper/distil-large-v3)。
- [TextAttack model zoo](https://github.com/QData/TextAttack/blob/master/textattack/models/README.md)、[SST-2 checkpoint](https://huggingface.co/textattack/roberta-base-SST-2)、[SiEBERT dataset provenance](https://huggingface.co/siebert/sentiment-roberta-large-english)。

本阶段已停止：未运行 MILP/deployment search、full-pool benchmark、multi-replica、integer replica optimization、dynamic adaptation 或 predictor retraining。
