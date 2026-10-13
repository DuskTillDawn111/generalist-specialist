# Module14A — High Quality Heavy Specialist Pool

Gate：**HQ_HEAVY_POOL_PARTIAL**。严格 HIGHER_QUALITY + SLOWER + LARGER，不使用Pareto状态排除，不改变任务、数据、旧阈值或旧pool。此模块只测Q/s0/M，不运行deployment、structural replay、lowQPS、multi-hotspot或g(x>1)。

## 1–2. 固定旧五类 reference 与真实 Q/s0/M

Reference来自原真实baseline：ASR/LID/Text沿用13B固定ratio anchors；Image为原ResNet50；OCR为原PP-OCRv6-medium-rec。13B用于nearest-target规则选出的ASR代表Whisper-small不重定义原Distil-large-v3 reference。未根据deployment选择比较分母。

| Task | Old specialist | Metric | Q | s0 ms | Idle MiB | Safe MiB |
|---|---|---|---:|---:|---:|---:|
| asr | distil-whisper/distil-large-v3 | wer | 0.037693516 | 146.223666 | 3672.0 | 4079.055724 |
| lid | openai/whisper-small | accuracy | 0.980000000 | 27.084135 | 1616.0 | 1795.139992 |
| text | distilbert/distilbert-base-uncased-finetuned-sst-2-english | accuracy | 0.925000000 | 3.769309 | 750.0 | 833.140466 |
| image | torchvision.models.ResNet50_Weights.IMAGENET1K_V2 | accuracy | 1.000000000 | 5.497822 | 604.0 | 670.955789 |
| ocr | PaddlePaddle/PP-OCRv6_medium_rec | exact_match | 0.960000000 | 6.067525 | 568.0 | 630.965047 |

safe M仍为原factor 1.110853955254881 × 真实post-warmup process idle NVML MiB；不是参数、权重大小或模型卡memory。

## 3. 现有 ASR/LID/Text 严格三重审计

| Task | Existing measured candidate | Origin | Status | Failed criteria |
|---|---|---|---|---|
| asr | facebook/wav2vec2-large-960h-lv60-self | NEW_MEASURED_CANDIDATE | FAIL_MULTIPLE | ["LATENCY", "MEMORY"] |
| asr | openai/whisper-large-v3 | NEW_MEASURED_CANDIDATE | FAIL_QUALITY | ["QUALITY"] |
| asr | openai/whisper-medium | NEW_MEASURED_CANDIDATE | FAIL_QUALITY | ["QUALITY"] |
| lid | facebook/mms-lid-256 | NEW_MEASURED_CANDIDATE | FAIL_QUALITY | ["QUALITY"] |
| lid | openai/whisper-medium | NEW_MEASURED_CANDIDATE | HQ_HEAVY_ELIGIBLE | [] |
| lid | speechbrain/lang-id-voxlingua107-ecapa | NEW_MEASURED_CANDIDATE | FAIL_MULTIPLE | ["QUALITY", "LATENCY", "MEMORY"] |
| text | philschmid/roberta-large-sst2 | NEW_MEASURED_CANDIDATE | HQ_HEAVY_ELIGIBLE | [] |
| text | textattack/facebook-bart-large-SST-2 | NEW_MEASURED_CANDIDATE | HQ_HEAVY_ELIGIBLE | [] |
| text | textattack/xlnet-base-cased-SST-2 | NEW_MEASURED_CANDIDATE | HQ_HEAVY_ELIGIBLE | [] |
| asr | distil-whisper/distil-large-v3 | FROZEN_EFFICIENCY_BASELINE | FAIL_MULTIPLE | ["QUALITY", "LATENCY", "MEMORY"] |
| asr | openai/whisper-small | FROZEN_EFFICIENCY_BASELINE | FAIL_MULTIPLE | ["QUALITY", "MEMORY"] |
| asr | openai/whisper-large-v3-turbo | FROZEN_EFFICIENCY_BASELINE | HQ_HEAVY_ELIGIBLE | [] |
| lid | openai/whisper-small | FROZEN_EFFICIENCY_BASELINE | FAIL_MULTIPLE | ["QUALITY", "LATENCY", "MEMORY"] |
| lid | openai/whisper-large-v3-turbo | FROZEN_EFFICIENCY_BASELINE | HQ_HEAVY_ELIGIBLE | [] |
| text | distilbert/distilbert-base-uncased-finetuned-sst-2-english | FROZEN_EFFICIENCY_BASELINE | FAIL_MULTIPLE | ["QUALITY", "LATENCY", "MEMORY"] |
| text | textattack/roberta-base-SST-2 | FROZEN_EFFICIENCY_BASELINE | HQ_HEAVY_ELIGIBLE | [] |

九条新增13B paths中ASR没有eligible；LID Whisper-medium、Text三个模型eligible。完整13B真实operating-point表还包含旧pool替代模型：Whisper-turbo ASR/LID与RoBERTa-base Text均eligible。按GPU前公开声明的默认范围纳入这些真实替代checkpoint，故ASR/LID/Text不需新gap filling。旧pool checkpoint可以在独立HQ regime中复用，原EFFICIENCY内容不删除。参考模型自身的三项相等全部不满足严格门槛。

## 4–5. Image/OCR 公开候选搜索

image: torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1, torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1, torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1, torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1。
冻结测量：torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1 @torchvision0.29.0:convnext_large-ea097f82.pth, torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1 @torchvision0.29.0:vit_l_16_swag-4f3808c9.pth, torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1 @torchvision0.29.0:swin_v2_b-781e5279.pth。

ocr: PaddlePaddle/PP-OCRv4_server_rec_doc, PaddlePaddle/PP-OCRv4_server_rec, PaddlePaddle/PP-OCRv5_server_rec, PaddlePaddle/ch_SVTRv2_rec, microsoft/trocr-large-str, microsoft/trocr-large-printed。
冻结测量：PaddlePaddle/PP-OCRv4_server_rec_doc @c810daf80801417895f51421f16b925c3eb4bef8, PaddlePaddle/PP-OCRv5_server_rec @b26c3587fda8da3c8ec0ce357214b4d661ff1558, microsoft/trocr-large-str @1105e441eaa192e99f3991d2958134348c623f4f。

公开quality仅是搜索证据：Image官方ImageNet1K top1优于ResNet50-v2不等于冻结Imagenette200能提高；OCR跨版本公开分数包含不同评估集，不能直接比本机IIIT5K200。原OCR是单词裁剪图recognition-only，不含detector、orientation/cropping pipeline；TrOCR single-line/word input兼容，并使用已训练STR checkpoint，不做本地finetuning。所有source/revision、metadata与网页依据见sources/、source_audit.md和冻结shortlist。

## 6–7. Image/OCR 真实测量与eligibility

| Task | Candidate | Q | s0 ms | Safe MiB | Latency ratio | Memory ratio | Status |
|---|---|---:|---:|---:|---:|---:|---|
| image | torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1 | 1.000000 | 7.690 | 1501.9 | 1.399 | 2.238 | FAIL_QUALITY |
| image | torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1 | 1.000000 | 36.590 | 1915.1 | 6.655 | 2.854 | FAIL_QUALITY |
| image | torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1 | 1.000000 | 18.844 | 953.1 | 3.428 | 1.421 | FAIL_QUALITY |
| ocr | PaddlePaddle/PP-OCRv4_server_rec_doc | 0.895000 | 7.021 | 679.1 | 1.157 | 1.076 | FAIL_QUALITY |
| ocr | PaddlePaddle/PP-OCRv5_server_rec | 0.915000 | 7.992 | 651.0 | 1.317 | 1.032 | FAIL_QUALITY |
| ocr | microsoft/trocr-large-str | 0.925000 | 41.314 | 2934.9 | 6.809 | 4.651 | FAIL_QUALITY |

原Image accuracy=1.0，任何真实新模型在相同metric上都不能严格更高。即便新模型同为1.0且更慢更大也FAIL_QUALITY，不能更换数据子集、改用ImageNet1000类或放宽为≥。

## 8. Gap filling

最多允许一轮、仅缺失ASR/LID/Text，规则在新GPU前固定。本次完整已有实测candidate universe三task均有eligible，gap filling rounds_used=0；没有反复搜索或Outcome条件。

## 9–10. 最终每task HQ选择、真实提升与资源比

| Task | Selected HQ specialist | Old Q | New Q | Quality gain | Latency × old | Safe M × old |
|---|---|---:|---:|---:|---:|---:|
| text | philschmid/roberta-large-sst2 | 0.925000 | 0.970000 | 0.045000 | 2.293 | 2.464 |
| image | NO_ELIGIBLE_HEAVY_SPECIALIST | 1.000000 | N/A | N/A | N/A | N/A |
| ocr | NO_ELIGIBLE_HEAVY_SPECIALIST | 0.960000 | N/A | N/A | N/A | N/A |
| asr | openai/whisper-large-v3-turbo | 0.037694 | 0.035001 | 0.002692 | 1.166 | 1.057 |
| lid | openai/whisper-large-v3-turbo | 0.980000 | 0.995000 | 0.015000 | 3.071 | 2.402 |

Quality gain以原metric单位表达：accuracy/exact-match为new-old，ASR为oldWER-newWER；乘100为百分点。选择顺序严格为eligible中质量最高→相同质量下latency低→safe M低→full ID@revision。不会为了heavy去选最慢/最大，也不会移除dominated候选以改变研究问题。

## 11. 是否完成五类替换

NO。仅3/5任务有合格HQ Specialist；missing=image, ocr。保留部分HQ pool证据，不用旧lightweight baseline填空，不宣称完整第二regime READY。

## 12. 一次性HQ SLO冻结

构造规则在GPU前固定，实际数值在所有冻结测量完成后一次性写入hq_quality_slo.json：两真实质量值算术中点；higher-is-better用≥，WER用≤；保留原semantic admission。每个有效值比旧SLO更严、排除旧reference、允许所选HQ模型。旧LID仅semantic admission，新HQ task增加独立accuracy SLO，原LID政策未改。Image没有可行严格增强阈值，值为null，不能伪造accuracy>1。

| Task | Old SLO | Old Q | New fixed HQ SLO | HQ Q |
|---|---|---:|---|---:|
| text | >=0.925000000 | 0.925000000 | >=0.947500000 | 0.970000000 |
| image | >=1.000000000 | 1.000000000 | NO_VALID_HQ_SLO | N/A |
| ocr | >=0.960000000 | 0.960000000 | NO_VALID_HQ_SLO | N/A |
| asr | <=0.042181000 | 0.037693516 | <=0.036347319 | 0.035001122 |
| lid | semantic-only | 0.980000000 | >=0.987500000 | 0.995000000 |

同期Image/OCR control独立另报，历史分母不改。image s0 ratio=1.047593, safe M ratio=1.000000; ocr s0 ratio=1.019443, safe M ratio=0.985915。漂移与所有有效run/outliers保留，不重测以挑更有利结果。13B旧controls drift仍见原文件；不同时间实测对比较存在限制，不能将本模块作为最终deployment证据。

## 13. 模型/SLO选择是否使用deployment/V3 outcome

**NO**。比较仅来自真实Q/s0/M固定reference，资格和质量优先规则及SLO公式在新GPU之前冻结。不调用Module9/12 optimizer，不进行新的structural或V3 replay；不引用后续2G/Generalist胜负决定任何模型或阈值。

## 14. Gate

**HQ_HEAVY_POOL_PARTIAL**。不降低三重门槛。

## 15. 若READY下一阶段g(x)

当前不是READY，不启动g(x>1)。部分pool所选checkpoint保持单独证据，未来阶段若授权再判断需补曲线；旧checkpoint已有g曲线不因本模块被重定义。

完整性：7496个既有受保护文件未变；冻结inputs/代码哈希、样本顺序与strict eligibility独立核验通过。新GPU task paths=6，全部quality和3run measurement success=True。实际RTX4090为49140MiB设备，未声称标准24GB本机验证。所有失败兼容修复与raw evidence保留。完成后停止。

汇总兼容修复：首次分析因latency-only control缺少描述性revision字段中止，尚未写入selection或SLO。complete_analysis.py只补入冻结baseline revision，再调用未修改的原analyzer；全部30个raw GPU summaries哈希未变，未重跑测量、改estimator或模型选择规则。失败日志及修复证据见postprocessing_compatibility_amendment.json。
