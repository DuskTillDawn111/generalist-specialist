# Module14B — Final Heavy Specialist Pool

正式第二regime：**QUALITY_NONDEGRADED_HEAVY_SPECIALIST_REGIME**；gate：**HEAVY_SPECIALIST_POOL_READY**。旧EFFICIENCY_OPTIMAL_SPECIALIST_REGIME不修改；14A历史文件不重写。仅Image采用显式授权的ceiling exception，其余四task严格提高质量并增加s0/safe M。

## 1–3. Image ceiling-aware reclassification 与最终选择

| Candidate | Q | s0 ms | Safe MiB | Latency ratio | Memory ratio | Eligible |
|---|---:|---:|---:|---:|---:|---|
| torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1 | 1.000 | 7.690 | 1501.875 | 1.399 | 2.238 | True |
| torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1 | 1.000 | 36.590 | 1915.112 | 6.655 | 2.854 | True |
| torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1 | 1.000 | 18.844 | 953.113 | 3.428 | 1.421 | True |

选择 **torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1**，Q保持1.000；三模型质量相等，选择最低真实s0，无deployment输入。数据复用14A，没有新Image GPU测量；checkpoint-native transforms和原closed-set10冻结语义不变。

## 4. 最后一轮 OCR 搜索与冻结名单

| Candidate | Family | Published IIIT5K quality | Frozen measurement |
|---|---|---:|---|
| baudm/parseq:parseq-tiny | PARSeq | N/A | False |
| baudm/parseq:parseq-patch16-224 | PARSeq | N/A | True |
| baudm/parseq:parseq | PARSeq | 0.990 | True |
| baudm/parseq:abinet | ABINET | N/A | True |
| baudm/parseq:trba | TRBA | N/A | True |
| baudm/parseq:vitstr | VITSTR | N/A | False |
| baudm/parseq:crnn | CRNN | N/A | False |
| Topdu/OpenOCR:SVTRv2-B-Union14M | SVTRv2 | 0.987 | False |
| byeonghu-na/MATRN:best-train-matrn | MATRN | 0.967 | False |
| wangyuxin87/VisionLAN:LA-final | VisionLAN | 0.959 | False |
| FangShancheng/ABINet:ABINet-LV | ABINet | 0.970 | False |

SVTRv2、MATRN、VisionLAN、原ABINet及STRHub多个release均检索。冻结四路径按公开checkpoint可复现性和native STR架构/输入规模多样性选择：PARSeq128x32、ABINet、PARSeq224x224、TRBA。GoogleDrive folder checkpoint未独立resolve、VisionLAN旧Python2/Torch1栈及MATRN旧fastai栈作为metadata局限如实记录，不宣称这些模型不存在。

四个checkpoint均固定publisher release v1.0.0文件及SHA前缀，implementation commit=1902db043c029a7e03a3818c616c06600af574be。权重完整SHA在GPU前execution lock中固定。官方GitHub浏览下载超时用同一release asset REST ID修复传输；未换权重，早期失败与完整哈希保留。独立新STR environment不修改旧runtime；native decoding defaults，不人为增加refinement/beam/生成长度来放慢模型。

公开quality来自其他IIIT5K3000/source evaluation，不能替代本机冻结200评估；不同checkpoint的家族score不冒充exact release score。原OCR是cropped-word recognition-only；仅保留native recognizer内部TPS等组件，不调用外部detector、裁剪生成或document understanding。

## 5–7. 本机 OCR Q / standalone s0 / safe M

| Model | Local success | Exact match | s0 ms | Safe MiB | Latency ratio | Memory ratio | Strict eligible | Failed criteria |
|---|---|---:|---:|---:|---:|---:|---|---|
| baudm/parseq:parseq | True | 0.965 | 10.436 | 635.408 | 1.720 | 1.007 | True |  |
| baudm/parseq:abinet | True | 0.970 | 12.937 | 693.173 | 2.132 | 1.099 | True |  |
| baudm/parseq:parseq-patch16-224 | True | 0.980 | 11.786 | 635.408 | 1.943 | 1.007 | True |  |
| baudm/parseq:trba | True | 0.965 | 11.880 | 793.150 | 1.958 | 1.257 | True |  |

性能固定3 fresh-process repetitions×30既有样本，原worker完整handler timer和warmup；s0为全部90样本算术均值，不挑最快run或删除outliers。原safe M=1.110853955254881×实测post-warmup idle NVML，peak另报；不使用参数/文件大小/模型卡memory替代。

## 8–9. 最终 OCR 与一次性 HQ SLO

最终 **baudm/parseq:parseq-patch16-224**，exact match=0.980。新OCR SLO：exact_match≥0.97，等于(0.960+真实选中质量)/2。

## 10–11. 最终五task pool / 相对旧模型的变化

| Task | Frozen heavy model | Old Q | New Q | Signed quality delta | s0 ms | Safe MiB | Latency ratio | Memory ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| asr | openai/whisper-large-v3-turbo | 0.037693516 | 0.035001122 | -0.002692394 | 170.522 | 4312.335 | 1.166 | 1.057 |
| lid | openai/whisper-large-v3-turbo | 0.980000000 | 0.995000000 | 0.015000000 | 83.179 | 4312.335 | 3.071 | 2.402 |
| text | philschmid/roberta-large-sst2 | 0.925000000 | 0.970000000 | 0.045000000 | 8.643 | 2052.858 | 2.293 | 2.464 |
| image | torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1 | 1.000000000 | 1.000000000 | 0.000000000 | 7.690 | 1501.875 | 1.399 | 2.238 |
| ocr | baudm/parseq:parseq-patch16-224 | 0.960000000 | 0.980000000 | 0.020000000 | 11.786 | 635.408 | 1.943 | 1.007 |

Quality delta统一new-old；ASR负WER delta代表改善。ASR/LID复用同一Whisper-turbo权重，但task-path Q/s0独立保留，shared residency取已测safe envelope max，不把两个latency混为一个值。

五任务完整=True，missing=。

SLO：ASR≤0.036347318824321295、LID≥0.9875、Text≥0.9475完全保留14A；Image accuracy≥1.000，同时pool固定ConvNeXt-large身份；OCR按本轮一次性中点/null。原efficiency policy未改。

## 12–13. Outcome evidence / gate / 停止

模型与SLO选择是否使用deployment/V3 outcome：**NO**。未运行optimizer/deployment/lowQPS/multi-hotspot/Task-Rich/replica frontier/g(x>1)，Module9不修改。

Gate：**HEAVY_SPECIALIST_POOL_READY**。OCR final search round已用尽，无第三轮模型搜索。

核验：7717个历史受保护文件未变；新GPU运行19次，成功19次。freeze timestamps、source/inputs hashes、样本顺序、资格和SLO核验通过。

机器为RTX4090，实际49140MiB，不宣称标准24GB本机验证。同期OCR baseline control漂移独立记录，历史分母不重定义。这里只确认单replica operating points，不提供deployment性能结论。

Primary sources： [PARSeq/STRHub](https://github.com/baudm/parseq)、[PARSeq paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880177.pdf)、[SVTRv2](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md)、[MATRN](https://github.com/byeonghu-na/MATRN)、[VisionLAN](https://github.com/wangyuxin87/VisionLAN)、[ABINet](https://github.com/FangShancheng/ABINet)。完整metadata、release、implementation源码与raw evidence均保留。完成后停止。

所选OCR的memory改善门槛裕量较小：三轮idle均572MiB，相对历史568MiB只增加4MiB，safe memory增加0.704225%。同期baseline idle560MiB，s0=6.6511946222222225ms（比历史高9.619569%）；控制漂移原样保留，比较分母与门槛均未调整。资格是原冻结规则下的本机点估计判断，quality仍仅是原冻结200样本结果。
