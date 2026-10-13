# Module 13A — Real Specialist Candidate Discovery

Gate: **REAL_SPECIALIST_SHORTLIST_READY**. ASR/LID/Text 各冻结3个 task-model paths，共9条、8个唯一 checkpoint。所有新候选未本机实测；BALANCED_SPECIALIST_REGIME 目前是待13B验证的 proposal，不能作为正式实验模型。

本轮CPU/WEB/METADATA only：GPU runs=0，权重下载=0，模型加载=0。Module12是COUNTERFACTUAL SEARCH TARGET；13A是真实公开模型发现；13B才是真实本地质量/延迟/内存测量。EFFICIENCY_OPTIMAL_SPECIALIST_REGIME、Module9及全部历史结果保留。

## 1–4. 候选和当前 references

Text语义：SST-2英文句子negative/positive情感二分类；200条validation冻结manifest。LID语义：VoxLingua107 dev音频识别语言，十种reference labels；模型在自身完整language class space上预测。ASR：LibriSpeech test-clean英文转录、corpus WER。

| Task | Module12比例reference | 已测service ms / idle MiB | 原quality admission |
|---|---|---|---|
| ASR | distil-whisper/distil-large-v3 | 146.223666144 / 3672.0 | WER ≤ 0.042181 |
| LID | openai/whisper-small | 27.084134878 / 1616.0 | semantic validity only；无数值accuracy门槛 |
| TEXT | distilbert/distilbert-base-uncased-finetuned-sst-2-english | 3.769309222 / 750.0 | accuracy ≥ 0.925 |

ASR baseline同时保留Whisper-small、Whisper large-v3-turbo、Distil-Whisper large-v3；LID保留Whisper-small和turbo；Text保留DistilBERT和RoBERTa-base SST-2。完整版本和原始来源见baseline_references.json。

公开candidate universe（参数仅作模型metadata；GB是实际选定权重文件十进制磁盘大小；均以FP32作为13B主operating point）：

| Task | Candidate | 参数metadata | 权重GB | 公开质量证据 | Preliminary | 13B |
|---|---|---|---|---|---|---|
| ASR | [openai/whisper-medium](https://huggingface.co/openai/whisper-medium/tree/abdf7c39ab9d0397620ccaea8974cc764cd0953e) | 763.9M | 3.0555 | wer=0.029 | QUALITY_LIKELY_PASS | 冻结 |
| ASR | [openai/whisper-medium.en](https://huggingface.co/openai/whisper-medium.en/tree/2e98eb6279edf5095af0c8dedb36bdec0acd172b) | 763.9M | 3.0555 | wer=0.031 | QUALITY_LIKELY_PASS | 保留未入选 |
| ASR | [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2/tree/ae4642769ce2ad8fc292556ccea8e901f1530655) | 1543.3M | 6.1734 | wer=0.027 | QUALITY_LIKELY_PASS | 保留未入选 |
| ASR | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3/tree/06f233fe06e710322aca913c1bc4249a0d71fce1) | 1543.5M | 3.0871 | family/card evidence；无对应frozen-set数值 | QUALITY_LIKELY_PASS | 冻结 |
| ASR | [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/tree/ae9ad07059c7c739ffaf932226a8fe64ae2620b0) | 600.0M | 2.4722 | wer=0.0169 | QUALITY_LIKELY_PASS | 保留未入选 |
| ASR | [facebook/wav2vec2-large-960h-lv60-self](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self/tree/54074b1c16f4de6a5ad59affb4caa8f2ea03a119) | 315.0M | 1.2621 | wer=0.019 | QUALITY_LIKELY_PASS | 冻结 |
| LID | [openai/whisper-medium](https://huggingface.co/openai/whisper-medium/tree/abdf7c39ab9d0397620ccaea8974cc764cd0953e) | 763.9M | 3.0555 | family/card evidence；无对应frozen-set数值 | QUALITY_LIKELY_PASS | 冻结 |
| LID | [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2/tree/ae4642769ce2ad8fc292556ccea8e901f1530655) | 1543.3M | 6.1734 | family/card evidence；无对应frozen-set数值 | QUALITY_LIKELY_PASS | 保留未入选 |
| LID | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3/tree/06f233fe06e710322aca913c1bc4249a0d71fce1) | 1543.5M | 3.0871 | family/card evidence；无对应frozen-set数值 | QUALITY_LIKELY_PASS | 保留未入选 |
| LID | [speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9) | UNKNOWN | 0.0852 | error_rate=0.067 | QUALITY_LIKELY_PASS | 冻结 |
| LID | [facebook/mms-lid-256](https://huggingface.co/facebook/mms-lid-256/tree/edc73fd00996e671dfc59d16436a29b12b10588a) | 966.2M | 3.8650 | accuracy=0.934 | QUALITY_LIKELY_PASS | 冻结 |
| LID | [facebook/mms-lid-1024](https://huggingface.co/facebook/mms-lid-1024/tree/6216408c37833201358b509224b019312577dd70) | 967.0M | 3.8682 | accuracy=0.928 | QUALITY_LIKELY_PASS | 保留未入选 |
| TEXT | [philschmid/roberta-large-sst2](https://huggingface.co/philschmid/roberta-large-sst2/tree/7d2599d698b7a805b6831e15e830e60a0b07bdb4) | 355.0M | 1.4216 | accuracy=0.9644495412844036 | QUALITY_LIKELY_PASS | 冻结 |
| TEXT | [philschmid/bert-large-cased-whole-word-masking-sst2](https://huggingface.co/philschmid/bert-large-cased-whole-word-masking-sst2/tree/5c930994422145867b5768ba7113a2aa330b19d2) | 334.0M | 1.3345 | accuracy=0.9438073394495413 | QUALITY_LIKELY_PASS | 保留未入选 |
| TEXT | [textattack/facebook-bart-large-SST-2](https://huggingface.co/textattack/facebook-bart-large-SST-2/tree/8a4baa9662afa056a97a2e9441ae665a06504979) | 406.0M | 1.6295 | accuracy=0.953 | QUALITY_LIKELY_PASS | 冻结 |
| TEXT | [textattack/xlnet-base-cased-SST-2](https://huggingface.co/textattack/xlnet-base-cased-SST-2/tree/9ceeb077dcd5cf5ae790572b2bd6aec755a263be) | 110.0M | 0.4693 | accuracy=0.944 | QUALITY_LIKELY_PASS | 冻结 |
| TEXT | [textattack/bert-base-uncased-SST-2](https://huggingface.co/textattack/bert-base-uncased-SST-2/tree/95f0f6f859b35c8ff0863ae3cd4e2dbc702c0ae2) | 110.0M | 0.4380 | accuracy=0.924 | QUALITY_UNCERTAIN | 保留未入选 |
| TEXT | [vroomhf/deberta-v3-base-finetuned-sst2](https://huggingface.co/vroomhf/deberta-v3-base-finetuned-sst2/tree/297a5dfce6be14d776bd06e98e2b98ab3326c627) | 184.4M | 0.7377 | accuracy=0.9599 | QUALITY_UNCERTAIN | 保留未入选 |

## 5–7. 靶区接近度与memory/latency区别

新候选：**目前无法确认谁最接近γ≈1.2–1.25，也无法确认谁接近γ≈1.9–2.0；全部gamma_est、mu_est为空且分类UNKNOWN**。这表示证据不足，不能声称现实生态不存在这些点。Whisper medium/full-large和base/large Text覆盖真实架构/容量变化，CTC/ECAPA等高效替代也保留；它们是测量探索点，不是已证实near-target候选。

旧ASR baseline可作已测定位参考：turbo γ=1.1662、small γ=1.3005，相对Distil-large-v3位于first-split靶区两侧；旧Text RoBERTa-base γ≈1.4609，相对DistilBERT仍低于Text per-task[1.5,2.0]搜索区间。这些是已有模型的历史测量比例，不是新候选的13A估计，也不是新deployment outcome。

没有证据认定某个新候选“主要是本机memory更大而latency不变”。最接近该假设的metadata对照是MMS-1024 vs MMS-256：同48层backbone，只扩分类头，文件约增0.0815%；不等于μ≈1.05。Whisper full-large-v3相对turbo扩decoder，LID只检测语言token，但仍经过decoder层，不能声称其latency不变。

参数量不能当VRAM；FP16磁盘权重也不能当FP32本机resident memory。例如large-v3磁盘约3.087GB，13B主点会转FP32，尚未计入CUDA context、workspace、activations、KV、allocator。CSV中的size_over_reference_resident_bytes仅是显式标注的disk/idle-byte proxy，不作μ估计或靶区分类。

## 8–9. 公开质量与“大模型不一定更适合”

每个候选的来源、数值/缺失、数据集、checkpoint归属强弱和限制均在quality_evidence.json/source_audit.md；官方卡已固定到SHA。Whisper-medium/medium.en/large-v2的greedy test-clean WER来自论文Table8；large-v3仅用官方跨语言改善证据，不能给它套用large-v2的具体WER。MMS数字是论文对应语言规模system evidence，不能冒充HF checkpoint实测。

BERT-base公开accuracy约0.924，紧贴且略低于0.925，标QUALITY_UNCERTAIN。DeBERTa-v3-base卡报告0.9599但evaluation dataset未注明，不能按名称确认为SST-2 likely-pass。BERT-large/RoBERTa-large/BART-large已有同任务classifier，但参数更大并不保证更接近靶区；MMS-1024更多语言对当前十个reference languages无必然收益。所有未入选候选保留；没有在缺失latency/memory时宣称真实dominance。

## 10–12. 冻结短名单、反cherry-picking、13B下载

短名单：ASR=Wav2Vec2 large-960h-lv60-self、Whisper large-v3、Whisper medium；LID=MMS-LID-256、Whisper medium、SpeechBrain VoxLingua107 ECAPA；Text=RoBERTa-large SST-2、BART-large SST-2、XLNet-base SST-2。每个task3个，medium共享checkpoint下载但分别测ASR/LID，不能把两task的memory重复加成两套模型。

**有没有为了让Generalist赢而挑模型：NO。** 已冻结selection-rule evidence：原quality admission → 真实Q/latency/idle-memory Pareto（含旧baseline）→ 距离Module12靶区union最近 → higher measured quality → lower measured memory → full model ID@revision。Generalist/V3 outcome不进入选择；旧baseline若支配或按规则胜出，保留旧baseline并报告BASELINE_RETAINED，不能强行制造balanced结果。

13B完整测量顺序：ASR→LID→Text，各task按完整model ID排序；先所有9条200样本quality，之后仅admitted paths做3×30单worker测量。每轮task先重测原reference control，再测候选；无low-QPS、hotspot或deployment campaign。历史分母原样固定，同期control比例另报；所有失败保留，不根据结果换checkpoint、调precision/backend或补选。

13B需要下载以下8个精确版本及其必要tokenizer/processor/config文件（本轮尚未下载权重）：

| Checkpoint | Frozen revision | Tasks |
|---|---|---|
| facebook/mms-lid-256 | `edc73fd00996e671dfc59d16436a29b12b10588a` | LID |
| facebook/wav2vec2-large-960h-lv60-self | `54074b1c16f4de6a5ad59affb4caa8f2ea03a119` | ASR |
| openai/whisper-large-v3 | `06f233fe06e710322aca913c1bc4249a0d71fce1` | ASR |
| openai/whisper-medium | `abdf7c39ab9d0397620ccaea8974cc764cd0953e` | ASR,LID |
| philschmid/roberta-large-sst2 | `7d2599d698b7a805b6831e15e830e60a0b07bdb4` | TEXT |
| speechbrain/lang-id-voxlingua107-ecapa | `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9` | LID |
| textattack/facebook-bart-large-SST-2 | `8a4baa9662afa056a97a2e9441ae665a06504979` | TEXT |
| textattack/xlnet-base-cased-SST-2 | `9ceeb077dcd5cf5ae790572b2bd6aec755a263be` | TEXT |

FP32主点、单请求/单replica、原样本与metric/threshold已冻结。SpeechBrain需独立版本环境和本地pretrained_path override；XLNet需SentencePiece。基本推理代码路径来自公开文档，本轮未做runtime smoke；13B先CPU加载/接口核对并锁adapter/environment SHA，再出现新GPU性能结果。

## 13. Gate与停止边界

**REAL_SPECIALIST_SHORTLIST_READY**：各task≥2个公开、无需用户fine-tuning、可本地推理的合法checkpoint且路径有证据；该gate不认证质量已过、γ/μ落靶区、标准24GB实测通过或新regime已成立。13A结果不足以建立真实Pareto frontier。13B后才允许新Specialist成为正式实验模型。

完成Module13A后停止；未开始13B、正式GPU benchmark、最终deployment、low-QPS、multi-hotspot、Task-Rich，也未做人工sleep/slowdown或memory scaling。
