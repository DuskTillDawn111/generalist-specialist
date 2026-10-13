# Module26 CPU证据审计

Stage A：**LEGAL_DEVELOPMENT_POSITIVE_EXAMPLE**。+0.4767982826570303由1500/1500完整GPU配对的20A系统U重算得到。仅为合法开发正例，不是独立验证。

原配置：Qwen、Whisper-large-v3、RoBERTa-SST2、ConvNeXt-tiny、PARSeq各1副本；背景text4/image2/ocr5/lid.03/VQA.2，ASR1.52，ASR G份额18%。对照只将ASR改为全Whisper，其他驻留、请求、路由完全匹配。预算34567.5534MiB，原20A权重/refs不变。

24B改为Text/Image/OCR各2副本，共8worker，ASR13.1833%，Text/Image另有极小G流量。QPS、checkpoint、GPU、输入池相同，但seed、输入cohort、运行顺序不同。不是同配置的独立重复。具体逐任务/worker服务与等待、输入分布见JSON和CSV，以及25原始归因输出。

| 历史实验 | G/S U | S-only U | ΔU |
|---|---:|---:|---:|
| 24A | -1.779873957 | -2.256672240 | 0.476798283 |
| 24B | -1.625463080 | -1.518752524 | -0.106710556 |

24A S-only为steady，G/S为recoverable_congested（峰值拥塞告警但最终恢复），均无硬unstable或持续增长；各物理worker重建和safe/peak显存通过。全部请求完整、输入/到达/路由匹配、warmup完全drain且先于测量，E2E由scheduled到completion重算一致；启动READY修复前无任何测量，失败原样保留。

原正收益主要是背景Text/OCR的mean/P95改善，不能归因于单独ASR收益或自动称分流必然产生改善。逐项贡献见historical_utility_decomposition.csv；Text/OCR含数百请求，但LID仅2、VQA32，后者尾部很不确定。两次历史GPU温度/时钟证据不足，不能唯一分离热状态、副本共置、输入、顺序的物理原因。

质量按原supported path准入，不用短时样本准确率改准入；原始WER/correct保留描述性报告。全部原始JSON/CSV/log及manifest已读取并hash冻结；本轮没有开发预测器或全池搜索。
