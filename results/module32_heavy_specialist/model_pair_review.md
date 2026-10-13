# Module32 配对审查

ASR质量门槛固定WER≤0.042181；安全显存因子固定1.110853955254881，B3=34567.5534MiB。所有单副本的历史内存与能力来源见JSON、质量provenance和Module14F memory audit。

|候选|G WER|S WER|安全表显存MiB|G ASR历史ms|S ASR历史ms|GPU顺序|
|---|---:|---:|---:|---:|---:|---|
|C1 qwen + large-v3|0.04173210679829482|0.03948844514247252|30883.961663996204|654.0784572333333|494.912528|先测；压力点不合法，不进入正式对比|
|C2 phi4mm + large-v3|0.019519856405654026|0.03948844514247252|25765.88720748521|1129.588914977778|494.912528|候选1预检失败后测|

候选1：Text→RoBERTa SST2，Image→ConvNeXt tiny，OCR→PARSeq，LID/VQA→Qwen，ASR→heavy S / Qwen；五副本。

候选2：Text→RoBERTa SST2，Image→ConvNeXt tiny，OCR→PARSeq，LID→Whisper-small，VQA→Phi，ASR→heavy S / Phi；六副本。其余五任务均0.2QPS，ASR负载按新共置容量数据选择，正式比较前冻结。

Whisper-large-v3的Module14B FAIL_QUALITY是相对Distil baseline的高质量重型pool判据；原目标ASR准入通过。200条原始WER已复算，无门槛放宽。FP32、batch1、native attention/生成设置与已有13B worker一致；只复用24/26的READY任务适配，未加入推理延迟或额外计算。

候选1G实际执行比历史明显变慢，短容量画像不能覆盖随后S服务膨胀；已有负结果不筛除。候选2执行一组有界受控预检；最多一个通过预检的候选执行三臂正式比较，不扩大模型搜索。
