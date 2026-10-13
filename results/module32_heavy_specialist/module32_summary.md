# Module 32：Heavy Specialist Positive-Split Feasibility

最终 Gate：**HEAVY_S_NO_POSITIVE_GAIN**。候选1正式比较=0；正式候选=C2；正式已测臂=3/3；三臂均合法稳定=True。

所有新数据、脚本与审计位于本目录。Module14F–31A历史文件、模型权重与正式优化器保护审计=True。不执行独立长窗口复现。

## 配对与容量

两配对均通过目标ASR历史质量准入：Qwen WER=0.0417321，Phi WER=0.0195199，Whisper-large-v3 WER=0.0394884，冻结门槛0.042181。大模型S原始200条质量已复算。原14B严格高质量pool筛选与目标任务准入是不同判据；没有放宽门槛。

C1：Qwen、heavy S、RoBERTa、ConvNeXt、PARSeq各1，LID/VQA→Qwen。最初实测S容量1.875QPS，但后续S-only在1.68和唯一一次下调0.92QPS均未完整排空；G 0.6QPS诊断亦失败，0.2QPS完整。所有过载窗口NULL U，未用其作效用对照。

C2：Phi、heavy S、Whisper-small、RoBERTa、ConvNeXt、PARSeq各1，LID→Whisper-small，VQA→Phi。S实测闭环容量0.575QPS，mean/P95=1.669/3.539s；0.8QPS上界诊断完整排空但原判据hard-unstable。按测前80%容量规则选择0.46QPS；G .15QPS诊断完整稳定，native ASR mean/P95=3.972/6.366s，基于实测服务推算剩余ASR能力约.234QPS，非长窗容量认证。

|预检|ASR QPS|完成/到达|S执行mean ms|S等待mean ms|预检通过|原始稳定性|
|---|---:|---|---:|---:|---|---|
|B_S|1.4|180/180|516.3151395904762|531.3441831904763|True|recoverable_congested|
|B_G|0.6|122/145|None|None|False|unstable|
|B_G_SAFE|0.2|148/148|None|None|True|recoverable_congested|
|B_PRESSURE_S|1.68|300/317|920.2443137880435|11306.510936304347|False|unstable|
|B_ADJUST_S|0.92|227/228|1720.9375510485438|27261.267832300975|False|unstable|
|B_C2_S|0.8|148/148|1613.9469565079362|6823.4714752222235|False|unstable|
|B_C2_G|0.15|116/116|None|None|True|recoverable_congested|
|B_C2_PRESSURE_S|0.46|166/166|1144.484717489796|697.9490152653061|True|recoverable_congested|
|B_C2_PRESSURE_G|0.46|166/166|1162.0364595227272|739.4687415909092|True|recoverable_congested|

共置服务需求有明显变化，历史standalone S=494.913ms，不能直接用作本轮稳定容量。Qwen同输入25样本native服务944→3117ms且预测文本一致；worker源码及Qwen/Specialist/OCR依赖版本与31A一致，温度/频率/驻留诊断未唯一识别膨胀因果。源数据与完整所有背景延迟见 colocation_profile.json、S_service_expansion_by_arrival_block.csv、G_service_drift_audit.json。

## 冻结与三臂

正式冻结ASR=0.46QPS，五背景各.2QPS，B3=34567.5534MiB；同一六模型池、各1副本、输入、预算、原Module20A效用。每臂600秒，30秒warmup另计，最大60秒drain，顺序5%→S-only→10%。请求轨迹与SHA在frozen_load/experiment_manifest中；负载选择从未使用ΔU。

|臂|目标/实际G份额|实际G/S ASR|合法U|合法ΔU|稳定性|
|---|---|---|---:|---:|---|
|A0|0/0.0|0/280|-2.2201052231044724|0.0|recoverable_congested / VALID|
|A1|0.05/0.060714285714285714|17/263|-3.092861788800474|-0.8727565656960015|recoverable_congested / VALID|
|A2|0.1/0.12142857142857143|34/246|-2.5465509908313138|-0.3264457677268413|recoverable_congested / VALID|

严格复用Module20A：Q/T/L/P/M=0/.25/.30/.30/.15，offered QPS权重、逐任务原参考时延、窗口吞吐损失、safe-memory定义均不变；质量只硬准入，原stability_v2与完整cohort/有界排空/资源判据不变。仅合法匹配双方计算ΔU。Module32按用户冻结的“一个合法配置ΔU>+.01”接受探索性正例；旧Module30样本及环境额外检查保留为诊断，不替代该定义。

## 效用与机制

- A1：同输入保留S等待减少 276.796ms/请求、总72.797s；G新增绝对执行/等待=72.169/2.302s。G相对原S执行增量=2798.115ms/请求；保留S执行变化=-104.920ms；ASR整体mean变化=-244.358ms。
- A2：同输入保留S等待减少 322.450ms/请求、总79.323s；G新增绝对执行/等待=92.411/10.852s。G相对原S执行增量=1202.421ms/请求；保留S执行变化=-21.733ms；ASR整体mean变化=-243.671ms。

A1：ASR mean/P95效用贡献变化=0.17532260419370924，五背景贡献变化=-1.0479827626656526；T/M贡献另见system_utility_comparison.csv。全部Text/Image/OCR/LID/VQA mean/P95变化保留，不按改善任务挑选。

A2：ASR mean/P95效用贡献变化=0.2083740909125886，五背景贡献变化=-0.5347138106929663；T/M贡献另见system_utility_comparison.csv。全部Text/Image/OCR/LID/VQA mean/P95变化保留，不按改善任务挑选。

Module31A：S执行227.57ms、等待36.06ms，5%/10%同输入S等待减少1.472/4.816ms。本轮预检S等待更大；正式可释放等待、G代价、共置服务膨胀以及背景抵消程度以本轮匹配分解为准。跨模块G和LID路径、负载及运行状态不同，只比较机制，不宣称单一模型重量因果。

未找到满足本轮Gate的合法探索性正例。未提出或实施正式优化器修改；不合法与负收益结果全部保留。

5%实际G样本较少，G路径P95及单次效应不确定；保留实际份额与全部样本，不外推可复现性。

## 审计与限制

最初READY任务声明适配遗漏在测量前修复，保留initial_ready_failure；G容量探测一次降载、无请求备用worker的审计修正、C1唯一压力负载调整均有归档与时间戳。没有重跑已测GPU窗口、按正收益选种子、人工延时、修改原推理设置或增加无意义计算。所有正式窗口仅一次探索。

温度、频率、功率、显存、CPU与异常原始样本见raw_runs、GPU_CPU_environment.csv与resource_comparison.csv；完整请求journal及collector/sender/reader退出、GPU释放见原始审计。显存为NVML采样，非瞬时allocator最大值。Git元数据不可用，源码/checkpoint SHA及依赖冻结代替版本身份记录。

窗内bootstrap仅诊断，不含独立跨运行不确定性，不满足Module20A独立holdout规则。未自动执行独立长窗口、修改正式优化器或启动下一模块；到此停止。

详细Stage D四项判断：heavy_specialist_mechanism_analysis.md / .json。

正式分流未观察到保留S服务变慢：5%/10%同输入执行分别变化−8.386%/−1.770%。但本轮S-only共置执行mean/P95=1263.049/2691.957ms，相较历史独立画像494.913/960.253ms有明显膨胀；跨画像输入顺序/负载/运行条件不同，不能唯一归因GPU共置。两分流臂的VQA尾部损失是背景负效用的主要来源。

最终数据审计与派生表交叉核对均通过：final_data_acceptance.json、final_artifact_review.json。数据索引见README.md；图见utility_and_queue_analysis.png / .pdf。继承的transport文字标签说明另存provenance_annotations.json，实际新冻结轨迹与SHA保持不变。
