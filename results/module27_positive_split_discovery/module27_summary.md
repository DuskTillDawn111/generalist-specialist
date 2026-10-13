# Module27：GPU实测正分流发现

Gate：**NO_POSITIVE_EXPLORATORY_CANDIDATE**。停止；不自动进入后续Module。

固定一个五模型池：Qwen2.5-Omni、Whisper-medium、RoBERTa-base-SST2、ConvNeXt-tiny、PARSeq，均1副本。B4=34567.5534MiB。

目标LID=10.5/11.5QPS，Text/Image/OCR/ASR/VQA各0.2QPS；ASR与LID共享Whisper-medium，VQA走Qwen。只改变LID的0/10/20%G份额。

预注册StageB最多6次（用户上限12），每次600s；实际启动5次。完整搜索空间、失败/跳过及理由见discovery_protocol.json、candidate_screening.csv、stage_b_campaign.json。

效用严格沿用20A的offered-QPS加权任务mean/P95、窗口吞吐损失及safe memory；pooled P95单报，不能代替任务P95。原始质量准入为硬约束，未补齐缺失质量分数。

所有native service、queue/E2E、worker busy/backlog、实际route、热状态/时钟/功率、READY/模型身份、退出及释放均保留。缺乏精确worker receipt ACK，host send-start不冒充接受时刻；显存峰值为0.5s NVML采样高水位。

探索比较：

|负载|G份额|ΔU|合法稳定|温度可比|进入独立复现|
|---|---:|---:|---|---|---|
|lid_10.5|0.1|-0.16630431254723366|True|True|False|
|lid_10.5|0.2|-5.2594018428238245|True|True|False|
|lid_11.5|0.1|-0.6543351692334687|True|True|False|

600s开发窗口的P95采用逐任务实际n报告条件order-statistic区间；至少72个样本才有有限equal-tail95上界，仍依赖iid，队列时间相关不保证。60s block区间仅单次配对的条件敏感性，开发正例不能算独立验证。

没有合格的开发正候选，StageC独立GPU未执行。

独立正收益门槛未通过，StageD正式优化器检查未执行；不能由未执行断言算法必定失败。

六个问题：

1. 真实正分流：开发候选=False，独立复现=False。
2. 胜出配置：None；G份额=None，模型/副本同上述冻结池。
3. 三项分流均未改善全系统U：S路径服务时间增加，S等待未得到有效缓解，G慢路径及VQA等待增加；损失主要来自mean/P95，详见下方实测归因。
4. 独立GPU复现：NOT_EXECUTED。
5. 正式优化器：NOT_EXECUTED。
6. 若未找到，仍缺少在同驻留、稳定可比且采样充分的独立配对上同时改善S队列并限制G慢路径/背景干扰的正效用证据；不得靠改效用、重试种子或扩充搜索制造结果。

历史16226文件metadata与读取科学来源SHA256不变，14F–26原实验/gate保持原样。所有负结果及不稳定的U=null证据保留。

设备与运行会计：第一次预启动仅做NVML身份核验，发现Module18历史GPU UUID与当前Module26 GPU不同；模型未启动，GPU推理/测量数为0，U不存在。该失败的全部原始输出及首次gate单独原样归档于initial_prelaunch_failure/。
当前设备身份在任何模型推理前另行冻结于hardware_identity_amendment.json；模型/QPS/预算/效用/输入/种子不变。为了不增加原6次预算，这次失败占1个检查名额，最多5次实际探索；预先取消11.5QPS的20%格点，保留0/10%，不是按GPU收益事后筛除。
历史Module18/19不同GPU的服务时间只是筛选先验，不是当前GPU容量认证；同机当前GPU配对才是效用证据。GPU型号、总显存及driver相同仍不能证明不同UUID性能等价。

## 完整实测负收益归因与验收

|LID QPS|G份额|ΔU|mean贡献差|P95贡献差|吞吐贡献差|显存贡献差|条件block范围|
|---:|---:|---:|---:|---:|---:|---:|---|
|10.5|10%|-0.166304|-0.078524|-0.087713|0.000000|-0.000067|[-0.362664, 0.020598]|
|10.5|20%|-5.259402|-0.865974|-4.393324|-0.000036|-0.000067|[-8.637562, -2.251789]|
|11.5|10%|-0.654335|-0.175393|-0.478875|-0.000067|0.000000|[-0.757046, -0.343581]|

三个区间均只是60s block的单次配对条件敏感性；10.5QPS/10%跨零，其他两项条件范围为负，不能把它们当独立复现CI。全部五次测量完整完成35,343个请求，每臂600s，全部排空成功且通过原硬稳定性/显存判据；10.5QPS/20%的Qwen与11.5QPS的Whisper有recoverable_congested记录，不能称为完全无拥塞。

10.5QPS/10%：同输入Whisper LID服务mean 69.20→74.13ms，等待181.72→179.42ms；VQA等待6.76→79.67ms，LID整体E2E mean/P95由249.78/797.84→270.96/820.41ms。卸载几乎未减少S等待，却增加G慢路径与背景等待。

10.5QPS/20%：同输入S LID服务69.16→78.71ms，等待183.31→180.15ms；VQA等待6.76→1205.85ms，VQA P95 700.45→4019.95ms，LID P95 797.84→1989.14ms。效用损失主要是P95；Qwen可恢复拥塞且最终清空，不截尾计算。

11.5QPS/10%：同输入S LID服务69.16→74.60ms，等待293.84→318.23ms；VQA等待20.49→161.16ms。S少接请求后的服务及等待仍变慢，不能用固定服务时间假设宣称容量获益。这些是匹配输入的运行观测，尚不能唯一归因到物理GPU干扰、热状态或CPU竞争中的某一种机制。

三项实际G份额分别为9.962%、19.811%、9.477%，G/S均实际承担目标请求。最高safe memory为26960.43MiB，NVML采样峰值24270MiB，均低于B4。六任务完整mean/P95与条件区间见per_task_latency.csv；逐模型任务服务/等待/质量输出另见negative_split_attribution.json。背景任务每臂约102–135请求，P95仍有采样与相关性不确定性。

原冻结CPU验收脚本的样本字段路径错误及失败日志保留；post_run_verify.py按原生item.sample.id逐条核验，不修改GPU原始记录或冻结配置，不重跑GPU。analysis_acceptance_amendment.json记录此CPU修正；最终验收见final_verification.json。没有正开发候选，独立复现与正式优化器阶段未执行，不能宣称正分流不存在于所有场景，也不能宣称优化器已成功或漏选。缺少新的、同驻留合法稳定且能同时改善S等待并限制G代价的正收益配对证据。
