# Module27：GPU实测正分流发现

Gate：**NO_POSITIVE_EXPLORATORY_CANDIDATE**。停止；不自动进入后续Module。

固定一个五模型池：Qwen2.5-Omni、Whisper-medium、RoBERTa-base-SST2、ConvNeXt-tiny、PARSeq，均1副本。B4=34567.5534MiB。

目标LID=10.5/11.5QPS，Text/Image/OCR/ASR/VQA各0.2QPS；ASR与LID共享Whisper-medium，VQA走Qwen。只改变LID的0/10/20%G份额。

预注册StageB最多6次（用户上限12），每次600s；实际启动1次。完整搜索空间、失败/跳过及理由见discovery_protocol.json、candidate_screening.csv、stage_b_campaign.json。

效用严格沿用20A的offered-QPS加权任务mean/P95、窗口吞吐损失及safe memory；pooled P95单报，不能代替任务P95。原始质量准入为硬约束，未补齐缺失质量分数。

所有native service、queue/E2E、worker busy/backlog、实际route、热状态/时钟/功率、READY/模型身份、退出及释放均保留。缺乏精确worker receipt ACK，host send-start不冒充接受时刻；显存峰值为0.5s NVML采样高水位。

探索比较：

|负载|G份额|ΔU|合法稳定|温度可比|进入独立复现|
|---|---:|---:|---|---|---|

600s开发窗口的P95采用逐任务实际n报告条件order-statistic区间；至少72个样本才有有限equal-tail95上界，仍依赖iid，队列时间相关不保证。60s block区间仅单次配对的条件敏感性，开发正例不能算独立验证。

没有合格的开发正候选，StageC独立GPU未执行。

独立正收益门槛未通过，StageD正式优化器检查未执行；不能由未执行断言算法必定失败。

六个问题：

1. 真实正分流：开发候选=False，独立复现=False。
2. 胜出配置：None；G份额=None，模型/副本同上述冻结池。
3. 原因必须以匹配请求的S/G服务、共享队列和全部六任务U分项为依据；见exploratory_utility_comparison.csv/independent_paired_utility.csv的matched_path_service_effects、per_task_latency.csv及utility_decomposition.csv。不能把名义容量或单目标吞吐当真实U收益。
4. 独立GPU复现：NOT_EXECUTED。
5. 正式优化器：NOT_EXECUTED。
6. 若未找到，仍缺少在同驻留、稳定可比且采样充分的独立配对上同时改善S队列并限制G慢路径/背景干扰的正效用证据；不得靠改效用、重试种子或扩充搜索制造结果。

历史16226文件metadata与读取科学来源SHA256不变，14F–26原实验/gate保持原样。所有负结果及不稳定的U=null证据保留。
