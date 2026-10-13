# Module28：Queue-Aware Dynamic G/S Routing Feasibility Study

Gate：**DYNAMIC_NO_UTILITY_GAIN**。完成后停止，不自动进入下一Module。

固定Qwen2.5-Omni、Whisper-medium、RoBERTa、ConvNeXt、PARSeq各1副本，六任务QPS：Text/Image/OCR/ASR/VQA各0.2，LID11.5。B4=34567.5534MiB；20A效用、归一化和质量准入未改。S-only、开发预选最佳已测非零固定10%、Dynamic对照；0%同样是固定路由，且是历史更强对照。非零比例并非全局连续最优。G-only因服务需求明显超过单副本容量而预先排除。

探索1/6次，每臂600s，独立验证0/9次，预算15次/9000s正式窗口。warmup30s、prime、冷却与排空单报；失败不重试，全部记录保留。

|阶段/轨迹/组|对照|ΔU|合法稳定|匹配热状态|
|---|---|---:|---|---|

逐任务效用贡献、六任务mean/P95及IID条件P95区间见utility_decomposition.csv、per_task_latency.csv；全系统pooled mean/P95、offered/completed QPS、safe/peak显存、实际路由比例和开销见all_gpu_runs.csv；逐请求状态/原因、worker queue/service与匹配输入干扰见routing_decision_trace.csv、worker_metrics.csv和worker_interference_analysis.csv。pooled P95不代替效用中的逐任务P95。

独立效应：[]，t95区间：None。n=3小样本强依赖近似正态，符号检验最小单侧p=0.125；单配对60s block仅条件敏感性，队列相关影响IID区间；不推广到其他模型池。

质量保持原冻结准入路径，实测子集accuracy/WER/CER/closed-set有效性同时保留，不填缺失指标，不把小子集得分当新准入认证。GPU显存为0.5s NVML采样高水位，瞬时allocator峰值UNSUPPORTED。无worker receipt/START ACK，在线只用已入队未完成状态和已收到RESULT；原始handler时间仅用于事后分析。混合运行服务变化不能唯一归因到GPU/CPU/热状态中的某一种。

未得到满足预注册门槛的动态正收益候选；独立复现不执行，停止扩展参数/负载网格。

停止原因：{'run_id': 'C_steady_s_only', 'reason': 'ILLEGAL_UNSTABLE_OR_INFRASTRUCTURE_FAILURE', 'errors': [], 'worker_states': {'qwen#0': 'steady', 'openai/whisper-medium@abdf7c39ab9d0397620ccaea8974cc764cd0953e#0': 'unstable', 'roberta_sst2#0': 'steady', 'convnext_tiny#0': 'steady', 'baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt#0': 'steady'}}。缺失比较或不完整运行不可计算正常U。

五个问题：

1. 真实动态正收益：探索=False；独立复现=False。
2. 路由到G时的状态：S保守队列代价高，G没有host可见未完成请求、目标在途为0，token/冷却/准入/容量通过；没有把平均GPU利用率当空闲。具体收益和状态必须结合实测表，不能由路由条件直接证明因果。
3. 优于S-only和最佳已测固定对照：探索=False，独立=False；逐组差值全部报告。
4. 独立GPU收益复现：False。
5. 当前证据不支持集成进正式部署优化器；不宣称优化器创新成功。

历史16350文件metadata和已读科学来源SHA256保持不变，全部负结果保留。
