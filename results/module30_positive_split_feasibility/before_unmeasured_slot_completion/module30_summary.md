# Module30 GPU-Measured G/S Positive-Split Feasibility Screening

Gate：**NO_POSITIVE_FEASIBILITY_CANDIDATE**。已停止；不自动进入下一Module或独立GPU复现。

Stage0：2100秒CPU并发压力、传感器/写盘/退出故障注入及200请求原生运输fixture通过；Module29错误、原代码及Gate保持不变。新遥测通过runtime.lock快照与串行采样修复并发冲突，日志与raw可逐请求核对。

StageA：39个已准入G/S共同任务配对排名；最多两个组合，Qwen+Whisper-small与Phi+Whisper-small ASR。两组五模型各1副本；Text/Image/OCR/LID/VQA各0.2QPS，ASR1QPS；显存34567.5534MiB；0/5/10%比例、样本、轨迹和顺序GPU前冻结。历史负收益和服务漂移保留，无共置预测倍率。

StageB：尝试6/6，完整5，合法稳定3。每次600秒，固定30秒预热、60秒有界排空、热状态控制。未运行槽和技术停止见campaign_completed.json。

|组合|方案|合法U|ΔU vs S-only|稳定/合法|
|---|---|---:|---:|---|
|qwen + whisper-small|s_only|None|基线|steady / False|
|qwen + whisper-small|split05|None|None|steady / False|
|qwen + whisper-small|split10|-1.2243815964928317|None|steady / True|
|phi4mm + whisper-small|s_only|-0.8558152695888688|基线|steady / True|
|phi4mm + whisper-small|split10|-1.4808522241946103|-0.6250369546057415|steady / True|

最终六问：
1. 测前优先级第一是Qwen+Whisper-small ASR；本轮实测最佳合法差值：{'candidate': 'C2', 'arm': 'split10', 'status': 'LEGAL_MATCHED', 'delta_U': -0.6250369546057415, 'all_three_legal_stable': False, 'positive': False, 'Generalist': 'phi4mm', 'Specialist': 'whisper-small', 'G': 52, 'S': 520, 'G_fraction': 0.09090909090909091, 'split_U': -1.4808522241946103, 'S_only_U': -0.8558152695888688, 'temperature_difference_c': 1.158908280439256, 'postwarm_temperature_difference_c': 0.0, 'sm_clock_relative_difference': 0.0018877492780529082, 'environment_comparable': True, 'sample_gate': True, 'T_delta': 0.0, 'L_delta': -0.07683078089689765, 'P_delta': -0.5480422814279456, 'M_delta': -0.00016389228089830343}。优先级不是预言正收益。
2. 实际观测到合格G/S优于S-only：False。有限制的合法正信号：[]。不稳定U为NULL，未用截尾延迟计算收益。
3. 物理变化逐组合见positive_split_mechanism.md与worker_interference_analysis.csv：S同输入服务/等待、G额外执行与等待、背景mean/P95及最终U均分列；不把服务变化唯一归因纯GPU。
4. 背景任务最大负效用贡献：{'candidate': 'C2', 'arm': 'split10', 'task': 'visual_qa', 'n_split': 120, 'n_control': 120, 'mean_delta_ms': 190.46646344166672, 'p95_delta_ms': 1465.3032675, 'mean_U_delta': -0.03568648712631452, 'p95_U_delta': -0.2745445326536471, 'legal': True}；全部六任务和有效n见per_task_latency.csv/task_utility_changes.csv。
5. 值得另行独立长窗口验证的合格候选：无；本轮只提供开发证据，独立验证未自动启动。
6. 若无正收益，应结合实际S/G性能差距、共享S服务膨胀、VQA等待和S-only队列压力判别，不能把一个负载点推为所有模型/负载不存在。对本轮这些因素的数值分解已完整保存，StageD无正候选时不执行。

局限：每组合只有一条配对轨迹、三个600秒开发窗口；未证明长时域稳定或论文独立复现。G样本仅21–53左右，尾部估计仍有限；只测两个固定比例/一个负载点，不宣称全局最优。所有已准入质量门槛、20A定义/参考与原stability_v2阈值未改。

完整性：17642个历史文件/模型权重元数据和冻结科学源码hash保持不变。正式优化器未修改。

## 必须阅读的遥测修正

初版SafeMonitor.check与NVML采样共用长锁，CPU替身复现250ms阻塞，第一窗口背景提交延迟均值35–55ms。前两个完整600秒窗口全部原始请求保留，但合法可用于物理可行性筛选的U为NULL；原始完整cohort算式保存在v1_observer_affected_diagnostics.json，禁止当正收益证据。没有截尾效用、没有补跑已执行槽。

当前窗口清洁退出并释放GPU后停止初版；v2错误检查和phase发布使用独立短状态锁，传感器I/O与journal串行锁不进入请求dispatch路径。250ms传感器阻塞下12次控制检查均微秒级；原生80请求/40请求中止测试与2100秒慢传感器并发压力在剩余GPU之前重新通过。完整证据见telemetry_fix_audit.md和frozen_gpu_manifest_v2.json；初版manifest、源码、错误与数据保留。

四个未执行的原冻结槽继续测量：C1_split10及C2三方案。模型、负载、seed、输入和比例均不改，总正式窗口6次。C1跨监测器版本不作合法ΔU；只有C2有完整同版本三臂对照。当前权威Gate为 **NO_POSITIVE_FEASIBILITY_CANDIDATE**；初版表观差值-0.2654730641914709只作技术诊断。未自动启动独立验证或下一Module。
