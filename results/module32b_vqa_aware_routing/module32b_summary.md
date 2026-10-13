# Module 32B：VQA Interference Attribution and Background-Aware Dynamic Routing

最终 Gate：**DYNAMIC_EXPLORATORY_POSITIVE**。正式运行3/3，三臂合法稳定=True，同轨迹=True，动态真实G ASR=4。历史及模型保护=True，最终审计=True。

## 1. Phi ASR为何损害VQA

Phi worker同步串行处理ASR与VQA，同一输入流先入先出且不可抢占。历史Fixed5/Fixed10的VQA等待分别有91.16%/95.42%与Phi ASR执行区间重叠；各臂VQA最慢5%全部被ASR挡住。已有VQA时接纳ASR的次数仅2/4，不能解释全部损失；Phi空闲时接纳ASR，随后新VQA到达也形成严重头部阻塞。

## 2. 内部队列还是GPU共置

主机制分类A（内部排队为主）。历史VQA服务mean/P95：S-only 283.957/396.402ms；Fixed5 292.431/373.232ms；Fixed10 232.012/332.081ms。等待mean/P95却从11.697/36.678ms增至475.000/3509.407、279.644/1943.949ms。历史VQA整体ΔU贡献−0.968148/−0.496859。P95代数Shapley分解显示排队增加主导，不能将服务P95与等待P95直接相加。

Whisper及若干Specialist与Phi ASR墙钟区间重叠时有服务变化关联，不能唯一归因GPU。现有native handler同步墙钟和NVML采样缺少kernel/context因果观测。E2E保持host monotonic；跨worker区间用同机Unix，不混减时钟。详见vqa_interference_attribution.md与clock_and_cuda_timing_audit.json。

## 3. 可控制的损失和冻结策略

只控制接纳：Phi有任何在途请求、VQA排队/在途、S低压或预计边际风险过高时走S；否则用当前音频时长分桶成本、S在途服务需求估计和未来VQA泊松到达风险做检查。S压力开/关=2500.0/1250.0ms；G执行尾部上限=2610.101ms；最大G在途1、接纳份额≤已到达ASR的5%、间隔≥10s、预计VQA暴露预算3%。参数与代码在GPU前冻结，正式窗口后未调整。

当前空队列无法知道未来VQA；ASR不可抢占，未来背景损失只能风险约束而不能保证消除。仅用当前输入元数据和past pending/host时间，无未来请求、reference label、native start/end或事后最佳请求ID。只读机会回放发现6个候选状态，不更新反事实队列、不计算反事实U。最短输入桶G只有11个校准样本，预测有明显不确定性。CPU风险模型定义修正保留于cpu_draft_variants与CPU_model_definition_revision，全部在GPU前。

## 正式GPU对照

同Module32六模型池/单副本，ASR=.46QPS，五背景各.2QPS，B3=34567.5534MiB。三臂同输入、同arrival/route uniforms轨迹，871到达（280 ASR、133 VQA），30s warmup、600s正式、60s最大drain及原热控制。顺序Fixed10→S-only→Dynamic。历史Module32 U不用于严格对照，三臂全部重测。

|臂|U|ΔU vs S-only|G/S ASR|实际G份额|完成/到达|原始稳定性|
|---|---:|---:|---|---:|---|---|
|A0|-1.102778931764138|0.0|0/280|0.0|871/871|steady|
|A1|-1.4706584235410125|-0.3678794917768744|34/246|0.12142857142857143|871/871|steady|
|A2|-1.0695570741086375|0.033221857655500564|4/276|0.014285714285714285|871/871|steady|

|臂|VQA执行mean/P95 ms|VQA等待mean/P95 ms|VQA E2E mean/P95 ms|VQA数|
|---|---|---|---|---:|
|A0|206.201/332.334|5.890/0.628|213.889/341.898|133|
|A1|204.366/319.905|149.146/1059.570|355.632/1211.756|133|
|A2|200.478/331.647|5.635/0.558|207.580/337.241|133|

六任务完整mean/P95与U贡献见gpu_per_task_latency.csv、gpu_task_utility_decomposition.csv。T/L/P/M按原Module20A公式，不稳定U=NULL，完整cohort与有界排空不变。每个模型backlog和原判据见raw_runs、formal_acceptance、gpu_backlog_timeseries.csv。GPU/CPU显存、温度、频率、功率、吞吐、异常和路由开销见gpu_sensor_samples.csv与gpu_resource_comparison.csv。

## 4. 是否保留Whisper实际排队收益

Dynamic与同输入S-only对照，保留S等待平均减少2.344ms，总减少0.647s；保留S服务均值变化-6.250ms。G新增绝对执行/等待=4.775/0.001s，相对同输入原S额外执行=3.493s。单次顺序有运行漂移，不能把所有等待差归为路由因果。

动态真实G执行=4，实际份额=0.014285714285714285；VQA等待被动态G ASR覆盖请求=0、覆盖总0.000ms；ASR期间未来新VQA到达=0。原型不读取这些未来到达，全部是事后诊断。

## 5. 六任务总效用是否提高

Dynamic ΔU vs S-only=0.033221857655500564；vs Fixed10=0.40110134943237497。只有双方均合法稳定、真实G/S协作且两差均>+.01，才标记DYNAMIC_EXPLORATORY_POSITIVE。完全回退S-only不能算协作正例。

Dynamic vs A0：ASR L/P贡献+0.003813、五背景+0.029409（其中VQA+0.002815）；完整T/M分项保留。

Dynamic vs A1：ASR L/P贡献+0.154993、五背景+0.246003（其中VQA+0.262455）；完整T/M分项保留。

动态决策原因分布：{'S_low_pressure': 271, 'eligible': 4, 'Phi_busy_backpressure': 3, 'G_execution_tail_risk': 2}。预测与实际native服务、匹配基线等待的差见dynamic_prediction_diagnostics.csv。逐决策允许状态重放与past时间检查见causal_router_replay_audit.json。

## 6. 是否值得独立复现

值得另行设计独立匹配复现，但本次仅探索性正例，不宣称可靠或可重复；未执行独立长窗口。

所有历史实验与Gate不变，权重与正式优化器不变。GPU已释放。未自动启动下一Module或长窗口，到此停止。

## 历史服务需求漂移

本轮重测Fixed10与Module32 Fixed10的871个同输入/同路由输出全部一致，但保留S执行mean从1206.418降至569.456ms、G ASR从2717.969降至1815.156ms，其他Specialist也有变化。相同native源码、参数、驻留与GPU身份不等于服务需求不漂移；跨轮不能唯一识别原因。原冻结负载与动态参数没有随变化调整，高S压力机会可能减少。正式效用只使用本轮重测的三个匹配臂，仍属于单次序列探索，详见cross_run_same_input_service_drift.json。

## 探索性正收益敏感性

相对S-only的+0.033222主要依赖Text尾部变化（L/P贡献+0.040279）；ASR仅+0.003813。减去Text变化的代数诊断残差−0.007057，并未重定义U或Gate。4次动态ASR期间没有未来VQA到达，不能证明背景保护的可重复性。值得另行独立复核，但不支持部署或优化器升级；详见positive_evidence_limitations.md及窗内bootstrap诊断。
