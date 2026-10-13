# Module28：Queue-Aware Dynamic G/S Routing Feasibility Study

Gate：**DYNAMIC_EVIDENCE_GAP**。本轮证据不足；不能判定动态分流有益或无益。已按冻结停止规则结束，不自动进入下一Module。

Stage A读取Module19–27原始日志，重点审计26/27。历史27的11.5QPS S-only存在真实S排队/G空队列窗口：6806个LID到达中1010个满足S≥8个未完成请求且G空队列，737个满足S≥10。native handler区间支持这些窗口的历史G闲态。平均GPU利用率未作为闲态证据。固定10%曾增加S服务69.13→74.60ms、等待294.02→318.23ms，VQA等待20.49→161.16ms；收益机会仅为可补测假设，未执行路由的反事实U为UNSUPPORTED。

Stage B原型与参数在GPU前冻结，独立于正式优化器。在线仅使用host已入队未完成工作和已收到RESULT服务时间；S等待排除首个可能执行中的请求，G代价含慢路径、S服务膨胀和背景外部代价。G最多1个在途目标请求、0.575rps令牌上限、约0.506s完成后冷却、65.32/32.66ms进入/退出滞回、显著服务预测误差或S服务膨胀时30s回退；准入或容量不满足回退S。工程安全系数及开发来源全部记录，未用最终GPU结果调参。原型CPU检查通过，但动态GPU行为尚未验证。

固定Qwen2.5-Omni、Whisper-medium、RoBERTa、ConvNeXt、PARSeq各1副本。Text/Image/OCR/ASR/VQA各0.2QPS，LID11.5QPS；B4=34567.5534MiB。20A效用、归一化、质量准入、原显存与容量/稳定性约束未改。预注册探索6次，每次600s；稳态与突发各比较S-only、历史最佳已测非零固定10%与Dynamic。0%是历史更强固定对照并单独纳入；未宣称10%是连续全局最优。G-only单副本服务需求明显>1，GPU前排除。

Stage C仅完成首个稳态S-only窗口：7607/7607请求全部完成，窗口末有32个积压，峰值39；最终150s的bin平均队列斜率+0.120146请求/s、增长12.776、递增比例75%。原分类器以`sustained_positive_tail_growth_with_nontrivial_end_backlog`判为unstable。虽然2.445s排空、无运行错误、全部worker干净退出、显存合规，仍不允许恢复合法U。这个判定是冻结有限窗口规则，不证明无限时域容量必然超限，也不证明动态路由失败。

|指标|本次S-only观察|
|---|---:|
|合法系统U|NULL：稳定性硬约束失败|
|全系统pooled mean/P95|580.306/1703.958ms|
|LID mean/P95|605.007/1718.010ms|
|LID服务mean/等待mean|70.209/534.251ms|
|offered/completed cohort QPS|12.678333/12.678333|
|窗口内completed QPS|12.625000|
|safe memory/NVML采样峰值|27020.412/24324MiB|
|实际目标G/S|0/7005|

六任务mean/P95、质量子集、worker执行/等待及热状态全部保留。有限raw arithmetic U仅为INELIGIBLE诊断，不可排序、不可计算合法ΔU。固定分流和动态臂、突发组未启动，不存在动态路由GPU开销测量；S-only到达状态检查开销约99.435μs mean/181.441μs P95仅是基础框架诊断，不能冒充动态决策开销。共置路由变化的干扰对比也为UNSUPPORTED。

Stage D未执行：不存在完整合法稳定的正探索候选，独立ΔU与置信区间均不存在。Stage E未执行，不进入正式优化器集成分析。没有更换种子、降低负载、修改阈值或扩大网格。GPU探索预算使用1/6次、正式窗口600s；其余5次及独立9次未用。prime、30s warmup、冷却、排空独立记录。

五个问题：

1. **是否发现真实动态正收益？** 未发现可验证证据；动态GPU臂未执行，不能断言动态分流无收益。
2. **收益时S/G什么状态？** 尚无实际收益状态。历史机会是S明显排队、G没有未完成工作且native handler闲；这是补测依据，不是正收益证明。
3. **是否优于S-only和最佳固定分流？** 无法判断：首个S-only不满足原稳定性硬约束，完整对照未执行，ΔU=NULL。
4. **独立GPU是否复现？** 未执行，无独立正收益证据。
5. **是否值得集成正式优化器？** 当前证据不足以支持集成；正式联合部署优化器未修改。

输出：opportunity_audit.md、frozen_candidate_manifest.json、dynamic_router.py、routing_decision_trace.csv（六任务全部7607请求）、gpu_exploration_runs.csv、utility_comparison.csv、worker_interference_analysis.csv、module28_gate.json及本摘要。缺失比较用NOT_EXECUTED/NULL明确标记，不生成伪测量。原控制器曾把“没有合格候选”统一标为DYNAMIC_NO_UTILITY_GAIN；初稿在initial_controller_report/原样保留，analysis_acceptance_amendment.json仅澄清科学标签，不改变任何实验、参数或验收规则。

历史16350文件metadata和已读科学来源SHA256保持不变。GPU已释放，完成后停止。
