# Module 28R：Dynamic G/S Routing Follow-up

状态：已完成并停止。容量 Gate：**DYNAMIC_CAPACITY_IMPROVEMENT**；效用 Gate：**DYNAMIC_EXPLORATORY_POSITIVE_UTILITY**。

历史Module29及以后不作为证据；原Module28 Gate、20A定义、模型池、B4预算、准入、副本、Fixed10和动态阈值均保持不变。HIGH LID11.5、LOW10.5；其余五任务各0.2QPS。30s warmup、原冷却门槛、600s measurement、60s有界排空；两个场景交错运行，不补种子、不增加GPU运行。

|场景|方案|系统状态|合法U|G / S目标数|窗口完成QPS|末端积压|
|---|---|---|---:|---:|---:|---:|
|HIGH|s_only|unstable|None|0 / 7005|12.636666666666667|25|
|LOW|dynamic|steady|-3.981932590440189|12 / 6332|11.566666666666666|6|
|HIGH|fixed10|unstable|None|720 / 6285|12.633333333333333|27|
|LOW|s_only|steady|-4.094784034228045|0 / 6344|11.565|7|
|HIGH|dynamic|recoverable_congested|-7.038958724998053|69 / 6936|12.643333333333333|21|
|LOW|fixed10|steady|-4.522586955503298|647 / 5697|11.563333333333333|8|

所有U均按完整cohort及原硬约束；不稳定延迟仅INELIGIBLE诊断，不用于效用排名或成功比较。有效任务数与质量子集见per_task_latency.csv；pooled mean/P95见gpu_routing_comparison.csv，不与加权归一化P混同。worker单独稳定性见stability_comparison.csv。NVML峰值仅0.5s采样高水位，未声称瞬时硬件峰值；worker allocator MEMORY_END输出在gpu_campaign.log，安全显存约束按原驻留系数及采样峰值检验。

## 六个结论

1. 原Module28 S-only有真实末150s队列增长与32请求积压，原分类器unstable、U=NULL；全窗口S需求约0.897、570s恢复至0、2.445s排空。**长期持续容量不足证据不足**，局部波动与边界效应无法排除。
2. 同轨迹新HIGH的容量结果：DYNAMIC_CAPACITY_IMPROVEMENT；以新S-only和Dynamic的系统及所有worker原分类器为准。若新S-only也稳定，只能报告本窗口稳定，不能声称恢复了已测容量缺口，更不能追溯修改原Gate。
3. LOW三个方案均合法稳定：True；recoverable_congested仅在原最终恢复规则通过时合法。
4. LOW真实系统正效用候选：True；两项预注册ΔU=[('s_only', 0.11285144378785583), ('fixed10', 0.540654365063109)]。必须两项都>+0.01、G/S共同承担，三个方案完整合法后才接受。
5. S同输入服务和等待、G慢路径及背景任务变化逐项见worker_interference_analysis.csv、worker_metrics.csv和utility_decomposition.csv。服务膨胀是共置运行关联证据，不能唯一断言物理GPU、CPU或热态机制。
6. 值得独立复现的正收益区间：仅LOW单点探索候选，已冻结，建议独立长窗口；本轮不启动。G-only未测，同副本历史服务需求筛选约HIGH4.41、LOW4.03 handler-s/s且还含VQA；历史筛选不是本轮实测稳定性，不宣称优于所有合法方案。

## 限制与停止

每负载一个匹配轨迹、每方案单窗口，不能提供独立复现置信区间。历史元数据与来源SHA校验在historical_integrity.json；正式优化器未改。所有异常、warning和原始完整请求保留，未执行项目留NULL。

## 数值归因补充

HIGH fixed10 vs S-only：合法ΔU=NULL；以下服务/等待仅完整观测诊断。

HIGH fixed10目标S路径：份额89.722%，同输入服务Δ=+6.199ms、等待Δ=+25.507ms、E2EΔ=+31.837ms，对目标整体mean差贡献=+28.565ms。

HIGH fixed10目标G路径：份额10.278%，同输入服务Δ=+287.903ms、等待Δ=-223.902ms、E2EΔ=+64.164ms，对目标整体mean差贡献=+6.595ms。

HIGH fixed10背景asr：n=124，同输入服务Δ=+11.001ms、等待Δ=+27.752ms。

HIGH fixed10背景visual_qa：n=105，同输入服务Δ=-27.538ms、等待Δ=+167.512ms。

HIGH fixed10背景text：n=113，同输入服务Δ=+0.429ms、等待Δ=-0.005ms。

HIGH fixed10背景image：n=122，同输入服务Δ=-0.646ms、等待Δ=+0.015ms。

HIGH fixed10背景ocr：n=138，同输入服务Δ=-0.034ms、等待Δ=-0.036ms。

HIGH dynamic vs S-only：合法ΔU=NULL；以下服务/等待仅完整观测诊断。

HIGH dynamic目标S路径：份额99.015%，同输入服务Δ=+0.036ms、等待Δ=-30.190ms、E2EΔ=-30.140ms，对目标整体mean差贡献=-29.843ms。

HIGH dynamic目标G路径：份额0.985%，同输入服务Δ=+320.484ms、等待Δ=-1169.547ms、E2EΔ=-848.987ms，对目标整体mean差贡献=-8.363ms。

HIGH dynamic背景asr：n=124，同输入服务Δ=-13.290ms、等待Δ=-21.814ms。

HIGH dynamic背景visual_qa：n=105，同输入服务Δ=-13.549ms、等待Δ=+12.556ms。

HIGH dynamic背景text：n=113，同输入服务Δ=+0.704ms、等待Δ=-0.017ms。

HIGH dynamic背景image：n=122，同输入服务Δ=-0.504ms、等待Δ=+0.001ms。

HIGH dynamic背景ocr：n=138，同输入服务Δ=-0.906ms、等待Δ=-0.042ms。

LOW fixed10 vs S-only：ΔU=-0.427803，分项T/L/P/M={'T': 0.0, 'L': -0.11905632450232972, 'P': -0.30867911171608275, 'M': -6.74850568404689e-05}。

LOW fixed10目标S路径：份额89.801%，同输入服务Δ=+5.073ms、等待Δ=+6.510ms、E2EΔ=+11.667ms，对目标整体mean差贡献=+10.478ms。

LOW fixed10目标G路径：份额10.199%，同输入服务Δ=+272.038ms、等待Δ=-63.841ms、E2EΔ=+208.298ms，对目标整体mean差贡献=+21.243ms。

LOW fixed10背景asr：n=124，同输入服务Δ=+24.368ms、等待Δ=+7.855ms。

LOW fixed10背景visual_qa：n=105，同输入服务Δ=-5.612ms、等待Δ=+135.726ms。

LOW fixed10背景text：n=113，同输入服务Δ=-0.297ms、等待Δ=-0.075ms。

LOW fixed10背景image：n=122，同输入服务Δ=+1.568ms、等待Δ=+0.015ms。

LOW fixed10背景ocr：n=138，同输入服务Δ=+0.926ms、等待Δ=-0.006ms。

LOW dynamic vs S-only：ΔU=+0.112851，分项T/L/P/M={'T': 0.0, 'L': 0.034266279964220825, 'P': 0.07860444526844601, 'M': -1.92814448115447e-05}。

LOW dynamic目标S路径：份额99.811%，同输入服务Δ=-0.478ms、等待Δ=-8.085ms、E2EΔ=-8.557ms，对目标整体mean差贡献=-8.541ms。

LOW dynamic目标G路径：份额0.189%，同输入服务Δ=+357.027ms、等待Δ=-967.985ms、E2EΔ=-610.933ms，对目标整体mean差贡献=-1.156ms。

LOW dynamic背景asr：n=124，同输入服务Δ=+1.603ms、等待Δ=-9.489ms。

LOW dynamic背景visual_qa：n=105，同输入服务Δ=+9.778ms、等待Δ=+0.886ms。

LOW dynamic背景text：n=113，同输入服务Δ=-0.266ms、等待Δ=+0.078ms。

LOW dynamic背景image：n=122，同输入服务Δ=+1.699ms、等待Δ=+0.111ms。

LOW dynamic背景ocr：n=138，同输入服务Δ=-0.062ms、等待Δ=+0.040ms。

HIGH Dynamic原判据状态=recoverable_congested、合法=True，但末端积压21、末段斜率+0.087234请求/s、增长8.519396、递增比例0.50。原四项联合拒绝条件中，递增比例0.50未达到0.60，因此通过；不能事后改变规则，也不能将这个通过写成长时域持续容量恢复。

完整校验：16735个历史文件元数据、原始来源SHA和冻结GPU实现均未变化；6次正式GPU探索，未重试。启动前CPU语法失败（模型启动/推理0次）保存在prelaunch_cpu_failure/；另有两次GPU结束后的CPU报告错误（重复status字段、numpy布尔序列化），修复及继续剩余槽位记录在两个report amendment文件，修正时负载、比例、动态阈值与数值公式未变。
