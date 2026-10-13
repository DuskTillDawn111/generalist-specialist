# Module 18：Cross-Task Three-Phase Deployment Validation

Gate：**CROSS_TASK_THREE_PHASE_REAL_VALIDATED**（冻结六任务 system-cohort 判据；LID HIGH有局部worker拥塞）。完成后停止，不扩大实验。

完整19模型池（Qwen/Phi/MiniCPM与全部16 Heavy Specialists）使用14F实测 memory/g、原始质量准入、rho≤0.8，资源优先再按相同J优化。完整池三项LOW的资源最优解都驻留目标Specialist，所以没有三阶段；完整池与受控池CPU结果独立保存在cpu_pool_comparison.csv与cpu/*.json。不得把受控成功称为完整池成功。

预定义受控COMMON_G池要求同一个Generalist具有Text/Image/LID所有比较任务及ASR/VQA背景的原始准入/真实路径；Qwen满足，Phi缺Text/Image/LID，MiniCPM缺Text/LID。排除依据于CPU结果前声明，全部16 Specialists与所有实测副本计数保留。此为共同跨任务能力控制，不能证明包含较小、窄能力Generalist时的全池资源最优相变。

预算B2=27980.1894 MiB。非目标背景Text/Image/OCR/ASR/LID各0.1 QPS，VQA=0.2 QPS，只覆盖目标QPS，全部六任务真实请求。Text=0.1/20/60，LID=0.1/6/10按原候选；Image原HIGH40在CPU仍S-only，按GPU前冻结容量公式CS+0.1CG得到137.2973242711638，LOW0.1/MID10不变。无GPU结果反馈移动点。Image预测Specialist达到操作rho=0.8，仍有物理20%余量且联合操作容量剩余4.4302 QPS；此为较靠近操作边界的独立验证点，无并发可迁移容量保证。

全部点最小资源竞争/同成本J证书/原V3独立CPU基线见frozen_task_manifests.json。Qwen必须承接VQA/ASR等背景，PARSeq承接OCR。Text MID/HIGH新增XLNet，Image新增ConvNeXt-tiny，LID新增Whisper-medium；都是优化器选择。ASR背景只走Qwen，未复用Module16 ASR1/LID12失败负载，也未通用化Module17局部2.06x。

顺序Text→Image→LID，LOW→MID→HIGH，每点最多3个独立进程重复；10s warmup+120s测量，seeds18101/18202/18303，同原native serving/stability_v2。unstable或运行/显存失败立即停止当前任务；完成阶段存在recoverable_congested同样不算通过并跳过下一阶段。各任务独立，失败不改点或重试。

|任务|池|LOW steady|MID steady|HIGH steady|结果|
|---|---|---|---|---|---|
|text|CONTROLLED_COMMON_G|3/3|3/3|3/3|THREE_PHASE_REAL_VALIDATED|
|image|CONTROLLED_COMMON_G|3/3|3/3|3/3|THREE_PHASE_REAL_VALIDATED|
|lid|CONTROLLED_COMMON_G|3/3|3/3|3/3|THREE_PHASE_REAL_VALIDATED|

|run|状态|目标G/S完成数|目标G实际份额|末尾积压|safe预算|通过|
|---|---|---|---:|---:|---|---|
|text_low_rep1|steady|12/0|1.0|0|True|True|
|text_low_rep2|steady|12/0|1.0|0|True|True|
|text_low_rep3|steady|12/0|1.0|0|True|True|
|text_mid_rep1|steady|0/2400|0.0|0|True|True|
|text_mid_rep2|steady|0/2400|0.0|0|True|True|
|text_mid_rep3|steady|0/2400|0.0|1|True|True|
|text_high_rep1|steady|247/6953|0.034305555555555554|1|True|True|
|text_high_rep2|steady|247/6953|0.034305555555555554|0|True|True|
|text_high_rep3|steady|247/6953|0.034305555555555554|1|True|True|
|image_low_rep1|steady|12/0|1.0|0|True|True|
|image_low_rep2|steady|12/0|1.0|0|True|True|
|image_low_rep3|steady|12/0|1.0|0|True|True|
|image_mid_rep1|steady|0/1200|0.0|0|True|True|
|image_mid_rep2|steady|0/1200|0.0|0|True|True|
|image_mid_rep3|steady|0/1200|0.0|0|True|True|
|image_high_rep1|steady|59/16417|0.0035809662539451324|1|True|True|
|image_high_rep2|steady|59/16417|0.0035809662539451324|1|True|True|
|image_high_rep3|steady|59/16417|0.0035809662539451324|1|True|True|
|lid_low_rep1|steady|12/0|1.0|0|True|True|
|lid_low_rep2|steady|12/0|1.0|0|True|True|
|lid_low_rep3|steady|12/0|1.0|0|True|True|
|lid_mid_rep1|steady|29/691|0.04027777777777778|0|True|True|
|lid_mid_rep2|steady|29/691|0.04027777777777778|0|True|True|
|lid_mid_rep3|steady|29/691|0.04027777777777778|0|True|True|
|lid_high_rep1|steady|227/973|0.18916666666666668|1|True|True|
|lid_high_rep2|steady|227/973|0.18916666666666668|0|True|True|
|lid_high_rep3|steady|227/973|0.18916666666666668|1|True|True|

worker_task_service.csv严格分开worker执行service、排队与arrival-to-result E2E；原始requests保留offered/submitted/completed对账，submitted为host send-start，未伪造worker ACK。per_model_metrics.csv单列handler busy wall fraction、真实throughput、host CPU、process GPU SM（无样本留空）和设备利用率；busy wall不等于GPU SM。worker_queue_evidence.json逐worker时间线和原stability判定；raw保存READY/checkpoint、PID、完整请求/遥测/排队/退出与释放。

Module15 ASR为历史9/9 steady的两任务实例；Module18是六任务低背景与COMMON_G受控池，范围不同。论文主实验集合：asr, text, image, lid，每个结果必须注明池与任务范围。新增模型=0；本轮检验无需增加模型，不进入新模型搜索、动态部署或第二Generalist实验。

所有历史8307个受保护文件以及冻结代码/表/样本hash完成核验，所有GPU worker PID独立与释放通过。仅声称冻结离散120秒点及重复观察，不声称尚未定义的E2E SLO、连续稳定边界或新的可迁移干扰模型。

语音/队列复核：27/27六任务合并cohort steady，72个独立worker-cohort中69 steady、3 recoverable_congested、0 unstable。唯一局部拥塞为LID HIGH三次的Qwen；分别有51, 23, 42个请求排队超过1秒，worker警告门槛为15请求。六任务系统门槛为64请求，因此system steady与worker recoverable两个标签可以同时成立，不得只报前者或改变门槛。局部队列均最终恢复，无持续积压。详细结果见worker_stability_summary.csv及speech_colocation_review.json。

LID HIGH Qwen handler busy wall fraction=0.8111–0.8894，LID执行service/s0=2.193–2.608；Medium LID执行均值=84.72–87.46ms。CPU rho≤0.8证书来自冻结standalone s0/g；此处Qwen实测busy>0.8，不能声称真实保留20%worker余量，也不能据这些serving点建立可迁移g或通用折减系数。实际throughput不是极限capacity。

三阶段触发机制：Text HIGH60与LID HIGH10仍低于各自Specialist的冻结操作容量，分流是同资源成本下J优化的结果；Image HIGH137.2973高于Tiny操作容量136.8051，CPU约束要求分流。GPU只验证冻结分流方案，没有S-only反事实对照，未证明实际单S一定不稳定。

论文使用边界：按预冻结系统判据，ASR/Text/Image/LID可组成注明各自任务/候选池范围的行为集合；LID必须标注上述局部拥塞及真实余量未证实。如果论文主张每个worker均无拥塞，主集合应限于历史ASR参考与Text/Image，LID只作边界案例。完整池三项均无本轮理论witness，成功不能外推到完整Generalist池。

本轮最大observed safe memory=25694.05 MiB；最大sampled owned peak=23130.00 MiB。全部offered=submitted=completed，errors=0；GPU已释放。完成后停止。
