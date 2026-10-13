# Module 17：Full-Pool Runtime Capacity Diagnosis & Minimal Calibration

Gate：**CAPACITY_CALIBRATION_INSUFFICIENT**。定位了实际瓶颈，局部实测服务包络通过独立holdout检验，但没有形成足以支持新完整模型池/新语音QPS/新G/S分流的可迁移容量约束。按停止条件结束，没有启动新的三阶段GPU验证。

## Stage A：CPU-only历史重建

先校验Module16与全部历史hash，读取三个MID原始request、worker、queue、CPU/GPU日志；在Stage A落盘前没有新GPU。每个worker的checkpoint/count/task mapping、预测与实测QPS/service/rho、排队时间、E2E、资源遥测在mid_worker_diagnosis.csv；逐path与六秒队列bin分别在mid_task_path_diagnosis.csv与mid_worker_queue_timeline.csv。module17_mid_worker_diagnosis.csv为同内容兼容文件。

三个MID repetition均由**whisper_turbo#0首先持续积压**，前30秒即形成满足原slope/growth/increasing/end阈值的rolling局部证据。该first-witness分析不是新的stability gate。medium同checkpoint支持ASR/LID，但MID实际仅执行LID；turbo同时执行ASR与LID，单PID单handler串行、没有两个task并行执行。

|MID rep|首个worker|turbo预测rho|turbo服务需求/预测|turbo窗口末积压|medium末积压|turbo后半段slope req/s|
|---|---|---:|---:|---:|---:|---:|
|1|turbo|0.507934|2.071489|26|1|-0.002176|
|2|turbo|0.507934|2.063760|27|1|-0.000914|
|3|turbo|0.507934|2.064739|61|1|0.494303|

原MID3总62积压中turbo61、medium1。前两次turbo形成积压后较接近平台，第三次后半段继续增长；不能仅用末尾积压断定具体干扰源。各worker所有offered、host enqueue/send-attempt、最终handler start、completed计数一致，未发现重复、丢失、dispatch任务错配、重叠handler、错误或未清理worker。worker receipt ACK/timestamp原协议不存在，因此host send-start不冒充精确acceptance时刻。所有启动/prime在测量前，warmup有bounded drain，未发现持续启动拖尾。原日志能检查shutdown/任务/计数异常，但不能完全排除未记录的CPU或CUDA内部机制。

服务时间只用worker start→end（包含native frontend、model和同步），queue_wait、sender queue、write block与arrival→result E2E严格分列。串行慢worker造成pipe/sender排队是可观测背压，未证明为queue实现bug。历史standalone turbo ASR闭环capacity约5.233–5.333 QPS、handler mean186.65–189.85ms；medium LID约14.567–14.733 QPS、handler mean67.33–68.18ms。相比并发明显变慢，不能把失败实验completed吞吐直接当精确极限。

## Stage B：冻结最小诊断

仅2配置×2rep，seeds17101/17202，10s warmup+120s测量。配置A保持原MID六任务；配置B相同六模型驻留且ASR=1、LID=12、各speech path routing不变，只把四个非语音task offered traffic置0。对应ASR/LID实际native sample序列、分配worker及相对scheduled offsets全部核验一致，fresh PIDs。只是诊断，不是正式三阶段验证；诊断unstable保留为结果，未改变/重试条件。

|condition/rep|stability|窗口末积压|safe after MiB|错误|
|---|---|---:|---:|---|
|FULL_MID_REFERENCE/1|unstable|61|24358.805531|[]|
|SPEECH_ONLY_SAME_RESIDENCY/1|recoverable_congested|9|23932.237612|[]|
|FULL_MID_REFERENCE/2|unstable|65|24374.357486|[]|
|SPEECH_ONLY_SAME_RESIDENCY/2|recoverable_congested|9|24367.692362|[]|

关闭非语音任务有改善但未消除turbo接近持续忙碌和队列拥塞；其它活跃模型不是单一充分解释。两个语音worker runnable CPU等待很小，arrival/enqueue和result collection延迟也很小，没有支持CPU调度饥饿/负载发生器错配的证据。CPU run时间包含CUDA同步时可能的spin，不等于纯frontend时间。GPU/CPU活动一起隔离，不能声称已区分纯GPU竞争与CPU/frontend执行；两种条件都保留medium/turbo及ASR/LID共享，也不能独立分解同模型共享与跨语音模型竞争。不重开9A，也不新增第3配置。逐worker flags中的INSUFFICIENT_TELEMETRY机制说明与diagnostic_comparison.json保留这些限制。

## Stage C/D/E：校准范围与停止

ρ_pred=D_pred/g(x)使用历史standalone均值与相对副本gain；g(1)=1并不证明standalone服务时间在完整并发下保持有效。仅对语音压力路径提出training-only均值包络，max(旧三rep均值最大值, 单侧95% t上界)，未依据新诊断结果抬高边界：

|path|s0 ms|冻结均值包络 ms|单path等价gain|
|---|---:|---:|---:|
|openai/whisper-medium@abdf7c39ab9d0397620ccaea8974cc764cd0953e/lid|69.333216|124.837318|0.555389|
|whisper_turbo/asr|170.522292|397.930581|0.428523|
|whisper_turbo/lid|83.178565|165.059922|0.503930|

独立holdout 12/12个path均值检验通过；局部包络支持=True。这不是逐请求deterministic上界，也不是精确容量边界。冻结原MID路由对应包络语音demand={'openai/whisper-medium@abdf7c39ab9d0397620ccaea8974cc764cd0953e': 0.991648086421598, 'whisper_turbo': 1.0674923867947355}，已没有ρ≤0.8所要求的余量。

低开销实现可以是单独scoped线性service-demand guard，保留原s0/g/M、J、quality和V3；但当前只支持已测试的驻留/count与固定语音负载/路由，非语音背景为原值或0。新完整池最优部署涉及其它模型、count、ASR路径或G/S分流，不允许把未知条件默认当已校准，也不允许统一给所有模型乘折减系数。

所以fullpool_reoptimization.json明确记录未启动正式重优化，新的可认证最优部署和完整理论witness均为空；这不证明所有合法组合数学上不可能。没有new_three_phase_manifest.json，没有正式gpu_validation.csv，也没有新的9/9结果。四个诊断结果独立保存在diagnostic_gpu_runs.csv，不能冒充三阶段验证。

## 最终八项回答

1. 三个MID都先由whisper_turbo#0持续积压；MID3末尾61/62归turbo。
2. 确认串行worker的实测服务需求升高并触发排队，未发现可修复的runtime/dispatch/queue bug；唯一底层干扰来源仍证据不足。
3. turbo加权服务需求约为预测2.06倍；medium LID约为standalone1.72–1.77倍。不是把E2E算进service，也不是宣称精确最大容量下降同等倍数。
4. 局部guard可以低开销实现，四次holdout支持当前局部包络；尚不能用于新完整池三阶段，因此没有部署未经支持的修正。
5. 原MID胜出路由失去可信ρ余量；修正后的全池最优部署尚未认证，历史V3独立保留。
6. 没有形成新的完整理论三阶段，未靠改QPS或模型制造witness。
7. 没有运行正式三阶段GPU，不能声称新9/9 steady；4次仅为诊断。
8. 对固定驻留及固定语音流有独立背景隔离验证的局部包络，尚未形成跨新语音负载/分流/部署可迁移的容量约束；按停止条件结束。

全部历史与冻结sources/catalog/samples/hash核验，GPU释放。完成后停止，不扩大实验。

额外核验：mid_waiting_and_execution_timeline.csv把send-attempt→handler-start等待、handler执行、host-unsent sender等待分别重建；排除正在执行的1个请求后，三个MID仍首先由turbo在前30秒形成waiting增长证据。所有native service=(end−start)与queue=(start−submit)方程核验通过。没有receipt ACK，不把virtual waiting计数称为精确worker内部已接收队列。
