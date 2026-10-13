# Module 32B：VQA 干扰归因

CPU-only；读取Module32全部原始raw/journal/遥测/文本日志（含归档失败），正式逐请求比较仅A0/A1/A2完整匹配600秒cohort。归档重复数据不重复加入估计。未执行反事实路由，未将代数时延替换视作真实收益。

## 时钟与CUDA

E2E=(host perf completion−scheduled)，worker服务=(Unix end−start)，worker等待=(Unix start−host Unix submit)。跨worker区间仅同机Unix口径。Unix/perf偏移散布及native公式逐请求核验见clock_and_cuda_timing_audit.json。Phi原ASR/VQA生成在开始、输入转GPU后、generate后显式cuda synchronize；服务是包含I/O、处理、GPU及同步的handler墙钟时间，不是CUDA event的纯kernel时间。Phi协议同步串行，VQA无法抢占在执行的ASR。

## VQA排队与执行

|臂|VQA数|服务mean/P95 ms|等待mean/P95 ms|E2E mean/P95 ms|等待与Phi ASR执行区间覆盖|VQA被ASR阻塞数|
|---|---:|---|---|---|---:|---:|
|A0|133|283.957/396.402|11.697/36.678|297.011/435.150|0.00%|0|
|A1|133|292.431/373.232|475.000/3509.407|768.956/3735.271|91.16%|16|
|A2|133|232.012/332.081|279.644/1943.949|512.926/2155.081|95.42%|17|

逐请求等待区间与同Phi ASR handler的交集是直接日志证据；同Phi ASR/VQA handler无并行，不能称GPU内两个任务同时执行。VQA未来到达也会被已接纳的ASR挡住，当前空队列不能保证背景安全。

## 完整系统效用损失

|臂|任务|mean贡献ΔU|P95贡献ΔU|合计ΔU|
|---|---|---:|---:|---:|
|A1|text|-0.015177|-0.010432|-0.025609|
|A1|image|-0.000674|+0.001510|+0.000836|
|A1|ocr|-0.017385|-0.030890|-0.048276|
|A1|asr|+0.040204|+0.135119|+0.175323|
|A1|lid|-0.002215|-0.004573|-0.006787|
|A1|visual_qa|-0.121130|-0.847017|-0.968148|
|A2|text|-0.019879|-0.016443|-0.036322|
|A2|image|+0.003279|+0.011132|+0.014412|
|A2|ocr|-0.004626|-0.007802|-0.012428|
|A2|asr|+0.040091|+0.168283|+0.208374|
|A2|lid|-0.002089|-0.001428|-0.003517|
|A2|visual_qa|-0.055417|-0.441442|-0.496859|

A1系统ΔU=-0.872757，T/M贡献变化=-0.000096。VQA P95差=3300.121ms；对配对E2E向量的三项Shapley代数分解：排队3348.916ms，服务-47.759ms，host残差-1.037ms。三项相加等于真实P95差；不得相加各自P95冒充E2E P95。
Phi ASR=17，路由到忙Phi=2，已有VQA在途=2；路由时Phi空闲但ASR期间新VQA到达=7。最高5% VQA尾样本中被ASR阻塞=7，全部尾请求与blocking ID见VQA_tail_requests.csv。

A2系统ΔU=-0.326446，T/M贡献变化=-0.000106。VQA P95差=1719.930ms；对配对E2E向量的三项Shapley代数分解：排队1768.019ms，服务-45.884ms，host残差-2.204ms。三项相加等于真实P95差；不得相加各自P95冒充E2E P95。
Phi ASR=34，路由到忙Phi=6，已有VQA在途=4；路由时Phi空闲但ASR期间新VQA到达=12。最高5% VQA尾样本中被ASR阻塞=7，全部尾请求与blocking ID见VQA_tail_requests.csv。

## 共置执行关联与可观测状态

cross_worker_service_overlap.csv完整保留同输入其他worker的服务变化及与Phi ASR handler区间的交集。交集取决于请求持续时间，并可能受输入难度、热状态、CPU与次序混杂，不能唯一识别GPU资源争用因果。正式G/S中保留S的配对服务均值没有膨胀；对真实重叠子群与非重叠子群分别报告，不能将总体均值直接推广至某时刻。

运行时能够因果获取：到达时已入队/发送尚未收到RESULT的Phi VQA/ASR数、S在途数、已完成请求统计、最新有限频率遥测、当前输入的元数据。原协议没有START ACK，因此“正在执行哪一任务”只能用在途保守近似，后台pending快照能保守禁止在忙Phi加ASR，不能读取事后native start/end。Phi空闲时未来VQA不可知；须以冻结到达率与ASR服务尾部风险保护。

现有日志可直接识别内部串行排队阻塞，但无法唯一归因服务膨胀中的GPU、CPU、同步等待部分。若要kernel级因果，需要独立CUDA events/NSys与受控扰动；本轮不把缺失量补为零。机制分类和是否进入原型由dynamic_opportunity_gate.json另行冻结。

## 辅助worker与到达状态口径

跨worker重叠子群：Fixed5/Fixed10保留Whisper与Phi ASR handler重叠n=34/44，同输入服务比1.0093/1.0552；非重叠组为0.9014/0.9644。其他Specialist的重叠组服务变化有正有负：Fixed10 Text/OCR/LID约+69.5%/+27.8%/+18.4%，Image约−8.6%。样本小、输入时长与重叠互相影响，只有关联，不能判定这些变化均由GPU竞争引起。

原fixed路由在cohort构建时预先按uniform指定，并未使用实时状态。Phi_ASR_arrival_state.csv记录的是host send时点；补充Phi_ASR_state_clock_basis.csv分别用scheduled arrival、实际enqueue与实际send重建“已入队而尚未收到RESULT”的因果可观测pending。动态原型在实际到达dispatch时同一runtime锁下快照并接纳，绝不读取native start/end。跨worker ASR/VQA重叠与未来VQA到达计数仅用于事后机制解释。全部path probe请求另存historical_path_probe_timings.csv，排除于正式U。
