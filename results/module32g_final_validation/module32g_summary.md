# Module32G最终独立效用验证

Gate：`VALIDATION_INVALID`。StageA CPU工程回归触发原控制检查门槛，按预先声明的技术失败STOP_NO_RETRY停止。正式GPU运行0/6，独立配对0/3，U、三个ΔU、均值和CI95均NULL。**本轮没有验证Dynamic的效用优势，也没有证据将它判为效用负收益。**

## 实际通过与失败

原32F Gate仍为GPU_PREFLIGHT_RECOVERED；本轮不修改该历史Gate。原六任务池、权重、质量准入、QPS、34567.5534MiB预算、safe factor和单副本未改。原32B路由源码/阈值/背压及32F增量runtime均byte-identical；CUDA隔离和Triton cache provider只做Module32G目录/包名移植，没有算子、precision或generation修改。

本轮1381次历史同状态决策逐字段等价，容差0。历史写入保护、phase transition与CPU local Phi cache解析通过。1x历史的完整2400秒规模/50x加速真实并发CPU回放产生1070个ASR路由计时、3419个control样本，路由P95=0.191203ms、P99=0.263617ms、max=30.575588ms；control max=0.033333ms。全部长尾保留，不以max路由替代原P95≤1ms验收。

随后8x历史段在m32g.monitor.IncrementalMonitor.check触发CONTROL_CHECK_OVER_10MS：原perf_counter墙钟15.338264ms>10ms。原检查包含state_lock等待和原异常谓词；未放宽门槛、删样本或重跑。CPU_validation_failure.json与cpu_validation.log保存完整异常栈，CPU_CONCURRENT_8.jsonl保存已写出的部分cohort。该段在失败前尚未写出raw_control_probes_8.csv及最终CPU_runtime_acceptance.json；thread_cpu/state_wait/state_hold等该次字段无法恢复，不能声称完整逐探针归因，也不能仅凭15.338ms断定GIL、锁或操作系统调度哪一项负责。此为加速CPU压力测试，不是GPU控制检查失败，不是CUDA304复发，更不是2400秒墙钟GPU验证。

## 预定配对的处理

已在生成前声明新种子320701/320702/320703和顺序P1_S→P1_D→P2_D→P2_S→P3_S→P3_D，没有换种子或机会筛选。StageA失败后未生成正式新轨迹、未调用正式freeze、未启动任何worker GPU测量。planned_validation_manifest.json仅保存计划与未执行原因，stage_A_source_snapshot.json仅是审计源码/环境快照，均不伪称正式GPU配置验收成功。未执行脚手架源代码保留但不会自动启动。

|组|S-only U|Dynamic U|ΔU|状态|
|---|---|---|---|---|
|1|NULL|NULL|NULL|NOT_EXECUTED|
|2|NULL|NULL|NULL|NOT_EXECUTED|
|3|NULL|NULL|NULL|NOT_EXECUTED|

没有新真实ASR分流比例、Whisper等待改善、VQA保护效果、Text偶然尾延迟或额外执行/共置成本的三组证据。去除Text P95单项的诊断敏感度也为NULL，不把32B探索性ΔU或32C无效配对替代独立结果。Module20A原六任务效用和归一化参考保持，未计算合法U；所有原始CPU数据及失败仍保留。

历史SHA/size/mtime/symlink前后核验=True，冻结32F源码/输入及依赖核验=True。历史32D旧progress不可逐字节恢复例外仅继续记录，不伪称恢复。GPU最终空闲1MiB、无计算进程，见nvidia_smi_after_stage_A.txt。没有修改正式优化器或任何历史Gate。

不进入优化器集成：路由真实GPU优势及优化器自主启用均未由本轮验证。本Module在工程失败Gate处停止，不开发新策略、不进行故障修复后重试或自动启动下一阶段。
