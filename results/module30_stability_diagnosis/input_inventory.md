# 输入盘点与字段证据

所有历史输出也作为输入冻结；完整309个文件的路径、大小和SHA-256见input_integrity_before_after.json。

|来源|字段/用途|缺口|
|---|---|---|
|Module28 raw_runs/C_steady_s_only.json|request_id/task/scheduled/enqueued/actual_submit/completion、worker_response start/end、service/queue、LID routing_decision|无collector commit和完整在线router状态；四个旧快照的reader/collector滞后不能用未来事件回填|
|Module29 stage2 gpu_validation_raw.json|同上，新增collector_commit_time_ns、decide及完整arrival开销|预测文本和reference已脱敏；G LID无valid_closed_set_output字段|
|Module29 stage2 decision_event_journal.jsonl|实际locked observation时点、全部pending、历史completion harvest、完整路由器状态|无worker receipt/START ACK，真实worker队列仍UNKNOWN|
|Module29 stage2 routing_decision_trace.csv|含warmup；本审计筛phase=measurement得到7005条|不得将warmup338条混入目标分母|
|原冻结轨迹|同一正式请求的pool_id/sample_id、任务、计划偏移和LID语言标签|不能因此认定两个时段/插桩不同运行构成合法匹配效用对照|
|原分类器stability_v2.py|20个30s bin、尾5个bin、OLS及全部硬/警告条件|原实现恢复/超时/全观察peak涉及排空，原样复算；排空不进入正式bin/OLS、不解除硬失败|

主时钟为每次formal start的perf_counter_ns；native Unix用每请求paired submit映射，仅用于事后执行分析。所有正式bin为[0,600]内，计数半开区间；边界事件单列核验。host enqueued pending、host submitted B、native未开始handler代理与真实worker等待队列分别命名。
