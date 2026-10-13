# 输入盘点（分析前）

Module28递归文件清单、路径、大小与SHA-256已先冻结于input_freeze.json。权重/缓存不在审计范围；请求正文、prediction、reference、sample内容不输出。完整raw只在服务器本地解析。

|所需信息|实际文件与字段|证据限制|
|---|---|---|
|正式请求ID、任务、路径|raw_runs/C_steady_s_only.json → phases.measurement.request_records[].request_id/task/model|仅正式窗口cohort；warmup与path_probe_records分开|
|计划到达|同上 scheduled_time_ns/frozen_offset_seconds|perf_counter_ns；计划到达不同于实际host入队|
|host入队、发送开始/写入完成|enqueued_time_ns/actual_submit_time_ns/submit_write_completed_time_ns|不是worker接收ACK；入队未完成数不等于等待队列|
|成功响应|status=completed及completion_time_ns|stdout reader捕获时间；collector更新status的时刻未独立记录|
|worker执行开始/结束|worker_response.start_ns/end_ns|Unix ns，只可回顾；从RESULT返回，执行期间不在线可见|
|时间基准锚点|measurement.window.start_time_ns/start_unix_ns/end_time_ns/drain_end_time_ns；actual_submit_unix_ns|host单调时钟字段减start_time_ns；Unix handler时间通过每请求submit的配对锚点映射，误差单报|
|S/G当时快照与代价|LID记录中的routing_decision.S_pending/G_pending/G_target_inflight及S_wait_est_ms等|7005个LID有实际快照；其他任务没有此快照|
|准入、容量|routing_decision.admitted/capacity_ok；worker_identities.READY；冻结catalog|实际布尔值在LID记录中；READY capability不自动等于质量准入，新路由压力容量未知|
|冻结P10/P95与惩罚|router_parameters.json、frozen_candidate_manifest.json.parameters|由GPU前Module27开发数据产生；不是对Module28未来数据重新拟合|
|开发服务分布|manifest.development_sources指向Module27五个raw请求worker_latency_ms|仅提取服务标量并核验来源哈希；请求正文不输出|
|EWMA|dynamic_router.py.observe_completion；gpu.py.arrival_snapshot|没有逐快照持久化service字典；可按已收到响应重建或界定，但collector/harvest竞态不能默认不存在|
|滞回、令牌、冷却、回退|dynamic_router.py.__init__/observe_completion/decide|S-only只调用estimate，没有调用decide；正式dynamic状态未实际运行和完整记录|
|增量journal|request_journals/C_steady_s_only.jsonl|每10s终态/检查点，非每次collector状态更新；可交叉核验事件，不可补造缺失状态|
|原Gate、合法U、原分类器|module28_gate.json、all_gpu_runs.csv、raw measurement.stability；原stability_v2.py|保持DYNAMIC_EVIDENCE_GAP、unstable、U=NULL，不做历史修订|
|完整路由CSV|routing_decision_trace.csv与target_routing_decision_trace.csv|衍生表，用raw校验；不能当作动态decide日志|

实际公式：C_S=sum(S按入队顺序除首项的任务P10)+目标P10；C_G=sum(G pending任务P95)+max(冻结G目标P95,已收到G目标服务EWMA)+冻结S膨胀系数×S_pending+冻结背景惩罚。entry/exit使用严格大于与严格小于判据；守卫按原代码优先级核对，禁止替换为固定13阈值。

在线快照在LID的decision_time_ns，而不是精确scheduled_time_ns；将分别分析计划到达和实际决策时点。600秒以[0,600)划界，排空[600,602.444690107]单列，warmup不并入正式cohort。
