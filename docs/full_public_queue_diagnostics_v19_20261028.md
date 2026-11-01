# v19：完整公开到达下的队列解释 sidecar

新入口是 `tools.predictor_validation_20261027.learned_availability_forecast_v19.LearnedAvailabilityForecast`。它继承 v16，接受原来的配置与显式未验证假设开关。在 v16 engine 的 `forecast(requests, now, deadline, drain)` 入口，使用实际完整 rows 和原 now 创建队列诊断；不重新构建模型或替换 compiled graph/work/release/phase。六项评分仍由原父类路径执行，scheduler、routes、service和稳定性规则没有新增惩罚或限制。

`FullPublicDagQueueDiagnostics` 复用 v6 的 FIFO等待、oldest age、剩余reference work及恢复episode。构造时只缓存完整公开输入的 profile/cost；**未来尚未到达的请求不 register，不增加当前 waiting/remaining work**。因此早cutoff时，只要冻结profile支持未来输入，100ms/1s/10s完整offered work envelope仍可已知；不再因为未来尚未register而变成None。未知template保留其请求与到达计数，work envelope为None并给出原因，不当成零也不删除。

SM/DRAM/L2没有归一化demand模型时保持None。reference work是原孤立GPU/copy/memset时长之和，不是实际busy time、资源利用率、服务能力或新的排队稳定性证明。未到达输入的已知reference work可以用于offered envelope，但未来service结果不可使用。

旧v8在drain边界会调用 `observe(delta=0, active仍非空, rates={})`。v19对子类的这一零时长调用只记录状态，不消费work，也不填补service rate。真实正时长interval仍严格使用engine的实际rates。遇到stall或未知输入，原engine的UNKNOWN原样传播；sidecar保留此前队列解释，不把不完整执行变成数值成功。

正常结果的 `forecast.queue_diagnostics` 包含增强解释；顶层 `queue_diagnostic_sidecar` 还保留完整cohort/now/deadline/drain、原UNKNOWN原因、原compiled身份与本地Python静态import闭包的source receipts。原result schema及CPU representation version保留，单独标记instrumentation version=19。侧录仅用于解释，不参与原六项U、rho、分流或执行决策。

CPU检查使用明确的synthetic小DAG与冻结编译表，比较开启/关闭时除diagnostics字段外整份原forecast逐字段相同；包括非零now、drain受限、同owner并行节点、晚到达、FIFO恢复、stall和未知template的原UNKNOWN。另直接调用原 `score_complete_forecast`，确认六项与full_U逐字段相同，部分drain仍被原评分器拒绝。测试也验证未来profile可用与当前backlog未增加两者同时成立。

本轮没有进行约4分钟真实训练provider初始化、没有运行新的完整57请求preflight、没有GPU、独立队列准确率或收益验证。实际软件接线经过小DAG入口测试；真实主预测器的物理模型修正及独立验证仍按原顺序继续。
