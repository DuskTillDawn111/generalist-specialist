# 独立 reader-visible 队列风险验证流程

v38 复用冻结 v36 从 v23 五个函数派生的执行体。原 arrival 注册、FIFO dispatch、自然共享、四驻留 owner、12 priming、57 measured、30 秒 arrival window、120 秒 drain、监视、磁盘预算、独占锁、watchdog 与清理不变。证据分为 risk_validation，training_only=False，使用新 nonce 和全部新请求 ID；不会进入训练 consumer。

唯一新增执行钩子在原 DeploymentRuntime.start 返回后、第一条 priming enqueue 之前。先核验 READY PID 属于各 worker，再写只含 READY 和原 context SHA 的文件。比较完整 READY payload，只有 PID 可变化；不读 service、queue 或结果。差异立即停，保留 UNKNOWN，不开始 priming。

目标扩展在任何 GPU 执行之前预声明。所有原 context SHA 保留，完整 context diff 保存；只有精确声明的证据 schema、training 标签、nonce 和 v36/v38 包装源追加经过显式等价映射。真实 execution/profiling 差异不会被删除。v36 原/派生 AST body 和 v38 startup 钩子 body 精确封存；99 文件完整 hook 静态源闭包也固定。

READY 通过后，仅更改模型 risk_domain_sha256 与 model_sha256。base/trees/support、训练类计数、原训练 run/row/context provenance、全部其他字段逐项完全相同。每次新执行有独立扩展域，只允许已经 READY 资格化的目标原 context。

v39 使用诚实的独立验证接口，不伪造 v30/v32 training report。严格顺序：

1. 审核完整新执行，独立导出 cutoff 可见 host prefix features，不导出标签。
2. 冻结评分只打开 features、READY 资格、三个 rebound 模型与域；CPU 测试禁止打开 raw/plan 证明边界。输出 raw 三 horizon 与固定等权 L2 PAVA joint，任何支持外均 UNKNOWN。
3. 封存预测报告及其 SHA 后才单独导出 outcomes。
4. 单独 evaluator 核验每个 score 等于冻结模型和对应 prefix 的重放值、整 population/grid、原 context/cutoff；同时报告 raw 与固定投影、coverage、Brier、AUC、reliability bins、同支持集训练类率 baseline。既不选择赢家，也不重拟合。

这属于 fresh execution / same recipe and content 的独立风险验证，预测方式为 causal receipt replay。它不声称 online rolling forecast、完整 GPU DAG/checkpoint、物理 kernel 服务、全过程 P95 或 ΔU 排序已合格，更不构成分流收益。

源冻结于 `artifacts/predictor-independent-risk-validation-v39-20261028-frozen01/report.json`，104 文件精确快照；准备对象为 `artifacts/predictor-ready-risk-holdout-v38-20261028-prepared01/campaign.json`。66 项 CPU 检查通过。新执行完成后，所有标签与预测都保留原样，结果不能用于调 threshold 或挑 raw/PAVA winner。
