# Matching producer CPU02：覆盖新生成计划的变更监视

CPU01 的 execute guard 只监视 manifest、sources 和 claim。sources 包含原
supplement seed plans，却未包含新 producer 的15份 plan.json，因此这些
新计划在运行期间“改写后恢复”可以躲过最终SHA检查。CPU01不可启动。

修改前的四份producer源码与原作者报告、manifest收据已保存于
`cpu01-author/source-snapshot-before-guard-fix/receipt.json`；CPU01工件不改。

CPU02先读取receipt检查过的manifest snapshot，再建立外层
SourceChangeGuard，包含manifest、sources和全部15份attempt计划。第一次
完整audit和plan读取发生在该guard内，并复核snapshot、plan和收据一致。
因此第一次audit期间、当前attempt及尚未执行的未来计划的改写恢复均拒绝。
claim仍在审计后以open('x')持久化，再单独建立claim guard；外层all-plan
guard持续覆盖supervisor的before_launch、after_run和返回后检查。

CPU测试只使用stub supervisor，不获取GPU lease、不启动child。
CPU02仍需要根对话独立审查；这些变更不认证native画像或预测效果。
