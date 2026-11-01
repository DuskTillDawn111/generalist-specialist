# H600 selected-v16 独立审阅

本次审阅只在 CPU 上进行。独立验证入口已经冻结；后续不得修改 `h600_selected_validation_v2.py`，也未修改 Gamma、数值修复、旧模型或旧失败报告。

选择审计已补齐：对 Gamma 与 WLS 两份报告分别用原始 LIMITS 重算评分；固定 seed 627601/627602、四条路线、每条 1200 个请求及总计 9600 个请求；UNKNOWN 保留惩罚，不缩小分母。两算法的 heldout plan、native raw、scored raw、observed_U、observed_metrics 必须逐项相同。两份报告和逐案例来源回执均需验证，WLS 必须绑定 numerical-v2，且满足 baseline report < Gamma protocol < Gamma report 的前瞻顺序。完全同分保留 WLS。

针对性 19 项 CPU 测试通过；主线程随后报告整体 23 项 CPU 测试通过。真实 selection 审计 PASS，full16 Gamma 数值重拟合验证 PASS，模型 SHA256 为 `a322044f243fd7f308526f3dbbe73281c3881cc2935ac398ffa1e439935cad34`。重复采集不进入拟合，独立验证来源为空。

训练比较仍然 FAIL，不能因选择 Gamma 而宣称预测验证通过。Gamma 与 WLS 都有 9600/9600 已知预测、决策错误 0、时序失败 3；Gamma 的归一化时序误差均值为 0.4447390992，WLS 为 0.4606774489，故按既定字典序选择 Gamma。保留的三项训练失败为 seed 627601 skewed 队列 MAE 2.6138613861（阈值 1）、low image P95 相对误差 0.2457283046（阈值 0.2）、low OCR P95 相对误差 0.4864239875（阈值 0.2）。原始失败报告与数值修复后的训练报告均保留。

独立阶段继续使用原 seed 627701、H600、drain300、warmup0 和原四路线公共计划；资源只覆盖预测候选，计划字节不改。完整预测及其来源必须先 seal，原生 raw 的 `startup_begin_ns` 必须严格晚于 seal 的 `created_ns`；每个候选启动前重新审查来源。实际测量的父进程收讫 E2E、原六项效用和旧门槛继续由原 reporter/gate 计算。独立结果只用于本轮验证，不能回流训练，也不自动授权最终四组收益结论。

训练与协议证据：`background-statistical/gamma-h600-v2-cv/{report,selection}.json`、`background-statistical/gamma-h600-v2-full/verification.json`、`flow-validation-h600-selected-v16/protocol.json`，均位于 `artifacts/predictor-gpu-20261027/`。下一步只读审计新独立 raw 的完整 cohort、质量绑定、原生时序及原稳定性；本审阅不启动 GPU。
