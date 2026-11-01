# 当前真实观察与 horizon-risk 的接线

可以接的是 **parent reader-visible observer → 冻结 risk sidecar**。现有 v21/v19 模拟器没有与其语义等同的当前观察 checkpoint，不能把模拟状态转称为真实 prefix。

`LifecycleDagPhaseModel._forecast` 要求所有输入 arrival>=now，从空 pending/running/active 初始化；明确拒绝 checkpoint、pause-after-event 与 pause-at-time。它的 `LifecycleSchedulingView.owner_lifecycle/handler_requests/pending_requests` 是模拟器 same-clock settle 后的状态，供 scheduler 决策，不是父 reader 收到的实测事件。

`QueueDiagnosticEngine.forecast` 把 `FullPublicDagQueueDiagnostics` 绑定到整次模拟，最终 sidecar 的 max-waiting/work/age 来自未来 forecast 路径。公共 future-offered envelope 可以是合法计划信息，但 sidecar 的全过程 peak 不能作为“当前队列”输入风险。v31 直接从 v30 完整公开计划重算 future count features，不使用其 simulated future maxima。没有修改这两个冻结 engine。

## 已实现的独立观察器 evaluator

`tools/predictor_validation_20261027/observed_horizon_risk_sidecar_v32.py` 提供 `ObservedHorizonRisk.predict_prefix` 和 `predict_runtime_records`。后者复用 v30 `snapshot_at`、公共 observed_state 和实际 reader receipt cutoff，保留 host enqueue/dispatch/write、handler/phase、完整公开 arrivals/原输入的因果观察。

`predict_prefix` 只接收精确 v30 envelope，拒绝追加 offline label 字段、v19 sidecar 或 forecast diagnostics；cutoff、clock、原 execution context 必须一致。模型和 domain SHA 每次核验，原 context 必须属于显式桥接成员；外域和训练支持外坐标保持 UNKNOWN。当前队列为空则返回 `NOT_AT_RISK`、risk_score=None，不制造“低概率”。

模型保持冻结，不拟合、不读取 label、不改 utility/rho。输出携带 model/domain SHA、原 execution context、visible-state SHA、当前 waiting count、H 和 score。score 是未校准候选输出，不能称为已验证概率。CPU fixture 有独立显式开关，不能成为 measured evidence。

CLI 的 `predict` 模式仅打开模型、domain、v30 report 的 features 文件，**不打开 labels 文件**，保存 sealed-predictions。冻结绑定精确 model/domain/visible-report/features receipts，并通过 AST 递归记录完整本地静态 import 源闭包（当前 84 文件，含包初始化文件），核验前后未变。

prefix 附带公共 host observed_state、自身 state SHA、allowlist phase metadata/host pipeline proof 与整个 prefix SHA。评分先验证 host state 的时间、context 与 hash，再由公共原输入 registry 和这些可见事实完整重算 prefix，拒绝只改派生 age/等待集合后重算外层 hash。此证明针对 reader-visible host state 的一致性；不声称 GPU cursor、kernel readiness、SM 服务或硬件进度已观察。

独立 `evaluate` 模式再加入 labels，只能评估冻结模型，拒绝训练 run 重叠、原身份不一致、模型 SHA 改变和 cohort 缺行。它验证封存报告 schema/status、冻结绑定、完整源闭包、每行 domain/context/cutoff/H 及有限 [0,1] score/status；并对精确冻结 feature prefix 使用同一冻结模型重新计算，逐字段比较预测，拒绝替换分数、状态或伪造空队列。重放不拟合、不依据标签选择参数。报告 Brier/AUC/reliability bins、UNKNOWN/不在风险人口数量。共享 recipe/content、密集窗口相关和支持覆盖限制仍在，不据这些 bin 自动宣称概率校准。

```bash
python -m tools.predictor_validation_20261027.observed_horizon_risk_sidecar_v32 predict \
  --model <已冻结的独立训练模型.json> --risk-domain <已资格域.json> \
  --visible-report <待评估整capture的v30 report.json> --output <新预测目录>

python -m tools.predictor_validation_20261027.observed_horizon_risk_sidecar_v32 evaluate \
  --prediction-report <新预测目录/report.json> --label-report <同capture v30 report.json> \
  --model <同一冻结模型.json> --output <新评估目录>
```

实际四 natural 数据未齐、bridge 未完成时，measured provider/fit 硬拒绝。后续每个 holdout capture 的模型只能由其余三个 whole run 冻结，再封存该 capture prefix 的预测，最后 join 标签；不能看留出标签挑模型/参数。本接口也可在后续新独立 run 上执行同样流程，但不声称它已支持输入家族泛化。

## 当前验证与未完成范围

八个 v32 测试实际通过观察器调用检查 late START receipt、未来 END mutation 不改预测、拒绝 simulator/label 输入、冻结模型不变、context/cutoff 约束、当前空队列、host hash/派生 prefix 篡改，以及独立 sealed label-join 的模型、状态、分数、域与训练 run 约束。加 v31/v30/公共 observed_state/旧 risk 回归共 **49 tests PASS**。CPU fixture 的预测与评估 artifact 独立保存，不能代替真实模型准确率。

历史 v30 training01–05 和 v32 cpufixture01 保留对应旧 implementation snapshot，未追溯重写成新 proof。当前 proof 数据使用新目录 training06/07；接口完成后封存源代码与完整闭包，后续修复必须新增版本。

这完成了风险观察 sidecar 的具体接线。没有 native DAG cursor/checkpoint，也没有把 observed state 恢复为 v21 scheduler 状态或执行 rolling forecast。要让全过程预测随真实进度重算，仍需独立设计完整 DAG/host/lifecycle checkpoint 契约并验证；当前 sidecar 只补真实可见风险诊断，不作为六项评分新项。
