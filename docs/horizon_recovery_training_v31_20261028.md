# 专用 horizon-recovery 训练诊断与显式执行域

v31 从真实 v30 reader-visible 队列构造条件风险：**当前 waiting>0 时，第一次清空是否严格晚于 H**，H 固定为 1/5/10 秒。它独立于旧 `recovery_risk.v2` 的目标/API，未改旧 fitter、gate、rho、六项评分或 rolling forecast。

## 二值目标不需要捏造精确恢复时间

若 H 内观察到首次清空，标签为 0（恰好在 H 清空也为 0）；若完整观察到 H、队列还没清空，标签已经确定为 1，尽管精确恢复时长仍右删失。绝不填造 H+1 的 recovery duration。H 前采集结束且尚未清空时标签 UNKNOWN；如果结束前已经看到 H 前清空，0 仍然可知。当前队列为空的样本不属于该条件风险人口，不混作负例。

`horizon_recovery_risk_v31.binary_target` 实现上述规则。`training_row` 只把 v30 prefix 投影到当前 count/oldest/HOL、各 owner waiting/handler/next-ready、已收到的 phase、未完成 pipe write、分 task 原 input size 和完整公开未来到达等坐标；不需要编造 source-bound service work。输入 size 仍是原单位协变量，不能称为毫秒、容量或吞吐。GPU cursor、ready DAG 和资源 share 不进入特征。

`legacy_stumps_kernel` 隔离复用旧 boosted stumps 数值循环，调用原 sigmoid；CPU fixture 对照旧 fitter，base_score 和全部 40 棵 tree 精确相等。新 schema 直接接收真实 binary label，不把删失时长塞进旧 recovery API。参数固定为 40 棵、learning_rate=.2、minimum_leaf=2，不用留出数据选参数。各 horizon 的独立模型暂不保证联合 survival 单调性。

## risk-domain 桥接保留原始 SHA

四个 natural route 的完整 context 逐字段检查后，差异只有 `execution_nonce`、`whole57_training_route_id`。其余 context 全部字段、plan 的 resource_configuration、原始 input registry 和 recipe 严格相等。`horizon_recovery_domain_v31` 仅移除这两个预先命名的运行/路由元数据，生成共同 execution projection、完整差异记录、所有原 context 文件 SHA 和 execution_context SHA。原 SHA 从未改写，light/profiling 域不能混入 natural。

准备对象在 `artifacts/predictor-horizon-risk-domain-v31-20261028-prepared01/risk-domain.json`。它已绑定四个原 context，但目前只附 natural/balanced 的实际 PASS review，因此状态仍为 `PREPARED_PROJECTION_EQUAL_WAITING_FOR_ALL_ACTUAL_MEMBERS`，不能拟合 measured 模型。

每个实际成员须有 PASS training review、完整 57+12 completed、raw/source audit PASS，且 raw/plan/context 严格绑定原成员。随后比较 READY 的完整运行 payload，仅排除实际 PID；torch、worker/tasks、clock、spec SHA 等不同会使共同域 UNKNOWN。保存各 run 的原 PID 作审计证据。任意 prepared 执行字段或实际运行 projection 不同都禁止合域拟合，不能通过 CPU fixture flag 绕过。

这证明的是明确字段范围内的执行契约相同，不能证明所有未观测环境/硬件状态一致，也不是性能转移或物理预测资格。

## 已完成的真实样本与 CPU 检验

natural/balanced 的 1382 个 v30 prefix 中，1093 个 cutoff 当前有正 backlog，产生 **3279 个条件训练行**：

| H | 已知 0 | 已知 1 |
| --- | --- | --- |
| 1 秒 | 91 | 1002 |
| 5 秒 | 241 | 852 |
| 10 秒 | 444 | 649 |

2503 个已知 1 正好来自 v30“窗口内精确 recovery 尚右删失”的标签；它们没有被丢弃，也没有被当作未知或负例。当前这组 retained 样本没有 H 前行政删失的 UNKNOWN，但相应逻辑已由 CPU fixture 验证。真实结果在 `artifacts/predictor-horizon-risk-v31-20261028-training01-natural-balanced/`，明确 `model_fitted=false`。

13 个 v31 测试覆盖 horizon 已知二值与精确时间删失、行政截断、H 边界、空队列人口、数值 kernel 与旧 fitter 等价、整 run 留出、共享 recipe group 保留、未知标签不当负例、未来标签不进入打分、原 context 保留、真实 runtime torch/spec 差异、未完成 actual bridge 的强制阻断和 NaN/训练支持边界。加 v30/公共观察/旧 risk 回归共 **41 tests PASS**。增加 v32 冻结观察 sidecar 检查后共 48 个测试，接线见 `docs/observed_horizon_risk_sidecar_v32_20261028.md`。

natural/skewed PASS 后也已独立提取 v30 training05，并产生 `predictor-horizon-risk-domain-v31-20261028-prepared02`：两次实际 READY 完整 payload 除 PID 外严格相等，仍等待 low/high。`predictor-horizon-risk-v31-20261028-training02-natural-balanced-skewed` 保存两组的 6624 行（每 H 2208 个正 backlog cutoff），1/5/10 秒 known 0/1 分别为 140/2068、437/1771、870/1338。没有实际拟合。旧 delivery01 的源码、测试和文档 snapshot 均保留且匹配其原 SHA。

## 后续执行入口

四 natural 实采完成后，分别先用 v30 adapter 产出各自可见 prefix/隔离 label，再建立全部 actual 证据的 risk-domain。命令中重复 `--actual-review` 四次：

```bash
python -m tools.predictor_validation_20261027.horizon_recovery_domain_v31 \
  --prepared-root artifacts/predictor-whole57-joint-training-v23-20261028-prepared03/whole57-joint-natural-v23 \
  --actual-review <balanced PASS review> \
  --actual-review <skewed PASS review> \
  --actual-review <spill_low PASS review> \
  --actual-review <spill_high PASS review> \
  --output <新的已资格域目录>
```

然后运行 `horizon_recovery_training_v31 --risk-domain <risk-domain.json> --visible-report <v30 report>`，重复 `--visible-report` 四次，使用新 `--output`。默认只导出 true horizon rows；显式 `--fit-diagnostic` 才执行四次整 capture 留出。该入口在所有 actual 成员未资格化、缺少任一原 context、没有四个不同实际 run 时硬拒绝。

每折只用三个完整 run 拟合，保留另一个 run 的全部目标行，禁止窗口/请求随机切分。UNKNOWN 特征支持与标签数量明确报告，训练 class-rate baseline 只在候选同支持集合比较。共享 recipe/content group 不改名、不伪造输入独立性；新入口明确叫 `same_recipe_content_whole_capture_training_diagnostic`，不冒充旧 independent group holdout。密集 receipt cutoff 相关，不提供伪独立置信区间、输入泛化或概率校准声明。任何实际拟合结果也仅是候选训练诊断，不能据此接入 rho/六项评分或声称分流收益。
