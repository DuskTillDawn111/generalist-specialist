# 连续路由 Phase 1：接口与复现

新算法位于 `gsdeploy.continuous`，独立于原 `DeploymentSystem.plan()`。基础依赖仍是 NumPy/SciPy；绘图额外需要 matplotlib。不会导入 torch、启动 worker 或 GPU。

## 一次小实验

在仓库根目录运行，输出目录必须尚不存在：

```bash
python experiments/compare_continuous_optimizers.py \
  --config configs/continuous_fixture.json \
  --algorithms de_enumeration de_milp milp_K2 dfs_K2 \
  --seeds 11 --outer 32 --predictions 1000 --seconds 15 \
  --output artifacts/continuous-manual-small
```

该配置是明确标记的 CPU synthetic fixture：共驻 slowdown、质量、显存和整板功耗全部为合成原语，由原 FIFO/评分函数评价；不能当成真实 GPU 实验或预测器精度验证。
预置完整对比包含低/混合/文本热点三个负载、三个优化 seed、K2/K4 Joint MILP 和 K2 DFS。

```bash
python experiments/benchmark_fixed_routing.py \
  --config configs/continuous_fixture.json \
  --output artifacts/continuous-manual-inner
python scripts/plot_continuous_routing.py \
  --source artifacts/continuous-manual-small \
  --inner artifacts/continuous-manual-inner \
  --output artifacts/continuous-manual-figures
python -m unittest tests.test_continuous_routing -v
python scripts/run_cpu_tests.py
```

## Python 接口

```python
from gsdeploy import SystemConfig
from gsdeploy.continuous import (
    Budget, ContinuousProblem, ContinuousRoutingEvaluator,
    DERoutingOptimizer, LegacyPredictorOracle,
)

config = SystemConfig.load("configs/demo.json")
problem = ContinuousProblem(config)
evaluator = ContinuousRoutingEvaluator(
    problem, LegacyPredictorOracle(), inner="enumeration",
    budget=Budget(seconds=20, max_predictions=500, max_outer=32),
)
fixed_route = evaluator.evaluate({
    "text": {"generalist": 0.3141592653589793,
             "text_specialist": 1 - 0.3141592653589793},
    **{t: {"generalist": 1.0} for t in config.workload if t != "text"},
})
# demo 能耗只覆盖少量历史身份；非网格路由可能返回 INCOMPLETE_EVIDENCE。
```

同一评价器可用于 `DERoutingOptimizer(evaluator, seed=11).solve()`；若先评价固定路由，这些成本仍在同一 budget 中，不能在公平实验中免费热启动。
`inner="milp"` 使用仅含模型副本 one-hot 的新 Master，没有旧路由候选列。两个内层共享完整候选评价及缓存。

## 接入已有完整预测器

调用方传入一个 oracle 对象，至少提供：

| 接口/属性 | 合同 |
| --- | --- |
| `version`、`check()` | 完整实现/画像/机制的不可变身份与失效检查 |
| `evidence_kind` | 如 existing_predictor 或 synthetic_fixture，不能伪装实测 |
| `supports_stress` | 是否支持本问题声明的 robust 情景 |
| `cost_lower_bound` | 全域成本下界；无证明设为 None |
| `state_costs(problem)` | 适用时提供 `(model,count)→可分成本下界`，否则 `{}` |
| `memory_states(problem)` | 适用时提供安全显存下界，不能把独占显存直接当共驻下界 |
| `predict(problem,D,routing_qps,trace,budget,deadline)` | 一个完整 trace/scenario bundle，调用前 `budget.consume("predictions", deadline)`，记录实际 leaf 数 |

完整 prediction 使用原 `U/constraints_eligible`；robust 使用 `scenario_U_low/stress_constraints_valid`。缺来源、约束或任意必要资源时明确标记 `evidence_unknown/constraint_unknown/performance_unknown/memory_unknown`，不把缺项当物理不可行。
若旧聚合 API 没有逐 stress 的缺项原因，adapter 保守返回 UNKNOWN；有确切支持时可声明 `stress_evidence_complete`。
`trace_sha256` 如返回，必须匹配输入冻结轨迹；通用 adapter 还须把样本/输入、执行 roster、路由规则和版本纳入自己的完整身份。

这里没有实现或训练新的真实共驻预测模型。`route_load_replay_v1` 仍保持原 `UNIMPLEMENTED` 语义；真实模型接入后才能开展该支持域内的真实预测算法比较。

既有 `calibrated_replay_v2` 已可用于其原验证支持域内的 CPU 搜索对照；这与未实现的 `route_load_replay_v1` 是不同模式，见 [真实回放诊断](continuous_routing_phase1_diagnostics.md)。

## 输出与成本口径

每次实验保存输入配置、每算法/seed 原始 JSON、内层 ledger、连续路由、完整预测、convergence、heldout 结果、JSONL、CSV 与源码摘要。
新算法的 `full_prediction_calls` 是 `(D,r,trace)` 情景 bundle；旧 predictor 每个 bundle 运行三个 stress leaf。额外报告 `predictor_leaf_calls`，不混算。
heldout 的成本单独保存，不计入优化预算；它不能反馈到赢家选择。总实验 wall time 包括所有实验及保存开销。

旧基线保持原实现：对 synthetic fixture 预先物化声明有限网格的精确服务/资源原语，计入准备时间，再由原 Joint MILP/DFS 搜索。
旧 DFS 的 state budget 比 prediction budget 更严格，结果显式披露；对时间和预测成本的对比需同时看实际预算和停止原因。
旧算法的多轨迹聚合训练目标或自定义 arrival calendar 没有公开接口，运行器会返回明确不支持，不将另一目标冒充公平基线。

## 证书

`certified=true` 仅认证当前固定连续路由下、有限合法副本域和指定 oracle；DE 的 `continuous_optimality` 始终 false。
预算停止可以返回最佳有效 incumbent，但未完成内层不会作为精确 V(r) 传入 DE。预测证据不完整时停在对应状态，不继续填造效用搜索。
MILP 使用完整身份 no-good 反馈与显式安全成本界；一般黑盒下可能穷尽所有部署。是否值得使用它由 Phase 1 成本结果决定。

## 内层效率改进：bounded（显式选择）

`inner="bounded"` 提供先计算安全界、再按界排序的有限域搜索，保留完整枚举与 MILP 对照：

```python
evaluator = ContinuousRoutingEvaluator(
    problem, oracle, inner="bounded", budget=Budget(30, 1200, 64),
    inner_bounds=True, inner_warm_start=False,
)
result = DERoutingOptimizer(evaluator, seed=11).solve()
```

`inner_bounds=False` 关闭界与按界排序；`inner_warm_start=False` 关闭上一可行部署的候选优先级，可用于消融。热启动只保存一个部署字典，必须在新路由重新评价；非法候选自然不进入当前部署域，不享有免费预测。实例仍受问题、轨迹及 oracle 身份检查保护。

通用 oracle 只使用它显式声明的全域/状态成本 floor、显存 floor。强界仅对精确类型 `SyntheticColocationOracle` 开启；继承类不会自动继承证明。合成模型复用同一资源函数计算显存、rho、功耗，以无排队最小服务时延界定 L/P，并调用原质量函数和评分函数形成 U 上界。缺质量时该项只取非负下界，不产生可行预测。完整证明、实测开销及采用范围见 [内层改进报告](continuous_routing_inner_improvement.md)。

每条结果增加 `bound_seconds`、界的 `scope`、剪枝原因和 `warm_start_evaluated_deployments`。界和排序时间计入共享绝对截止时间，预测预算仍按完整 trace bundle 计数。被实际评价为 UNKNOWN 的候选继续阻止认证。新搜索需要遍历候选域构造界，适用于当前小型有限域，不是大域可扩展性结论。

复现同路由消融与独立端到端 DE：

```bash
python experiments/benchmark_continuous_bounds.py \
  --config configs/continuous_fixture.json \
  --seeds 11 29 47 --outer 64 --seconds 30 --predictions 2400 \
  --output artifacts/continuous-bounds-manual
python -m unittest tests.test_continuous_bounds -v
```

输出目录必须不存在。结果包括枚举 DE、四种同路由消融及 combined DE，保存冻结轨迹、逐路由 ledger、独立 heldout、源码快照、逐文件 SHA256 和环境版本；逐路由 U/认证或端到端路由序列不一致即报错。研究 runner 也接受 `--algorithms de_bounded`，默认规划器入口保持原设置。

## 已验证回放界与不可变摘要缓存（可选）

```python
from gsdeploy.continuous import (
    Budget, ImmutableContinuousProblem, SessionPredictorOracle,
    ContinuousRoutingEvaluator, DERoutingOptimizer,
)

budget = Budget(8, 10000, 32)  # 建立 problem/session 也消耗此绝对预算
problem = ImmutableContinuousProblem(config, replica_options=replica_options)
with SessionPredictorOracle(problem) as oracle:
    evaluator = ContinuousRoutingEvaluator(problem, oracle, "milp_bounds", budget)
    result = DERoutingOptimizer(evaluator, seed=11).solve()
```

`milp_bounds` 使用原 Joint MILP 的任务/副本列安全界，通过原门禁后才启用；其他 oracle 保守退回普通固定路由 master。它不同于 synthetic `bounded`，不预枚举完整部署域，输出 `validated_bounds_enabled`、`bound_scope`、`bound_seconds` 和逐轮 `milp_rounds`。新模式的界由模式名显式开启；`inner_bounds` / `inner_warm_start` 仍是 `bounded` 模式参数。

缓存可独立选择：使用冻结问题和 session oracle 配 `inner="milp"` 即 cache-only；普通问题和 Legacy oracle 配 `inner="milp_bounds"` 即 bounds-only。`enumeration` 仍是独立参考；原可变问题继续每次检查全量身份。session 仅缓存不可变摘要，不按近邻路由、相同 FIFO 分配复用 U；每次优化应独立建立并关闭 session。

支持域、证明及 104 次消融结果见 [已验证界与缓存改进报告](continuous_routing_validated_improvement.md)。五模型固定路由可减少预测，端到端仍落后旧 DFS/Joint；因此这些能力保留为显式选项。

## Joint 初始解与固定部署连续精修

2026-11-02 补充：该入口现在接受显式实验模式 `paper_profiled_fifo_v1`，
以同一阶段共驻/FIFO 预测器评价 Joint 初始解和后续固定部署路由。
源码身份覆盖阶段模拟、生命周期和资源合同；新模式的执行合同校验不授予实测精度资格。
完整结果及暂缓的真实画像事项见 [当前状态](joint_refinement_current_status.md)。

小范围重评入口（输出目录必须不存在）：

```bash
python experiments/reevaluate_joint_refinement.py \
  --config configs/paper_fifo_cpu_fixture.json \
  --seconds 4 --routes 32 --seeds 11 29 47 --output /tmp/joint-refine-new
```

默认关闭部署邻域。当前示例仅为CPU软件画像；真实画像缺失时不能称为真实预测器的效用重评。

```python
from gsdeploy.continuous import JointMILPContinuousRefinement

result = JointMILPContinuousRefinement(
    config, replica_options,
    seconds=8, max_routes=128, seed=11,
    method="de",                 # 或独立命名的 "local" 成对流量转移基线
    initial_fraction=1.0,         # 初始Joint可使用剩余总预算，不重置截止时间
    neighborhood=False,          # 独立可选部署邻域；默认关闭
).solve()
```

流程使用原Joint完整结果热启动。固定部署阶段调用原完整预测器，无部署MILP；零流量模型仍驻留。`winner` 始终保留初始/后续有效候选中的最佳值；`initial`、`initial_winner`、`fixed_winner`、`histories`、`convergence` 与阶段计数全部保留。若初始阶段没有有效incumbent，返回明确空赢家，不把诊断分数用于精修。

`neighborhood=True` 生成有限合法的一次部署变化，默认最多4个。固定阶段用剩余路由/时间预算的40%，其余分给邻居；每次改变部署均重新预测。它与固定部署版本分别比较。当前没有观察到额外效用收益，见 [扩展性与精修报告](joint_scalability_and_refinement.md)。

独立 `FixedDeploymentRoutingEvaluator` 可用于固定某个部署的连续研究；其 `domain` 只允许该部署内支持任务的模型。`DERoutingOptimizer(initial_shares=...)` 先评价原始比例，并在完整域DE种群中注入对应连续坐标；未提供初始比例时保留原顺序。`PairwiseTransferSearch` 是有限步长的局部基线，不是MADS。
