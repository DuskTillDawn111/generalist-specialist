# 静态联合 MILP 与预测反馈

当前默认 `solver=milp`。模型、副本和路由由 MILP 主问题联合决定，再由同一个逐请求预测器计算完整效用。历史模块只作归档，不进入依赖；`solver=dfs` 保留为独立服务模型下的小规模参考。

本轮只有 CPU 验证，没有从旧 Pairwise 系数推导新模型池的干扰曲线，也没有拟合未经测量的能耗或 P95。

## 使用

```bash
python -m pip install -e .
gsdeploy plan --config configs/demo.json --solver milp \
  --route-grid 2 --milp-seconds 30 --output artifacts/milp-demo.json
gsdeploy plan --config configs/static_replicas.json --solver milp \
  --route-grid 4 --milp-iterations 1000 --milp-seconds 30 \
  --output artifacts/static-milp.json
python -m unittest tests.test_milp -v
```

固定部署仍通过 `--deployment` 指定。`--milp-columns` 限制主问题列生成；`--milp-gap` 设置绝对效用差容差。`--no-capacity-witness` 关闭补充容量见证，得到纯网格域。`--max-states` 仅控制 DFS。配置保存所有条件，运行继承该计划的求解和轨迹参数。

## 目标与候选域

五项成本与既有评分相同：

```text
C = -U = 0.2*T_loss + 0.2*L_norm + 0.2*P_norm + 0.2*M_norm + 0.2*E_norm
```

历史四项配置保留原权重。质量是任务模型路径的硬准入，没有必须保留通用或专用的约束。

副本域只有画像中的离散状态，包括零副本。分流比例取分母 1..K 的全部网格点的并集；K 增大不删除旧网格点。K=1 是单路径，K>=2 联合选择部署和分流。重复比例采用首次出现的规范表示。

默认 K>=2 且使用独立服务代理时，先解一次容量可行性 MILP：同时选择副本和连续任务流量，满足质量路径、显存、守恒、部署联动及家族压力约束。该辅助问题仅生成一条可行流量见证，最多使用总预算中的 5 秒；它的线性显存/服务目标不决定最终方案。若成功，将其逐任务分流及接收副本状态加入主问题列，允许主问题与网格列重新组合；与已有列重合时不重复加入。完整见证保存在 `domain.capacity_witness`。

这样避免粗网格完全漏掉容量可行的分流，同时仍是有限域。该见证不同于旧 DFS 依赖任务前缀的容量填充边界。若辅助问题未找到见证，主问题只搜索网格。`NO_FEASIBLE_GRID_PLAN` 只说明报告的网格及见证域内无合法方案，不等于连续路由或物理不可行。Oracle 对照必须冻结同一候选域；纯网格测试显式关闭见证。

## MILP 主问题

1. `z(m,n)` 为二进制，各模型恰选一个副本状态。
2. 每个任务 t 的离散列 c 记录分流请求率、接收模型、副本数和服务成本下界；`v(t,c)` 为二进制，每任务恰选一列。
3. 任务列必须与 `z(m,n)` 的副本数一致；不能分流到未部署或未准入的模型。每列流量之和为任务输入请求率。
4. 显存使用安全系数乘实测副本表，总和不超过 B。
5. 独立服务模式下，按任务列累积 `r(t,m)*s0(t,m)/(1000*g(m,n))`，约束家族总压力不超过 0.8。robust 使用 2 倍服务压力的必要约束。

主目标最小化连续变量 theta，初始约束为：

```text
theta >= exact_memory_cost(z) + sum_t service_floor(t, selected_column)
service_floor(t,c) = alpha_t / b_t *
                    (w_L * mean(service_samples) + w_P * P95(service_samples))
```

服务样本通过固定轨迹和同一个轮询路由器生成。实际延迟逐请求不小于服务时间，均值和对应样本分位数均不小于该下界。吞吐损失和能耗非负，只在数学下界中放宽为零；预测评分中的缺失能耗仍未知。robust 使用 2 倍服务样本。

为避免长轨迹在每列上重复执行轮询，按实际浮点权重序列、别名位置顺序和任务样本数缓存每个接收别名的计数。相同模式只模拟一次；不同模型服务值按计数计算精确均值和线性插值 P95。缓存保留真实轮询的有限样本分配，不用理论流量比例替代；FIFO 完整延迟仍由预测器计算。

**初始下界不是最终评价目标。** 每个完整候选都经过共享预测器，计算完整 T/L/P/M/E 并反馈后再求解。

## 完整预测与反馈约束

设完整身份包含 b 个选中二进制变量，合法实际成本为 C*，加入：

```text
theta >= C* * (sum(selected_identity_variables) - (b-1))
```

同一身份的右边为 C*；其他身份至少少选一个变量，右边不大于零，已有 `theta>=0` 保证不排除其他方案。这是离散身份反馈，不是把任意非凸预测器当作凸函数加切平面。

独立 FIFO 模式还有更强的路由反馈：任务列固定所有正流量模型的副本数，改变空闲驻留模型不改变 T/L/P，但可能改变显存和非可加能耗。因此仅跨这些部署复用延迟与吞吐成本：

```text
theta >= exact_memory_cost(z)
         + delay_cost * (sum(selected_task_columns) - (task_count-1))
```

robust 的 delay_cost 是三个场景中最大的 T/L/P 成本。不能跨部署复用整卡能耗；启用整方案共驻画像时也禁用路由反馈，因为空闲驻留模型可能改变服务画像。

约束不合法或排空失败的身份加入 no-good 约束，只排除该完整身份；不能根据一条坏路由排除整个部署。证据缺失的身份单独保留乐观下界，从当前可评分子问题移出，不认定为物理不可行。

## 界、停止与证书

合法完整成本提供上界，MILP 对偶界提供剩余域下界；未知身份移出后，全域下界取剩余主问题下界与未知身份下界的最小值。最大化 U 的乐观上界是成本下界的负数。

主问题下界达到当前方案成本的绝对容差、且没有影响证书的已发现未知身份时，返回 `OPTIMAL_MILP_FINITE_DOMAIN`。证书只覆盖冻结预测器、画像和有限网格，不能转为连续路由或真实 GPU 最优证明。

| 状态 | 含义 |
| --- | --- |
| `OPTIMAL_MILP_FINITE_DOMAIN` | 在声明域、冻结预测器与数值容差下完成求解 |
| `MILP_BUDGET_LIMIT` | 时间或反馈次数耗尽，保留当前方案和界 |
| `MASTER_BUILD_BUDGET_LIMIT` | 主问题未完整建立，不证明无解 |
| `MILP_SOLVER_ERROR` / `MILP_INVALID_WITNESS` / `MILP_NUMERICAL_UNRESOLVED` | 求解异常或数值问题，保留未解决状态 |
| `NO_FEASIBLE_GRID_PLAN` | 完整报告的网格及见证域内没有合法方案 |
| `INSUFFICIENT_COHORT` | 正需求任务在冻结窗口没有样本 |
| `FEASIBLE_INCOMPLETE_EVIDENCE_DOMAIN` | 有完整可评分方案，仍有未知候选，不给全域证书 |
| `INCOMPLETE_PERFORMANCE_DOMAIN` | 共驻画像缺失，没有已知合法可评分候选 |
| `DIAGNOSTIC_ONLY_NO_ENERGY` | 只有已知分项诊断方案，完整 U 为空 |

整池完全没有能耗画像时，可以优化已知分项诊断目标，权重不重新分配。`diagnostic_optimality_certified` 仅描述该诊断目标；完整五项 `optimality_certified=false`。部分候选有能耗时保留未知身份及其界。robust 比较最坏完整效用，任何场景能耗未知均不能补零。

算法最坏情况下仍可能访问大量离散身份。路由反馈复用队列成本，缓存减少列生成成本，下界剪掉不值得预测的方案，但不宣称解决了组合复杂度。时间预算、列预算、迭代次数、完整效用 gap 和缺能耗时的诊断 gap 分别报告。

## 共驻证据接口

`performance_mode=independent` 沿用单模型/副本常数服务代理，明确报告共驻竞争与真实服务尾部未测，不把独立家族容量当作真实单 GPU 总容量认证。

`performance_mode=profiled` 按完整部署、路由、负载、H、seed、到达过程和模型参数查冻结画像：

```python
from gsdeploy.performance import profile_key

key = profile_key(catalog, deployment, routing, rates, H, seed, arrival_process)
catalog["performance_mode"] = "profiled"
catalog.setdefault("colocation_profiles", {})[key] = {
    "kind": "measured",  # CPU 验证用 synthetic_fixture
    "scope": "whole_gpu_colocation",
    "coverage_valid": True,
    "service_time_ms": {
        "generalist": {"text": 12.0},
        "text_specialist": {"text": 3.0},
    },
    "provenance": "measurement record identity and protocol",
}
```

数字只解释格式。服务时间是应用 n/g 前的家族归一化值；每 worker 的实测服务时间需乘 g/n 转换。每条正流量路径必须有值。相同部署不同分流可对应不同服务时间；缺画像返回 `MISSING_COLOCATION_PROFILE`，不退回独立代理。P95 仍来自 FIFO 服务代理，不声称获得真实服务尾部分布。

有限画像无法提供未知组合的通用服务下界，所以 profiled 主问题将服务与容量放宽为非负零界，再由完整画像校验。零只用于松弛，不用于服务预测或评分。部分画像下可能较慢，但不会根据独立容量错误剪掉共驻方案。

能耗身份包含选中共驻画像内容；改变服务画像后旧电量不再自动适用。整方案能耗仍按整卡运行积分和启动摊销读取，无可加性假设。

## 验证与剩余证据

`tests/test_milp.py` 独立生成小规模完整网格，使用独立到达、轮询、FIFO 和分位数实现比较得分；覆盖多副本与分流、全专用、能力限制、非可加能耗、缺失/部分能耗、稳健场景、预算、长轨迹缓存、Poisson 突发排空失败、路由相关共驻画像及画像变更使能耗失效。另验证粗网格无解时容量见证恢复合法分流。原 DFS 测试显式选择 DFS，不混称同域。

后续需要测量更新后模型池的质量、不同输入与任务混合的服务时间、同卡干扰和容量、整卡能耗及启动成本、真实服务尾部。本轮完成求解器和证据接口，不替代这些 GPU 证据。

2026-10-15 本地验证：42 项 CPU 测试通过，含 13 项新增 MILP/见证测试；demo 的规划、真实 mock 子进程执行与离线评估通过，请求轨迹与计划条件一致，mock 正式 U 为空。现有静态模型池在总 21 QPS 下完成诊断目标求解；总 35 QPS、K=4 时补充容量见证找到了合法 ASR/VQA 分流，30 秒预算结束时保留方案，诊断效用差界约 0.00932，未宣称最优。真实模型池均缺整方案能耗，完整五项 U 为空。旧归档工作区保持原提交与干净状态。
