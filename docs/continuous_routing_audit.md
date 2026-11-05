# 连续路由—离散部署：Phase 0 代码审计

审计日期：2026-11-01（Asia/Shanghai）。代码基线：`e3a53ad8f52f4a6c1578c35b1e8432852bd78390`。
开发分支：`codex/continuous-routing-optimizer`。
依据：[用户计划书 V3.0](continuous_routing_plan_v3.md)；原文逐字节归档，SHA256 为
`6ccbe58955ab041abdd438aef91d1b16c7e3e5322a619e8572323ee70804a901`。

本次完成源码审计、两个 CPU 接口探针及 [Phase 1 设计](continuous_routing_design.md)。没有实施新求解器，没有启动 GPU、采集、拟合或恢复旧大规模实验。

## 1. 审计结论

1. 连续路由的底层执行语义已经存在：`routing.py`、`trace.expand_routes()` 和 `prediction.predict()` 接受浮点 QPS。主要离散限制位于 MILP/DFS 的候选生成层。
2. 两个旧优化器都没有固定路由、搜索部署的公开接口。`plan(deployment=...)` 固定的是部署，不能作为固定路由内层。
3. 主包 `route_load_replay_v1` 仍是明确的 `UNIMPLEMENTED` 占位；它的零值只用于数学松弛，完整评价返回 UNKNOWN。
4. 现有标量 service 接口与研究 DAG 完整预测接口是两条链。任意共驻 DAG 不能先压成单一 service，再由旧 FIFO 重算；需独立的完整候选 adapter。
5. 本课题按“预测器已给定”推进算法开发。真实 adapter 的支持域不足时保留 UNKNOWN；可用显式 synthetic oracle 做算法正确性验证，不开展预测器研发。
6. 内层首先建立独立枚举参考。MILP 的价值必须由调用数与总成本证明；在非可分共驻黑盒下，它可能退化为昂贵的部署枚举。

## 2. 当前调用关系

```text
SystemConfig → DeploymentSystem.plan()
                 ├─ JointMilpOptimizer.solve()
                 │    └─ columns(): fraction_rows + replica-state combinations
                 └─ UtilityOptimizer.solve()
                      └─ enumerate deployments → rows(): fraction_rows
                                   ↓
                         predict_scenarios(trace=...)
                                   ↓
                predict() → resolve_service() → per-alias FIFO
                                   ↓
               quality + memory + energy + components + constraints

DeploymentSystem.predict() → validate_plan() → predict_scenarios()
runtime → expand_routes() + Router → workers → evaluation
```

`predict_scenarios()` 每次实际执行三个 `predict()`，服务时间乘数为 1、1.25、2，使用相同 cohort；这不是三条独立到达轨迹，也不是统计置信区间。

## 3. 数学变量与代码字段

| 数学对象 | 当前字段/实现 | 新算法中的处理 |
| --- | --- | --- |
| 模型集合与能力 E(t) | `catalog.models[m].supported_tasks`；六项研究还要求 executable capabilities | 由配置生成合法边；模型角色不限制合法分流组合 |
| 副本状态 N_m | `capacity_factors[m]` 的整数键及相应 `memory_table_mib[m]` | 只采用声明且合法的状态；0 表示未部署，不插值生成副本画像 |
| 部署 D | `deployment` / `x`：`{model: positive count}` | 内层优化，零副本从序列化部署中省略 |
| 任务率 λ_t | `workload` / `rates` | 正负载任务进入优化；`SystemConfig.load()` 去除零负载任务 |
| 连续份额 r_t,m | 旧优化器由 `fraction_rows()` 得到 | 新域直接接收连续数值，保存精确零边界 |
| 实际 QPS λ_t r_t,m | 现有 `routing[task][model]` | 唯一份额→QPS adapter，不能把份额直接传给旧预测器 |
| 总正流量 Λ_m | 当前通过每任务路由和部署合法性间接约束 | 新 `activation.py` 跨任务求和，正流量模型禁止状态 0 |
| 轨迹 τ | `make_trace()`、`trace_digest()` | 评价前冻结，所有算法共享；还需绑定 sample/input 顺序 |
| 效用 U | `prediction.components()` 与模式对应资源/质量实现 | 原样复用，六项为负加权损失；不加第二套效用函数 |
| Nominal/Robust | `U` / `scenario_U_low` | 保持 nominal 与同轨迹 stress 语义 |
| 固定路由上界/认证 | 旧 `utility_upper_bound`、`optimality_certified` | 新结果独立命名 scope，不继承旧有限路由证书 |

## 4. 逐模块审计

### 4.1 `milp.py`

`solve()` 用 `state_index[(m,n)]` 建 one-hot 模型状态，`columns()` 用路由分母 1..K 与显式 pair grids 生成任务 route/count 列，再建立任务选列、状态链接、显存和聚合压力约束。

Master 最小化 `theta`（成本下界），候选调用 `predict_scenarios()`，完整 identity 反馈修正其成本；未知 identity 的乐观界单独保留。`result()` 的认证只针对冻结预测器和声明有限网格。

可复用：`solve_master()` 的单线程 SciPy/HiGHS 调用；`residual_identity_cut()` 的数学形式（新 one-hot blocks 必须重新代入验证）。
不可直接复用：`columns()`、依赖网格的任务选择变量、capacity witness、`solve(K=...)` 及 result 的域说明。

旧 `performance_mode=independent` 分支有跨部署的 route feedback，它依赖固定任务 route/count 列确定 FIFO 的特殊性质。不能转用于一般共驻 provider，也不能跨连续路由复用。

### 4.2 `optimizer.py`

`UtilityOptimizer` 先枚举部署，再按任务搜索网格路由；不是 fixed-routing enumeration。
部署前剪枝涉及 additive memory、capacity 和质量/solo-FIFO 界，均有模式条件，不能把该搜索循环整体搬成通用黑盒求解器。

`plan_key()` 将份额 `round(...,12)`，是历史有限域排除身份的一部分；不得作为新连续路由缓存键或 identity。
新枚举保留“逐有限部署调用同一 evaluator”的思路，自己维护固定路由域与终止状态，不调用旧 `solve()` 来伪装内层。

### 4.3 `routing.py`、`trace.py`

`Router` 为每任务维护 smooth weighted round-robin deficits，模型/alias 按名称稳定排序，平局选择排序在先者。`expand_routes()` 将一个模型的 QPS 均分到其 n 个 alias。

FIFO clocks 按 alias 保存，所以同一 worker 上不同任务共享队列，不是每任务各自独立 FIFO。副本数变化会改变 alias 集合及实际请求分配，即使 family 份额不变。

路由数值连续不意味着完成时间、P95、质量或 U 连续。有限 cohort 会产生 assignment 平台；rho、aggregate quality、能耗模型等仍可能直接依赖浮点负载，所以相同 assignment 也不代表完整预测相同。

`make_trace()` 返回路由无关的 `(arrival, task)` 元组。sample ID 由质量/回放链依据 input order 和 seed 决定。`trace_digest()` 本身不包含这些 sample 身份，不能单独作为完整实验缓存身份。

### 4.4 `prediction.py`、`performance.py`

`predict(..., trace=...)` 已有冻结轨迹接口。`DeploymentSystem.predict()` 没有显式 trace 参数；它从 `context()` 传 horizon/seed/arrival 等，再在底层生成轨迹。

`resolve_service()` 支持 independent、profiled、task-replica proxy 和已校准模式。profiled 查询绑定完整部署/路由合同，缺项返回 UNKNOWN；不保证支持任意连续路由。

主 `predict()` 用解析 service 和 n/g 缩放推进独立 alias clocks。它可以承接其适用域内的回放模型，不能由本次审计认定为支持任意共驻干扰预测。

正式 U 需要完整指标及约束通过；能耗或质量缺项保留 `U=None`。`diagnostic_partial_U` 不能成为新算法的可行 incumbent。
原完整约束包括能力/流量守恒、显存、rho≤0.8、drain 完成；六项研究没有默认绝对质量门槛。新计划提到的质量硬约束只在问题实例明确声明时适用，不能新增全局 Q_min。

### 4.5 `surrogate_bounds.py`、`coupled_mode.py`

`surrogate_bounds()` 只给经过版本与范围检查的冻结校准模型返回可分界；不是任意预测器的通用界。
在完整共驻模式下，独占服务时间、独占显存总和和独占容量不天然构成必要条件。没有适用证明时不能据此筛除部署，包括未接收流量的 resident。

`coupled_mode.validate_placeholder()` 仍强制 `status=UNIMPLEMENTED`；`unknown_service()` 返回 `ROUTE_LOAD_MODEL_NOT_IMPLEMENTED`。任何零界都不构成服务、显存、能耗或可行性预测。

### 4.6 `config.py`、`system.py`、`search_session.py`

`SystemConfig.solver` 白名单只有 `milp/dfs`；`DeploymentSystem.plan()` 没有 fixed-routing 参数。新开发先用独立实验入口与配置包装，旧 CLI 默认不变。

`SystemConfig.validate_plan()` 同时检查结构与调用 service 解析，是现有 predictor 模式专属验证器。新 whole-candidate adapter 应共享结构规则，但不能强行经过旧 service 分解来判定未知的 rho 或共驻显存。

`SearchSession` 复用冻结输入与 digest memoization，但内部加载特定历史数学源；不是通用完整预测缓存。可复用不可变问题/精确 float identity 的原则，不能把旧 session 当作新 oracle。

### 4.7 研究 prototype 与主包边界

`artifacts/predictor-integrated-small-closeout-v1-20261101/integration.py` 接了 typed DAG、batch、互补候选和矩阵 hook，但只验证 4 个 fixture 请求；`full_U=null`、deployment 未获资格，不支持据此搜索任意模型/副本/路由域。

既有 [whole-forecast provider 契约](production_whole_forecast_provider_contract_v1_20261029.md) 描述了同样的接线缺口及后续原型；那些设计/原型的存在不等于当前主包已注册通用 provider。
新算法只依赖显式 adapter 合同，绑定实际源码/画像/支持域；本轮不迁移整个历史研究链，也不重新校准。

## 5. 实际检查与结果

基线 CPU 回归已在本会话前一轮对相同提交执行：`python scripts/run_cpu_tests.py`，203 个测试通过，15.622 秒。
本次仅做下列短 CPU 探针，不修改配置文件或源代码：

| 探针 | 实际输入与输出 | 可支持的结论 |
| --- | --- | --- |
| 非网格比例 | demo text Specialist share `0.3141592653589793`，QPS `15.707963267948966`；constraints_valid 和 route_conservation 均 true | 路由、结构验证和 FIFO 可接浮点比例产生的 QPS |
| 非网格证据缺项 | 同次预测 reason=`MISSING_ENERGY`，U=null | 旧 demo 能耗 fixture 不覆盖任意连续路由，不能作为真实完整 U oracle |
| alias 映射 | 前 50 个 text 请求：G#0=34，S#0=16 | 有限请求映射是确定的整数分配，不等于逐窗口精确比例 |
| coupled placeholder | 合法 placeholder 配置，reason=`ROUTE_LOAD_MODEL_NOT_IMPLEMENTED`，performance_unknown=true，constraints_eligible=false，U=null | 占位安全拒绝，无实际共驻数值预测 |

## 6. Phase 1 必须处理的风险

| 风险 | 实现要求 |
| --- | --- |
| 份额/QPS 混淆 | 保存两者；守恒只在单一转换入口校验 |
| 把很小正值视为零 | 不设隐式 epsilon 激活阈值；检测浮点乘法下溢并明确报错 |
| 总流量为零便强制卸载 | 只禁止正流量模型取 0；零流量模型仍允许合法正副本，避免删掉共驻域 |
| 黑盒下继承独占剪枝 | 通过显式 bounds capability 启用；无证据时禁用 |
| 未求完结果当成 V(r) | 结果分开 incumbent、上界、终止原因、认证及未决部署 |
| UNKNOWN 当物理不可行 | 使用三态约束，不沿用仅 performance/memory 两个 unknown 标志的局限 |
| 缓存量化或遗漏来源 | float 精确编码；绑定完整轨迹、输入、预测器、约束和目标版本 |
| 偏置预测调用计数 | 分开外层评价、候选预测、轨迹、stress leaf 与缓存命中 |
| 多轨迹每条选不同 D | 同一候选 `(D,r)` 在全部轨迹评价后聚合，再内层选 D |
| nominal posterior 被当 stress | provider 显式声明 stress 支持，未支持 robust 保留 UNKNOWN |
| 旧网格基线被声称优化新目标 | 单轨迹直接匹配；多轨迹基线需明确聚合优化支持，否则仅作为冻结方案泛化评价 |

## 7. Gate 0 判定与下一步

代码关系、变量/边界、预测器接口缺口与实施方案已明确，Phase 1 设计见配套文档。
可以开始独立 CPU 算法实现：连续域与激活 → 枚举 → MILP → DE。
真实预测器实验的前提另列：adapter 对声明候选域提供完整效用与约束；未满足时仅标记 synthetic/受限域算法验证。
这不会触发新的预测器训练或恢复已取消的历史 campaign。
