# 部署候选筛选与自适应连续路由预算分配

本轮新增独立 `AdaptiveCandidateRefinementOptimizer`，保留原Joint MILP、DFS、DE、完整FIFO预测器、效用函数和默认规划流程。先完成历史候选诊断与连续域上界的正确性检查，再进行同预算消融。没有增加GP-BO、MADS或部署邻域。

实验的最终结论与对照表见第4–6节。下文“安全”仅针对当前经过门禁验证的冻结数学预测器，不代表物理GPU性能界；“候选”也不宣称是连续域全局Top-K。

## 1. 历史候选：收益、差距和计算投入

复用上一轮正式36次运行的原始数据，没有为历史诊断调用新预测器。对132个候选出现记录整理网格U、精修U、增益、当前有效最佳U与差距、模型集合、副本、显存/能耗贡献、连续维度、调用数、首次改善成本和种子波动。不同算法/k/轨迹的重复出现按记录保留，不冒充独立模型结构。

[候选级完整表](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-history-v2-20261102/candidate-potential.csv) · [种子变化](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-history-v2-20261102/seed-stability.csv) · [历史界核验](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-history-v2-20261102/audit.json)

定义 `observed_gain = U_refined − U_grid`，`gap_to_incumbent = U_best − U_grid`。前者是预算内已观察到的增益，不能用作未来收益上界。以下为trace43、utility-k5、优化seed11的一组记录；副本均为1，G=generalist_2，I/O/A/T=image/ocr/asr/text。

| 网格名次/部署 | 连续维度 | 与第一名差距 | 已观察增益 | 精修预测 | 首次改善前预测 | 搜索秒 |
|---|---:|---:|---:|---:|---:|---:|
| 1，G+I+O | 2 | 0 | 0 | 75 | 未出现 | 4.776 |
| 2，G+I+O+A | 3 | 0.026867 | 0 | 76 | 未出现 | 4.694 |
| 3，T+I+O+A | 0 | 0.036130 | 0 | 0 | 不存在连续自由度 | 0.011 |
| 4，G+I+O+T | 3 | 0.037952 | **0.013194** | 113 | 13 | 7.084 |
| 5，G+O+A | 2 | 0.081927 | 0 | 113 | 未出现 | 6.709 |

Text候选网格U为−0.344805975，精修至−0.331611573，超过原第二/第三名，升至第二；仍低于第一名−0.306854442。连续维度比赢家高，但与原第二名相同，所以不能用维度单独预测收益。其分流改善来自共享FIFO、逐输入服务/质量差异及尾部请求分配；额外Text显存成本在固定部署内不变，却在跨部署比较中形成显著负项。上一轮已分解出代表Text方案相对第一名的显存贡献差约−0.04317，足以抵消质量/延迟收益，见[原报告](/root/autodl-tmp/generalist-specialist-project/docs/multi_candidate_continuous_refinement.md)。

首次改善成本也不同：trace42的Text仅改善0.000222，首次改善在13次新预测后；trace43同为13次，却改善0.013194；trace45为14次，改善0.016543。其他多个候选投入75–127次预测仍无增益。完整表保留“未改善”为空值，不能将首次改善成本记为零。观察到无增益不是证明该部署永远不能改善。

![历史增益相对所需差距、维度与评价成本](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-history-v2-20261102/candidate-potential.png)

两个旧优化种子在trace43/45的Text改善相同；不同轨迹上其增益变化较大。多样性策略在trace42还曾漏掉Text候选，另一个低排名部署的种子增益为0.000500–0.007518，说明候选选择和有限搜索均存在变化。当前任务不把历史经验增益带入安全证明，也不跨轨迹复用预测。

## 2. 覆盖整个连续路由域的安全筛选

实现：[candidate_bounds.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/continuous/candidate_bounds.py)。先审计原 `prediction.components`、`quality`、`calibration[_v2]` 和 `surrogate_bounds`：

- 原效用是 `U = −Σ w_j C_j`，所有权重非负，六项成本Q/T/L/P/M/E非负。
- Q为质量**损失**，不是正向质量分数。CER/WER可以大于1，不截断为1，也没有额外绝对质量硬门槛。
- T为到达窗口内未完成请求占比；L/P为按任务到达率加权、按原任务参考归一化的Mean/P95。
- M是全部驻留模型的预测显存/预算，零流量不会移除模型；E包含板卡截距、busy项、驻留worker和摊销启动项。
- 硬约束仍包括任务合法覆盖、路由守恒、显存、原rho≤0.8和排空；不能把rho或网格上某一路由的表现当作整个域的证据。

### 2.1 可行性检查与能力门禁

独立于具体路由，可以安全排除非法副本、无法覆盖必需任务的部署。对原门禁支持的 `calibrated_surrogate` / `calibrated_replay_v2`，还能使用确切的固定驻留显存公式排除超预算部署。

实现仅接受精确的内置 `ImmutableContinuousProblem` / `SessionPredictorOracle` 类型和相同问题身份。复用现有validated gate；未知模式、未验证/改变负载、缺失数据不获得经验“补界”。不支持的预测模式返回UNKNOWN；结构性无覆盖/非法状态仍可按声明域判断。没有采用未经证实的质量硬约束。

当前真实候选由原Joint产生，已满足约束，因此本轮正式实验中的安全删除主要是效用上界支配，并非大量候选显存不可行。

### 2.2 上界的数学依据

固定部署D，允许所有任务在已部署合法模型之间取连续simplex路由。构造每项成本下界，再取负号得到 `UB(D)`。以下松弛**不使用Router选出的某条固定路由**，也不转用旧 `Joint.floor(task,x,row)`。

**驻留显存。** 冻结模式中的显存是与路由无关的确切预测值：

```text
memory(D) = max(safe_factor × memory_multiplier × Σ memory_table[m,n_m],
                training_deployment_memory[D])
M_floor = memory(D) / memory_budget
```

它包含零流量模型。此公式不能移植到缺少相同证据的profiled/coupled模式。

**能耗。** 已验证模式的三个功率系数、启动系数均为非负；实现再次检查。每任务busy成本为路由QPS的线性函数，simplex上的下界是其最小合法路径成本：

```text
E_floor = [intercept + worker_coef × Σ n_m
           + startup_joules_per_worker × Σ n_m / amortization
           + busy_coef × Σ_t rate_t × min_{m∈E(t)∩D} baseline_ms(m,t,n_m)/1000]
          / energy_reference_watts
```

这来自原预测的能耗表达式，不将经验功率拟合精度解释为物理下界。缺少必要路径时不删除该执行边，也不强行输出有用界。

**服务与FIFO。** 对每条已冻结请求j，保留其输入ID和合法模型副本状态，令：

```text
s_min[j] = min_{m∈E(task_j)∩D} predicted_service_ms(m,task_j,n_m,input_j)
```

包含原 `n_m/capacity_factor` 归一化。松弛允许每条请求独立选择最快模型，忽略比例路由的耦合、共享队列及等待；实际延迟逐坐标满足 `latency[j] ≥ s_min[j]`。因此同任务mean和经验P95也分别不小于这些最小服务值的mean/P95；仍用原任务权重和参考值归一化。不能用平均服务代替逐请求最小服务来证明P95界。

对 `arrival_j + s_min[j] > horizon` 的请求计入不可避免的T成本；若已超过 `horizon + drain`，则没有该部署的有效路由。实现使用保守数值余量；未知路径的零服务只放松界，不当作已验证性能。全程不计算某条特定路由的孤立FIFO作为跨路由界。

**质量。** v2逐输入质量样本与服务输入顺序由原gate绑定。每任务每条请求j的质量分子/分母为非负 `a[j,m]` 和正数 `b[j,m]`，则：

```text
Q_task_floor = Σ_j min_m a[j,m] / Σ_j max_m b[j,m]
```

任一真实路由的分子不小于分子下界、分母不大于分母上界，故比值安全；不同项取到的模型无需一致。该松弛覆盖CER/WER大于1和模型间分母不同的情况。再按原任务权重求和。不能验证逐输入质量的分支使用0松弛；缺质量仍由完整预测器返回UNKNOWN，不会因松弛而被当作有效方案。

**组合。** 各项乐观选择可以彼此不相容，仍然分别是所有可行路由的成本下界，故相加有效：

```text
UB(D) = −Σ_j weight_j × floor_j(D) + numerical_guard
```

nominal的松弛也覆盖原robust目标，因为服务倍数1/1.25/2均不小于1，robust取最坏效用。多条声明训练轨迹则按原权重聚合，不合并轨迹。实现的guard为 `1e−8 × max(1, |cost_floor|)`，提高上界以避免舍入导致过度删除。

只有 `UB(D) ≤ 当前最佳有效U` 才标记 `SAFE_PRUNED`。无需候选本身已连续优化充分；这与“目前增益很小所以放弃”是两种不同判断。

### 2.3 正确性验证与实际紧度

新增10项CPU测试覆盖独立有限网格/副本状态的完整枚举、非网格比例、robust比较、零流量驻留、门禁拒绝、UNKNOWN、预算和所有调度分支。原多候选8项一并通过后才启动正式消融。有限检查验证实现，数学证明承担整个连续域的适用性；没有宣称穷举了所有实数。

另外复核旧正式数据中7760条有有效用的预测事件记录（包含缓存命中），全部满足新增UB，没有新完整预测。历史audit记录了每个配置/部署的未舍入上界。

trace43的五候选上界如下：

| 索引/部署 | 网格U | 连续域UB | 相对第一名−0.306854442的判断 |
|---|---:|---:|---|
| 0，G+I+O | −0.306854442 | −0.291626275 | 尚不能排除 |
| 1，G+I+O+A | −0.333721784 | −0.329456337 | 安全删除 |
| 2，T+I+O+A | −0.342984773 | −0.341583814 | 安全删除 |
| 3，G+I+O+T | −0.344805975 | −0.315574115 | 安全删除 |
| 4，G+O+A | −0.388781899 | −0.359297721 | 安全删除 |

Text已有正收益并不与删除矛盾：其网格差距0.037952，而这个乐观松弛最多允许约0.029232的改善，仍不足以追平第一名。**本轮现在能够排除这个固定Text部署反超第一名的可能性，前提是当前冻结预测器成立。** 不能扩展为排除了未生成部署、所有工作负载或真实GPU上的可能性；原第一名部署自身的连续域UB仍高于其已观察U，尚无连续全局最优证书。

## 3. 两阶段候选预算原型

实现：[adaptive_candidate.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/continuous/adaptive_candidate.py)。显式使用 `AdaptiveCandidateRefinementOptimizer`，不导出为旧默认planner。

### 3.1 共享预算与初筛

总时间从预算对象创建开始，包含问题/会话建立、所有候选生成、界计算、初筛、重启及搜索内记录。每次Joint残余求解拥有独立反馈；只共享全部署排除及严格相同完整预测输入的缓存。候选始终根据完整U选择，保留原不受限Joint初始解和所有已观察有效incumbent。

生成阶段最多使用setup之后剩余总时间的50%，不续期全局截止时间。默认效用优先，支持原多样性选择配置；正式实验固定效用优先，以隔离预算策略。

安全检查后，对有自由度的每候选进行少量探测：原合法网格路由；一次只改变一个任务的两模型50/50中点（其他行保留网格路由）；合法纯路由端点；剩余槽位用未量化的Sobol点。这是针对上一轮有利子面搜索不足的诊断而增加的统一初筛，不替换原DE。相同探测点也用于重启均匀对照R和静态优先对照H。

最低初筛预算默认8次路由尝试、目标最多16次，初筛比例25%。比例为分配目标，最低配额可优先于比例目标；绝对总预算始终优先。按剩余预测/外层预算和待初筛候选数共同决定实际配额；预算不足时明确记录未达最低值，时间与调用上限优先。这里最低值是分配配额，不保证8次缓存未命中；命中和零维域不能人为制造新调用。

零维部署只有一条合法路由，直接复用候选的完整预测并记录 `SINGLE_ROUTE_EXHAUSTED`，不伪造连续搜索收益。比例为零仍保留模型驻留；模型集合从不因无流量自动改变。

### 3.2 自适应追加与状态

默认每次追加最多48次路由尝试，每候选累计追加上限112次。最多16次探测加112次追加，使其潜在总配额与旧均匀方法每候选128次上限可比；实际探测配额可因全局预算低于16。每段最多3秒，且受原绝对截止时间约束。

普通轮次使用可解释启发式评分：

```text
priority(D) = current_best_U(D) + 2 × observed_gain(D)
              + 0.02 / [1 + attempted_routes(D)/max(1,dimension(D))]
```

它鼓励已有有效方案和实际收益，同时保留对低投入候选的探索。**它不是UB，没有进入任何安全删除条件。** 每4轮安排一次按维度归一化投入最少的候选探索；已停滞候选在仍有活跃候选时最多获一次探索恢复。

连续2次追加无改善，则标记 `PAUSED_STAGNATION`。这只是启发式暂停，可能漏掉未来收益；如果全部候选都暂停，可以提前停止而不耗尽共同预算。达到单候选追加上限则记录 `EXTRA_ALLOCATION_CAP`。安全删除单独标记 `SAFE_PRUNED`，有界与当时有效incumbent。若全局时间/调用先用尽，尚为ACTIVE的候选属于预算暂停，汇总表另列 `budget_paused=true`，不是安全删除。

### 3.3 重启与精确缓存

使用原 `DERoutingOptimizer`，追加时**重新启动DE**，以该候选历史最优合法路由warm start，按 `seed + 1009×round` 生成确定性重启种子。没有修改SciPy内部种群，没有把重启称为继续种群。

每候选评价器及历史保留；所有阶段共享只对完整输入完全一致才命中的预测缓存，键含问题/目标、oracle版本、部署、精确路由和轨迹身份。零比例和闭simplex端点均保留，没有粗舍入或FIFO映射近似缓存。

原DE每次重新做端点/子面初始化，重启会消耗外层尝试、查缓存和Python时间。缓存命中不计新完整预测，但计相应搜索成本。报表同时保留路由尝试数和已完成路由数；若中途因预测预算异常停止，最后一次已开始的尝试可能没有完整结果。原DE本身保持不变。

## 4. 公平消融与实验结果

正确性通过后正式完成 **138次** 运行：core84、同外层预算24、独立轨迹18、正式k1补充12。每组seed11/29加独立seed47；先导12次单独保留，不混入正式中位数。k=1/3/5已覆盖，未出现提高系统最佳U的证据，因此未扩展k8。

A为原Joint；B为原多候选包装器k1；C为原多候选均匀DE；D为安全界后均匀运行原DE。E/F分别是不带/带安全界的自适应原型。另有R：与E相同探测点、分块重启与缓存，但round-robin均匀追加；H：同样的探测/重启，按初筛U静态优先。**R/E更适合隔离调度规则，C/E是完整方法的操作成本比较，不能把共同新增探测点的效果全归于自适应分配。**

共同预算轴：

| 对照轴 | 总秒上限 | 完整预测上限 | 路由尝试上限 |
|---|---:|---:|---:|
| predictions | 60 | 160 | 512 |
| wall | 8 | 1000 | 512 |
| outer | 60 | 1000 | 128 |

候选生成始终计费，各方法预算上限相同；早停方法无需耗尽。三种轴使用相同预测器/任务/配置，变化的是算法预算。以下各表为3种子的中位数，完整范围与逐种子值在CSV。

### 4.1 相同完整预测预算

trace43所有方法最终U均为 **−0.306854441615**，均保留原Joint的部署和路由。

| 方法 | k | 完整预测 | MILP | 路由尝试 | 总秒 | 达到Joint秒 |
|---|---:|---:|---:|---:|---:|---:|
| A 原Joint | 1 | 1 | 2 | 0 | 0.798 | 0.795 |
| B Top-1 DE | 1 | 128 | 2 | 128 | 9.186 | 0.911 |
| C 均匀DE | 3 | 108 | 9 | 108 | 7.388 | 0.782 |
| C 均匀DE | 5 | 160 | 19 | 161 | 10.417 | 0.730 |
| D 安全+均匀DE | 3 | 132 | 9 | 128 | 9.022 | 0.800 |
| D 安全+均匀DE | 5 | 137 | 19 | 128 | 9.204 | 0.792 |
| R 同探测/重启均匀对照 | 3 | 148 | 9 | 256 | 12.503 | 0.791 |
| R 同探测/重启均匀对照 | 5 | 153 | 19 | 484 | 18.683 | 0.798 |
| H 静态初筛优先 | 3 | 146 | 9 | 256 | 12.280 | 0.792 |
| H 静态初筛优先 | 5 | 155 | 19 | 484 | 18.702 | 0.799 |
| E 自适应 | 3 | 141 | 9 | 195 | 10.995 | 0.794 |
| E 自适应 | 5 | 154 | 19 | 232 | 12.100 | 0.787 |
| F 安全+自适应 | 3 | 93 | 9 | 112 | 7.294 | 0.856 |
| F 安全+自适应 | 5 | 98 | 19 | 112 | 7.785 | 0.785 |

k5时F从C的160次完整预测降至98次，减少 **38.75%**，中位总时间10.417→7.785秒，最终方案相同。k3时F为93次，较C的108次减少13.89%。

这不是自适应单独稳定有效的证明：k3时E用了141次，**高于**C的108次；D也用了132次。删除候选后，均匀方法可能把更多额度重新给仍活跃的赢家，从而增加调用。k5时E比C少少量预测（154 vs 160），但总时间12.100秒仍比C慢。

R/H的大量外层尝试是重启端点/子面初始化和缓存检查的真实成本，不计为新增完整预测。本轮E相比同重启对照R减少了这些重复尝试和耗时，但这不能掩盖其相对原C的成本限制。MILP成本在k5的多候选方法间相同；筛选发生在生成之后，不能省掉已经支付的候选生成。

### 4.2 相同8秒预算

| 方法 | k | 完整预测 | MILP | 路由尝试 | 总秒 | 达到Joint秒 |
|---|---:|---:|---:|---:|---:|---:|
| A 原Joint | 1 | 1 | 2 | 0 | 0.807 | 0.803 |
| B Top-1 DE | 1 | 120 | 2 | 120 | 8.018 | 0.804 |
| C 均匀DE | 3 | 74 | 9 | 73 | 5.754 | 0.877 |
| C 均匀DE | 5 | 115 | 19 | 114 | 8.007 | 0.776 |
| D 安全+均匀DE | 3 | 116 | 9 | 112 | 8.015 | 0.791 |
| D 安全+均匀DE | 5 | 108 | 19 | 99 | 8.069 | 1.021 |
| R 同探测/重启均匀对照 | 3 | 101 | 9 | 128 | 8.005 | 0.800 |
| R 同探测/重启均匀对照 | 5 | 113 | 19 | 126 | 8.008 | 0.749 |
| H 静态初筛优先 | 3 | 109 | 9 | 139 | 8.011 | 0.792 |
| H 静态初筛优先 | 5 | 109 | 19 | 124 | 8.022 | 0.794 |
| E 自适应 | 3 | 106 | 9 | 136 | 8.004 | 0.810 |
| E 自适应 | 5 | 113 | 19 | 126 | 8.016 | 0.792 |
| F 安全+自适应 | 3 | 93 | 9 | 112 | 7.693 | 0.975 |
| F 安全+自适应 | 5 | 98 | 19 | 112 | 7.423 | 0.799 |

一个明确反例是k3：C只用74次预测、5.754秒，而F用93次、7.693秒，效用相同。原C为后续候选预留份额，零维候选又不需要搜索，因此可能未用完总预算就结束。F删除候选后精修赢家至停滞，本轮这部分投入没有收益。因此“组合方法总比均匀DE省调用/更快”不成立。

8秒内的实际评价数受主机负载、配额与重启成本影响；轻微超时保留真实时间。部分ACTIVE候选仅因全局时间结束而暂停，不能称为安全删除或连续求解完成。

### 4.3 相同外层预算与局部收益阈值

| 方法 | k | 完整预测 | MILP | 路由尝试 | 总秒 | 达到Joint秒 |
|---|---:|---:|---:|---:|---:|---:|
| A 原Joint | 1 | 1 | 2 | 0 | 0.783 | 0.780 |
| B Top-1 DE | 1 | 128 | 2 | 128 | 8.402 | 0.736 |
| C 均匀DE | 5 | 131 | 19 | 128 | 8.687 | 0.790 |
| D 安全+均匀DE | 5 | 137 | 19 | 128 | 9.086 | 0.794 |
| R 同探测/重启均匀对照 | 5 | 117 | 19 | 128 | 8.209 | 0.782 |
| H 静态初筛优先 | 5 | 115 | 19 | 128 | 8.272 | 0.790 |
| E 自适应 | 5 | 115 | 19 | 128 | 8.153 | 0.791 |
| F 安全+自适应 | 5 | 98 | 19 | 112 | 7.386 | 0.788 |

外层尝试不是完整计算成本：缓存命中、候选生成和MILP数不同。`runs.csv`另列共同64/128次尝试前的最佳U、已用完整预测及是否实际达到该前缀；提前停止后不伪造后续评价。在已到达的共同前缀上，系统U仍等于Joint初始解，没有新增效用优势。

更高系统阈值统一为“该轨迹Joint U+0.001”。全部正式运行均未达到，时间/调用记录为null，不写0，也不生成虚假的达标速度。当前Joint阈值则都可在第一条有效预测达到；其优势来自Joint初始化，不来自多候选或新筛选。

另外记录每个部署相对其自身网格U提升0.01的局部阈值。它只用于诊断候选内探索，不能冒充系统第一名提升；安全删除的Text部署未搜索该局部阈值，是按系统目标正确取舍。

| 方法，trace43/k5 | 局部+0.01达标种子 | 候选内首次改善前新预测 | 达局部阈值时总预测 | 达局部阈值时总秒 |
|---|---:|---:|---:|---:|
| C 均匀DE | 3/3 | 13.000 | 83.000 | 5.695 |
| R 同探测/重启均匀对照 | 3/3 | 3.000 | 27.000 | 2.457 |
| H 静态初筛优先 | 3/3 | 3.000 | 27.000 | 2.403 |
| E 自适应 | 3/3 | 3.000 | 27.000 | 2.414 |
| F 安全+自适应 | 0/3 | 未达到/未搜索 | 未达到/未搜索 | 未达到/未搜索 |

这张表的候选内与整个优化运行的调用数采用不同起点，分别标明；局部达标不代表超过系统第一名。

### 4.4 k、独立种子与冻结轨迹

正式k1结果中，A/B来自主批次，E/F来自单独正式补充，参数一致；各有三个种子。k3/5的F仍生成全部候选，之后才安全删除，故相同效用下增加k的主要代价是生成开销，不能认为候选越多越好。


相同预测上限，k1：

| 方法 | k | 完整预测 | MILP | 路由尝试 | 总秒 | 达到Joint秒 |
|---|---:|---:|---:|---:|---:|---:|
| A 原Joint | 1 | 1 | 2 | 0 | 0.798 | 0.795 |
| B Top-1 DE | 1 | 128 | 2 | 128 | 9.186 | 0.911 |

E 自适应：89次预测、6.701秒，U=-0.306854441615。

F 安全+自适应：89次预测、7.004秒，U=-0.306854441615。

相同时间上限，k1：

| 方法 | k | 完整预测 | MILP | 路由尝试 | 总秒 | 达到Joint秒 |
|---|---:|---:|---:|---:|---:|---:|
| A 原Joint | 1 | 1 | 2 | 0 | 0.807 | 0.803 |
| B Top-1 DE | 1 | 120 | 2 | 120 | 8.018 | 0.804 |

E 自适应：89次预测、6.889秒，U=-0.306854441615。

F 安全+自适应：89次预测、6.949秒，U=-0.306854441615。


trace45是重新冻结的一条轨迹，方法在该轨迹上分别重新优化；不是对同一学习路由的泛化检查。所有方法最终U均为 **−0.298403112797**。

| 方法 | k | 完整预测 | MILP | 路由尝试 | 总秒 | 达到Joint秒 |
|---|---:|---:|---:|---:|---:|---:|
| A 原Joint | 1 | 1 | 2 | 0 | 0.790 | 0.788 |
| B Top-1 DE | 1 | 128 | 2 | 128 | 8.577 | 0.866 |
| C 均匀DE | 5 | 160 | 17 | 160 | 10.595 | 0.785 |
| D 安全+均匀DE | 5 | 136 | 17 | 128 | 9.185 | 0.787 |
| E 自适应 | 5 | 157 | 17 | 236 | 12.694 | 0.795 |
| F 安全+自适应 | 5 | 97 | 17 | 112 | 7.485 | 0.731 |

三个优化种子均未得到高于该轨迹原Joint的系统U。完整波动范围保留在comparison/runs CSV；本轮不能据三种子宣称统计显著性或对任意模型池稳定占优。

![不同预算下的效用、调用和时间](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-final-report-20261102/budget-comparison.png)

![达到相同效用的时间与预测次数](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-final-report-20261102/anytime.png)

![各候选的调用分配与局部收益](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-final-report-20261102/candidate-allocation.png)

![k与成本及最佳效用](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-final-report-20261102/candidate-count.png)

![不同松弛项的上界紧度；只有完整连续域UB用于删除](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-final-report-20261102/safe-bounds.png)

## 5. 研究判断与失败案例

**候选筛选有价值，但仅限证明支持的范围。** 本轮的逐请求松弛比单独驻留显存/能耗界更紧，能安全删去四个靠后候选，且不要求连续搜索充分。单独的显存/能耗上界仍较宽；逐项数值见stages CSV的floors及UB分层，不能只因Text显存大就直接按启发式删掉。

**自适应单独未显示普遍优势。** k3存在增加调用的反例；R/E对照显示它能减少重启造成的重复尝试，但与原C相比，预测调用和时间没有一致改善。不要将重启缓存带来的少量调用减少全部归因于更好的候选选择。

**组合方法在部分预算设置下能以更少预测返回相同方案。** k5/160次预测上限的减少明确，独立轨迹亦复验；但8秒/k3反例说明收益受预算与早停规则影响。它没有提高系统U，而原Joint只用1次完整预测，仍是当前最省的默认方案。

**预算不充分和启发式遗漏风险仍存在。** 一次只改变一个任务的初筛能较早触及Text的有利子面；多维重启仍会反复消耗端点/面初始化，少量追加可能到不了充分的随机搜索。`PAUSED_STAGNATION`不等于该部署不可能再改善。原DE未恢复种群，R/H的重复初始化开销已经体现在路线尝试和总时间中。

安全筛掉Text后不再重现其“第四升第二”是有证明的系统目标取舍：该固定部署的连续域UB低于当前第一名。其他候选因启发式暂停或墙钟结束而未充分搜索，则没有这种保证。所有方法保留原有效incumbent，所以未观察到全局方案退化，也不能由此反证未搜索区域不存在更优点。

目前未发现候选集合缺少某个部署是主要瓶颈的证据；也没有新增模型/副本切换提高系统U的观测。**本轮不开展部署邻域搜索，不开发GP-BO/MADS，不替换默认优化器。** 不能把“未开展”填成邻域零收益。

下一步优先：保留低K Joint；在更大且门禁支持的域检验连续域安全界的适用性和构建成本，再研究将其用于候选选择/生成，减少为最终被删部署支付的重复MILP开销。经验性候选优先级与停止策略应继续以R/E和独立轨迹消融验证。若后续证据显示原候选集合成了主要限制，再启动有限一层邻域；当前不足以支持扩大复杂连续优化器。

正式138次运行共记录 **14938次完整预测**，没有系统U高于Joint的运行。独立审计重建14938条逐运行部署/路由完整FIFO映射，核验15226条有有效用的事件；安全删除120个候选出现记录，未知事件0，未认证候选0。这些计数包含跨运行重复输入，不是全局唯一模型结构或独立样本数。

CPU全套 **299项全部通过**（含新增10项）；正式实验前的新增/多候选18项组合检查也已通过。完整日志与最终源码见交付清单。

## 6. 证据范围、数据与复现

本轮只使用原验证域内的五模型池、单副本状态、四任务等到达率、冻结Poisson轨迹43/45，1800秒到达与300秒排空。显存49140MiB，Q/T/L/P/M/E各1/6，延迟参考仍为1000/250/10000/3000ms。没有切换到工作区另行讨论的一秒参考，也没有改变质量或能耗预测。

当前模式的服务是逐输入独立服务中位数乘冻结全局校准系数，再进入共享FIFO；缺少独立的随路由执行活动改变的动态GPU共驻服务项。结论只覆盖其FIFO、输入/质量分配、驻留资源与期望能耗效果，不证明真实动态干扰机制。更改负载/任务被既有门禁拒绝的范围见[上轮范围审计](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-core-20261101/scope-audit.json)。本轮没有绕过门禁或新增GPU测量。

正式方法共用同一配置/副本域；setup、全部候选生成、界、初筛、DE重启、精确缓存和搜索内记录均计入总时间。图表和完整FIFO映射导出在优化计时结束后统一完成。主机共享、Python深拷贝和冷热启动影响绝对秒数；不把包装器某次第一解更快解释成加速了原Joint。预算是共同上限，不强迫提前停止的方法继续耗尽。正在执行的单次操作不被强制抢占，轻微超时据实报告。

每次完整预测包含三个原服务敏感性分支，leaf调用数为3倍；原nominal效用定义不变。每次master调用单独计MILP次数。Joint的1次预测和通常2次MILP来自一次完整顶层求解，二者不要混用。

### 原始数据与审计

- [历史候选诊断](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-history-v2-20261102/audit.json)：132条候选记录、原始预测事件的界核验；没有新完整预测。
- [12次先导试验](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-pilot-20261102/summary.json)：k1/3/5及四个主策略，先导追加上限96；正式冻结为112，使最多16次初筛加追加额度与128次旧配额可比。先导不混入正式中位数。
- [正式core](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-core-20261102/summary.json)：trace43，同预测/同时间预算，三个优化种子，k3/5及Joint/Top1/重启对照。
- [同外层预算](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-outer-20261102/summary.json)：128次路由尝试上限的k5对照。
- [独立轨迹复验](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-stability-20261102/summary.json)：trace45，三个优化种子，k5主方法。各轨迹分别重新优化，不是固定一条路由的holdout泛化评估。
- [正式k1](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-k1-20261102/summary.json)：正式参数下的E/F，两种成本预算、三个种子。
- [完整分析JSON](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-final-report-20261102/analysis.json)、[逐运行CSV](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-final-report-20261102/runs.csv)、[逐候选CSV](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-final-report-20261102/stages.csv)、[候选决策CSV](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-final-report-20261102/decisions.csv)、[各次重启CSV](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-final-report-20261102/segments.csv)。
- [正式独立审计](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-final-report-20261102/audit.json)、[源码/日志/校验清单](/root/autodl-tmp/generalist-specialist-project/artifacts/adaptive-candidate-closeout-20261102/completion.json)。

每批保留plan、逐case配置、全部 `gsdeploy` 源码哈希及快照、逐run完整JSON。原始JSON包含候选完整预测、各次残余Joint结果、评价历史、缓存/预算、界及其成本项、调度动作、最终方案和逐请求映射。CSV和图表不替代原始记录。

历史诊断第一次读取大JSON时以SIGKILL/137退出，原因未独立确认；没有新预测。失败记录在 `artifacts/adaptive-candidate-history-20261102/failure.json`。改为读取优化账本时跳过报告用的大映射块，原始文件保持完整，v2重跑成功。正式映射审计则逐条流式读取并重建，不因为降低报告内存而省略正确性核验。

原 `gsdeploy/milp.py`、`optimizer.py`、`prediction.py`、`routing.py`、`quality.py` 等受保护文件与HEAD一致；历史manifest中的既有优化源码也逐项匹配。新增文件导致完整manifest增加条目，不等于修改了原预测器。各正式批次使用同一最终新增源码；先导保留自身快照与参数。

### 显式API

```python
from gsdeploy import SystemConfig
from gsdeploy.continuous.adaptive_candidate import AdaptiveCandidateRefinementOptimizer

config = SystemConfig.load(
    "artifacts/adaptive-candidate-core-20261102/p5-n1-K1-trace43-config.json"
)
result = AdaptiveCandidateRefinementOptimizer(
    config, {m: [1] for m in config.catalog["models"]},
    k=5, safe_filter=True, policy="adaptive", seed=11,
    seconds=60, max_predictions=160, max_routes=512,
    min_screen_routes=8, screening_fraction=.25,
    screen_routes=16, chunk_routes=48, max_extra_routes=112,
).solve()
# result["winner"] / ["stages"] / ["decisions"] / ["segments"] / ["budget"]
```

### 运行命令

从 `/root/autodl-tmp/generalist-specialist-project` 运行，输出目录必须不存在。正式复现依赖原本地冻结画像和输入；没有用模拟数据替代它们。测试fixture明确为CPU合成验证问题，不纳入真实场景性能结论。

```bash
python -m unittest tests.test_adaptive_candidate tests.test_multi_candidate -v
python scripts/run_cpu_tests.py

python experiments/analyze_candidate_potential.py \
  --sources artifacts/multi-candidate-core-20261101 artifacts/multi-candidate-stability-20261102 \
  --output artifacts/adaptive-history-reproduce

python experiments/adaptive_candidate_refinement.py \
  --output artifacts/adaptive-core-reproduce --axes predictions wall \
  --traces 43 --seeds 11 29 47 --ks 3 5

python experiments/adaptive_candidate_refinement.py \
  --output artifacts/adaptive-outer-reproduce --axes outer \
  --traces 43 --seeds 11 29 47 --ks 5

python experiments/adaptive_candidate_refinement.py \
  --output artifacts/adaptive-stability-reproduce --axes predictions \
  --traces 45 --seeds 11 29 47 --ks 5 \
  --methods joint top1 uniform safe_uniform adaptive safe_adaptive

python experiments/adaptive_candidate_refinement.py \
  --output artifacts/adaptive-k1-reproduce --axes predictions wall \
  --traces 43 --seeds 11 29 47 --ks 1 --methods adaptive safe_adaptive

python scripts/report_adaptive_candidate.py \
  --sources artifacts/adaptive-candidate-core-20261102 artifacts/adaptive-candidate-outer-20261102 \
    artifacts/adaptive-candidate-stability-20261102 artifacts/adaptive-candidate-k1-20261102 \
  --output artifacts/adaptive-report-reproduce

python scripts/audit_adaptive_candidate.py \
  --sources artifacts/adaptive-candidate-core-20261102 artifacts/adaptive-candidate-outer-20261102 \
    artifacts/adaptive-candidate-stability-20261102 artifacts/adaptive-candidate-k1-20261102 \
  --output artifacts/adaptive-audit-reproduce.json
```

固定输入、种子和探测/重启规则可复验；触及墙钟预算时，主机负载可能改变实际评价数量，不保证不同机器的耗时或路径逐位一致。三个种子只支持本轮复现判断，不宣称统计显著性或任意场景优势。
