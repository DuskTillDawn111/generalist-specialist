# 激活感知 BO–MADS 双层算法：实现与 CPU 核对

2026-11-04。当前目标以用户提供的[算法原文](activation_bilevel_target_20261104.md)为准，SHA-256 为 `a64b980f126368755fc1d9d769b0b9a4c4fe20bd8f45830059258082b2a733fe`。本轮将此前 C1-Lite 的候选池和渐进内层用于完整目标算法，新增激活子空间调度、BO、网格自适应局部轮询及协同反馈。

原文在本仓库按原始字节保留。本轮新指令更新了旧任务书“暂不开发 BO/MADS、暂缓 Joint 初始化组合分支”的实现顺序；这不表示旧的真实预测或收益 Gate 已通过。

## 本轮改变与架构

`max_r V(r)`，`V(r)=max_D U(D,r)` 的定义保持。外层搜索路由及正流量激活组合，内层固定路由搜索合法副本与可选驻留模型；完整预测仍由现有共驻阶段/FIFO/生命周期与六项效用接口执行，不引入第二套排队语义，也没有重训真实共驻规律。

| 部分 | 本轮实现 |
|---|---|
| 激活组合 | `ActivationSpaces` 按模型集合构造各任务闭 simplex，显式探索 full、可覆盖任务的单模型、纯专用组合及增删/交换邻域 |
| BO 全局探索 | 认证观测上的 RBF Gaussian Process + Expected Improvement；显式保留新激活空间和随机探索配额 |
| MADS 局部精修 | 独立局部网格、变化的正生成方向、完整轮询状态及成功/失败/未解决三类更新 |
| 部署结构共享 | 复用 `DeploymentCandidatePool`，新路由重新预测，不跨路由复用 score |
| 按需固定路由 MILP | 复用 `ProgressiveFixedRoutingSolver`，保留每路由 ledger、LB/UB/UNKNOWN 与追加预算 |
| 不完整观测 | 与精确 GP 观测分开存储，调度竞争路由细化，不把 LB 写成 V |
| 可选 Joint 初值 | 低 K 原 Joint 搜索；计入同一总预算，后续继续固定路由 MILP 与动态扩池 |
| 旧算法 | Joint K1/K2、原 C1 和 C1-Lite 保留独立对照入口，未替换默认系统规划器 |

主入口是 `gsdeploy.continuous.ActivationAwareBilevelOptimizer`。源文件为 [activation_bilevel.py](../gsdeploy/continuous/activation_bilevel.py)、[activation_search.py](../gsdeploy/continuous/activation_search.py)、[bayesian_search.py](../gsdeploy/continuous/bayesian_search.py) 和 [mesh_search.py](../gsdeploy/continuous/mesh_search.py)。

## 激活与 BO 的具体语义

每个激活集合对应一个闭路由子空间：集合外份额精确为 0，集合内按任务能力分配。中心点与 Dirichlet 内部点会使用该集合；边界点允许实际激活减少，并登记实际正流量模型。模型在另一任务仍有流量时不会误判为停用；正流量要求合法副本，零流量不强制卸载已驻留模型。

子空间按需发现，默认最多 32 个；输出发现列表、维数、访问数及容量截断。不是预展开全部路由网格，也不声称遍历了所有覆盖组合。每次全局提案还包含任务内的精确 0/1 边界，避免只处理整个模型停用而漏掉任务边界。

GP 的输入包含完整份额与实际模型激活指示；目标只取已认证固定路由的可行 U（认证带现有内层数值容差）。无解/UNKNOWN/未认证下界不进入回归，最多保留 128 个训练观测，并保留最好认证点。核长度尺度与数值 jitter 显式固定，未做大规模超参数选择。

部分结果保留原始 LB/UB、pool_version 与状态，周期性给仍可能改善全局方案的旧路由追加内层预算。缺 UB 时依靠探索调度，不宣称安全剪枝。master 已穷尽但证据 UNKNOWN 的身份不会通过重复预算变成已认证；需要外部证据更新后建立新问题身份。

这里选用了原文允许的“追加内层求解”方案，没有实现区间 GP 或把求解误差伪装成普通噪声。GP 的后验方差只用于提案，不能当作数学效用上界。

## MADS 实现边界

局部 poll 在子空间的 stick 坐标中执行。层级为 `l` 时 mesh 为 `4^-l`、frame 为 `2^-l`；对随机正交矩阵按 frame/mesh 比例整数化，检查满秩后取正负列方向，额外加入坐标方向以覆盖盒角的可行轮询。越界候选直接拒绝，不通过 clipping 假造网格方向。随网格收细，方向分辨率提高，区别于仅固定坐标方向的 pattern search。

支持跨多次调度保留未完成 poll。只有完成且证据充分的无改善 poll 才缩小网格；已证明改善则扩大 frame。未认证 LB 的改善可以启发式移动中心，但记录为 heuristic，不当作已证明成功；区间重叠先追加有限内层预算，仍未解决则不判定失败收缩。

这是项目内实现的随机方向 MADS 风格 mesh-adaptive poll，没有调用 NOMAD，也未声称复现某个命名的 OrthoMADS/LTMADS 算法。BO 重启局部中心、有限方向/预算、最大层级、非精确内层与激活边界使本实现**不输出经典 MADS 驻点或连续全局最优证书**。未来若需要特定 MADS 实现或理论保证，应单独绑定其方向规则、收敛假设与验证。

## 协同、预算与初始化

BO 和 MADS 共用同一观测账本、oracle、精确预测缓存、部署池及 global Budget。全局/局部调度概率根据历史改善与预测/master 工作量更新，保持 .25–.75 的概率边界；计数驱动的分配使相同计数预算下可复现，wall 时间仍单独计入总预算。

MADS 比较前刷新过期的未认证 parent/trial 池下界；认证值与池结构无关可直接保留。所有刷新、细化、预测和 MILP 计入预算。更好的可行联合方案独立保存，池淘汰及求解退出不会抹去它。

Joint 初始化默认关闭，显式 `joint_seed_k=1` 可启用；默认仅使用剩余时间/预测/master 配额的 20%，剩余预算留给连续搜索。它使用同一 oracle/session/trace 和预测缓存，不承诺原生 MIP warm start。即使在 Joint 后立刻耗尽外层预算，已完整评价的 Joint incumbent 也保留。Joint 初值不会限制后续部署域。

## 软件验证与 CPU 对照

新增 [test_activation_bilevel.py](../tests/test_activation_bilevel.py) 的 **22 项测试全部通过**，覆盖激活/QPS、GP 排除部分值、正 U、整数网格与变化方向、证据不足不收缩、独立固定路由枚举、预算不续期、UNKNOWN、Joint 初值保留、单路由终止、真实 v2 软件接口与确定性。

相关回归在首批 18 项新测试时运行，共 **153 项，35.488 秒，全部通过**；随后新增的 4 项边界测试也包含在最终 22 项运行中（1.616 秒通过）。两轮合计覆盖 **157 个不同测试**，并非声称单次跑了 157 项。回归原始日志见 [tests-regression.log](../artifacts/activation-bilevel-20261104/tests-regression.log)。CLI 另完成一次 CPU smoke。

同一 v2 CPU fixture 的 24 次工程对照：1 任务、2 模型、3 个非空部署、2 个请求；优化种子 11/29/47，底层同一输入/轨迹。统一上限 4 秒、96 次预测、48 次外层评价，各方法可以提前停止。采用独立 session，执行顺序按种子轮换。运行期间有并行 CPU 回归，短 wall 时间仅作工程记录，不作严格速度估计。

| 方法 | U：11 / 29 / 47 | 完整预测：11 / 29 / 47 | master：11 / 29 / 47 |
|---|---|---|---|
| 原 Joint K1 | -3.294667 / -3.294667 / -3.294667 | 4 / 4 / 4 | 5 / 5 / 5 |
| 原 Joint K2 | 同 K1 | 5 / 5 / 5 | 6 / 6 / 6 |
| 原 C1（DE + 完整内层） | -3.282971 / -3.281039 / -3.281400 | 50 / 50 / 50 | 98 / 98 / 98 |
| 前轮 C1-Lite budget | -3.294667 / -3.280012 / -3.280611 | 16 / 14 / 13 | 8 / 7 / 9 |
| 激活 BO+MADS | -3.280892 / -3.279866 / -3.279806 | 30 / 31 / 31 | 27 / 25 / 27 |
| 激活 BO | -3.284368 / -3.280819 / -3.280330 | 40 / 40 / 40 | 29 / 29 / 29 |
| 激活 MADS | -3.279806 / -3.279806 / -3.279806 | 26 / 26 / 26 | 26 / 26 / 26 |
| 激活 BO+MADS，Joint 初始化 | 同无初始化组合 | 30 / 31 / 31 | 31 / 29 / 31 |

组合算法三次运行中实际 BO EI 提案为 10/12/15 次，MADS poll 评价为 19/16/16 次，GP 接收认证路由 24/22/24 个，排除部分路由 4/7/5 个。由日志可确认两种搜索均有执行，而非仅更名。原始 [JSON](../artifacts/activation-bilevel-20261104/reference-v2-cpu/comparison.json)、[CSV](../artifacts/activation-bilevel-20261104/reference-v2-cpu/comparison.csv)、[图](../artifacts/activation-bilevel-20261104/reference-v2-cpu/comparison.png)及[源码摘要](../artifacts/activation-bilevel-20261104/reference-v2-cpu/source-manifest.json)已保存。

这里新组合优于旧 K1/K2 和原 C1 的最终 U，但 MADS 单独也很强；BO 协同、Joint 初始化的额外价值尚未成立。旧 Joint 在这个小例更便宜，不能声称新方法全面优于旧 MILP。各方法初始化/路由序列不同，这是完整方法比较，不是单机制因果实验。

该 fixture 使用分析构造的阶段与 aggregate quality proxy，两个请求还会产生映射台阶；出现内部解不能当作稳定真实分流证据。真实大模型池、路由维数与副本域上的收益仍需后续比较。

## 使用与复现

单次算法入口，不覆盖已有输出目录：

```bash
python experiments/solve_activation_bilevel.py \
  --config configs/paper_fifo_reference_v2_cpu_fixture.json \
  --output artifacts/activation-bilevel-demo \
  --seconds 4 --predictions 96 --routes 48
```

加 `--joint-seed-k 1` 使用可选 Joint 初值；加 `--exact-inner` 要求每个路由认证，否则提前退出。真实配置仍要求合法冻结参考及匹配预测支持；入口不会用合成标量填补 pending 参考。

```bash
python experiments/benchmark_activation_bilevel.py \
  --output artifacts/activation-bilevel-comparison \
  --seconds 4 --predictions 96 --routes 48 --seeds 11 29 47
python -m unittest tests.test_activation_bilevel -v
```

比较脚本仅接受显式 CPU fixture；单次求解入口可接合法真实预测配置，但不自动授予物理预测资格。本轮未启动 GPU 采集，也未改变效用权重、FIFO、显存或 rho 规则。后续 GPU 画像/参考/独立确认仍按[预测资格缺项](c1_lite_hybrid_prediction_qualification.md)推进；它们影响真实结论，未阻止本轮先实现目标算法。

## 随后的 GPU 实测进展（持续更新）

上述“本轮”指算法实现和 CPU 对照阶段。随后已在同型号 RTX 4090 上开展持续采集；实时结果见 [GPU 状态](../artifacts/activation-gpu-20261104/STATUS.md)，原始报告和失败历史见 [实验记录](../artifacts/activation-gpu-20261104/CAMPAIGN.md)。GPU UUID 只记录来源。

四任务孤立参考及两轮完整低负载能耗参考已冻结；动态共驻、FIFO、驻留空闲及单/双 OCR 副本训练已完成。严格年龄模型曾通过一个专用域的训练回放，但独立验证失败。新 schema 的有界年龄候选完整训练回放支持 2760/2760 请求，随后首轮独立测试虽无覆盖缺口，时延和 work 准确性仍失败。训练通过不会解除正式排名的资格要求。

两轮五模型压力探针保留全部专用，再对 text/ASR 做小比例分流，实测出现质量、时延与能耗的权衡，尚无稳定内部路由优势。额外 text/ASR 专用副本对照、观察器资格和预测结构改进仍在推进。效用权重、原显存安全系数和 rho<=0.8 不变；预测/完整 episode 资格尚不支持发布新算法与旧 Joint MILP 的物理排名。


随后单/双OCR有界候选的两轮独立检验均完成且失败，覆盖已完整，时延/work准确性仍未全部合格。单OCR终点时间目标候选逐路径ASR/image/OCR时延达标，但text P95=.5535仍失败。GPU短探针同时证实专用副本改变跨任务干扰，能耗变化不固定为正。新实验阶段模型位于独立experiments入口，保留原PHASE和混合host/transfers/CUDA语义，不读未来token；其物理资格仍等待新观察器及新训练。PEFT包装导致旧观察器遗漏text forward的问题已定位，另建底层观察器和新上下文进行配对检验，不追认原失败。完整结果仍以GPU状态及原始报告为准。

修复后的PEFT底层观察器现已通过四轮GPU配对及跨轮重复门槛，text forward缺口已补齐。对应六轮全新阶段训练正在采集，完整预测资格仍等待拟合、FIFO/能耗回放和新独立检验；原失败均保留。

新一轮GPU结果：PEFT底层观察器六轮训练1872请求已完成，host-call band阶段模型仍被原work/重复支持门槛拒绝；隔离v2修复的是长连续空闲描述域，不是服务准确性。两轮受控串行历史对照也未证明简单前置类别能稳定解释所有band波动。原失败保留。

另一个实质进展是新G2+四S运行域：显式在加载后将ExLlama重构阈值设为1、保留原4bit权重和生成规则，配合严格确定性与PEFT底层观察；四轮656请求的输出/跨轮重复与8路径观测时延门槛全部通过。该运行配置的两轮全池solo、实际质量复评分正在执行，完整共驻/FIFO/能耗资格仍待新域训练及独立检验。若最终使用此运行配置比较新旧算法，二者必须共用它，并重建匹配完整episode参考，不能把后端变化记为优化算法收益。
