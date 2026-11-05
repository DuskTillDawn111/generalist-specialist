# 连续路由 Phase 1 实际结果

全部数值来自 CPU synthetic fixture；不证明真实 GPU 预测精度、质量或部署收益。

Phase 1 工程交付完成：连续域/激活、完整候选 adapter、精确缓存、固定路由枚举、固定路由 MILP、DE、独立 CLI、配对实验与图表。原 Joint MILP/DFS 源码和默认行为没有修改。使用方式见[复现说明](continuous_routing_usage.md)。

最终源码的可移植 CPU 回归：`python scripts/run_cpu_tests.py`，238/238 通过，17.672 秒；其中 35 项为新增连续优化测试。新增测试包括一个非网格内部最优的独立目标，DE 获得比 K2 点更好的值；这是组件正确性证据，不是实测系统收益。

## 端到端配对结果

同一负载、训练轨迹、效用与资源约束；三个优化 seed。以下时间为各算法运行时间中位数，U 为三 seed 中位数。

| 负载 | 算法 | 中位 U | 中位秒数 | 中位完整预测 | 有限网格认证次数 |
| --- | --- | ---: | ---: | ---: | ---: |
| low | de_enumeration | -0.136352359 | 3.421 | 576 | 0/3 |
| low | de_milp | -0.136352359 | 4.305 | 576 | 0/3 |
| low | dfs_K2 | -0.136352359 | 0.419 | 208 | 3/3 |
| low | milp_K2 | -0.136352359 | 14.031 | 208 | 3/3 |
| low | milp_K4 | -0.214355861 | 30.054 | 196 | 0/3 |
| mixed | de_enumeration | -0.145829219 | 4.556 | 576 | 0/3 |
| mixed | de_milp | -0.145829219 | 5.395 | 576 | 0/3 |
| mixed | dfs_K2 | -0.145829219 | 0.468 | 208 | 3/3 |
| mixed | milp_K2 | -0.145829219 | 14.489 | 208 | 3/3 |
| mixed | milp_K4 | -0.235773198 | 30.051 | 193 | 0/3 |
| text_hot | de_enumeration | -0.154548490 | 6.029 | 576 | 0/3 |
| text_hot | de_milp | -0.154548490 | 6.902 | 576 | 0/3 |
| text_hot | dfs_K2 | -0.154548490 | 0.523 | 208 | 3/3 |
| text_hot | milp_K2 | -0.154548490 | 14.401 | 208 | 3/3 |
| text_hot | milp_K4 | -0.154548490 | 30.052 | 232 | 0/3 |

## 固定路由内层

独立枚举与 MILP 的 9 对固定路由全部同值、同认证。MILP/枚举耗时比中位数 1.212；完整预测次数全部相同：True。

局部连续扰动诊断中，1 类实际请求分配对应 41 类效用。相同请求映射不能直接作为全预测缓存等价证明。

## heldout 与停止状态

实际终止状态：`{"OUTER_LIMIT": 18, "OPTIMAL_MILP_FINITE_DOMAIN": 9, "MILP_BUDGET_LIMIT": 9, "OPTIMAL_ENUMERATED_DOMAIN": 9}`。

每个训练赢家在两条 validation 与两条 test 轨迹上冻结 D/r 后评价；完整数值和独立成本位于对应 raw JSON 的 holdout 字段，未用于挑选赢家。

旧 Joint MILP 的 wall-time 曲线只展示终点；原实现没有每次改进的时间戳，未补造过程时间。DFS state budget 是更严格的额外限制，需同时阅读原始停止原因。

## 阶段判断

1. 两种内层的正确性链已建立；一般黑盒下，MILP 的身份反馈并不自动减少预测调用。
2. 连续搜索结果和网格结果按实际预算报告；高于某个网格结果也不表示连续全局最优。
3. 保留枚举作为当前小域推荐内层，MILP 作为可运行的对照及未来安全界入口。
4. 暂不推进 BO/MADS 或混合算法；先确认完整真实 oracle 的支持域，以及能够降低内层成本的有效结构。

## RQ1–5 与推进决定

- **RQ1 内层 MILP 是否优于枚举？** 本 fixture 中否：9 对同值、同调用数，MILP 耗时中位数为枚举的 1.212 倍。另有独立反例测试证明，在显式安全 state floor 下 MILP 可以提前停止；不能把该测试推广为现有共驻模型已有强界。
- **RQ2 连续 DE 是否比旧网格更优？** 三负载、三 seed 均与 K2/DFS 同 U；DE 每次 576 个预测 bundle，K2/DFS 为 208。DE 比本次 K2 Joint MILP 快，但 DFS 更快、调用更少。本轮不支持“连续双层架构整体更高效”的结论。K4 的低/混合负载较差结果来自 30 秒预算中止，不是已证明的 K4 最优值。
- **RQ3 激活边界是否重要？** 是，本次所有训练赢家都在边界：低负载只用 `s_shared#0`；混合与文本热点用 `s_shared#0+s_text#0`。已采样严格内部点的最好 U 分别为 −0.264857480、−0.268081720、−0.260746277，低于对应边界赢家。这是样本比较，不是整个内部区域的界。
- **RQ4 FIFO 阶梯是否妨碍搜索？** 局部 41 次扰动落在同一个请求分配类别，但有 41 个不同 U，说明负载相关共驻/质量/能耗项仍会变化。当前结果不足以断定阶梯对所有目标的影响；已避免按 assignment 合并预测缓存。
- **RQ5 开销在哪里？** 一次路由的多个部署完整评价是主要重复工作；此次内层没有利用非可分模型的强界，MILP 又增加 Master 开销。外层/预测/MILP 时间均在 JSON 的 budget 中独立记录。

**Gate 1：工程正确性与小域算法可行性通过；性能优势未建立。** 当前不进入 Phase 2/3，也不替换原优化器。依计划书“如果主要瓶颈在内层，应优先分析内层，而不是立即增加外层算法”，保留两种可运行内层，以枚举作为小域推荐值。

覆盖限制：此次为 2 任务小域、3 个合成模型、有限副本及 Poisson 到达；维度诊断扩展到 1–3 维，不是 ASR/OCR/七模型真实画像实验或渐近复杂度证明。没有重新设计、训练或验证 GPU/FIFO 预测器。

主对比总 wall time（含循环、验证与保存）：499.577 秒。主实现摘要：`6b841dcce61ecf19f80d48311e6e00562df290f5ef3ad4a0e87e86cc33b99bf3`。

原始结果：`/root/autodl-tmp/generalist-specialist-project/artifacts/continuous-routing-phase1-v1-20261101`；固定路由/平台诊断：`/root/autodl-tmp/generalist-specialist-project/artifacts/continuous-routing-inner-cost-v1-20261101`。

最终防御性加固增加了公开候选结构/轨迹检查、实际 MILP 调用计数、witness 检查、严格界停止与来源绑定；最终源码复核的 6 组训练赢家与首轮部署/路由/U 逐项相同，见 `final-source-parity.json`。首轮和最终运行各保存 130 个源码/配置文件的 `source-snapshot/` 与逐文件 manifest；首轮连续模块摘要经逐字节内容核对与原摘要一致。完整冻结日历、配置和域保存在各结果目录的 `problems.json`。

图表位于 `artifacts/continuous-routing-phase1-figures-v2-20261101/`，包括 U 对时间/预测调用/外层评价、内层成本、维度成本、负载路由和种子分布。计时对比为实际串行运行记录，没有新增硬件收益声明。

## 额外多轨迹与维度检查

`/root/autodl-tmp/generalist-specialist-project/artifacts/continuous-routing-multitrace-v1-20261101`：6 次运行，总 wall 32.718 秒。
- mixed_multitrace / de_enumeration / seed 11：U=-0.15491221962577995，calls=640，status=OUTER_LIMIT。
- mixed_multitrace / de_milp / seed 11：U=-0.15491221962577995，calls=640，status=OUTER_LIMIT。
- mixed_multitrace / de_enumeration / seed 29：U=-0.15491221962577995，calls=640，status=OUTER_LIMIT。
- mixed_multitrace / de_milp / seed 29：U=-0.15491221962577995，calls=640，status=OUTER_LIMIT。
- mixed_multitrace / de_enumeration / seed 47：U=-0.15491221962577995，calls=640，status=OUTER_LIMIT。
- mixed_multitrace / de_milp / seed 47：U=-0.15491221962577995，calls=640，status=OUTER_LIMIT。

`/root/autodl-tmp/generalist-specialist-project/artifacts/continuous-routing-dimensions-v1-20261101`：6 次运行，总 wall 15.406 秒。
- dim1_image / de_enumeration / seed 11：U=-0.1238007461586366，calls=396，status=OUTER_LIMIT。
- dim1_image / de_milp / seed 11：U=-0.1238007461586366，calls=396，status=OUTER_LIMIT。
- dim2_text / de_enumeration / seed 11：U=-0.07092785708641972，calls=298，status=OUTER_LIMIT。
- dim2_text / de_milp / seed 11：U=-0.07092785708641972，calls=298，status=OUTER_LIMIT。
- dim3_mixed / de_enumeration / seed 11：U=-0.14513866893117014，calls=320，status=OUTER_LIMIT。
- dim3_mixed / de_milp / seed 11：U=-0.14513866893117014，calls=320，status=OUTER_LIMIT。

`/root/autodl-tmp/generalist-specialist-project/artifacts/continuous-routing-final-source-v1-20261101`：6 次运行，总 wall 35.876 秒。
- low / de_enumeration / seed 11：U=-0.1363523585025562，calls=576，status=OUTER_LIMIT。
- low / de_milp / seed 11：U=-0.1363523585025562，calls=576，status=OUTER_LIMIT。
- mixed / de_enumeration / seed 11：U=-0.14582921922846204，calls=576，status=OUTER_LIMIT。
- mixed / de_milp / seed 11：U=-0.14582921922846204，calls=576，status=OUTER_LIMIT。
- text_hot / de_enumeration / seed 11：U=-0.15454849001070675，calls=576，status=OUTER_LIMIT。
- text_hot / de_milp / seed 11：U=-0.15454849001070675，calls=576，status=OUTER_LIMIT。
