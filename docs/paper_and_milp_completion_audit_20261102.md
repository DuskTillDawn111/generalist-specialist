# 论文适配与 MILP 优化完成情况核对

**2026-11-03 后续实施：** [CPU 阶段交付](paper_structured_completion_20261103.md)进一步落实了完整原生记录转换、共享 FIFO 界、映射级事件缓存、一次 Joint 后无部署 MILP 的结构化搜索、资源配置选择及 busy 续推。183 项直接相关测试通过。以下旧表中的相应缺口须与新报告合读；尚未完成原论文完整复现、真实画像资格及 GPU 收益验证。

**后续实施更新：** 用户已授权补齐可实现部分。[补齐报告](paper_milp_completion_20261102.md)记录了新增训练流水线、可选凸上下文插值、新阶段预测器候选界、精确路由列缓存和 CPU 回归。下表保留实施前状态；“新模式界不可用”“无训练入口”“未做新模式多候选核对”已由后续实现更新。真实精度及资源执行闭环缺项继续保留。

2026-11-02。用户要求先核对论文及MILP工作是否完成，不急于验证。本轮只读代码/报告和更新记录，不执行训练、GPU或搜索实验。

结论：**两条工作线均未全部完成。已经有可运行的软件机制和独立优化原型，但还没有完成“论文适配的实测动态预测规律＋对应高效联合优化”的完整交付。** 不能把测试通过、接口接通或实验数量当作任务完成。

## 论文机制

| 工作 | 已落实 | 尚未完成 |
|---|---|---|
| J3O项目适配 | paper_algorithms.py：Joint初始解、固定部署连续LP提案、连续转移、单编辑部署前瞻、完整FIFO效用接受 | 有限启发式适配，不是原J3O分层贪心拉格朗日算法；没有原近似保证，也未证明端到端收益 |
| USHER项目适配 | 合法配置枚举、计算/显存互补分组、放置提案，已有SM/DRAM画像用于候选排序 | 没有完成需求感知资源配置与实际GPU执行的闭环；排序/装箱不构成干扰规律。OG-Merger、真实隔离等完整复现未完成，也不能不加判断地列为本项目必须全部照搬的机制 |
| FIFO适配 | 有序阶段推进、活动变化重算速度、共享队列；handler完成、结果接收、许可释放、READY分离；进入主预测和Joint+连续精修 | 匹配实测训练数据导出及可迁移规律未完成；当前入口只支持fresh/drained、batch1，高级busy-state/twoFIFO/kernel分支未全部接入 |
| 路由相关共驻干扰 | 支持按输入/阶段/完整活动上下文使用显式progress law | 现有精确查表不等于从有限实测训练出的通用服务模型；未见上下文、真实生命周期/资源规律、独立精度资格仍欠缺 |

论文适配不要求盲目复现每篇论文的所有系统组件。仍须明确本项目采用哪些机制、模型假设、输入证据和接受规则，并完成这些机制的实际链路；当前软件原型只能算部分完成。未开发GP-BO/MADS符合用户要求，不计作遗漏。

## MILP与连续搜索

| 工作 | 实际状态 | 有效范围 |
|---|---|---|
| 固定路由部署MILP | 已实现独立one-hot master、身份排除反馈、预算与未知证据处理 | 不替换原Joint；与完整枚举做过一致性比较 |
| 缓存/会话 | 已实现不可变快照、摘要memo、精确预测缓存 | 减少重复解析/哈希/预测；不能把缓存收益全归给连续算法 |
| 固定路由安全界 | validated_bounds.py复用旧Joint任务界及资源/rho必要条件 | 新coupled模式由opt.coupled检查拒绝启用，不能沿用旧独立服务界 |
| Joint＋连续精修 | 已实现DE和独立局部基线，热启动、闭simplex/端点、保留incumbent；已接入新模式 | 固定D阶段无部署MILP；新模式只有软件样例结果，旧域对照未提高系统最佳U |
| 多候选与自适应预算 | multi_candidate.py / adaptive_candidate.py已有独立原型及旧域实验 | 重复残余Joint生成部署；尚未完成新动态模式下完整适配/实验验证，不能继承旧域收益 |
| 连续部署上界/安全筛选 | candidate_bounds.py已实现旧标量预测器的乐观松弛 | 明确只接受calibrated_surrogate/calibrated_replay_v2，新模式available=false |
| 高K构列与Joint结构加速 | 已完成瓶颈诊断 | 任务列缓存/按需构列、共享G多任务队列紧界等仍属未完成研究，不能称已解决 |

旧冻结域曾将同路由内层成本显著降低，五模型DE+MILP相对自身基线改善，但仍落后原Joint。固定Text部署出现非网格收益，也未提高完整联合问题最佳U。没有已经证明“新算法同时效用更高且端到端更快”的结论。

## 正确的后续依赖顺序（当前仅记录，未启动）

1. 先补算法设计和实现边界：明确J3O/USHER/FIFO各自采用机制；完成训练观测到动态progress/生命周期/资源规律的可执行路径，先检查已有资料与可辨识性，不能靠直接多跑GPU掩盖缺失算法。
2. 在同一主链路中明确Joint初始解、连续提案、完整FIFO接受和部署候选生成的关系；为新非可分预测器设计可证明安全的下界/上界，证明不足的启发式只能排序，不能剪枝。没有界时保守运行是正确性措施，不是性能优化完成。
3. 优先解决有证据的优化成本：重复构列/会话成本、候选复用、共享G多任务排队界和不必要路由评价。保持原优化器作为基线，用独立模块落实改进，不改变效用及硬约束。
4. 小范围CPU反例/枚举核对只检验正确性。算法需要什么训练证据明确之后，再进行定向GPU采集及独立预测验证；随后才评价同域Joint、Joint+连续、多候选/结构加速的效用和总成本。

实测数据、GPU分流验证和后续重评继续保留。用户尚未恢复这些计算任务；本次核对不构成启动授权。

## 依据

- docs/paper_mechanism_integration_20261102.md、gsdeploy/paper_algorithms.py、gsdeploy/paper_forecast.py。
- gsdeploy/forecasting/profiles.py、lifecycle.py、simulator.py。
- docs/continuous_routing_validated_improvement.md、docs/joint_scalability_and_refinement.md。
- docs/multi_candidate_continuous_refinement.md、docs/adaptive_candidate_refinement.md。
- gsdeploy/continuous/validated_bounds.py、candidate_bounds.py、refinement.py、multi_candidate.py、adaptive_candidate.py。
