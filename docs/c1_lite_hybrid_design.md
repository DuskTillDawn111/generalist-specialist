# C1-Lite 实现设计与证书范围

2026-11-03，执行依据：[任务书 V2](c1_lite_hybrid_task_v2.md)。本版完成 CPU 软件原型；真实预测与场景资格、正式算法比较及 Hybrid 分支尚未完成。结果见[实验报告](c1_lite_hybrid_experiments.md)。

## 实现入口与保留的架构

外层给连续路由，内层固定该路由搜索模型集合与合法副本，再以同一完整预测器评价效用。新增组件通过 `gsdeploy.continuous` 导出，原 Joint、DFS、`DERoutingOptimizer` 和固定路由 MILP 继续保留。

| 组件 | 文件 | 职责 |
|---|---|---|
| `DeploymentCandidatePool` | [deployment_pool.py](../gsdeploy/continuous/deployment_pool.py) | 有容量上限的跨路由部署结构池 |
| `ProgressiveFixedRoutingSolver` | [progressive.py](../gsdeploy/continuous/progressive.py) | 每路由 ledger、池评价、限额 MILP、追加求解与全域界 |
| `InnerSearchTrigger` | 同上 | always / gap / budget 三种策略及显式探索配额 |
| `C1LiteOptimizer` | [c1_lite.py](../gsdeploy/continuous/c1_lite.py) | 独立渐进 DE，处理部分内层观测和池版本变化 |

未接入 BO/MADS。C1-Lite 本身没有 Joint 初始化；现有 Joint 多候选固定部署精修仍作为 C3 对照，不能称为 Hybrid C1。

## 共享池与路由 ledger

规范部署键包含模型名与副本数，显式拒绝 bool、非法副本和空部署。池保存来源、发现顺序、任务覆盖、历史最好完整评价及路由；历史效用仅用于评价顺序与容量管理。

新路由先按全部任务的正流量模型筛选兼容结构，再经过原评价器检查硬约束与完整证据。显存标记为需要上下文预测，结构合法不代表峰值显存已通过；零流量模型不会被自动卸载。不同路由不共享效用、未知状态或 no-good 排除。

池支持 utility/recent 策略和模型集合多样性倾向，扩池/淘汰改变 `pool_version`；历史最好值更新不改变结构版本。版本代表结构集合，历史排序不构成池值证书。默认最多 16 个结构；淘汰是启发式，不是安全剪枝。全局 incumbent 的结构受保护，完整赢家还独立保存在求解器中；路由 ledger 不随淘汰丢失。

每个精确路由及不可变身份有单独 ledger，保存已评分、已剪枝和 UNKNOWN 身份、最好可行值、剩余域上界及 master 轮次。追加预算复用 ledger 和精确预测缓存。SciPy 接口未提供本实现可用的原生暂停恢复/MIP start，因此每次按 ledger 重建 one-hot 部署 master，加入路由局部排除；重建计入总 wall，输出 `mip_start_used=False`。

## 下界、上界与 UNKNOWN

当前完整可评分的最好可行值是 `LB(r)`。采用显式声明的全局成本下界、状态成本下界、显存必要下界及适用的逐候选安全界；没有证明时返回 `UB=None`。v2 不沿用旧的非负 Q 假设，允许全局成本下界为负及 U 为正。

全域 UB 同时覆盖：未访问的 master 域、从 master 排除但证据未知的部署、已有可行最好值。UNKNOWN 的 UB 被保留；本版采取保守策略，只要有 UNKNOWN 就不给全域证书。剩余 master 不可行只说明 master 域已穷尽，不能抹去这些未知身份。

认证要求所有证据与界合法，并满足有限域穷尽或当前路由 UB–LB 容差。全域无解也需要完整排除，池中没有可行部署不足以证明无解。完整预测若违反已声明 UB，输出 `SOLVER_ERROR`，清除证书和 UB；未测身份仍依赖 oracle 下界契约的正确性。

返回值包含路由、部署、LB、UB、gap、约束状态、认证状态、退出原因、池版本、本次预测/master 数、累计会话数和未解决身份。`inner_search_completed` 表示取得当前路由证书；即使所有 master 访问结束，UNKNOWN 仍会令它为 false。

证书仅针对冻结 oracle、当前路由、声明的有限副本域和数值容差。它不证明真实硬件精度，也不证明连续路由全局最优。

## 触发器与渐进 DE

- always：每次未认证评价触发内层，仍服从局部及全局预算；`exact_inner=True` 才取消局部限额并要求完整内层。
- gap：有效 UB 未收敛时触发；若可证明无全局改善则跳过。缺 UB 依赖显式探索配额。
- budget：无可行池结果时搜索，其余按探索配额及竞争路由细化；池下界够强不构成精确值。

独立控制器使用 DE/rand/1/bin 风格提案和现有闭 simplex 编解码，允许精确端点及有限激活面初始化，给内部探索保留预算。它与原 SciPy DE 的初始化及迭代细节不同，因此完整运行差异不能全部归因于池。

主模式明确标为 `lower_bound_guided`：部分值只作启发式可行 LB 排序；UNKNOWN 保持 `None`。区间重叠时周期性细化竞争路由，未分离时使用声明的 LB 比较规则。每次选择记录 parent/trial 的 LB、UB、证书、版本与启发式标记。

trial 发现新结构后，先刷新全部过期且未认证的种群成员，再做选择；细化再扩池时同样刷新。已认证值与池无关可直接保留。刷新只使用池，不推进探索计数；刷新也消耗全局外层/预测/时间预算，不能隐去其成本。

## 预算、身份与原型限制

复用不可变问题、独立预测 session 与 `Budget`。初始化、池评价和细化共用一个 deadline 及预测/master/外层计数；局部限额通过临时收紧后恢复原上限实现，不续期全局预算。非抢占调用实际超时保留在 wall 中。

结果分列 bundle 预测、叶子、master、外层评价与 cache hit。组件计时不等于总 wall：构造、身份检查及部分外围开销体现在总 wall，尚未全部拆成独立组件；缓存计数包含 ledger 命中，池 `evaluations` 仅统计可评分可行观测。这些指标不能当作精确的池命中率或 FIFO 引擎次数。

本版尚未完成所有任务书中的指标图、池多样性/探索配额扫描、同初始种群的完整外层因果消融，以及规模化正式比较。候选池提供配置接口，实验脚本只覆盖默认策略、容量 1 与 16 及三种触发策略。真实采集接线缺口见[预测资格](c1_lite_hybrid_prediction_qualification.md)。

## 验证

新增 22 项算法反例与 5 项资格准备测试；与原连续路由/安全界、论文预测集成、v2 参考、系统和 MILP 模块合计 135 项通过。同一组 6 个固定路由，新旧内层认证值一致。测试及运行记录见[实验报告](c1_lite_hybrid_experiments.md)，这些是 CPU 软件证据。
