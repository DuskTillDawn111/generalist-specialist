# 论文适配、结构化部署与连续路由：CPU 阶段交付

2026-11-03。按“先做到没有 GPU 进行不下去”的要求，补齐当前确定方案的数据转换、预测复用、共享 FIFO 界、部署/路由搜索和资源配置决策链路。没有启动 GPU 推理、长训练、全池扫描、GP-BO 或 MADS；原 `JointMilpOptimizer`、`UtilityOptimizer`、六项 U、rho≤0.8 与 g(n) 本轮没有改动。

**结果：当前方案可以在 CPU 上完成从完整训练记录到优化方案及资源绑定包的执行。减少了重复 MILP 和事件模拟，但尚无真实 GPU 精度、效用提升或端到端加速结论。** “当前项目适配链路可执行”不等于所有论文完整复现，也不等于所有优化研究已经结束。

## 这次补齐了什么

| 缺口 | 已实现 | 范围与限制 |
|---|---|---|
| 原生记录仍需手工转成训练格式 | `paper_capture.normalize_capture` → `paper_training.fit_config` → 共同阶段 FIFO 预测器 | 复用已有生命周期导出器；必须有完整有序阶段、同机时钟、驻留名单和资源窗口。旧 handler 总时间不能拆造 GPU/host 阶段 |
| 多候选生成反复运行残余 Joint | 独立 `PaperStructuredOptimizer`：一次 Joint 初始解 → 已声明画像部署 → 固定部署连续精修 | 初始解时间全部计入；初始解后部署 MILP 为零。候选范围、未支持部署、未访问及预算截断分别输出 |
| 新预测器的固定路由界过松 | `PaperDeploymentBounds` 新增跨任务共享 owner FIFO 时间下界；接入 `bounded` 和 `milp_bounds` | 只证明当前声明预测器的乐观界，不是物理 GPU 保证；旧独立服务界仍不用于新模式 |
| 多个连续比例产生同一 FIFO 映射时重复模拟 | 独立 FIFO 结果 LRU，默认 128 项/16 MiB 序列化预算 | 精确绑定部署、请求/输入/alias 映射、情景、horizon/drain、能耗需求；每个比例仍重算 Q 和 rho |
| 资源配置停留在排序/装箱 | `ProfiledResourceConfigurationSearch` 以完整 U 选择合法配置，再生成资源绑定包 | 每种控制配置必须有自己的执行绑定画像；不同配置禁止复用相同执行上下文；不把 MPS 百分比换算成速度 |
| 当前模式无法从 busy 状态续推 | `PaperFifoSession.forecast_continuation` 使用同一 providers 和 FIFO 引擎 | 校验当前 observer、完整 pending cohort、实际已分配 owner、显式执行游标、资源身份、handoff/gate；只输出剩余过程，不虚构全阶段 U |
| LP 与原 MILP 混用会报求解器错误 | 新 LP 统一使用与原 Joint 相同的单线程 HiGHS 设置 | 发现并修复进程级 scheduler 配置冲突；未修改原 MILP |

保留上一轮训练工作守恒 NNLS、可选凸上下文插值、路由无关部署上界和精确 Joint 路由列缓存。本轮沿用它们，历史交付见 [11 月 2 日报告](paper_milp_completion_20261102.md)。

## 与论文的实际关系

J3O 的项目适配是“离散部署候选与连续路由交替，LP 生成路由提案，完整预测 U 决定接受”。新增搜索器将此结构与 Joint 热启动连接，固定部署阶段不再调用部署 MILP。LP 只处理原 simplex/rho 及线性提案；实际 P95、T、动态共驻、M/E 全部由共同 FIFO 计算。没有继承原 J3O 分层节点问题的子模性或近似比。

USHER 的适配分成需求分组/放置提案和**有独立画像的资源配置选择**。SM/DRAM 互补只排序，不剪枝；显存容量、带宽需求及 MPS ceiling 分开处理。现在配置选择、完整 U 评价和真实 worker registry 的资源绑定能在 CPU 上贯通；实际资源控制是否生效仍需平台证据。

FIFO 部分采用基于冻结到达轨迹的事件模拟、活动阶段变化时重新定价，以及 handler、parent receipt、permit release、READY 的分离。busy 续推不会用“已运行多久”猜测 GPU 剩余工作；缺当前游标或队列成员即 UNKNOWN。续推没有完整历史耗能等信息时 `U=None`，应另用已有 `score_forecast` 和完整历史评分。

InferLine 的事件回放思想已用于本项目；iGniter 的原始资源估计/配置算法、MuxServe 的完整 prefill/decode/KV 管理、USHER 的 OG-Merger 没有因此变成已复现。这些原系统机制不能凭相似接口算作完成。`mps_static_sm` / `green_context` 的原生适配器也**仍未实现**，并非仅缺测量；本轮没有采用这两个模式，选择它们的绑定包继续显示 `RESOURCE_CONTEXT_ADAPTER_NOT_IMPLEMENTED`。当前方案使用已有 shared/request_admission 路径或保留平台验证阻塞的 MPS 配置，不绕过执行资格。

## 搜索及界的正确性边界

对部署 D 和固定连续路由 r，先用实际 Router 重建精确 alias 请求映射。对每个请求 j，每段以该执行身份下声明上下文的最大进度运行，得到 handler 下界 h_j。每个 alias 只有一条跨所有任务的 READY 时钟：

`start_j = max(arrival_j, ready_alias)`

`end_j = start_j + h_j`

`receipt_floor_j = end_j + receipt_delay_floor_j`

`ready_alias = end_j + ready_delay_floor_j`

去掉干扰和准入等待只会让这条时钟更快。receipt 与 READY 分开，不能让第二条请求在第一条结果接收时提前开始。相同 alias 的 text/image 共享同一时钟，不能为每个任务各建一条空队列。整数 ns 舍入使用向下余量。逐请求 floor 导出 Mean/P95、必然超窗比例及必然排空失败，资源和原 rho 必要条件也分别检查。

缺 stage/context 服务证据时使用零松弛，不能移除该执行器或宣称实测不可行。插值速度是声明供体的凸组合，最大表速度仍覆盖它。所有安全剪枝仅针对声明模型；LP 的不可行性使用 HiGHS 数值证书及余量，不称精确算术证明。没有非线性连续全局最优性证书。

固定部署的 `CapacityAwareTransferSearch` 使用 Joint 投影路由、原容量 LP 提案、**优先精确端点**、连续混合与按目标模型共享容量余量裁剪的比例转移。没有重新量化到旧 K 网格，也没有称为 MADS；预算不足会记录端点覆盖和终止原因。未知内部点不终止其他端点的搜索。路由变为零时模型仍驻留。

`joint_seed=False` 是单独开关：可以完全跳过 Joint 构列，直接在已画像部署域搜索。它不能保证重现未遍历 Joint 域的最好解，也不提供连续全局证书。高 K 首次构列的组合增长、完整动态目标的更紧松弛和更大模型池可扩展性仍是研究问题；它们没有被归咎于缺 GPU，也没有被宣称解决。

## 小范围实验与结果

原始数据及图位于 [最终实验目录](/root/autodl-tmp/generalist-specialist-project/artifacts/paper-structured-completion-20261103-final/REPORT.md)。使用明确的 CPU 解析样例：1 任务、2 模型、3 合法画像部署、每条轨迹 2 请求、K=2。8 方法 × 2 搜索种子，共 16 次；所有方法同预测器、同轨迹、同质量/资源约束。各方法统一 6 秒、48 次完整预测查询、24 次连续路由评价的上限，原算法自身上限及有限域也保留。未截断输入轨迹，没有伪造新共驻测量。

下表取 seed=11 的一次记录，只说明这次运行；不是统计加速比：

| 方法 | 总时间 s | 完整预测查询 | 实际 FIFO 引擎调用 | MILP master 求解 | 连续路由评价 |
|---|---:|---:|---:|---:|---:|
| 原 Joint | 0.272 | 5 | 5 | 6 | 0 |
| 原 DFS | 0.113 | 5 | 5 | 0 | 0 |
| DE + 固定路由 MILP（新界） | 2.079 | 24 | 6 | 50 | 24 |
| Joint + 固定部署 DE | 0.174 | 5 | 5 | 6 | 1 |
| 多候选 | 0.688 | 13 | 6 | 13 | 13 |
| 自适应多候选 | 1.283 | 19 | 6 | 13 | 24 |
| 结构化候选 + 容量转移 | 0.689 | 14 | 5 | 6 | 12 |
| 结构化候选 + DE | 0.873 | 12 | 6 | 6 | 13 |

两个种子全部返回 **U=-3.0157777778，部署 G×1，text 的连续比例 G=1、QPS=2，实际两条请求均为 G#0**。最终是端点；搜索过程包含非网格内部比例。双模型固定部署的端点仍使用 40 MiB 驻留资源，删去 S 后才使用单 G 的 20 MiB。这些数值全部是解析样例，不是硬件测量。

“完整预测查询”是名义/压力打包接口调用，不等于实际事件模拟次数。每个打包接口检查三情景，样例只声明 nominal；未支持压力保留 UNKNOWN。事件引擎计数单独保存。

独立缓存消融：同一固定双模型部署的 8 个不同连续路由点，FIFO 引擎 **8→2** 次，6 次按完全相同映射复用；U、Q、rho、映射、完成时刻逐项相等。默认 FIFO 缓存仅占 14,258 字节。不能只缓存整份 score：相同映射仍可能因为聚合质量代理或 rho 比例不同而有不同 U/资格，测试包含同映射跨 rho 可行边界的反例。

小样例为何仍不能证明效用或总速度更优：

- 一次 Joint 已经覆盖了这个很小的离散域，之后搜索仍有额外费用。结构化方案将多候选的 13 次 master 降为一次 seed 的 6 次，但仍需建界、LP、连续提案、校验及缓存复制；这个成本不随 master 次数一起消失。
- seed=11 的结构化容量转移约 0.689 秒中，MILP 0.0188 秒（2.7%），完整预测接口 0.1909 秒（27.7%），路由搜索及其校验 0.3562 秒（51.7%），其余构界/LP/初始化/Python 工作 0.1228 秒（17.8%）。完整分解在 `comparison.json`；不能把“少解 MILP”称为“端到端更快”。`outer` 计时扣除了其内部预测和 MILP 时间；没有将整个外层 wall time 与预测时间重复相加。
- 只有两条请求，许多连续比例映射相同。样例的 Q 明确使用 aggregate_proxy，相同有限 FIFO 映射并不保证 Q 相同；它不是从实际输出得到的连续分流质量收益证据。
- 最好部署只有一个模型，固定它的路由维度为零，因此 Joint+固定部署精修无需继续搜索。多候选能访问双模型部署并探索内部点，但仍未超过全 G。

183 项直接相关测试通过，耗时 **7.879 秒**，包括新反例、既有训练/连续接线、共享 FIFO/生命周期、observer、资源控制及 kernel baseline。测试只证明软件合同。另一次扩展回归进入旧多负载穷举后被主动停止；其中断日志保留在 `paper-structured-completion-20261103-before-endpoint-priority/tests.log`，不计为完成的回归，也不使用其性能结果。

## 复现入口

```bash
# 当前独立搜索器；默认包含一次 Joint 初始解
python experiments/structured_paper_search.py \
  --config configs/paper_fifo_cpu_fixture.json --output /tmp/structured-new \
  --seconds 6 --max-predictions 48 --max-routes 24
# 可加 --method de 或 --without-joint-seed，分别记录不同实验

# 显式资源候选 JSON：[{id, config: 配置路径, profile_bindings: 绑定表}, ...]
python experiments/structured_paper_search.py \
  --resource-candidates /absolute/path/to/resource-candidates.json \
  --output /tmp/resources-new

# 原生完整训练记录 -> 规范记录 -> 规律；不采集，不运行 GPU
python experiments/normalize_paper_capture.py \
  --config /absolute/path/to/template.json --captures /absolute/path/to/captures.json \
  --output /tmp/normalized-new
python experiments/fit_paper_predictor.py \
  --config /absolute/path/to/template.json --training /tmp/normalized-new/training.json \
  --output /tmp/fitted-new
```

转换器要求 `gsdeploy.complete_stage_capture.v1`，完整示例保存在实验目录的 `native-capture-CPU-FIXTURE.json`；那是软件样例，不能修改 kind 后当作真实采集。模板仍需独占阶段基线和执行配置，训练器不会从共跑总延迟推造它们。训练和验证角色严格分开，转换/拟合不读取留出标签。

## 到哪里需要 GPU

2026-11-03 的只读 `nvidia-smi -L` 返回退出码 126、`Permission denied`；另检查到 `/dev/nvidia*` 设备节点为空、该命令不可执行，记录见 `gpu-access.txt`。没有加载模型或发起 GPU 验证。

当前确定链路下一项必需输入是**匹配实际模型、执行后端、驻留集合及资源配置的真实阶段/资源/生命周期训练证据**。历史 346 条留出记录缺有序阶段/READY，其他训练记录执行身份或前史不同，不能自动补字段后复用。CPU 样例、插值可解或训练残差小均不能替代这些证据。

下一次只安排一个匹配部署的定向小采集：先确认容量和阶段记录完整；每个拟合上下文至少两个独立训练轮次；再用独立短轨迹核对服务、FIFO、READY、全驻留显存/整板能耗与路由排序。优先恢复此前记录的 G2+image+OCR/OCR 热点小任务，不开展全池扫描；训练/留出先声明，旧 seed 不重新称为盲测。通过这个检查后，才用相同轨迹和完整成本比较 Joint、Joint+连续与新结构化搜索。

真实预测资格、真实连续分流收益与真实联合优化重评仍未完成；`native_eligible`、`independent_accuracy_qualified`、`final_gain_eligible` 全部为 false。GPU 待办及其原有依赖保留，见 [当前状态](joint_refinement_current_status.md)。
