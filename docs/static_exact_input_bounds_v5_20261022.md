# v5：同一冻结预测器的逐输入搜索界

此改动只收紧搜索放松，不改 v2 预测器、系数、效用合同、质量评分、输入顺序或候选域。八个 `predictor_code_bindings` 文件保持 v4 原字节。v4 快照及其未完成认证记录保留。

## 为什么有效

对于固定部署和一个已指定的任务路由，运行时 Router 的状态按任务独立。该任务第 i 个请求选择的 worker alias 与其他任务是否存在无关。SamplePool 按任务独立洗牌、循环，选择发生在 Router 之前，因此第 i 个请求的输入也不变。

v2 的 worker 服务时间是该输入在该 path/count 上的冻结中位数乘全局 development 比例，再乘预测器原有 `n/capacity_factor`。固定输入、alias、服务时间后，删除同 worker 的其他任务请求只能使 FIFO 完成时间提前。由 FIFO 递推归纳，每个保留请求的响应时间均不大于完整计划；其平均、线性插值经验 P95、超出测量窗口的计数均不增。因此，单任务队列产生的 L/P/T 成本是完整计划对应项的下界。

任务请求数 <=1024 时，MILP column 和 DFS 已分配任务使用上述单任务 FIFO。更长 trace 使用同样的逐请求 sample→alias→服务时间序列，但去掉全部等待；每个服务时间仍不超过其完整响应时间，所以其经验 mean/P95 是安全下界。长 trace 的 T 放松为零。

robust 搜索的目标取 1/1.25/2 倍服务场景的最低效用。所有损失权重非负，固定队列中的完成时间关于倍数单调，因此 2 倍场景成本是三场景最大成本；以该场景建立的成本下界同样有效。Q/E/M 不因服务倍数改变，仍用原分解。

DFS 未分配任务仍用所有可用模型的跨输入最小服务下界；没有提前决定路由。rho 仍严格用完整输入池等权均值；功率 busy 项仍用原 standalone aggregate baseline，不能混为逐输入延迟界。

## 缺失路径

只有完整输入池通过 `replay_path_statistics` 的 path/count 才建立逐输入查找。缺失任一样本的 path/count 整体返回零成本放松。求解器仍枚举该候选；完整预测保持 UNKNOWN，不能借放松将其当作已知可行解或删除它。失败/变更版本仍无法激活这些界。

## 验证

`tests/test_replay_search_bounds_v2.py` 增至 12 项：合成全部网格/partial、nominal/robust、混合任务/多副本、periodic/Poisson、长 trace、缺失路径，以及 MILP/带界 DFS/穷举 DFS 对独立网格 oracle。另对真实完成 v2 的十个实测计划和 generalist_2 双副本计划，检查 704 个 partial 上界与 44 个完整 MILP 成本下界。真实候选即使不满足最终约束，也用完整预测的诊断成本检查界的数值有效性，不改变可行性定义。

新增短/长 trace 对照明确验证：逐输入界严格强于原跨输入 minimum，而不超过完整预测成本。41 项原 calibrated/DFS/MILP/residual/基础 MILP 回归同样通过。以上均为 CPU 检查，不启动 GPU。
