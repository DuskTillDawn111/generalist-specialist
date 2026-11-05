# 论文预测机制与 MILP 结构优化补齐

后续实现见 [2026-11-03 CPU 阶段交付](paper_structured_completion_20261103.md)：共享 G 跨任务 FIFO 界、原生捕获转换、映射缓存及结构化候选搜索已补充；本页为 11 月 2 日版本及其原始实验记录。

本轮按“完成能落实的代码，小范围验证，不启动 GPU、大扫描或 Phase 2”实施。补齐了训练到共同预测接口、受限上下文泛化、新阶段预测器候选界和精确路由列复用。**不是所有论文机制的完整复现，也没有新的真实 GPU 效用或效率结论。**

## 实际完成

| 原缺口 | 新实现 | 接入位置 |
|---|---|---|
| 阶段、生命周期、资源缺少统一训练入口 | `paper_training.fit_config/fit_deployment`：完整阶段导出 → 工作守恒 NNLS → 独立轮次检查 → 生命周期及整板资源拟合 → 新配置 | `experiments/fit_paper_predictor.py`；输出配置供主 Joint / DFS / 连续精修使用 |
| 精确查表无法处理未见共跑组合 | `ConvexContextLaw`：显式可选的资源特征凸插值 | `ProfileProvider` → 共同 FIFO 引擎；默认仍精确查表 |
| 旧独立服务界不适用新预测器 | `PaperDeploymentBounds`：最快阶段进度、自由执行器选择、去排队的乐观松弛及原 rho/simplex 必要条件 | `ContinuousDeploymentBounds` → 自适应多候选连续精修 |
| 残余 Joint 每次重新构列 | `JointColumnCache`：精确域身份、容量 witness、K、列上限、压力策略及额外分母键；LRU、16 MiB 序列化预算、防御性复制 | `JointDeploymentCandidates`；多候选与自适应算法共用 |
| 新模式与多候选流程未核对 | 列缓存/候选界开关对照、端点及非网格点反例、多候选和自适应集成 | `tests/test_paper_completion.py` 和独立 CPU 演示 |

本轮未修改 `gsdeploy/milp.py` / `optimizer.py` 的算法代码，未改 U、`rho<=0.8` 或 `g(n)`；保留工作区中已有的预测模式接线变更。列缓存仅包装实验实例方法，不共享 master、no-good、反馈或部署排除历史。首次高 K 构列复杂度未因此消失。

## 训练输入与证据

新增 `gsdeploy.complete_stage_training.v1`，每个部署提供独立训练 run_id、来源、执行身份摘要、measured/CPU 样例类型、同机 `host_perf_counter_ns` 时钟、完整记录窗口、驻留/活动 roster 完整声明、逐请求有序阶段、生命周期观测及资源窗口。显式模板提供独占基线阶段与执行配置，不能从共跑总耗时拆造独占 GPU 时间。

阶段须连续覆盖 host/sync/cleanup 等全部工作；同一 batch1 owner 不得同时执行两条请求。资源窗口活动集合必须与阶段跨度一致，不能跨状态切换后把平均值伪装成固定上下文观测。功率用整板均值，显存用全部驻留 worker 的原始峰值，再由原预测接口应用 safe_factor。

生命周期 id 为 `run_id:request_id`，handler_end 必须匹配该请求最后阶段结束。receipt、permit release、READY 分开保留、不互相代填。进度的每个目标/上下文、资源上下文及生命周期输入检查独立轮次；同轮多条记录不能充当重复运行。验证记录禁止进入拟合。

秩不足、残差过大、缺重复轮次、缺 READY 或资源时，相关规律保持 UNKNOWN，有效部分可输出 `PARTIAL_TRAINING_LAWS`。原配置不改，未训练部署不进入输出域。roster/同机声明是数据生产器契约，字段本身不能证明采集真实完整。本轮未把历史不匹配数据改造成新测量。

```bash
python experiments/fit_paper_predictor.py \
  --config /absolute/path/to/template-system.json \
  --training /absolute/path/to/complete-training.json \
  --output /absolute/path/to/new-fit
# 明确接受插值假设时才添加 --interpolate-contexts
```

输出配置、拟合诊断及输入 SHA-256；输出目录必须为新路径。演示保存完整 CPU-FIXTURE 输入示例。

## 受限的共驻进度模型

USHER 的资源画像/需求感知放置提供结构参考，没有提供适用于任意 GPU 的现成减速公式。新增一个明确的统计假设：同执行身份、同目标 alias/profile/segment，在训练覆盖的资源特征域内估计共跑速度。

特征为 host/gpu/sync/cleanup 数量及 SM/DRAM/L2 需求和；目标速度的特征排除目标自身，整板资源特征包含全部活动阶段。对查询 f，在训练顶点 f_i 上求：

`w_i >= 0, sum(w_i)=1, sum(w_i*f_i)=f`。

速度/功率取凸加权值，显存取实际供体的最大值。完整驻留部署始终不变，零流量不会卸载模型。每个目标需同身份训练支持；资源需已知功率/显存；缺 GPU 计数器不能用零代填。不跨硬件、执行配置、驻留集合、目标输入或阶段任意转移。

只对表中完全不存在的上下文插值；已有上下文的缺失字段/情景仍未知。范围外、LP 超时或失败、支持不足返回 UNKNOWN。单 LP 限时 0.1 秒、最多缓存 128 个上下文；FIFO 输出保留服务/功率/显存证据，插值能追溯供体及权重。

这是有限泛化，不能解决所有覆盖缺口。相同资源特征不保证物理干扰相同；显存供体最大值也不是物理安全保证。独立精度、原生执行和实测收益资格仍为 false。没有宣称完成 OG-Merger、真实 MPS/SM 隔离、跨模型权重共享或 busy-state 全分支。

## 候选界的依据与限制

对驻留 D，每阶段取允许上下文中的最大速度 v_max。请求耗时至少为 `sum(duration/v_max)+receipt`；去掉队列、READY 等待、准入，并允许每请求自由选最快执行器，得到 Mean/P95 下界。向下保护整数 ns 舍入误差。

资源取全上下文最小已知功率/显存，缺项退回零，startup 按原式摊销。Q 使用原逐样本误差的乐观分子/分母，无法解释的聚合质量退回零；T 当前保守为零。六项非负损失下界取负即 U 上界。robust 最差 U 不超过 nominal U，但缺压力情景不被判为无可行解。

凸插值速度不超过训练最大值，功率/显存不低于训练最小值，因此界仍覆盖该声明模型。原 rho/simplex LP 和全上下文显存下界提供必要条件检查；用 HiGHS 数值判定及向外余量，不声称精确算术证明。

界针对数学预测器，不是 GPU 物理保证。缺证据不等于不可行。资源互补启发式只排序，不安全剪枝。新界目前用于候选级筛选，未把旧固定路由任务列界套到非可分模式；共享 G 排队紧界和按需构列仍未解决。

## 小范围核对与交付

受影响模块回归 **62 项通过，31.787 秒**。覆盖训练时钟/身份、独立轮次、缺生命周期、资源窗口、插值域外、缺计数器、模型零流量驻留、FIFO 共跑重估价、界覆盖端点/非网格值、缓存隔离及新模式多候选接线。只验证软件正确性。

追加两项证据边界后，新功能 16 项复核通过（14.210 秒）；原全过程/有序阶段/生命周期相关 70 项通过（0.433 秒）。分批共覆盖 134 项不同测试。训练 CLI 输出与 API 逐字段一致。最终报告列出实际计数和时间：构列从 3 次降至 1 次，但小样例总时间未更快，最佳 U 仍持平。

```bash
python experiments/complete_paper_milp.py --output /absolute/path/to/new-cpu-demo
python -m unittest tests.test_paper_completion tests.test_paper_integration \
  tests.test_paper_continuous_refinement tests.test_multi_candidate \
  tests.test_adaptive_candidate tests.test_continuous_validated -v
```

最终原始数据、图、manifest 和结果摘要见 `artifacts/paper-milp-completion-20261102-final/REPORT.md`。此前同名不带 final 的目录为开发过程记录。演示值全部为明确的 CPU 软件样例，不是共驻测量；小样例计时不支持加速结论。

## 尚需证据及下一步

真实可迁移规律仍需匹配训练资料和独立验证。旧 346 条留出请求缺有序阶段/READY；其他训练记录驻留集合或初态不匹配，不能补字段冒充新规律。本轮未启动 GPU、长训练、广泛扫描、GP-BO 或 MADS。

后续按训练契约先复用或定向补齐一个部署的少量数据，独立核对进度、FIFO、资源和路由排序，再比较同域 Joint / Joint+连续 / 多候选。若有限泛化假设失败，先修正规律；若预测资格通过后重复部署求解仍占主导，优先结构化搜索加速。当前没有真实系统上 U 和速度同时超过原 Joint 的证据。
