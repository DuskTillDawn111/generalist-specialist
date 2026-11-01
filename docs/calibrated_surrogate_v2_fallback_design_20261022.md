# v1 校准失败时的 v2 后备设计（仅设计，未启用）

后续实现与正式种子分配见 `docs/calibrated_replay_v2.md`：新 holdout 固定 seed46/47；下文早期建议的 seed43 已取消，43/44/45 保留给 formal confirmation。

本文件不改变 v1 protocol、阈值、样本分组、实现或执行快照，也不代表 v1 已失败。只有完成既定 6 个训练条件和 2 个 holdout 后，才能作出 v1 的正式判断。若失败，保留原始失败报告；不得提高 U 误差阈值或继续用原 holdout 宣称独立验证。

## 选择依据和最小改动

当前 FIFO 对同一 `(model, task, n)` 的每个请求使用相同服务时间。已有 standalone `raw_service_samples` 带 `sample_id`、`protocol_sha256`、`request_sha256`、`replica`、`service_ms`，真实静态请求也保留同样的输入身份，足以建立逐输入的服务回放。输入服务时长变化会通过共享 FIFO 的队头阻塞放大；只拟合全局均值比例不能恢复这种变化。

v2 的首选变化是**逐输入服务时间重放**，保持原有全局服务比例、非负整板功率模型、启动能耗和内存模型。先不增加按任务的自由系数，也不拟合依赖实际 holdout 完成时间的校正项。这样新增结构用于恢复已测到的输入变化，新增参数数量仍受训练条件数量限制。

## 冻结输入回放及拟合规则

1. 对每条路径状态 `(model, task, n)`，仅从冻结 standalone 文件建立 `sample_id + protocol/request identity -> median(service_ms)`。协议、输入、模型运行参数、checkpoint 和源文件摘要全部绑定。不同副本的同输入观测可按预先声明的规则合并；跨 `n` 状态不得借用。
2. 和运行时 `SamplePool` 一样，以 `seed:samples:task` 打乱每个任务池并循环取样；每个任务请求先确定 sample，再调用相同的 Router 分配 family/replica。所有候选共用相同到达和样本序列。不能按模型各自消费一份样本池，否则路由改变了任务负载。
3. 每个请求服务时间是 `global_multiplier * baseline_input_median(model, task, n, sample)`。全局比例以训练请求的实际 worker service 总和除以逐输入 baseline 总和拟合；预测不得读取当前候选的真实服务时间或质量结果。
4. 缺输入或缺状态时默认返回 UNKNOWN，报告覆盖率。不能用同一 holdout 请求的服务时间填洞。若确需 aggregate fallback，必须在新 protocol 中预先声明，并将适用范围单独标记。
5. 能耗先保留 v1 功率特征含义及训练方式，使评估能区分“队列结构改善”和“新增能耗模型”。若更换为逐输入 demand 特征，应视为另一个预先指定的模型变更，不能看新 holdout 后再选择。
6. 不从少量测量中拟合任意“U 修正常数”。分别报告 Q/T/L/P/M/E 的误差及其加权 U 贡献，确定误差是否来自输入差异、跨 worker GPU 争用、功率或内存。

同输入中位数属于固定有限样本预测，不提供 GPU 随机运行置信区间；v2 同样只能证明冻结预测器有限决策域内的最优。

## 新 protocol 与最多两个新增 GPU 条件

若 v1 失败，则旧 8 条条件一律成为 v2 development 数据；原 holdout 应在报告中明确标记“已查看，不再独立”。已有三 worker adaptive 条件可作为额外 development 数据，但不得将 confirmation 结果改作训练。v2 的服务回放规则和拟合方式先冻结；可用旧 development 数据分析原因，不把两个新 holdout 用于选择模型结构或调参。

建议两个新 holdout 使用同一个三 worker 部署 `{generalist_2:1, image_specialist:1, ocr_specialist:1}`，文本/ASR 全给 generalist，并分别使用：

- H1：image 专用份额 0.75，OCR 专用份额 1.0。
- H2：image 专用份额 1.0，OCR 专用份额 0.75。

这两条覆盖相邻的不同分流方向，并避开已经计划的 image/OCR 都 100% 专用的 adaptive 路由。使用固定静态工作负载、600 秒、预先指定的新 seed=43（原校准 seed=42），使样本和到达顺序也独立。生成 protocol 时必须核实该部署和两条路由尚未执行，内存和任务支持均可行；若不可行，在运行前替换并记录原因。这里的 family 名称应解析为冻结配置中的真实模型键。

新 protocol 至少冻结：全部 development IDs、两个新 holdout IDs、原始服务文件及输入池摘要、逐输入归并规则、缺失处理、拟合公式、特征/参数数量限制、阈值、新 seed、到达过程、H、GPU/运行时版本和模型权重。沿用 v1 数值阈值（包括 U 绝对误差 0.10）有利于比较，但必须明写到 v2 protocol，而不是隐式继承。

两个新条件仅足以进行有限 holdout 检查，不能宣称统计上证明全空间物理最优。如果 v2 仍失败，应停止正式 surrogate 搜索并保留失败；继续补测需要重新设计实验，而不是反复修改同一 holdout。

## 与 MILP / DFS 的衔接

新 schema 必须与 v1 区分，两个求解器调用同一个逐输入预测器和质量、能耗、约束评价。v1 的常数服务均值下界不能直接作为 v2 下界：某个路由可能恰好分到较快输入，使常数均值大于真实无队列成本。

可先使用每条路径冻结输入服务时间的最小值构造安全但较弱的界；更强的版本可针对已经固定的任务路由列，重放其 sample 分配，计算逐输入无等待时延的 mean/p95 下界。等待时间非负，所以同一请求集的无等待时延按分量不大于 FIFO 时延。部分未分配路由只能使用仍保持乐观的下界；rho 约束也需按预测器正式定义一致处理。没有证明的新项使用零下界，不能沿用不再适用的公式。

正式启用前重新执行：任意完整/部分候选下界不高于实际 surrogate 成本、小域 Cartesian/DFS/MILP 最优一致、名义与 robust 目标一致、两种到达过程一致、拥塞分流选择正确、失败及身份变更返回 UNKNOWN。v2 独立 holdout 未通过前，不产生正式可用 U 或最优性证书。
