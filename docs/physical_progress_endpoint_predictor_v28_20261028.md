# 物理端点证据与替代执行门控

本交付仅使用两次旧 training capture，在 CPU 上重建事件前缀。没有运行 GPU、使用 validation 标签、修改生产预测器或测量最终分流收益。

## 二元 context 门控已被端点证据否定

推荐证据是 `artifacts/predictor-physical-progress-evidence-v26-20261028-training02/report.json`。784,974 个 kernel 全部保留；378 个 kernel 的 START、END 严格落在另一 owner 的合格成对 context envelope 内，且该 kernel 的整个观测生命周期没有 own envelope 或 gap。两次 capture 分别是 374、4，不能把这一差异抹平成稳定比例。370 个来自 text#1 decode 在 G2 envelope 内，另八个来自 G2 在 text envelope 内。其生命周期总和只有 1,581,245 ns。

因此，在已审计 owner 对应下，不能以“当前 context owner 不是自己”推导该 kernel 的物理端点不可能发生。原始 context 还有 31、32 个配对问题，事件域的解释也有边界；这里证明的是确定性 envelope gate 与观测端点不相容，不是所有 context 标记都有已知 SM 调度语义。

9,571,354,335 ns 的 other-envelope 生命周期重叠不是 SM service。156,355 个 kernel 与其他 owner 的 kernel 生命周期重叠也不能证明同时占用 SM。生命周期包含等待、挂起和未分类区间，不能换算服务量、资源份额或吞吐。4,309 个 kernel 在 launch API 返回前已经 START，450 个已 END；signed offsets 必须保留，不能把负数裁成排队时间零。

## 可运行提取器与 schema

入口为 `tools.predictor_validation_20261027/physical_progress_evidence_v26.py`，输入 `--supervision artifacts/predictor-context-initial-transition-v13-20261028-training02/report.json`，输出必须使用新的 `--output` 目录。`owned-kernel-prefix-labels.jsonl.gz` 保存全部内核，`launch-signature-vocabulary.json` 保存 627 种 launch 描述。

`prefix_features` 只使用 cutoff 之前的原始 API/context/START/END 事件：owner、请求与语义 phase、signature、API entry/return、是否已 START/END、已 START 未 END 数量及当前未闭合 context。后续 SAVE 配对、错误和未来 END 不得反向修改前缀。多个 context 同时打开保持 ambiguous。

`observation_labels` 才保存完整 START/END、signed lag、生命周期、成对 context 诊断和生命周期分区。同 stream 上一 kernel 的 END、dependency certification 属于单独诊断。描述符含 grid/block/register/shared memory，但这些不是服务需求或吞吐。

该前缀依据离线 timestamp 重建，尚未证明 CUPTI 事件交付延迟或描述符在实际预测时可见。training01 保留原样，其源码 snapshot 同目录可核验；它包含未来配对 context 的特征问题，不应继续使用。training02 已纠正该问题。

## 两个最小候选均未通过

v27、v28 都固定 SHA256 modulo 32 选择 24,186 个 kernel，按整个 capture 做两折留出，不能把 kernel 数当作独立训练数据量。目标是 launch API return 后的 START/END 首达 wall time，包含等待和抢占，不是执行服务。

| END 留出指标 | signature baseline | v27 状态 residual | v28 加 issued queue |
| --- | --- | --- | --- |
| capture2 中位绝对误差 | 78.429 us | 66.529 us | 57.712 us |
| capture2 平均绝对误差 | 12.646 ms | 28.845 ms | 42.090 ms |
| capture1 中位绝对误差 | 74.173 us | 63.118 us | 55.438 us |
| capture1 平均绝对误差 | 13.198 ms | 30.567 ms | 43.240 ms |

v28 的 END P95 绝对误差分别为 220.510、234.725 ms。典型值改善，但尾部恶化，**两者均不具备替代主预测器的资格**。模型、完整 fold 对照和源码分别保存在 `predictor-kernel-endpoint-v27-20261028-training01`、`predictor-issued-queue-endpoint-v28-20261028-training01`，不能将这些结果称为独立请求预测改善或分流收益。

v28 issued queue 只包括已返回的被测 launch API 到已观测 END，区分 owner/stream；API 尚在执行、未测 copy/memset/event 依赖不在其中。它是可见 issued 未完成数量下界，不是完整 ready queue。

## 替代模型与 whole57 所需证据

执行状态至少区分 API issued、dependency blocked、已 START 未 END、completed。事件驱动预测需要在每次 arrival/issue/dependency-release/START/END 上，先处理旧状态下已经存活的区间，再更新条件首达分布。可以从 signature empirical survival/KM 与可见 issued 队列的分层条件 residual 开始；不能再用 held==0 强制服务率为零，也不能把 kernel 生命周期或 occupancy 当作吞吐。已完成 END 释放真实 DAG 依赖；排队等待只计一次，不再次当作 service。

whole57 v23 应在现有同次 raw/plan/context 下保存完整 request lifecycle；native launch/copy/memset、stream/event/sync 的依赖与身份；API entry/return 和 GPU START/END；context 原始事件及未知语义；公共时钟转换与误差；observer delivery 时间及丢事件信息。只有 native DAG 完整时才认证 dependency-ready；light engine 的 ready frontier 不能代替 issued backlog。现有 v23 数据不足的字段必须显式 UNKNOWN。

训练应覆盖 solo、pair、triple、background、输入长度和启动偏移的自然变化，保留完整公开到达计划，不加入任意全局一/二请求 permit。验证至少按 capture 和输入分离，并依次报告 START/END survival 与尾部校准、请求完成/全过程队列/P95、候选排序与 ΔU 符号，最后才做六项评分内的端到端分流对照。

计算/显存互补需要可核实资源需求或独立训练的 phase 条件时延，描述符不自动提供该能力。batch、公共权重子矩阵和 cache 复用需要真实后端以及单独的开销、驻留与收益画像。计数器权限不可用时保持 UNKNOWN，不编造数据。
