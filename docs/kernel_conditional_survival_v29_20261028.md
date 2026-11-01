# 事件端点条件 survival：v29 训练诊断

入口为 `tools.predictor_validation_20261027/kernel_survival_candidate_v29.py`。输入 `--queue-report artifacts/predictor-issued-queue-endpoint-v28-20261028-training01/report.json`，输出必须另建目录。推荐结果是 `artifacts/predictor-kernel-survival-v29-20261028-training02/`，保存全量模型、两折训练模型、151,695 个 landmark 标签、完整留出指标和同支持集合对照。training01 与对应 implementation snapshot 保留，training02 仅增加 fold model 保存及同支持集合对照，算法未根据留出误差调参。

本次仅用原两次 training capture，固定原 SHA256 modulo 32 的 24,186 个 kernel，未启动 GPU，也未读取 validation 标签。kernel/landmark 高度相关，独立单位只有两次 capture；这些 capture 还共用输入模板。

## 已实现的模型

1. **signature-origin KM**：在 launch API return 后，按 owner/phase/launch signature 拟合 START、END 首达 wall time 的 Kaplan–Meier survival。在后续事件已等待 age 而未完成时，使用 `S(age+r)/S(age)` 更新剩余时间分布，不能重新从原始均值开始倒计时。
2. **landmark-queue KM**：在相同 signature 下增加事件种类、当前同 stream 已返回 launch 且未 END 的 issued 数量桶、是否已经 START，直接拟合该 landmark 的条件剩余 wall time。事件包括 launch return、后续第一个观测 issue、自己的 START（仅 END 风险）、100us/1ms/10ms/100ms elapsed checkpoints。

队列只统计被测 kernel，缺失其他 CUDA 操作及完整 native 依赖，所以始终是 issued 下界，不能称为 ready 数。当前 context 不决定是否允许物理端点发生。模型未估算 SM service、reference-work rate、资源份额或吞吐。

观察到 API return 前 START 的 4,309 个、已 END 的 450 个 kernel，其 signed offsets 保留在原始证据中；对已经发生的目标不生成伪零时长。KM 支持右删失，事件/删失同刻先处理事件风险；已存活截断、未知/稀疏组、超出经验支持和无法识别的尾部分位数均有显式行为。每组至少五条观测；未知不回退为跨 signature 均值。

行政观察窗口事先规定为每个 full capture 最后一个被测 launch API return。未来物理端点仍保留在 v26 原证据中，只在本次窗口内被截为右删失。**固定 sample 本次没有实际删失（0/151,695）**，因此只能说删失代码经过解析测试，不能说真实删失校准已经验证。独立删失假设也未证明。

## 留出结论：典型误差降低，尾部仍不可靠

以下比较均使用两种方法同时有预测的同一集合，预测指标是 END 的中位数剩余时间。

| 留出 capture / 事件 | 共同支持数 | signature 中位绝对误差 | landmark queue 中位绝对误差 | signature P95 绝对误差 | landmark queue P95 绝对误差 |
| --- | --- | --- | --- | --- | --- |
| capture2 / API return | 10,755 | 76.19 us | 48.38 us | 58.27 ms | 58.98 ms |
| capture1 / API return | 10,779 | 79.01 us | 45.39 us | 51.68 ms | 49.64 ms |
| capture2 / 下一 issue | 7,323 | 121.10 us | 89.39 us | 121.60 ms | 121.06 ms |
| capture1 / 下一 issue | 7,238 | 113.48 us | 86.16 us | 142.12 ms | 142.14 ms |

API return 的 95% 分位覆盖从 signature 的 93.95%、94.47% 降为 queue 的 92.64%、92.01%。下一 issue 的 queue 覆盖为 91.89%、92.00%。已经存活 100ms 的 END 队列分层与 signature 条件 survival 在共同支持上相同，95% 覆盖仅 91.18%、89.12%；该尾部没有因加 queue 桶而改善。

所以 **v29 未通过主预测器准入**。它提供了可运行的首达 survival 基线和失败模式证据，不能宣传为请求排队/P95/ΔU 改善。完整 report 同时报告每组 coverage、UNKNOWN、Brier 固定时限评分和右删失下 coverage/Brier 界；有删失时，uncensored-only 绝对误差不代表总体尾部准确性，比较时也不能悄悄改变支持集合。

该实现是事件 landmark 原型，没有把随事件变化的风险连续积分进全过程 engine；它不具备反事实 CUDA issued 过程、完整依赖 ready 预测或 CPU/GPU observer delivery 契约。这些仍需 whole57 新训练域支撑。

## 与真实 whole57 输出对齐

与 gpu_environment 子对话核对后，新增外部只读入口 `whole57_endpoint_contract_v29.py --review …/balanced/report.json --output …`。推荐结果是 `artifacts/predictor-whole57-endpoint-contract-v29-20261028-training02/report.json`。它直接读取小型 review/raw 和 199MB SQLite schema，用 streaming SHA 核验，不加载数 GB `parsed-measured.json`，不修改采集器和 reviewer。

实测 whole57 light/balanced 是 57 measured +12 priming，review PASS，原 review SHA 为 `b16ef5367798c8406f1c9cd1b4d464f1b99a9fe965d17984adced29269611a3b`。SQLite 中 kernel/runtime/NVTX/memcpy/memset/synchronization/CUDA-event/context-switch 的所需列全部存在。context 原表名是 `GPU_CONTEXT_SWITCH_EVENTS`，字段为 timestamp/tag/contextId/globalPid/gpuId，不能假定是 start/end duration 表。

现有 raw 包含 phase_start_ns、deadline_ns、arrival_loop_end_ns、observation_end_ns、lifecycle_boundaries、clock_anchors、ready。现有 native-first-api-kernel 只覆盖首端点，完整 kernel/API/copy/event 链需直接从 SQLite 流式提取。表和列存在不证明依赖 DAG 已完整，也不证明 GPU-ready。

最小新增端点契约保留 capture/request/owner/correlation/stream 身份、API entry/return、kernel START/END、事件或右删失标志、同 native 域观察终点、timestamp clock、observer delivery 或 UNKNOWN、dependency certification 或 UNKNOWN。raw perf_counter 的 observation_end 不能直接用作 native GPU 删失时间。session systemClockNs 不能替代有误差的跨钟校准；现有 causal intervals 也无 drift/device calibration 上界。diagnostics 可继续审核事件丢失与采集边界，但本次 schema inventory 未认证 loss completeness。

observer delivery 未采集，完整 DAG readiness、native observation-end certification、跨钟误差上界和 runtime launch 描述符可见性仍为 UNKNOWN。后续应先流式提取 native 事件链、证明依赖/身份/观察窗口，再做按独立 capture 与输入划分的动态风险/请求完成验证，不能把旧训练域的 KM 直接搬到新 domain。
