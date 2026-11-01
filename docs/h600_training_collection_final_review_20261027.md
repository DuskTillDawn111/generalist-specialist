# H600 新训练画像：独立完整性审计结果

10 条预声明长流全部完成，12000 个测量请求全部完成并有逐请求质量证据。八条 primary 共 9600 请求是新训练数据，两条相同 trace repeat 共 2400 请求只作重复性诊断；H30 预检继续排除在 H600 拟合之外。

最终机器可复核摘要为 `artifacts/predictor-gpu-20261027/continuous-training-v14-h600/independent-final-collection-audit.json`。严格 source/event/telemetry 映射、native/scored 不变绑定、原 catalog 逐请求质量重算全部通过。总任务数为 text11100、image300、ASR300、OCR300。collection、runtime、telemetry、qualification、未完成请求、质量 annotation mismatch 和 strict source mapping failure 七类计数均为零。每条流各有 source/quality audit、原六项与稳定性诊断、输入组合诊断和实际 idle history 四项报告；最终摘要逐个核验了其收据。

十条流均通过原实际稳定性规则，状态均为 `recoverable_congested`，最终 drain 后 backlog 为零。这不是无拥塞：skewed 的两 seed drain 分别为 98.840 和 96.630 秒；其余八条为 0.387–3.138 秒。原规则、阈值及稳定性结论保留，不能把 scheduled 到达 backlog 的增长忽略为“没有等待”，也不能自行改用它替代原 host-send backlog 判定。

|相同 trace 组|相同输入/到达/路由/资源|text#0/#1 服务逐请求绝对差均值|text#0/#1 E2E P95，primary→repeat|
|---|---|---|---|
|seed627601 low|1200/1200 完全一致|26.0 / 24.2 ms|3513→3395 / 3699→3814 ms|
|seed627601 balanced|1200/1200 完全一致|17.7 / 19.3 ms|5575→5792 / 5550→5436 ms|

low 的 G2-text、ASR、image、OCR 服务逐请求绝对差均值分别为 104.0、62.2、45.5、27.2 ms。balanced 的 ASR、image、OCR 分别为 43.3、31.1、34.7 ms。两组均说明相同输入与到达不能消除真实执行波动，且 P95 改变的方向并不一致。这只是两组冷启动运行对的描述，不是独立输入样本数扩大，也未校准总体方差或置信区间。

输入混合对时间分组均值的解释不可遗漏。例如 seed627602 skewed 的轻载 text#0 三段原服务均值为 1016.7/1011.0/790.0 ms，以该 alias 全 cohort 的输入频率重权后为 934.7/940.4/939.4 ms；直接把原均值下降归因于运行年龄会误判。每条流的各 alias/input 三段计数、区间/P95、缺少支持，以及前序真实 handler END→当前 START 的 idle gap 都单独保留。固定 n≥5 只标描述覆盖，稀疏单元不替代、不回填，不改变当前 OCR 模型。

纯 CPU 审计 watcher 已完成十条并退出码 0，完成标记为 `completed-independent-training-audits.json`。它仅更新 root progress 的 `training_v14_h600`，最终已核对 `completed_main_runs=10`、`audited_measurement_requests=12000`。未启动额外 GPU，也未改冻结数值组件。

本次 PASS 的范围是训练采集完整性、质量与来源，以及原实际稳定性。预测精度、全队列/P95、ΔU 符号/排序、统计 rho 适用性及最终分流收益仍由各自固定的 CV、独立验证和后续对照决定；此摘要不授予那些结论。
