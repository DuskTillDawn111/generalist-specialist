# 冻结预测验证失败与下一步

2026-10-28。known-input sealed03 的四候选 GPU 采集、原 observed adapter、正式 gate 和最终源码复核均已完成。228/228 测量请求完成，48 次原预热完成，所有实际 token/forward/beam 工作匹配预封假设，质量与资源证据完整。源码冻结期间 361 份生产 Python 字节未变。门槛结果为 **FAIL**，不是缺数据造成的 UNKNOWN。

| 候选 | 封存预测 U | 实测 U | 服务 MARE | E2E MARE | 等待深度 MAE |
|---|---:|---:|---:|---:|---:|
| shared_balanced | -1.44218 | -1.94054 | 43.84% | 76.59% | 1.119 |
| shared_skewed | -2.24109 | -2.31457 | 49.86% | 26.15% | 0.218 |
| shared_spill_low | -1.63733 | -2.58202 | 43.80% | 64.63% | 1.455 |
| shared_spill_high | -1.41901 | -2.58532 | 48.73% | 88.01% | 2.267 |

ΔU 符号正确率 2/3，六对候选排序正确率 2/6，原要求分别为100%。high 相对 balanced 预测 +0.02317，实测 -0.64478。没有新的最终分流收益资格。四个 shared 固定路由、同公开 known-content 新请求的局部验证，不是完整未见内容验证或原/优化执行×禁/允分流四组对照。

正式证据：

- `artifacts/predictor-gpu-20261027/known-input-dag-co-run-v6-20261028-evaluation01/gate-report.json`
- 同目录 `README.md` 与 `final-source-freeze-review.json`。
- `artifacts/predictor-gpu-20261027/known-input-dag-co-run-v5-20261028-collected01/` 保留全部原 raw、scored raw、预采集审核、逐请求实际生成工作审核和监督记录。
- `artifacts/predictor-observed-queue-decomposition-v6-20261028-final01/` 为四候选逐请求 service/queue/reader-gap 与真实队列恢复分解。
- `artifacts/predictor-frozen-pilot-error-attribution-v6-20261028-final01/report.json` 分解各原效用项对 ΔU 误判的贡献。

## 已定位的证据与仍待区分的原因

balanced 中，长文本服务预测766ms、实测919ms，短文本预测573ms、实测340ms；skewed 换副本后相同输入的偏差方向保持。父进程 receipt gap 的绝对误差很小。low 中 G2 长/短文本分别预测1234/273ms、实测3364/917ms，随后等待被显著放大。此处只做离线诊断，不用这些标签拟合或选择新情景。

仅用原训练证据已经确认：旧 host v5 按 family/task 混合长短输入和两个文本副本。两输入 DAG issue depth 几乎相同（约19900），但原自然端点对应的 issue-depth 参数短约14–16us/step、长约41–43us/step，均值约28.66us/step。这是缺失输入条件的可修复问题，不是已观测到的真实 CPU launch 分解；原端点仍只能支持部分识别的 host-supply 情景。

原 SSG 的108个语义组、328044个 kernel 和18个请求 DAG 可与两份 solo capture 的实际 launch/shape/拓扑严格对应。两份 SSG natural 中，G2 handler 中位比约2.81/2.79，prefill call墙钟比约2.82/2.86，decode合计比约1.19/1.18。阶段性干扰必须分开解释。natural call墙钟不是 GPU busy；只有一份 SSG light capture，不能当作独立 kernel 减速验证。资源状态模型与静态 occupancy 也不提供已识别的 SM/DRAM/L2 需求。

high 的 ΔU 预测误差约 -0.668，其中原 P 项贡献约 -0.561、L 项约 -0.109；Q相同，M/E贡献小。主要决策缺口是服务/等待传播造成的尾时延预测，不应通过修改六项权重掩盖。完整错误分解见上述 artifact。

## 当前实施顺序

1. 保留失败版本与独立标签；解除冻结后只增加新版本，不回写 sealed03 或重报本批通过。
2. 输入条件 host v7：按原 public content、协议/生成工作、原 DAG 与 owner 精确绑定，每组仍需至少两个独立训练 raw，UNKNOWN 不回退 family 均值。源闭包需从原 strict raws 重推参数；训练内端点拟合和 SSG 跨 raw 组件检查分别报告，不把两个 known inputs 叫做未见 shape 泛化。
3. 补齐四任务 component-generation 与 startup 接线，使用原上下文/路径、v3/v4 strict loader、完整 flow/priming 绑定及原 M/E 算术。原十二模板 CPU 诊断只验证数值与接线，不授予独立精度资格。
4. 实现并独审原子 checkpoint 续算，为之后的可见状态纠偏保留完整预算、cohort和评分窗口。预测 cursor 不能成为观测 cursor；provider选择/参数与内部堆、诊断缓存也必须绑定并核验。
5. 用原训练的严格序列/DAG对应分开诊断 host supply、sync、kernel lifetime 与阶段干扰。先明确可识别性和所需的新画像；不能整体乘一个 SSG 系数。原已看训练与当前失败验证保持各自身份。
6. 新模型和 source 稳定后重新冻结，使用新鲜请求检验服务、队列、P95、ΔU符号和排序。通过后再扩大未见输入验证、真实batch/cache/资源机制消融与最终公平四组对照。

原六项等权、rho/g(n)、稳定性与完整 offered cohort 不变；没有默认全局 cap1/cap2。goal 仍在进行。
