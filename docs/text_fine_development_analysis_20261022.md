# text1.85 小份额开发实验终态分析

工具：`tools/spillover_context_20261022/analyze_fine_development.py`。只接受终态 queue，固定核验已声明 seed107 不分流/1⁄32、seed108 1⁄32/不分流四条件。包括未执行、失败、资源不全或评分不合格的条件，不能因无收益删去其中一项。

```bash
python tools/spillover_context_20261022/analyze_fine_development.py \
  --packet artifacts/static-spillover-study-20261022/text185-fine-development-v1/experiment-packet.json \
  --queue /absolute/path/to/queue-progress.json \
  --output /absolute/path/to/new-analysis-directory
```

输出目录必须不存在。不会加载模型或调用 GPU。

核验包括 packet bindings、原 raw SHA、声明 job、静态模型池/计分参数、完整 executed identity SHA、native checkpoint manifest、OCR绑定时扩展的文件 allow-list、GPU UUID、实际记录中的原到达 trace/输入 SamplePool/扩展 alias SWRR、原生响应身份与协议、服务与排队时序恒等、内存起止覆盖及独立 whole-board 测量/启动能量积分。

实际 native output 用冻结答案重评分，包含 OCR prediction 的原 page_ocr_text 规范化，与冻结 quality/evaluate_run 的 Q 精确对齐；整个冻结 evaluate_run 结果需与 raw 保存结果一致。审计失败保留 traceback 与 raw 状态计数，不自动作数值修复。

每个 seed 的两路方案必须有相同路由无关到达/输入 cohort digest。只有双方原效用 eligible、资源审计通过、干净退出，才报告可比较的 ΔU。报告六项 norm/contribution 差值、诊断 U 与稳定性原始证据；不可比较时 ΔU 保持 null，诊断算术与失败仍保留。

每个 alias 报告 text 输入计数、真实服务总和/到达时长、measurement end backlog、drain后未完成数、到达尾段25%的 host presubmit／worker queue／e2e、完整 host/worker ledger分解。两文本副本工作量差异单列。服务总和/H 是实际有限样本工作量，包含 drain完成的 cohort 请求，不能当作正式 rho 或可持续容量。

1⁄32 小权重会改变原 SWRR 中两个专用副本收到的输入序列；有限32输入池与输入服务时间不等会产生工作量重分配。观察到的收益可能同时来自通用承担请求、专用输入重分配以及共同运行服务变化，不能全归因于容量分担。

验证使用保留的 text2.6 half-split 成功 native run（1631 请求）测试完整审计、精确重放、Q独立复算、原 evaluator全对象一致，以及篡改输入/alias/时间/服务证据和缺完成输出的拒绝。另有 queue helper mock 测试错误 UUID、GPU占用、内存/能量无效停止，以及效用不合格但资源有效仍可完成声明配对。

这是开发阶段四条件比较，不是算法已自然选中部署、完整七模型池最优证明、预测器通过独立验证或正式效率实验。独立验证 seed81/82 和确认 seed71/72/73 保留。
