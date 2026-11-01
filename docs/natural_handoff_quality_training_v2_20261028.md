# 两次自然执行训练：交付边界与质量期望

新三域训练的两次独立 natural capture 各有 8 个完成请求，strict v3 和
三域工作对应审核全部通过。`natural_lifecycle_fit_v2.py` 与
`natural_quality_fit_v2.py` 只读取这些完成的训练记录；smoke 和旧独立验证
没有进入拟合。原 raw、域身份、各次采集 run ID 均保留。

## Handler END 后的不同边界

产物：`artifacts/predictor-gpu-20261027/natural-lifecycle-training-v2-20261028-two-captures/fit/`。
`lifecycle.json` 为原 `LifecycleProvider` 格式，16 个观测形成 6 个
family/task/input 配置；文本配置各有 4 个观测，G2 与 OCR 配置各有 2 个。
每条原始联合延迟向量仍保留，条件剩余延迟模型可读取。

平均 parent receipt 延迟约 0.493–0.837ms，WORKER_READY 延迟约
0.374–0.626ms，均相对真实 handler END。它们是不同事件；共享执行下
permit release 是不适用，不填成零。没有从下一请求 START 反推 readiness。
源绑定的本地 Linux perf-counter 证据位于父目录 `clock-evidence.json`，
并不声称 CUDA 与 host 的微秒级校准。

只支持对应训练输入的经验均值，不证明负载无关、尾部正确或新输入泛化。
三域的原 context 不相同；与 light 内核模型组合时必须使用单独可审核的
域桥接适配，不能把 lifecycle 的 hash 修改为 light hash。

## 原质量单位下的期望

产物：`artifacts/predictor-gpu-20261027/natural-quality-training-v2-20261028-two-captures/`。
先用原 catalog/reference 和原 scorer 对当次实际输出计分，再按配置训练
每请求 numerator 与 denominator 的算术均值，保存为预测器要求的
`expected_per_request_totals`。没有把训练请求总数当作单请求数据。

两个训练 OCR 输入实测分别为 10/187 和 4/790，因此任务 fallback 的
每请求预期总量为 7/488.5。预测最终 Q 时仍按 offered 请求累加这些总量后
相除，不能平均逐请求 CER；CER/WER 大于一也不会截断。
文本两个训练输入的原质量损失分别为 1 与 0，fallback 为 0.5。
这些是极小训练域的观测，不是质量提升结论。

`by_input` 只含训练输入；fallback 明确假设未来同任务输入来自相同分布。
不会读取未来真实输出或验证 reference 来预测 Q。image/ASR 尚未训练，
相应质量配置没有默认补值。

两个产物各保存 gate 的 training-index。合并索引时按真实
`(run_id,request_id)` 去重；相同原请求被多个模型使用，不能伪造新的 run。
拟合报告的 `physical_accuracy_validated=false` 保持不变。

CPU 检验：handoff 专测与原 lifecycle 测试共 20 项通过；质量单位与原
Q 总量测试共 6 项通过。它们检验代码行为，不是 GPU 泛化验证。
