# H600两条预测的事件预算耗尽：数值固定点

冻结的H600训练CV报告为 `background-statistical/run-aware-v14-H600-report/report.json`，SHA `c319441c89d8e085274b0aebeaff199f8a95b056f1b24424ccb64ffb8fe5ab42`。627602的shared_spill_low/high均返回 `FULL_HORIZON_EVENT_BUDGET_EXCEEDED_NO_TRUNCATION`。原100000预算、模拟器、模型、阈值及报告均未修改。

新增 `h600_event_forensics.py` 只以同一公开plan的到达、输入及静态路由，原heldout fold模型和同fold资源模型重新运行模拟；没有读取实际未来START/PHASE。使用Python trace读取冻结forecast局部变量，不写clock/progress等状态。在连续九次完全相同的零delta状态后终止诊断前缀。结果位于 `background-statistical/run-aware-v14-event-diagnostic/diagnostic.json`，完整保存前后事件、progress、rate、duration、当前clock ULP及源收据。

| 路由 | 首次停滞事件 | clock (ns) | 已完成请求 | 已提交未来请求 |
|---|---:|---:|---:|---:|
| shared_spill_low | 5626 | 571791700004.1077 | 1140 | 1143 |
| shared_spill_high | 5430 | 552417363854.8057 | 1100 | 1104 |

两者均停在G2的finalize：条件service为0.4716724721000537ms，余fraction为1.148153794261475e-10，大于模拟器的1e-10完成判据。余时长为5.415525384903663e-05ns，小于当前clock半个ULP；clock ULP为0.0001220703125ns。因 `clock + remaining_duration == clock`，每轮boundary等于clock、delta等于0，progress不减少，phase永远不能完成。future arrival尚未到达，另一路text decode仍活动，所有后续事件状态逐项相同。这是数值固定点，不是需要100000个真实有效事件的负载复杂度。

最小Python浮点复现：

```python
import math
clock = 571791700004.1077
service_ms = 0.4716724721000537
fraction = 1.148153794261475e-10
remaining_ns = fraction * service_ms * 1e6
assert fraction > 1e-10
assert remaining_ns < math.ulp(clock) / 2
boundary = clock + remaining_ns
delta = boundary - clock
assert delta == 0.0
assert fraction - delta / (service_ms * 1e6) == fraction
```

前一轮完整finalize的理论时长471672.4721000537ns，rounded边界相减得到471672.47204589844ns，留下约5.4e-05ns尾量。该尾量低于绝对clock的可表示分辨率，却稍高于固定fraction判据，两个判据不一致。

建议新增版本的事件模拟器保持事件身份：计算每个phase的相对完成时长及最早完成alias，在选择segment/arrival/window/drain最早事件时携带“哪个phase-end事件被选中”；推进时间后，只有被选中的phase-end alias精确置为完成，其余alias仍按实际delta扣减。它避免把“大数相加后再相减”的浮点残差当成未发生的phase-end。应保留原资源/输入/排队/准入逻辑及预算，不通过调高预算、扩大通用fraction epsilon或过滤请求处理此问题。若多个phase-end严格并列，均应完成；arrival/window/drain与其并列时保持原事件处理顺序。

实现前需独立审阅数值方案；至少覆盖本最小数值复现、超短阶段配大clock、两alias并列完成、完成与到达/测量截止/drain并列、未选中活动phase不提前完成、无事件推进时明确UNKNOWN。修复应使用新模块和新预测来源收据、在新目录对完整八条H600主流重新预测，原失败报告保持原样，门槛不变。当前仅提出修复，不自动替换或重算原报告。

资源缺项与此问题不同：heldout627601资源模型未见G2 ASR+OCR且无text的任务组合，实测仅约0.571ms；本次fresh627601四候选的resource unknown_reasons均为空，故当前资源缺项未触发这些预测。627602两条预测的UNKNOWN由上述数值停滞导致，不能以H90或全训练资源回填解决。
