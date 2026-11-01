# text26 预声明输入暴露与固定画像质量审计

本阶段仅使用冻结packet、runtime-inputs.jsonl及冻结v5的make_trace、expand_routes、SamplePool、Router和质量预测器。没有读取开发run.json、progress、原生输出或独立验证数据，没有启动GPU；没有更改seed、路由、输入、画像或评分。结果属于 **METADATA_DERIVED_NOT_NATIVE**，预测质量不作为实测U。

工具：[audit_text26_predeclared_exposure.py](/root/autodl-tmp/generalist-specialist-project/tools/spillover_discovery_20261022/audit_text26_predeclared_exposure.py)。产物及中文表格：[report.md](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/text26-predeclared-exposure-v1/report.md)，机器结果：[report.json](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/text26-predeclared-exposure-v1/report.json)。每job文件保留逐计划请求与每输入计数，包括零计数；protocol/manifest记录输入与源码SHA。新目录不可覆盖。

重建以各job identity中的seed51、H600、Poisson及workload为准。测量窗口的SamplePool和Router均从初始状态新建，与runtime.window一致；priming/warmup不消耗测量窗口输入序列或路由状态。四条件重建出的trace SHA都与冻结identity相同；每任务直接汇总已分配输入的画像误差分子／分母，与独立调用冻结predict_quality所得损失核对一致。质量取packet内的历史逐输入画像，不打开其source路径指向的历史原生文件。

| 条件 | 专用text路径覆盖 | G2/text覆盖 | 专用各alias覆盖 | 有限轨迹alias周期 | 固定画像预测text精度 | 固定画像预测Q损失 |
|---|---:|---:|---|---:|---:|---:|
| n2、专用4/7 | 32/32 | 32/32 | 32、32 | 7 | 0.693026152 | 0.305501410 |
| n2、专用1/2 | 16/32 | 16/32 | 9、9 | 未发现≤256的精确周期 | 0.688667497 | 0.309810347 |
| n1、专用1/2 | 16/32 | 16/32 | 16 | 2 | 0.688667497 | 0.309810347 |
| n3、专用4/7 | 32/32 | 32/32 | 32、32、32 | 21 | 0.692403487 | 0.306116973 |

每条件计划1606条text请求。半分流n2专用各alias并非简单各8输入：text#0收到402请求，8个主输入加1个仅一次的输入；text#1收到401请求，同样8个主输入加1个仅一次的输入。第5个task_position就已偏离首4步重复序列，不能把理想有理权重的周期当成实际浮点SWRR周期。路径整体仍各16/32，两个专用alias覆盖集合相交2个输入。此处只记录精确重建结果，不修改Router的浮点实现或tie规则。

Q是各任务原生质量**损失**（text/image为错误率，OCR为CER，ASR为WER）按冻结配置的任务到达率加权，越小越好；表中text精度仅为1−text loss。不能将不同任务指标直接称为统一精度，也不能把Q误称为六项总效用U。固定画像沿用历史输出，实际并发／副本／失败请求下的输出复现尚未验证。各条件的Q有差异本身不证明模型质量变化：确定性输入循环与路由会改变输入分配，而有限窗口末尾也会产生计数不整齐。

这份预声明暴露审计是后续原生诊断的参照：先核对实际计划cohort与此序列是否一致；再对有效完成请求区分offered覆盖与completed覆盖，缺失完成保留UNKNOWN。服务均值、倍率与输出质量跨条件比较，应报告共同输入交集和逐输入重复数。4/7覆盖32并不代表每次输入发生时GPU共驻／背景负载相同；n2与n3还同时改变副本数与驻留资源，因此无法凭这些非随机配对条件单独识别因果干扰。

复现命令：

```bash
python tools/spillover_discovery_20261022/audit_text26_predeclared_exposure.py \
  --output /absolute/path/to/new-exposure-directory
```
