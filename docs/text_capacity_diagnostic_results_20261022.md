# 第一批6次容量诊断：执行结果与下一步

六次GPU运行及独立审计均已完成，1152/1152条文本测量成功，无重试。
固定驻留为ChatQA文本专用×2、MiniCPM-o4.5 GPTQ（G2）×1、OCR×1。
每个活动alias接收相同32条原文本输入，每条重复3次；两块分别按ABC和CBA顺序新启模型。
全驻留21条路径初始化、30秒闭环warmup与正式测量分开记录。

## 本批数据

| 状态 | 第1块总测量秒 | 第2块总测量秒 | 测量请求／块 |
|---|---:|---:|---|
| A：两专用活动、G2空闲驻留 | 81.13 | 82.31 | 两专用各96 |
| B：G2活动、两专用空闲驻留 | 48.47 | 47.33 | G2 96 |
| C：两专用与G2共同活动 | 125.92 | 126.91 | 三alias各96 |

六次安全显存34307.61–38000.09MiB，小于49140MiB。独立核对了模型、协议、
权重、输入、原生响应、每alias单在途、时间账本、进程退出、显存和整卡能量。
全部诊断U=null，闭环吞吐不是开环稳定容量。

## 最重要的新发现

两个C条件中，两专用最后handler在约86.60/87.77秒结束，G2继续处理至约126/127秒。
G2真正与所有专用存在正handler重叠的请求仅18/14条，覆盖18/14个不同输入；
其余78/82条无三alias共同handler重叠。第2块有少数请求在两个专用结束时刻之间，
不能将“无三alias重叠”全部等同“两专用均空闲”。

在有正共同重叠的请求中，对照B的同输入、同重复轮次，G2的服务时间比中位数
分别为9.49和11.25；无三alias重叠请求的相同比值约1.00和1.01。
慢段与共同活动相关，**并不唯一证明GPU调度、缓存或其他物理机制是原因**。
活动状态与时间、顺序、轮次共同变化，尚未做因果识别。

更容易误判的是：两块中32/32个输入的“三次服务中位数”都取自两个专用已经
结束后的尾段。因此，整段逐输入中位数C/B接近1，并不能说明持续共同活动没有
代价。这批诊断识别出了“快尾段掩盖慢联合段”的问题；若直接用该中位数校准，
算法会再次高估通用可接流余量。

多数配对慢请求的prediction和生成token与B相同：正重叠组为17/18和13/14，
整块为94/96。第1块最慢G2请求17.44秒，与B的prediction及24个生成token相同。
已观测输出长度差不足以解释这些慢请求。缓存回收只记录是否执行及显存数值，
没有独立耗时，不可据此断言回收导致变慢。

原生输出质量也需保留：两专用各次EM均0.71875；G2的B为0.65625/0.63542，
C两次均0.63542。原冻结质量记录不改；这些输出差异不代表负载相关质量模型
已经建立，正式效用仍须重新评分实际输出。

## 这批之后怎么推进

当前结论为：**第一批诊断完成；尚不足以进入两档开放负载实验或正式搜索。**
原因是G2完整输入的持续共同活动覆盖不足，正式背景路径的全输入／联合活动也
未由本批建立。不能从三种闭环状态拟合连续负载／分流函数。

下一小批应先研究持续共同活动：每个alias完成其目标cohort后，继续提交预声明的
均衡支持请求，直到所有alias的目标cohort完成，再停止并排空。目标与支持请求
全部保留、分别标识，不把支持样本当独立重复或正式效用；设置有限wall timeout、
支持请求上限和原资源门禁。该方案需新版本运行器、CPU核验和冻结包，不能改写
本次结果。完整共同活动覆盖和跨块重复的判断在新运行前声明，失败如实保留。

持续共同活动证据足够后，另核对G2/image、G2/asr、OCR/ocr完整输入服务和正式
背景联合活动。随后才用精确正式SWRR输入重建、整池rho与有限FIFO风险情景，
选择两档负载及0/1/8/1/4份额。新预测器冻结并独立验证后，再接联合MILP及同域DFS，
扩展七模型池证据并做强对照。原稳定性、rho≤0.8、六项权重、质量及模型池不变。

## 证据与更正

- [六次独立审计v2](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/text-capacity-diagnostic-v1/all-six-independent-audit-v2/audit.json)
- [原始诊断概览](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/text-capacity-diagnostic-v1/analysis-v1/report.md)
- [共同活动／尾段逐请求诊断](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/text-capacity-diagnostic-v1/joint-service-diagnostic-v1/report.md)
- [时间与服务对照图](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/text-capacity-diagnostic-v1/joint-service-diagnostic-v1/joint-service-phases.png)

全六审计v1曾错误比较历史任务协议SHA和当前worker合并协议SHA，导致工具误报。
原源码和失败输出保留，修正收据在`all-six-independent-audit-correction-v1/`；
最终v2核对当前合并协议并全部通过。六条raw、冻结runtime、评分和稳定性未改。
相关51项CPU检查通过；实际GPU数据及审计证据均单独绑定SHA。
