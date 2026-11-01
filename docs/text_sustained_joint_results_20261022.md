# 持续共同活动诊断：真实慢段与后续分流尺度

本批按新冻结包执行，第一组B完成、第一组C触发600秒测量上限；队列按预声明规则停止，第二块两组未启动，无重试。GPU已空闲。原模型池、副本、质量入效用、六项权重、rho≤0.8及正式稳定性未改。此处600秒是有限诊断的执行上限，不是新增正式稳定性准入。

## 实际执行与截止状态

固定驻留：ChatQA文本专用×2、MiniCPM-o4.5 GPTQ通用G2×1、OCR×1。B仅G2活动；C两专用及G2活动。每活动alias预注册相同32条文本输入×3轮、共96条primary；先完成者继续处理独立预声明的support，直到所有primary完成。支持请求不增加独立样本，不进入正式效用。

| 条件 | 截止内primary | 截止后返回primary | 未发布primary | 测量时长 |
|---|---:|---:|---:|---:|
| B，G2单独活动 | G2 96/96 | 0 | 0 | 46.37046秒 |
| C，持续共同活动 | 两专用各96/96；G2 76/96 | G2 1条 | G2 19条 | 600秒上限触发 |

C共发布1205条support，其中1203条在截止内完成，另2条在干净退出过程中返回。G2第77条响应同样在截止后返回。三条响应保留，但不得混作截止内完成。19条未发布primary没有服务时间或输出，保持UNKNOWN。运行器实际记录结束为600.000483秒，审计按预声明600秒边界独立切片。

第一B独立成功审计PASS。C的独立失败审计状态是`EXPECTED_WALL_TIMEOUT_EVIDENCE_VERIFIED`，表示超时事实、原生身份、账本与可核验测量证据一致；研究门槛仍为`FAILED_INCOMPLETE_PRIMARY_COHORT`，cohort_pass=false、U=null。第二块未启动，不能声称跨块可重复性通过。

## 核心发现

C截止内76条G2 primary覆盖全部32输入，每输入至少有2条真实三方handler共同交叠比例≥0.90的观测。76条交叠比例最小约0.9446、中位约0.9952；因此本批已观察到持续共同活动中的完整输入二次覆盖，但没有完成原协议要求的三次及第二块。

对照B的同输入、同重复轮次，76条C/B服务时间比值中位数为**14.8235**。C服务时间中位约6779.58ms，均值7839.28ms；配对B服务时间中位约408.26ms。这是删失开发样本的配对描述，不是总体精确性能倍率、因果证明、独立验证或可迁移容量。

本批去除了先前“专用先结束，通用三次服务中位数由快尾段占据”的主要覆盖缺口，显示通用接收端在这个持续共同活动状态下处理很慢。不能由此声称任意轻负载混部都慢14.82倍，也不能断言唯一物理原因是GPU争用或缓存。低分流下共同活动的占空比、时间和顺序仍会改变，需要真实开放负载测量。

两个专用截止内primary EM均0.71875；G2在截止内76条上的EM为0.63158。B完整96条EM为0.65625，两者输入/轮次计数不同，不能直接当整池精度下降的因果证据。实际输出及逐请求评分全部保留，冻结历史质量未替换。

C已采样安全显存峰值38106.74MiB，低于49140MiB；但失败退出路径最后一条显存采样距测量终点0.316秒，该段显存UNKNOWN，不能声称全窗资源完整。整卡功率完整覆盖测量窗口，独立积分约214609.95J。未来另版运行器需要在失败finally关闭monitor前采样；不改写本次源或原始数据。

## 已完成的CPU尺度筛查

新敏感性分析保留一个明确的不完整开发情景：仅截止内、共同交叠≥0.90的primary，逐输入取已有2或3次服务中位数，再按32输入等权平均。G2得到7847.27ms，专用两个alias分别861.54/837.91ms；专用敏感性使用较大的alias均值。第三轮删失、仅单块和背景路径缺失均未消失，该表不能注册为已验证画像或概率上界。

还有一个时间覆盖限制：专用primary在前约84秒结束，G2观测横跨约596秒。两个向量不是同一平稳时段的联合服务表；专用后续support的漂移尚未进入此表，不能把两者组合当作任意份额下同时成立的保守容量上界。后续筛查前须核对support时间分层，支持数据仍不成为独立样本或正式效用。

在这个观察情景中，按原capacity_factor计算，G2文本工作量项为`lambda_text × share × 7.84727`。以下用lambda_text=1.8作为**CPU尺度示例，未选定实验负载**，且尚未加入未知背景；正式rho保持UNKNOWN：

| 通用文本份额 | 情景中G2的文本工作量项 |
|---|---:|
| 1/4 | 3.5313 |
| 1/8 | 1.7656 |
| 1/16 | 0.8828 |
| 1/32 | 0.4414 |
| 1/64 | 0.2207 |

306个CPU网格点覆盖lambda0.5–3.0（步长0.05）与份额0、1/64、1/32、1/16、1/8、1/4，**全部保持NO_GO**，没有选择负载、路由或预测效用。这一分析解释为什么不应继续默认1/8和1/4能在专用高压力附近承接分流；也不证明小份额一定有收益。原份额保留CPU压力对照；只有独立风险证据支持时才值得安排GPU。

1/32、1/64超出现有正式K8（分母1至8并集）路由域，目前仅属于开发敏感性网格。若后续证据支持需要将它们正式加入，必须另版本统一路由域、精确SWRR输入/质量、预测器、MILP及DFS，再在同域比较最优；不能只改实测路由便宣称原算法已选到该解。

## 下一阶段调整

不再为补齐第三轮反复跑饱和闭环，也不启动原先默认的两档×0/1/8/1/4六次开放实验。先核对已记录support的时间分层及输入覆盖，再用明确局限的逐输入压力情景做小份额的精确SWRR暴露与FIFO风险筛查，明确新开发域；0分流必须保留同负载对照。尚未覆盖的G2/image（64输入）、G2/asr（32）及OCR/ocr（32）按背景设计逐路径补证，不把21条priming当全池覆盖。

只有候选在完整背景与压力情景下有明确风险依据，才封一个有限开放开发包，优先0分流与一个有依据的小份额，必要时再加一个相邻份额。其任务是检验真实低频背景、接收端余量、原生质量和效用是否出现可复现改善，不保证混合获胜。之后才冻结并独立验证新的预测器，统一搜索域，与DFS核对。正式搜索、DFS与最优证明仍未启动。

## 证据

- [失败独立审计](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/text-sustained-joint-diagnostic-v1/failed-attempt-independent-audit-v1/audit.json)
- [首组B独立审计](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/text-sustained-joint-diagnostic-v1/first-baseline-independent-audit-v1/audit.json)
- [CPU敏感性网格](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/text-sustained-joint-diagnostic-v1/partial-risk-sensitivity-v1/screen.json)
- [严格截止窗口报告](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/text-sustained-joint-diagnostic-v1/failure-window-v1/report.md)
- [目标／支持请求时序图](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/text-sustained-joint-diagnostic-v1/failure-window-figures-v1/primary-support-timeline.png)
- [同输入服务比值与共同活动图](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/text-sustained-joint-diagnostic-v1/failure-window-figures-v1/generalist-paired-service-overlap.png)
- [背景覆盖设计](/root/autodl-tmp/generalist-specialist-project/docs/text_background_joint_coverage_design_20261022.md)

原始队列、attempt、审计、未执行job和失败均保留；不选择较好轮次替代本批失败。
