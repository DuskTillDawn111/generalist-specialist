# 七池9个缺失状态的已有证据审计（2026-10-22）

只读CPU审计，未启动GPU、未改既有证据/画像/评分。9状态均保持UNKNOWN；0条完整合格baseline，0条全域不可行证明。唯一可补充的完整输入诊断候选来自G1/text/n3失败运行；它不解除资源及迁移门禁。

扫描423份非当前GPU/非保留验证的原生资源profile元数据，精确匹配4个终态FAILED；其余5状态未执行。逐项SHA、覆盖、原生请求身份和完整输入列表见同名JSON。

|模型/任务/n|正式成功/返回|有效输入覆盖|既有记录/停止原因|结论|
|---|---:|---:|---|---|
|generalist_1/text/3|51/51|32/32|RuntimeError: Peak board memory exceeds safety-adjusted GPU budget|可补充32输入失败上下文诊断；全域仍UNKNOWN|
|generalist_1/text/4|0/0|0/32|MEASURED_FAILURE|UNKNOWN，不排除|
|generalist_1/image/4|0/0|0/64|RuntimeError: Warmup failed; see warmup.json|UNKNOWN，不排除|
|generalist_1/ocr/4|1/1|1/32|RuntimeError: Peak board memory exceeds safety-adjusted GPU budget|UNKNOWN，不排除|
|generalist_2/text/4|0/0|0/32|TWO_INCREMENT_THROUGHPUT_PLATEAU|UNKNOWN，不排除|
|generalist_2/text/5|0/0|0/32|TWO_INCREMENT_THROUGHPUT_PLATEAU|UNKNOWN，不排除|
|generalist_2/image/5|0/0|0/64|TWO_INCREMENT_THROUGHPUT_PLATEAU|UNKNOWN，不排除|
|generalist_2/ocr/5|0/0|0/32|TWO_INCREMENT_THROUGHPUT_PLATEAU|UNKNOWN，不排除|
|generalist_3/ocr/4|1/1|1/32|RuntimeError: Peak board memory exceeds safety-adjusted GPU budget|UNKNOWN，不排除|

## G1/text/n3：覆盖能补，合格baseline不能直接补

首轮44/44成功，32/32输入全覆盖；第二轮中断前7/7成功，合计51条。各输入仅1–3次，各alias分别覆盖17、13、16个输入，不是每alias完整32输入×3轮。51/51请求与既有n1相同input的request SHA、native protocol SHA和effective generation一致，模型spec也一致。可保留逐输入观测/中位数作为明确标记failed-context的未验证诊断清单，不能重新登记为合格资源画像。

测量峰值47245.6875MiB，乘safe_factor后52483.060942MiB>49140。profile里的safe_memory_mib=32421.037702是预热后驻留值，不能拿它遮盖测量峰值超限。全局状态继续UNKNOWN：当前GPU/运行语义迁移及低频合法路径均未证明。

## 其他失败及未测状态的范围

G1/image/n4：0/64正式输入；startup4个READY，前3个进程共9个串行warmup成功，第四进程第一个最大图像warmup-3-0 OOM。冻结runner是每进程load后逐个串行warmup，因此不是单纯测量期四路同时请求失败。该启动/priming/驻留上下文是精确失败证据；但历史设备UUID不同，且它不是对所有初始化、低频静态路由或新设备的通用不可行证明。

G1/ocr/n4及G3/ocr/n4：各仅1/32正式输入；safe peak分别49858.529510、53293.290078MiB，均超49140。最多保留单输入服务诊断，不能补全池服务或把饱和失败推广到所有低频。

G1/text/n4：因n3失败而未测；不能没有单调性证明就继承n3失败。G2/text/n4、n5、image/n5、ocr/n5：因吞吐平台启发式停止而未测；平台既不是OOM也不是不可行证明。来自G2/asr/n5或同族其他task的成功/失败都不能补这些具体n/task的服务值。

## 身份与下一动作

4个失败profile均来自GPU-533a287b-0e15-a7c8-8e58-2b21bdcbb13a；当前要求设备为GPU-8a262fe0-598c-46b7-c953-41d9da84a47e。相同型号/显存不等于精确设备身份。还需明确当前worker/source/检查点/任务协议的适用性，不能仅凭一个protocol_sha字段就断言等价或不等价。此审计未重哈希全模型权重，未认证新共驻或开放负载服务。

优先在CPU保存G1/text/n3的32输入观测与身份链，作为可复用诊断候选；其余8状态无完整目标输入服务证据。若需解除全域证书缺口，应另行声明特定n/task与目标活动上下文的合法测量或可证明排除，不修改现有61条画像，不静默删状态，不直接把4个历史FAILED纳入不可行剪枝。本文没有安排新GPU。

机器报告：/root/autodl-tmp/generalist-specialist-project/docs/seven_pool_missing_state_evidence_audit_20261022.json；SHA256=444f55febf1b6c7147d25c7c7ed752a98687e2d8b23d391bee8e4e895d1e5699。所有读取source路径和SHA在JSON逐项保留，输出auditor脚本也有SHA。
