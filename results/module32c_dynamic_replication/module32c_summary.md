# Module32C：独立配对复现与效用归因

最终 Gate：**DYNAMIC_REPLICATION_INCONCLUSIVE**。完整三组=False；双方合法稳定=False；每组真实G/S=False；三组ΔU均>+.01=False。

原计划三组六臂，实际一组两臂完成。P1_D触发原32B冻结1ms路由开销验收失败后停止；其正式U与ΔU=NULL，后两组NOT_EXECUTED，无三组均值/置信区间。完整性保护通过，实验验收未通过。
Stage A CPU-only审计通过：Module32B完整文件SHA、raw、journal、worker身份/输出、GPU遥测与全部六任务U复算均合格。Text P95正贡献+.044249（总Text+.040279）主要来自handler服务变化，排队很小；原Text P95由115样本第109/110排序请求插值。具体请求、原四次触发与代数分解见 utility_attribution.md、text_tail_audit.md。

## 冻结独立配对

原Phi + Whisper-large-v3六任务池、六单副本、ASR=.46QPS、五背景各.2QPS、B3=34567.5534MiB、原质量准入不变。原32B路由源码、全部阈值、音频元数据、背压规则按字节相同；只把计量固定600s分母改成冻结窗口2400s，CPU对照复算验证，无原优化器改动。

三个预先固定新seed=320301/320302/320303，组内相同输入、Poisson到达、route/replica uniforms；相同 frozen sample pool 的新顺序/到达轨迹，不是新数据集质量验证。预定顺序P1 S→D、P2 D→S、P3 S→D，无seed替换。每臂30s warmup完全排空、原热控制、2400s正式、有界60s排空，完整cohort，原Module20A U与稳定性阈值，不稳定/不合法U=NULL。

|组|S-only U|Dynamic U|ΔU|G/S ASR|合法稳定|温差°C|
|---|---:|---:|---:|---|---|---:|
|1|-1.3852584985619942|||10/1060|False|7.0057670782143475|

## 独立组统计

完整合法三组不足，不报告预定三组均值/置信区间为有效复现证据。

## Text 尾与总效用来源

|组|Text mean贡献|Text P95贡献|ASR L/P贡献|VQA L/P贡献|诊断减Text P95|
|---|---:|---:|---:|---:|---:|

仅合法配对才列上述正式贡献；不合法配对保持NULL，另以明确标记的ineligible_task_arithmetic_diagnostic.csv提供完整六任务代数诊断。减去Text贡献只是代数敏感性，官方U仍含六任务全部原权重；没有删请求、重定义效用或执行反事实路由。全部测量和T/M分项见 per_task_utility_decomposition.csv、gpu_independent_pairs.csv与完整raw。

Text与VQA服务/等待/E2E分别统计；P95分项不能相加。Text_VQA_p95_component_attribution.csv是匹配已测请求向量的代数Shapley，并非执行过的counterfactual。直接Phi ASR overlap按同host Unix区间；不存在直接重叠也不能排除共享GPU/host资源或状态漂移。没有内核级共享资源干预，所以Text改善与路由因果机制仍不能唯一识别。


## Whisper 实际收益与 VQA 非抢占风险

|组|保留S数|G数|保留S等待均值减少ms|保留S等待累计减少s|G相对同输入S新增执行s|
|---|---:|---:|---:|---:|---:|
|1|1060|10|15.247291416037735|16.162128901|10.090752986|

G真实接纳状态、实时预测/实际handler时间、实际等待和同输入基线均保留。当前空Phi不能知道未来VQA，不可抢占执行会阻塞新VQA；下面的未来暴露均是事后诊断，路由器不读取。
- P1: G执行10次；期间新VQA到达5；VQA等待覆盖Phi ASR的请求5；VQA E2E mean/P95 S=218.559/346.180ms，D=214.406/345.187ms。

## 稳定性、环境与停止

每个系统/worker沿用原stability_v2判据，每任务一秒backlog轨迹完整记录；没有用排空截尾补出正值。吞吐分within-window和full-cohort两口径。所有600秒开发数据只用于审计，不与新2400秒窗口混作配对基线。

GPU UUID/驱动/内存与各worker Python环境冻结，温度、SM/显存时钟、功率、CPU、VRAM与外来进程逐样本保存。组内原5°C/15%热状态诊断见gpu_independent_pairs.csv，并报告绝对服务需求可能随顺序/环境漂移。Unix与perf时间仅按各自定义计算，handler含原CUDA同步与预处理，不声称纯kernel服务时间。

数据合法性、完整性或真实协作未支持有效三组复现；保留完整已尝试结果和未执行原因，不宣称失败策略的效用为有效正例。

## 本轮停止原因与未完成配对

实际执行2/6次正式GPU运行，仅P1完整两臂；P2/P3未执行。P1_D路由wrapper P95开销=1.221860ms，超过原32B冻结1ms验收阈值，触发原技术错误停止政策。Dynamic虽然完整、steady且10次真实分流，正式U与配对ΔU均为NULL，不用原始代数数值挽救验收。

平均GPU温度还相差7.006°C，原5°C环境诊断未通过；SM时钟差0.282%，无外来GPU进程。温差不是独立增加的停止阈值，但进一步限制因果解释。

原32B状态投影扫描累计runtime.pending，其中完成请求未删除。wrapper分段P95随2400s窗口增长：前300s约.288ms，最后300s约1.335ms。CPU-only源码/重放画像支持历史扫描扩展问题；纯router.decide很快但不能替换原完整wrapper验收。未修改冻结实现。

仅非合法完整cohort代数差=-0.013985735，Text P95诊断贡献=-0.015412520；二者明确不是正式ΔU。P1 Text P95由58.638增至62.398ms，原Text尾改善未在这一已测轨迹出现。缺少三组有效差值，不报告三组均值/置信区间。

后续若另行开展研究，先处理宿主在线状态历史扫描并独立冻结/验收，再设计新的配对与环境控制；不得放宽阈值或沿用当前非法差值宣称成功。本轮只提出此诊断，不实现修复、不重跑GPU。

完整失败诊断见long_window_failure_diagnosis.md；六个计划slot状态见planned_run_disposition.json，后两组新轨迹原样保留，未换种子。

最终全权重SHA、历史全部Module32/32B SHA和Module14F–31A/正式优化器metadata保护、组内输入和逐决策因果复算见 final_data_acceptance.json。该文件passed包括六次完成和技术验收，本轮为false；integrity_passed单独表示文件/权重/环境/因果重放完整性。所有既有Gate不变；GPU已释放，未启动下一阶段，到此停止。
