# 开发运行终态文件的服务与时间账本诊断工具

工具：[diagnose_text26_service.py](/root/autodl-tmp/generalist-specialist-project/tools/spillover_discovery_20261022/diagnose_text26_service.py)。它分析同输入／路径／副本／alias的实际服务相对原始逐输入基线的倍率、输入覆盖、固定池循环与路由对齐、分位数及时间相关性；另外独立核算时间账本和两种实际服务FIFO回放。它不拟合参数、不改评分／packet／冻结源码、不运行GPU。

## 运行门禁

现在可执行下列检查；**不读取queue progress、开发raw或partial文件**，只核对已声明packet、基线及源码元数据：

```bash
python tools/spillover_discovery_20261022/diagnose_text26_service.py --check-only
```

对已在显式队列收据中登记为 `completed`、`completed_with_failures` 或 `failed` 的任务，可分析其已原子发布的最终 `run.json`。`eligible=false`、非零退出码、运行错误或异常shutdown不阻止诊断；它们原样披露，不生成效用值或排名。原队列停止后，剩余独立条件登记在第二份收据中；两个路径必须显式传入：

```bash
python tools/spillover_discovery_20261022/diagnose_text26_service.py \
  --packet artifacts/static-spillover-study-20261022/text26-development-v1/experiment-packet.json \
  --queue-progress artifacts/static-spillover-study-20261022/text26-development-v1/runs/queue-progress.json \
                   artifacts/static-spillover-study-20261022/text26-development-v1/remaining-queue-progress.json \
  --output /absolute/path/to/new-text26-service-diagnostics
```

也可重复使用 `--queue-progress`。工具合并两份收据的 `jobs` 和 continuation 的 `preserved_attempts`，支持原队列 `output` 路径及 continuation 的 `raw={path,sha256}`。原队列在runner非零退出时可能只登记 `EXECUTION_ERROR` 和输出目录；工具仅将该目录规范为 `run.json`，将该状态保留为 `wrapper_status`。它本身不是可分析的raw终态：必须由另一份显式收据以 `raw={path,sha256}` 和实际终态共同绑定，才允许分析。两份收据的读取顺序不影响合并；缺失终态收据或SHA则保持waiting。同一job重复登记必须规范路径、已有SHA、实际终态、资格及退出码一致；显式attempt必须为1。有冲突就拒绝，不选择最新attempt，也不按U挑选。不存在于packet的job登记被拒绝。终态与eligible在读取后还须与raw一致，已登记SHA在JSON解析前核验。没有外部SHA的原队列文件计算并保留读取SHA，在分析结束复核。

默认要求packet的全部job均已登记终态且最终文件存在；有等待项则退出码3、`WAITING_REGISTERED_TERMINAL_JOBS`、`raw_files_read=0`，不读取其他已完成raw或创建输出目录。显式增加 **`--partial`** 可只分析已登记终态且最终文件存在的条件，报告保留完整waiting名单；若没有可分析条件，仍退出3。非终态、未登记、文件未发布的任务不会打开。此选项只表示条件子集；文件名仍必须严格为 `run.json`，不接受任何 `.partial` 文件。

读取时检查读前后文件身份、完整换行与JSON、terminal状态、packet声明的job/plan/seed/执行身份及执行身份摘要，只接受开发seed51–59。分析结束再次核对所有输入、源码与工具SHA。不扫描runs目录，不读取61/62或71–73，不更改评分、packet或冻结代码。`--check-only` 在progress读取之前返回，始终零raw、零progress读取。

失败、未完成、缺失必要计时或无法匹配服务基线的请求保留原record和SHA，标记 **UNKNOWN**，`service_ms/ledger/fifo_diagnostic=null`，不丢弃、不当成零服务。只有状态completed且必要计时、映射和服务字段有效的请求进入服务账本。报告逐项提供offered、recorded、analyzed completed、UNKNOWN及未记录请求数；缺失cohort、drain、异常shutdown和错误均披露。缺失记录只记数量，不凭空创造服务值；offered未知则缺失数量也为null。完成请求分布可能存在幸存偏差，不能代表失败请求或完整运行。

为避免跳过UNKNOWN后把未知服务误归入IPC/emit，凡alias含UNKNOWN、priming/warmup结束时间未知，或全局cohort/drain证据不完整、window标记aborted时，该alias的完整FIFO及一步间隙均标为 `UNKNOWN_CENSORED`，不计算残差；有效完成请求自身的ledger仍可诊断。只有完整可验证的alias轨迹才计算下述FIFO。所有诊断输出的 `diagnostic_U` 固定null、`ranking_performed=false`，不恢复或比较无效运行的U。

## 时间语义与独立账本

已检查冻结v5的runner、deployment、transport、runtime/protocol，以及packet真实worker使用的source-qa ProfileAdapter和PP-OCR适配器；输出同时记录冻结与真实worker源码哈希。

| 量 | 实际计时范围 | 不能误称什么 |
|---|---|---|
| service_ms | 原生ProfileAdapter在初始CUDA synchronize之后开始，覆盖infer、后同步及cache回收；PP-OCR适配器另含输入校验、图像加载和det/rec流程 | 不是整个handler；PP-OCR不套用generic cache回收语义 |
| worker_latency_ms | worker以wallclock记录整个handler起止，包括内层service外的同步／wrapper工作 | 不含请求在worker开始前的等待 |
| actual_submit_time_ns | host sender在stdin JSON write／flush**之前**记录的monotonic时间 | 不是worker收到请求或开始服务 |
| worker_queue_wait_ms | worker start_unix减host submit_unix | 包含IPC、stdin读取／解析、之前handler及emit间隙，不能解释为纯队列等待 |
| completion_time_ns | host reader解析完整stdout行后，事件入队时的monotonic时间 | 不是GPU完成时间，也不是worker handler结束 |

对每请求保留这条账本：

```text
observed E2E (host monotonic)
 = host schedule→enqueue
 + host enqueue→submit
 + worker start_unix − host submit_unix
 + worker end_unix − worker start_unix
 + host completion_unix − worker end_unix
 + residual_clock_span
```

host write完成时刻另存作诊断，不能再加进上式，否则会重复计入submit→start。stdout emit、read／parse与线程调度尚没有细分时间戳。单机wallclock与monotonic是不同域，工具只比较同域时间差，再报告跨度残差、负时间和字段残差，不直接相减绝对monotonic与unix时间，也不假定NTP调整不存在。

FIFO按每alias的host实际提交顺序，使用真实逐请求service序列和真实handler时长分别回放：

```text
oracle_handler_end = max(actual_host_submit_unix, previous_oracle_handler_end)
                     + observed_handler_duration
oracle_service_end = max(actual_host_submit_unix, previous_oracle_service_end)
                     + observed_service_duration
```

起始状态使用同alias最后priming／warmup handler结束；没有该记录则从首个host submit初始化。该回放有意省略未观测的stdin到达／解析、上一响应emit和调度开销，所以必须保留`observed_worker_end − oracle_end`残差。它只是“已知完整实际服务后”的oracle记账诊断，**不是把host submit假定为真实服务到达**，也不是可部署预测器。每条另算`actual_worker_start − max(host_submit, previous_actual_worker_end)`的一步未观测间隙，可区分累积service外开销与主机排队。不会用逐请求实际间隙把回放强行修正为零残差。

## 服务、覆盖与相关性输出

每条请求保留原始record、record SHA、原始run SHA、sample_id、n、alias、baseline输入身份和service值；明确只是同pool_id/n状态的历史基线比较，不据此声明历史协议与当前执行完全等价。倍率不乘旧全局service系数，也不拟合路径常数。

每路径报告service／handler／倍率的mean、min、P50、P95、max及sum-ratio；每alias另报告服务倍率和FIFO残差；每输入×alias报告重复次数和服务／倍率分布。按所有保留的有效身份记录（包括UNKNOWN）提供alias／sample序列、请求状态及模pool-size余数分配，并披露序列覆盖完整性，以揭示32输入池与固定路由的周期对齐。不能以global arrival_index间隔当任务路由周期。

时间相关性报告lag1/2/4/8/32的倍率相关，以及扣除同输入倍率中位数后的相关、四个时间块的分布、输入／生成token与service的描述性相关。合并路径可能交错多个并发alias，alias统计更适合序列解释。所有相关均为单运行描述，样本中位数与生成token只作已完成运行诊断，不作为新方案预测特征，不计算因果干扰结论，也不使用实际future overlap。

输出包括protocol.json、report.json、逐job完整请求／路径／输入／FIFO JSON及manifest.json；所有输入和语义源码有哈希，分析结束再次核对不变。新输出目录必须不存在，不覆盖既有证据。

## 已完成的CPU验证

[测试脚本](/root/autodl-tmp/generalist-specialist-project/tools/spillover_discovery_20261022/test_diagnose_text26_service.py)的17项CPU测试覆盖合成handler／service FIFO差异、wallclock跨度残差、失败ineligible终态分析、completed但资源超限仍保持eligible=false、UNKNOWN与零完成请求、缺失cohort、未知warmup、abort伪drain标记、身份和非终态拒绝、双收据合并与冲突拒绝、EXECUTION_ERROR目录与raw终态收据的双向合并及无收据零raw、注册SHA、默认等待零raw、partial只读已登记终态、check-only零progress/raw以及partial文件名拒绝。

另外使用两条**已完成ASR1.9 seed51开发运行**进行CPU smoke，结果位于[service-ledger-asr19-smoke-v2](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/cpu-discovery/service-ledger-asr19-smoke-v2/report.json)。未使用独立验证数据。两条运行的host/worker分解残差绝对值最大约.0148ms，原始alias顺序／handler重叠检查无异常；handler时长FIFO结束残差最大约5.035ms，说明未观测间隙确实需要单列，不能把所有差异都归到内层service倍率。

该smoke也实际显示ASR分流后G2/asr只收到4/32输入，工具能披露这种覆盖限制。此结果仅验证工具对这些完成文件可运行及保留账本残差，不证明text模型正确或分流收益。本次扩展对text26只执行check-only，未读取其progress或运行结果。扩展版另用相同两条已完成ASR开发运行做了[smoke-v3](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/cpu-discovery/service-ledger-asr19-smoke-v3/report.json)：2504条请求、UNKNOWN为0、diagnostic_U=null，保留原残差结果。v2/v3是各自工具源码SHA下的历史收据，后续源码若更新不覆盖它们。

## text26终态正式诊断的资源字段

[text26-service-diagnostic-v2](/root/autodl-tmp/generalist-specialist-project/artifacts/static-spillover-study-20261022/text26-service-diagnostic-v2/report.json)显式保留原eligible来源、resource_valid、safe_memory、预算、原M_norm及稳定性状态。completed从不推出eligible；n3实际resource_valid=false。中文初稿曾把n3资格误写true，机器v1与原raw一直是false；v1不覆写，v2仅补充原证据字段，服务账本不变。更正与后续n2校准建议见[终态分析](/root/autodl-tmp/generalist-specialist-project/docs/text26_terminal_service_diagnosis_20261022.md)。
