# 持续共同活动 primary/support 分析工具

`tools/spillover_capacity_20261022/analyze_sustained.py` 是只读CPU分析器，不启动GPU、不拟合预测器、不计算正式效用，不修改冻结的配置/评分/稳定性。它复用旧 `analyze.py` 的精确半开区间交叠、原生质量评分、冻结worker身份检查和终态文件SHA读取；分析产物同时记录两个源文件SHA。

## 读取边界

`--check-only` 只验证packet、全部binding、四个job、每alias的96条primary与1024条独立预声明support、固定门槛和冻结质量参考，不读取队列或raw。2026-10-22已通过新packet `text-sustained-joint-diagnostic-v1/experiment-packet.json` 的检查，packet SHA256为 `4097f86a366088bb62047f420f3aba93b5b8b0e24e02af6e6ebc5fc7cbe2b577`。

正式读取须显式提供 `--queue-progress`；仅接受队列登记终态 `completed/completed_with_failures/failed`、绝对路径、精确文件名 `run.json` 和已登记SHA。默认等待四个终态及完成队列发布；运行中队列不会导致分析器读取live文件。显式 `--partial` 可以分析已登记终态，但总状态仍不足，不选择attempt、不重试、不覆盖任何分析输出。已失败队列中后续未运行job仍保持等待/缺失。

```bash
python tools/spillover_capacity_20261022/analyze_sustained.py \
  --packet artifacts/static-spillover-study-20261022/text-sustained-joint-diagnostic-v1/experiment-packet.json \
  --check-only
```

```bash
python tools/spillover_capacity_20261022/analyze_sustained.py \
  --packet artifacts/static-spillover-study-20261022/text-sustained-joint-diagnostic-v1/experiment-packet.json \
  --queue-progress /absolute/path/to/queue-progress.json \
  --output /absolute/new/analysis-directory
```

## 完整保留与角色分离

每条primary从packet重建，缺失/失败/未释放均保留UNKNOWN，不能用support补齐。raw内support必须属于packet预声明模板且是每alias从0开始的连续前缀。所有已登记support全量保留请求身份、时刻、原生输出、评分、tokens及handler区间；未释放support只列数量，不伪造响应。

- 只有primary进入每输入3次服务中位数、原生primary质量和同输入同轮B/C配对；每alias独立输入簇仍为32。
- support单独列请求数、完整性、服务和质量描述。support不会增加独立样本数，也不进入任何正式Q、产量或效用分母。
- primary与support的所有可验证handler区间共同构造真实活动掩码。一个alias持续补发请求不代表其handler100%连续忙，IPC补发间隙按实际时间保留。
- 单活动B的all-active条件是平凡成立，明确标为不适用，不能当作联合活动证据。

## 预声明门槛

每个C alias均须完整包含96条primary；所有96条实测三方共同handler交叠必须为正，每个输入3轮至少2轮共同交叠比例≥0.90。UNKNOWN、只一次高覆盖、0交叠、缺失输入或缺失alias均使覆盖不足，不能跨输入平均。

跨block持续共同活动服务可重复性使用每alias/每输入**仅交叠比例≥0.90的primary服务**中位数，两块各至少2次，第二块/第一块须在闭区间 `[0.5, 2.0]`。完整96个alias×输入比较缺一也不足。全3次primary服务中位数另行保留，不能代替covered-primary中位数；报告同时保留两块全部逐输入、逐轮B/C比值和输出/token一致性。

输出漂移、逐轮服务漂移和全量质量保留描述；本次未另设输出一致率阈值，不把观察到的答案差异隐去，也不据此修改冻结质量。

测量窗、最后primary完成窗、support admission stop延迟与支持排空窗分开报告。检查primary完成声明、各alias完成数、primary+support release数、support不能早于本alias primary完成或晚于admission stop、一条在途、串行worker区间和时刻ledger。raw失败、原生评分缺失、资源无效、能量覆盖缺失、priming/warmup原生失败、非正常shutdown或时刻合同问题均保留并使证据不足。

资源与能量字段来自runner原始声明；此分析器不独立重算显存峰值或积分电能。独立审计属于另一工具和另一产物，不能把本分析状态冒充审计结果。

## 状态与解释

四job全部终态、各job完成/原生/资源/时刻检查通过、C覆盖通过且跨block门槛通过时，总状态为 `COMPLETE_SUSTAINED_DEVELOPMENT_DIAGNOSTICS`；否则为 `INSUFFICIENT_DIAGNOSTIC_EVIDENCE`。这个complete只表示预声明开发诊断证据充分，不代表开放负载稳定容量、正式质量迁移、容量函数通过、分流收益或全池最优。

`U=null`、`fitted_parameters=false`、`ranking_performed=false`、`independent_formal_validation=false` 始终保持。没有用观察到的handler交叠作为未来预测特征，也没有把交叠解释为GPU利用率或特定物理因果。

CPU验证：`python -m unittest discover -s tests -p 'test_spillover_capacity_sustained_analysis.py' -q`，14项通过。测试覆盖support维持交叠但不扩充primary、早结束缺覆盖、缺失primary、原生失败support、模板/重复/非前缀拒绝、admission边界、原生与资源失败门槛、covered median筛查、2/3且96全正交叠、inclusive阈值、B/C配对只使用primary，以及运行中不读raw和终态SHA边界。

## 终态分析绘图

`plot_sustained.py` 只读取已发布分析目录的 `manifest.json`、已登记SHA的 `report.json` 和job detail JSON；它不会沿report内的原始raw路径读取任何文件，不会读取GPU或队列。输出新的PNG/PDF各两张、逐区间/配对点数据和源SHA：

1. 四个job的三alias handler时序：primary与support不同颜色，保留真实请求间隙；虚线为最后primary的host完成边界。UNKNOWN单独标注，不补造区间。B中两个不活动专用显示空行。
2. G2逐primary同输入同轮C/B服务比值与实际共同handler交叠比例散点，服务比值用对数轴，标出0.90覆盖线；block与重复轮次均保留标记。此图只描述关联，不证明因果或容量。

```bash
python tools/spillover_capacity_20261022/plot_sustained.py \
  --analysis /absolute/published/analysis-directory \
  --output /absolute/new/figures-directory
```

绘图CPU测试3项通过，覆盖只依赖分析产物、原raw路径不存在也可完成、篡改SHA或路径越界被拒绝，以及不覆盖既有图件。实际终态产物须在全部原始终态发布后另行分析/绘图，本开发过程未读取运行中的raw。

## 本次600秒失败的截止修正

既有 `ea65736…` 分析工具按原readiness源码成功产生 `analysis-v1` 的partial报告，状态INSUFFICIENT；它保留shutdown期间才完成的原生response并给出 `Completion after measurement end` 等合同flag。因此其77条G2已完成服务是**终态总数**，不是截止窗内有效数，不能直接用于窗口内服务或覆盖。

不修改原工具、readiness或analysis-v1，新增 `failure_window.py` 只读已验证SHA的终态analysis，产出 `failure-window-v1`。截止为 `min(measurement_end, measurement_start+600秒)`，以host完成时间为准：C有76/96条G2 primary在截止前完成，1条在截止后完成，19条未释放；两专用primary各96完成；support在截止前1203条，截止后2条。截止后原生输出和服务保留在terminal字段，窗口资格为UNKNOWN，不计入有效服务、质量、B/C比值或覆盖。观察到的跨截止handler只截取截止前区间作为活动上下文，不伪造完成。

`plot_failure_window.py` 只读新切片的manifest/report/detail，生成 `failure-window-figures-v1`：handler时序到600秒为止，G2配对散点只有76个有效点。所有图件和INTERPRETATION明确PARTIAL/INSUFFICIENT，不能读成96完成或3轮全部成功。两个图已视觉检查。

独立失败审计指出显存末约0.316秒没有样本括住，全窗口资源覆盖UNKNOWN；观测峰值合法不补足缺口。完整primary与跨block门槛失败，未启动第二block，不自动重试。5项新增CPU fault测试通过，验证截止后原生保留但不资格化、跨截止handler裁剪、未释放与截止后完成分开、600秒硬截止和配对只使用有效服务。
