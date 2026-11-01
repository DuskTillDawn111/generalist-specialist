# 开发数据一页汇总

最终审计文件生成后运行：

```bash
python tools/spillover_discovery_20261022/build_development_data_report.py \
  --output artifacts/static-spillover-study-20261022/development-data-report-v1
```

默认仅读取三个明确来源的audit.json：five-development-point-audit-v1、asr19-replication-development-audit-v3、text26-development-audit-v1。可用 `--five-audit`、`--asr-audit`、`--text-audit` 显式换已审计JSON位置；按各自schema分派，不能把v1/v2或其他研究资料自动猜成v3。拒绝validation/confirmation目录和非audit.json输入，绝不追读引用的原始run.json或未完成raw。

输出目录必须新建，已有目录不可覆盖。缺少审计文件或源审计尚未完成时输出PENDING；其余已完成资料仍展示，缺失项不补造。源审计不完整时保留已有成本但U和所有效用比较为空。失败、不稳定、被保留的暂停旧attempt同样不参与比较。只有源审计确认的同trace、双方有效配对才显示ΔU，且与行内U重新核对。

产物：README.zh.md中文单页、conditions.csv（负载/seed/部署/精确路由/完整性/稳定性/六项/U）、paired-comparisons.csv、sources.csv、report.json与manifest.json。CSV/JSON保留完整数字，中文页成本按6位小数显示；路由JSON保留原浮点数，不四舍五入替换。原seed51对在两个开发资料中重复出现时，按raw SHA标记同一证据，不作为新增重复样本。ASR v3保留被用户暂停的旧attempt，U为空。

请求完整性与效用资格独立展示：README及CSV分列offered、completed、未完成数、原status_counts、运行status、源utility eligible、utility_complete、效用原因、最终效用有效性与稳定性。计数仅来自审计内cohort、measurement_window、status_counts、request_summaries.task、per_task明确的窗口及drain完成数、utility明确计数；不从eligible或utility_complete推断完成数。缺少明确计数写UNKNOWN；若计数矛盾则报错。ASR seed52 split为完整1227/1227但效用无效且unstable，nosplit为未完成229、failed且unstable；暂停旧attempt独立保留WINDOW_ABORTED。

若审计memory_evidence.resource_valid=False，单列memory_resource_valid/safe_memory_mib及MEMORY_RESOURCE_INVALID原因；完整、recoverable_congested但内存不合格的运行仍U为空，不误称unstable。text26正式源为修复attempt日志发现问题后的text26-development-audit-v2；使用`--text-audit`显式指定，v1保留。原审计确认同trace与双方可比较分别记录：同trace不代表无效效用能参与比较。

汇总不重算或修改生产评分、权重、稳定性、门禁、统计假设；所有结论仅限开发证据，不能称可靠普遍收益、独立最终确认或全池最优。

CPU检查：`python -m unittest tests.test_development_data_report -v`。覆盖三schema适配、缺测PENDING、失败/暂停不比较、源审计partial保守处理、同证据去重标记、错误schema与heldout拒绝、仅audit.json数据读取及输出不可覆盖。
