新 `unified_kernel_dag_intake_v1.py` 使用独立 unified-v8_1 原schema入口，复用已有 strict load_review/trace_metadata、validate_request、parse_sqlite、audit_sqlite、request_plan、validate_mapping、AuditedOccupancyResolver 和冻结 node-path compiler。没有修改原producer、原isolated intake或GPU controller，没有启动export/profiler/CUDA/模型/fit。

真实来源是 `unified-corun-v8_1-20261028-prepared01/unified-light-v8_1/strict-review-v8_1/report.json` 的PASS单SQLite、原raw/context与31个measured phases。旧kernel_sequence.load_collection/request_cuda_dag.convert硬限isolated原schema，因此不能直接复用其入口；新模块只适配原unified字段，先角色门禁再读labels，由原request事件重建anchors、只读重parse既有SQLite，与saved phases/clock_mappings完全相等。GPU lifetime始终取原nsys_start_ns/nsys_end_ns，同一SQLite即同一native ns坐标；不做跨owner median校时。request_plan内原host release映射仅保留描述性handler坐标，未作为跨owner GPU lifetime。

全量结果在 `artifacts/predictor-unified-kernel-dag-intake-v1-20261029-cpu01/`：4个offered请求，text0/1各19423、G2 11552、OCR5493，共55891kernels精确canonical DAG bijection，0missing/unresolved/cycle。OCR kernels保留streams7/13，包含1129eventwait与全部copy/memset/host-sync节点。wholeSQLite为371244GPU、23368copy、4360memset；其中315353个非measured kernels也纳入peer完整性扫描，与measured union重叠0ns，未被当作可忽略的潜在并行peer。

旧occupancy audits不覆盖本SQLite，所以没有按名字/配置从isolated借用。使用现有host-only工具新建 `artifacts/predictor-occupancy-declaration-unified-training-v1-20261029-cpu01/audit.json`，在原设备完整UUID声明上读取exact-source完成launch，1679个严格静态等价组覆盖全部371244，known371244/UNKNOWN0；原branch sensitivity全通过。该命令只用g++重新构建CPU header calculator，不链接CUDA或执行device代码；补充56份完整host/CPP/header依赖receipts于extended-source-receipts.json。缺attrs、别源或不匹配hardware只能UNKNOWN或拒绝，未放宽。

exact静态join后，冻结compiler完整编译55891/55891 lifetime paths，0rejected，全部六列保留、same-owner peers包含。路径内结构秩为6，但55891个exact-node baseline全是singleton，nuisance-adjusted derivative rank=0。每个自由node baseline可以对任意beta吸收唯一响应的积分路径，breakpoint不是独立response replication。因此没有调用fit，numerical/SM service/物理/预测/部署资格仍false。原compiler仍native_sources_authenticated=False；本intake的native_mapping_authenticated=True只说明source-bound CUDA/NVTX/DAG映射，不替代响应法则资格。没有与isolated context池化。

作者9项CPU测试通过，覆盖原schema角色早于labels、native端点/同owner多流分母、duplicate/missing/wrong-PID、未测peer半开区间重叠、exact静态source/hardware/launch匹配、UNKNOWN保留及路径rank6仍singleton rank0。543份real intake sources在运行结束核验无变化；作者测试扩为544份sources+test并再次前后核验。fullcatalog、capture与paths独立保存供root独审。

复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/bin/python -m unittest tests.test_unified_kernel_dag_intake_v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/bin/python -m tools.predictor_validation_20261027.unified_kernel_dag_intake_v1 --review artifacts/predictor-gpu-20261027/unified-corun-v8_1-20261028-prepared01/unified-light-v8_1/strict-review-v8_1/report.json --device-attributes artifacts/predictor-device-attributes-v2-20261027-extended/device-attributes.json --occupancy-audit artifacts/predictor-occupancy-declaration-unified-training-v1-20261029-cpu01/audit.json --output NEW_UNSTARTED_OUTPUT
```

后续需要同执行身份下的独立response replication及跨capture exact-input/work、静态与DAG节点共同模板对齐；raw/request IDs与native provenance必须各自保留，不能直接把不同OCR work按名字合并。当前没有新的GPU补采或拟合计划被执行。
