cpu02 已重新完成真实八请求 CPU 审计，结果仍为 `PASS_CPU_MAPPING_ONLY`。这是来源闭包和多任务处理补强后的重算，不构成 native 动态预测资格。此前 cpu01 的报告、static coverage、seal 和原文档全部保留；修改前的 audit source、测试和原文档精确快照位于 `artifacts/predictor-kernel-dag-training-mapping-v1-20261029-cpu01-source-snapshot/`，由 `snapshot.json` 绑定原 SHA。

新结果位于 `artifacts/predictor-kernel-dag-training-mapping-v1-20261029-cpu02/`：`report.json`、同一次执行产生的 `static-coverage.json`、`tests.json` 和 `seal.json`。实现不再手列七个文件，而以现有 `implementation_sources(__file__)` 扫描全部本地静态 import 闭包，当前为 61 个文件，包括 package initializers、v9 engine、checkpoint v9→v7 以及静态 occupancy resolver 的传递实现。`strict_references` 从 strict review、原 raw/plan/context、完整 saved DAG、canonical audit sources 和 occupancy audit 中收集精确嵌套 receipts，冲突拒绝、逐份读取前校验，执行完成后再次验证所有 491 份来源和完整实现闭包。这里只声明本地源码闭包和有精确 receipt 的数据文件，未将第三方 Python runtime/库或无 receipt 的外部对象描述为同等认证。

static coverage 已成为审计器正式可选步骤（`--occupancy-audit`），不再依赖一次单独的交互式 inventory；本次与主报告共用 491 个来源 receipts，并在 resolver 执行后重验。主报告绑定 companion 文件 SHA。覆盖仍为 129,545/129,545 kernels，包括全部 OCR，实际 UUID 唯一绑定仍为 `GPU-533a287b-0e15-a7c8-8e58-2b21bdcbb13a`。这只是 audited static allocation，不代表实际 SM 驻留、吞吐、DRAM/cache counters 或速度。

`replay_catalog` 现在对同一个 owner 的全部 profile 取 supported_tasks 并集，验证 family 一致；不同 family 被拒绝，不再由最后一个 task 覆盖前面的 task。每个 request 与 plan key 保持独立，原 FIFO 到达次序保留。本次实际 full8 的 generalist2 只有 text，因此这一修复没有改变原结果；新增反例明确检验未来同 owner 的 text+OCR 多任务域。

重算数据：8/8 requests、129,545/129,545 canonical kernels、0 unassigned、四 SQLite missing_runtime=0/unresolved=0/cyclic=false；fresh request DAG 与 training03 完整内容一致。OCR profile:3/7 保留 2/3 kernel streams 和 1,129/3,282 event-record-to-wait 边。v9 profiled replay 完成 161,984/161,984 DAG nodes、143,835 events；OCR 最大并行 activity nodes=3。source receipts 全部前后相同。

新增四项 CPU 反例后共 22 tests PASS，覆盖同 owner 多 task 保留、family 冲突拒绝、实际传递 checkpoint/package 代码纳入、嵌套来源冲突与文件篡改拒绝。真实 full8 路径也已完整重跑，没有以 fixture 代替真实数据审计。

原文档 `docs/kernel_dag_training_mapping_audit_v1_20261029.md` 中的 node-keyed rate 接口、same-owner peers、serial 单 active 限制、v8 rate≤1 限制、独立 six-launch 初始 pilot、全六列实际 rank gate、完整 launch 留出和 counters 缺口继续适用。仍是明确假设的 profiler replay：使用实际 kernel duration/API issue release/handler-end anchor，context/copy/default-stream 和 instant handoff 等机制尚未识别。没有真实 kernel fit、自然无 profiler 预测验证或动态 response 接入；没有 GPU/profiler/fit 启动，也没有修改 serial v1/v2、旧 native 闭包或运行中的 controller/driver。

复现（输出目录必须未使用）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/bin/python -m unittest tests.test_kernel_dag_training_mapping_audit_v1 tests.test_request_cuda_dag_v5 tests.test_kernel_sequence_audit_v2

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/bin/python -m tools.predictor_validation_20261027.kernel_dag_training_mapping_audit_v1 --review artifacts/predictor-gpu-20261027/kernel-training-bridge-v3-20261028-prepared00/light-kernel/strict-training-review-v3/report.json --profiles artifacts/predictor-gpu-20261027/request-cuda-dag-v2-20261028-training03/profiles.json --output NEW_UNUSED_ARTIFACT_DIRECTORY --replay-v9 --occupancy-audit artifacts/predictor-occupancy-declaration-light-training-v3-20261028/audit.json
```
