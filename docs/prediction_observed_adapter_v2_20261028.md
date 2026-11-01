# 原生实测接入 v2 gate 与采集前封存检查

新增 `tools/predictor_validation_20261027/prediction_observed_v2.py`，只执行 CPU 证据检查、实测算术与格式转换。没有发射 kernel、运行模型、拟合预测器或改动当前冻结采集源。旧 V4 数据及失败阈值不变。

## 当前六项口径

沿用[上一轮最终结果所明确的当前实验口径](predictor_validation_final_results_20261027.md)：

- `T=1-N_win/N_off`，分母保留全部计划请求，正式截止后的排空完成不补回窗口吞吐。
- `L/P` 按计划到达到 **parent reader receipt** 的每任务完整 cohort 延迟求均值/线性经验 P95，使用冻结负载权重和参考值。handler END 不替代 parent receipt。
- 实测 `Q` 对每任务累计 numerator/denominator 后按冻结请求率加权。OCR/ASR 不平均每请求 CER/WER，也不截断大于 1 的值；质量来源或完整实测分子分母缺失则 UNKNOWN。
- `M=f*max(startup-through-drain owned sampled MiB)/B`。包含空闲 resident、启动、priming、warmup 和排空；安全系数只应用一次。不是旧历史段落的两次驻留快照，也不是瞬时峰值。
- `E=(J_run/H+J_startup/A)/P_ref`，正式窗口和启动期均须有板级功率覆盖。启动包括加载/priming/warmup，drain 能耗不进入 `J_run`。不以 handler/NVTX/kernel 时间制造板能耗。

适配器复用 `flow_metrics.score_flow`，先核验身份、PID 显存求和及时间戳，再处理该 helper 的“仅完成请求诊断”边界：缺少完整 parent receipt 时，gate 的 `T/L/P` 保留 UNKNOWN，不让剩余请求的漂亮均值代表完整 cohort。与 `FullHorizonModel` 的当前公式一致；预测质量仍使用训练期预期值，实测质量必须来自当次输出。原 g(n)、rho 和稳定性不修改，报告保留已有断言但不代替原资格评估器，更不自动开启最终四组实验。

## 在实际采集之前

`verify_before_collection(protocol_path, predictions_path, plan_path, *, run_id, candidate_id, output_root)` 验证：

1. v2 protocol、model/source/training 文件、预测 SHA 和封存顺序。
2. flow plan 本身已列入 `protocol.sources`，候选、baseline、execution context、完整公开输入 cohort、六项权重及队列网格一致。
3. plan 的依赖 receipts 及 context 内冻结执行源与当前执行文件一致。
4. 目标采集目录不存在或为空，候选已有封存预测；拒绝重用已有 raw/trace 的目录。

函数独占写入 **目标目录的同级** `<output-name>.precollection.json`，返回 `receipt_path`。采集目标目录保持不存在，使使用 `mkdir(exist_ok=False)` 的新原生 collector 可以正常创建。调用方在真正开始采集时必须记录 `time.time_ns()`，并把以下字段写入 immutable raw：

```python
raw['split'] = 'validation'
raw['precollection'] = receipt(check['receipt_path'])
raw['collection_started_unix_ns'] = actual_collection_start_unix_ns
```

这里是本机源回执和显式启动时间证据，不能证明不诚实元数据的真实性，也不是外部公证时间。此函数本身不证明调用方给 `seal_predictions` 的数字确实由模型计算；实际模型执行应由独立 sealer 完成并绑定。

CLI：

真实 `native_flow.execute` 现已内接前置检查。应直接使用下列命令启动新验证，
不要先单独执行 `precheck` 占用其独占回执路径。protocol 必须冻结实际
`native_flow.py`、plan 及预测依赖；plan 需事先声明上述队列边界。

```bash
python -m tools.predictor_validation_20261027.native_flow execute \
  --plan /absolute/flow-plan.json --candidate baseline \
  --output /absolute/new-native-run --protocol /absolute/protocol.json \
  --predictions /absolute/predictions.json --run-id validation-run
```

三个验证参数必须同时提供；训练计划不能通过此入口重新贴为验证。
获取 GPU 独占锁后再核对封存文件，避免排队等待期间源码或模型变化；
随后在首个板级采样及 worker 启动前记录真正的 `collection_started_unix_ns`。
原 `raw.json` 保留 split/precollection/开始时间，`scored-raw.json` 只增加质量
证据并绑定原 raw。以下单独 `precheck` 命令仅供其他正确集成这些字段的 collector。

```bash
python -m tools.predictor_validation_20261027.prediction_observed_v2 precheck \
  --protocol /absolute/protocol.json --predictions /absolute/predictions.json \
  --plan /absolute/flow-plan.json --run-id validation-run \
  --candidate-id baseline --output /absolute/new-native-run
```

## 采集后适配

`build_observed_run(..., run_id=..., collections=[...], output_dir=..., allow_cpu_fixture=False)` 的每项 collection 包含 `candidate_id`、`raw_path`、`precollection_path`。完整候选使用同一个声明 run ID；各候选物理执行可以有自己的真实启动时间，gate run 元数据使用其中最早启动时间，逐候选时间在 source-bound raw 保留。

接受 `gsdeploy.native_flow_collection.v1` 加上述显式验证字段；拒绝 smoke、training、phase collection 及 profiler-perturbed flow。原 raw 必须绑定相同 plan、实际 objective/resource context。若传质量注释文件，`native_raw` receipt 必须指向原生 immutable raw，且注释不得改写其中任何字段。

时序核验使用每请求恰一个实际 START、READY PID、handler END、reader receipt 及完整计划到达。`service_ms=handler_END-START`、`queue_ms=START-arrival`、`e2e_ms=reader_receipt-arrival`；不把 collector 处理或 permit release 混入终点。没有 START 时不从 receipt 反推服务和等待。

队列定义必须事先写入 plan：

```json
{"queue_depth_boundary":"waiting_until_handler_start"}
```

冻结 grid 上的深度为 `arrival <= grid < actual START`，包含准入、准备及配批等待，排除 active 请求；不把所有未收到响应的请求称作队列。缺少有关 START 的点保留 null，不按零计数。

显存逐样本核对 NVML PID→MiB、owned alias、READY PID、foreign PID 和实际 owned 求和。加载期间允许部分 owner 尚未出现，但 READY 后必须所有 resident 都在；遥测必须包络启动至排空。板能耗独立检查，不因缺显存把有效板功率伪造成零。缺数保留 null，矛盾身份/非单调时钟/哈希变化直接拒绝。

输出目录独占创建，保存 `adapter-report.json`，以及满足真实采集先后证据时的 `observed-run.json`。后者直接供 `gsdeploy.forecasting.prediction_gate` 使用；独立审计报告绑定原始 raw、质量来源、precollection、adapter/scorer 源，并单列缺失原因。缺采集前证据时不输出 gate 文件；缺指标时保留完整候选/请求及 null，让 gate 返回 UNKNOWN。运行失败不会只留下可通过 gate 的六项数字。

```bash
python -m tools.predictor_validation_20261027.prediction_observed_v2 adapt \
  --protocol /absolute/protocol.json --predictions /absolute/predictions.json \
  --plan /absolute/flow-plan.json --run-id validation-run \
  --collections /absolute/collections.json --output /absolute/new-adapter-report
```

14 项 CPU 测试覆盖手算完整六项、OCR ratio-of-sums、排空显存峰与安全系数一次、能耗区间、等待队列、缺失端点、缺失遥测、NVML 外来进程/错误求和、immutable raw、CPU 标识、source drift、倒序采集和失败采集。命令：

```bash
python -m unittest tests.test_prediction_observed_v2 -q
```

测试 fixture 的 PASS 仅证明适配与 gate 算术；新 GPU 预测准确度、真实互补及最终收益仍未由该工具证明。
