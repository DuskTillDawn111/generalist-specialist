# 原四候选主预测验证执行准备

本文件准备原 v5 gate 的自然 GPU 验证入口，本次没有启动 GPU，也没有读取新验证标签。
代码为 `tools/predictor_validation_20261027/main_prediction_gate_supervisor_v1.py`。
它只在四个完整 CPU 预测、独立 v48 artifact review 和真实 prediction seal 都完成之后允许执行。

## 保持的实验口径

原计划为 `artifacts/predictor-gpu-20261027/known-input-availability-v11-20261028-prepared01/plan.json`，
run ID 为 `fresh-availability-v11-r01`。四候选按 `shared_balanced`、`shared_skewed`、
`shared_spill_low`、`shared_spill_high` 顺序独立运行，每次原 57 个测量请求（56 text、1 OCR）、
12 次原生 readiness priming，30 秒 offer + 120 秒 drain，warmup=0。所有候选保留原四 owner 和
uncapped shared 执行；没有新增 cap1/cap2、丢弃、改路由、改请求 ID、改输入或窗口。

六项 Q/T/L/P/M/E 均为 1/6，原 startup amortization=3600 秒、rho/g(n)/稳定性规则不改。
gate 仍要求 service/e2e/P95 MARE≤0.2，queue-depth MAE≤1，ΔU 符号及候选 pair 排序准确率=1。
完整请求的到达至父 reader receipt、真实 handler START 排队边界及原遥测口径由原 v11 collector、
v5 observed adapter 和原 flow_metrics 实现，supervisor 不另写评分公式。

## 启动前拒绝条件及预算

v48 preparation 须通过四份 lossless gzip 预测的独立 reprojection、完整训练索引、来源闭包核对。
v48 protocol 必须冻结 supervisor 的全部 `implementation_sources` 闭包（当前165个来源），
不能只封存顶层文件。`--additional-execution-sources` 用显式入口清单扩展该闭包。
supervisor 再核对 v5 `_bindings`、原计划、公有57请求、四份57行 numeric timings、六项 numeric terms、
完整 numeric queue grid、精确 production/preparation/independent-review receipts。
任何 UNKNOWN、来源漂移、缺封存、非 PASS review 都禁止启动。

每个候选启动前 free disk≥18 GiB；运行中 free disk<13 GiB、相对启动时磁盘减少>6 GiB、
或整组 elapsed>2400 秒都会停止**该组自建 session/process group**，保留失败 raw 和日志。
监测间隔0.2秒，SIGINT→SIGTERM 各最多8秒清理，再SIGKILL；清理可能使实际退出略超2400秒。
四个候选**各自2400秒**，不是四组总共2400秒；计时包括等GPU锁、加载、预热、30秒offer、
120秒drain和关闭。磁盘减少是所在文件系统总体空闲变化，其他进程写入也可能触发保守停止。

原 v11 child 自己持有 `artifacts/.gpu-experiment.lock`；外层 supervisor 不持同一锁，避免锁嵌套死锁。
child 在锁内重查绑定、写真实 collection start、验证四 owner 的 READY PID 归属、保留所有请求。
本方案不运行 nsys、不产生新的 profiler sqlite 或 trace 派生物。原 collector 自身只保证原启动/关闭/
drain 边界，新增 supervisor 才提供整个进程的2400秒和磁盘守护。

## seal 完成后的命令

以下 `MAIN_SEAL` 必须指向 v48 成功生产目录，不能指向 CPU preflight 目录。
`MAIN_GATE` 选取全新、其父目录已存在的独占输出根；各候选不能重试覆盖。
这些命令仅为执行准备，本次未运行。

```bash
MAIN_PLAN=/root/autodl-tmp/generalist-specialist-project/artifacts/predictor-gpu-20261027/known-input-availability-v11-20261028-prepared01/plan.json
MAIN_SEAL=/absolute/path/to/successful-v48-seal
MAIN_GATE=/absolute/path/to/new-main-gate-output
mkdir "$MAIN_GATE"
python -m tools.predictor_validation_20261027.main_prediction_gate_supervisor_v1 prepare \
  --plan "$MAIN_PLAN" --protocol "$MAIN_SEAL/protocol.json" \
  --predictions "$MAIN_SEAL/predictions.json" --production "$MAIN_SEAL/production-receipt.json" \
  --output "$MAIN_GATE/execution-manifest.json"
```

四次分别执行并核对 `execution.json` 状态，失败即保留证据并暂停后续，不通过改阈值或再训练覆盖该次验证。

```bash
python -m tools.predictor_validation_20261027.main_prediction_gate_supervisor_v1 execute --manifest "$MAIN_GATE/execution-manifest.json" --candidate shared_balanced --output "$MAIN_GATE/shared_balanced"
python -m tools.predictor_validation_20261027.main_prediction_gate_supervisor_v1 execute --manifest "$MAIN_GATE/execution-manifest.json" --candidate shared_skewed --output "$MAIN_GATE/shared_skewed"
python -m tools.predictor_validation_20261027.main_prediction_gate_supervisor_v1 execute --manifest "$MAIN_GATE/execution-manifest.json" --candidate shared_spill_low --output "$MAIN_GATE/shared_spill_low"
python -m tools.predictor_validation_20261027.main_prediction_gate_supervisor_v1 execute --manifest "$MAIN_GATE/execution-manifest.json" --candidate shared_spill_high --output "$MAIN_GATE/shared_spill_high"
```

四组成功后，`collections.json` 为 JSON 数组，每项格式：

```json
{"candidate_id":"shared_balanced","raw_path":"/absolute/MAIN_GATE/shared_balanced/collection/scored-raw.json","precollection_path":"/absolute/MAIN_GATE/shared_balanced/collection.precollection.json"}
```

其他三项使用各自 candidate 名。adapter 读取 scored copy 的 native_raw receipt，原始 `raw.json` 不改。

```bash
python -m tools.predictor_validation_20261027.known_input_observed_v5 adapt \
  --plan "$MAIN_PLAN" --protocol "$MAIN_SEAL/protocol.json" --predictions "$MAIN_SEAL/predictions.json" \
  --run-id fresh-availability-v11-r01 --collections "$MAIN_GATE/collections.json" --output "$MAIN_GATE/observed"
python -m tools.predictor_validation_20261027.known_input_gate_v5 \
  --protocol "$MAIN_SEAL/protocol.json" --predictions "$MAIN_SEAL/predictions.json" \
  --observed "$MAIN_GATE/observed/observed-run.json" --output "$MAIN_GATE/gate-report.json"
```

主验证不得使用 `--allow-cpu-fixture`。最终依据原 gate PASS/FAIL/UNKNOWN，不依据 subprocess exit code
或 supervisor 的 COLLECTED 状态宣称准确。即使 gate PASS，也只支持当前已训练 public content 的独立新共跑
请求预测，不支持 unseen-content、kernel级真值、最终净收益或原/优化×禁/允分流四组收益结论。

## CPU 验证

`python -m unittest tests.test_main_prediction_gate_supervisor_v1 -v`：5项通过。
覆盖完整四预测数值性、未知/漏请求/漏队列/NaN拒绝、精确封存及独立review、闭包缺失、
低磁盘启动前拒绝、真实CPU独占进程组 timeout/disk-floor/incremental 清理、子进程失败证据保留、
来源漂移拒绝和禁止同输出重试；测试不启动 CUDA。
