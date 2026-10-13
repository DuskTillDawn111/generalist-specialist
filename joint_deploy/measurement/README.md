# Stage 2A：固定 deployment / routing 的真实测量

入口接受 `(deployment counts, workload req/s, task×model routing req/s)`，启动已有 worker，
生成时间轴上的请求，保存实际结果。只执行显式配置，不调用 Stage 1 求解器或选择组合/路由。

## 运行与配置

```bash
cd /root/autodl-tmp/generalist-specialist
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m joint_deploy.measurement.run_measurement \
  --config joint_deploy/measurement/configs/smoke_text.json \
  --output results/stage2a/smoke_text.json
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m joint_deploy.measurement.run_measurement \
  --config joint_deploy/measurement/configs/smoke_multitask.json \
  --output results/stage2a/smoke_multitask.json
```

两个 smoke 依次运行。默认复用正式 1200 条六任务 pool；可用 `--pool` 指定其他已有 JSONL pool。
`--profile` 默认六任务 metadata，只用于 capability 校验和 provenance，不读取 service time 做预测。
示例命令显式设进程级 OMP_NUM_THREADS=1；所有相关线程环境值记录在结果中。

`MeasurementConfig` 的 JSON 字段：

| 字段 | 意义 / 默认 |
| --- | --- |
| deployment | model type → 非负整数实例数；不区分 generalist/specialist 的变量形式 |
| workload_rps | task → req/s |
| routing_rps | task → model → req/s，逐任务总和必须等于 workload |
| warmup_seconds | 10 秒；示例 5 秒 |
| measurement_seconds | 30 秒；示例 20 秒 |
| arrival_process | 当前仅支持 deterministic_periodic |
| random_seed | 42 |
| routing_source | explicit_config；若手动导入 Stage 1 witness，应写 stage1_feasibility_witness |
| startup_timeout_seconds | 每个 worker 等待 READY 上限，300 秒 |
| drain_timeout_seconds | 每个阶段等待在途请求上限，60 秒 |
| shutdown_timeout_seconds | 所有子进程共享 graceful shutdown 等待预算，10 秒 |
| max_pending_per_model | 每个 host sender queue 最多 10000 条；满时明确记录 request error |

schema 接受多个 generalist type 及 count > 1；当前 runtime 单 GPU、每种 model type 最多一实例，
在启动前明确拒绝 count > 1。不新增 replica 调度。新增模型需 capability metadata 和 worker launcher。
默认 launcher 复用已有 WORKERS（含 phi4），加上现有 Qwen launcher；当前 smoke 验证 Qwen、
DistilBERT、Whisper，ResNet/PaddleOCR/其他 launcher 尚未在本阶段真机 smoke 中覆盖。

所有 rate 必须非负有限，正 routing 只能指向已部署且支持 task 的模型。routing 守恒按
1e-7 相对容差校验（零 workload 无绝对余量）。不重做 Stage 1 Capacity 筛选，允许测量过载。

## 复用代码与执行过程

- 复用 `run_static_hybrid.Worker` 的进程启动、PaddleOCR library path、stdout parser、send/shutdown。
- 复用 `run_pure_qwen` 中 Qwen Python/script 配置及 percentile 计算，复用 `read_jsonl`。
- inference 完全由现有 `qwen_worker.py`、`distilbert_worker.py`、`whisper_worker.py` 等处理。
- 协议保持原有 `READY|JSON`、`RESULT|JSON`、`ERROR|JSON`、`BYE|JSON`，请求保留
  request_id / pool_id / arrival_index / task_type / sample / submit_ns。

runtime 给现有 reader 增加接收时间戳和有序 EOF 事件，不改 worker。每个模型一个 sender
线程与 FIFO queue，pipe 阻塞不会阻塞其他模型或 arrival loop。没有等待上一个请求完成才
生成下一个请求。所有错误、队列满、超时均保留原始请求记录。

worker READY 后，warmup_seconds > 0 时先让每条正 routing 路径执行至少一个 priming 请求，
再运行配置时长的周期 warmup，并 drain。所有预热请求必须完成才开始 measurement。
因此实际预热总时间包含 priming、periodic warmup、drain，并非恰好等于配置秒数。
priming/warmup 各自保存记录，与正式统计完全分开。

每个任务流：`phase_k + n / lambda_k`，phase 由 seed 和 task 名独立产生，落在首个 period 内，
避免全部在 t=0 同步 burst。正式阶段重置 routing 和样本序列，预热时长不会改变正式选择顺序。
样本按 task 分组，用固定 seed 洗牌后循环；记录 sample_id、pool_id 及 pool 哈希。

路由使用各 task 独立的 smooth weighted round robin：累加配置 rate，选择当前 deficit 最大
的模型，再减去总 rate。无随机路由。有限时长请求数受相位/取整影响，实际数量完整保存。
该规则只执行给定 routing matrix，不选择或调整 routing。

## 时间与指标口径

host 的 scheduled、submit、completion 均用 `perf_counter_ns`，不能和 worker wall-clock
时间戳直接相减。completion 在 stdout reader 放入事件队列时记录，不含 collector 消费队列延迟。
submit 是开始向 stdin 写入前的时刻，另保存 write-completed 时间以识别 pipe backpressure。

- schedule_delay = submit − scheduled：含 arrival loop 延迟、host sender queue 等待。
- e2e_latency = completion − scheduled：包括排队、执行与返回；只对成功完成的正式请求计算。
- runtime_latency = completion − submit：包含 pipe/worker queue 等待及执行/返回，不是纯 GPU 耗时。
- worker_latency_ms：原 worker handler 自测墙钟时间，保持原名；缺失为 null。
- worker_queue_wait_ms：原协议的 start_ns − submit_ns，使用 wall clock，包含传输/worker 等待。
  clock 调整可能影响该辅助量，但不会影响 host 单调时钟的 e2e/runtime 指标。

每个 task 输出 offered/completed/failed/incomplete、target/actual rate、e2e mean/P50/P95/P99、
runtime/worker/schedule_delay mean/P95、routing counts/fractions/rates，以及非 null 观测数。
`offered = completed + failed + incomplete`；ERROR 不当作成功 completion。失败和 incomplete
不进入成功 latency 分布，读 latency 时必须同时查看 completion ratio，以识别完成样本偏差。

正式 cohort 为 scheduled 落在 `[measurement_start, measurement_end)` 的请求。
窗口后等待配置的 drain 上限；超时后标为 incomplete，shutdown 中收到的迟到响应另存
`late_response_after_drain`，不会再改变正式计数。

- `achieved_total_throughput_rps` / task `achieved_completion_throughput_rps`：正式 cohort 截至
  drain deadline 成功完成数 / **配置 measurement_seconds**，按需求定义。
- `within_window_completion_throughput_rps`：实际在窗口内完成数 / measurement_seconds。
- `completed_during_drain`：窗口后才完成的 cohort 请求数。
- `completion_throughput_including_drain_time_rps`：成功数 / 从窗口开始到 drain 结束的实际时间。

cohort completion throughput 不能单独作为持续服务能力的证据。结果同时保存窗口、drain 时间及
actual offered load，不用一个 overall latency 替代各 task 指标。

模型 received_requests 以成功 pipe write 或 worker response 为证据，并非独立 worker ACK。
模型级保存 routed/received/completed counts、按 task 的 received counts、已观测 handler 时间总和。
handler 时间包含 host 工作，不称为 GPU busy utilization。本版未添加 NVML 采样。

## 结果与测试

一次执行输出完整 JSON，包含配置、窗口、READY、shutdown、模型/任务/system 指标，及
request_records / warmup_request_records / priming_request_records。source 保存 Git commit
（非 Git checkout 时 null 并注明）、profile/pool/protocol/worker/harness 哈希、线程环境与 GPU snapshot。
输出路径已存在时会覆盖，保留实验历史请为每次运行选择新路径。

完成且无请求错误/不完整、进程退出正常/BYE 收齐时 exit=0；启动、执行或清理失败 exit=1，
保留失败 JSON；无效配置在启动前拒绝。shutdown 有界，必要时只 terminate/kill 本次启动的子进程。

```bash
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m unittest discover \
  -s joint_deploy/measurement/tests -v
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m unittest discover \
  -s deployment_core/tests -v
/root/miniconda3/envs/gsdeploy/bin/python benchmark/scripts/validate_benchmarks.py
```

CPU integration fixture 使用原 `worker_protocol.run_worker`，仅测试 orchestration，不算 GPU smoke。
不训练 predictor、不推断性能、不做 deployment/routing optimization；Stage 1 保持封存。
