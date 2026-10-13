# Stage 2B-1：Routing Characterization

复用封存 Stage 2A，固定五模型 deployment 和 workload，仅改变显式 task×model routing_rps。
不修改 feasibility、worker 或 measurement harness；不建立 predictor，不选择 routing/deployment。

## Protocol

- 固定 deployment：Qwen、DistilBERT、ResNet50、PaddleOCR、Whisper，各一实例。
- 固定 workload (text,image,ocr,asr,visual_qa,lid) = (2,1,1,0.3,0.5,1) req/s，总计 5.8 req/s。
- 生成标签 specialist_share ∈ {0,.25,.5,.75,1}；最终配置始终存完整 6×5 routing_rps。
  VQA 固定全部交给 Qwen；其他任务在 Qwen 与当前唯一对应 specialist 之间分配。
  该生成器仅针对本实验的当前模型池，不改变通用 deployment-count schema。
- 每种 routing 3 次：seed=101/202/303。每轮进程完全重启、priming、warmup=10 s、measurement=100 s。
  每轮实际 scheduled 为 text 200、image 100、OCR 100、ASR 30、VQA 50、LID 100，共 580。
- 时间长度按 ceil(30/min(active lambda)) 自动得出；正式实验不改变 workload。
- 相同 repetition 中不同 routing 用相同 seed，使到达相位和 task 样本序列可对齐。
  不同 repetition seed 不同。正式 pool 固定 mixed_formal_1200_6task.jsonl。
- 每个 repetition block 中 routing 顺序按固定 seed 打乱，实际顺序保存 plan.json。
  一个 run 一个 GPU measurement，串行运行。OMP_NUM_THREADS=1，其余环境由 Stage 2A 记录。
- 使用原 Stage 2A：有正 routing 的路径先 priming，再周期 warmup 后 drain，measurement 样本/路由计数重置。
  specialist_share=0 时 specialists 仍加载/READY，但不发 priming、warmup 或正式请求给它们。
- drain_timeout=60 s；异常和 incomplete 保留，不能用 cohort throughput 隐藏积压。

## Full-deployment smoke gate

正式实验前使用相同 deployment/workload，specialist-first routing，warmup 10 s、measurement 20 s。
文件在输出目录 `configs/full_deployment_smoke.json`、`full_deployment_smoke.json` 和 `.log`。
`validate_smoke` 必须确认五个 worker 都实际完成请求、所有正式请求完成、无错误、全部 BYE/exit=0。
任一失败都阻止 characterization。20 s smoke 不计入 15 次正式 repetition。

## Running / resuming

```bash
cd /root/autodl-tmp/generalist-specialist
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python \
  -m joint_deploy.characterization.routing_sweep --execute
```

默认输出 `results/stage2b/routing_characterization/`。省略 `--execute` 只准备/聚合，不启动 GPU。
支持 `--output-dir`，新目录应先生成并通过其 full-deployment smoke。
已有 raw result 不覆盖、不自动重跑，失败点同样保留。resume 前核对封存 protocol/data 哈希，
逐轮核对 config 哈希。数据点异常会在汇总标记，不被删除；无 artifact 或外部 timeout 则停止，
保留日志供检查。需要明确重做实验时用新输出目录，避免混淆 protocol。

15 轮完成后执行跨 run 校验和描述性分析（均不启动 GPU）：

```bash
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python \
  -m joint_deploy.characterization.validate_dataset
OMP_NUM_THREADS=1 /root/miniconda3/bin/python \
  -m joint_deploy.characterization.analyze_results
```

描述性绘图使用已有 base Python 的 matplotlib；gsdeploy 不含 matplotlib，模型环境未安装或修改依赖。

文件：

- `plan.json`：固定配置、seed、实际执行顺序、原协议/数据哈希。
- `configs/`：15 个完整 MeasurementConfig。
- `raw/`：15 个未经平均的原 Stage 2A JSON，含完整 request records。
- `logs/`：各轮 stdout/stderr。
- `routing_characterization_summary.json`：所有 run、每次 audit、各指标 mean/sample std/min/max/values。
- `routing_characterization_summary.md`：system throughput、逐 routing 完整矩阵及逐任务 latency 表。
- `dataset_validation.json`：跨 routing 的同 seed 样本/相位对齐、独立 seed、配置及数据来源审计，以及 raw SHA256。
- `routing_characterization_findings.json/.md`：端点配对差、各任务敏感度、latency 分量与具体排队时间线。
- `routing_latency.png/.pdf`、`routing_latency_components.png/.pdf`：实测曲线与分解，可直接导出。

审计逐条复建原 arrival 和 deterministic routing，核对 timestamps、routing conservation、实际数量、
最小 task 样本数、完成数及 shutdown。routing 取整允许与目标数相差 ≤1；同时要求记录逐条符合原路由器。
completion ratio<1、worker ERROR、不完整或不正常 shutdown 都标记 overloaded_or_unstable，
它是需要分析的观测状态，不是将错误必然归因于负载过高。

## Statistical interpretation

三次重复分别计算，再报告 run-level mean/sample std(ddof=1)/min/max。三轮 P95 的平均不是 pooled P95。
per-task latency 独立保存，不能用 overall mean 替代。低频 ASR 每轮只有 30 请求，P99 接近极端样本，
需同时看三轮 values/std；不根据三次短实验宣称普遍规律或统计显著性。

Stage 2A cohort throughput=截至 drain 完成的该窗口请求数/100 s，另保留窗口内完成吞吐。
E2E = schedule delay + runtime；runtime 包含 handler、worker/pipe queue 和响应传输。
characterization 额外对原始记录计算 mean(runtime−worker)，仅为观测分解，不做排队预测。
worker mean 也随 routing 的模型混合比例改变，不能把所有变化都叫 GPU interference。

## Tests

```bash
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python \
  -m unittest discover -s joint_deploy/characterization/tests -v
```

测试包括 15 个配置的不变量与请求数、seed 变化、实际路由取整、保留 incomplete、错误 routing/config、
shutdown/worker errors 标记，以及 sample std/missing 数据行为。没有模拟数据混入真实实验目录。
