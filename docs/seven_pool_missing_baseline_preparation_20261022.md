# 原七模型池缺失 standalone 画像：当前设备准备合同

本阶段只准备 CPU 校验通过的原生执行包，不启动 GPU，不读取正在进行的实验输出。
原始来源是 `execution-packet-v2.json`、原七模型池 recalibration requirements 和九状态证据审计。
目标是 `GPU-8a262fe0-598c-46b7-c953-41d9da84a47e`、device 0、49140 MiB，沿用 safe_factor 1.110854。
历史 UUID `GPU-533a287b-0e15-a7c8-8e58-2b21bdcbb13a` 的失败不能证明本设备不可行。

## 五个独立上下文，九条路径

| context-id | 正式采集目标任务 | 每任务全局 primary 数 | full-capability priming 数 | 历史 raw residency 代理 MiB |
|---|---|---|---|---|
| generalist_1-n3 | text | 96 | 36 | 29238 |
| generalist_1-n4 | text、image、ocr | 96、192、96 | 48 | 38984 |
| generalist_2-n4 | text | 96 | 48 | 33960 |
| generalist_2-n5 | text、image、ocr | 96、192、96 | 60 | 41390 |
| generalist_3-n4 | ocr | 96 | 48 | 34888 |

原始池为 text32、image64、ocr32、asr32。每任务采集单位是**全局完整池每输入三次**，不是每个副本每输入三次。三轮分别预先打乱，按全局 index modulo n 分配副本；每 alias 的请求量和 unique input 数在 packet 与 cpu-risk-check.json 中逐项列出。合计 1056 条 primary、240 条 priming。每 alias 不一定覆盖完整输入池，不可宣称逐 alias 画像完整。

上表仅为历史驻留代理，不是数学下界、full-capability 峰值上界或当前设备可行性证书；任何行都不能据此剪枝。cpu-risk-check.json 另外列出 safe 代理和借用最近历史 n 得到的工作量估计；估计不是待测 service、稳定容量或运行时保证。

## 请求与失败合同

每个 context 启动全新原生 worker；先等全部 READY 并校验 model、spec SHA、四任务能力，然后按 alias、原支持任务顺序，对每任务 frozen size proxy 最大的三个输入串行 priming。所有副本的 text/image/ocr/asr lazy 路径都会驻留。

这比旧单任务画像更强。full-capability priming 或后续请求失败，证明的只是这个特定驻留/活动合同失败；不能据此排除只激活单任务的原路由域，更不能跨 n、跨任务、跨设备传播不可行结论。准备包不解决该更弱合同的所有未知状态。

同一 context 的目标任务依声明顺序测量，共享已 primed 的驻留 worker；每个任务先独立 warmup30秒并完全 drain，warmup 游标与 primary/support 分离。各 alias 一次最多一条未完成请求。primary 完成较早的 alias 继续使用预声明独立游标发送**同任务 support**，直到全局最后一条 primary 完成；已发 support 必须在同一 hard wall 内 drain。每 alias 每任务最多256条 support；support、warmup 和 primary 分开记录，support 不混入全局三次统计。support 上限耗尽、超时、原生错误、缺失响应均使当前 phase 失败，不回退到 tail 数据。

单请求上限300秒；每 alias READY 上限300秒；full priming 总上限3600秒；每任务 warmup hard wall330秒；text/image measurement hard wall600秒，OCR3600秒，包含 support drain 与 snapshot 开销。context hard wall 按启动、priming、全部 phase 上限加120秒冻结在 packet。shutdown 使用冻结 runtime 的 grace10秒与 terminate/kill 各3秒；强制结束或缺 BYE 单独标记失败。

失败停止该 context，保存原始响应、异常、时序、最后一次资源样本与清理状态；后续任务仍为 NOT_STARTED_UNKNOWN。所有未释放 primary 在运行前已登记，时间保持 null，最终标为 incomplete；不伪造 service。没有自动重试、自动下一 context 或自动排队。

## 原生绑定、资源与正式搜索边界

CPU prepare 复用冻结 v5 `bind_native_job`，只为 G1/G2/G3 重新散列当前完整 checkpoint 与质量报告。为身份覆盖构造的三个 family n1 部署没有执行，也不代表同时驻留可行。原七模型全部配置、质量、task/n 域、数学约束保留；只更新目标 UUID、身份绑定及种子/时间元数据。

执行入口复用冻结 v5 `DeploymentRuntime` / `MeasurementConfig` / `registry_for` / `PowerMonitor`，worker 使用原始多任务 `scripts/static_study_worker.py` 和冻结 native `ProfileAdapter`，不使用旧单任务 profiler 的 wire protocol。每个 family 四条任务路径通过 `verify_quality_evidence` 独立校验，无绝对质量准入。`MeasurementConfig` 只是运输元数据，实际请求由冻结 roster 的 closed loop 释放。

未知 service 上不调用正式 `SystemConfig.validate_plan`，不临时填入历史 service 来通过准入。正式 rho≤0.8、SWRR、score 与稳定性规则完全不变，正式 feasibility 仍 UNKNOWN。本工具是饱和 standalone 诊断，不是正式低负载稳定性验证或 U 比较。

内存从 worker 启动前开始采样，加载期间每次 startup event wait 都检查故障与 hard wall；safe_factor×已观测 owned peak 超预算、foreign PID 或查询错误立即锁存失败。READY 前尚未创建 CUDA context 的 owned PID 缺失被原样记录；全部 READY 后缺失 owned PID 无效。结束时在 worker 退出前最后采样，保留失败峰值；采样峰值不能证明连续物理峰值上界。Whole-board power 连续采样，保留 startup、warmup、measurement 区间积分与覆盖检查，不扣除监控开销。只保留原生 prediction 供后续冻结 scorer 使用，所有 U 为 null，不拟合、排名、激活模型或宣称全池最优。

## 使用与校验

工具目录：`tools/seven_pool_baseline_20261022/`。

```bash
python tools/seven_pool_baseline_20261022/prepare.py
python tools/seven_pool_baseline_20261022/run.py --packet artifacts/static-spillover-study-20261022/seven-pool-missing-baseline-preparation-v1/experiment-packet.json
python tools/seven_pool_baseline_20261022/run.py --packet artifacts/static-spillover-study-20261022/seven-pool-missing-baseline-preparation-v1/experiment-packet.json --launcher-check
python -m unittest discover -s tests -p 'test_seven_pool_missing_baseline.py'
```

默认 runner 和 checker 仅查 CPU 元数据、SHA 与确切 roster；无 subprocess、GPU 查询、模型加载。显式 `--launcher-check` 为三个 family 启动 `--check` CPU 子进程，对160个输入逐项构建请求；仍无 GPU 查询、模型加载与推理。

将来执行必须同时给出 `--execute --context-id <上表一个ID> --output <不存在的目录>`；工具只执行这一项。当前准备阶段没有调用该入口。输出是新的 `run.json`、`live.json`、`progress.json`，不覆盖旧证据。

packet 冻结了原始输入、源文件、元数据、请求顺序、工具及本文 SHA；preparation-receipt 另外记录 packet 字节 SHA 和 CPU 校验结果。CPU 检查不能替代运行前 UUID/idle 检查，也不证明当前设备可行。
