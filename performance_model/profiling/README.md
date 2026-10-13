# Module 2B: Qwen feature-directed profiling

本目录只做受控 profiling、数据审计与描述性分析。正式 feature extractor 仍是
`performance_model.features.FeatureExtractor`；inference、arrival generator、router 与原始结果保存
复用原 `joint_deploy.measurement.run_measurement`，不改 worker。

## Design

- A: Qwen-only，沿用已有 `(text,image,ocr,asr,vqa,lid)=(2,1,1,.3,.5,1)` 的比例；
  根据 catalog 的 s0 转换成 demand mix，u_Q=0.1…0.7，每点三轮，逐点升序。
- B: Qwen-only，u_Q=.4；text/vision/audio dominant demand vectors 反算请求率。
  balanced 直接复用 A_u_0.40。不是相同 req/s 的比较。
- C: 固定 Qwen+DistilBERT+Whisper+PaddleOCR 的驻留环境，Qwen u=.4 与 balanced mix 不变；
  no-active-contender 与三种单路径竞争负载比较，各三轮。共 runner u=.2，仅作流量尺度，不作真实利用率解释。
  三条路径由离线资源特征 min-max normalization + farthest-first distance 选出；不遍历 deployment combinations。
  低 kernel-time anchor 为 DistilBERT/text，后续路径为 Whisper/ASR、PaddleOCR/OCR。

最多42 个独立 run，全新进程；seeds=1101/2202/3303。同一 seed 在各配置复用。
B/C 每个 repetition 内按固定 seed shuffle condition 顺序。
每轮先原 harness 的逐路径 priming，再 warmup=10s，再正式 measurement。
measurement 时长按最慢 positive service path 约30请求计算，限制60–900s；低样本tail明确标记。
OMP/MKL/OPENBLAS=1。数据池与 inference 协议固定。

## Execution / resume

```bash
cd /root/autodl-tmp/generalist-specialist
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m performance_model.profiling.plan
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m performance_model.profiling.run
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m performance_model.profiling.analyze
```

已有结果由 runner 复用，不静默重跑或删除异常点。不要在实验进行中重新生成计划。
原始数据与 configs/logs 按 family 分目录保存；每轮真实 x/r/lambda 都经共享 extractor 重算，
另将 cohort count/window 得出的 observed rates 经同一 extractor 计算，以区别目标与有限窗口取整。

A 每点完成三重复，任一出现completion_ratio<.98、持续backlog增长、drain>30s、incomplete/worker error，
就停止A的更高负载。drain硬上限60s；整个run安全上限2400s，超时先SIGINT让原harness保存已有请求并清理，
再限时强制终止。无完整窗口的异常不会伪装成正常结果。B/C是独立实验族，异常条件仍保留三重复。
A_u_0.40 即使因 A 提前停止而未在升序扫描中执行，仍作为 B 的 balanced anchor 补跑。
`execution_coverage` 区分这一延后执行点和真正跳过的高负载点，核对剩余 run 与完整性。

持续backlog判定：10等长区间的时间平均backlog；后半段增长至少max(10 requests, offered*2%)，
至少3/4相邻区间均值增长，窗口结束backlog至少为上述增长阈值一半。是预先固定的操作性标记。

## Interpretation and artifacts

`profiling_plan.json` 保存所有配置、选择依据、source hashes、expected u/q/C、seed/phase。
`profiling_summary.json/.md` 保存逐run与每配置 mean/sample std；`dataset_validation.json` 是完整审计。
`dataset_rows.jsonl` 由现有 dataset builder 转換原始请求，保留result绝对路径，不混入合成数据。
`load_axis.png`、`task_mix_axis.png`、`colocation_axis.png` 仅供 characterization。
报告同时保存 runtime 与 schedule delay，避免将主机sender排队误归为worker队列。
汇总延迟只用成功请求；incomplete/error保留，截断轮次的P95不能代表完整offered群体。
`dataset_validation.json` 另外核对既有核心源码/审计文件哈希及实际应执行的重复数。

A通过完整性、三重复、样本数、稳定性检查的点才可写入 `piecewise_load_points.json`，
初始化既有 piecewise baseline，无训练/accuracy报告。异常点仍完整保留在原始数据与汇总中。

C 的限制：原harness按task而非service-path独立生成到达流，竞争者与Qwen可能共享task。
添加竞争路径会改变Qwen实际到达间隔、任务样本subsequence；analysis保存配对seed的sample IDs比较、
音频时长分布。因此 C 只能做描述性关联，不能因果归因于某一资源维度。
资源相关性只用4个condition均值（含control），不是把三重复当作12个独立pressure条件。

本阶段不实现deployment search、routing optimization、最终performance predictor或多副本。
