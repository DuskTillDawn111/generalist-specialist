# Same-donor actual two-FIFO 训练诊断（CPU）

`actual_worker_twoFIFO_training_diagnostics_v4.py` 将冻结的 joint request work/protocol law
接入实际 READY 释放 worker 的 v4 执行器。每次 whole-training-capture leaveout 只从另外两份训练 capture
选择 donor，保留 v3 的相同 common capture quantiles `.25/.75`、seeds `0/1`、public-key ordinal draws。
response work、所有 START/PHASE/query/RESULT/READY marks 与 notification slacks 保持同 donor。
训练 donor 的 d/w 通过 law 向量和按 ordinal 配对的 provenance 关联到 FIFO raw audit，
而非以 vector hash 当作唯一键或查找 target request 的执行答案。

每个原 job 使用完整 public arrivals 和原 arrival offsets，保留 60s 诊断 drain。
生成过程中不读取 heldout peer path；heldout completion 仅用于事后误差统计。
每个 old-v3 重算 RESULT reader latency 均与 training05 冻结记录逐条精确对齐。

两个 transport sensitivity 世界使用同 donor 的联合 pair：`d=START-submit`、`w=write_record-submit`。
`line_min` 取 `x=0,pre=d`；`line_max` 取 `x=min(d,w),pre=d-x`。
每个 offered public-key ordinal 都显式构造 `TransportDraw(0,0,w,x,d-x)`，并移除 response plan 中旧 idle handoff，避免双计。
两个零 extra 是为了保持 v3 的 immediate dispatch baseline，属于显式假设，没有测得 sender CPU demand 为零。
移植 idle d/w 到 busy transport 的资格仍为 false。sender/worker 初始 free、notification pending 初始为空同样是显式世界假设。
历史 priming/control/background 初始状态未得到认证。

报告：`artifacts/predictor-actual-worker-twoFIFO-training-v4-natural-20261028-diagnostics01/report.json`，
SHA256 `3dc40b82c7c4087b9248a6b1768ba3cddedff81e04b3b0efa5da9cf28c09cfe1`，617 个 sources 复核通过。
原三份 capture 各 26 job、50 requests，四个 joint donor worlds 和两个 line endpoints 合计 624 worlds、1200 offered world-requests。
所有 offered 均保留并完成；每个 endpoint 有 600 world-requests 和 150 个完整四世界 ensembles，UNKNOWN 为 0。

最关键的支持限制：每个原 job 内每个 owner 最多只有一条请求。
在明确的 per-job empty-initial 世界中，READY processing 和实际 READY 从来不会竞争释放该 owner 的第二条请求；
而两个潜在分解对首次 idle START 都恰好给出 `submit+d`。
因此所有 624 worlds 的 START 与 RESULT reader 对照都与同 donor v3 精确相同，line 两端结果也相同。
这不能说明两种 FIFO policy 在 busy queue 等价，不能估计 busy policy error，也不能认证 idle pair 的 busy 迁移。
实际差别由单独的 queued CPU 反例证明：相同首 START=1 时，旧替代 policy 的次 START=120，实际 FIFO 为 12。

两个 endpoint 得到相同的训练诊断统计：

| 统计 | ms |
| --- | ---: |
| 单个随机世界的平均绝对误差 | 160.038387 |
| 四世界 ensemble mean 的 MAE | 153.806160 |
| 四世界 ensemble mean 的 RMSE | 253.359223 |
| 四世界 ensemble median 的 MAE | 153.900813 |
| 四原子经验分布的 CRPS | 119.849972 |

这些是固定四世界经验分布的训练诊断，不是校准后的总体预测分布，也未以该结果选择参数或模型。
独立复制单位仍是三份训练 capture，26 jobs 和 draw worlds 不增加 capture 样本量。
没有资源/质量模型、完整原任务 utility 或最终 gain 资格。
query-bound wrapper 逐 world 核验完整 public cohort 和原窗口/drain，保留 fingerprint 供下游资源 timeline 使用。
所有实现仅 CPU，无验证标签、GPU 初始化、新采集或 Nsight export；native CUDA API 数值认证仍为 **UNQUALIFIED**。
