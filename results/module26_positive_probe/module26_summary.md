# Module26：正开发探针审计与独立复现

最终Gate：**PROBE_NOT_REPRODUCIBLE**。不开发预测器，不搜索新池，不自动进入后续Module。

## 历史开发正例审计

24A +0.476798为完整1500/1500 GPU配对的系统U，S-only=-2.256672、GS18=-1.779874。原配置是五个模型各1副本，固定六任务负载，ASR18% G分流；原对照仅将ASR改为Whisper。质量路径、safe/peak显存、完整到达输入匹配及原队列规则通过，G/S有峰值拥塞告警但恢复，无硬unstable。合法开发正例，不是独立验证。

24A分项ΔU：mean=+0.136153，P95=+0.342342，吞吐=+0.000000，memory=-0.001697。收益主要来自Text/OCR背景延迟改善。

24B负值来自另一配置：Text/Image/OCR各2副本、共8worker、ASR13.1833%且Text/Image有极小G流量，seed/输入cohort/顺序也不同。checkpoint、GPU、QPS、效用参考相同，但不是原18%单副本配置的独立重复。历史温度/时钟不足，不能唯一归因到硬件共置、热状态、输入或顺序。逐任务、逐worker及完整重算见positive_probe_audit.json和historical_*.csv。

原窗口仅120s，LID2、VQA32，尾延迟不确定；warmup与测量窗口分离且完全drain，未完成请求/窗口截尾/归一化重算错误均未发现。启动READY修复前没有测量结果，原失败保留。

## 独立复现

用户选择每臂2400s；原模型、副本、18%路由、六任务QPS、输入池、质量和U/refs全部冻结。三组新Poisson轨迹，顺序control→proposed/proposed→control/control→proposed。冷启动温度≤38℃、连续3样本，原prime/10s warmup/2s idle；温度、时钟、功率及队列原始数据均保留。

首次前台控制进程在约200s测量后消失，无完整raw请求记录，不能计算合法U；进度和日志作为不完整基础设施失败保留，不算正或负收益。根据继续指令，仅一次基础设施修复为独立后台进程+10s增量请求journal，同一冻结配置和种子，全部负结果保留，无按收益重试。源manifest及统计措辞修正均原样可审计。

| 配对 | G/S U | S-only U | ΔU | 可比 |
|---|---:|---:|---:|---|
| 1 | -1.975265919791881 | -1.6602028068650534 | -0.31506311292682754 | False |
| 2 | -2.072836721846691 | -1.6507906496250644 | -0.4220460722216266 | True |
| 3 | -1.994230871035093 | -1.6283064279913222 | -0.3659244430437707 | True |

独立配对t95区间：{'mean_delta_U': -0.3676778760640749, 'low': -0.5006116098821861, 'high': -0.23474414224596377, 'n': 3, 'df': 2, 'method': 'Approximate paired-effects Student t95; independent sessions, assumed normally distributed effects', 'small_sample_uncertainty': 'n=3: heavy assumption dependence; report all effects, no broad model-combination generalization'}。n=3强依赖效应近似正态；三次同正的单侧精确符号p最小0.125，不能声称分布无关5%显著。逐组60s block bootstrap只是条件敏感性，不冒充独立重复CI。LID的P95 order-statistic区间可能无有限上界；59样本支持最大值的一侧95%覆盖，72才支持一侧97.5%，统计说明单独保留。

每任务mean/P95、服务/等待、原始质量指标见per_task_latency.csv；pooled mean/P95、offered/completed/within、safe/peak显存和实际G份额见gpu_paired_runs.csv；各分项和worker busy/backlog/slope见utility_decomposition.csv及worker_stability.csv。不稳定/不完整U为null，完成子集统计仅诊断。


Stage B没有通过可重复正收益gate，Stage C未启动，没有新模型池搜索或附近路由GPU重试。

## 六个问题

1. 为什么24A与24B不同：副本数、路由、输入、运行顺序不同；效用口径相同，背景mean/P95贡献翻转，尚无唯一物理因果分离。
2. 24A是否合法稳定真实正例：是，按原稳定性规则recoverable且完整，是开发正例，独立性不能升级。
3. 独立GPU是否复现正收益：False；ΔU=[-0.31506311292682754, -0.4220460722216266, -0.3659244430437707]，Gate=PROBE_NOT_REPRODUCIBLE。
4. 是否来自系统mean/P95：历史24A主要来自背景mean/P95；本轮以system_utility_comparison.csv逐组分项和per_task_latency.csv为准，不把目标吞吐或pooled P95代替系统U。
5. 正式优化器能否选中：False；未执行Stage C不能据此断言其算法必定失败。若有正配置但未选中，报告为“存在正收益配置，但算法未成功发现”。
6. 继续LOW/MID研究的HIGH正例依据：False；自主LOW/MID正式验证准备条件：False。仅HIGH正配置但未自主发现时，先研究固定池算法排序，不能宣称已有三阶段收益。

历史16146个文件metadata和全部读取科学输入SHA256不变，24A/24B/25原gate未修改。所有失败、负结果和中断证据保留。完成后停止。
