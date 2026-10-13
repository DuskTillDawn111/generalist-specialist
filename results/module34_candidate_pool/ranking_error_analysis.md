# 排名与决策误差



20A在A/B的ASR32B及TEXT10四格中，生成集内U排名正确，但漏掉更好的部署/路由；这是候选生成搜索误差。另核对原生20A连续方法与网格适配的九组最终方案，二者驻留/U相同、路由按冻结1e-10归一化指纹等价（C有约3e-15的MILP浮点残差）（existing20A_native_vs_grid_decisions.json），因此本轮四个终点路由遗漏不是仅由整数网格适配新产生。二者中间候选域仍不同，原生结果单独保留。最大预测regret10.844309（A TEXT10）。同一G2部署中，最快服务路由把图像/ASR/VQA留给忙于文本的Qwen；Oracle将这三类任务转给MiniCPM，降低队列竞争。B TEXT10同理，V3控制已找到相同U参考最优。这证明队列敏感候选生成是已有20A的关键缺口，不能仅改名为U优化就声称解决。

V3控制的三处U差距LOW18/C .004391、ASR32B/A .355421、ASR32B/B .041308，对应其已找到J参考最优，但该J方案并非U最优。其余C ASR/TEXT10虽没有选择全局J方案，却在原多指标接受规则下选到U最优；这再次说明全局J差距不是该引擎唯一“自身目标搜索误差”。J拥塞代理与U的周期FIFO性能接口不同，J/U两种精确Oracle差异同时反映目标定义及预测代理差异；不能将其全量归为纯目标权重误差。详见objective_and_search_error.csv和oracle_margins.json。另对每组所有合法驻留状态（C302状态/组）先选择精确J网格路由，再使用原NoiseAwareRuntimeObjective逐项与控制版最终方案比较，九组均无原接受规则判为严格更好的候选，见V3_original_acceptance_audit.json。这是完整有限域的多指标“无可接受改进”审计；比较器含容忍门槛且可能依赖路径，不能说成唯一标量全局最优，也不能外推到原生连续域。正式V3原生数值失败完整保留，不在GPU后修复重验收。

原20A预测器固定使用120秒周期到达、phase0/.37及1.25×/2×服务压力场景；名义U来自phase.37，压力上下界不是统计置信区间，也不是Poisson请求模拟。真实GPU为预冻结Poisson协议（开发120秒、独立240秒），故到达突发性、画像设备、并存干扰及时间窗都属于预测适用性差异，不能全部归因于搜索。per_task_prediction_diagnostics.csv分列模型服务、worker队列及主机传输/调度残差，不通过改预测器消除本轮负结果。

真实预测误差依据每个有效GPU窗口的观测U与同一预算下冻结预测U计算，见gpu_results_table.json；排名结论必须依据独立观测及其不确定性，不以预测自证真实最优。prediction_search_decomposition.csv按恒等式“观测有符号差=冻结预测差+差分预测误差”分列；两类误差可以抵消，不强行拆成正百分比。method_empirical_regret.csv将每方法映射到实测相同方案，报告A/C两方案及B四个同TEXT10已测方案集合的观测均值regret，仍不是完整CPU池真实Oracle；未测方法保留缺失，不伪造实测值。





原GPU清单在观察任何GPU测量前冻结六方案、开发seed340101/120秒、独立seed340201/2/3/240秒和AB/BA/AB次序。CPU选择记录在gpu_selection_record.json：增加A最大搜索缺口病例，保留原CPU freeze；B最大20A/最快路径预测regret；C预定ASR32B Phi与Qwen比较。开发测量建立这三个两方案集合的排序；独立轨迹验证各自预选方案，未按开发结果临时换竞争结构。每窗口全六任务Poisson工作负载、共同输入轨迹与路由均匀数、30秒暖机、60秒有界drain。稳定准入沿用原分类器：unstable不合法；recoverable_congested必须完整完成并确认恢复才可获得U，因此“有效”不等于全部窗口全程steady（逐项gpu_stability_audit.csv）。严格检查所有请求、原worker时间戳、系统/每worker稳定性、安全显存、外部GPU进程、温度/时钟、持久终端日志及两套原U算术。原module26.arithmetic在未调用其旧_path观察器时，components.U/null和components.eligible/false保持默认占位；它的raw_arithmetic_U是纯算术结果，本轮顶层gpu_observation.eligible与U才是经全部冻结检查后的准入值（两套算术误差≤1e-12），不以占位字段替代本轮准入。不稳定或硬约束/技术无效方案U为NULL。

| 比较 | 开发优胜 | 独立n | 观测ΔU（Oracle所选−对手） | 95%CI | 判断 | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| A_TEXT_HALF_MID18 | ORACLE | 3 | 21.658290567158588 | [15.152162008032839, 28.164419126284336] | PREDICTED_ORACLE_SUPPORTED | 0.014517247592876027 |
| B_TEXT_HALF_MID18 | ORACLE | 3 | 0.08579362982853134 | [-2.7103006615055394, 2.881887921162602] | STATISTICALLY_UNRESOLVED | 1.0 |
| C_ASR32B | ORACLE | 3 | 0.05087046901498019 | [-0.22385006103782198, 0.3255909990677824] | STATISTICALLY_UNRESOLVED | 1.0 |

ΔU=CPU U Oracle所选方案减对手。误差线是三个独立种子配对差值的95%t区间（n=3，依赖配对差值近似正态，统计功效有限）；同时报告三比较Holm校正p，不将跨零微小差距当唯一赢家。对照温度均值差≤5℃、时钟相对差≤15%时才作可比性判断。开发与独立记录全部保留，包括负结果、故障和不稳定窗口。

A Oracle纯G与B Oracle Mixed复用相同TEXT10独立输入，可以检查同一B域内两种结构的观测对比：{'matched': [{'rep': 1, 'G_only_U': -35.259163651219644, 'Mixed_U': -13.362332743439989, 'Mixed_minus_G': 21.896830907779655, 'temperature_mean_delta_c': 1.535849056603773, 'clock_relative_delta': 0.0008137715179967753}, {'rep': 2, 'G_only_U': -15.301565687603606, 'Mixed_U': -11.498823367225897, 'Mixed_minus_G': 3.8027423203777087, 'temperature_mean_delta_c': 0.9981694266080865, 'clock_relative_delta': 0.0005512992222683755}, {'rep': 3, 'G_only_U': -19.426897318318424, 'Mixed_U': -11.834489765562912, 'Mixed_minus_G': 7.592407552755512, 'temperature_mean_delta_c': 0.7238805970149258, 'clock_relative_delta': 0.0002683510857899307}], 'statistics': {'n': 3, 'mean': 11.097326926970958, 'sd': 9.542659761146284, 'ci95': [-12.607954056094922, 34.80260791003684], 'p_two_sided': 0.181580347059335}, 'scope': 'Two preselected competing plans admitted in B; not all-B measured oracle.'}。这仍只是两个预选方案的比较，不能代替全部B配置GPU测量。

Module33八组有效开发GPU中，只有三组符合本轮C成员、副本1与端点路由域；已按49140预算重算U，原U/raw/Gate均只读。OCR2及0.5分流方案明确排除，详见module33_GPU_reuse.csv。重算历史开发数据不是独立新轨迹，也不把33旧预算下的全八配置Oracle自动移植为34全C真实Oracle。另外复用33三组原独立240秒LargeASR结果，按新预算重算，并对本轮SmallASR独立结果作次要未配对Welch比较（historical_large_vs_new_small_analysis.json）；seed不同，绝不称配对验收。该分析在本轮C独立结果产生前锁定，不改变原三项主配对或方法，环境不匹配时不作明确胜负结论。software_environment_versions.json记录本轮实际Python与包版本；14F未提供完整可比包版本，所以不根据相同env路径推断历史软件完全相同。本轮未安装/修改原环境，原worker代码和权重保护。历史14F性能画像来自另一GPU UUID；33/34测量使用当前同一4090（49140MiB）UUID，设备来源差异保留。

共六个预选方案全部测量也只建立各两方案集合内观测最佳，不覆盖CPU枚举全集；故GPU_ORACLE_INCOMPLETE保持true，MIXED_EMPIRICAL_OPTIMALITY_SUPPORTED不自动通过。GPU排名反转时记录诊断，把性能预测改进留到后续研究，原方法结果不替换。

