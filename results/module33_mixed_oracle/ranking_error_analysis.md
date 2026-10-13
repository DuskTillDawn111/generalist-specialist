# 排序误差与归因

完整开发GPU Oracle=True；参考与优化器同方案=True。

搜索层：四组U网格精确Oracle与20A/原生V3预测部署匹配，输出相对U网格Oracle的regret=0；原生连续空间最优性未知。MILP网格适配与原生连续结果都保留。原V3接受原CPU预测性能、原噪声规则；没有把正式流程替换成最大U排序。

目标层：正式V3的J routing/ranking与20A总U定义不同，J全局最优在CPU下反而比U最优多部署副本，U成本差见objective_error_decomposition.csv。正式V3没有单一全局J部署目标；其候选生成/Top2与真实噪声接纳规则同时决定输出，不能把J-global gap默认解释为搜索算法错误。

预测层：保持20A独立standalone s0/g/M不变，CPU为周期到达+常数服务，GPU为冻结真实输入+Poisson到达；输入变异、共置、runtime和窗口到达差异均可贡献误差。每个U分项的预测−真实误差和原生执行/等待保留，不做本轮后验拟合。此实验不能唯一识别纯GPU干扰/CPU/frontend/arrival机制各自因果贡献。

实测诊断：{"n": 8, "Spearman": 1.0, "Kendall": 0.9999999999999998, "U_MAE": 1.0987012365410898, "all_frozen_candidates_measured": true, "false_nominal_stable_predictions": 0, "omitted_invalid_or_unstable_from_sorting": 0, "point_reversals": 0, "reversals_with_within_trace_CI_excluding_zero": 0}。

排名层：8个配置均按同一清单竞争；技术无效和unstable U为空。开发最佳是该窗口观测最大，并非潜在期望U最大或连续全局最优。bootstrap仅为同轨迹分块不确定性诊断，不删微小反转或用不显著差距强称严格排名。

独立配对：{"n_valid_pairs": 3, "expected_pairs": 3, "mean_oracle_minus_selected_U": 0.0, "CI95_low": 0.0, "CI95_high": 0.0, "all_same_candidate": true, "interpretation": "Same candidate is an identity comparison, not independent rank validation against other candidates."}。

初始失败及环境：Phi缺失隔离cache预置导致READY前失败，与任何实测收益无关；byte-identical恢复缓存后未改配置/算法。每个运行的传感器/worker请求/队列/退出保留在raw_runs与request_journals。没有改写历史负收益。

下一轮改进：仅当新实测排序与CPU不一致时，在下一轮另行做预测器/目标/搜索改进和新独立验收；本轮到此停止。

## 最终证据边界补充

正式V3保留原create/search/接纳逻辑，性能接口明确使用冻结20A CPU预测；随后独立测量输出部署。没有开展完整的GPU反馈式V3搜索，因此下面的速度只指CPU预测后端的求解开销，不能解释成完整在线GPU搜索加速。

Stage C范围限制：原生正式V3路由器仍是连续变量，最终结果都落入冻结网格，但搜索中存在网格外路由（formal_route_domain_audit.json逐项列出）。它与网格Oracle的最终输出regret比较有效，却不是严格相同有限路由域的搜索比较；只有20A有限域适配这一开发算法执行了相同路由网格。正式V3的J和噪声规则未改，不能用20A适配结果冒称正式V3的同域验收。本轮保留该未满足条件，不后验修改正式路由器重新验收。

在GPU启动前记录的首次CPU计时，大池正式V3耗时比84.38–91.25×，原生连续20A耗时比15.84–20.47×，20A网格适配同域加速26.88–32.11×。原始耗时见native_optimizer_speedup.csv。正式V3/原生连续20A与网格枚举不是相同内部路由域，耗时比仅供描述。上表V3三次中位数包含与首次GPU启动/warmup重叠的后两次计时，保留但不能视为严格隔离的CPU效率实验；枚举每格只计时一次，所有速度均为单机描述性结果。两池不仅模型数量从6增至7，副本上限也扩大；不能把开销变化唯一归因于增加一个模型，更不声称渐近复杂度。

最高两项不确定性：{"best": "C2f410cf3d145f0f3", "runner_up": "C6494564b5e815e73", "delta_U": 0.0020740720826062375, "CI95_low": -0.024788140052446117, "CI95_high": 0.05037670837657149, "strict_ranking_supported": false, "scope": "Unadjusted paired 30s block diagnostic on one development trajectory. Memory fixed; no between-run uncertainty."}。最高点差约0.0021，低于原20A正收益门槛0.01，且区间跨0；不能据此宣称OCR一副本严格优于二副本，也不能宣称优化器严格优于J-routing基线。

获选部署独立轨迹复测的实际U：{"candidate_id": "C2f410cf3d145f0f3", "n": 3, "observed_U": [-1.1829417168373317, -1.1535286577592934, -1.1917216998671385], "mean_U": -1.1760640248212546, "offered_completed": [{"run_id": "PAIR1_C2f410cf3d145f0f3", "offered": 356, "completed": 356, "within": 356, "T_norm": 0.0}, {"run_id": "PAIR2_C2f410cf3d145f0f3", "offered": 367, "completed": 367, "within": 367, "T_norm": 0.0}, {"run_id": "PAIR3_C2f410cf3d145f0f3", "offered": 326, "completed": 326, "within": 326, "T_norm": 0.06963470319634701}], "CI95_low": -1.2257563483971916, "CI95_high": -1.1263717012453176, "scope": "Three independent frozen request traces; descriptive t interval, same deployment. Does not validate other candidates independently."}。Oracle与算法为同一配置时，经验regret=0是身份关系；三条独立轨迹复测提供该部署的效用与波动数据，不构成竞争候选的独立排名确认。

独立轨迹均无请求丢失。第三条Poisson轨迹只有326个到达，少于240秒×1.46 QPS=350.4，原20A固定QPS分母导致T_norm=0.069635；保留此原定义，不能解释成系统丢失请求，也不后验改写分母。三次窗口/样本量不同于开发120秒窗口，开发与复测U差值不能直接归因于单一机制。

本Module不满足20A冻结的通用holdout支持规则（至少3个独立匹配部署/负载组、跨至少2模型组合等）；20A历史UTILITY_MODEL_INSUFFICIENT未被改写或升级。CPU四格搜索U regret为0，性能预测U MAE已量化，但不识别各性能机制的单独因果贡献。

基线比较：ASR32B的小池中，fastest-service静态启发式和最佳一副本不分流静态类也选择同一个观测最佳部署，差值为0；本轮不证明优化器优于这些简单基线。最小显存启发式实测U=-2.514951，获选方案U=-1.042465；J网格参考U=-1.044539，其与获选方案差距无法可靠区分。BEST_ONE_REPLICA_NO_SPLIT是受限静态类别内的参考最大，不能当作无需搜索的廉价启发式。所有相同配置复用该配置原始测量，没有额外伪造独立试验。

空间限制：SMALL6的六任务覆盖强制5个模型族驻留；自由度主要是额外ASR Specialist是否驻留、OCR1/2副本及ASR离散路由。LARGE7增加替代Generalist和实测副本选择。不能把这个充分测量的小问题代表为原始完整模型选择问题。

首个有效GPU记录曾因journal缺少phase末端才派生的3个字段被评估器误判。用原native annotate_records从相同时间戳在内存重建后，完整记录逐项相等；原raw字节、初始失败评估与修正说明均保留，未重跑GPU或修改U/门槛。原native harness的raw provenance引用初始v1清单；run_manifest_provenance.json提供实际v3/v4启动清单映射，原raw保持原样。

统一三结构选择框架可以继续扩展，但原合法Specialist集合没有visual_qa路径，六任务纯S在当前质量准入下不可行；不能把仍驻留VQA Generalist的ASR S-only路由改称纯S。纯G须纳入已准入的OCR Generalist并重新冻结预算/容量。详见unified_structure_extension_readiness.json。本轮未启动Module34。

[实验图PNG](/root/autodl-tmp/generalist-specialist/results/module33_mixed_oracle/module33_benchmark.png) · [可导出PDF](/root/autodl-tmp/generalist-specialist/results/module33_mixed_oracle/module33_benchmark.pdf)
