# Module33：Mixed Deployment Oracle Benchmark and Optimizer Evaluation

Gate：`MIXED_OPTIMALITY_SUPPORTED`。CPU 与 GPU 结论分别判断；历史保护=True。

证据限定：正式V3最终输出匹配有限网格Oracle，但内部仍使用连续路由；Stage C严格同有限搜索域条件未完全满足。完整8配置GPU观测最大与独立同方案复测成立，不是跨候选独立排名或通用全局最优证明。详情见文末与final_scientific_evidence.json。

正式算法是冻结 V3：J 决定任务路由与候选排名，真正的部署接纳使用原 NoiseAwareRuntimeObjective(1ms, 5%) 的稳定性、原始 mean/P95、显存比较；没有单一全局 J 或 U 部署目标。20A 是直接最大化预测 U 的开发算法，其 UTILITY_MODEL_INSUFFICIENT 历史 Gate 保留。Module25 的扩展未执行、32B 动态策略未修改。此区别不能用“J改成U后的表现”掩盖。

Stage A 冻结 6/7 模型两池：SMALL6 的 Phi/Whisper-large/Roberta/ConvNeXt/PARSeq/Whisper-small；LARGE7 加 Qwen 与部分实测双副本。Whisper-small 的 ASR 被明确排除，仅保留原32B LID路径，原质量准入不变；受限空间不代表14F完整模型池。六任务均有正QPS。LOW18来自18的text_low_rep1，ASR32B来自32B A0。预算34567.5534MiB、safe_factor、20A权重与逐任务归一化参考不变。全部路径网格为QPS的0/0.5/1弱组合；最优性只覆盖这个离散空间。

20A有限域适配只对原MILP路由变量做单位换算/整数网格约束；原42个生成尝试、目标、权重、.8/.95容量种子和U排序未改。原生连续20A另外执行，四组最终结果均恰好在网格内。正式V3使用原生create/search及原始接纳流程，RuntimeEvaluator明确用原CPU预测接口提供性能；不是dry_run仅返回seed，也不是实测GPU自适应搜索。四组native V3最终结果也均在网格内。

|池/负载|全部副本向量|覆盖后网格组合（显存前）|合法组合|Oracle U|20A/V3 U regret|CPU穷举秒|20A中位秒/加速|V3中位秒/加速|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|SMALL6/LOW18|96|8|8|-0.527970|0 / 0|0.021133|0.055337 / 0.38×|0.035767 / 0.59×|
|SMALL6/ASR32B|96|8|8|-0.521009|0 / 0|0.022212|0.055692 / 0.40×|0.024915 / 0.89×|
|LARGE7/LOW18|1458|79704|2101|-0.527970|0 / 0|5.144888|0.188437 / 27.30×|0.060971 / 84.38×|
|LARGE7/ASR32B|1458|79704|2101|-0.521009|0 / 0|5.789698|0.180770 / 32.03×|0.063449 / 91.25×|

四组最优均为6模型各一副本，ASR全部走Whisper-large，VQA走Phi，其他四任务各走对应S；没有ASR分流。U目标选择、模型驻留、副本和路由均与U Oracle一致，无约束违规。每次20A42个MILP尝试，只评估3–5个唯一候选，包含四个预测场景；小池访问3/8=37.5%，大池约0.14%–0.24%。内部MILP分支并不等于显式候选访问。小空间枚举更快，大空间才有速度收益；仅两个真实画像规模与单机时序，不声称渐近复杂度定理。

CPU Oracle另有独立笛卡尔路由枚举，逐项覆盖计数/集合一致；独立标量FIFO回放与原predict名义U误差<1e-9。全部候选使用相同predict_scenarios，名义phase=.37、120秒、rho<1；原V3仍保留rho<=.8。J空间Oracle单独在原.8约束下取最小值。J最优偏好更多副本，其U相对U Oracle差约0.0104–0.0478；这说明目标的取舍不同。正式V3最终也不必达到全局J最小，因为原接纳噪声门槛是另一项规则，不能把这种J差距直接叫算法搜索失败。详见objective_error_decomposition.json，正式选择在原pairwise接纳规则下的有限候选改进数量也单独列出。

GPU全部配置和种子在实测前冻结。8个方案共同使用六任务ASR32B负载、同输入/Poisson轨迹/route与replica uniforms；开发窗口120s、warmup30s、drain60s、逐次fresh PIDs。配置包含有/无Whisper-large、OCR1/2副本、Phi ASR份额0/0.5/1；显存/质量/系统与逐worker稳定性失败的U=NULL。GPU范围不是CPU前几名，也没有更改动态分流策略。

|候选|OCR副本|Large驻留|ASR Phi份额|预测U|开发实测U|合法|
|---|---:|---|---:|---:|---:|---|
|C01b83ef70ec4e147|1|False|1.0|-1.038176|-2.514951|True|
|C8ca12d7cd7d8bc7c|1|True|1.0|-1.073413|-2.581927|True|
|C7c17dae4499ffa3a|1|True|0.5|-0.928145|-1.733013|True|
|C2f410cf3d145f0f3|1|True|0.0|-0.521009|-1.042465|True|
|C626b54e4a880b416|2|False|1.0|-1.048604|-2.571987|True|
|Cc2ce1b5bead0ce10|2|True|1.0|-1.083841|-2.659571|True|
|C534ebceea3ea8d52|2|True|0.5|-0.938573|-1.804356|True|
|C6494564b5e815e73|2|True|0.0|-0.531437|-1.044539|True|

GPU完整性：8/8个配置完成测量，8个合法可排序。真实参考方案=C2f410cf3d145f0f3；优化器方案=C2f410cf3d145f0f3。经验开发regret=0.000000。

独立验证：3/3个有效配对；参考−优化器平均ΔU=0.000000，t95 CI=[0.000000, 0.000000]。同方案比较=True；同方案的0差值是恒等关系，不能伪造为其他候选的独立排名验证。独立轨迹用于复测获选方案；其它候选只具有完整开发空间测量证据。

预测与实测：{"n": 8, "Spearman": 1.0, "Kendall": 0.9999999999999998, "U_MAE": 1.0987012365410898, "all_frozen_candidates_measured": true, "false_nominal_stable_predictions": 0, "omitted_invalid_or_unstable_from_sorting": 0, "point_reversals": 0, "reversals_with_within_trace_CI_excluding_zero": 0}。排序微差的30秒配对block CI见gpu_ranking_pairs.csv。CI跨0时不做严格排名，内窗bootstrap没有识别序列次序、热状态、显存测量误差或跨运行随机性。不同配置环境差异完整保留。

基础设施负结果保留：新增评估脚本曾在任何worker启动前语法失败；其后首个配置在Phi READY前因漏预置隔离动态代码缓存失败，无测量窗口/U。恢复32E/F已有的byte-identical缓存预置后按原配置/种子重新启动，失败raw与NULL结果保留且run_id不同。没有根据GPU效用更改算法、模型、QPS、候选或门槛。任何真正测量失败都不会补值、删除或调整后验验收。

最终六项回答：
1. 在四个冻结有限空间中，原生20A和正式V3的CPU预测后端输出均匹配有限网格U最优；20A有限域适配自己的网格U目标regret=0；原生连续域全局最优性未知。正式V3并不以U或全局J为单一部署目标，J-routing精确Oracle及原噪声接纳规则的结果分别报告。
2. GPU开发空间参考−优化器ΔU=0.000000；独立配对平均ΔU=0.000000。证据范围为全部8个有限配置，不外推连续/完整模型池。
3. CPU有限网格参考下的输出regret为0，原生连续空间搜索误差未获证明。目标函数差异确实存在，J最优与U最优不同；GPU预测误差由实测U/分项误差、排序反转及稳定性误判独立量化。不能将GPU逆序直接归为搜索失败，也不能以相同胜者忽略绝对预测误差。
4. 求解速度见上表：小空间固定求解开销使其可能慢于穷举；大空间明显更快。V3和20A的时间分别报告，三次重复与目标调用计数留档。
5. 受限离散CPU最优性证据可复现；完整GPU排序证据=True，独立配对完整=True。适用于论文的范围是带限制、失败和不确定性的benchmark结果；没有通用真实全局最优证据，论文用GPU结果准备条件=True。
6. 可在另行授权下定义纯G/纯S/Mixed统一候选域并复用Oracle与测量框架；本次混合空间强制覆盖的G/S必要路径不能当作三结构竞争证据。原预测器仍需按GPU结果评估是否可迁移。本轮不进入Module34。

历史前后完整SHA256：True，文件数=21488，包括历史结果、源码、Gate与模型权重；全部新输出/缓存位于Module33隔离目录。

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
