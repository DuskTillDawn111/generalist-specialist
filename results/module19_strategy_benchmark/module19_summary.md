# Module19：Deployment Strategy Benchmark & Routing Ablation

**V3定义修正：原Module19包装错误使用dry_run=True，原engine在初始化/排序后退出，没有执行原V3的接受/回滚GPU反馈搜索。已测数据仅为初始化诊断，不能证明Proposed优于或劣于完整原V3。完整Baseline C标为UNRESOLVED_RUNTIME_FEEDBACK_REQUIRED；不虚构CPU P95或GPU反馈。其余A/B/Proposed和固定驻留路由消融不受影响。**

发现后在完整run保存和GPU释放边界停止旧控制器，新控制器仅跳过尚未执行的V3初始化诊断。所有真实配置/QPS/预算/路由保持原冻结，已测数据未删除或重试。

用户明确选择保持静态实验范围：完整V3未验证，保留初始化诊断，不开展GPU反馈搜索。见user_v3_scope_resolution.json。

Gate：**BENCHMARK_COMPLETED_WITH_FAILURES_AND_UNVERIFIED_V3**。完成后停止。

研究结论：LOW的资源优先部署相对独立S-oriented实测safe memory节省3.23%–16.15%，四例均有三次一致支持，但目标E2E更高。系统级路由结论由全部原始任务请求计算：Text HIGH、Image HIGH（理论非法S-only诊断）、LID MID/HIGH均NO_MEASURABLE_BENEFIT；系统mean/P95三次均退化。ASR HIGH两臂首次unstable并截尾，INSUFFICIENT_EVIDENCE，不用完成子集延迟证明优势。完整原V3按用户选择未验证，现有结果不足以建立独立部署优化算法贡献。

ASR保持Module15两任务VQA0.2、原24模型E/H并集与B1；Text/Image/LID保持Module18六任务低背景、17模型COMMON_G Heavy子池与B2。全部exactQPS/预算/准入/g/M/质量/原generator不变，Image HIGH使用原精度137.2973242711638，不按GPU结果调整。OCR为必要背景。LID仅边界，保留历史HIGHQwen3/3recoverable。

基线A只限制目标任务由G服务，背景合法S保留；基线B只限制目标由S服务，背景VQA等G保留。两者独立按相同safe memory主目标/J次目标选择模型与实测count。V3初始化诊断原搜索/目标/router不改，失败不修复成另一个优化器。Proposed完整最小资源与同成本J证书。部署比较独立选驻留；路由消融固定Proposed全部驻留与非目标路由，仅改目标到S，两者分开。

CPU36项正式A/B/Proposed策略独立求解，另12项仅V3初始化诊断，Proposed完全复现历史。G-only MID/HIGH四任务均理论不可行，不能上GPU制造合法G-only吞吐对照；Specialist-oriented ASR HIGH理论不可行。V3初始化诊断 Image HIGH返回rho0.8028785的S-only路由，违反共同0.8 gate，记returned-plan INFEASIBLE而非整个池不可行；LID HIGH原数值求解失败，保留未修复。V3初始化诊断 Jrouter的稳定域rho<1与外部操作门槛rho≤0.8区别如实保留。LOW四任务V3初始化/G-only/Proposed实际配置等价，无显存/延迟算法优势。

GPU前冻结21个新physical配置、最多63次新run；实际55次新run，历史6次仅作等价配置/单臂描述。每配置3个原始seed对应重复，rep1顺序、rep2逆序、rep3轮换，独立进程/native10s warmup+120s measurement。同配置别名共用观察，未伪造独立胜利；18LOW完全相同六任务配置共用3次，不与15混合。首次system或worker unstable/持续积压/运行/显存失败停止该配置未来重复，其他独立冻结对照继续。

LOW：下表positive saving表示Proposed比S-oriented少驻留的实测safe MiB；延迟cost为Proposed−control，目标任务E2E，每runP95分别配对，不合并请求后重新算。

|任务|valid pairs|safe saving MiB|safe saving %|mean cost ms|P95 cost ms|memory evidence|
|---|---:|---:|---:|---:|---:|---|
|asr|3|1775.8851898008027|11.564298433948846|4052.1047810611103|9775.547679183335|CONSISTENT_POSITIVE_NOMINAL_95CI|
|text|3|1086.415168239273|4.883902805476871|169.73997163888887|621.1986276499998|CONSISTENT_POSITIVE_NOMINAL_95CI|
|image|3|707.2436848456078|3.2335746612984795|334.2598981666667|742.1456995|CONSISTENT_POSITIVE_NOMINAL_95CI|
|lid|3|4075.3528771784077|16.149937279459426|382.59832244444453|769.5528255333335|CONSISTENT_POSITIVE_NOMINAL_95CI|

MID：G-only均不可行，不能给出合法同QPS GPU G-only性能差。Proposed的实测targetthroughput/E2E及B/V3对照见gpu_strategy_comparison.csv；ASR/Image MID所有合法策略实际等价，只复用历史3次作描述，不将其叫独立性能收益。Text MID只与更换OCR背景checkpoint的V3初始化诊断有部署差异；LID MID比较S-oriented及V3初始化诊断，包含真实target/background指标。

HIGH固定驻留路由消融：positive latency effect=S-only−GS，正值GS更快；throughput为GS−S-only。ASR/Image S-only在冻结opgate理论不可行，只作为诊断；即使真实steady也不是合法baseline，不以其throughput重写g表。Text/LID是真实合法路由对照。

|task|control legal|valid pairs|mean improvement ms|P95 improvement ms|within-window throughput gain|mean evidence|P95 evidence|
|---|---|---:|---:|---:|---:|---|---|
|asr|False|0|None|None|None|NO_COMPARABLE_PAIRS|NO_COMPARABLE_PAIRS|
|text|True|3|-3.9077995932870153|-1.6225732000000008|0.0|CONSISTENT_NEGATIVE_NOMINAL_95CI|CONSISTENT_NEGATIVE_NOMINAL_95CI|
|image|False|3|-1.0733838599781425|-0.5308720833333332|0.0|CONSISTENT_NEGATIVE_NOMINAL_95CI|CONSISTENT_NEGATIVE_NOMINAL_95CI|
|lid|True|3|-93.33119927944449|-421.7766306833329|-0.00277777777777762|CONSISTENT_NEGATIVE_NOMINAL_95CI|INCONCLUSIVE_OR_INCOMPLETE|

每一次paired delta、spread、df2 nominal95%tCI见paired_comparison_analysis.json及routing_ablation.csv。要求3次同方向且CI排除0；CI跨0或符号不一致记inconclusive，不把微小Image差异/非零G流量当性能收益。n3多指标未经多重比较校正，只作为development证据，非普遍显著性结论。未完成或unstable时不以completed-only低延迟证明优势。

V3初始化诊断比较：LOW四任务与Proposed完全相同，ASR MID/HIGH、Image MID也相同；Text MID/HIGH与LID MID只有OCR背景checkpoint及可能的运行时竞争差异；Image HIGH原返回路由gate失败，LID HIGH原CPU失败，无合法V3GPU控制可用于声称实测性能胜利。全部独立solve_seconds与V3snapshot/audit时间分开，CPU每policy一次计时，不据单次CPU时间声称统计显著加速。

相对V3初始化诊断具有3次同方向且nominalCI排除0的正例：[]；负例：[]。所有zero、负结果、inconclusive与失败完整保存。目标latency改善不能代替全任务指标，详见per_task_e2e.csv。

LID边界：所有新run的逐worker稳定性、service/s0、handlerbusy、queue episode与tail slope见worker_stability.csv和worker_task_service.csv；systemsteady不代表每个worker steady。保留历史Module18 Qwen3/3recoverable与2.193–2.608×LIDservice/s0，实测busy0.811–0.889，不宣称真实20%余量。基线S-only可能改变这类局部排队，结果按新观察独立记录。未用Module17局部2.06作为全模型折减。

停止的配置及首失败：{"asr_high_proposed": {"run_id": "asr_high_proposed_rep1", "system_stability": "unstable", "worker_unstable": ["phi4mm#0", "whisper-small#0"], "valid_session": false, "memory_passed": true, "errors": [], "reason": "First unstable/sustainedbacklog/runtime/memory failure; no retry, laterrep skipped."}, "asr_high_fixed_s_only": {"run_id": "asr_high_fixed_s_only_rep1", "system_stability": "unstable", "worker_unstable": ["whisper-small#0"], "valid_session": false, "memory_passed": true, "errors": [], "reason": "First unstable/sustainedbacklog/runtime/memory failure; no retry, laterrep skipped."}}。没有重试/移动点。

独立算法贡献判断：当前证据仅能支持所测合法候选池中的资源/延迟取舍、特定路由收益或损失以及求解可行性差异；不能单凭非零GS分流、受控子池相变或V3初始化诊断数值失败建立独立部署优化算法贡献。要声称全面优于V3或通用并发capacity，现有证据不足。此处不扩大实验、不加入模型、不动态部署、不进入Module20。

所有历史8377文件与冻结输入/源码hash核验，所有新workerPID独立且GPU释放。未定义latencySLO不宣称达标。

补充量化（沿用GPU前冻结的配对规则；无新配置、阈值或实验）：

|LOW目标|safe saving MiB / %|GPU peak saving MiB / %|mean cost ms|P95 cost ms|
|---|---:|---:|---:|---:|
|asr|1775.885 / 11.564|1598.667 / 11.564|4052.105|9775.548|
|text|1086.415 / 4.884|978.000 / 4.884|169.740|621.199|
|image|707.244 / 3.234|636.667 / 3.234|334.260|742.146|
|lid|4075.353 / 16.150|3668.667 / 16.150|382.598|769.553|

LID LOW的B还将ASR背景从Qwen转至Medium，这是独立部署优化的真实结果，不能将该差值解释成只改变LID路由。safe memory与采样GPU peak分别报告，未把safe factor乘peak。

MID实测（目标任务；每run mean/P95等权平均，未将三run请求合并）：

|目标|策略|已测重复 / 源|系统steady次数|全worker steady次数|窗口完成QPS|mean ms|P95 ms|
|---|---|---|---:|---:|---:|---:|---:|
|asr|PROPOSED|3 / HISTORICAL_MODULE15|3|3|2.442|210.165|415.577|
|asr|BASELINE_B_S_ORIENTED|3 / HISTORICAL_MODULE15|3|3|2.442|210.165|415.577|
|asr|DIAGNOSTIC_V3_INITIALIZATION|3 / HISTORICAL_MODULE15|3|3|2.442|210.165|415.577|
|text|PROPOSED|3 / NEW_MODULE19|3|3|19.997|16.120|20.912|
|text|BASELINE_B_S_ORIENTED|3 / NEW_MODULE19|3|3|19.997|16.120|20.912|
|text|DIAGNOSTIC_V3_INITIALIZATION|3 / NEW_MODULE19|3|3|19.997|14.524|19.328|
|image|PROPOSED|3 / HISTORICAL_MODULE18|3|3|10.000|15.618|20.699|
|image|BASELINE_B_S_ORIENTED|3 / HISTORICAL_MODULE18|3|3|10.000|15.618|20.699|
|image|DIAGNOSTIC_V3_INITIALIZATION|3 / HISTORICAL_MODULE18|3|3|10.000|15.618|20.699|
|lid|PROPOSED|3 / NEW_MODULE19|3|3|6.000|88.437|127.706|
|lid|BASELINE_B_S_ORIENTED|3 / NEW_MODULE19|3|3|6.000|76.014|121.337|

同一个config_id表示共享等价配置，不是额外独立GPU重复。ASR/Image MID历史三次只作单臂描述，不用于证明当前运行时收益。四任务G-only MID/HIGH均无合法CPU计划，因而没有合法GPU G-only变化百分比；这是可行性边界，不是测得的延迟提升。

V3初始化诊断配对（非完整V3算法比较；正差值=初始化诊断−Proposed；吞吐正差值=Proposed−诊断；单位MiB/ms/QPS）：

|case|comparison status|memory effect|mean effect|P95 effect|throughput effect|
|---|---|---:|---:|---:|---:|
|asr_low|INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3_COMPARISON|0.000|0.000|0.000|0.000|
|asr_mid|INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3_COMPARISON|0.000|0.000|0.000|0.000|
|asr_high|INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3_COMPARISON|—|—|—|—|
|text_low|INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3_COMPARISON|0.000|0.000|0.000|0.000|
|text_mid|INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3_COMPARISON|0.000|-1.596|-1.583|0.000|
|text_high|INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3_COMPARISON|0.000|0.134|0.065|0.000|
|image_low|INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3_COMPARISON|0.000|0.000|0.000|0.000|
|image_mid|INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3_COMPARISON|0.000|0.000|0.000|0.000|
|image_high|INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3_COMPARISON|—|—|—|—|
|lid_low|INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3_COMPARISON|0.000|0.000|0.000|0.000|
|lid_mid|INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3_COMPARISON|—|—|—|—|
|lid_high|INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3_COMPARISON|—|—|—|—|

只有预冻结规则给出3次同方向且nominal CI排除0，才标记一致性支持；上表单个均值差本身不构成显著性。全背景任务指标及任何回退见per_task_e2e.csv。

HIGH实际运行状态（不排除失败配置）：

|case / config|run|CPU合法|系统 / worker|目标窗口QPS|offered / completed|end backlog|
|---|---|---|---|---:|---|---:|
|asr_high / asr_high_proposed|asr_high_proposed_rep1|LEGAL_STRATEGY|unstable / [{"worker": "phi4mm#0", "status": "unstable", "warning_reasons": ["large_service_demand_backlog_peak", "large_but_completed_drain", "large_queue_wait_episode"]}, {"worker": "whisper-small#0", "status": "unstable", "warning_reasons": ["large_submitted_backlog_peak", "large_service_demand_backlog_peak", "large_but_completed_drain", "large_queue_wait_episode"]}]|1.783|526 / 311|101|
|asr_high / asr_high_fixed_s_only|asr_high_fixed_s_only_rep1|NON_BASELINE_DIAGNOSTIC|unstable / [{"worker": "phi4mm#0", "status": "recoverable_congested", "warning_reasons": ["large_but_completed_drain"]}, {"worker": "whisper-small#0", "status": "unstable", "warning_reasons": ["large_submitted_backlog_peak", "large_service_demand_backlog_peak", "large_but_completed_drain", "large_queue_wait_episode"]}]|2.333|526 / 370|80|
|text_high / text_high_proposed|text_high_proposed_rep1|LEGAL_STRATEGY|steady / []|60.000|7200 / 7200|0|
|text_high / text_high_proposed|text_high_proposed_rep2|LEGAL_STRATEGY|steady / []|60.000|7200 / 7200|0|
|text_high / text_high_proposed|text_high_proposed_rep3|LEGAL_STRATEGY|steady / []|59.992|7200 / 7200|1|
|text_high / text_high_s_oriented|text_high_s_oriented_rep1|LEGAL_STRATEGY|steady / []|60.000|7200 / 7200|0|
|text_high / text_high_s_oriented|text_high_s_oriented_rep2|LEGAL_STRATEGY|steady / []|60.000|7200 / 7200|0|
|text_high / text_high_s_oriented|text_high_s_oriented_rep3|LEGAL_STRATEGY|steady / []|59.992|7200 / 7200|1|
|text_high / text_high_original_v3|text_high_original_v3_rep1|V3_INITIALIZATION_DIAGNOSTIC_ONLY|steady / []|60.000|7200 / 7200|0|
|text_high / text_high_original_v3|text_high_original_v3_rep2|V3_INITIALIZATION_DIAGNOSTIC_ONLY|steady / []|60.000|7200 / 7200|0|
|image_high / image_high_proposed|image_high_proposed_rep1|LEGAL_STRATEGY|steady / []|137.292|16476 / 16476|1|
|image_high / image_high_proposed|image_high_proposed_rep2|LEGAL_STRATEGY|steady / []|137.292|16476 / 16476|1|
|image_high / image_high_proposed|image_high_proposed_rep3|LEGAL_STRATEGY|steady / []|137.292|16476 / 16476|1|
|image_high / image_high_s_oriented|image_high_s_oriented_rep1|LEGAL_STRATEGY|steady / []|137.292|16476 / 16476|1|
|image_high / image_high_s_oriented|image_high_s_oriented_rep2|LEGAL_STRATEGY|steady / []|137.292|16476 / 16476|1|
|image_high / image_high_s_oriented|image_high_s_oriented_rep3|LEGAL_STRATEGY|steady / []|137.292|16476 / 16476|1|
|image_high / image_high_fixed_s_only|image_high_fixed_s_only_rep1|NON_BASELINE_DIAGNOSTIC|steady / []|137.292|16476 / 16476|1|
|image_high / image_high_fixed_s_only|image_high_fixed_s_only_rep2|NON_BASELINE_DIAGNOSTIC|steady / []|137.292|16476 / 16476|1|
|image_high / image_high_fixed_s_only|image_high_fixed_s_only_rep3|NON_BASELINE_DIAGNOSTIC|steady / []|137.292|16476 / 16476|1|
|lid_high / lid_high_proposed|lid_high_proposed_rep1|LEGAL_STRATEGY|steady / [{"worker": "qwen#0", "status": "recoverable_congested", "warning_reasons": ["large_queue_wait_episode"]}]|9.992|1200 / 1200|1|
|lid_high / lid_high_proposed|lid_high_proposed_rep2|LEGAL_STRATEGY|steady / [{"worker": "qwen#0", "status": "recoverable_congested", "warning_reasons": ["large_queue_wait_episode"]}]|9.992|1200 / 1200|1|
|lid_high / lid_high_proposed|lid_high_proposed_rep3|LEGAL_STRATEGY|steady / [{"worker": "qwen#0", "status": "recoverable_congested", "warning_reasons": ["large_queue_wait_episode"]}]|9.992|1200 / 1200|1|
|lid_high / lid_high_s_oriented|lid_high_s_oriented_rep1|LEGAL_STRATEGY|steady / []|9.992|1200 / 1200|1|
|lid_high / lid_high_s_oriented|lid_high_s_oriented_rep2|LEGAL_STRATEGY|steady / []|10.000|1200 / 1200|0|
|lid_high / lid_high_s_oriented|lid_high_s_oriented_rep3|LEGAL_STRATEGY|steady / []|9.992|1200 / 1200|1|

未提交/accepted host-send-start/完成/窗口吞吐/drain计时见request_accounting.csv；worker真实完成路由见actual_task_worker_routing.csv。host-send-start不是worker ACK，日志不足时不声称精确worker acceptance。未完成请求的执行忙碌比例只具下界意义，GPU utilization另由NVML采样报告。失败run只报告被截尾的完成请求延迟，不能用其较小均值证明性能胜利。

历史运行时变化见historical_runtime_drift.csv：必须结合same_scheduled_samples及same_task_family_assignment判断，不能把样本差异当纯服务时间变化。ASR LOW新/历史同种子服务时间约3.60×、4.12×、1.30×，没有拟合固定惩罚，也未由失败吞吐推出精确容量。当前benchmark不能证明跨运行状态可迁移的容量模型。

Text/Image/LID结论的候选池边界是Module18预定义COMMON_G17，未开展完整3-Generalist池新GPU对照；ASR为Module15原24模型两任务池。没有把两种池/预算/背景直接混合排名，没有增加模型，原V3与Module9–18未修改。

系统级固定驻留路由补充（按用户追加要求，只重新分析已有日志）：

所有完成请求按实际数量合并得到系统mean和P95；P95来自原始请求分布，绝不平均任务P95。第二种mean按冻结offered QPS加权每任务完成队列mean；不按完成比例重新归一化。完整请求流两者接近/相等，但失败截尾时均须结合未完成数，不能据低延迟宣称收益。ASR原实例是ASR/VQA两任务，另外三例为六任务，未虚构ASR背景。

LID MID的Proposed与独立求解B恰好得到同一驻留且所有背景路由完全一致，仅目标G/S对比S-only不同，因此也纳入这项系统统计；原实验类型仍记部署策略比较，不伪称为另一个预先强制固定驻留实验。重复物理控制只统计一次。

全系统P95按实际请求数定义，会由高QPS任务主导，不能代表每类背景任务都满足某个tail约束；各任务P95和拥塞必须同时检查。六任务背景中的低QPS任务每run通常只有12或24条请求，其P95/quality估计粒度有限。这里不额外赋予任务权重、不修改原J，也不声称未定义的latency SLO通过。

|case|valid pairs|结论|system mean effect ms|system P95 effect ms|QPS-weighted mean effect ms|throughput effect QPS|
|---|---:|---|---:|---:|---:|---:|
|asr_high|0|INSUFFICIENT_EVIDENCE（理论非法S-only诊断）|—|—|—|—|
|text_high|3|NO_MEASURABLE_BENEFIT|-3.805933|-1.884067|-3.805933|0.000000|
|image_high|3|NO_MEASURABLE_BENEFIT（理论非法S-only诊断）|-1.093457|-0.505018|-1.093457|0.000000|
|lid_high|3|NO_MEASURABLE_BENEFIT|-96.068001|-552.285571|-96.068001|-0.002778|
|lid_mid|3|NO_MEASURABLE_BENEFIT|-11.206034|-121.442153|-11.206034|0.000000|

效果正值表示G/S更好；延迟为S-only−GS，吞吐为GS−S-only。下面完整保留每次运行、每任务代价、预测差异与worker风险，系统收益不要求每任务改善。若多个系统指标方向冲突，不能声称全面提升。n3 CI未经多重比较修正且此系统级扩展为追加分析；NO_MEASURABLE_BENEFIT表示未检出收益，并非证明效果恰为零。

asr_high：分类 ['INSUFFICIENT_EVIDENCE']；S-only理论合法=False。

|rep / arm|offered / completed|system mean / P95 ms|offered-QPS mean ms|system window QPS|CPU J ms|safe MiB|system / worker warnings|
|---|---|---:|---:|---:|---:|---:|---|
|1 / GS|550 / 333|50742.373497 / 100113.549020|51018.037646|1.900000|877.365553|15338.671414|unstable / [{"worker": "phi4mm#0", "status": "unstable", "warning_reasons": ["large_service_demand_backlog_peak", "large_but_completed_drain", "large_queue_wait_episode"]}, {"worker": "whisper-small#0", "status": "unstable", "warning_reasons": ["large_submitted_backlog_peak", "large_service_demand_backlog_peak", "large_but_completed_drain", "large_queue_wait_episode"]}]|
|1 / S_ONLY|550 / 394|27040.734151 / 87181.194119|27532.366267|2.533333|1091.783704|15314.232627|unstable / [{"worker": "phi4mm#0", "status": "recoverable_congested", "warning_reasons": ["large_but_completed_drain"]}, {"worker": "whisper-small#0", "status": "unstable", "warning_reasons": ["large_submitted_backlog_peak", "large_service_demand_backlog_peak", "large_but_completed_drain", "large_queue_wait_episode"]}]|

|task|3-pair mean effect ms [nominal CI95]|P95 effect ms [nominal CI95]|方向证据|
|---|---|---|---|
|asr|— None|— None|NO_COMPARABLE_PAIRS / NO_COMPARABLE_PAIRS|
|visual_qa|— None|— None|NO_COMPARABLE_PAIRS / NO_COMPARABLE_PAIRS|

|rep|CPU J预测mean improvement ms|实测system mean improvement ms|实测system P95 improvement ms|实测QPS-weighted mean improvement ms|预测/实测mean方向相符|valid pair|
|---|---:|---:|---:|---:|---|---|
|1|214.418151|-23701.639346|-12932.354901|-23485.671379|None|False|

三重复有效证据flag：{'TARGET_TASK_BENEFIT': False, 'SYSTEM_LEVEL_BENEFIT': False, 'CAPACITY_OR_STABILITY_BENEFIT': False, 'NO_MEASURABLE_BENEFIT': False, 'INSUFFICIENT_EVIDENCE': True}；CPU J与实测mean方向相符次数=0/0，全任务样本/非目标路由/实际驻留同一性见routing_system_pair_differences.csv。

持续积压/截尾/质量fault runs：['asr_high_proposed_rep1', 'asr_high_fixed_s_only_rep1']；一致性延迟退化任务：[]。严重拥塞由原worker/system分类与backlog记录识别；没有新增或倒推延迟SLO。

text_high：分类 ['NO_MEASURABLE_BENEFIT']；S-only理论合法=True。

|rep / arm|offered / completed|system mean / P95 ms|offered-QPS mean ms|system window QPS|CPU J ms|safe MiB|system / worker warnings|
|---|---|---:|---:|---:|---:|---:|---|
|1 / GS|7272 / 7272|15.072582 / 15.798851|15.072582|60.600000|31.031319|22343.716456|steady / []|
|1 / S_ONLY|7272 / 7272|11.436536 / 14.092761|11.436536|60.600000|31.437689|22343.716456|steady / []|
|2 / GS|7272 / 7272|14.450406 / 16.425986|14.450406|60.600000|31.031319|22117.102249|steady / []|
|2 / S_ONLY|7272 / 7272|11.428130 / 14.322033|11.428130|60.600000|31.437689|22117.102249|steady / []|
|3 / GS|7272 / 7272|15.907242 / 16.135316|15.907242|60.591667|31.031319|22272.621803|steady / []|
|3 / S_ONLY|7272 / 7272|11.147765 / 14.293156|11.147765|60.591667|31.437689|22274.843511|steady / []|

|task|3-pair mean effect ms [nominal CI95]|P95 effect ms [nominal CI95]|方向证据|
|---|---|---|---|
|text|-3.907800 [-5.765611425784005, -2.049987760790068]|-1.622573 [-1.7638132335041448, -1.4813331664958453]|CONSISTENT_NEGATIVE_NOMINAL_95CI / CONSISTENT_NEGATIVE_NOMINAL_95CI|
|image|0.985003 [-0.2591152218732641, 2.2291217774288303]|11.355493 [-15.47752664605485, 38.18851267938818]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|
|ocr|-0.697413 [-1.820419129031205, 0.42559351792009326]|-0.655727 [-7.737637299651743, 6.426183532985078]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|
|asr|30.511336 [-72.8453753210292, 133.86804765436247]|23.332623 [-117.3366019242029, 164.00184745753603]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|
|lid|17.275071 [-48.87701463945382, 83.42715625056496]|9.629990 [-49.76335380060557, 69.02333430060555]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|
|visual_qa|-4.894760 [-30.277549968047865, 20.488030079158953]|-10.952302 [-48.11162904454868, 26.207025677882058]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|

|rep|CPU J预测mean improvement ms|实测system mean improvement ms|实测system P95 improvement ms|实测QPS-weighted mean improvement ms|预测/实测mean方向相符|valid pair|
|---|---:|---:|---:|---:|---|---|
|1|0.406370|-3.636046|-1.706090|-3.636046|False|True|
|2|0.406370|-3.022275|-2.103953|-3.022275|False|True|
|3|0.406370|-4.759477|-1.842160|-4.759477|False|True|

三重复有效证据flag：{'TARGET_TASK_BENEFIT': False, 'SYSTEM_LEVEL_BENEFIT': False, 'CAPACITY_OR_STABILITY_BENEFIT': False, 'NO_MEASURABLE_BENEFIT': True, 'INSUFFICIENT_EVIDENCE': False}；CPU J与实测mean方向相符次数=0/3，全任务样本/非目标路由/实际驻留同一性见routing_system_pair_differences.csv。

持续积压/截尾/质量fault runs：[]；一致性延迟退化任务：['text']。严重拥塞由原worker/system分类与backlog记录识别；没有新增或倒推延迟SLO。

image_high：分类 ['NO_MEASURABLE_BENEFIT']；S-only理论合法=False。

|rep / arm|offered / completed|system mean / P95 ms|offered-QPS mean ms|system window QPS|CPU J ms|safe MiB|system / worker warnings|
|---|---|---:|---:|---:|---:|---:|---|
|1 / GS|16548 / 16548|7.634690 / 6.920836|7.634718|137.891667|30.622592|22023.790517|steady / []|
|1 / S_ONLY|16548 / 16548|6.518881 / 6.408149|6.518907|137.891667|30.465814|22023.790517|steady / []|
|2 / GS|16548 / 16548|7.233002 / 6.997123|7.233025|137.891667|30.622592|21708.307994|steady / []|
|2 / S_ONLY|16548 / 16548|6.349617 / 6.510647|6.349640|137.891667|30.465814|21708.307994|steady / []|
|3 / GS|16548 / 16548|7.985325 / 6.936251|7.985355|137.891667|30.622592|21866.049255|steady / []|
|3 / S_ONLY|16548 / 16548|6.704150 / 6.420361|6.704180|137.891667|30.465814|21866.049255|steady / []|

|task|3-pair mean effect ms [nominal CI95]|P95 effect ms [nominal CI95]|方向证据|
|---|---|---|---|
|text|-14.576418 [-55.67974961438646, 26.526913725497586]|-40.055720 [-261.2781889614647, 181.1667490614648]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|
|image|-1.073384 [-1.5333457487235964, -0.6134219712327036]|-0.530872 [-0.5934617681955431, -0.4682823984711223]|CONSISTENT_NEGATIVE_NOMINAL_95CI / CONSISTENT_NEGATIVE_NOMINAL_95CI|
|ocr|0.217672 [-0.821021210816755, 1.2563657663723098]|-1.683340 [-5.438927842522175, 2.0722471758555097]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|
|asr|13.978640 [-86.38104859227633, 114.33832903672078]|14.600992 [-99.02944559586594, 128.2314303958659]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|
|lid|0.903573 [-108.75237452869595, 110.55952063980709]|-24.663358 [-160.99637995570458, 111.66966332237132]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|
|visual_qa|-17.322031 [-61.836397549536855, 27.192335466203513]|-20.187485 [-76.64423495625645, 36.26926558958999]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|

|rep|CPU J预测mean improvement ms|实测system mean improvement ms|实测system P95 improvement ms|实测QPS-weighted mean improvement ms|预测/实测mean方向相符|valid pair|
|---|---:|---:|---:|---:|---|---|
|1|-0.156778|-1.115809|-0.512687|-1.115811|True|True|
|2|-0.156778|-0.883385|-0.486476|-0.883384|True|True|
|3|-0.156778|-1.281175|-0.515890|-1.281175|True|True|

三重复有效证据flag：{'TARGET_TASK_BENEFIT': False, 'SYSTEM_LEVEL_BENEFIT': False, 'CAPACITY_OR_STABILITY_BENEFIT': False, 'NO_MEASURABLE_BENEFIT': True, 'INSUFFICIENT_EVIDENCE': False}；CPU J与实测mean方向相符次数=3/3，全任务样本/非目标路由/实际驻留同一性见routing_system_pair_differences.csv。

持续积压/截尾/质量fault runs：[]；一致性延迟退化任务：['image']。严重拥塞由原worker/system分类与backlog记录识别；没有新增或倒推延迟SLO。

lid_high：分类 ['NO_MEASURABLE_BENEFIT']；S-only理论合法=True。

|rep / arm|offered / completed|system mean / P95 ms|offered-QPS mean ms|system window QPS|CPU J ms|safe MiB|system / worker warnings|
|---|---|---:|---:|---:|---:|---:|---|
|1 / GS|1272 / 1272|197.654188 / 667.613617|197.654188|10.591667|178.417875|25376.347754|steady / [{"worker": "qwen#0", "status": "recoverable_congested", "warning_reasons": ["large_queue_wait_episode"]}]|
|1 / S_ONLY|1272 / 1272|119.252901 / 317.457274|119.252901|10.591667|225.373598|25376.347754|steady / []|
|2 / GS|1272 / 1272|221.312635 / 980.585829|221.312635|10.591667|178.417875|25247.488695|steady / [{"worker": "qwen#0", "status": "recoverable_congested", "warning_reasons": ["large_queue_wait_episode"]}]|
|2 / S_ONLY|1272 / 1272|109.661617 / 285.533696|109.661617|10.600000|225.373598|25229.715032|steady / []|
|3 / GS|1272 / 1272|235.902615 / 1012.253651|235.902615|10.591667|178.417875|25694.051985|steady / [{"worker": "qwen#0", "status": "recoverable_congested", "warning_reasons": ["large_queue_wait_episode"]}]|
|3 / S_ONLY|1272 / 1272|137.750917 / 400.605415|137.750917|10.591667|225.373598|25694.051985|steady / []|

|task|3-pair mean effect ms [nominal CI95]|P95 effect ms [nominal CI95]|方向证据|
|---|---|---|---|
|text|-283.161102 [-564.4763911128227, -1.8458126093993883]|-467.693582 [-1185.9983181884418, 250.61115352177524]|CONSISTENT_NEGATIVE_NOMINAL_95CI / INCONCLUSIVE_OR_INCOMPLETE|
|image|-218.023261 [-350.89298479923906, -85.15353681187196]|-688.717034 [-1604.8636632991288, 227.429594299129]|CONSISTENT_NEGATIVE_NOMINAL_95CI / INCONCLUSIVE_OR_INCOMPLETE|
|ocr|0.048614 [-12.146446957247583, 12.243675290580914]|-4.807178 [-8.542311277580604, -1.0720453890860564]|INCONCLUSIVE_OR_INCOMPLETE / CONSISTENT_NEGATIVE_NOMINAL_95CI|
|asr|30.412329 [-145.57585338909024, 206.40051205575696]|-23.782720 [-412.772468765794, 365.2070279657936]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|
|lid|-93.331199 [-130.20559383049715, -56.45680472839176]|-421.776631 [-888.4946137831594, 44.94135241649343]|CONSISTENT_NEGATIVE_NOMINAL_95CI / INCONCLUSIVE_OR_INCOMPLETE|
|visual_qa|-189.682371 [-622.928568068691, 243.56382562424662]|-478.947095 [-1737.7867772263944, 779.8925880930601]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|

|rep|CPU J预测mean improvement ms|实测system mean improvement ms|实测system P95 improvement ms|实测QPS-weighted mean improvement ms|预测/实测mean方向相符|valid pair|
|---|---:|---:|---:|---:|---|---|
|1|46.955723|-78.401287|-350.156343|-78.401287|False|True|
|2|46.955723|-111.651018|-695.052133|-111.651018|False|True|
|3|46.955723|-98.151698|-611.648236|-98.151698|False|True|

三重复有效证据flag：{'TARGET_TASK_BENEFIT': False, 'SYSTEM_LEVEL_BENEFIT': False, 'CAPACITY_OR_STABILITY_BENEFIT': False, 'NO_MEASURABLE_BENEFIT': True, 'INSUFFICIENT_EVIDENCE': False}；CPU J与实测mean方向相符次数=0/3，全任务样本/非目标路由/实际驻留同一性见routing_system_pair_differences.csv。

持续积压/截尾/质量fault runs：[]；一致性延迟退化任务：['text', 'image', 'ocr', 'lid']。严重拥塞由原worker/system分类与backlog记录识别；没有新增或倒推延迟SLO。

lid_mid：分类 ['NO_MEASURABLE_BENEFIT']；S-only理论合法=True。

|rep / arm|offered / completed|system mean / P95 ms|offered-QPS mean ms|system window QPS|CPU J ms|safe MiB|system / worker warnings|
|---|---|---:|---:|---:|---:|---:|---|
|1 / GS|792 / 792|112.647886 / 308.710402|112.647886|6.600000|126.867061|25376.347754|steady / []|
|1 / S_ONLY|792 / 792|103.905792 / 196.366038|103.905792|6.600000|127.337842|25376.347754|steady / []|
|2 / GS|792 / 792|108.118116 / 318.082702|108.118116|6.600000|126.867061|25247.488695|steady / []|
|2 / S_ONLY|792 / 792|95.408770 / 151.963984|95.408770|6.600000|127.337842|25229.715032|steady / []|
|3 / GS|792 / 792|119.589872 / 293.425447|119.589872|6.600000|126.867061|25694.051985|steady / []|
|3 / S_ONLY|792 / 792|107.423211 / 207.562070|107.423211|6.600000|127.337842|25694.051985|steady / []|

|task|3-pair mean effect ms [nominal CI95]|P95 effect ms [nominal CI95]|方向证据|
|---|---|---|---|
|text|-12.153296 [-34.501022243390835, 10.194429687835267]|-33.921088 [-255.34549372308356, 187.50331702308375]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|
|image|-17.433441 [-55.731147312931654, 20.86426547959832]|-95.387968 [-241.64504557335778, 50.86910954002478]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|
|ocr|1.141591 [-2.0953117106407015, 4.378493432862925]|1.810510 [-2.7487353706698716, 6.369755137336536]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|
|asr|41.621982 [-2.5133331049364287, 85.75729732715874]|70.728401 [-69.96611359583935, 211.42291512917276]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|
|lid|-12.423394 [-15.665979218088392, -9.180809699504199]|-6.368679 [-15.052214253240738, 2.314855419907368]|CONSISTENT_NEGATIVE_NOMINAL_95CI / INCONCLUSIVE_OR_INCOMPLETE|
|visual_qa|-3.685703 [-65.04569170133566, 57.674285145780125]|-56.049689 [-163.47255163087505, 51.373173230875345]|INCONCLUSIVE_OR_INCOMPLETE / INCONCLUSIVE_OR_INCOMPLETE|

|rep|CPU J预测mean improvement ms|实测system mean improvement ms|实测system P95 improvement ms|实测QPS-weighted mean improvement ms|预测/实测mean方向相符|valid pair|
|---|---:|---:|---:|---:|---|---|
|1|0.470781|-8.742094|-112.344364|-8.742094|False|True|
|2|0.470781|-12.709346|-166.118718|-12.709346|False|True|
|3|0.470781|-12.166662|-85.863377|-12.166662|False|True|

三重复有效证据flag：{'TARGET_TASK_BENEFIT': False, 'SYSTEM_LEVEL_BENEFIT': False, 'CAPACITY_OR_STABILITY_BENEFIT': False, 'NO_MEASURABLE_BENEFIT': True, 'INSUFFICIENT_EVIDENCE': False}；CPU J与实测mean方向相符次数=0/3，全任务样本/非目标路由/实际驻留同一性见routing_system_pair_differences.csv。

持续积压/截尾/质量fault runs：[]；一致性延迟退化任务：['lid']。严重拥塞由原worker/system分类与backlog记录识别；没有新增或倒推延迟SLO。

quality/capability admission按冻结目录逐请求path核验；各任务correct数、accuracy、ASR corpus WER及无效closed-set输出见routing_system_task_metrics.csv。背景每task仅约12或24请求，不能将小样本accuracy波动直接认定为原始质量SLO失效；也不能以准入合格替代报告实测错误。逐worker队列、end/tail/drain backlog和service/queue严格分开见routing_system_worker_backlog.csv。

CPU J是原始全系统预测，不是目标任务J或真实延迟上界；逐rep报告预测/实测方向是否相符，方向错误完整保留，没有根据结果修路由或加干扰系数。系统级结论优先于仅凭目标任务延迟的整体判断，目标代价和其他任务收益/损失均独立可见。
