# 新预测器画像计划与训练转换

`tools/predictor_validation_20261027/profiling.py` 只在 CPU 上准备计划、审计已有原生观测、转换训练数据，不启动 CUDA 或 worker。

```bash
python -m tools.predictor_validation_20261027.profiling plan --input PLAN_INPUT.json --output NEW_PLAN.json
python -m tools.predictor_validation_20261027.profiling audit --input OBSERVED_TRACE.json --output NEW_AUDIT.json
python -m tools.predictor_validation_20261027.profiling convert --input TRAINING_TRACE_LIST.json --output NEW_FIT_INPUT.json
```

计划输入为 `{inputs:[{task,pool_id,input_size},...],execution_context_sha256:...}`，text/OCR 各至少四个输入。size 是公开输入长度，不是答案、标签或实际生成长度。固定驻留 `text#0,text#1,generalist_2#0,ocr#0`；分别覆盖 S0/S1/G/O/SS/SG/SSG/SSGO。多路采用 0/50/200 ms 间隔与正反启动顺序，各条件默认两次训练、一次独立验证；固定长度排序后每四个输入留一个为完全未见输入，始终不用于训练。训练数据不能混入验证请求。

当前 ProfileProvider 按精确 pool_id 查表，因此验证需要分别报告：已见输入的独立重复实验预测误差、未见输入的 coverage/UNKNOWN。没有输入特征泛化模型时，不能声称已验证跨输入泛化。此计划用于微观画像，不能代替全过程排队、P95、ΔU 的独立运行验证；完整计划也可能很大，应在原生执行前显式冻结选定的先导子集。

原生 trace schema 完整例见 `tests/test_predictor_profiling.py::trace`；测试数值都是 CPU 构造。执行证据必须是 `kind=measured,backend=native`，使用同一 `host_monotonic_ns` 时钟，并给出不可变源文件 SHA receipt。`observation_window_ns={start,end}` 包含前后空闲阶段；功率和显存样本均须覆盖这个完整窗口。

每个请求须包含真实 worker START/END，及按原始观测整理的 host/gpu/sync/cleanup 非重叠连续完整分段。GPU 段需明确 `observation_level=phase|kernel`、观察方法和资源计数器归一化证据。并行 kernel 必须由 exporter 整理成不重复计时的时间段；这里不把 kernel 耗时相加，也不将 submit 或 handler envelope 变成 GPU 段。缺失段、误用提交作为 START 或不完整时间覆盖返回 UNKNOWN。

硬件计数器无权限（如 `ERR_NVGPUCTRPERM`）时，GPU `demand` 可逐项为 null，并显式提供 `demand_evidence="UNKNOWN:ERR_NVGPUCTRPERM"` 或 `{status:"unknown",reason:...}`。真实阶段重叠仍可用于经验进度拟合，但计算/访存互补分类保留 UNKNOWN；不能用 0 填补未知需求。非 GPU 段 demand 必须全为 0。阶段同步本身可能改变执行，需要在独立验证中检查 instrumentation 开销，不能把 profile 行为效果移植给不同执行后端。

`power_samples=[{time_ns,watts}]` 是整板功率，通过所有内部采样节点的分段线性积分计算 context 平均功率，不解释成单 kernel 的能耗。`memory_samples=[{time_ns,owned_used_mib,owned_aliases,foreign_pids}]` 覆盖全部四个驻留 alias（包括空闲者），foreign_pids 必须为空；输出使用 raw MiB，不预乘安全系数。任一 context 区间内部没有显存样本，那个 context 的显存保留 null；采样峰值不是连续真实峰值。

转换仅接受同一执行上下文的 training traces。每个 profile/segment 至少两次实际 singleton 活动作为 solo baseline，单路活动仍须保持全部模型驻留。完整重叠上下文只为实际见到的组合建表，不枚举 kernel 笛卡尔积。输出 `{document,observations,conversion}` 可交给 `fit_progress(document,observations)`；未通过满秩或重构误差检查的组合继续 UNKNOWN。profile id 必须将输入、模型、任务、执行配置和稳定段序列绑定，不能在不同观测中改变含义。

务实起步应采用少量真实阶段，先固定输入子集和完整驻留集，改变启动偏移以获得可辨识性；每个 target 的上下文列数可能随其他任务阶段和输入混合迅速增长，不能用少数 SSG 实验外推完整多路组合。训练重构误差不是独立预测精度或可信界。

## CPU 同域搜索核验

`search_check.py` 直接用 `FullHorizonModel + score_forecast` 对相同有限候选域分别执行完整 DFS 参照与有界搜索；不读取旧结果表。域为固定部署/执行配置下，当前一个尚未启动的 text 请求的目标 alias × 后续静态 text 分流率（0、1/8、1/4、1/2、3/4、7/8、1）。其他任务保留输入声明的固定路由，所有候选使用相同未来情景。这里只核验这个有限域，不能声称覆盖部署、副本、资源配置、联合batch、动态未来策略。

上界来自六项非负代价：Q 可由固定候选路由和共同输入直接算出，M 不小于安全系数乘已有历史峰值除以预算，其余项以 0 为下界。仅严格小于当前最优时剪枝，保留所有最优并列。UNKNOWN 不算不可行，也不认证整个候选域。CPU 规划仍检查 family 平均 rho≤0.8、完整 drain 和显存预算，但不生成原生稳定性证据。

```bash
python -m tools.predictor_validation_20261027.search_check \
  --template artifacts/predictor-cpu-20261027/analytical-request.json \
  --objective configs/resource_spillover_objective_20261024.json \
  --output NEW_CPU_SEARCH_DIRECTORY
```

已有可复查产物 `artifacts/predictor-gpu-20261027/cpu-search/revalidated-v2/`。使用明确 `cpu_fixture` 的原分析请求，14候选×3情景，两种算法各执行42次新模拟，最优 U=-0.3620833333333333、全部4个并列最优一致。弱上界未剪枝，因此不声称搜索效率提升。该 fixture 为2请求、两alias的小域功能核验，不是完整四任务画像下的收益/稳定性/全域最优结论。上一级初次产物早于 UNKNOWN认证逻辑修复；以 revalidated-v2 中带最终源SHA的报告为准。

## 原生采集的直接映射与拟合诊断

`map_native_collection(raw_path,export_paths=())` 接收 `native_collect.py` 的完整 `raw.json` 与可选的 `trace_export.py` Nsight导出。每批保留独立 trace，验证请求不进入 training；补全的 worker_control host 段只对应明确 worker START/PHASE/RESULT 边界之间的实际控制代码，不通过 kernel 时长相减推断。mixed native inference 阶段可能包含主机工作、传输、CUDA启动和等待，其 `kind=gpu` 表示已观察的加速器调用阶段，不表示全程GPU忙碌。

`fit_native_collection(...)` 在原始满秩与15%工作重构误差要求下运行 `convert_training + fit_progress`，输出转换证据和每组诊断。独立验证仍需在冻结训练模型后另行采集；本次采集中交错的 validation 只作为诊断。没有GPU demand计数器时分类UNKNOWN；无kernel export时不声称已经解释实际kernel重叠。

```bash
# INPUT.json: {"raw_path":"/absolute/path/raw.json","export_paths":[]}
python -m tools.predictor_validation_20261027.profiling map-native --input INPUT.json --output NEW_TRACES.json
python -m tools.predictor_validation_20261027.profiling fit-native --input INPUT.json --output NEW_FIT_DIAGNOSTICS.json
```

可选kernel报告将发起kernel的phase与kernel实际起止分别保留。时钟映射有效时，单列越过所属phase结束的kernel数量和最长尾部；sync的资源demand=0仅表示host等待类段不参与资源需求分类，不代表等待期间GPU空闲，也不将异步尾部自动归为host。kernel union busy 与kernel duration sum均保留，不重复计时。无有效映射则尾部数据UNKNOWN。

短暂worker_control段可能没有实际NVML样本，M应保持未知。多phase多输入共驻的上下文组合也可能导致拟合欠定；这时不能把训练工作重构或局部服务误差升级成完整六项反事实效用保证。

## 真实训练画像初次审计结果

训练 `native-training-v3` 保持完整四worker驻留，144批、354请求完成，无采集错误。CPU拟合结果见 `artifacts/predictor-gpu-20261027/profile-audits/native-training-v3-comparison.json` 及两份原始拟合报告。

正式 fine v1：实际见到549个上下文，只7个可完整支持，542个因未辨识或工作重构不通过而丢弃，7个均缺段内显存采样。12个GPU目标组全部欠秩；G2每输入32完整观测却有72/93/87上下文列，rank31/31/27。不能因输出局部 `known=true` 就宣称完整预测通过验收。

预声明开发备选 `macro_profiles.py` 将START到post_sync结束作为macro0，之后到END作为macro1；PP OCR整handler为macro0。它保留输入特定solo基线，但显式假设同family alias可交换、输入solo归一化后共享完整family×macro多重集上下文进度。实际24上下文、5个目标组全部满秩，却全部超过固定15%训练工作重构阈值：G2 macro0/1分别75.1%/62.0%，OCR34.8%，text macro0/1分别56.1%/144.1%。不提高阈值、不core启用、不用验证集调参。每alias/pool solo handler两次相对range大多低于6.2%，最大8.7%，不能简单用普通solo抖动解释这些大残差。

现有证据指出两个具体瓶颈：fine精确输入/段组合维度过大、设计矩阵欠定；macro虽然降维，但阶段内部进度、输入差异和异步尾部等状态被压缩过度，恒定共享进度不足以重构训练工作。当前应判预测器验收未通过，不能进入最终收益确认或宣称正分流收益。下一轮模型开发必须另作预声明与冻结，再用独立采集验证。

后续分析模块为 `profiling_analysis.py`，保留完整plan/split绑定审计；原 `profiling.py` 已恢复为训练v3快照相同字节，SHA `42223cf357f57fa5dd47fb70d9e3c5d88a149245beb8e792d0742c9bac870f0f`，以允许相同执行身份的独立验证。二者差异仅分析映射/审计，不改变采集执行代码。新的 macro/handler开发工具使用分析模块，未修改冻结执行源。

## 输入独立 whole-handler 开发诊断

`handler_profiles.py` 为另一个显式训练开发模型：目标 family/task/pool 独立solo基线和系数；上下文仅共享完整active family多重集，保留同族副本数量，不跨input共享系数，也不从pairwise外推multiway。START到END始终称 mixed handler，不称kernel阶段。仍要求NNLS满秩、所有速度严格为正、最大工作重构误差≤15%。额外报告矩阵条件数、奇异值、per-alias残差与solo偏移，以及逐handler留一重拟合的系数敏感性，后者不是置信界。

报告 `profile-audits/native-training-v3-handler-fit-v2.json` 在11个family上下文、9个target组中仅1组接受。G2三输入最大工作误差17.7%、54.1%、32.6%均拒；OCR一组欠秩、一组9.1%误差但非正速度而拒、一组24.3%拒；text两组53.5%、59.0%拒。唯一接受的text长输入最大工作误差13.8%，但矩阵条件数约26795，留一系数最大变动61.8%，预测稳健性仍无保证。这些均仅训练数据诊断，没有读取独立validation-v3或调节验收阈值；不core启用。

这进一步表明，即使不跨输入共享且将上下文减少至5–8列，整个handler恒定进度仍不足。下一步若继续开发，应引入能在预测时得到的稳定执行阶段/工作进度，不能以实际未来kernel信息作在线特征，也不能在同一验证数据上不断改模型后继续称独立验证。

## Semantic阶段与可辨识的局部证据

`semantic_profiles.py` 将真实handler映射为 prepare、prefill、decode、finalize；OCR保留mixed whole handler。prefill为首次forward开始至第二次forward开始，decode延续至真实post_sync结束，避免把forward返回后的异步GPU尾部丢掉。单次CUPTI对应实验中G2的forward返回后仍有51.6ms尾部；这只证明该次尾部存在，不是通用尾部界。所有这些阶段仍包含CPU、传输、启动和等待，硬件需求计数器仍UNKNOWN。

`semantic-training-v5` 的144批354请求完成后，76个实测上下文、27个target组，没有组同时通过原满秩、正速度及≤15%工作重构误差条件。`partial_profiles.py` 保留可独立支持的solo证据：whole-handler 9/9、semantic 23/27、fine 45/75；它不把solo支持升级成multiway共驻支持。训练进度模型未通过的结果仍保留，不放宽阈值。

## 资源画像的独立验证与全任务扩展

训练v5的资源模型 `statistical-resources/training-v5-resources.json` 使用直接完整family-count上下文的整板条件功率及独立batch拥有进程显存采样峰值。mean/max均为经验情景，max不是容量或置信上界；MiB保持raw，调用者只乘一次原safe_factor。功率是条件观测，不是因果归因。

前瞻封存的 `statistical-resources/validation-v6-protocol.json` 之后采集独立v6，144批354请求。单次正式评估报告 `statistical-resources/validation-v6-report.json`：mean整窗能量误差8.22%、raw拥有显存峰值误差9.34%，通过事先声明的两个10%描述阈值；但mean显存低估3149MiB。max能量误差10.70%未过阈值，显存仍低估2060MiB。这没有建立安全上界、阶段进度精度、队列/P95或ΔU验收结论。

`background-training-v7` 为全四任务训练，264批690请求，输入覆盖text、image、ASR、OCR。`background_resources.py` 的一次训练产物为 `background-statistical/training-v7-resources.json`，SHA `9055141785fcb330a07ff1abc7a797b92212d626371f35d96596c6900e351ac6`。共12个family-count上下文、24个完整family/task多重集上下文，覆盖G2的idle/text/image/ASR × text活跃副本数0/1/2 × OCR活跃0/1的完整24组合。15412个全部拥有、无foreign样本，最大采样间隔70.70ms、最大查询62.74ms，raw idle/run拥有显存采样峰值分别29960/31634MiB。

后续使用 `TaskConditionalResourceProvider(document,execution_context_sha256)`：直接读取同一完整family/task多重集的time-weighted mean_watts，以及mean/max_trial_sampled_peak_mib与run idle floor，不使用family pooled均值。未见任务组合为UNKNOWN；task条件power max未保存，始终UNKNOWN，不借用family max。它的mean功率、mean/max显存在24个已见上下文均可读取，但仍须独立验证，未证明跨输入分布或执行方式可迁移。

## 全到达流六项算术

`flow_metrics.score_flow(raw,plan)` 从完整父进程记录计算诊断六项：T用deadline前父进程收到的完成数；L/P用从公开scheduled到父进程completion的E2E，包含有界drain内完成；Q先按任务汇总错误数/参考单位再按到达率加权。OCR/ASR错误率允许超过1，不把逐请求CER/WER直接平均。

M使用包含加载、prime、warmup及drain的全部拥有进程raw采样峰值，原safe_factor只应用一次。缺失/NaN/负/bool显存使M UNKNOWN，不能跳过坏样本后继续报告峰值；时间戳必须为严格递增整数。E按原runner语义，startup_begin到measurement开始的完整加载/prime/warmup能量进行摊销，measurement能量只积分至deadline，不包含drain能量。父进程READY标记不截断startup能量。缺失或无效功率使E UNKNOWN。

该helper不授予物理eligible；调用者另行检查冻结源身份、相同输入/到达队列、完整完成率、原稳定性与shutdown/资源协议。部分请求未完成时，完成子群的L/P仅为诊断，不能视为最终分流收益验收。

## 真实统计模型的有限静态候选搜索核验

`flow_search_check.py` 对native_flow plan的完整公开输入/到达轨迹和所有静态候选路由，分别用朴素leaf DFS和有界leaf DFS重新构造 `FullHorizonModel` 模拟，不读取seal预测或实测最终结果。每个叶节点读取训练模型、直接task资源provider和训练family/task质量代理。模型有限目标采用完整有界drain及M预算；原planning rho通过独立 `flow_eligibility.original_rho` 保留，但不授予缺失证据的原gate约束最优性或物理资格。

上界为 `U_upper=-wQ*Q_lower-wM*M_lower`：已知训练质量代理时，Q由冻结输入和静态路由精确算出；未知时只用非负下界0。M下界取同一候选明确传入的prephase峰值×safe_factor/budget，缺失则0；这里prephase可以是训练经验代理，仅对同一预测目标函数成立，不是物理显存界。其他非负代价下界0。只有上界严格小于incumbent减1e-12时才剪枝，保留容差内全部并列。

首次真实运行 `cpu-search/flow-pilot-v8/` 使用v7训练统计模型、24task资源上下文、训练质量和90秒181请求/4候选plan，两种搜索各做4次全新forecast。4/4候选均因 `UNKNOWN_SHARED_SEGMENT_SERVICE:text#0/#1` 返回UNKNOWN；没有可报告的最优U或规划可行集，0剪枝。两算法同为空ties只表示输出一致，summary状态为UNKNOWN，不能作为最优性认证，更不能把UNKNOWN剔除后宣布剩余最优。新模型补齐训练域后须在新目录重新核验。

v8工具初版中的 `rho` 是finite forecast累计active service/H/副本数的load proxy；虽然保留.8阈值，它不等于原 `planned_routing_rps * resolve_service / capacity_factor(n)` 规则。该次所有forecast本已UNKNOWN，未授予资格。v9起proxy只做独立诊断，正式原rho调用原resolver并检查当前执行/资源绑定；没有绑定时 `rho_admission_passed=None`，即使旧catalog算术rho<.8也不默认通过。

补域训练v8后的 `cpu-search/flow-pilot-v9/` 对新181请求/4候选plan，两算法各重新模拟4次，每候选893次状态转移；4/4预测完整，无UNKNOWN、无剪枝。模型comparison U：balanced=-1.810026864147645，skewed=-4.42309532771573，spill_low=-1.8853012701309182，spill_high=-1.5918418335489957。两种搜索给出相同唯一模型最优 `shared_spill_high`，仅在固定resident/执行方式的这4个静态候选及相同模型目标内成立，不证明有界搜索加速。共同startup按plan明确省略，absolute U仍UNKNOWN；原rho正式准入四者均UNKNOWN，原gate约束最优性与物理最优性均未认证。预测spill_high相对balanced的comparison ΔU约+0.2182需等待独立实测检验，不能当成分流真实收益。

```bash
python -m tools.predictor_validation_20261027.flow_search_check \
  --plan NEW_FLOW_PLAN.json --model TRAINING_MODEL.json \
  --resources TRAINING_RESOURCES.json --quality TRAINING_QUALITY.json \
  --output NEW_SEARCH_DIRECTORY
```

## 连续到达流训练映射

v9独立全流验证未通过ΔU符号/排序和部分延迟误差验收后，旧validation始终保持validation。新增 `flow_profiles.map_flow_collection(raw_path)` 只允许由提前冻结且 `created_ns < startup_begin_ns`、`phase=predictor_training` 的plan严格派生training split；不修改旧runner或原始raw。显式 `allow_validation_diagnostics=True` 可只读映射validation，但输出固定 `split=validation,not_for_training=True`，不能进入训练拟合。

每个run输出一个完整continuous measurement cohort，不切成独立per-request trace，不制造solo观测。所有公开到达、输入、静态路由和父进程完成均与冻结plan逐项核对，实际START/PHASE必须与worker结果中每个边界相符；同alias请求必须FIFO且handler不重叠。语义阶段连续覆盖START到END，保留原native stages和scheduled/actual-submit/parent-receipt时间。request_id用完整raw SHA命名空间区分不同run，原id保留在native_request_id。

源审计包括invocation、plan/profile-input plan自哈希、冻结执行源快照、执行身份、READY PID/spec/clock/task、完整shutdown/BYE、全部NVML PID/alias/显存和、单调时间与原2秒采样gap限制。加载期可只有部分已拥有进程，READY之后必须覆盖全部驻留者。非shared资源必须另有同一context显式资源绑定，不能继承shared身份。

priming/warmup仅进入单独resource stage history，不进入service训练trace。resource_evidence保留startup/prime/warmup直到measurement开始的能量、measurement至deadline能量、单列drain能量，以及全部raw拥有显存采样峰值；不预乘safe_factor，不声称连续峰值或容量界。首组新v10 training只读审计报告为 `profile-audits/continuous-training-v10-first-flow-audit.json`：181个measurement完整映射，18个prime排除；其余新训练流完成后再拟合新版本，不反复覆盖冻结模型。

## v11一次训练与整流/整seed交叉验证协议

`flow_fit.py` 将背景训练v7、补域训练v8与提前声明的全部8条v10连续训练流合并。`background-statistical/training-v11-fit-protocol.json` 在实际拟合前冻结，源文件SHA为 `29163eea193b05f14554c8290c33401fbca173675f0c7cbcb81dd937e28627fe`；其中绑定的 `flow_fit.py` SHA为 `ea4870bbcff379717fdcad064e14a6a0193299d47602245acc3ab25dd468c69a`。旧v9独立验证不参与任何模型拟合。输出必须使用新目录，失败保留started标记，不能原地覆盖重试。

数值引擎仍为原bounded ridge、alpha=.1、经验数值界[.05,8]。每个target handler/phase各有一个等权回归行，长流因观测数更多而增加其target内权重，不按采集文件平均。只有整个handler与其余handler均无重叠的真实solo可进入算术solo基线，连续流不拆成伪solo；新增真solo可能改变基线。诊断明确报告旧/新观测占比、真solo数量、基线变化、phase顺序冲突和逐target拟合残差/矩阵秩。

相对工作损失也存在量纲归一化带来的偏向：单上下文、无正则时，最小化 `E[(d/b*s-1)^2]` 得到 `s=b*E[d]/E[d²]`，隐含时长为 `E[d²]/E[d]`，并非算术均值。工具报告实际单上下文子集的一阶/二阶矩和该解析量，明确它不是多上下文ridge的实际预测公式。逐阶段W重构保留联合向量、UNKNOWN和每个源组分布，不能称物理进度辨识或独立验证。

保留8次整流留一诊断，并额外做2次整seed留一：一个seed的四条路由全部排除，基线与系数都重新拟合，只保留另一seed四流和旧画像。模型开发比较优先使用整seed留出，避免同输入序列另一条路由使整流留一过于乐观；不启用fold模型集成。两类manifest分别供 `flow_crossfit_report.py` 新鲜全到达模拟，均标记 retrospective training cross-validation，不替代下一批前瞻独立验证。

连续raw原样保留，通过严格映射的派生training evidence wrapper作为直接拟合源；wrapper显式记录原始raw/plan收据、完整trace、映射报告与传递源。`verify_training_model(path)` 从模型直接源追到所有原始源，重新严格映射，验证训练plan在启动前冻结、执行身份一致、原始/派生trace一致和完整/排除流覆盖。原始raw同时保留于模型证据树和最终manifest，不把缺少逐行split的原raw直接当作已标注训练采集。

实际完整拟合产物位于 `background-statistical/continuous-v11/`，内部model SHA为 `0b4a60e1c24250a5af52a8097790b19025cc77ab5eb26230762fdf43d762f5e9`。10组原始训练源、410完整trace、2462请求，15 handler target、51 semantic target、15 profile，全部8流通过直接与传递源核验，validation sources为空。连续流在各target回归观测中的份额为11.1%–68.9%；真实solo算术基线变化为−2.03%至+1.39%，15 target均无phase顺序冲突。

2462/2462联合W向量均可读取，但text的两个短finalize阶段仍很差：e0ff输入平均W=.274、平均绝对工作残差.790；daaf输入平均W=.415、残差.654。某个e0ff单上下文finalize子集的时长算术均值1.452ms、无正则解析隐含时长8.701ms，相对偏差约+499%；这是描述损失偏向的解析子集诊断，不是该上下文多参数ridge的预测量。覆盖完整并未解决所有模型失配，须由整seed留出的新鲜队列/延迟/ΔU报告决定后续开发，再用下一批独立前瞻GPU数据验证，不能将训练W重构或交叉验证直接称预测验收通过。

一次完整+8整流+2整seed拟合全部完成。模型文件SHA为 `fe5afd2e4902bfb2210db9c574336fd75f0115b5b6c5650b4e3359f81c8bef20`；最终 `manifest.json` SHA为 `83bfad79245e47ad6c1d8bef6876a655831af189439fed4a54d7b579702bfd11`。每个整流fold严格保留7个训练流，留出181/181联合W可读取；每个整seed fold只保留另一seed四流，原seed四流全部排除，留出724/724联合W可读取。所有W均用对应fold的新模型重新重构，不沿用冻结v8诊断残差。供后续fresh replay的入口为 `crossfit-manifest.json` 和 `seed-crossfit-manifest.json`；这里的known仅指模型域支持，不能当作预测精度通过。

## 时长加权损失的预声明训练开发对照

新增独立 `duration_profiles.py`、`background_duration.py` 与 `duration_fit.py`，未改旧统计引擎、采集源、旧模型或冻结mapper。矩阵X仍为观察上下文时长除以目标solo算术基线，新增行权重 `w_i=1/sum_j(X_ij)=b/T_i`，不做平均权重归一化；目标为 `mean_i[w_i*(X_i*s-1)^2]+alpha/p*sum_j(s_j-prior_j)^2`。alpha=.1、速度数值界[.05,8]、handler先验1、phase匹配handler先验、solo基线与full family/task/phase上下文及UNKNOWN规则均保持。新definition明确duration-weighted estimator，原冻结BackgroundStatisticalProvider可直接读取。

单上下文、alpha=0且界不活跃时，新目标隐含时长为样本算术均值；含固定ridge时变为 `s=(1+alpha*prior)/(mean(T)/b+alpha)`，不能称严格无偏均值或渐近一致。多上下文设计来自真实未来阶段边界，因此这里只称conditional work WLS，不声称精确联合执行MLE或因果速度。独立数值审阅及10项新tests覆盖alpha0非等时长、固定ridge闭式、多上下文加权正常方程、数值界、完整task UNKNOWN及来源/定义篡改；共203项相关回归通过。

读取任何CV误差报告前，`background-statistical/duration-v12-selection-protocol.json` 封存两算法选择规则，SHA `397ef2f6b266effb2b75e530fc4e5ba650e1a0a6b5ce2307b59fbd4e1de6d75c`。固定字典序最小化：两seed合计ΔU符号与所有候选对排序错误数（缺失各计错）；原service/E2E/queue/每候选每任务P95门槛超标或UNKNOWN项数；上述误差各除原门槛后的均值（UNKNOWN为正无穷）。完全相同保留旧损失，并完整报告逐项tradeoff，不声称逐维占优。duration均值偏差/MSE另外描述，不临时加入选择标准。

v10训练plan未存validation_thresholds，前瞻附录 `selection-threshold-source-addendum.json` 绑定既有冻结 `flow-validation-v11/plan.json` 的原阈值，SHA `0c3decd5b63f8581c503a004d632d754a9afdd4c5b76c0e51560156263f78cc6`。service/E2E/P95仍各.2、queue MAE仍1请求、符号/排序准确率仍1.0，无观察后调门槛。随后冻结 `duration-v12-fit-protocol.json`（SHA `e1af2a6139fee57a54315b31c0d6167024654c11f1bc6bfbf5d3dba220687c56`），只执行2次整seed留出，不重算8次整流或全训练模型。选择仅用于模型开发；获胜者若仍不满足原CV门槛，仍需如实判未过，不能自动进入最终收益实验。

两次duration整seed拟合完成，每次406完整trace、15/51/15 handler/phase/profile，heldout724/724联合W可读取。`duration-v12/manifest.json` SHA为 `53d4fed6d36677f6a7e10f9464cca5b02ef8b461f8914fb9359d560c608c41f9`，`seed-crossfit-manifest.json` SHA为 `48975ee8adc7ee58219575137e0a05198238968581816fecc8bf32f21c8ddcc1`。后续fresh rollout和预声明selection由独立入口执行；duration获选仅表示训练开发比较结果，保留原训练CV未通过。

获选后新增独立 `duration_full_fit.py` 及4项必要CPUtests，不更改已冻结duration CV driver。`prepare` 只接受明确duration获选的training selection，递归核所有447个去重来源，并绑定原selection规则/阈值收据；旧损失获选则拒绝prepare/fullfit。full verifier严格要求全部8流无排除，保留新数值definition、所有原始/派生训练源及full工具收据，并输出兼容独立验证入口的 `verified/training_flow_count/omitted_flow_ids/omitted_seed`。207项相关回归通过。

全训练拟合协议 `duration-v12-full-fit-protocol.json` SHA为 `845402e20e12711a3341964d6dbc7166c8bb7398a0b5533cde5df335e40494f7`；新full fitter SHA为 `07db271a9aabe405a2a9918effbe32d8c9e8771cebbae6216d60e4ca987a2aa6`。此协议仍明确 `original_training_cv_thresholds_passed=False`；拟合全部训练画像不改变CV结论，更不能称独立验证通过。全模型待新的前瞻独立GPU数据检验。

授权的一次全训练拟合已完成：`duration-v12-full/training-duration-full-model.json` 文件SHA为 `84efe37340728765032436e5c2b22209c924809e4cea81a1d60036de30c33027`，内部model SHA为 `2e4571d936cc9a621c74bd9354ec785ba18ebdbede0e5a953104e547f83fa21b`，最终manifest SHA为 `2f546053a80c8c16f26cd5f002b7c5676e2c47675ddb395742e6ea9f91d9ca84`。410完整trace、15 handler/51 phase/15 profile、2462/2462联合W可读取；full source verification确认8流全纳入、无排除seed/flow、validation sources为空。来源/覆盖通过仍只说明训练模型可追溯，独立服务/排队/P95/ΔU验收待另行完成。

## OCR候选的新独立验证入口预备

`ocr_trained_validation.py` 为新OCR-handler完整模型建立独立入口，保持原冻结 `flow_validation.py` 与 `flow_trained_validation.py` 字节不变。它延迟调用 `variant_full_fit.verify_variant_full_model`，严格要求ocr_handler、full_model_verified、全部8训练流、无排除流/seed、validation_sources为空，并在seal/report前后和gate前重核。training audit必须绑定实际OCR入口与实际full verifier源码收据，不能借用其他入口的既有receipt。服务/E2E/队列/P95/六项算术仍由原reporter执行，5项新tests与21项相关tests通过。

仅CPU预备的 `flow-validation-v13/plan.json` 固定seed627502、H90秒、181请求、原四shared候选、120秒bounded drain，原v11阈值逐数值保持，split明确validation、not_for_training=True。计划SHA为 `d0ed881db71da47b22046c147d747f44357e7080b1e9f2991f8231c48cc6bcc3`，新入口SHA为 `3652b2a3b7ae2a5b589bb3712d440ecb932a948cf3dbe94ba49810d21cfd84cc`。计划生成未读取独立v11 raw/report/gate，未封存新预测或启动GPU；未来须用新OCR入口配套的suite audit/run链，旧runner的training-audit入口不能直接替代。

只读审阅发现冻结full verifier未额外将document.variant锁到OCR，因此新增 `variant_full_audit.py` 包装核验器；OCR入口现在调用该wrapper，额外要求 `strict_variant_full_identity_verified=True`。旧未执行计划字节完整保留，新的同seed/同参数计划为 `flow-validation-v13-ocr-audited/plan.json`，SHA `c1df8decd4594feae1b1a2fad9cea4c2053b349347c83235879af4b78aa14faa`。最终OCR入口SHA为 `eede948d37c5f740aa26d42d890e810c749559b7b1f6be913334202f8f1ea670`，由新计划绑定；原准备计划的旧入口receipt已不适用于新流程。

新增 `ocr_validation_run.py`（SHA `388ad6eefad4931c149850d2c441312fe9a4efb8b325ac78a31cf53f6807fbb2`）使用OCR checker与原prospective纯结构检查、资格protocol核验，prepare和run前均重核所有来源、完整候选/cohort/时序与确定性顺序；审计不打开runtime路径。明确run操作才调用native execute，首失败即停止，无重试或seed替换。4项真实入口tests覆盖未调用旧checker、空请求预测、错source阻止execute、首失败不继续；连同新wrapper与原验证链共41项相关tests通过。新计划目前仍未seal、未执行；训练CV仍有缺口时先新增训练，不能将已开发模型强行称通过。

## H600主流训练的run-aware准备

`run_aware_training.py` 独立支持旧v7/v8画像、H90 v10八主流与新H600八主流共16条完整流。未来两次相同输入repeat仅用于运行波动诊断，不进入solo基线/回归行，不算独立输入实现；preflight也排除。模型只使用已经选择的OCR handler duration数值方案，无新参数或模型搜索。每run保留run_id、来源角色、seed/候选、H/drain、请求数、runtime identity及完整capacity_condition/hash；可共同拟合phase系数，但 `capacity_evidence_pooled=False`，不同H/drain不共用容量资格。

准备协议不读取新运行raw/log/outcome，要求所有主流计划prospective training、同runtime、唯一原始输出；repeat必须逐输入/路由等同其H600 primary。`background-statistical/run-aware-v14/protocol.json` 已封存，SHA `b399ada6ab3823b4665300909ea1029532912089dbc79f41dc0b362a24c7132b`；`primary16-mapping-protocol.json` SHA为 `8483f3ba8c2aff4a623f6f07e2bb9210dd9f45f97fc08d1a3bcd8b6586d1291b`；新driver SHA为 `de6a2a15fa652368c3702051cb765a808f2e2f1bf659225a86a328518f1e46b3`。尚未运行实际拟合。

默认fit只做两个H600整seed留出：每次排除新seed四候选、保留另外12主流和旧画像，只在四条新H600流测试；旧H90交叉验证不能当H600确认。显式 `--full` 后续才生成16主流全模型。新 `verify_run_aware_model` 重新映射全部直接/传递原始源，核验12/4或16/0覆盖、来源标签、OCR数值definition以及无repeat拟合。4项真实CPU映射tests包括无outcomes仍能prepare、数据未齐不创建fit输出、旧H90全保留的H600留出、删除repeat raw不影响拟合、全16及标签篡改、repeat输入漂移；217项相关回归通过。

`run_aware_report.py` 已在实际拟合/报告前冻结，SHA `2f7329082be5957607b5e5c487cc5c75ceb423a6ccedb09dcebdfd6bb8f3b035`。报告只对新H600八主流逐条执行全新forecast，严格核验每fold完整12训练流/四留出流、原raw/plan/trace身份及无path/hash/trace泄漏，旧H90与repeat不成为测试行。每行保留H/drain/run标签、全部请求及UNKNOWN分母、service/E2E/queue/各任务P95与六项效用，按原baseline/tie判断ΔU符号和候选排序；原阈值由冻结附录与OCR入口常量共同锁定，不变更门槛，不开展新方案选择。

可通过 `--resource-manifest resources-h600-v1/fitted/manifest.json` 使用同一H600整seed留出资源模型。调用新资源helper的严格重算核验，只在预测候选副本中覆盖同路由训练startup能量均值和全history拥有显存采样最大值，实测评分候选保持原plan。资源document全部传递source也进入报告收据。未传该参数时明确标记旧资源模型为 `UNVALIDATED_H600_CONDITIONAL_PROXY`；传入时仍仅为 `H600_WHOLE_SEED_TRAINING_CV_ONLY`，经验采样最大值不授予物理界或容量资格。报告、训练、资源及相邻核心共40项CPU回归通过，其中包含真实资源engine的八流synthetic manifest重算与report接口集成。当前仅完成实现和CPU检查，尚无实际H600拟合或误差报告。

## H600独立验证入口的CPU准备

新 `h600_trained_validation.py` SHA为 `a70da89d6bd2f5962772c6bdd26617d95724ea956e28432526ba39373a99b28e`，`h600_validation_run.py` SHA为 `70729b0dc08114c20e726063ee3e992252d091390d8eb1c0daef7b82995f42ae`。只读独立审阅与56项相邻CPU回归通过；未改冻结的旧验证模块。入口严格要求run-aware full16、无omitted流/seed、无repeat拟合、完整直接/传递training来源以及OCR handler完全绑速定义；资源使用严格重算的H600 primary full manifest，不借CV fold或旧资源模型。

`flow-validation-h600-v15/protocol.json` SHA为 `cb9fb84af88b71ae0fdf1e0ec87788ed8e5e14a521748be161bad952d7e34bdc`，仅绑定容量协议原先预留的 `capacity-h600-v1/reserved-target-seed-627701/plan.json`，不复制或修改执行计划。seed627701/H600/drain300/warmup0、完整1200请求与四条shared路由保持原样。原阈值通过既有threshold-source addendum与入口LIMITS共同锁在新的验证protocol，缺少plan内阈值不会临时改变门槛。资源startup均值与full-history拥有显存经验最大值只覆盖用于预测的候选深拷贝，并在forecast/audit中保存具体值与资源manifest；sealed.sources.plan始终指原reserved receipt，因此同一raw可用于容量和预测验证，不出现两个target plan身份。

suite复用原完整cohort/时序/确定性顺序检查，并核验容量seal与预测seal绑定同一capacity protocol及原plan。每候选启动前重核来源，首失败停止，不重试或换seed。report要求startup_begin晚于完整forecast seal，先校验scored/native immutable绑定，再以显式validation-only模式严格映射实际START/PHASE/FIFO、拥有进程telemetry与shutdown；评分后重核所有native/scored/quality/mapping来源，gate重走并逐项核对。评分与原门槛仍沿用冻结reporter/gate，最终收益和物理容量资格保持未授予。本次只prepare验证protocol，未seal、未拟合、未执行GPU或读取新独立结果。

## H600同域CPU搜索核验入口准备

新增 `h600_search_check.py` 使用同一H600独立验证protocol审核原reserved plan，严格要求run-aware full16和H600 full资源manifest。原计划原样保留，只将对应训练startup能量与全history拥有显存经验峰值覆盖到候选的内存副本，再分别调用冻结 `flow_search_check.search` 进行plain/bounded两次全新模拟。仅搜索原四条shared静态路由，固定驻留副本、完整公开输入/到达、600秒窗口、300秒drain和原显存safe factor/budget；未搜索其他部署、资源份额、整请求准入、batch/cache或自适应路由。

UNKNOWN候选保留在完整域，返回UNKNOWN；两算法都没有模型可行候选时返回 `NO_MODEL_FEASIBLE_CANDIDATE`，不将空ties当成最优性PASS。完整已知且有可行解时才比较最大模型U及容差内全部ties。原catalog resolver rho逐候选独立记录，缺失适用证据时保留UNKNOWN，不认证原rho约束内最优性或物理最优性。上界只适用于相同经验资源预测目标；零剪枝不声称加速。四项新CPUtests覆盖资源覆盖与原计划字节不变、两次fresh评估及ties、UNKNOWN分母、无可行候选、质量context漂移前置阻断；连同旧搜索/入口共20项回归通过。当前仅准备入口，未用未完成模型实际运行。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m tools.predictor_validation_20261027.h600_search_check \
  --protocol artifacts/predictor-gpu-20261027/flow-validation-h600-v15/protocol.json \
  --model FULL16_MODEL.json \
  --resource-manifest artifacts/predictor-gpu-20261027/resources-h600-v1/fitted/manifest.json \
  --quality artifacts/predictor-gpu-20261027/background-statistical/training-v8-quality.json \
  --output NEW_EXCLUSIVE_CPU_SEARCH_DIRECTORY
```
