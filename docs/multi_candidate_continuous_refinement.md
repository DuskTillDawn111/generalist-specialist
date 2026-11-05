# 多候选部署与连续路由协同优化实验

本轮完成 Step 1–3：独立包装原 Joint MILP，生成不同部署，再复用原 DE 对每个部署连续精修；补齐 k=1/3/5/8、效用优先与多样性策略、同预算基线及冻结轨迹复验。

**发现了靠后候选之间的真实排名反转，但没有候选超过原 Joint MILP 第一名。** trace43 的 Text 部署从第4名升至第2名，trace45 从第3名升至第2名，均在两个优化种子复现。正式36次运行最终系统最佳效用均与原 Joint 相同；多候选增加了搜索成本。依据任务书 Step 4 的条件，本轮不新增部署邻域搜索，H4 保留为未检验；原默认优化流程保持不变。

本报告的“最佳”指声明的冻结预测器和预算内最佳已知结果。Joint 的认证只覆盖所声明有限网格；没有连续全局最优或无条件全局 Top-K 声明。

## 1. 配置、范围与公平口径

沿用上一轮的真实输入、模型画像及预测器：`artifacts/static-study-gpu-20261022/search-v2-v5-K8/system-config.json`。本轮不采用工作区另行讨论的一秒归一化配置。

| 项目 | 本轮设置 |
|---|---|
| 模型池 | G2=`generalist_2`，以及 `text/image/ocr/asr` 四 Specialist |
| 模型身份 | MiniCPM-o-4.5-GPTQ；Llama3-ChatQA-1.5-8B；SigLIP2；PP-OCRv6；Granite Speech |
| 正式副本域 | 每个模型未部署或1副本；合法任务覆盖、显存等仍由原规则判断 |
| 前导副本扩展 | 每个模型可1/2副本，另含未部署状态；只使用已有画像支持状态 |
| 到达率 | text/image/ocr/asr 各0.0748360842705697 QPS |
| 冻结轨迹 | Poisson seed42；另外在43/45分别重新优化 |
| 时间窗口 | 到达1800秒，排空300秒 |
| 目标 | 原 `six_term_quality_v1`，Q/T/L/P/M/E 各1/6，U越大越好 |
| 延迟参考 | text/image/ocr/asr 分别1000/250/10000/3000ms |
| 资源与策略 | 显存49140MiB，`search_policy=nominal`，其余原约束不变 |
| Joint | K=1；反馈上限1000轮，候选列上限50000，绝对间隙1e−8 |
| 正式共同预算 | 每次40秒、400次完整预测、384次连续路由评价；先到预算先停止 |
| DE | 原 `DERoutingOptimizer`，`best1bin`，Sobol，popsize=3，maxiter=100，seed11/29 |
| 候选精修 | 每候选最多128次路由评价，剩余时间/调用预算按待精修候选数分配 |

K=1 在每个任务的可执行模型间包含纯模型端点；K不是一般意义上的等间距1/K网格。本轮选择低K以控制候选生成成本，不重复上一轮高K扩展性试验。模型副本与任务的共享能力由现有合法执行边决定；共享多任务Specialist、多执行模型的通用可行域另有单元测试，本轮真实五模型池的Specialist仍为各自专用任务。

所有被比较方法共享配置、任务、原效用、原完整预测器及同一冻结轨迹。运行顺序固定种子打乱。总优化时间从预算创建开始，包含不可变问题/隔离会话建立、所有候选MILP、连续精修、精确缓存与搜索内记录开销。用于报告的逐请求映射导出及JSON写盘在计时结束后完成，各方法使用相同处理方式。上限为协作式检查，正在执行的单次调用/收尾可造成小幅超时，例如40.081秒；没有把这部分裁成40秒。

400次“完整预测”指原 `predict_scenarios` 口径，每次包括1/1.25/2倍服务时间的三个分支，分别计为3次leaf；主效用使用nominal。缓存命中不计新调用。MILP次数按master实际调用计数，不等于原优化器顶层调用次数：Joint第一名的一次顶层调用在本轮通常含2次master求解、1次完整预测。

### 预测器有效性边界

模式为 `calibrated_replay_v2`：逐输入独立服务中位数乘冻结全局校准系数，再通过原Router、共享FIFO、逐输入质量回放和资源回归计算完整效用。已有独立条件留出验证；本轮没有进行新GPU测量。原 `hardware_capacity_certified=false`、能耗 `measured=false` 等证据字段保留。

**当前服务函数没有随部署集合及路由活动变化的独立动态共驻干扰项。** 本轮能研究FIFO、输入分配、质量、期望负载、驻留显存和能耗预测的权衡，不能证明真实GPU上的动态共驻干扰优化。

[门禁审计](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-core-20261101/scope-audit.json)显示：改变低/高负载、ASR/OCR热点或任务集合均被 `V2_REPLAY_INPUT_CONTEXT_OR_WORKLOAD_CHANGED` 拒绝；另两个旧配置返回 `TRANSFER_SPEC_OR_CODE_CHANGED`。这些不纳入实验，也没有修改门禁、画像或权重强行运行。因此结果不覆盖高负载、热点负载、新任务或任意模型池。

## 2. Step 1–2：不同部署候选的提取

实现位于 [multi_candidate.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/continuous/multi_candidate.py)，通过显式导入使用，不接入默认planner。

1. 每次创建新的原 `JointMilpOptimizer`，第一次求解不限制部署。
2. 保存其完整有效预测、部署、网格路由、真实U、状态、认证、用时和调用数。
3. 将已得到的**整个部署向量**加入 `exclude_deployments`，重复求解剩余域。零副本与未部署按原模型语义处理。
4. 每次反馈状态独立；不把 `exclude_plans` 的单条路由排除当作部署排除，不跨路由复用未经证明的反馈割。
5. 候选选择以完整预测U排序；始终保留最初不受限Joint的第一名。预算提前结束时记录实际数量及未认证状态，不将master界当作真实U。

实际有效运行中所有候选的残余有限网格求解均完成认证，未出现UNKNOWN。每次证书仅针对当次排除后的有限域；代码仍明确输出 `global_top_k_certified=false`，不把重复受限求解包装为连续域Top-K证明。若后续出现提前终止，同一接口会保存未认证候选和有效incumbent，单元测试覆盖此分支。

候选策略如下：

- **A，效用优先**：生成k个不同部署，保留第一名，其余按完整预测U排序。
- **B，效用与结构多样性**：先生成2k个不同部署，再选k个。保留第一名；在距离已生成最佳U不超过0.15的候选中，按与已选集合的最小部署距离最大化选择，效用用于平局排序。距离为激活集合Jaccard距离与归一化副本差的均值；窗口无候选时回退到剩余候选。0.15是预声明的选择启发式，不是最优性界。全部2k次候选生成成本计入策略B。

### trace42 的前五个候选

下面为utility-k5、seed11的一个完整生成记录，副本均为1。模型缩写G=G2，T/I/O/A=text/image/ocr/asr；“路由”列写每个任务的执行模型，比例均100%。显存是原预测中的安全估计，并非新增硬件实测。

| 网格名次 | 部署 | text/image/ocr/asr路由 | 完整U | 显存MiB | 最大rho | 新预测/MILP | 该次求解秒 |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | G+I+O | G/I/O/G | −0.309303 | 24967.9 | 0.1015 | 1/2 | 0.034 |
| 2 | G+I+O+A | G/I/O/A | −0.332252 | 37854.9 | 0.0510 | 2/3 | 0.095 |
| 3 | T+I+O+A | T/I/O/A | −0.341342 | 39060.0 | 0.0510 | 2/4 | 0.159 |
| 4 | G+I+O+T | T/I/O/G | −0.343243 | 37694.8 | 0.0625 | 1/4 | 0.038 |
| 5 | G+O+A | G/G/O/A | −0.389585 | 27361.8 | 0.0545 | 4/6 | 0.255 |

全部满足原约束。这里单次求解时间不含前面的公共setup；端到端对照表包含setup及全部成本。每次预测的约束、六项贡献、显存、部署与路由均保留在原始JSON；[派生诊断表](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-final-report-20261102/diagnostics.json)可直接查看本表的未舍入值。

trace42 多样性k5从10个生成候选中选择索引 `[0,2,4,7,6]`，没有选中索引3的Text部署。索引7在seed11改善0.007518，但最终仍只有−0.416671。多样性本身不保证保留最有路由潜力的部署；本轮没有显示它优于效用优先。

## 3. Step 3：固定部署连续精修

每候选固定模型集合及副本数，复用 `FixedDeploymentRoutingEvaluator` 与原DE。原网格路由作为warm start和保底有效解；闭simplex参数化允许精确零、纯模型端点及多个执行模型，不把连续比例重新量化到网格。零流量的已部署模型仍驻留，显存及相关资源按原固定部署计算。

精修过程**没有部署MILP**。若精修未改善或预算用尽，保留原网格路由。总结果始终保留所有阶段已观察到的最佳有效方案，保证不退化。生成阶段使用setup完成后剩余时间的50%，仍处于原绝对总截止时间内；精修不会重新获得一份完整总预算。

缓存仅复用语义完全一致的完整输入：不可变问题/目标身份、oracle版本、部署、完整精确路由及轨迹身份。缓存内容深拷贝隔离；没有比例四舍五入，也没有仅凭相同FIFO映射复用整份预测。

### 排名变化：观察到了什么

正式实验共36次运行，生成132次候选出现记录，精修92个候选阶段，其中12阶段有正收益；6个严格成对排名反转分布在4次运行。它们不代表6个独立实验：同一部署可同时反超两个候选，同一配置有两个优化种子。

| 轨迹 | Text候选网格U及名次 | 精修后U及名次 | ΔU | 原Joint第一名U | 超过第一名？ |
|---|---|---|---:|---:|---|
| 42，多维多候选DE | −0.343243，第4 | −0.343021，第4 | +0.000222 | −0.309303 | 否 |
| 43，多维多候选DE | −0.344806，第4 | −0.331612，第2 | +0.013194 | −0.306854 | 否 |
| 45，多维多候选DE | −0.342265，第3 | −0.325722，第2 | +0.016543 | −0.298403 | 否 |

上述utility-k5的结果在优化seed11/29均出现。trace43的Text路由为50%给Text、50%给G2，image/OCR仍纯Specialist，ASR给G2。K1只含端点，因此这是原K1未覆盖的有效分流；不能由此宣称它超越K8或任意更细网格。trace42 的轻微收益未造成排名反转。

在trace43/45分别重新运行算法，说明这些冻结轨迹上都存在候选内收益；这**不是**把同一条学习到的路由放到独立轨迹做holdout评估，也不足以证明统计显著性或泛化。

![候选精修前后的排名，包含trace43的真实反转](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-final-report-20261102/candidate-ranks.png)

![候选效用变化与增益；图示trace42，并非所有轨迹最大收益](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-final-report-20261102/candidate-gains.png)

### 搜索是否充分：独立单任务DE检查

不能因为多候选算法未超过第一名，就认定连续路由没有更好解。检查trace42的Text候选发现：seed11的115条多维评价中，只有4条同时将image/OCR保持在Specialist端点；这4条的Text比例仅为1、0、0.5和0.0624958。原DE先进行全维搜索，预算有限时未必到达有利子面的充分搜索。

为检验这一问题，另做**独立诊断**：重新生成5个Joint候选，选首个含G2及Text的部署，仅Text路由可变，其余行固定为原候选网格路由；仍使用同一DE。生成成本计入，40秒、240次预测、128次路由上限。该检查不混入主算法排名或正式对照表，也没有称作MADS或新算法。

| DE seed | Text给S比例 | 实际Text请求 G/S | 最佳U | 相对该部署K1的ΔU | 全部预测/MILP/路由评价 | 总秒 |
|---|---:|---:|---:|---:|---:|---:|
| 11 | 0.779931263684 | 29/102 | −0.327145 | +0.016099 | 133/19/125 | 11.625 |
| 29 | 0.436274661910 | 74/57 | −0.328461 | +0.014782 | 136/19/128 | 9.325 |
| 47 | 0.535787858784 | 61/70 | −0.328429 | +0.014814 | 136/19/128 | 9.504 |

其他任务保持image145请求给I、OCR143给O、ASR148给G2，567请求守恒。三个连续比例均非K1端点，FIFO请求映射确实改变；三个结果都可反超该轨迹的网格第二名−0.332252，但仍不及第一名−0.309303。主算法的候选内搜索预算/任务分配存在不足，这已得到正面证据；连续全局最优仍未知。

### 为什么局部改善仍追不上第一名

用本轮独立检查seed11的记录分解，与原Joint第一名比较：

| 六项效用贡献的变化 | 连续Text部署 − 原Joint第一名 |
|---|---:|
| 质量 Q | +0.00222646 |
| 吞吐 T | 0 |
| 平均延迟 L | +0.00016630 |
| P95 P | +0.02377509 |
| 显存 M | **−0.04316539** |
| 能耗 E | −0.00084412 |
| 总U差 | **−0.01784165** |

额外Text驻留令预测显存从24967.9升至37694.8MiB，多12726.9MiB。质量和延迟的收益抵不过显存等资源代价。相对**同一四模型部署**的K1路由，显存贡献不变，连续收益约0.01610主要来自P95改善；相对**三模型原赢家**，显存差成为决定性负项。不能把两类对照混为“分流无收益”，也不能把残余差归因于已经测得的动态GPU干扰。

## 4. FIFO等价、缓存与改善成本

每次完整输入对应的请求分配保存索引、到达、任务、模型、副本与完整映射哈希；逐项重新使用原路由逻辑重建验证。实际QPS与有限请求实现分配分开记录，未把模型计数相同误当映射相同。

正式92个精修阶段中，连续两条DE评价有44对完整映射相同，这44对U仍全部不同。原因是能耗、rho等还使用连续期望QPS；映射相同不意味着完整目标或约束相同。这里的“相邻”仅指DE评价顺序，**不是**把一维比例排序后的相邻采样，更不是对实数区间平台的证明。上一轮的细扫平台诊断仍见[路由收益报告](/root/autodl-tmp/generalist-specialist-project/docs/routing_benefit_diagnosis.md)。

[stages.csv](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-final-report-20261102/stages.csv)逐候选列出有效预测调用、精确缓存命中、不同路由/映射数、改善事件数、首次改善前预测数、每次改善的平均预测数，以及赢家路由。没有改善的候选成本栏为空，不能把它算成零成本改善。正式实验对系统第一名的改善为0，故不存在有限的“每次系统最佳改善成本”；局部候选改善成本只能说明候选内部搜索效率。

已对前导与正式48次有效运行的10787条逐运行部署/路由映射记录完成重建，再核验独立单任务3次的405条记录。共11192条记录全部一致；此为跨运行含重复输入的记录数，不能称为11192个全局唯一预测输入。

## 5. 四组核心对照与k消融

算法A为原Joint；B为当前DE+Fixed-Routing MILP实现（使用已有 `milp_bounds` 安全界与隔离预测会话，原DE不变）；C为本轮包装器k1；D为多候选k3/5/8。C不等于只计精修时间，包含第一次Joint和setup。E邻域增强未执行，不能给出其收益或成本对照。

trace42中两优化seed的中位数如下。所有方法最终U均为 **−0.309302874267**，本表没有效用提升。预算相同指同一上限，不强迫各算法耗尽预算。少于400次预测或384次路由也可能因单候选128次上限、DE结束或时间分配停止。

| 方法 | 总秒 | 完整预测 | MILP次数 | 路由评价 | 达到共同目标秒 | 达标前预测 |
|---|---:|---:|---:|---:|---:|---:|
| A 原Joint | 2.279 | 1 | 2 | 0 | 2.245 | 1 |
| B DE+MILP | 18.476 | 170 | 554 | 384 | 5.972 | 61 |
| C 效用优先 k1 | 10.251 | 128 | 2 | 128 | 0.850 | 1 |
| D 效用优先 k3 | 22.818 | 181 | 9 | 179.5 | 1.858 | 1 |
| D 效用优先 k5 | 32.956 | 372 | 19 | 369.5 | 0.896 | 1 |
| D 效用优先 k8 | 25.605 | 384 | 32 | 384 | 0.808 | 1 |
| D 多样性 k1 | 10.964 | 130 | 5 | 128 | 0.831 | 1 |
| D 多样性 k3 | 20.092 | 265 | 25 | 257 | 0.855 | 1 |
| D 多样性 k5 | 33.678 | 386 | 66 | 362 | 1.808 | 1 |
| D 多样性 k8 | 27.844 | 400 | 116 | 384 | 0.809 | 1 |

整数指标的两样本中位数可出现0.5。达到共同目标按完整预测结束时的有效观察计时；候选生成的后续认证/精修仍计入总耗时。运行时受共享主机、冷热启动及强记录影响，原Joint两次耗时范围1.503–3.054秒。包装器内第一解约0.8秒，**不是包装器加速原Joint的证据**，也不能据此给出独立Joint的稳定加速比。

相对B，C将554次MILP降到2次，完成搜索成本较低；D-k5降到19次MILP，但增加到372次预测，总时间反而更长。多样性k5需生成10个部署，66次MILP、386次预测，也未改善系统第一名。新增候选减少重复部署求解的同时增加了路由与完整预测成本，不能只看MILP次数。

所有Joint初始化方法第一条预测就达标，B需约61条预测。这来自**原Joint初始化**，k1已经获得；不能归因于多候选。k增大时每候选的预算反而下降，并且部分运行触及时间预算，因此耗时/调用数不是随k单调增长，不能用k8比k5快证明大k更高效。

| 额外冻结轨迹 | 方法 | 中位总秒 | 中位预测 | 最终U |
|---|---|---:|---:|---:|
| 43 | Joint / DE+MILP / k1 / k5 | 0.894 / 17.854 / 8.594 / 30.051 | 1 / 170 / 128 / 387 | 全部−0.306854442 |
| 45 | Joint / DE+MILP / k1 / k5 | 0.852 / 16.235 / 11.306 / 22.867 | 1 / 164 / 128 / 363 | 全部−0.298403113 |

两个优化seed只能说明本次可复现情况，不能做统计显著性结论。时间范围、各seed未舍入结果、实际候选数和终止状态见[comparison.csv](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-final-report-20261102/comparison.csv)、[seeds.csv](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-final-report-20261102/seeds.csv)。

![总时间上的最佳有效效用](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-final-report-20261102/anytime-time.png)

![完整预测次数上的最佳有效效用](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-final-report-20261102/anytime-predictions.png)

![k与最佳效用、总时间和预测数](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-final-report-20261102/candidate-count.png)

## 6. 假设判断与阶段决定

| 假设 | 本轮判断 | 证据和边界 |
|---|---|---|
| H1 靠后部署反超原第一名 | **未观察到** | 有较低名次之间的反转，但所有有效运行均未超过第一名；不是不存在更好连续解的证明 |
| H2 收益与部署结构有关 | **得到本轮支持** | 原赢家精修无增益，Text等额外驻留部署出现正增益；支持范围限本轮画像与轨迹 |
| H3 多候选优于单部署精修 | **系统效用上未成立** | 找到k1未评价的候选内收益，但最终联合最佳与k1相同，成本更高 |
| H4 邻域继续改善 | **本轮未检验** | 根据Step 4条件暂停扩展，不能填写零收益或声称失败 |
| H5 合理求解成本 | **仅初始化达标成本受支持** | Joint初始化比双层DE少调用即达同等U；多候选额外成本没有换来更高系统U |

本轮对应任务书的情况C与D：连续精修有局部收益，尚未提高联合最佳；多候选增加的完整预测成本没有兑现为系统收益。因此完成Step 1–3及A/B/C/D同预算对比后，不扩展Step 4，Step 5中的E和“前五名+邻域”消融也不执行。缺少邻域前后图是条件分支未进入的结果，不能伪造对照值。此前独立邻域实验也不能冒充本轮多候选邻域的结果。

下一步优先级：

1. **保留低K Joint作为默认。** 原始最优器、效用、画像及FIFO不变，不迁移到新包装器。
2. **先解决候选与任务预算分配。** 本轮有明确的有利子面搜索不足证据；研究在保留端点、原路由与闭simplex的前提下，如何给少量有希望的任务行分配预算。继续沿用DE做可解释消融，暂不增加BO/MADS。
3. **研究结构化安全加速。** 使用原约束内可证明的驻留资源界、任务覆盖检查、合法副本预筛选、Joint列生成与部署剪枝。若预测器不能提供有效界，保留UNKNOWN，不能用经验上界认证剪枝。
4. **控制“再加一个模型”的机会成本。** 本轮Text的显存负项超过路由收益；候选选择要同时考虑激活成本和可能的路由改善，而非仅按网格名次或结构距离扩池。启发式潜力估计不能冒充安全界。
5. **进入邻域的触发证据仍待获得。** 若后续有限预算分配能稳定提高联合最佳或识别出原候选未覆盖的有希望合法结构，再做独立一层邻域消融；每次改变部署重新预测资源与FIFO，单独计费。

## 7. 数据、失败记录与验收

| 批次 | 完成运行 | 完整预测 | 用途 |
|---|---:|---:|---|
| corrected pilot v2 | 12 | 3269 | k1/3/5及1/2副本域的小规模验证 |
| formal core | 20 | 4834 | trace42，两策略k1/3/5/8及Joint、DE基线 |
| formal stability | 16 | 2684 | trace43/45，Joint、DE、效用k1/k5 |
| independent face v2 | 3 | 405 | 单任务DE的搜索充分性诊断 |
| 合计有效批次 | **51** | **11192** | 正式对照仅取core+stability的36次/7518次预测 |

前导+正式共48次的审计统计为168个候选出现记录、128个精修阶段、16个正增益阶段、4074次MILP、11937次连续评价；正式36次单独为132/92/12、3952次MILP、8686次连续评价。上述都是运行记录/阶段计数，不是全局去重模型结构数。

原始数据位置：

- [corrected pilot](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-pilot-v2-20261101/summary.json)
- [formal core](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-core-20261101/summary.json)
- [formal stability](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-stability-20261102/summary.json)
- [independent face](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-face-check-v2-20261102/summary.json)
- [汇总原始分析](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-final-report-20261102/analysis.json)，[候选CSV](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-final-report-20261102/candidates.csv)，[反转CSV](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-final-report-20261102/reversals.csv)
- [48次审计](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-core-20261101/combined-audit.json)，[单任务审计](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-face-check-v2-20261102/audit.json)
- [交付清单、源码与日志](/root/autodl-tmp/generalist-specialist-project/artifacts/multi-candidate-closeout-20261102/completion.json)

每个主批次保存 `plan.json`、逐case配置/副本域、`source-manifest.json`、优化源码快照和逐run完整JSON。后者包含候选原预测、各次残余MILP、连续评价历史、事件、赢家、预算和逐请求映射；汇总CSV用于阅读，不能代替原始记录。源码及预测输入身份在审计中核对。

失败/未完成记录另存，不纳入正式效果结论：

- 初版16秒pilot保存8次运行、362次预测；setup前计算生成分段预算，导致部分请求5候选却仅得到3个或无初始解，主动停止。已修正生成阶段相对setup后的剩余预算计算；原全局预算没有续期。记录在 `artifacts/multi-candidate-pilot-20261101/stopped.json`。
- 大预算补充运行在最终run保存前以SIGKILL/137退出，原因未确定。`artifacts/multi-candidate-budget-supplement-20261102/failure.json`保留，不能解释成“加预算仍无收益”。
- 第一版单任务诊断把冻结行中已排除的零比例键传入，触发 `UNSUPPORTED_ROUTE`。仅修正诊断初始行投影，未放宽域/门禁；失败记录在 `artifacts/multi-candidate-face-check-20261102/failure.json`，重跑结果为v2。
- 汇总派生表首次同时保留多个大JSON时也以137退出；改为逐文件释放对象后成功，不涉及新的优化运行或预测。失败及重跑信息在交付日志中记录。

51次有效运行之外，初版pilot的8条完整记录单独保留；不把它们并入正式样本膨胀结果。两个没有最终run结果的优化/诊断尝试没有完整调用账本，不能纳入调用总数。

CPU回归 **289项全部通过**，其中新增 `tests.test_multi_candidate` 8项，覆盖独立笛卡尔网格排名、全部署排除、不泄漏反馈/限制、合法域与共享多任务Specialist、精确缓存、预算、初始解保留以及未认证候选。原 `gsdeploy/milp.py`、`optimizer.py` 与HEAD一致；全部 `gsdeploy/**/*.py` 的当前哈希与各有效批次manifest一致。测试与审计验证软件语义和证据一致性，不替代新GPU测量。

## 8. 使用与复现

以下命令均从 `/root/autodl-tmp/generalist-specialist-project` 执行，输出目录必须不存在。运行依赖上述本地冻结画像/预测输入；不提供虚构替代数据。CPU测试可单独运行。

```python
from gsdeploy import SystemConfig
from gsdeploy.continuous.multi_candidate import MultiCandidateContinuousRefinement

config = SystemConfig.load(
    "artifacts/multi-candidate-core-20261101/p5-n1-K1-trace42-config.json"
)
result = MultiCandidateContinuousRefinement(
    config,
    replica_options={m: [1] for m in config.catalog["models"]},
    k=5, strategy="utility", seed=11,
    seconds=40, max_predictions=400, max_routes=384,
    per_candidate_routes=128,
).solve()
# 查看 result["winner"]、["candidates"]、["stages"] 和 ["budget"]。
# 此接口为显式实验入口，不替换 DeploymentSystem.plan()。
```

```bash
python -m unittest tests.test_multi_candidate -v
python scripts/run_cpu_tests.py

python experiments/multi_candidate_refinement.py --mode pilot \
  --seconds 40 --predictions 400 --routes 384 --per-candidate-routes 128 \
  --replicas 1 2 --traces 42 --seeds 11 29 --ks 1 3 5 \
  --strategies utility --K 1 --output artifacts/multi-candidate-pilot-reproduce

python experiments/multi_candidate_refinement.py --mode full \
  --seconds 40 --predictions 400 --routes 384 --per-candidate-routes 128 \
  --replicas 1 --traces 42 --seeds 11 29 --ks 1 3 5 8 \
  --strategies utility diversity --K 1 --output artifacts/multi-candidate-core-reproduce

python experiments/multi_candidate_refinement.py --mode full \
  --seconds 40 --predictions 400 --routes 384 --per-candidate-routes 128 \
  --replicas 1 --traces 43 45 --seeds 11 29 --ks 1 5 \
  --strategies utility --K 1 --output artifacts/multi-candidate-stability-reproduce

python experiments/multi_candidate_face_check.py \
  --output artifacts/multi-candidate-face-reproduce

python scripts/report_multi_candidate.py \
  --sources artifacts/multi-candidate-core-20261101 artifacts/multi-candidate-stability-20261102 \
  --output artifacts/multi-candidate-report-reproduce

python scripts/audit_multi_candidate.py \
  --sources artifacts/multi-candidate-pilot-v2-20261101 \
    artifacts/multi-candidate-core-20261101 artifacts/multi-candidate-stability-20261102 \
  --face artifacts/multi-candidate-face-check-v2-20261102 \
  --output artifacts/multi-candidate-audit-reproduce.json

python artifacts/multi-candidate-final-report-20261102/derive_diagnostics.py
```

固定输入和种子支持确定性复验；若触及墙钟预算，主机负载变化可能改变实际评价数量和最终路径。原始数据、顺序、软件版本及源码哈希保留，不能保证不同机器上的用时逐位一致。
