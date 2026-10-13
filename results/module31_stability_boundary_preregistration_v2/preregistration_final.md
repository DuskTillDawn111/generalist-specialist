# Module 31：稳定性边界探索协议（审阅冻结版）

**当前状态：仅预注册完成，等待审核。GPU 执行未获授权；本阶段 GPU 会话预算为 0。** 本文件的“冻结”指固定可审核的协议提案，不表示已获执行批准。新增数值及草案候选值统一标为 **待批准**；已核对的历史值、由历史值直接计算的数值和本阶段 CPU 计数核验分开记录。不得启动 worker、调用推理、自动运行探索或三臂对照。

## 1. 问题、范围与冻结来源

研究问题是：在原硬件、原五模型各一副本、六任务与原动态路由器条件下，平均强度和 LID 短时突发如何影响 **600 秒有限窗口**的原稳定性分类？主要终点是每个条件、种子、arm 的完整 `stability_state_v2`、全部硬失败/警告原因与恢复情况。探索结果不是动态收益的正式证明，也不是无限时域稳定、容量上限或跨模型泛化证明。

完整阅读并据实际调用核验：Module 30 的 `next_experiment_preregistration_draft.md`；Module 28 的 `freeze.py`、`gpu.py`、`dynamic_router.py`、`report.py`；Module 29 stage2 的 `prepare_validation.py`、`run_controlled.py`、`instrumentation.py`；`module21/trajectory.py`、`joint_deploy/measurement/load_generator.py`、`performance_model/local_calibration/stability_v2.py`。分类、效用、窗口和停止细节也核对了 `module26/common.py`、`module26/report.py`、`module11b/measure.py`、原 runtime/schema/metrics、质量政策、效用定义与参考 manifest。输入的完整路径、字节数、SHA-256 与前后核验见 `input_integrity_report.json`。

保持 Module 28 `DYNAMIC_EVIDENCE_GAP`、Module 29 第一阶段 `EVIDENCE_INCOMPLETE`、第二阶段 `CONTROLLED_SPLIT_LOGGING_VERIFIED`、Module 30 仅针对预注册准备的 GO。历史 S-only 与 Dynamic 都为 `unstable`，历史合法 U、ΔU 仍为 NULL。历史文件、冻结路由器、正式优化器和既有结果禁止修改。

## 2. 实际可控制的负载参数

|参数|代码能力与冻结方式|证据状态|
|---|---|---|
|逐任务平均到达率|`arrivals(rates,duration,seed,label)` 用逐任务独立 `expovariate(q)`；可传入正 QPS 向量|已有实现，已核对|
|正式时长、warmup、种子|phase 从轨迹 duration 读取；生成器由 seed 与 label 分流 RNG|已有实现；本协议沿用 600/30 秒|
|样本顺序与循环|`SamplePool` 每任务按角色字符串种子 shuffle，随后循环；冻结池每任务 30 个独特样本|已有实现；不能声称每请求都是新输入|
|固定分流|`choose` 使用既有 route uniform 与 replica uniform；固定比例为期望比例，非每次恰好 10%|已有实现；确认性对照沿用 fixed10|
|短时突发|Module 28 `freeze.py` 仅对 LID 时间做保均值 warp：每 10 秒前 2 秒 2 倍、后 8 秒 0.75 倍|已有实现；并非 PoissonConfig 内建 burst 开关|
|周期到达|通用 `periodic_arrivals` 支持确定周期到达；默认 MeasurementConfig 仅验证 periodic，Module22 PoissonConfig 复用结构校验|本协议不新增 periodic arm|
|动态节流参数|令牌率 0.5750000000000001 rps、冷却 0.5056366968 秒、在途上限 1 等来自原参数|不是本实验的可调负载旋钮；禁止随强度缩放|

负载只改变平均 QPS 和预定到达模式，不改变任务语义、输入长度筛选、背景组成、模型、副本、prompt、解码、batch 或质量阈值。正式测量由冻结轨迹驱动，不能把名义 transport config 当成实际生成算法。

## 3. 两个扫描轴与任务组成

基准为 LID 11.5 rps，Text、Image、OCR、ASR、VQA 各 0.2 rps。所有任务共同乘以 a；期望任务权重为 LID 0.92、其余各 0.016，不要求实现请求数严格保持比例。

|a|LID rps|其他每任务 rps|总 rps|600 秒期望总请求|其他每任务期望请求|
|---:|---:|---:|---:|---:|---:|
|0.60|6.9|0.12|7.5|4500|72|
|0.80|9.2|0.16|10|6000|96|
|1.00|11.5|0.20|12.5|7500|120|

以上是草案基准缩放的确定计算，不是实测请求总数。**网格 [0.60,0.80,1.00] 用于本新协议仍待批准。**

平均强度扫描：在 steady、burst 两种模式内分别按升序扫描三个 a，对两 arm 分别报告分类和趋势。到达模式扫描：在每个固定粗网格 a 比较 steady 与 burst，所有点现在列出，不根据动态触发次数、延迟或 U 选择模式。steady 为六任务平稳 Poisson；burst 仅改变 LID，其余五任务保持平稳 Poisson。

令原 LID 事件 t=10c+u：若 u<4，t'=10c+u/2；否则 t'=10c+2+(u−4)/0.75。周期 10 秒、相位 0，2×2+8×0.75=10；600 秒与 30 秒均为整周期，warp 保留同种子基础流的精确请求数和样本。两模式使用不同预定种子，因此模式比较是分布层面的对比，不伪装成同一输入逐请求配对。每模式内两 arm 则使用完全相同轨迹。不同强度共用同模式 seed，使用 common random streams；跨 level 结果有关联，不能当作独立复现。

## 4. 请求生成、种子与运行顺序

`load_grid.json` 冻结唯一生成规则：先用原 `arrivals` 生成事件；每 phase 新建同 seed 的 SamplePool；在基础事件顺序中绑定样本后再做 LID warp，按 `(offset_seconds,task)` 排序。到达、route、replica 流字符串分别为 `{seed}:{label}:arrivals:{task}`、`...:route:...`、`...:replica:...`；样本 shuffle 为 `{seed}:samples:{task}`。warmup 与 measurement 到达流独立，但两 phase 的样本序列从同一 seeded permutation 开始，沿用原实现。不能重采样、替换困难输入、按语言挑选路由、填补请求或为覆盖率换 seed。

本阶段只做标准库 CPU 到达计数核验，未物化新轨迹、未启动 worker。核验共覆盖 53 个可能的 (profile,level,seed) 组合；确认性三枚 seed 的计划总请求为 5954、6057、5800，各任务最小计划请求为 83、90、82。这些是生成规则计数，不是成功观测。部分探索点（含 .80 steady 的 seed300102 最小71）低于72，保留原seed，不因此改分类或单会话原evaluate准入；72是后续正式三臂证据覆盖门槛。**执行批准后且任何 GPU 运行前**，须一次生成并哈希冻结所有粗扫、所有可能细扫/复核分支、确认性对照的 warmup 和 formal 轨迹；冻结生成器版本、顺序、请求 ID、样本引用、route/replica uniform 与原文件哈希，所有 arm 重用字节一致的轨迹。生成成功不授权 GPU。

|用途|steady seed|burst seed|状态|
|---|---|---|---|
|粗扫，每条件 2 次|300101、300102|300111、300112|待批准；steady 来自草案候选|
|唯一细扫，每条件 2 次|300201、300202|300203、300204|待批准|
|边界复核，每条件 3 次|300211、300212、300213|300221、300222、300223|待批准|
|独立确认性三臂，每条件 3 次|300301、300302、300303|不执行 burst 确认性 arm|待批准、须单独授权|

所有分析 RNG 也在 `seed_manifest.json` 中冻结：确认性 paired 60 秒 block 条件敏感性（原 500 draws），相对 S-only 为 300401–300403，相对 fixed10 为 300411–300413；无随机运行顺序、无新的 bootstrap 或有利 seed 搜索。记录内未发现这些候选种子用于结构化 seed/RNG 字段；草案提及不是已执行。外部或未记录的开发史仍 UNKNOWN，启动前须审核；发现冲突则 NO_GO，不自动顺延替换。

`execution_order.json` 给出所有 slot 与互斥分支的精确顺序。粗扫按 a 升序，同 level steady 后 burst；模式内按 seed 顺序，每 pair 的 S/D 或 D/S 由零基 `(level_index+seed_index+profile_index)` 奇偶冻结。粗扫结束后 steady 细扫、burst 细扫、steady 复核、burst 复核。细扫和复核顺序使用零基 `(seed_index+profile_index)` 奇偶。所有会话串行，不能同时执行两个 arm。

每种模式最多一次细扫：只有相邻已完成粗网格点中，**至少同一个 arm** 从低点两 seed 均非 unstable，变成高点至少一个 seed unstable，才选最低相邻对的算术中点。可能值仅为 0.70 或 0.90。仅正常完成、没有基础设施故障的原分类能触发；没有变化、变化方向相反或缺失证据不细扫。不得向网格外扩展，或为动态收益挑选中点。

每模式复核只选择已扫描、两 arm 各两 seed 均满足原分类、完整恢复、原合法性和配对输入/热态的**最高已完成 level**，使用该模式固定的三枚复核 seed。没有共同通过点就跳过；三 seed 任何不通过都保留，不改到更低点再复核。探索中非零 G 不是选点要求：路由未启用也必须保留。复核是探索边界可靠性的检查，不是确认性效用结果。

## 5. 窗口与硬件

沿用已核对的正式 600 秒、warmup 30 秒、每个已支持 worker/task 路径 sequential prime 3 次、各 phase 排空上限 60 秒，postwarm/postmeasurement idle 各 2 秒；prime、warmup、冷却和排空分别记账，不能混入 formal。每会话重新启动五 worker；每 phase 新建原 DynamicRouter，warmup 状态不传入 formal。

原 manifest 要求的设备为 NVIDIA GeForce RTX 4090，UUID `GPU-31f2671b-99c5-ef57-be45-5c9a871e11f4`，driver `595.71.05`，报告总显存 49140 MiB。这里只引用历史记录，未实时验证，也不凭商品型号推定实际显存。执行前必须逐字段匹配；不匹配即 NO_GO，不临时换 GPU。

固定 Qwen2.5-Omni、Whisper-medium（revision abdf7c39ab9d0397620ccaea8974cc764cd0953e）、RoBERTa SST2、ConvNeXt Tiny、PARSeq（parseq-bb5792a6.pt）各一副本。注册器、解释器路径、launchers、环境 overrides、模型完整 ID、s0、质量准入和 catalog 使用原 manifest，详见 JSON。B4=34567.5534 MiB，安全因子 1.110853955254881；0.5 秒监测 NVML owned/total 显存及 foreign PID，不把采样峰值称为瞬时 allocator 峰值。

原热态：prelaunch≤38°C，3 次连续合格，2 秒采样间隔，最长 300 秒；resident premeasurement≤40°C，3 次连续合格，最长 180 秒；同配对/三臂组 postwarm idle 均温最大差≤5°C。不改时钟、功率、MPS。启动前无外部 GPU 进程、used≤128 MiB；关闭后使用原 release 核验。单 worker READY timeout 300 秒（原 runtime 是逐 worker，不能误报为全局 300 秒），drain 60 秒，shutdown 配置 15 秒。live GPU 隔离、环境包版本和 checkpoint 全量 attestations 本阶段未确认，列为 P06。

## 6. 原稳定性分类完整保留

唯一分类器为 `performance_model/local_calibration/stability_v2.py` 的 `reconstruct/classify`，全量 DEFAULT_THRESHOLDS 原样冻结在 `load_grid.json`。不调用 scaled_thresholds，不修改硬阈值或警告阈值。

B(t)=实际 host send-start 提交数−本 cohort 已成功记录的响应数；不是计划到达，也不是 worker ACK/真实等待队列。系统及每个有请求的 worker 分别重建。正式 600 秒为 20 个 30 秒积分 bin，尾部 25%=5 个 bin=150 秒。尾部硬失败 `sustained_positive_tail_growth_with_nontrivial_end_backlog` 必须同时满足：

1. 尾 5 个 bin **时间积分均值**的 OLS slope≥0.02 请求/秒。
2. 尾 bin 最后均值−第一个均值≥5 请求。
3. 尾相邻 bin 均值严格递增比例≥0.6（4 个相邻差，至少 3 个为正）。
4. 正式窗口末端 B≥5 请求。

四项为 AND，不能换成 OR、瞬时峰值或不同尾窗。排空完成不解除已经触发的尾部硬失败。B 的全观察峰值包含有界排空，formal-only peak 另报。

其他原硬规则全部保留：incomplete>0；drain timeout；completion_ratio 缺失或<0.98；worker/request/run errors；unclean shutdown 或失败状态；不足提交时间戳；以及非完全恢复。`final_recovery_confirmed` 必须 completed=offered 且没有排空超时，不能只凭≥98%完成认定恢复。

原警告阈值全部保留：B 峰值≥30，请求全 s0 服务需求积压峰值≥5 秒，drain≥5 秒，等待≥1000 ms 的请求数达到 max(10,ceil(0.05×offered))。输出仍是 `unstable`、`recoverable_congested`、`steady`；后两者分别报告，不合并成长期稳定。没有某项时间戳或服务路径时按原实现处理，不补造零值；额外日志缺口另记 UNKNOWN，不能虚构分类通过。

边界报告使用可复核的有限网格括区间，按每模式/每 arm 给出全 seed 分类表、最高共同通过点和复核全结果。两端都有证据才称观察到 transition interval；混合 seed、非单调、全部通过、全部失败或安全截断分别记不确定、非单调、右侧未括定、无共同通过点、证据截断。不得拟合不存在的“真实极限 QPS”，不得将不触发 unstable 自动等同长期稳定。

## 7. 后续独立确认性三臂匹配条件

**现在预定 a=0.80、steady，即 LID 9.2 rps，其他每任务 0.16 rps；600 秒；seed 300301、300302、300303。负载不是探索后挑出的最高共同通过点。** 这是草案中点的设计选择，待批准，没有确认性结果。若该点不具备准入证据，结论 NO_GO/EVIDENCE_INCOMPLETE，不改成其他 level 或模式。

三 arm：S-only（全部 LID→S）；fixed10（原 `choose` 的 uniform 抽样，名义 10% LID→G，90%→S，实际比例全部报告）；冻结 Dynamic。其余任务路径始终 Text→RoBERTa、Image→ConvNeXt、OCR→PARSeq、ASR→S、VQA→G。G、S 与背景模型三 arm 始终驻留各 1 副本，不能给 S-only 移除 G 来节省显存。Dynamic 名义 5% route 仅用于 READY 结构检查，不作为实际分流比例；实际选择只由原 decide 决定，令牌率和其他参数不缩放。

每组三 arm 共享同一 warmup/formal 请求总量、样本、计划偏移、task、uniform、worker 模型环境、日志覆盖与预算。确认性顺序为：第 1 组 S/fixed10/D，第 2 组 fixed10/D/S，第 3 组 D/S/fixed10（事先拉丁方）。热态差限用于全组三 arm，不只挑一对。全部组保留，不按 ΔU 或 G 路由比例早停或换 seed。

统一新 harness 的实现尚未交付，不可直接复用旧 launch/main：旧脚本绑定历史 OUT，会覆盖或向历史目录写入。后续须仅在新执行目录实现观察 sidecar，collector 原 AST 保留；实际持锁后时点、commit、harvest、完整 pending 与决策前后状态覆盖三 arm。S/fixed 也按同一观察规则执行与记录，在没有动态选择时记录 estimate/控制路径；不能以省略观察降低其测量开销。观察差异只因路由选择逻辑产生，开销计入 E2E，不额外加第二遍。统一日志/语义导出的设计、实现 SHA-256 与 CPU replay/collector 核验属于 P04，不能宣称现在已有三臂 harness。

|评价项|冻结口径|
|---|---|
|延迟|scheduled→reader successful response 为 E2E，含 host、路由及日志开销；正式 cohort 包含排空成功的 mean/P95，六任务分别报告。另报 runtime、schedule_delay、service、transport-inclusive worker wait 与完整分解；pooled P95 只作诊断，不代替逐任务 P95。|
|吞吐量|planned/offered/sent、窗口内成功、排空成功、失败/未完成分别计数；正式分母固定 600 秒。窗口内 throughput 不含 drain；cohort throughput 与包括 drain 时长 throughput 单列。|
|队列|原 B(t)、host-enqueued outstanding、S/G/task pending、积分 bin、peak/end/post-drain、tail 指标和等待分布分列；真实 worker receipt/waiting queue 没 ACK 仍 UNKNOWN。不得把 host pending 叫纯等待队列。|
|质量|沿用完整冻结 catalog 的已测量路径准入与政策；Text accuracy≥0.925，Image≥1.0，OCR exact_match≥0.96，ASR WER≤0.042181 是历史准入值。当前负载子集分数不重新拟合准入。VQA/LID 没有 numeric accuracy 阈值，报告错误、闭集、语言、sample_id、correct/WER/CER 观测分母；不能把推理成功当正确。|
|语义证据|原 `evaluate` 的 explicit `valid_closed_set_output=False` 门控原样保留，缺失不是 True。Qwen LID 输出 prediction/reference/correct，但原 stage2 sanitizer 丢弃类别且 worker 没 flag；未来 sidecar 仅从已收到响应导出规范化语言类和按原语义定义的诊断，不让答案进入策略，不改 handler/评分。不能新增基于小样本准确率的排除阈值；政策与 explicit flag 门控的适用性须 P07 审核。缺失字段保留 UNKNOWN，确认性证据不充分。|
|背景干扰|五个非目标任务保持同输入、到达、路径、模型；按同 request_id/sample/offset 和共同实际路由比较 service/wait/E2E、ASR/S 与 VQA/G 上下文；报告 S/G native overlap、背景 queue、热态/功率/CPU/RAM 和路由开销。分层重叠是描述性关联，不能当单一 GPU 干扰因果量。|
|覆盖|每个 formal arm 六任务成功观测各≥72（原 manifest 要求）；Dynamic 目标 G 与 S 路径正式成功各>0；prime/warmup 不算。不足不补跑、不合并种子凑数。|

原 metrics 用 completion<end 计窗口内成功，而原 utility arithmetic 用 completion≤end。本协议分别保持各函数原口径，额外输出 exactly_at_end 计数；若出现边界事件，显式解释差别，不悄悄修改 comparator 或吞吐定义。延迟 P95 在效用中沿用原 numpy percentile 实现；其他显示函数分位计算方法保留名称与版本，不混写。

## 8. 合法 U、ΔU 与证据门槛

沿用 Module20A 定义及 Module28 `evaluate`，不修改正式优化器。Q 是硬约束，wQ=0，Q_norm=NULL 不填缺失质量。U=−0.25T−0.30L−0.30P−0.15M。

T=1−min(1, completed/offered, completed_in_window/(sum(offered_QPS)×600))；L、P 按冻结 offered rate 权重分别加权六任务 mean_E2E/ref 与 P95_E2E/ref；M=安全因子×max(postwarm_idle_mean,postmeasurement_idle_mean)/B4。refs、全部权重和来源哈希见 JSON，不用本次观测重新归一化。

U 合法需 valid_session、全 cohort 成功/覆盖、READY capability、原已测单副本/容量结构、路由守恒、原质量/语义路径、safe 及 sampled peak 显存≤原预算、系统和每 worker 不 unstable、原恢复确认、无错误且清洁释放。`recoverable_congested` 只有明确恢复且原其余约束都通过才可能合法。失败 U=NULL，不能用完成子集算正常 U，不填为 0，不排序；原始算术或延迟只可标 CENSORED/诊断。

ΔU 只在两个单 arm 原合法性及预注册输入/部署/背景/热态匹配均通过后形成合法对比；三臂正式证明另需全组覆盖≥72和 Dynamic 两路径非零。缺失/不合法 arm 不能被删除以制造“最强对照”。每组明确报告 D−S 和 D−fixed10；主要预定效应为 D−max(U_S,U_fixed10)，两对照必须都合法。这是预先固定的严格 benchmark，不是在比例网格里事后挑 best-fixed。

探索阶段只以稳定性/合法性决定分支，U 与 ΔU 如计算仅作明确标记的探索诊断，不能驱动条件选择，也不能转入正式优化器。S-only unstable 而 Dynamic 非 unstable 仅为有限窗口过载救援候选；S-only U=NULL、合法 ΔU=NULL，即使 Dynamic 可恢复也不证明效用优越。

若将来单独批准确认性阶段，沿用原三组全部 ΔU>+0.01，且三组主要配对效应 Student t 双侧 95% 区间下限>0 的原门槛。全部三组都合规才可能 GO；不改成只看平均值。报告 n=3、df=2、全部单组效应、均值、区间及近似正态假设；三组同号的分布无关单侧符号检验最小 p=0.125，不能声称一般 5% 显著证明。60 秒 block/500 draws 仅条件敏感性，不是独立 GPU CI。没有跨模型池、全局最优固定比例或正式优化器创新结论。

## 9. 预算、异常、停止与审核

`budget_and_stop_rules.md` 与 `gate_criteria.md` 为执行约束；预算均待批准。两模式各按草案最多 22 会话，因此探索上限 44 会话/26400 秒 formal（7 小时 20 分），并非已经获准的原 22 次预算。确认性 9 会话/5400 秒 formal（1 小时 30 分）单独批准；两者合计最多 53 会话/31800 秒 formal（8 小时 50 分）。新增每 attempt wall cap 3600 秒、prime cap 300 秒以及探索 44 小时/确认性 9 小时总 wall cap 为保守待批准提案，覆盖准备/冷却/加载/排空/关闭；不是从历史测量推定的收费时长。

OOM、预算/显存超限、foreign GPU process、worker/任务/语义路径/日志错误、缺少必要时间戳、排空超时、热态无法达到、关闭/释放失败均按全局异常规则停止，保留全部预定请求和失败记录，不重试。没有新增温度运行中报警值、准确率 SLO、throughput SLO 或延迟 SLO；原记录不能确认的数值不伪装成已有阈值。正常完成且安全可继续的科学 unstable 不跳过其已注册配对另 arm。

本版对草案的可见改动：加入在全部粗网格点的到达模式扫描及其独立种子，探索上限扩大为两模式各 22；确认性 load 改为现在固定 0.80 steady，而不是观察后选最高共同通过点。两处均为待批准提案，目的为区分扫描轴及避免探索条件进入正式收益证明。新增复核点规则仍只用原稳定性/合法性，不用动态收益。

七份文件提交后立即停止。本版不包含 GPU launcher，不自动创建实验、运行三臂、重调策略或集成优化器。审阅若修改任何待批准值，须在任何新结果前形成新版本并重新冻结七份文件哈希，保留本版，禁止追溯修改历史。
