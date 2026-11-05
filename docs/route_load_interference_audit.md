# 路由、负载与 GPU 共驻执行干扰审计

2026-11-02。**当前 GSDeploy 主实验使用的 `calibrated_replay_v2` 没有随G/S路由、实际负载或其他驻留模型活动而改变逐输入执行服务时间的能力。它有逐输入差异、合法副本画像和共享 FIFO，有经过两个留出条件验证的全局共驻校准，但没有实现 `s_G=f(D,λ_G,λ_S)` 的动态执行干扰函数。**

结论针对本阶段配置实际进入的预测路径。仓库还有独立kernel、twoFIFO及全过程预测研究工具，存在相关文档不代表这些模型已接入本轮Joint/连续算法的正式效用oracle，也不继承其验证资格。

## 1. 审计文件和证据分类

审计入口为 `prediction.predict_scenarios → predict → performance.resolve_service → calibration_v2.replay_service/input_service_ms`，资源走 `replay_auxiliary`，请求映射走 `trace.expand_routes + routing.Router`。搜索界另走 `surrogate_bounds` 和 `milp.floor`。

| 路径/模式 | 实际含义 | 可否作为本轮正式性能 |
|---|---|---|
| `calibrated_replay_v2` | 固定逐输入standalone中位数×全局校准比率，共享FIFO；整板能耗/显存回归转移 | 当前原门禁接受时可评价冻结U；不是每个组合的实测 |
| `calibrated_surrogate` | 较早固定路径均值×全局系数 | 不表示动态路由干扰；本轮未切换 |
| `task_replica_proxy` | 已测同任务/副本平均服务，转移到FIFO | 非混合负载干扰模型；本轮未切换 |
| `profiled` | 按完整部署、路由、负载、环境身份取对应共驻服务画像 | 可以有特定完整条件证据；不是未测连续路由的通用函数 |
| `independent` | 常服务独立副本 | 独立服务假设；本轮未切换 |
| `replay_transfer_v1` | 有自身身份/工作负载/评分合同的迁移 | 两份已有ASR配置当前被拒绝，不能借用旧资格 |
| `surrogate_bounds` / `floor` | 同一冻结预测器成本的数学下界 | **不能当GPU实际服务或测量** |
| `route_load_replay_v1` | `coupled_mode.py` 明确要求 `status=UNIMPLEMENTED` | 不可；完整预测返回未知 |

`coupled_mode.unknown_service()` 返回 `None` 与 `ROUTE_LOAD_MODEL_NOT_IMPLEMENTED`；schema错误返回 `UNSUPPORTED_ROUTE_LOAD_SCHEMA`。其零服务、零压力、零显存、零能耗仅是非负成本松弛，关闭standalone回退和capacity witness。相关测试即使注入合成完整oracle，也不构成生产动态预测器。

本轮没有更改这些路径，也没有使用 `_allow_unvalidated` / `_calibration_validation` 绕过证据门禁。
`profiled` 的画像类型还可以是显式 `synthetic_fixture`；只有类型及来源确为真实测量的记录才计作GPU测量，合成画像只用于正确性测试。

## 2. 当前服务模型依赖哪些变量

[`calibration_v2.py`](/root/autodl-tmp/generalist-specialist-project/gsdeploy/calibration_v2.py:313) 的逐输入服务为：

\[
\widehat s(m,t,n,i)=\mathrm{median\_standalone}(m,t,n,i)\times1.040352322436219.
\]

`prediction.py` 再按原 `n/capacity_factor(n)` 和场景倍率缩放后执行FIFO。v2要求合法校准状态的capacity factor等于n，因此正常主场景下这一步不会引入新的跨模型干扰。

| 检查项 | 当前情况 |
|---|---|
| 模型/任务/输入 | 是，使用相应冻结逐输入服务，不是统一均值 |
| 副本数n | 是，选择n对应画像；缺路径不借用n1 |
| 共驻模型集合D | 用于合法性、驻留、能耗和队列；**固定(m,t,n,i)服务不随其他驻留模型改变** |
| Batch/精度/执行配置 | 通过冻结模型/画像/环境身份间接绑定；不接受任意新Batch的运行时服务函数 |
| 每模型实际接收请求数/实际λ | 影响分配与队列；**不作为逐输入服务函数自变量** |
| 任务路由比例r | 影响Router、rho和资源期望；**不进入逐输入服务函数** |
| GPU计算/访存共享竞争 | 全局系数吸收开发条件的平均效应；没有随活动或重叠变化的计算/访存竞争项 |
| 路由改变后的重新预测 | 会重新算完整请求映射、FIFO和U；服务仍为同一冻结函数 |
| FIFO使用什么服务 | 当前被分到模型/副本的具体输入服务，副本内跨任务共享FIFO |

注意 `replay_service()` 返回完整输入池均值供rho使用；真正FIFO循环再调用 `input_service_ms()`，不会给每条请求都填pool mean。`replay_input_ids()` 在Router之前按任务和seed固定输入序列，因此改路由不会偷偷更换请求输入。

同一固定D中：`λ_G,t=q_t(1-r)`、`λ_S,t=q_t r` 成立；G的**总**负载还要加未改变的其他任务。本文路由曲线不会把G总QPS误写成单任务QPS。

## 3. 实验 C：固定部署的受控验证

在原seed42、输入序列、四任务到达率、环境合同、副本和驻留集合不变下，分别扫描：

- G2+OCR，只改变OCR路由；其他任务给G2。
- G2+ASR，只改变ASR路由；其他任务给G2。
- G2+image+OCR+Text，只改变Text；image/OCR维持给S，ASR给G2。

每条21点，r=0,0.05,…,1，共63次完整预测。每点保存完整567条请求的输入ID、模型、副本、服务、等待、起止时刻和映射SHA。独立事件观察器逐任务重算Mean/P95、窗口完成数、排空数及 `L=W+s`，全部与原预测一致。另9个激活对照也重建账本，总计72份核对。

- [原始逐点JSON/事件](/root/autodl-tmp/generalist-specialist-project/artifacts/optimization-difficulty-20261102/controlled)
- [路径级服务、等待及实际QPS](/root/autodl-tmp/generalist-specialist-project/artifacts/optimization-difficulty-20261102/controlled-paths.csv)
- [每模型实际负载与队列](/root/autodl-tmp/generalist-specialist-project/artifacts/optimization-difficulty-20261102/controlled-models.csv)
- [系统U/Mean/P95与六项贡献](/root/autodl-tmp/generalist-specialist-project/artifacts/optimization-difficulty-20261102/controlled-points.csv)
- [服务不变性核验](/root/autodl-tmp/generalist-specialist-project/artifacts/optimization-difficulty-20261102/service-invariance.json)

### 服务与等待必须分开解释

固定路径的全输入池平均服务在全部路由点**严格相同**。例如OCR：G2约8124.51ms、OCR S约435.15ms；Text：G2约520.72ms、Text S约600.33ms。该结论也由服务函数的参数依赖直接支持，而不只是稀疏采样看起来平直。

实际分配输入均值则可能改变。OCR S在r=0.5时被分配子集的平均服务为381.02ms，在r=1为435.73ms；两者使用同一逐输入表，差异来自输入子集，不能解释成负载导致GPU减速。r=0的零请求S平均服务/等待留空，不能填0并画成“执行服务为零”。

![固定部署的服务曲线：实线完整输入池，虚线实际分配子集](/root/autodl-tmp/generalist-specialist-project/artifacts/optimization-difficulty-report-complete-20261102/06-route-service.png)

G2+OCR对照中：

| OCR给S比例 | G2总计划QPS | OCR S计划QPS | G2 rho / S rho | G2平均等待ms | S平均等待ms | 系统Mean / P95 ms | U |
|---:|---:|---:|---|---:|---:|---|---:|
| 0 | 0.299344 | 0 | 0.725069 / 0 | 8509.13 | 无请求 | 10951.81 / 28466.83 | −8.533955 |
| 0.5 | 0.261926 | 0.037418 | 0.421066 / 0.016282 | 2409.50 | 0 | 3678.94 / 11505.95 | −3.215756 |
| 1 | 0.224508 | 0.074836 | 0.117063 / 0.032565 | 51.03 | 6.83 | 541.92 / 1340.53 | −0.413332 |

这证明当前预测器具有**路由—共享FIFO排队耦合**。它不证明具有**路由—GPU执行服务干扰耦合**。平均rho<0.8也不排除长OCR请求引发队头阻塞；G-only的等待很大，不能说整个系统低压。

![固定服务律下的等待变化](/root/autodl-tmp/generalist-specialist-project/artifacts/optimization-difficulty-report-complete-20261102/07-route-waits.png)

63点中OCR、ASR切片最佳均是S端点；Text粗扫最佳r=0.65，U=−0.338344410，优于同部署两个端点，但没有本轮粗扫之外的最优性证书。此前独立DE已在该Text部署找到−0.325372398；本轮粗扫结果不能覆盖或否定历史更好候选。

对到达率的控制实验无法在现有正式预测门禁内执行：原负载×0.5/2/4、ASR/OCR×3，以及仅±1%均返回 `V2_REPLAY_INPUT_CONTEXT_OR_WORKLOAD_CHANGED`。拒绝的完整配置、原因单独归档；没有在失败后绕过门禁绘制服务—负载“预测曲线”。已有Poisson不同seed只代表自然轨迹变动，不是受控突发或新到达率验证。

## 4. 实验 D：驻留和能耗能解释什么

v2显存使用 `max(additive×1.2216725559481743, 开发条件部署峰值)`，固定D下不随r改变。零流量模型仍保留权重驻留和对应worker/启动资源状态。激活或删除模型必须重新预测；本轮均如此。

能耗为整板非负回归，系数 `[72.78783325893261,126.5546063226936,0]`，另每worker启动能量6773.594640927618J。第二特征是计划QPS×原standalone均值的总忙碌需求，不是从实际GPU执行重叠重建功耗。即使两个比例产生相同有限请求映射，期望资源项和rho也可能不同，不能将其完整U自动视为相等。

原赢家额外驻留Text、仍保持Text给G，延迟/质量/吞吐完全相同，却损失U=0.043862258，其中显存0.043165386、能耗0.000696872。纯Text或连续Text可收回部分收益，尚未收回全部驻留成本。ASR的激活也有同方向的六项分解；详见[难度报告的实验D](/root/autodl-tmp/generalist-specialist-project/docs/gsdeploy_optimization_difficulty_diagnosis.md)。

因此当前数据支持“某些候选受到显存固定成本约束”。它不支持将差值归因于动态共驻干扰、GPU利用率竞争或Specialist饱和队列；OCR全任务rho约0.0326、ASR约0.0510，恰好说明这些S当前没有形成双边高压。

## 5. 实验 E：覆盖范围和实际验证强度

### 输入和路径画像

冻结输入池：Text32、image64、OCR32、ASR32。按当前catalog合法正副本状态及任务支持枚举，共70条 `(model,task,n)` 路径，其中61条具有完整冻结逐输入表，9条缺失：

- G1：Text n3/n4、image n4、OCR n4。
- G2：Text n4/n5、image n5、OCR n5。
- G3：OCR n4。

逐条详情及原始输入身份类型见 [path-coverage.csv](/root/autodl-tmp/generalist-specialist-project/artifacts/optimization-difficulty-report-complete-20261102/path-coverage.csv)。部分OCR历史数据使用 `historical_profile_input_binding`，不是新补出的request/protocol哈希，也不升级为协议完全等价；原始绑定与原验证限制保留。

### 实际共驻校准不是整个搜索域的密集测量

spec包含8条开发条件、2条独立留出条件，**所有条件到达率都是相同四任务负载**，窗口均600s：

| 类型 | 实际部署/路由 | seed | 说明 |
|---|---|---|---|
| 开发8条 | G2-only；四S-only；G2+Text、G2+ASR的S比例1和0.75；G2+OCR、G2+image的S比例0.75 | 42 | 用于拟合全局服务比率与整板资源 |
| 留出1 | G2+image+OCR，image给S0.75、OCR给S1 | 46 | 服务相对误差8.39%，U绝对误差0.009697 |
| 留出2 | G2+image+OCR，image给S1、OCR给S0.75 | 47 | 服务相对误差1.59%，U绝对误差0.004604 |

两留出通过原阈值，排序一致；原spec明确“无physical all-pool optimality或statistical coverage claim”。G1/G3、多副本以及大量其他D/r虽可在已知路径下进入数学转移预测，并没有因此成为独立实测验证过的共驻组合。

[measured-calibration-conditions.csv](/root/autodl-tmp/generalist-specialist-project/artifacts/optimization-difficulty-report-complete-20261102/measured-calibration-conditions.csv)保存逐条件真实packet、run来源与完整比例；[predictor-evidence.json](/root/autodl-tmp/generalist-specialist-project/artifacts/optimization-difficulty-report-complete-20261102/predictor-evidence.json)保存原验证误差与环境身份。本轮没有重新拟合、重签合同或读取留出服务来填新预测。

### 可用域表

| 模型组合/副本 | 任务与负载 | 路由范围 | 共驻服务 / FIFO | 正式联合U资格 |
|---|---|---|---|---|
| G2-only n1 | 四任务原负载 | 全给G2 | 原画像全局转移 / 共享FIFO | 通过原硬检查时可用 |
| G2+单S n1 | 四任务原负载 | 对应任务闭Simplex含0/1 | 稀疏实测校准后转移 / 重新排队 | 通过原硬检查时可用；非密集路由测量 |
| G2+image+OCR n1 | 四任务原负载 | 支持模型之间比例变化 | 两条混合留出条件 / FIFO | 冻结预测可用；非动态服务验证 |
| 四S-only n1 | 四任务原负载 | 各S执行本任务 | 开发条件及转移 / 独立模型队列 | 通过原硬检查时可用 |
| 3/5/7池与其他合法副本 | 四任务原负载 | 每任务的已知路径Simplex | 已知路径转移 / 按副本FIFO | 条件可用，缺路径/资源不授资格 |
| 高/低/热点/任务子集 | 改变到达率或任务集合 | 不适用 | 原门禁拒绝 | 不可作当前正式比较 |
| 旧ASR transfer配置 | 原配置指定负载 | 原配置指定范围 | `TRANSFER_SPEC_OR_CODE_CHANGED` | 当前不可用 |
| `route_load_replay_v1` | 无验证工作负载 | 无正式域 | 未实现 / 无合法服务输入 | 不可用 |

机器可读完整表见 [predictor-coverage-table.csv](/root/autodl-tmp/generalist-specialist-project/artifacts/optimization-difficulty-report-complete-20261102/predictor-coverage-table.csv)。

当前数据确实在负载维度很窄，也缺动态服务耦合；这**限制了研究问题的代表性**。但尚无高负载合法对照，不能因果断言“窄负载是低K简单的唯一原因”。有效域过滤、强目标下界和驻留惩罚已有直接解释，缺失压力场景另列为未回答问题。

![门禁覆盖和拒绝原因；A/C重复探测保留为探测次数](/root/autodl-tmp/generalist-specialist-project/artifacts/optimization-difficulty-report-complete-20261102/09-coverage-gates.png)

### 本环境的真实Profiling状态

只读 `nvidia-smi` 调用遇到 `Permission denied`；记录见 `hardware-access.json`。没有建立可用GPU测量环境，没有启动模型、改变设备权限或运行新Profiling。新增真实GPU测量数为0。当前所有服务/等待曲线明确标记为冻结预测，历史实际校准条件独立列表，不拿模拟值填补负载覆盖。

## 6. 最小补齐清单与验收条件

下一阶段先做**测量设计与预测资格扩展**，不需要先挑GP-BO/MADS，也不预设真实GPU一定存在显著路线干扰。建议分两步，以下是待执行方案，尚无测量结果。

**先识别动态效应。** 固定G2+OCR和G2+Text两种共驻部署、各n1、同硬件/输入/Batch/解码设置/后台四任务，分别测S比例0、0.5、1。原负载和一个预先声明的更高负载各一档，每条件3个独立到达seed，共 **2×3×2×3=36个共驻运行**。较高档需要根据真实安全容量与排空检查预先确定，不能把尚未获预测门禁的2倍负载直接写成正式已验证域。

保留r=0时S权重驻留，另以G-only和S-only的匹配输入/执行画像做对照，分开“闲置驻留影响”和“活跃竞争影响”。原standalone画像只有在硬件/软件/输入协议完全匹配时才复用，否则补采。单个S不能覆盖四任务；它的standalone对照只评本任务执行，不冒充四任务系统U。36个运行只是识别矩阵，不包括可能需要补采的standalone及最终盲测。

每个请求至少记录：输入ID与请求/协议SHA、模型/副本、计划到达、入队/出队、真正执行起止（排除队列等待）、完成、成功状态、批次/精度/执行配置；同时记录GPU内存峰值、功率积分/启动能量、模型活动重叠和设备/驱动/checkpoint身份。若采集GPU计算/访存/带宽指标，要明确采样口径，不能用系统Mean反推执行干扰。按相同输入比较G/S服务随另一模型活动的变化，避免输入子集均值造成伪减速。

**再验证可转移性。** 若上述数据显示稳定服务变化，才提出带D、n、路由导出的每模型/每任务计划负载及活动特征的候选服务模型；保留独立seed、部署或比例作为盲测，不用拟合数据自证。至少报告逐模型服务分布、Mean/P95 FIFO、六项U和候选ΔU符号/部署排序误差。热点评价再追加ASR和OCR不对称负载，多副本至少补充最有潜力的n2共驻条件。此时重新建立正式门禁，而非放宽旧阈值/重签旧数据。

建议验收目标是回答“是否存在可复现、足以影响部署排序的route/load执行效应”，而不是确保新预测模型或连续算法一定胜出。若服务变化很小或路由收益仍不能覆盖激活成本，保留低K Joint，把工程工作集中在已确认的构列、界限和会话建立成本。

## 7. 已确认与未确认结论

- **已确认：** 冻结服务函数没有实际路由负载自变量；占位模式未实现；63点池服务不变而FIFO变化；驻留Text的固定成本可逐项解释；实际校准仅一个四任务到达率向量、两个留出条件；新负载门禁失败。
- **已有局部证据：** 连续Text路由改善固定部署尾延迟和效用，部分独立轨迹有收益；未提高Joint系统最佳值，既有比例跨轨迹不总稳定。
- **尚未确认：** 真实GPU服务随λG/λS变化的强度、强干扰时纯路由是否仍最佳、高负载是否改变部署排名、连续全域最佳方案。

因此当前研究优先补齐C/D方向的画像与独立验证；高K的A方向有清楚计算瓶颈，可作为独立工程工作。不能用零数学下界、全局校准系数或队列等待下降，声称已经验证了路由相关GPU动态执行干扰。
