# 有文献依据的六项归一化重设计

原设计日期：2026-11-01。2026-11-03 **已完成软件与正式研究入口切换为 `six_term_reference_v2`；真实参考值仍待独立测量。** 接线、当前状态与冻结流程见[正式切换记录](utility_normalization_switch_20261103.md)。下文保留原设计论证及建议用语。

建议保留质量、有效吞吐、平均延迟、P95、显存和能耗六项，采用文献中的两种成熟方法：**固定参考范围归一化**与**相对固定参考值归一化**。把分母改为可追溯的质量锚点、实验 SLO、GPU 容量或实测基线能量。每项均有具体来源，下文明确区分原文方法与本项目的适配。

重要的新证据是：OECD/JRC 手册不只支持“做无量纲化”这一动机，而是明确给出了 Min–Max 和参考比值的公式；μ-Serve 也直接使用“尾延迟 / SLO”。因此可以为六项的归一化运算提供文献依据，同时诚实说明整套六项组合仍是我们针对问题的设计。

## 1. 共用的方法论依据

**[R1] OECD / European Union / JRC，2008，*Handbook on Constructing Composite Indicators: Methodology and User Guide*.**

- [DOI](https://doi.org/10.1787/9789264043466-en)，[官方全文](https://www.oecd.org/content/dam/oecd/en/publications/reports/2008/08/handbook-on-constructing-composite-indicators-methodology-and-user-guide_g1gh9301/9789264043466-en.pdf)。已下载并核验全文。
- §5.3，印刷 p.85 / PDF p.87：`(x-min)/(max-min)`；也讨论用初始时期冻结的范围，以及新数据可能超出 `[0,1]`。
- §5.4，印刷 pp.85–86 / PDF pp.87–88：`x/x_ref`，或 `(x-x_ref)/x_ref`；参考可为外部基准，不要求等于候选集合的当次极值。
- §6.10，印刷 p.103 / PDF p.105，式 (31)：归一化指标的加权求和。
- §1.6，印刷 pp.31–33 / PDF pp.33–35：等权很常见，但权重仍表达价值判断；相关指标可能重复计权。
- §1.7，印刷 p.34 / PDF p.36：应分析归一化、权重和聚合方法的敏感性。

这是综合评价的方法论文献，应用背景是国家指标。我们将数学方法用于部署评价，**不声称该手册规定了推理系统的六个指标或参数**。

**[R2] Krishnan, 2022，*Past efforts in determining suitable normalization methods for multi-criteria decision-making: A short survey*. Frontiers in Big Data.** [DOI](https://doi.org/10.3389/fdata.2022.990699)，[公开全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9433668/)。

该综述比较了多种归一化与适用情形，显示方法选择和比较准则可能改变排名。它用于说明为什么本项目需要做归一化敏感性，而不是证明某个方法“普遍最优”。

以下 PDF 页码均指本次实际保存的 PDF 物理页码；HTML 来源使用节号。原文摘录和文件 SHA256 见本地审计目录 `artifacts/utility-normalization-redesign-20261101/evidence.json`。

## 2. 推荐的六项公式

设 `alpha_t=lambda_t/sum(lambda)`。全部成本越小越好：

```text
Q_v2 = sum_t alpha_t * (loss_t - a_t)/(z_t - a_t)
T_v2 = 1 - N_good / N_offered
L_v2 = sum_t alpha_t * mean_e2e_latency_t / D_t
P_v2 = sum_t alpha_t * p95_e2e_latency_t / D_t
M_v2 = owned_sampled_peak_memory / B
E_v2 = J_episode / J_episode_ref

U_v2 = -(Q_v2 + T_v2 + L_v2 + P_v2 + M_v2 + E_v2)/6
```

`a_t` 是预先冻结的可信最好质量损失，`z_t` 是可信最弱质量损失，且 `z_t>a_t`；`D_t` 是固定实验延迟目标；`B` 是固定可用显存预算；`J_episode_ref` 是固定参考部署在相同工作负载协议下测得的基准能量。`N_good` 的精确定义见下文。所有参考值在评价新候选之前冻结，对同一任务所有模型、部署和路由相同。

|项|归一化的含义|直接来源及适配边界|
|---|---|---|
|Q|0 对应可信最好质量，1 对应可信最弱质量；统一为“一个已冻结质量跨度”的损失|R1 §5.3 的范围归一化；R3 Cascadia §3.3 的 utopia–nadir 质量范围。我们将“质量不足的约束惩罚”改为连续质量成本，不引入绝对质量门槛|
|T|未能按实验延迟目标交付的输入需求比例；0 为全部按时完成|R4 MuxServe 的 SLO attainment 请求比例；R5 USHER 的 goodput 目标。我们取其补数作为损失，并保留固定 offered cohort|
|L|平均端到端时间占实验延迟目标的比例；1 表示均值等于目标|R1 §5.4 的参考比值；R6 Proteus、R5 USHER 的基线倍数 SLO。平均值的比值成本是二者在本项目中的组合，不冒称论文提出了相同总效用|
|P|P95 端到端时间占同一目标的比例；1 对应经验 P95 等于目标|R7 μ-Serve §4.2 **直接以 P99/SLO 归一化**。我们保留当前 P95，因此是分位点适配，不能写成复现它的 99% 服务保证|
|M|占用可用 GPU 显存容量的份额；1 是预算边界|R8 DRF §4 / Algorithm 1 的 `usage/resource_capacity`。我们将资源份额用于 GPU 进程 owned 采样峰值，而非搬用 DRF 的公平分配算法|
|E|完成同一评价 episode 所需 GPU 能量相对固定参考部署的倍数；参考为1|R9 Zeus Fig.1 的基线能量归一化，结合 R1 §5.4；R10 MLPerf Power 的时间对齐、边界一致和能量积分原则。Zeus 是训练论文，移植的是参考比值；episode 边界为本项目适配|

### 2.1 Q：改为冻结质量范围，而非直接混合四种原始损失

原生指标继续使用 `1-EM`、`1-accuracy`、CER、WER；先按各任务原协议聚合，再做归一化：OCR/ASR 仍分别累计编辑错误数 / 累计参考字符或词数，不改成逐请求比率的简单均值。

在**开发阶段的固定可信路径集合**上，为每任务得到 `a_t=min loss`、`z_t=max loss`，同时保存实现这两个锚点的路径、检查点、数据划分和指标统计。这个集合包含已验证可执行的通用与专用路径，不包含故意加入的坏模型或不支持的路径。在确认集上比较候选时保持锚点不变；新增候选也不自动扩大范围。

**[R3] Cascadia，ICLR 2026，§3.3 / PDF p.6，附录 D / PDF pp.15–16。** 原文以最大模型和最小模型定义质量的 utopia / nadir，质量短缺除以两者质量之差，解释为让惩罚无量纲。论文使用：

```text
max(0, (q_min - Quality)/(quality_best - quality_worst))
```

我们的修改是：四任务模型大小不必对应质量顺序，故用固定可信路径的开发集质量来定端点；`(loss-a)/(z-a)` 衡量相对最好锚点的连续损失，不加绝对精度门槛。R1 §5.3 则直接支持这一范围变换。

不自动截断：超出最好锚点时 Q 的任务项可小于0，劣于最弱锚点时可大于1。R1 对冻结历史范围后超界已有讨论。这样保留实际改善/退化；若硬裁剪，超过最弱锚点的不同差方案将被打成相同分数。

若 `z_t=a_t`，范围归一化无定义，必须报告该参考协议不可用，不能偷偷加 epsilon 或删掉质量权重。若范围相对估计误差很小，点估计仍可计算，但其敏感性必须报告；应增加独立样本来改善证据，不能通过选择更差的锚点人为放大范围。

**敏感性对照**：R6 Proteus（PDF p.8）按家族最佳准确率归一化准确率。对应的另一种成本可写成 `1-(1-loss)/(1-a)`。它不依赖最弱模型，却表达“相对最佳正确程度”的偏好；主设计选择范围法以比较不同任务可实现的质量跨度，同时报告该替代法是否改变结论。对 OCR/ASR 使用 `1-loss` 是本项目的方向转换，它可为负数，不将其冒称常规定义的 accuracy。

### 2.2 T：采用 SLO 有效吞吐的供需比例

**[R4] MuxServe，ICML 2024，Metrics / PDF p.7**：SLO attainment 是在 latency target 内完成的请求百分比；同时报告聚合吞吐。

**[R5] USHER，OSDI 2024，PDF p.2 / p.4**：[论文页](https://www.usenix.org/conference/osdi24/presentation/shubha)。目标包括在延迟约束下提高 goodput；实验用单请求平均推理延迟的2倍设定 SLO。

```text
formal cohort = 计划到达时间落在正式窗口内的全部请求
N_good = 此 cohort 中，输出成功返回且
         completion_time - scheduled_arrival_time <= D_task 的请求数

goodput = N_good/H
offered_rate = N_offered/H
T_v2 = 1 - goodput/offered_rate = 1 - N_good/N_offered
```

这里“成功返回”是执行/传输成功，输出质量由 Q 评价，不加正确率准入门槛。拒绝、失败、未按时完成均不能计为 good。窗口结束前到达、在窗口之后但自身 deadline 之前完成的请求仍可计为 good，修正旧 T 将它们一律当成窗口内吞吐损失的边界效应。

测量必须覆盖正式 cohort 的最后一个 deadline；若参考值未满足排空时限、或运行提前中断且缺失评价所需证据，应标记不可比较，不利用早停“节能”。已确认超时/拒绝算 bad；遥测丢失不能伪装成已确认超时。固定完整 `N_offered`，不把失败请求从分母删除。

这是六项中**指标语义也发生变化**的一项，名称应从“窗口内吞吐损失”改为“SLO 有效吞吐损失”。T 是实际 cohort 的整体比例，其任务权重等于实际 offered 计数占比；Q/L/P 的 alpha 则继续是预先指定的请求率占比，报告二者的有限样本差异。

T 的完成率、L 的平均值和 P 的尾部分位各有侧重，但都受延迟影响，存在相关与重复计权。不能因为三项分别有来源就声称它们互相独立。

### 2.3 L / P：用实测参考路径生成实验 SLO

**[R6] Proteus，ASPLOS 2024，§6.1.2 / PDF p.8**：[DOI](https://doi.org/10.1145/3617232.3624849)。每个模型家族选择能在 CPU 运行的最快变体，以最小 batch 的画像延迟的2倍设定家族 SLO，并分析 SLO 敏感性。

R5 USHER（PDF p.4）也明确使用 NVIDIA V100 上单请求平均推理延迟的2倍。**[R11] AlpaServe，OSDI 2023，§3 / PDF p.6，§6.2 / PDF p.11**：[全文](https://www.usenix.org/system/files/osdi23-li-zhuohan.pdf)，扫描独立执行延迟的倍数，并使用5倍的默认实验点。这说明倍数需要说明，2倍并非普适定律。

建议为每任务预先指定一个参考路径 `r_t`，优先使用当前已保留的专用路径，在同一 GPU、相同输入/输出协议、batch=1、无竞争的参考运行中测端到端均值 `s_ref,t`：

```text
D_t = k * s_ref,t,  主分析 k=2；预先声明 k∈{1,2,4,8} 的敏感性
L_v2 = sum_t alpha_t * mean_e2e_t/D_t
P_v2 = sum_t alpha_t * p95_e2e_t/D_t
```

借鉴 Proteus 的“同一模型家族共用目标”，我们改为**同一任务所有可替换模型共用同一参考路径产生的目标**。不能每个候选除以自己的单模型延迟。选择专用路径是本项目可复现的基线政策，不是文献证明了专用路径应优先；需要记录并冻结该政策。

不把旧 handler/service-time 画像直接当作 e2e 参考，也不把 LLM 的 TTFT/TPOT 当作整请求时间。参考路径、GPU、输入池、输出长度或预处理改变时需新参考版本。

**[R7] μ-Serve，USENIX ATC 2024，§4.2 / PDF p.11**：[论文页](https://www.usenix.org/conference/atc24/presentation/qiu)。原文：“We measure the 99th-percentile (p99) latency ... and normalized it to the SLO latency.” 这是尾延迟除以 SLO 的直接来源。我们使用 P95，是沿用本项目尾延迟指标后的适配，且“各任务 P95 加权”不等于全局 P95。

延迟比值保留大于1的值，不在目标处裁平。T 表示是否达标，L/P 表示耗时大小；裁平 L/P 将丢失超时程度。P95 的样本量与区间必须报告；稀疏任务不能凭几次请求认证长期95%服务水平。

**与上一版1秒的关系**：统一1000ms仍是上一版有效配置。此方案提出有来源的替代参考政策，并不将1秒包装成公认SLO；只有新参考运行完成、分母冻结后才启用新评分。`k=2` 得到的值可能小于或大于1秒，它是文献风格的实验目标，不是已验证的业务承诺。

### 2.4 M：容量份额保留，安全系数移出实测成本

**[R8] Ghodsi et al., NSDI 2011，*Dominant Resource Fairness: Fair Allocation of Multiple Resource Types*.** [官方全文](https://www.usenix.org/legacy/events/nsdi11/tech/full_papers/Ghodsi.pdf)。§4 / PDF pp.3–4，Algorithm 1 使用 `u_i,j/r_j` 表示某资源份额，举例以占用 GB / 总 GB 表示内存份额。

```text
M_v2 = startup-to-drain owned sampled peak GPU memory / B
```

使用**真实 owned 采样峰值**作为占用，包含模型、缓存、临时张量；所有候选采用相同测量边界、采样方法与 GPU 可用预算。B 沿用有硬件核验记录的固定预算，不从候选峰值的最大值反推分母。

当前 M 将安全系数 f 乘入成本。建议新版本将 f 保留在既有规划/显存安全约束中，实测归一化成本使用原始峰值，避免把不确定性裕度误称实际资源占用。已有硬合法性规则不放松；保守规划代理仍需单独标识，不能和原始实测 M 混为同一证据。

DRF 并未规定我们的 owned 归属、采样峰值或安全系数；它直接支持的是“用量 / 容量”这一归一化，而测量口径由实际部署问题决定。

### 2.5 E：相同 episode 的基线能量比，去掉无依据的450W及默认摊销期

**[R9] You et al., NSDI 2023，*Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training*.** [官方全文](https://www.usenix.org/system/files/nsdi23-you.pdf)。Fig.1 / PDF p.3 明确报告相对固定 Baseline 的 normalized energy；§3.1 / PDF p.5，式 (2) 用硬件 MAXPOWER 统一能量/时间量纲。这是归一化能量与多目标折衷的直接先例，应用本身是训练。

**[R10] *MLPerf Power: Benchmarking the Energy Efficiency of Machine Learning Systems from μWatts to MWatts for Sustainable AI*, HPCA 2025.** [DOI](https://doi.org/10.1109/HPCA61900.2025.00092)，[全文 v2](https://arxiv.org/html/2410.12032v2)。§IV-A 强调完整测量边界、可比性和透明度；§IV-B 说明功率与活动阶段时间对齐、积分得到能量，并按场景评价能效。

本地同时保存该论文正式题名的PDF全文（arXiv v2 / HPCA 2025 版式），§IV-A/IV-B 位于 PDF pp.5–6。

建议主评价使用固定 finite episode：从部署启动开始，经统一预热、正式到达窗口 H，至正式 cohort 排空结束。对候选和参考都积分同一类型的区间：

```text
J_episode = integral(P_GPU_board(t), startup_begin ... cohort_drain_end)
E_v2 = J_episode / mean(J_episode_ref over frozen reference runs)
```

在同一工作负载协议下预先选一个**可行、稳定、覆盖四任务的固定对照部署 R**，冻结其模型、副本、路由、资源控制、输入协议、GPU和参考种子集合。参考能量取该开发阶段重复运行的均值；所有评价候选和确认种子共用这个固定分母，不逐候选、逐确认种子重算。

这将能耗解释为“交付同一服务 episode 的相对能量”，无需假设450W参考功率或3600s生命周期。候选的启动/排空更长时对应能量直接计入。它也意味着 H 是**显式评价协议参数**；不能将这个结果直接外推为任意长期部署的能耗排名。

本项目目前只有整张 GPU 板卡遥测，因此称 GPU 能量，**不称 MLPerf 完整系统测量或整机能耗**。不减空闲功耗作为主能量，以免空闲基线噪声造成负值；若现有文件仅保存 run/startup 电量而缺少 drain 区间，新 E 记为未知，不据旧 E 分数倒推补全。

常驻长期服务应另报 `J_steady/J_steady_ref` 与启动能量，而非把一个猜测的 A 混进主分数。有限 episode 是本项目的可测量适配；MLPerf 并没有规定我们的启动/排空边界，Zeus 也没有规定推理系统的部署寿命。

## 3. 为什么不强制六项都压成0到1

R1 同时承认范围法和参考比值法，不要求一切都落在 `[0,1]`。主设计选择可解释的端点与分母：Q 一个可信质量跨度，T 一份输入需求，L/P 一个延迟目标，M 一份硬件容量，E 一次基线能量。

T 在 `[0,1]`，合法实测 M 在 `[0,1]`；Q 对冻结开发范围之外的数据可以超界，L/P/E 可以超过1。**这些都符合参考归一化的数学定义，但并不保证六项影响均衡。**

若产品目标明确要求每项有界，可采用固定上下锚点的 Min–Max，再裁剪至 `[0,1]`。但延迟和能耗的坏端点仍需业务预算或明确参考协议，裁剪还会让超差方案失去区分度；不能仅为实现0到1而编造“最大可接受值”。因此本次不推荐用 `x/(1+x)`、随候选变化的 min/max，或一个无来源的 sigmoid 作为默认方案。

## 4. 权重、相关性与验证

六项各 `1/6` 可作为保留用户现有六项偏好的**主比较设定**。R1 支持这种常见聚合形式，却同时明确指出其权重含义是 trade-off，不是科学测得的客观重要性。我们不声称归一化能消除这种选择。

至少预先做以下检查：

1. 参考敏感性：`k∈{1,2,4,8}`；Q 的固定范围法与 Proteus 风格相对最佳质量法；E 的 episode 与 steady-only 结果分别标识。
2. 权重敏感性：主分析六项等权；对照按质量、有效吞吐、延迟、显存、能耗五组等权，L/P 各0.1，其余各0.2；另对各项作预先声明的单项权重变化。
3. 同时报告原始单位、六项成本、六项贡献、资格条件及 Pareto 关系。不能只给总 U，也不能为某结论重新挑参考锚点。
4. Q 锚点、D、能量分母都使用独立的开发参考证据，确认集不参与定分母。小样本质量与稀疏任务尾分位的不确定性须保留；仅增加同一输入的重复请求不能扩大语义质量样本量。

## 5. 与现有实现的差异及落地条件

|项|现有实现|拟议 v2|
|---|---|---|
|Q|四种原始任务损失直接加权|先按冻结质量跨度变换，再按任务流量加权|
|T|正式窗口结束前完成比例的补数|正式 offered cohort 中按各自 deadline 成功返回比例的补数|
|L/P|手工研究尺度，最新统一1秒|固定参考路径实测均值的2倍作为主实验 SLO；所有候选共用|
|M|采样峰值乘安全系数后除预算|原始采样峰值除预算，安全裕度继续用于规划/合法性|
|E|运行平均功率加启动摊销，再除450W|完整固定 episode 的GPU能量 / 冻结参考部署能量|

新定义不能只改 JSON 分母就宣称生效：Q 需保留任务原始损失和锚点版本；T 需从逐请求 arrival/completion 重建 deadline 达标数；E 需完整 episode 的功率覆盖；预测器、原生测量和离线评分必须一起升级，另立 `six_term_reference_v2` 契约，并核验新公式对预测器/优化器的适用性。

现有旧分数、实验配置和最优性证书继续保留原定义。当前的旧实测可能能重建部分 v2 分项，但缺参考值或完整能量时只能报告部分结果，不能冒称完整 v2 总效用或新最优部署。

本次交付的是**每项可追溯、公式明确、改动边界明确的重设计**。尚无独立冻结的新 `a/z/D/J_ref` 数值，所以没有把现有数据事后挑成正式参考，也没有在运行入口中默默启用新总分。

## 6. 可用于论文的方法表述

> 我们采用固定参考的六指标评价。质量损失按照开发阶段冻结的可信最好与最弱质量跨度归一化；有效吞吐表示实验延迟目标内交付的输入需求比例；平均与P95端到端延迟除以固定任务SLO，该SLO通过独立参考路径画像按预先指定倍数产生；GPU显存除以可用容量，GPU能量除以固定参考部署交付同一评价episode的能量。归一化分别依据复合指标的范围/参考比值方法及模型服务、资源份额和能量基线文献，并为四任务部署作适配。我们报告原始指标、六项贡献及参考/权重敏感性，不将等权组合或实验SLO解释为通用业务标准。

参考材料中的 R3/R4/R6 论文原始PDF来自用户提供的附件；题名、原文页码和本地文件哈希均保存在证据清单中。上述引用支持各自对应的方法，不表示已有一篇论文提出了本项目完整的六项组合。
