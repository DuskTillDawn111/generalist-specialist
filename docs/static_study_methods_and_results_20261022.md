# 静态负载下通用／专用联合部署：方法与最终结果

现有流程已经完成混合补测、质量入效用、冻结预测器验证、七模型联合搜索及同域 DFS 效率对照。
正式 GPU 独立配对确认已完成 **15／15 条**，全部严格验证，并通过独立算术及统计复核。
自由搜索方案在三个种子中均领先其余四个已测物理方案；最优证书仍仅覆盖**冻结预测器在声明离散域内的部署与路由**，
有限候选实测不构成物理全组合最优或 0.05 水平统计显著的证明。
下面直接对应最初的五个问题说明机制、结果及证据范围。

本轮研究池沿用已确定的七模型：Baichuan-Omni、MiniCPM-o、Qwen2.5-Omni 三个通用模型，
以及 QA、图像、OCR、ASR 四个专用模型；纯通、纯专作为结构控制。
没有将旧 A/B/C 重新命名为新池。四任务到达率各约 0.074836 请求/秒，采用固定种子的 Poisson 到达；
“静态”指负载分布和路由比例固定，不要求请求等间隔到达。
正式窗口 H=1800 秒、开发种子 42、确认种子 43/44/45。
副本采用冻结画像中声明的离散状态，允许零副本；路由为分母 1..K 的有理网格并集，分别研究 K4、K8，关闭额外容量见证。

完整条件及逐文件 SHA 见[最终研究总报告](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/study-summary-final-v1/report.zh.md)
及其[机器可读证据](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/study-summary-final-v1/report.json)。

## 1. 混合组合很多，需要补测什么，如何控制数量？

需要测混合部署，但没有逐一运行所有部署／副本／路由组合。
本轮先执行 **8 条 H600、seed42 条件**：共享通用、多专用共驻、QA 和 ASR 各自的全专用路由与
75% 给专用／25% 给通用路由、OCR 与图像的分流条件。
这些条件用于观察共享队列、共驻资源和路由变化，而不是为每个组合拟合独立参数。
每条都保留实际输出评分、端到端延迟、整板能耗、显存、启动成本和运行身份。

第一版模型按事先固定的 6 训练＋2 留出执行，**验证失败**：两条留出 U 绝对误差为 0.284082、0.139401，
超过预设 0.10，且排序反转。失败模型和原始结果完整保留，没有进入正式搜索。
随后显式启用事先封存的 v2 逐输入回放方案：原 8 条作为 development，
再补 **2 条新的 H600、seed46/47 独立留出**，不复用旧留出充当独立验证。
v2 使用冻结逐输入服务基线回放共享 FIFO，用少量全局系数迁移服务、整板功率／启动与显存；
两条新留出 U 绝对误差为 0.009697、0.004604，全部预设误差检查和排序检查通过。
两个留出使用不同种子，其排序一致是模型验证检查，不能当成同轨迹分流因果对照。

因此，本轮校准／验证新增测量是 **8＋2 条**，而非枚举混合组合。
v2 验证通过后才冻结并搜索，正式确认数据禁止回流拟合。
这只是验证了有限条件上的预测能力，不能保证任意未测组合的误差都有覆盖。
未覆盖证据或预算不足的域继续保留 unknown、上界和 gap；不能按零损失或“已不可行”处理。

证据：[八条件原生观察](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/eight-point-observation-v1.json)、
[v1 失败报告](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/surrogate-v1/calibration-report.json)、
[v2 验证报告](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/surrogate-v2/calibration-report.json)、
[v2 独立复核](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/v2-success-independent-audit.zh.md)。

## 2. 质量如何进入总效用，是否仍有绝对精度准入？

当前正式配置使用六项等权，每项 1/6，效用越大越好：

```text
U = -(Q_loss + T_loss + L_norm + P_norm + M_norm + E_norm) / 6
```

| 项 | 评价内容 |
|---|---|
| Q | 按固定任务流量权重汇总的质量损失 |
| T | 正式窗口内未完成的计划需求比例 |
| L | 各任务平均端到端延迟相对冻结参考值的成本 |
| P | 各任务 P95 端到端延迟相对冻结参考值的成本 |
| M | 显存归一化成本 |
| E | 整板运行能耗与统一摊销启动成本 |

QA 使用 1−EM，图像使用 1−accuracy，OCR 使用 CER，ASR 使用 WER。
OCR／ASR 先累计编辑错误和参考字符／词数再计算，不能把不同长度样本的错误率简单平均；CER／WER 不截断。
四任务等到达率，因此本轮任务质量权重各 1/4。

**当前六项契约已移除绝对精度门槛。** 可执行路径不会因为未达到某个 EM／accuracy 下限或 CER／WER 上限被淘汰。
专用模型的质量优势通过真实质量记录进入 Q，而不是由“专用”角色直接加分。
预测按冻结输入与路由回放历史逐输入输出；GPU 实测则重新评分当次实际输出，不能用历史精度代替。
输入、参考、协议、检查点和运行身份仍需有效；这些是证据检查，不是精度准入。
缺少质量或完整整板能耗时不产生完整 U，也不重新归一化剩余权重。
等权仅表示惩罚系数相同，不保证六项实际贡献大小相同，报告同时保留六项成本。

定义见[六项效用规范](/root/autodl-tmp/generalist-specialist-project/docs/utility_specification.md)，
冻结数值见[K8 配置](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/search-v2-v5-K8/system-config.json)。

## 3. 除纯通／纯专外，与谁比较，如何减少完整评价次数？

联合 MILP 同时选择模型、副本和路由。主问题使用有效下界提出方案，完整预测器计算六项效用后反馈修正，
直到全域效用上界与可行方案 U 的差达到容差；未知域也必须纳入是否能发证的判断。
这使最优性依赖可核验的界，而不依赖“只比几个基线好”。

另外重新求解六个角色：自由搜索、排除胜出部署后的最强其他部署、固定胜出部署并排除原方案后的最强其他路由、
全域不分流、纯通、纯专。**这些竞争方案都重新优化，包含混合部署与路由竞争者。**

v5 的 K8 实际结果如下；表中全部是冻结预测器结果，非 GPU 实测排名。

| 角色 | 预测 U | 有限域最优证书 | 完整方案预测次数 | 外部搜索墙钟秒 |
|---|---:|---|---:|---:|
| 自由搜索方案 | -0.309303 | 有，gap=0 | 3 | 18.988 |
| 最强其他部署 | -0.321209 | 有 | 4 | 20.580 |
| 同部署最强其他路由 | -0.325602 | 有 | 5 | 0.376 |
| 最优不分流 | -0.309303 | 有，等于自由方案 | 2 | 5.733 |
| 纯通预算候选 | -5.676148 | 无，600 秒预算耗尽 | 160 | 600.352 |
| 纯专域最优方案 | -0.341342 | 有 | 1 | 0.288 |

自由方案为 MiniCPM-o、图像专用、OCR 专用各一个副本：QA／ASR 给通用，图像／OCR 给对应专用。
最强其他部署将 OCR 改为两个副本；同部署最强其他路由将图像任务的 1/6 分给通用。
自由方案不分流，因此与不分流角色共用一个正式物理方案。
完整预测 3／4／5 次不是 GPU 运行次数，也不是总搜索操作数：仍有列构建、显存预筛选和主问题求解开销。

纯通结果需特别说明：**不能称该预算候选为纯通最优。**
嵌套 K4 已发现更强纯通候选 U=-3.367339，K8 重评分一致；它部署 MiniCPM-o 两副本、Qwen2.5-Omni 一副本，
QA 给后者，其余任务给前者。K8 包含 K4，预算内更差的 incumbent 只反映搜索尚未完成，不能解释为细网格最优值下降。
额外候选已单独记录，不替换已冻结正式确认名单。
主 K8 纯通域上界为 -0.555303，仍低于全池已知可行 U=-0.309303，
所以足以排除纯通域赢得本轮全池预测最优；**这不证明纯通域内哪个候选最优**。

原始搜索见[K8 六角色结果](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/search-v2-v5-K8/search-results.json)，
额外纯通证据见[独立说明与重评分](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/pure-generalist-k8-extra-evidence/README.zh.md)。
求解入口为[冻结六角色脚本](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/execution-source-v5/scripts/solve_static_session_study.py)。

最终 GPU 确认按物理方案去重后为 **5 个方案×3 个种子＝15 条 H1800 作业**，不测全部搜索组合。
15 条现已全部完成。它们检验有限组强竞争方案的实际表现，不能直接证明物理全组合全局最优。

## 4. DFS 是否搜到同样最优，效率优势有多大？

正式效率对照使用 G2＋四专用的五模型子域，各模型允许 0／1 副本、K4 路由。
该域在显存／rho／排空等进一步拒绝前有 4097 个完整身份；plain DFS 实际完整评价中位数为 1696。
MILP、带界 DFS、plain DFS 采用相同冻结预测器、相同输入轨迹、相同预算和同一不可变 session 缓存机制，
串行新进程运行固定五轮并交错次序。
v5 的缓存复用相同摘要检查；代表性真实候选的完整预测 JSON 与 v4 逐位一致，没有重拟合预测器。

**五轮均同目标 U、同部署、同路由，15／15 次得到各自有限域证书。**

| 算法 | 搜索墙钟中位秒 | 整进程墙钟中位秒 | 完整预测次数中位数 |
|---|---:|---:|---:|
| MILP | 0.252850 | 0.792042 | 1 |
| 带界 DFS | 0.371656 | 0.911831 | 5 |
| plain DFS | 23.662888 | 24.207291 | 1696 |

搜索墙钟包含求解器构造与求解；整进程墙钟还包含导入、配置验证和结果传输。
以“竞争算法时间／MILP 时间”计算，MILP 相对 plain DFS 的比值为 **93.58 倍（搜索）／30.56 倍（整进程）**；
相对带界 DFS 为 **1.47 倍／1.15 倍**。评价次数减少与墙钟比是不同指标，不能互相替代。
该结果支持这一个同域、同优、有证书实验上的效率优势，不能外推到任意配置池或七模型全副本域。

此前 v4 的 120 秒 pilot 三算法均未证完，未作为同优加速证据；失败与预算记录继续保留。
正式数据见[五轮效率报告](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/five-model-v2-v5-K4-final/report.json)，
早期预算结果见[v4 pilot](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/five-model-v2-K4-pilot/report.json)，
预测等价证据见[v4／v5 对照](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/search-session-equivalence/real-v4-report.json)。

## 5. 能否主动把专用任务分给空闲通用，实际是否选了分流？

可以。任务流量是决策变量，通用副本的共享 FIFO 会反映所有已路由任务的竞争。
当专用排队收益超过通用侧新增排队、质量下降、显存和能耗成本时，算法可主动选择正的通用分流比例。
它优化的是**固定负载下的静态分流比例**；当前证书没有覆盖“每个请求按实时队长重新决策”的动态控制策略。

机制已有合成 CPU 正例：通用 QA 精度 0.80、专用 0.95，六项等权时仍主动将 25% QA 分给通用，
U=-0.223655，好于最佳不分流的 -0.236821；质量权重提高到 0.6／0.9 后恢复全专用。
这些服务与能耗数值是明确标注的 synthetic fixture，只证明算法能表达并优化该权衡，不能充当真实 GPU 分流收益。
原始记录见[合成分流与权重敏感性](/root/autodl-tmp/generalist-specialist-project/artifacts/static_joint_cpu_v6_20261021/diagnostics-final/diagnostics.md)。

真实校准反而提供了两个分流负例。同一混合部署、同 seed42 下，把对应任务从 100% 专用改为 75% 专用／25% 通用：

| 校准对照 | 100% 专用路由实测 U | 75% 专用路由实测 U | 分流方案减不分流 |
|---|---:|---:|---:|
| QA | -4.647848 | -4.993744 | -0.345896 |
| ASR | -5.007372 | -5.338051 | -0.330678 |

因此不能预设分流总有收益，也不能为了展示混合分流而强制算法选它。
本轮固定负载下，自由搜索最优和最优不分流相同；图像 1/6 给通用的强替代路由预测 U 更低。
这些结果与“算法具备分流优化能力”一致：它根据当前总效用选择是否分流。
真实负例来自[八条件完整原生结果](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/eight-point-observation-v1.json)，
尚不构成所有负载下分流有害的结论。

## 正式 GPU 最终结果与证据边界

冻结的 5 个物理方案×seed43／44／45＝**15 条 H1800 作业全部完成**，严格报告状态为 `ALL_FROZEN_JOBS_VERIFIED`，无缺失作业。
同一 seed 的五个方案使用相同到达轨迹、样本顺序及任务质量分母；seed43 为 527 个请求，seed44／45 各为 529 个请求。
所有计划请求均最终完成，实际输出有效；窗口内未完成的请求仍按冻结契约计入 T，不能把排空后的完成比例替代 T。
以下均为当次实际输出、计时及硬件采样重新计算的实测值，数值保留六位小数；完整精度见严格 JSON。

| 物理方案 | seed | Q | T | L | P | M | E | U |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 自由搜索／最优不分流 | 43 | 0.155143 | 0.000000 | 0.344961 | 0.690283 | 0.459488 | 0.213391 | -0.310544 |
| 自由搜索／最优不分流 | 44 | 0.159017 | 0.000000 | 0.354484 | 0.732432 | 0.462291 | 0.215205 | -0.320572 |
| 自由搜索／最优不分流 | 45 | 0.160723 | 0.000000 | 0.342794 | 0.653510 | 0.459894 | 0.214724 | -0.305274 |
| 最强其他部署 | 43 | 0.155143 | 0.000000 | 0.348631 | 0.701367 | 0.509130 | 0.213939 | -0.321368 |
| 最强其他部署 | 44 | 0.159017 | 0.000000 | 0.355028 | 0.725959 | 0.511933 | 0.215107 | -0.327841 |
| 最强其他部署 | 45 | 0.157074 | 0.000000 | 0.344051 | 0.632717 | 0.509673 | 0.214812 | -0.309721 |
| 同部署最强其他路由 | 43 | 0.165223 | 0.000000 | 0.384070 | 0.810111 | 0.461748 | 0.213908 | -0.339177 |
| 同部署最强其他路由 | 44 | 0.175684 | 0.000000 | 0.378352 | 0.778604 | 0.462291 | 0.216055 | -0.335164 |
| 同部署最强其他路由 | 45 | 0.160980 | 0.000000 | 0.368882 | 0.735998 | 0.460030 | 0.213296 | -0.323198 |
| 纯专域最优方案 | 43 | 0.139401 | 0.000000 | 0.343487 | 0.636729 | 0.690927 | 0.207838 | -0.336397 |
| 纯专域最优方案 | 44 | 0.142093 | 0.000000 | 0.352406 | 0.643695 | 0.692374 | 0.209795 | -0.340060 |
| 纯专域最优方案 | 45 | 0.140025 | 0.000000 | 0.347682 | 0.623732 | 0.690927 | 0.207313 | -0.334946 |
| 纯通预算候选 | 43 | 0.213000 | 0.017078 | 6.604153 | 28.666268 | 0.506056 | 0.383401 | -6.064993 |
| 纯通预算候选 | 44 | 0.214217 | 0.003781 | 5.431602 | 20.504065 | 0.506010 | 0.370859 | -4.505089 |
| 纯通预算候选 | 45 | 0.212370 | 0.001890 | 7.496197 | 26.424122 | 0.506056 | 0.382248 | -5.837147 |

三种子均值为以下结果；自由方案与不分流共用同一物理作业，不能计作两组独立实验。

| 物理方案 | 平均 Q | 平均 T | 平均 L | 平均 P | 平均 M | 平均 E | 平均 U |
|---|---:|---:|---:|---:|---:|---:|---:|
| 自由搜索／最优不分流 | 0.158294 | 0.000000 | 0.347413 | 0.692075 | 0.460558 | 0.214440 | -0.312130 |
| 最强其他部署 | 0.157078 | 0.000000 | 0.349237 | 0.686681 | 0.510245 | 0.214619 | -0.319643 |
| 同部署最强其他路由 | 0.167296 | 0.000000 | 0.377101 | 0.774904 | 0.461356 | 0.214420 | -0.332513 |
| 纯专域最优方案 | 0.140507 | 0.000000 | 0.347858 | 0.634718 | 0.691409 | 0.208316 | -0.337135 |
| 纯通预算候选 | 0.213196 | 0.007583 | 6.510651 | 25.198151 | 0.506041 | 0.378836 | -5.469076 |

**自由搜索方案的实测平均 U=-0.312130，在这五个冻结候选中最高，并在三个种子内均领先四个竞争方案。**
五候选三种子均值排序与冻结预测排序一致：自由方案、其他部署、替代路由、纯专、纯通预算候选。
单种子并非完全同序，例如 seed43 的纯专 U=-0.336397 优于替代路由 U=-0.339177，与预测两者次序相反；自由方案的领先未反转。
纯专方案实测平均质量损失最低（Q=0.140507），自由方案 Q=0.158294；自由方案胜在总效用权衡，尤其显存成本较低。
图像任务 1/6 给通用的替代路由在三个种子中均低于自由不分流方案，本轮真实条件未显示这一分流带来总效用收益。

配对差定义为 ΔU=自由方案 U−竞争方案 U，使用预先声明的三个种子块配对；95% 区间为 Student t 区间（df=2）。

| 竞争方案 | seed43 ΔU | seed44 ΔU | seed45 ΔU | 平均 ΔU | 95% Student 区间 | 精确双侧符号翻转 p | Holm 校正 p |
|---|---:|---:|---:|---:|---|---:|---:|
| 纯通预算候选 | 5.754448 | 4.184517 | 5.531873 | 5.156946 | [3.046747, 7.267145] | 0.25 | 1.00 |
| 最强其他部署 | 0.010824 | 0.007269 | 0.004447 | 0.007513 | [-0.000425, 0.015452] | 0.25 | 1.00 |
| 纯专域最优方案 | 0.025853 | 0.019489 | 0.029672 | 0.025005 | [0.012226, 0.037784] | 0.25 | 1.00 |
| 同部署最强其他路由 | 0.028632 | 0.014593 | 0.017923 | 0.020383 | [0.002160, 0.038606] | 0.25 | 1.00 |

四组配对均完整。三个种子的精确双侧符号翻转检验最小 p=0.25，本次四组均为 0.25，Holm 后均为 1.00，
**没有 0.05 水平显著证据**。其他部署这一最近竞争者的 Student 区间跨零；其余区间虽为正，也不能替代精确检验或多重校正结论。
Student 区间依赖种子差独立且近似正态的假设；符号翻转检验依赖种子块独立及零假设下配对差对称。
种子块才是重复单位，结论条件于冻结的有限输入池。QA／OCR／ASR 各复用 32 个唯一输入、图像复用 64 个，
不能把每条 527／529 个请求当作同样数量的独立质量泛化样本。串行作业还可能受到运行时间漂移影响。

纯通角色仍是冻结的**预算候选**，不能称其为纯通最优；前述更强 K4 候选 U=-3.367339 只有预测与 K8 重评分证据，
**未纳入这 15 条正式 GPU 实测**。因此该纯通实测差不能代表自由方案相对最优纯通方案的实测差。
这些结果支持有限已测候选中的领先；全池最优证书来自冻结预测器及离散域的有效界，两者都不证明物理全组合全局最优。

独立算术审计对全部 15 条重新处理实际输出／参考文本、编辑距离、端到端均值与 P95、轨迹／路由、
显存样本和整板能量积分，没有调用生产评价器或直接采信缓存 U。共 **390 项检查全部通过**，
独立重算 U 与保存值的最大绝对差为 **1.78×10⁻¹⁵**；编辑距离原语另与动态规划交叉核对 103 次。
同种子到达／样本／质量分母及共享检查点身份一致。部署所含 worker 源文件集合不同可能导致 runtime 摘要不同，
审计逐条核对各自冻结作业与源文件 SHA，不要求跨部署 runtime 摘要相同。
独立统计审计另行用解析 t(df=2)、8 种符号枚举和独立 Holm 步进复核全部四组，状态为 `REAL_15_OF_15_STATISTICS_VERIFIED`；
它验证统计计算，不新增硬件观察。

能耗沿用冻结契约：E 计 H1800 正式窗口整板能量，另将加载、priming 与 warmup 启动能量按 A=3600 秒摊销，
参考功率 450 W；**CPU 能耗及窗口后排空能耗未计入 U**。排空请求仍计入质量和延迟，T 记录其未在正式窗口完成的损失。
纯通预算候选在 seed43／44／45 分别排空 9／2／1 个请求，排空时长为 28.020／12.206／5.382 秒，
按契约排除的排空整板能量分别约 6806.01／3119.14／1027.08 J；其余 12 条均无请求在排空期间才完成。
因此 E 是上述窗口与摊销规则下的指标，不是含 CPU 和全部排空的完整生命周期能耗。显存仍为记录样本的 owned-process 峰值加安全因子，
不能据此证明未采样时刻的连续硬件最大值。

追溯入口：

- [全部 15 条严格确认报告](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/confirmation-report-v1/report.zh.md)及[完整 JSON](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/confirmation-report-v1/report.json)。
- [最终研究汇总](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/study-summary-final-v1/report.zh.md)及[机器可读汇总](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/study-summary-final-v1/report.json)。
- [全部算术独立审计](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/confirmation-all15-independent-audit/report.zh.md)及[审计 JSON](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/confirmation-all15-independent-audit/report.json)。
- [最终配对统计独立审计](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/confirmation-statistics-final-audit-v1.json)。
- [配对效用细节与部署路由补充图说明](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/confirmation-paired-detail-figures-v1/README.zh.md)、[配对效用图](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/confirmation-paired-detail-figures-v1/confirmation-utility-paired-detail.png)、[部署与路由图](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/confirmation-paired-detail-figures-v1/selected-deployment-routing-detail.png)。纯通图示仍须按预算未证候选解释。
- [最终证据范围复核](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/final-evidence-scope-review.zh.md)及[完成回执](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/final-completion-receipt.json)。
- [冻结正式 packet](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/confirmation-v2-v5-K8/experiment-packet.json)及[串行队列状态](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/confirmation-runs/queue-progress.json)。
- [严格确认汇总脚本](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/execution-source-v5/scripts/report_static_confirmation.py)、[研究汇总入口](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/study-summary-source-v2/scripts/report_static_study.py)、[绘图入口](/root/autodl-tmp/generalist-specialist-project/artifacts/static-study-gpu-20261022/confirmation-plot-source-v1/scripts/plot_static_confirmation.py)。
