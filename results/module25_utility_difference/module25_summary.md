# Module25：系统效用差预测与正收益分流发现

最终Gate：**INDEPENDENT_DELTA_SIGN_OR_UNCERTAINTY_NOT_VALIDATED**。完成Stage A/B/C后按用户停止规则停止。Stage D没有执行，Stage E没有执行，新GPU会话=0；没有声明任何新池或三阶段成功。

## 错误复现与可核验归因

24A开发探针实际ΔU=+0.476798，但20A预测=-0.029255、最终24A校准预测=-0.007317；双副本自主候选预测+0.031224才是24B验证的对象。24B实际ΔU=-0.106711。二者五个checkpoint和QPS相同，物理副本为5与8、计划ASR G份额18%与13.1833%、输入seed与cohort均不同；只能称不同配置下收益方向变化。各自配对内部匹配，原效用口径一致，没有截尾完成集效用。

原J优化的是未归一化的平均拥塞surrogate，没有20A任务参考值、尾延迟和显存效用的完整目标。21的Poisson常数service及24的固定路径残差仍不能表达路由改变后的service动态；未知副本回退先验和共置画像失配使小正预测超过真实余量。20A本身在本轮5个合法负例组上均预测为负，不能混称所有旧算法都错误预测为正。

24B的actual-minus-B24 ΔU误差中，mean项=-0.037677、P95项=-0.100411，吞吐+0.000163，memory约-0.000010；实际损失主要来自Text/OCR背景，而不只是ASR排队。将旧预测仅代入11.4943%实际份额，ΔU仍+0.003363，没有解释真实负收益。逐worker service/queue/P95、输入时长/类别差异、mean服务/等待/传输分解和观测service oracle均保存。P95没有错误地相加拆分。物理根因仍缺少同轨迹n1/n2×路由的因子实验和kernel/时钟/预处理证据，不作唯一GPU因果断言。

## 新模型与支持范围

Separate checkpoint/task/count的ridge拟合log服务均值和33个服务分位形状，输入包括own load、同worker其他任务load、foreign Generalist/Specialist load、驻留worker数量和idle-gap。同组合/workload内先合并整次重复会话，再按组平衡权重；4个以上组才估计route slopes，否则保留稀疏中位值并明确斜率未识别。Generalist背景与Specialist服务均允许随完整路由变化，无全模型统一干扰系数。

Poisson队列由项目Module21 FIFO变换扩展到经验服务分布与随机副本thinning；周期实验使用原periodic_arrivals/Router和物理worker FIFO。所有多任务共享队列，E2E加实测transport残差。仍有stationary/独立service、未测GPU状态/输入混合的局限。实际有效容量的点预测由条件service产生，不把isolated g(n)当确定共置扩展比。缺失路径/副本/组合明确unsupported，区间无支持则unbounded。实际U始终用20A冻结公式和hard quality/stability/memory约束。

全部125个原始会话纳入证据清单；20A无新GPU、21D重分析不重复计样本、23无GPU。最后模型完全排除24A/24B整池的6个会话（包括启动失败与preflight），预测验证每折排除比较两边的整个checkpoint组合，跨所有workload/副本/模块传播排除。完整deployment/count/QPS/hardware/request协议键用于inner calibration分组。没有请求随机训练/测试划分。已知历史结果的留出属于retrospective out-of-sample，不冒充盲测。

## 严格独立排序结果

| 打分器 | 合法独立对照组 | 可预测/拒绝 | 点方向正确率 | ΔU MAE | 错误正号 |
|---|---:|---|---:|---:|---:|
| J | 5 | 5/0 | 0.0 | None | 5 |
| A20 | 5 | 5/0 | 1.0 | 0.49397509557295827 | 0 |
| NEW | 5 | 3/2 | 0.6666666666666666 | 0.20666978440447992 | 1 |

5组均为实际负收益，来自4种checkpoint组合；重复配对先聚合为一个对照组。24A正开发探针不计独立正例，Module22 MID是部署策略差异，几乎零G流量，不能计作真实正GS/S对照。Module19原operationally-illegal Image/ASR比较器保留为辅助诊断，不扩大合法主验证分母。

新模型仅3组有点预测，其中2组方向正确=66.67%；按全部5组计覆盖正确2/5，另外2组拒绝而不是“正确识别”。24B仍误判+0.002205（真实-0.106711），小于+0.01阈值但符号依然错误；不构成可推荐收益。整池留出的24A正开发样本仍被预测为负（-0.006831），说明没有证明识别正收益的能力。

每折inner calibration仅0–1个独立ΔU对照组，95% split-conformal无法产生有限界；不把训练最小/最大或16次模拟变动当统计CI。有效有限区间=0，coverage未知。所有保守决策均ABSTAIN，不能把零错误推荐当预测已成功。2次实际unstable均仍被预测nominal stable，和20A一样；仅标uncertified不等于正确检出。

| 同一3组可预测子集 | 方向准确率 | ΔU MAE |
|---|---:|---:|
| J | 0.0 | None |
| A20 | 1.0 | 0.17136324361958222 |
| NEW | 0.6666666666666666 | 0.20666978440447992 |

必须比较相同分母，不能用新模型3组MAE低于20A5组MAE宣称改进。J没有系统U量纲，ΔU MAE不适用，未用经验换算系数。候选仅两个时，排序正确即方向正确；跨不同workload的增益排序是描述性Spearman，不冒充同一可执行候选列表的top-k搜索。

## 计算开销与后续搜索

同一24B候选对，热态10次、首轮丢弃，median：J=0.000035s，20A=0.003372s，新模型=0.036040s；新/20A=10.69倍，新/J=1038.69倍。新模型没有推理速度改善；扩展服务分布/队列有额外成本。整池优化搜索因gate失败未执行，不能给出搜索加速或新算法优化收益。原Module24B旧搜索耗时只能作背景，不是本轮等条件对照。

HIGH-first筛选规则已在拟合前冻结：全部现有质量准入模型及已测副本参与，B4预算、固定背景4/2/5/.03/.2 QPS，HIGH ASR=1.52、降低到.02/.80；使用真实20A Generator及全合法部署/自由路由的评分扩展，竞争纯G、纯S和混合。不强制A/B结构。module25/optimizer.py提供guarded adapter，当前gate会在求解开始前拒绝执行。high_first_pool_search.csv明确NOT_EXECUTED而非伪造失败候选；frozen_positive_candidates.json为空且frozen=false。

## 六个问题

1. 旧预测为什么误判：J目标不同、standalone/共置service不匹配、固定残差不随路由变、周期/Poisson与副本thinning差异，背景mean/P95损失遗漏；并非归一化算错或简单份额偏差。
2. 新预测器能否独立判断：未通过，3个可预测合法组正确2个、拒绝2组，无独立正例/有限误差区间，unstable漏检仍有2次。
3. 找到值得分流的新组合：没有形成可信候选；未执行新池搜索，不能断言所有池无收益。
4. 同一六任务系统形成A或B：未建立，未强制结构，也没有输出未验证成功。
5. GPU证明真实总效用优势：本轮GPU=0，既有24B负结果保留；24A单个开发正探针不是正式优势证明。
6. 相比J的准确率/开销：同一3组负例J0/3、新2/3、20A3/3，不能推广到正例；推理更慢，完整搜索速度改进未测。

历史16083个文件metadata不变，所有读取的科学输入SHA256不变，24A/24B gate原样保留。首轮大CSV字段限制及NumPy布尔JSON序列化失败日志保留；仅补齐缺失诊断artifact并独立确认点预测未变，没有重拟合或结果调参。所有科学输出、失败候选、原始负实验都保留。完成后停止。
