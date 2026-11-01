# 阶段组合响应 provider：训练前固定的模型与可识别性门槛

这是新训练前的 CPU 设计与提取合同，不是已拟合预测器。阶段/算子的可组合响应是主线；完整 cohort 的联合轨迹只保留作残差、相关性和反事实排序校验，不能成为只记忆完整路由的主预测器。物理 context 仅作独立诊断，不再开关 reference service。

## 先能拟合什么

当前真实可观测量是三路语言 forward 的 BEGIN shape/index、host BEGIN/END、current-stream event elapsed、query 的 host bracket 和实际 parent processing availability；OCR 只有较粗的 host native call。GPU SM/DRAM/L2 counter 不可用不阻止拟合响应分布。它限制资源类型的物理解读。

第一版组件是每个真实 language forward 和 disjoint host stage 的条件响应模型。单路隔离训练提供基线分布 `W0(owner, phase, input/shape, execution identity)`。两个 text pool 是主 57 请求的已知内容域，OCR 的已知内容也有交集；这些内容的独立重复可以检验同域组合，不需要等待新域泛化。不同 routing、offset 和并行集合必须通过动态 covariate 路径组合，不能以完整 cohort identity 查整段耗时。

对 forward，预先声明三种候选：baseline、visible_count、visible_phase_aware。后两者的响应虚拟时钟是：

`dV_i(t)/dt = exp[-β_n n_i(t) - β_s s_i(t)]`，完成条件为 `V_i = W0_i`。

visible_count 固定 `β_s=0`；baseline 两参数均为 0。两参数是自由实数，只有 rate 为正，不限制减速必须大于 1，允许观察到加速。初版先拟合全局两个干扰参数，隔离的 owner/phase/input 基线分别保存；不立即增加 per-owner×phase×peer 的大参数表。全局模型可能失败，这是可检验假设，不用经验乘数补洞。

`n_i` 是当前已经处理的 BEGIN、END、query 构成的 last-visible-unconfirmed peer owner 数。`s_i` 是其中存在同名 semantic phase 的 peer owner 数。多个未确认 forward 可同时属于同一 owner，owner 数仍只计一次；同一 owner 的多个 phase 描述保留。host END 不清除该 forward 的未确认 stream evidence，成功 query 到 parent processing 后才清除；false query 不证明 truth-now 仍未完成。OCR 的 host-active call 单独标记为粗可见证据，不能补成已知 device frontier。

这两个特征是可见证据，不是实际同时运行数量、SM 占用或 USHER 的资源相似度。观察器/通知延迟是执行身份的一部分。特征变更只改变后续积分，不能重置 forward 的 attained age。运行时不知道的将来 phase 和 query outcome 必须由执行/响应模型产生，不能从验证轨迹回放。

## 可组合的执行合同

provider 输入至少包含：阶段/算子身份、训练支持的 shape/input、开始时刻与 attained age、已处理的事件前缀、host 发布及依赖状态、原执行配置和时钟域。输出下一 milestone 的响应分布/剩余分布、删失或区间、支持数及 UNKNOWN 原因。没有“holder 获得 context 就前进一纳秒”的规则。

组合引擎按真实 worker 的 host 顺序发布 forward 和外部阶段；CUDA 仍异步。已发射和 stream event 仅在其实际发布之后出现。已有静态 DAG 可标记已发射算子，但未来 template 节点不能成为当前 ready frontier。forward index 不是生成 token 数；已知内容的固定生成模板也必须通过独立实际 work 一致性验证，遇到不同停止行为返回 UNKNOWN，不能借验证结果预知终止长度。

host call wall 与 current-stream elapsed 必须成为不同 response channel。后者包括排队和调度间隙，不能加到 host forward wall，也不能把 host begin + stream elapsed 当作精确 device END。成功 query 给当前 stream event 完成上界，前一次 false query 的 begin 给保守下界；无前次 false 时用 end-event record call 的 begin 作保守下界。无 true 为右删失。query 时刻和通知时刻分别保存，直到通知实际可用才进入在线状态。

自然与 profiler-perturbed 各自有两个 response channel，总计四个域。只在训练中建立并经独立检验的迁移合同才能跨域使用；相同输入、相同 phase 序列不能自动转移纳秒耗时。

不能把 `native_inference` 的包围时长再加上内部 forward、stream elapsed 和 post_sync。host 组合必须先构造互不重叠的 partition：pre_sync、inference 内各 forward 和 forward 外 gap、post_sync、cache reclamation、final_sync、response/READY 尾部。query/host launch 开销位于真实 call/gap 内，只计一次。post_sync 必须条件在模拟出的未确认 completion 状态；不能独立抽样一个包含同一等待的总时长后再加 GPU 等待。

当前 extractor 将 native_inference 明确标为 inclusive_parent_not_additive；它还没有生成全部 disjoint gap 节点或实现联合 host/completion provider。后续实现必须在隔离和共跑训练中验证重组后恰好恢复 handler→response/READY，才能接主全过程引擎。G2 语言 forward 外的多模态阶段及 OCR 算子尚未有同等细粒度探针，只能使用明确的粗 host-stage 条件响应，不能宣称已识别逐算子服务。没有跨流依赖证明时，不能升级为 all-stream 可组合 device service。

如果只能在 parent 收到 BEGIN 后知道实际 shape，应保留 BEGIN 的延迟进入与已过 age。host 已在 BEGIN 通知到达前结束的样本仍进缺失/删失分母，不能当成剩余时间 0 的普通训练行。可以先以公开输入和冻结的训练模板做 prior，收到实际 BEGIN 后再条件更新；不能用迟到的实际 shape 伪称更早已知。

## 固定 fit 与 rank 检查

仅训练数据估计基线及系数。单路基线参数应先冻结，或与共跑参数一起进入有明确定义的 likelihood；不把同一 noisy baseline 同时当无误差标签又当自变量。可使用 lognormal/AFT 响应误差或基线经验分布的时间变换 likelihood；具体误差族在读取独立验证前冻结。按 whole capture 保留 cluster，而非把所有 forward/kernel 当独立样本。

两干扰参数最少需要：有隔离 `(n,s)=(0,0)` 的基线、有同 phase peer 的 `(1,1)`、有异 phase peer 的 `(1,0)`，且各组覆盖相同目标 phase/input 基线。只有 `(n,s)=(1,1),(2,2)` 时两列完全共线；增加相同状态的一百万 kernel 不会增加 rank。双专用/专用+G2/SSG/加 OCR 检验 additivity 和遗漏 peer-class 效应；两个启动顺序及偏移检验序列/相位影响。

extractor 的 rank 仅检查真实已可用 prefix states 的两列支持和 whole capture 数，不能当最终可识别性。实际拟合必须计算积分模型对所有自由参数的 Jacobian（同时包括基线 nuisance 参数），检查 full rank、奇异值/条件数、profile likelihood 或信息矩阵的不确定性，以及 capture 留出稳定性。若事件路径不同但积分列仍成比例，仍不可识别。rank 满但参数区间过宽同样不得推广。

若现有 26 job 没产生同 phase/异 phase 两种独立路径，最小补采是相同输入/配置、单 peer 数为 1 的两种 phase 对齐。offset 根据训练隔离轨迹预先确定，再冻结新的采集 manifest；不按验证结果选 offset。无 rank 不补默认 β，也不把 physical context hazard 乘进来。

## 最小训练矩阵与范围

26 jobs 包含两个 text length/content 层、9 个同步活动 context，以及 SG 的 50/200ms 双启动顺序。四 owner 始终实际 resident，活动集合另记。这一轮的一次 capture 用于覆盖/可运行性和必要 rank 诊断，不能估计跨 capture 方差或独立资格。

初始训练下界为相同 26-job 设计至少 3 次完整 capture/mode。当前审计通过的一次可计入，只新增两次；natural/kernel 分别建模，即共 156 个 job 执行。三次仅满足初始重复设计，不承诺置信精度或 P95。若各参数和响应域支持不足，按预先定义的精度规则增加独立 capture。整次模式执行顺序和温度/热状态保留，不能当模式变换已校准。

冻结后至少两个 known-input 独立 capture/mode 起步，并另保留 unseen-input 检查。known-input 检验不同路由/启动状态的响应组合；unseen-input 若缺少 shape model 支持，应 UNKNOWN。阶段模型选择只看训练 capture 留出，不能在 holdout 上选 baseline/count/phase-aware 或阈值。

模型的全过程资格仍要求同域 57 请求及完整已知到达的独立测试：过程队列、准入、响应/READY、P95、六项效用的 ΔU 符号、候选排序与选择 regret。公开未来到达率可以完整纳入队列规划，但实际未来 completion 不是输入。没有完整支持和误差精度时，不把阶段模型的低训练误差当最终分流收益。

## USHER 与 Roomie 的落地点

static topology/shape 的 nominal FLOPs、declared parameter/activation/KV bytes 可以作为结构特征。必须有精确模型 config、算子语义、dtype/quantization/adapter、batch、Q/K 和 cache 长度的 receipt。实际缺少的字段保持 None；cache-presence 不等于已知 K，quantized 模型存储大小不等于一次 forward 的 DRAM 读流量。不能把 nominal bytes 写成带宽计数或 cache hit。

有这些结构证据后，可预先定义 compute-/memory-sensitive proxy，产生互补候选，并在独立自然配对中检验是否加速。资源 similarity 是未来独立 candidate feature；当前同 semantic phase 特征不能冒充它。即使 counter 一直不可用，也能用这种可证伪的候选生成+自然响应验证继续推进。

Roomie 的序列/相位思路通过真实 phase、双顺序、offset、整请求准入及 batch 等执行条件进入模型。offset 本身暂不增加自由系数，其效应由 covariate 路径体现，减少与 phase 共线。共同 batch、公共权重子矩阵与 cache reuse 必须有真实执行后端、相同权重身份和净收益测量。shared-text 三 owner 是新的执行身份，不能继承四 owner baseline；配批等待、额外操作、host 和显存成本必须进六项效用。

## 本次代码交付

`stage_response_features_v1.py` 不在冻结 profile closure 内。它只接受带 receipt 的 v6 training plan/context/executed-jobs，拒绝 smoke/validation、未完成、缺请求或带错误的 raw；priming 完全排除。BEGIN feature 只从执行事件中取得，response 文档只提供训练标签及身份复核，不读取 prediction、reference、答案、生成 token 数作为特征。

输出包括 ordered parent-visible states、带独立 cutoff 的动态 covariate 路径、shape/index、host wall、独立 completed-stream elapsed、interval censoring、训练用 evidence exposure、支持/rank 和 whole capture 分母。未来 END 用于训练标签/积分窗口，明确不作为 runtime feature。static resource 字段目前全部 None，有明确缺失原因。代码没有执行拟合、没有运行 GPU，也没有读取独立验证标签。
