# 队列感知部署／路由：完整执行记录

本轮已完成。原固定路由8条、新联合流程24条（14开发＋4独立验证＋6最终确认）全部收齐。最终结果见[实验结论](/root/autodl-tmp/generalist-specialist-project/artifacts/queue-aware-study-20261023/final-closure-v1/最终实验结论.md)与[逐轮研究报告](/root/autodl-tmp/generalist-specialist-project/artifacts/queue-aware-study-20261023/final-report-v1/研究报告.md)。实测选中混合驻留、文本仅专用队列均衡；动态分流能够触发，但本轮没有优于该强对照。

## 固定目标

让算法在明确候选域中比较部署副本与路由策略，以原 Q/T/L/P/M/E 六项总效用选择最优，并完成同域 DFS 核验、搜索效率比较、真实 GPU 强替代比较与独立验证。分流是否胜出由结果决定。

所有时间对用户按北京时间。原七模型池定义保留，模型没有被从科学问题中删除；当前开发预测对 G1/G3 的混合上下文仍缺证据，因此局部搜索的最优性不能外推成七模型池物理最优。未知与不可行分别记录。

## 实验合同

当前预声明为 [protocol-v3](/root/autodl-tmp/generalist-specialist-project/artifacts/queue-aware-study-20261023/full-workflow-protocol-v3/protocol.json)，开发实验包为 [packet-v2](/root/autodl-tmp/generalist-specialist-project/artifacts/queue-aware-study-20261023/native-joint-shortlist-packet-v2/experiment-packet.json)，实际执行根为 [execution-v2](/root/autodl-tmp/generalist-specialist-project/artifacts/queue-aware-study-20261023/native-joint-workflow-v1/execution-v2/progress.json)。v1/v2 预声明与旧执行记录原样保留；protocol-v2 在动态 GPU 测量之前加入不同部署的强替代，protocol-v3 则记录启动前可选证据断言的修订。

实际启动凭据为 [launch-v2.json](/root/autodl-tmp/generalist-specialist-project/artifacts/queue-aware-study-20261023/native-joint-workflow-v1/launch-v2.json)，监督器 PID 为 809487。packet-v2 SHA256 为 `1dd71e5fde83fce3195891b251a8a48e67348434a2143ac79226e6a9d5e0fe5f`；protocol-v3 SHA256 为 `03d9fc2c06afb72a7f3629c30b549762b147ec4a116023eb887f31ce3830b80d`。每个阶段开始及运行间边界核验已绑定文件；这里列出的是执行身份，不代表已产生测量结果。

负载 text 1.85 req/s，image/OCR/ASR 各 0.01 req/s，Poisson 到达。每个真实运行保持 600 秒测量、30 秒 warmup、原 drain 配置、49140 MiB 预算、1.110854 安全系数、rho≤0.8、原六项等权评价。绝对质量准入关闭，实际输出精度计入 Q。原稳定性源码与阈值保持不变。

新增事前路由边界：动态 text→G2 使用 token bucket，rate=1.85/8=0.23125 req/s，burst=1。不足额度时在专用副本间继续比较，不丢请求或主动延迟请求。该约束给出长期路由速率上界，有限突发单列；它沿用原 task-replica proxy 准入语义，并不声称已证明共享 GPU 的物理稳定性。

此 guard 会改变先前无 guard 的 CPU 结果。阶段一无 guard 的约 5.2% 分流比例不能冒充当前策略结果；新开发回放中质量感知策略的文本分流约 3.81%、3.55%，相对仅专用均衡的 L 改善约 4.53%、3.96%，均尚非 GPU 实测。

历史 text-n2 画像来自物理 GPU `GPU-533a287b-0e15-a7c8-8e58-2b21bdcbb13a`，当前各 GPU 批次固定使用 `GPU-8a262fe0-598c-46b7-c953-41d9da84a47e`。两者均为 RTX4090、49140 MiB、driver 595.71.05，但不能称为同一物理 GPU。standalone 容量 proxy 属于迁移口径，不是当前卡的物理容量证书；本轮最终有效性仍依赖本轮 native 资源计量、因果路由审计与原稳定性判定。

## 启动前修订：可选证据来源顺序

首次联合工作流 `execution-v1` 的监督器 PID 为 803609。启动前独立核验发现，新增纯专与另一混合部署的可选 `capacity_evidence` 内，`service_evidence.sources` 列表由集合遍历产生；跨 Python 进程时列表顺序可能不同。guard、rho、显存估计及全部数值阈值相同，完整字典的严格相等断言仍会因来源列表排列而误拒。差异证据保存在 [capacity-evidence-mismatch.json](/root/autodl-tmp/generalist-specialist-project/artifacts/queue-aware-study-20261023/native-final-prelaunch-audit-v1/capacity-evidence-mismatch.json)。

旧监督器在 `WAIT_EXISTING_GPU_BATCH` 等待期停止，状态保留为 `STOPPED_FOR_DIAGNOSIS_NO_RETRY`；本联合执行尚未启动任何新 GPU 运行。因此这是启动前准备问题，不计作新实验的 GPU 测量失败，也没有消耗开发113/114、独立验证81/82或最终确认71–73。

修订采用新增 packet-v2 的方式：仅把新增两部署、两开发种子的四个 job 所附可选 `capacity_evidence` 设为 `None`，与原十个 job 的调用口径一致。每次运行仍独立重新计算并严格核验容量 envelope、guard、rho 与显存；没有关闭事前准入，没有放宽数值阈值。policy contract、job ID、测量身份与冻结源码均未改变。完整修订凭据为 [revision-receipt.json](/root/autodl-tmp/generalist-specialist-project/artifacts/queue-aware-study-20261023/native-joint-shortlist-packet-v2/revision-receipt.json)。旧 packet、旧协议、旧执行及失败断言证据均保留，后续使用 protocol-v3／packet-v2／execution-v2 的新登记组合。

## 软件与核验

- `protocol.py`：冻结 resident replica 状态、策略、服务估计、聚合质量先验和源文件身份。
- `runtime_window.py`：保留原 trace 与 SamplePool；在到达时加锁读取全部 host／worker 管道未完成工作，选择 alias 后原子入队。记录完成事件何时实际对策略可见。priming、warmup、measurement 的策略状态隔离。
- `guarded_policy.py`：可重演的 token 状态与限制路径审计。
- `native_runner.py`：复用冻结 worker、largest-three priming、监控、功率积分与关闭流程；独立动态实验身份。
- `admission.py`：校验全部潜在执行路径与质量身份；由事前 rate envelope 按原代理服务口径检查容量与显存。
- `evaluation.py`：原 trace／输入核对与逐决策因果重演；复用原六项数值函数与稳定性分类，缺失计量保持 UNKNOWN。
- `search.py`：有限部署与策略搜索；静态界与动态界分别证明，plain DFS 使用相同目标、候选与新 oracle。
- `analysis.py`：逐方案总效用与贡献、实际路由、逐副本排队与服务统计，完整成对运行的差值。缺任何声明种子时不取部分均值排名。
- `workflow.py`：等待原 GPU 批次完成后串行执行、冻结开发选择、独立验证；资源或源文件错误保留并停止，无自动重试。
- `reporting.py`：只读已登记终端 raw、阶段报告与搜索 artifact，生成中文 Markdown、结构化 JSON、SHA256 manifest 及可选 PNG/PDF；不运行推理或修改实验文件。进行中、启动前停止与实际已运行后的不完整状态分别标记。

## 阶段与决策规则

1. **原固定路由批次继续。** 原八条 seed109–112、rank1/rank2 对照完成并由原 supervisor 审计，不更换正在运行的源码或条件。
2. **联合开发搜索。** 40 个覆盖四任务的部署，7,508 个静态部署／路由方案，加每部署三个动态策略共 120 个，总 7,628。动态只改变 text，背景按明确固定规则选择；包含纯通、纯专及混合结构。报告完整六项开发诊断 U，正式 U 保持 UNKNOWN。与 plain DFS 比较最优身份、值、oracle 次数及墙钟；有并行负载时不冒称独占重复基准。
3. **真实开发对照。** text×2＋G2×1＋OCR×1，比较 SWRR 不分流、SWRR 1/32、仅专用均衡、混合仅局部延迟、混合局部延迟＋质量。另加入开发搜索中的最佳纯专部署（text×2＋image＋OCR＋ASR，专用均衡）和另一种强混合部署（text×2＋G2＋image＋OCR，混合局部延迟）。seed113 七条件正序、114 七条件逆序，共十四条。局部不计质量的消融仍按包含 Q 的完整总效用接受最终评价。
4. **冻结选择。** 以两开发种子全部 eligible 的原六项平均 U 选最佳；再选另一类路由中的最强控制：混合感知队列 versus 静态／仅专用均衡。避免只比较两个几乎相同的混合消融而丢掉强控制。候选不足时记录选择未决，不用失败方案的诊断分数补位。
5. **独立验证与最终确认。** 冻结上述两个候选及两阶段 packet 后，先按 seed81、82 做四条独立验证，再按 71、72、73 做六条最终确认，跨五个种子交替 AB／BA。两阶段分别出报告，另附五对合并描述；不根据验证结果重新选参、补挑种子或改变效用权重。逐对报告 ΔU、各分项贡献、是否反转及实际分流；重复请求不算独立实验。
6. **结果闭环。** 原生有限候选表由优化选择与 DFS 另行核对；这只能证明已测候选表的排名一致。开发模型搜索效率与 GPU 实验成本分别报告。若 GPU 没有稳定分流收益，保留该结论并报告最强替代，不把 CPU 结果替代真实结果。

## 终态结果

CPU接入、准入、评价及真实mock subprocess已通过。原固定路由批次8/8完成并审计；联合流程24/24完成、全部eligible，阶段状态均为COMPLETE_AUDITED。北京时间2026-10-23 06:59:06联合监督器完成，未自动重试、未在验证后重选。历史批次独立结论见[结果简报](/root/autodl-tmp/generalist-specialist-project/docs/static_top_two_terminal_results_20261023.md)。

实测七候选冠军为ChatQA×2＋G2×1＋OCR×1、queue_specialist_only，开发平均U=−1.0022191872。独立验证两对平均领先冻结queue_latency对照0.1049804320，最终确认三对平均领先0.0375608308；五对均同向，合并平均领先0.0645286713。开发及两验证阶段的实测表算法与DFS同解同值。这是已测有限表的结论；开发预测模型最优原为queue_latency，与原生冠军不同，不能宣称物理预测正确。

完整开发域40部署、7628候选中DFS与有界搜索同解同值，oracle调用15256→3358，减少77.99%；冷进程墙钟572.62→139.52秒，非独占计时约4.10倍。该效率来自冻结开发模型的有限搜索，不是GPU推理效率或原始MILP全域证明。

机制审计确认分流是真实因果选择，但被选G2文本请求服务显著慢于冻结估计。开发动态分流相对静态不分流跨种子反转，相对仅专用均衡两种子均较差；五个预留种子也均较差。原稳定性规则、质量指标和权重保持不变，没有通过放宽规则制造分流收益。

## 报告与结果读取约定

报告 CLI 必须显式指定 `execution-v2`、`protocol-v3`、开发搜索 artifact 和全新输出目录，避免将旧执行或历史结果混入本轮。进行中报告快照示例为 [研究报告.md](/root/autodl-tmp/generalist-specialist-project/artifacts/queue-aware-study-20261023/report-generator-ready-v1/研究报告.md)，其状态为 `IN_PROGRESS`，不是最终研究结论。正式终态到达后生成新 artifact，不覆盖该快照。

报告生成器已检查以下呈现链路：

- 开发七方案均保留部署副本数、路由策略、完整候选 ID 与 policy contract SHA256；Markdown 表中短 ID 仅用于阅读，完整身份保存在同目录 `report.json`，原始 job／测量身份可通过对应 raw 路径及 manifest SHA256 回查。相同策略名在不同部署中不会合并。
- 原生逐轮报告实际 task→family 与 alias 请求计数、各副本服务／排队统计和 text→G2 决策审计示例；分流能力与最终收益分别评价，不把分流次数本身当成效用改善。
- 每轮列出原六项归一化成本和负效用贡献，开发均值表列六项平均贡献；完整整轮配对另列六项贡献差。缺测、失效或未收齐种子保留不可评价，不以0补值，不以 `diagnostic_U` 替代原生 U。
- 强纯专及其他混合部署保留完整组合身份与开发对照；验证只继承开发冻结的冠军和另一类路由最强对照，81/82与71–73分别报告，五对合并仅作额外描述，不重新选择。
- 同域 DFS／剪枝搜索报告 winner、目标、评价次数及计时；原生表搜索的读表时间不算 GPU 测量效率，诊断搜索的非独占 CPU 计时不冒称独占加速基准。收益反转、不可评价或不足以建立显著优势均如实保留。

可选 `--static-audit` 指向原 seed109–112 固定1/32／不分流批次的终态审计。只有其监督器明确完成且终态 audit 存在时才读取；否则仅记录路径和未完成状态，不读取半程排名。历史静态证据不并入新113/114筛选或保留种子验证。

## 已完成历史静态批次的质量相等审计

seed109、110、111 的已完成六条旧静态运行中，不分流与固定1/32分流的 Q 分别同为 `0.2810443031211888`、`0.2791938272181649`、`0.27808319351374355`。独立事后审计逐一匹配并重评分3,368对请求，复现了保存的六条 Q 及任务评分分子／分母。证据为实际 `worker_response.prediction`，旧静态评价与新动态评价均调用冻结的实际输出评分路径，不是把历史聚合质量代入实际 Q。完整证据、逐请求记录、SHA256及CPU复算脚本见 [质量相等审计](/root/autodl-tmp/generalist-specialist-project/artifacts/queue-aware-study-20261023/static-quality-equality-audit-v1/README.zh.md)。只读取前三对已完成运行，未提前读取seed112、新113/114或保留种子的结果。

这三对 text 分流35／34／35条，每个种子实际只反复分流两个输入，所有匹配请求的正式质量得分相同；输出字符串仍分别有18／17／17条变化。text正式指标是EM，有些输出的F1变化但EM不变，不能观察结果后换指标制造差异。

具体限制是输入循环与固定路由相锁：text池有32条，alias级SWRR分流步长31／33交替、64次到达形成重复周期，分流始终落在输入循环的第10／11位置。每个seed的shuffle改变对应哪两个输入，但不解除种子内覆盖限制。该分流子集不能用来估计对全池均匀随机1/32分配的平均质量损失。

同一冻结池的历史画像预测重新评分得到专用EM为23/32=0.71875，G2为21/32=0.65625：20条均正确、8条均错误、3条只有专用正确、1条只有G2正确。四条EM不同的输入均未落入这三对的分流子集。历史全池差异不是本次未执行路径的新原生反事实观测，不能将聚合差值强加到每次分流。旧静态数据、原Q与其零Q贡献差保持不变。

新动态路由依据队列可见工作选择路径，不使用固定1/32周期，但不能保证输入选择代表性，仍可能出现实际Q相同。新工作流的 `fixed_1of32_swrr` 控制仍有相锁风险。最终报告应同时保留实际分流次数、分流独立输入数与各pool计数，再结合实际输出Q评价；聚合先验只用于路由局部评分，评分真值不提供给路由器。新评价模块的实际输出评分函数已用旧seed109内存副本做CPU敏感性验证：只改一条原本EM正确的预测，Q增量为 `0.0008857268705594334`，匹配text任务权重／1111；扰动未写回、不作为实验结果。

报告可选参数 `--quality-audit` 指向 `artifacts/queue-aware-study-20261023/static-quality-equality-audit-v1/evidence-v2/report.json`，将该事后限制单独带入最终报告；不会加入新开发候选排名或保留seed配对。该补充只修改未绑定的报告模块及本记录，冻结运行源、种子、输入和packet均保持不变。

最终结果、逐轮路由、六项贡献、SHA256、图表及审计均已写入独立artifact。七模型池物理全局最优未获证明；本轮完成的是声明的局部搜索、32条原生运行及固定选择验证。未达到“分流优于强对照”的预期现象，负结果按原合同保留。
