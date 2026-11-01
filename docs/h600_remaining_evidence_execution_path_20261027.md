# H600 剩余证据与可执行的完成路径

本文仅为 CPU 设计，不冻结新计划、seed 或统计门槛，不新增 campaign 代码，不启动 GPU，不 seal 预测。下文 seed 是待检查冲突、评审和冻结的角色建议，不能据此称已有计划或数据。

目标仍包括真实画像、独立预测验证、降低干扰机制验证和最终四组实测。`NOT_ELIGIBLE` 记录当前不能作出的结论，不能替代尚未完成的阶段四，也不能因此把整个 goal 标为完成。反过来，完成实验并不要求结果必须为正：若完整证据说明该优化方式不可行或没有收益，应交付真实负结果与明确边界，不能继续调整直到出现正收益。

## 1. 当前 CPU 信号意味着什么

主线程的未封存 precheck 在原 H600 / drain300 下预测三模式均能完成：max1 最后完成约 845.86 s、text P95 约 236 s；max2 最后完成约 609.43 s、text P95 约 17.37 s；shared text P95 约 4.19 s。这里只引用主线程提供的预测摘要，不读取运行中的独立 raw，也不把这些数当成已经 seal 或实测。

这说明不能预设 max1 会因 drain 超时自动跳过，也不能把 max2 称为已有效的“优化”。max1/max2 都可能以降低同时活动为代价增加极大的准入等待。最后完成早于 900 s 只说明该点预测通过 bounded-drain 检查，不等于原稳定性、`.8 + 1e-8` 的原 rho、六项效用或机制净收益合格。阶段三仍须按固定原计划报告实际所有任务、等待、P95、原稳定性和激活；结果可以否定当前准入思路。

## 2. 最小共同域与模式选择

保持四驻留不变，最小共同路线域仍为 balanced、skewed、spill_low、spill_high。每模式禁分流域 `N={balanced,skewed}`，允许域 `A=N∪{spill_low,spill_high}`。只能称这个显式有限域内最佳，不能称全路线最优。

要得到最早可用的四组证据，建议**只为一个事先锁定的 admission 模式补齐完整模式画像**，shared 作为原执行方式。max2 可以是待评审的优先候选，理由是点预测等待代价小于 max1；这不是收益结论。锁定选择的规则、时间和已知阶段三诊断必须留下记录，不能到最终确认后挑模式。max1 保留阶段三结果；若不继续为它补画像，应明确未搜索/未合格，而非把它当作已排除的最佳方案。

若目标升级为“max1/max2 全优化域内最佳”，则两个模式均需完整 route 证据，预算相应增加；只验证 max2 不能支持该更强结论。最终“允许”集合包含禁分流，允许域最优若选择 no-spill，应如实保留重复策略，不能强制分流制造收益。

## 3. 一个模式需要的最小新增数据矩阵

下表给出可落地的起始预算，**不是足够统计精度的保证**。所有运行的 H600/drain300/warmup0、负载、已知输入域/生成分布、驻留、controller、alias route、priming、source 都精确绑定。每条含 1200 个测量请求，保留低频 image/asr/ocr 各 30 个，不能把 text 请求数量当低频任务独立重复数。

|角色|建议新 seed 范围，尚未冻结|所需 mode×route|次数/请求|用途及禁止事项|
|---|---|---|---|---|
|准入 primary training T1/T2|例如 628101、628102|锁定 admission×四 route×两 seed|8 次，9600 测量请求|模式专属资源、容量估计；若预声明需要，也可作服务训练。不是验证，不借用旧 validation 请求。|
|准入独立适用性 V1|例如 628201|同 admission×四 route|4 次，4800 请求|验证冻结服务迁移、队列/P95/决策、资源及容量。不能回流拟合或改门槛后重称独立。|
|最终 F1…FR|例如从 628301 开始，R 另行预定|shared/admission×各四 route×每 block|8R 次，9600R 请求|四锁定策略和完整 N 强基线对照；不选模型、不选模式、不回填画像。|
|若当前 shared gate 未通过：shared 新训练/新验证|另设不与上列重用的 training/validation 角色|shared×四 route×训练两 seed，另独立一 seed|另加 8+4 次|失败后修复所需分支，不能把原 heldout 改为训练或重新当未见测试。是否能复用原 shared 训练由新协议明确。|

seed 的整数不同不自动意味着输入实例不重叠。当前 public known-input 池可能重复同一 pool_id；应称“新到达/输入抽样序列的运行独立 block”，公开实例重叠，不能宣称未见输入泛化。若需真正实例不相交，先划分新输入池并补对应服务/质量/资源支持，这不是上述最小路径。

为什么不是只采 spill_low：阶段三该路线三模式无法告诉我们禁分流 balanced/skewed 在 admission 下的资源、等待与容量，也不提供 N/A 内效用差值或排序。为什么不是每 route 一次训练：虽然可以产生点均值，却无法完成整 seed 的训练 CV；两 seed×四 route 是与现 H600 工作流兼容的最小可交叉检查起始单元。两 seed 仍不能给出运行级容量上界或高置信区间。

无论服务是否重拟合，**8 次相同的训练采集可以同时供资源与容量估计使用**，不需要为 E、M、capacity 各再跑八次。它们用同一次 native 原始观测、各自独立数值与来源审计；不能把这说成三套独立实验。priming 不算额外测量请求，但完整 startup/priming/drain 能量与 owned peak 必须保留。

## 4. versioned training mapper 如何保证不污染验证

现 `flow_profiles.map_flow_collection` 对非 shared 要求匹配的资源/context；新 admission pilot 的 adapter 又明确只输出 validation / not_for_training。两者都应保持不变。不能删旧 mapper 的检查、给验证 raw 改 phase、覆盖旧 model context，或将 pilot adapter 的输出改标 training 后喂 fit。

新增 mapper 应是独立版本，例如将来另建 controller-aware training evidence 模块，先用 CPU fixtures 完成角色/身份反例测试，再在首个新训练 startup 前固定源版本与计划。最少合同如下：

1. 仅接受预先声明的 `phase=predictor_training`、`split=training`、`campaign_role=primary_training`、新 training suite receipt 和精确 job ID；`created_ns`/源 seal 必须早于 native startup。拒绝 `predictor_validation`、`mechanism_validation`、final/diagnostic 及任何事后新增 training 角色。
2. 保留 immutable base worker context，同时绑定 mode/controller 的标准化资源配置、admission/deployment/resources 源 SHA 和独立 controller identity。统计键包括 mode；同驻留/活跃 family multiset 不能让 shared 与 admission 被自动池化。
3. 验证 native raw 未经改写、plan/invocation/trace/source 相同、每请求 task/input/alias/scheduled arrival 完整，READY PID/spec/clock、START/PHASE/RESULT 与真实语义阶段连续，同 alias FIFO 非重叠、准入 observed contract、完整 shutdown 和 bounded drain。
4. 验证 startup-through-drain NVML 按真实拥有 PID 归属、无 foreign、峰值/进程和一致、原 2 s gap 限制；保留原测量语义，峰值是 sampled owned peak，不把各 worker 不同时刻峰值相加，不将其宣称物理上界。
5. 一个完整 measurement cohort 输出一个 continuous trace，priming/warmup 仅作 excluded history。只从预先冻结 training plan 推导 split，保留原 raw request ID 和来源，不把实际服务、ticket 顺序、输出标签变成未来预测特征。
6. 训练消费者再次核对 suite/mode/controller/route/source closure；不能仅信任一个 `split=training` 字段。两个 seed 的 leave-one-whole-seed-out 中，资源、启动、容量以及任何拟合的服务均一起排除 heldout seed，避免“服务 heldout 而资源看过”泄漏。

validation/final 使用另一个只输出 `not_for_training=True` 的版本化 adapter，并保持严格实际检查。两个方向都 fail closed：training mapper 拒 validation，validation mapper 不被训练消费者接受。测试至少包含伪 training wrapper 引用旧 validation raw、source/资源漂移、启动后计划、漏请求/低频任务、native raw 变更、PHASE/READY/NVML 错误及跨 mode 池化拒绝。

## 5. 先只新增资源/容量训练，服务能否继续迁移

**可以，但条件是冻结的 shared 服务模型在准入下先被独立证实适用，且需要保留这个假设的实际局限。** 资源/容量训练依然测到真实 worker service，这是容量原接口需要的观测；“不重拟合服务预测器”不等于不能记录该 service。

最早路径 A：

1. 保留当前 selected shared 服务模型及 SHA，不因阶段三残差调整其参数。阶段三只提供 fixed spill_low 的迁移诊断，不充当四路线通过结论。
2. 新 T1/T2 八条训练流只拟合 mode 专属 startup/conditional power/owned sampled memory 与输入加权容量；full service predictor 参数完全不变。逐任务/输入/phase 支持先在 CPU 检查，未见上下文 UNKNOWN。
3. 在看 V1 之前冻结“冻结 shared service + admission controller + 新 mode resource/capacity + 训练质量”完整预测、source closure、原误差与决策门槛。资源/容量训练不能把 V1 的真实活跃轨迹或 permit 顺序回填预测。
4. V1 四条独立流同时检查 full cohort 的 service/E2E/队列/每任务 P95、六项预测、ΔU 符号和候选排序，以及原容量适用性、rho 和稳定性。采用已有原门槛；资源单项附加门槛若尚无，应在 V1 前另行声明并说明不能替代原六项决策 gate。
5. 只有这些检查完整通过，才能把该模式标为统计适用的服务迁移并继续 qualified final 路径。V1 不能用于重新校准共享服务后仍报告同轮 PASS。

若 stage3 的真实观测已清楚否定 shared→admission 服务迁移，或 V1 失败，路径 A 不成立。可选路径 B 是在新的训练协议中，用明确的 admission training traces 拟合 mode-specific 服务模型。若 T1/T2 在采集前已声明可用于服务建模，它们仍可用，因为本来就是训练；这不会让 V1 变成训练。但模型/估计器选择必须在训练内 CV 进行，保留旧失败，随后使用**另一个新 V2** 做独立验证。是否需要更多 training seed/context 由训练残差与缺覆盖确定，不能事后把验证中缺的上下文当零成本插值。

mode-specific 服务和容量均反映同一原执行定义：服务是 START→END，容量 service 做原 g(n)/n 归一化，等待/parent tail 分别建模或留作误差。不得通过把准入等待挪进“服务”再改 g(n) 人为让 rho 过关，也不能把等待从实际 E2E 中删掉。

## 6. 精确资源/容量与原资格的最小实现缺口

现 `h600_resources_v1`、`h600_capacity_v1` 和 input-weighted 入口绑定 shared 四路线及指定原训练 seed，不能原地扩成新 admission suite。后续应另加独立入口，保留原数值定义并显式接受新 mapper/mode 条件，而不是改冻结的 POLICY、seed 或 resource identity。

|组件|必须绑定/实现|不能采取的捷径|
|---|---|---|
|资源|完整 mode/controller；每 route 的 startup→phase_start joules、整段 owned sampled peak；conditional whole-board power、真实 idle/active task contexts；whole-seed CV|复制 shared E/M、把 startup 跨模式省略、family-only 兜底未见 task context、重复安全因子|
|容量|精确 alias route/load/input generator/H/drain/warmup/mode；每 family/task 输入加权真实 worker mean、原 g(n)/n、每 run/input 数；新 seed 的如实标明 transfer|将 max_active 当 replica n，改 g(n)，给目标 seed 插入 fake measured key，按实际 counts/H 替换 planned rps|
|适用性|V1 每 active family/task 对冻结训练 service 的 mean 误差、完整真实资源/来源、原 `.8+1e-8` rho 和原 native_stability；当前20% mean-service规则如复用须原样预声明|text 大样本掩盖 OCR/ASR，或 `drain_complete=True` 代替 rho/稳定性|
|搜索/评分|mode-aware 六项 provider 和资源/容量来源；按原资格过滤的完整 N/A，fresh plain/bounded DFS/all ties；资源或容量未知为 UNKNOWN|仅把模型 best 改名为原资格 best；直接 generic core 加非 shared配置而不审计证据|
|最终报告|独立新 blocks，所有原六项、强禁分流基线、原稳定性与资格、失败/缺测完整分母|只有服务/平均延迟图，或把被准入卡住的请求不计入 P95/U|

资源/容量数值可在训练拟合后发现所有 candidate 都不合格，此时应保留实数与失败原因，而非只输出 UNKNOWN。UNKNOWN 是缺证据；有充分证据但违反原 gate 应写明确 FAIL/INELIGIBLE。已知不合格和未知不能混成“预测器没找到”一种状态。

## 7. global max2 可能严格原 rho 不可行：如何处理

max2 能在 900 s 内完成，不代表 `.8+1e-8` 规划余量足够。对每条 route 应从其 mode-specific 训练容量计算原公式；原 n 仍为两专用/一个 G2/一个 OCR。再由独立适用性检验证明该统计 profile 在相同条件下可用。训练校准和原 g(n)/n 可能使高负载家族仍超过原 rho，即使真实队列最终排空。

若某候选 rho 明确超限，按原资格排除并保留数值。若 max2 **四条** candidate 在可靠、条件匹配的证据下均违反原资格，则该声明有限优化域为空：报告 `NO_QUALIFIED_CANDIDATE_IN_DECLARED_DOMAIN`，best=null、all_ties=[]；不能把两个空集合一致写成 DFS 最优成功。这个结论是原统计规划规则下的有限域不可行，不是物理上永远运行不了或所有未列路线不可行。

若只测 spill_low，不能推出四条域为空；若容量适用性未证实，则是 `DOMAIN_QUALIFICATION_UNKNOWN`，也不能提前给空域证明。max1 是否同样不合格需要它自己的证据；不因 max2 失败就自动继承。若为了让 max2 过关而降低负载、改 drain、放松 rho 或改 n，那是新问题/新协议，不能用来完成当前固定条件的正收益结论。

优化域为空时分两条明确路径：

- **正式 qualified 优化结果**：C/D 不存在合格最优，不伪造可行 incumbent。报告当前固定条件无合格 admission 方案，不声称已经验证正收益。
- **真实负结果对照**：仍可在显存/硬件执行条件安全、原 bounded drain 和失败停止规则允许且另行预声明的诊断协议中，运行固定四组/八配置，量化共享与准入的实际六项、等待/尾延迟和原 gate 失败。标明 `diagnostic execution of planning-ineligible candidate`，原 rho FAIL、final gain eligible=false 不变。此时 C/D 是预先指定的准入策略或“有限域已测性能最佳”诊断，不能称“原资格最优”。

第二条没有绕过原规则：原 gate 仍 FAIL，诊断回答该方案为什么不适合以及是否有实际负收益；它不把不合格策略当合格部署推荐。若现有 runner/report 不支持如实记录这类诊断，需新版本协议/入口，不能 monkeypatch 资格字段或删除检查。若任何硬执行/显存条件不允许则不启动，给出明确不能实测的原因与保留分母，不能伪造四组原始结果。

因此，若用户目标是**完成真实四组比较并得到诚实结论**，负结果路径可以交付这一实测任务；若目标是**证明合格优化和分流带来正收益**，空域或负结果说明该目标没有实现。最终应分别说“实验/报告是否完整”和“算法是否成功”，不以 NOT_ELIGIBLE 文档代替实测，也不承诺一定能做出正收益。

## 8. 预算、最早路径与失败分支

以下仅估算测量与 drain 配额，实际还需 startup、priming、shutdown、冷却和 CPU 审计；不能把它当运行时间承诺。每条测量600 s，最大 drain300 s，故测量加 drain 是每条 10–15 min；短 drain 会少于上界，启动额外另计。

|路径|新增运行数|测量至最大排空配额|限制|
|---|---:|---:|---|
|一个 admission 的 T1/T2 + V1|8+4=12|2–3 h + 启动等|当前 shared 验证合格且服务迁移一次通过时的最早资格补齐起点|
|再做一个最终 block|12+8=20|3 h 20 min–5 h + 启动等|只能单块实测，不提供可靠运行级显著性|
|例：R=3 个最终 block|12+24=36|6–9 h + 启动等|三个独立 block 仍有限；不是自动达到95%显著性或精度|
|同时补 max1/max2 两模式且比较两者|训练16+独立验证8，再最终3模式×4route×R，即24+12R|R=3 时10–15 h + 启动等|共同域/模式选择或多重假设需另预声明，非最早路径|
|shared 失败后也需新8训练+4独立|在相应路径上另加12|另加2–3 h + 启动等|具体修复/新训练是否足够不能预保证|
|路径A迁移失败，训练模型变更后新增 V2|至少另加4（只一个模式）|至少再40–60 min + 启动等|若训练上下文/样本仍不足，额外训练另预算，不挪用V1|

无须先为每个指标重复采集一遍，也无需把最终确认升级为大规模部署搜索。最早按以下依赖执行：

1. 完成当前 stage2 selected 独立报告/gate 和 stage3 固定模式诊断，保留失败，不改其数据角色。若 stage2 FAIL，正式正收益路径暂不开放，但诊断和新训练准备继续完成。
2. CPU 完成新 training mapper、mode-specific resource/capacity 消费入口及 meaningful role/source tests；锁定有限域、拟合/迁移规则、seed角色和预算，再前瞻准备 T1/T2。当前文件没有执行这一步。
3. 新8条 primary 训练完成后，一次产生资源/容量与覆盖矩阵，按整seed CV；如模式原 rho 已明确无可行候选，选择预声明负结果分支或停止 qualified 优化扩展，不继续假称最佳搜索。
4. 原始 sources/模型/provider/全部四 route预测、原误差阈值/容量适用性及失败动作在 V1 startup 之前 seal。V1用4条新独立流同时检验服务迁移、资源/容量和决策；未通过则保留并走新版本分支。
5. 合格时才做原资格约束下 CPU fresh DFS/all ties。锁定四策略、全部N基线、最终 R、执行平衡次序、主差值/多重比较与失败规则。采用新 F blocks，两模式各四 route，共8R。
6. 若走负结果路径，也需在新 F 之前锁定诊断策略/八配置与统计协议，按同到达/输入收集真实完整数据，出具原六项、原资格、稳定性、强基线和不确定性报告；不能从 F 再挑模式、路线或回填模型。

最终核心仍保留 `locked_allow − locked_no_spill` 与预声明的 `locked_allow − max_{c∈N}(actual U_c)` 强基线差值。`max_A(actual U)−max_N(actual U)` 只是使用当块结果选优的包络诊断，按集合包含关系非负，不能当可部署分流收益。所有缺失/未合格候选和完整 1200 分母均保留；六项缺一不重归一，不以完成子集估计全群 P95。

## 9. 交付清单与未完成事项

达到完整阶段四实测交付至少应有：模式/route/角色来源矩阵；mode-specific T1/T2 画像与整seed CV；冻结服务迁移或新服务模型的独立 V1/V2 报告；原容量/稳定性与六项判定；完整有限域 DFS/all ties 或明确空域/UNKNOWN；预声明的新最终 blocks 原始观测及全六项配对结果。正结果还需统计和原资格支持，负结果同样需要真实数据与完整失败保留。

目前仍缺上述新训练 mapper/模式数值入口与新训练/独立/最终数据。本文只把它们缩成一个可执行、可计数、不会污染验证的有限路径，不生成这些证据、不把预测摘要当实测，也不将 NOT_ELIGIBLE 作为已完成全部目标的替代物。
