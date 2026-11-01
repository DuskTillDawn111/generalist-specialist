# 多副本分流实验：画像和预测器适用性审查

结论：现有画像支持将 ASR 作为首个候选，但现有 v2 的验证身份不能直接转移到高 ASR workload。应派生新实验和新验证协议，保持全部旧记录不变。此次只有 CPU 只读检查与本文写入，没有 GPU、拟合、搜索或冻结源码变更。

## 已执行的门禁验证

使用冻结 `execution-source-v5` 的独立 `SearchSession`，原配置可通过验证；仅把 workload 改为 ASR=2.6，其余任务各0.05：

- 直接调用 `validated_replay_spec` 抛出 `V2_REPLAY_INPUT_CONTEXT_OR_WORKLOAD_CHANGED`。
- 用同样新 workload 创建新 session 也被拒绝。
- 即使 `SystemConfig` 结构构造成功，完整 `predict` 仍返回 U=null、eligible=False、同一拒绝原因。因此 SystemConfig 可构造不等于预测器验证通过。

`calibration_v2.py::validated_replay_spec` 同时绑定完整 quality 文档、输入顺序、safe factor、measurement context、workload，以及每个模型的 baseline/model/memory/capacity 身份。只改 `spec.workload` 后重签外层 hash 不能让旧实测变成新负载验证；这会制造无依据的验证来源。

## 现有 v2 拟合入口的具体限制

`declare_replay_protocol` 与 `scripts/fit_static_replay_v2.py prepare` 是特定的旧实验入口：必须包含原 v1 的八个执行 ID；其 purpose 为 `legacy_v1_development`；两个新 holdout 固定 H600、seed46/47、显式 heldout purpose；43/44/45保留给正式确认。`fit_calibrated_replay` 又要求每条原生 run 的 workload 等于新的 base workload。

因此，把旧低负载八条放进新高负载 base 会在拟合合同检查失败；把新八条假装为旧 ID、改旧 protocol、把 confirmation 降格为 development 都不合法。旧八条、v2 两留出以及正式15条应继续归属于原研究身份。

新阶段需要单独的协议/入口，明确声明它是新的 workload 实验。可以复用逐输入数学模型、NNLS 规则、严格原生导入、质量评分和 packet 身份构造，但不得偷偷放宽旧函数的验证门禁。新数学/门禁源码必须重新封存并绑定新版本，旧v4/v5和旧证书不改。

## 两条可选推进路径

**优先低成本路径：冻结转移模型，仅做新范围验证。** 完整冻结旧 service ratio、power coefficients、startup、memory 规则和逐输入池；新分支明确标为未验证转移，开发种子筛查只用于挑选后续预声明研究点，不能出正式最优证书。选定点后，对高 ASR、专用n2与通用共驻的两个独立条件做新 holdout 验证；新协议按预定七项误差和排序检查决定激活/失败，不使用这些 holdout 重拟合。新 spec 的来源要同时绑定旧系数 receipt、新 workload 和新验证 run。此路径验证的是既定系数的转移能力，不能伪称“新负载重新训练”。

**若转移验证失败：独立重新校准。** 先保存失败；把已查看新 run 明确列为 development，补充必要的配置覆盖后冻结拟合，再预声明两条全新 holdout。若仍用最多3个功率参数、全局service比例、startup/worker与内存倍率，开发条件至少应足够满足原有2×参数数量及矩阵满秩要求；建议6–8条有物理差异的条件，不能用同一原生run重标重复充数。若仍采用旧v2的“至少8条”规则，则新入口明确执行8条新development，而不是混用旧workload来凑数。

两条路径都需新协议，不能直接调用现有“旧八条”prepare伪装成功。验证门槛、六项权重、质量文档、延迟参考、显存预算和startup摊销保持原规则。新 seed 集在执行前声明，开发/留出/正式三组互斥；若复用数字42/46/47/43–45，必须明确是不同workload研究的局部种子，不主张相对旧研究完全新随机性。更稳妥是新协议使用新 seed 范围，相关入口也需显式支持。

## 高 ASR 必须新增的真实条件覆盖

新增证据必须实际包含 ASR专用>=2副本、同卡通用及低背景任务，而不是只补 standalone ASR n2。至少覆盖：

1. 目标混合部署 ASR×2+G2×1+OCR×1，背景text/image由G2承接。在候选高负载点测两个不同的合法分流比例，例如2.6rps下专用2/3与5/7；两条在冻结rho代理中均满足必要约束。记录每请求服务、排队、实际输出和各专用副本的分配，核对没有把n2服务再乘错容量因子。
2. 同workload下有竞争力的其他部署/副本布局，例如ASR×1+G2×2，或其他通用组合；先判定是否有完整可行效用。若画像必要容量已失败，它是饱和/不可行诊断，不能当作可拟合的完整eligible utility。现有 strict importer 会拒绝未完成或不eligible的校准数据。
3. 至少一个相同部署的route变化，避免把“更多副本”和“分流优化”混为一项；资源/能耗拟合若不足秩，可预声明不同实际共驻副本总数的额外开发配置，但不能为了凑数制造无效条件。
4. 新留出必须保留未见seed/路由或配置的预测检验，尤其检查服务比例随高负载/ASR副本数是否系统变化、整板功率是否外推、n2常驻显存/碎片化峰值、P95与排空是否偏差。质量必须对实际输出重新评分，不能以旧离线Q替代。

H600可以用于初步验证，正式评价仍H1800并保留drain300秒、同seed跨候选相同arrival/input序列。其余任务各0.05时H600预期每类仅30请求，H1800也只有90，不能继续写旧合同“每task至少100预期请求”。应如实声明有限背景样本量，或在执行前统一选择更高背景/更长H；不能看结果再更改。

## 能合法复用的 packet 构造，以及必须重建的身份

`static_study._job` 可复用构造新身份；`execute_job` 本来就允许测量尚无已验证预测器的条件，采集时显式使用task_replica_proxy/standalone_proxy，不宣称转移最优。`_candidate_calibrations` 默认只提n1，且 `prepare_experiments` 用旧原型规则选6–8条；它不能自动覆盖本研究的n2分流。应新建明确计划清单，经新协议冻结后调用同一底层身份/执行框架。

应创建新的 input-bundle派生目录和packet目录。input/runtime样本、references、quality、worker/checkpoint来源可以保留原文件receipt；新manifest写清父实验路径+SHA，别重新计算或重写旧标签。至少更新：

- workload与产生规则、各task expected request count、H/seed集、所有trace manifest/hash；旧输入准备函数的等负载lambda规则不能沿用。
- study-contract、system-config、route K、背景请求数量说明、校准/留出/正式角色和每种phase/purpose；保留原权重/质量/参考/预算。
- 每个计划的绝对rps routing、deployment、worker alias数量；所有 `_job.identity`、job ID、source_problem_sha256、frozen_files_sha256。
- 新模型/协议数学版本bindings、runtime source receipts、输入bundle manifest与source-bindings；完整checkpoint/GPU实际执行身份按原严格流程重绑定。
- 逐输入baseline数值可复用原始standalone receipt，但 `base_problem_sha256` 随新base变化，冻结baseline合同也要新建，不能篡改旧冻结表。

派生新负载是新实验，不是source relocation；不能宣称“只搬路径、物理语义未变”。`validate_packet` 只检文件和job身份，并不检查所有科学合同。特别是 `prepare_experiments(--config)` 不自动让输入目录的旧study-contract workload同步；正式 `build_confirmation_packet` 会对二者不一致拒绝。新派生合同和config必须先一致，再构造所有jobs。

## 分流收益与不可行对照的区别

高点2.6–2.7较可能得到ASR×2且分流，因为ASR n2的rho≤.8能力约1.879rps，n3+任一通用超49140MiB。但这也意味着同部署不分流以及简单再加专用副本往往不可行。它们不能产生有效U，不能据此宣传“U优于所有强对照”。应同时筛查较低区域1.6–1.85，在ASR n2不分流仍可行时检查自由解是否仍愿意n2+分流；若没有，就如实说明当前画像只支持容量可行性收益，尚未观察到用户要求的可行强对照间效用改进。

最终仍须七模型完整自由搜索，不限制结构、不强留通用、不压低专用品质、不改效用权重来迎合期望。扫描点的选择规则应预声明，保留全部扫描结果；只有独立确认才能支撑选中点的真实局部优势。静态分流比例不等于在线根据瞬时队列调度，二者应分开命名和证明。
