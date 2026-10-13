# Module 31：GO / NO_GO 与证据门槛

**当前 Gate：预注册交付完成；执行 NO_GO（等待审核）。** 这不是科学阴性结论。GPU 未启动，合法 U、ΔU 没有新结果。Module 30 的 GO 只授权准备预注册，不授权运行。

## 批准清单与数值来源

|ID|固定可审核内容|为何待批准 / 证据状态|
|---|---|---|
|P01|两模式完整 coarse [0.60,0.80,1.00]；每模式唯一 0.70/0.90 细扫；两 arm；独立种子；共同比例六任务；生成规则与条件分支|基准率与 burst 参数是原实现；草案网格用于新实验未获执行批准；两轴组合及新增 seed 是本次提案|
|P02|探索≤44 attempt/26400 formal秒；另批确认性≤9/5400秒；prime≤300 wall秒、attempt≤3600 wall秒、44/9小时阶段wall上限；收费/占用时段|原草案只是最多22/13200秒提案，未批准；本版两模式各22和新deadline不能冒充已有实验预算；金额/单价 UNKNOWN|
|P03|最高共同通过点的探索复核只用于稳定性；独立确认性预定 a=.80 steady、seeds300301–303；若不合法则不换点|固定负载是现在提出的待批准选择，不能跟随未来收益择点；确认性须单独授权|
|P04|新执行目录的统一三臂观察/日志harness、完整代码hash、原collector AST/路由重放、arm观察一致、deadline监督及清理、三臂轨迹匹配|原 stage2 只验证 Dynamic，旧OUT不可重用；本阶段只编协议，没有新 launcher/harness，也没有 GPU 烟测授权|
|P05|全部seed/RNG用途、原 generator/tag stream、原500 draw/60秒条件敏感性；固定顺序|记录内候选seed无结构化碰撞；外部/未记录开发史仍UNKNOWN；不声明新seed就是新输入或全局推理确定性|
|P06|原GPU UUID/name/driver/total显存；五模型各1副本、原registry/env；质量/容量/显存/模型版本；解释器/包/权重完整attestation与GPU隔离|历史身份和源hash已核对；liveGPU、future环境与所有checkpoint二进制hash尚未从冻结记录全部确认；有差异就NO_GO|
|P07|原质量/语义政策适用性、G-LID规范化类别及scoring证据的只读导出；原evaluate explicit False门控；所有结果保存规则|原G-LID flag缺失，缺失不是True；数值LID/VQA准确率SLO和非劣界未指定。不得自行新增；新字段仅诊断，不暗改原合法U规则|

P01–P07 是审核条目，审批时间/批准人/未来金额/准确率非劣界/额外延迟吞吐SLO均未确认，不填写虚构值。若用户在任何新结果前批准当前确定提案，可据批准冻结对应版本；若要改值，必须保留本版并新版本重冻结，而不是执行时自由决定。

## 执行前 Gate

探索启动 GO 必须同时：明确授权探索；批准 P01/P02/P05 的确定数值与预算；P04统一观察harness仅改日志且CPU检查通过；P06旧冻结输入hash与环境吻合并证明隔离/热态/预算；P07语义与scoring路径可核验；全部可能分支及确认性trace一次生成、hash冻结，source与新harnesshash锁定；deadline/ledger/不得覆盖或重试实现通过审查。任何未知或不满足项为执行 NO_GO，不能靠原Module30 GO绕过。

本阶段的输入完整性 PASS、JSON 校验 PASS 或CPU请求计数通过，只支持预注册交付，不代表上述执行 GO。

## 探索结束 Gate

必须报告每profile/level/seed/arm的原三态分类、硬原因、四项尾指标、警告、恢复、六任务覆盖、系统与worker分类、资源/热态/日志状态；报告原配对输入一致性、真实G/S分流、全部异常和跳过。

- **FINITE_WINDOW_BOUNDARY_BRACKETED**：同profile/同arm低点两seed非unstable，高点至少一seed科学unstable；报告离散括区间与复核原结果，混合seed仍显式标记有限窗口不确定性。不升级为长期容量证书。
- **COMMON_REGION_RECHECK_PASSED**：预选最高共同合法通过点，在3枚该profile复核seed中两arm均通过原合法性/恢复/匹配；仅对该点、该模式、600秒成立。G始终0时附 `POLICY_DORMANT`，不能调高负载强迫分流。
- **EVIDENCE_INCOMPLETE**：无共同通过点、缺组、复核不通过、基础设施截断或日志不充分；不填边界值。
- **NO_TRANSITION_OBSERVED**：注册范围全部通过或未见合法相邻转换；分别给出右侧未括定/缺失，不扩大扫描。
- **NON_MONOTONE_OR_SEED_SENSITIVE**：观测非单调/混合种子，保留完整矩阵，不对局部通过点推广单调“稳定区”。

这些标签可并存，不能互相覆盖负结果；没有完整执行不声称终点已达。过载救援仅为 S-only unstable、Dynamic非unstable的候选；baseline U=NULL，合法ΔU=NULL。至少三枚预定探索复核seed仅在规则选定点复核；若该候选未被共同通过点规则覆盖，则保持未经三seed复核，不增加救援专项slot。

## 进入独立确认性对照的 Gate

探索完成后必须停止，单独审核。未来确认性始终是 **a=0.80 steady、三枚300301–303**，不选highest共同通过点、不选burst有利条件。进入GO需：steady .80粗扫两seed两arm都原合法通过且匹配；steady按预选规则的三seed边界复核通过，并明确若复核位于其他level不能替代 .80证据；没有安全/日志/语义/环境缺口；fixed10路径通过原准入/结构校验（探索没测fixed10，不声称其稳定已证实）；新确认性三armtrace相同且预定请求覆盖>=72/任务；另批9次预算、P03与P04/P06/P07完成。

这些只允许发起受控确认性测试，不能预先证明fixed10或任何arm最终合法。若 .80不具备上述证据或预生成请求不足72，NO_GO/EVIDENCE_INCOMPLETE，不能临时换成 .60/.70/其他seed。确认性若不获批准不执行。后续请求改变确认性条件须另立协议，不能仍称本协议的独立验证。

## 确认性合法性和动态收益 Gate（未来适用）

单 arm U 合法保持原 `evaluate` 全条件。组比较需所有三 arm各完成600秒、cohort全成功、系统和每worker非unstable（recoverable必须显式恢复）、capability/质量/容量/route守恒/原safe与peak显存合法、清洁关闭释放；相同全部输入、到达、部署和非目标路径、同观察日志与环境，postwarm组三臂最大温差<=5°C；每task每arm正式成功>=72、Dynamic目标G/S正式成功各>0。原legal U与group证据资格分层记录，不能把缺组但某arm有U当三臂合法ΔU。

主要效应预定为 Δ_r=U_D,r−max(U_S,r,U_fixed10,r)，必须两个对照都合法；另报告D−S和D−fixed10。全部三组完整满足上述条件、每组Δ_r>+0.01，且三组效应的双侧95% Student-t区间下限>0，才为 `DYNAMIC_POSITIVE_REPLICATED_WITHIN_THIS_PROTOCOL`。这个门槛继承Module28独立协议，冻结用于未来仍待单独审核。负效应不换seed、不调ratio或Dynamic、不选最好profile、不重权重、不删最强合法对照。

任一非法/缺失/不匹配/未启用两路径为 `EVIDENCE_INCOMPLETE`，非法U/合法ΔU=NULL；不能叫动态负收益。全组合法但收益门槛失败为 `DYNAMIC_GAIN_NOT_CONFIRMED`，保留所有正负单组效应。部分组合法可保留各合法pair诊断和对应NULL，但不得只用该子集计算正式三组通过CI或报告独立复现成功。

CI须披露n=3、df=2、独立会话效应近似正态假设、最小单侧sign p=0.125；conditional block区间与输入重复不提供更多独立GPU样本。GO不授权正式优化器集成、更多负载、跨模型实验或长期稳定结论。

## 当前交付验收

七份文件齐备且互相一致；原DEFAULT_THRESHOLDS完整且四项AND没有变化；强度/模式扫描分离，seed/branch/预算/确认性预定条件可复核；所有未知数和新设定标待批准；历史文件集合、大小、SHA-256核验不变；GPU启动=0、optimizer修改=false。满足即可提交并停止等待审核，不自动转换为执行GO。
