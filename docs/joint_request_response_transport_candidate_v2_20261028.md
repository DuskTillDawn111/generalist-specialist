# 联合 request residual 与 response-clock transport：训练候选 v2

本交付包括可执行的真实训练候选、完整 request 到 CPU reference 的桥接和整 capture 留出场景诊断。它替代了“先独立预测一个连续 W，再寻找恰好等于 W 的经验轨迹”的不可用接法；并不取得主预测器、真实原执行队列、native assimilation 或最终效用收益资格。

## 保留的 v1 诊断

`local_joint_macro_candidate_v1.py` 与 `predictor-local-joint-macro-candidate-v1-natural-20261028-training02/report.json` 保留精确经验原子方案。三个自然域训练 capture 共 150 请求、690 forward 原子，按 owner、公开 exact input/pool 身份、原 prompt 长度、batch、forward index、terminal kind 和实际 BEGIN 的 processed peer 数分为 152 个 cell。整 capture 留出，local cell 支持分别为 223/230、226/230、226/230；连续 response 的精确原子支持全部为 0/230。这是经验离散分布的支持限制，不是可以隐藏的拟合成功，也没有据此发明 bandwidth 或补零。

v1 final 原子保留最后 forward 和完整 native END→READY tail 的联合信息。三个 capture 共 1380 次 query：690 次 after-host-END false、552 次 next-prehook true、138 次 observer-context-exit true。最后一组的发生时间已经在 READY tail 内。query 的来源 forward 与发生所属 macro 是两种身份，不能把全部 query 压入来源 forward 的 `[0,W]`。

## v2 实现

`joint_request_response_transport_candidate_v2.py` 读取经过完整 raw/review/macro 重推的三个 natural training 文档，以及具有二阶数值资格的 phase-aware macro 响应方程。每个 macro 的训练 reference work 为：

`V(t) = ∫[BEGIN,t] exp(-β · processed_peer_state(s)) ds`。

β 来自原联合 nuisance/interference 拟合；整 capture 留出时只使用另外两个 capture 的 β、baseline 和 donor。实际计算将 rate 固定量化到百万分之一的正有理数，因而 CPU reference 可以精确处理这个明确的近似模型。这不是 native GPU service clock。

每个训练请求保存一个完整的 joint vector：entry、每个 forward、OCR 粗 macro、完整 READY tail 的 `V_end` 与 log reference-work residual，所有内部 H/query/native-END/RESULT mark 的所属 macro 内 `V_at`，以及它们各自成对保留的 occurrence→reader、reader→dequeue、dequeue→processing 三段延迟。READY 的三段延迟与 actual-submit→worker-START 也同 vector 保存。没有独立重配 W、H、query 或尾部边际。

抽样只按公开 `(owner, task, exact public input/pool identity)` 查找支持。先等权抽一个 capture 作为不同输入间共享的 latent block，再在该 block 的对应输入请求中抽整个 vector；同 key 的多次请求使用 caller 预抽的 ordinal draw 队列。历史 request/cohort ID 只在独立 provenance 中出现，不用于选择耗时。原绝对发生时间不会作为未来事件重放。

桥接将这些 work 与 mark 送入 `macro_response_queue_reference_v1.py`。未来完整已知 arrivals、每 owner FIFO、生成的 processed peer 状态共同决定新的 wall time；不会读取 heldout 的未来 peer path 来积分预测时钟。query 保留独立来源 forward identity，实际发生在哪个 macro，就用该 macro 的 work 坐标触发。当前 stream/device 的新 world 标识是符号身份；原 query event/stream/device/host bracket 无损保存在 provenance，符号重命名不构成物理作用域认证。

RESULT reader receipt 单独作为 E2E 终点；collector processing 控制 peer 可见性。两者的时钟不可混用。所有延迟必须为明确的非负整数，缺字段、负 leg、bool 冒充整数及不合法 handoff 一律拒绝，不把缺数据补零。

## 新假设和已暴露的阻塞

响应方程最初拟合的是 observed peer path 条件似然。把 work 和 mark 坐标迁移到重新生成的 peer path，是新的生成假设，数值收敛不证明这个假设。另有两个明确假设：一个 shared capture latent 加条件独立的 request draws；原 joint 三段通知延迟残差可以随 vector 迁移到新的 wall time。这些均需进一步整 capture 和 fresh validation 检验。

原通知延迟直接迁移可能违反同 producer FIFO，尤其原 host mark 的小间隔被新响应时钟压缩时。v2 不修补这个结果：整 scenario 返回 UNKNOWN，该 scenario 的所有 offered 请求仍在分母。顺利完成的子集 MAE 不能代表全部负载，也不能被当作分流排序资格。

训练 raw 中的 actual-submit→START 一般可以包含 pipe/worker FIFO 等待，而不是纯粹固定 handoff。实际冻结 runtime 的 sender 线程不等 WORKER_READY processing 才继续 send；worker 自身 stdin 循环才串行。因此 reference 的“READY processing 后派发”是明确的替代策略假设，不是已核实的原执行队列合同。v2 的 forward 诊断只能在该策略假设下解释，不能宣称恢复了原队列。后续需要单独连接 parent send FIFO、pipe/worker FIFO 与 worker 实际 READY/START 边界。

这三个 capture 的同执行源码 SHA 集合一致，但它们仍只有三个统计 block，不能把 1002 个 macro 或 26×3 个 offered job 当作独立 capture。精确已知内容、当前固定 generation/stop/shape 路径之外的支持仍 UNKNOWN；G2 TTS 的 full-N 前瞻模板没有因此取得资格。

## 产物与下一步

当前最终 v2 场景诊断产物为 `artifacts/predictor-joint-request-response-transport-v2-natural-20261028-training03/`。training01 是旧 processing endpoint/zero-handoff 诊断，training02 是 READY reader 接口完成前的过渡产物，均保留且被 training03 替代。

若通知 FIFO 冲突频繁，下一候选应显式建模协议时序：从完整同 producer 消息链的四时钟反推 reader/dequeue/processing 三站的非负递推 slack，再按新 occurrence 递推三站可用时间。那是额外的队列时序假设，不能称为已识别的 CPU service，也不能伪装成 silent clamp；需保存 RESULT/READY、跨 request pending、producer lifetime/序号和未采集 parent 背景活动的限制。v2 冲突分母必须保留，不能为了通过 gate 直接修改旧产物。

仍未交付真实资源、质量、能量/峰值 provider，故没有六项效用、P95、ΔU 或最终分流收益结论，也没有 GPU 实验或 validation 标签读取。
