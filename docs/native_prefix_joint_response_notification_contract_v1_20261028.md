# Native prefix 的联合响应／通知延迟合同 v1

本文是 CPU 设计、有限状态精确参考和真实训练反例核验，不是已拟合的 native adapter。v59 simulated checkpoint、新主预测 gate、stage-response provider 和冻结 profile 源均不改动。不能把这次交付称为物理预测精度改善。

## 真实问题及输入时钟

对 reviewed natural training 的 464 个阶段逐项与 host partition 对齐，137 个在 BEGIN 的 **collector processing** 时刻之前已经 host END，而且同一 cutoff 的 END processing 尚未来到。137 个全部为 host_stage：pre_sync 46、post_sync 44、final_sync 46、cache_reclamation 1。不是 137 个 language forward。98 个甚至在 BEGIN reader receipt 前已结束，其余 39 个由 reader→processing 的进一步迟滞跨过了实际 END。这个比例只是该 training capture 的分解事实，不能直接作为泛化后的先验概率。

真实 profile:0/pre_sync 的 host BEGIN=6117520440008385，host END=6117520440063574；BEGIN reader=6117520440214566、processed=6117520440281327；END reader=6117520440247024、processed=6117520440297082。取 cutoff=BEGIN processed，阶段已经结束，且 END 已到 reader，但 collector 尚未处理。若沿用 reader-cutoff ledger 看得到 END；若预测特征按 processing-cutoff生成，则此时仍看不到 END。两种观察 filtration 不能混用。

既有 `visible_device_event_ledger_v1.py` 是 reader-receipt 口径，stage v2 的 covariate 是 collector-processing 口径。新 adapter 应使用明确命名的 processing 可见状态或经核实的 reader→decision 发布协议，不能只换字段名。最小方案固定 `availability_endpoint=collector_processing_time_ns`，保留 reader/dequeue/processing 三个原值用于延迟训练。旧 ledger 不改为“实际 GPU 进度”；stream 观察仍只有其记录 stream 的范围。

## 后验对象与似然

令阶段实际开始 B、host response W、实际结束 E=B+W；BEGIN/END 的 reader 与 processing 额外延迟分别为 DB/DE，实际可用时刻 AB/AE 由冻结 endpoint 定义。Z 包括共同 parent queue、调度/捕获随机效应、潜在阶段/stream 状态；通知丢失为显式变量，不用 None 冒充 0。

应拟合并条件化 **联合** `pθ(W, DB, DE, Z | 原公开输入/执行身份, 过去可见信息)`。cutoff c 的后验与这个联合先验乘上可见 prefix 的 likelihood 成正比：已收到通知需匹配其身份、payload occurrence 和 availability；未收到通知只要求其 availability>c 或该通知丢失。在连续模型中，已知时间的贡献是密度/区间 likelihood，不能当零概率精确匹配事件；有限 CPU reference 使用离散原子精确匹配。

BEGIN 可见、END 不可见时，后验必须保留两个分支：

- `E<=c<AE`：实际已结束，通知在途；潜在 host residual=0，距离 END 通知仍可大于 0。
- `c<E<=AE`：实际仍在进行；host residual>0。

第三个分支是 END 通知丢失；未识别丢失机制时不能伪造它的概率。`P(W>c-B)` 不是完整 likelihood。BEGIN 迟到时也不能把 W 的原点重置到 AB，不能把“BEGIN 刚可见”当作物理存活至 AB 的 delayed-entry 条件。实际 host stage 结束不等于 request RESULT、worker READY、stream event 完成或全部 GPU 完成。

通知延迟不得默认与 W、peer 响应独立。共同 CPU/reader 队列和捕获状态可同时影响响应和 BEGIN/END 延迟。独立乘边际分布还可能制造违反同一 producer FIFO 的不可能通知顺序。推荐初版为共享 capture/parent-queue latent effect 的联合模型，或保留完整配对 `(W, DB, DE, availability path)` 的训练核／经验联合分布；它们仍需训练支持、可识别性及 capture 留出检查。单 capture 的 137/464 不用来选择模型或给混合权重。

## 最小输入字段与来源

|输入组|必须保留的字段|允许来源与缺失处理|
|---|---|---|
|执行与模型身份|execution_context、natural/profiler domain、observer/worker/protocol receipt、response-model 和 notification-model receipt|冻结训练证据；缺模型或跨域迁移未经核实则 UNKNOWN|
|原请求与阶段身份|request_id、owner/task/input、原 arrival/route、phase_id/semantic phase、forward_index、parent/leaf关系|原公开 cohort + 当前可见 BEGIN；未知未来 stage 数/shape 只能来自合格 prior，不能读未来输出|
|clock/visibility|共同单调时钟域证明、原 cutoff、availability endpoint、reader/dequeue/processing timestamps、decision发布协议|实际可见元数据；不同 clock domain 无映射拒绝|
|可见 prefix|完整已处理通知、固定序号/原始 payload receipt、已消费 prefix hash、registry complete-through cutoff|只先读 availability，再读已可见 body；滑动窗口/缺事件时 absence likelihood UNKNOWN|
|BEGIN信息|实际 BEGIN occurrence、实际 shape/index、BEGIN availability|只在该 BEGIN 可见后使用；之前只用公开输入和冻结训练模板|
|END/RESULT/READY|各自 occurrence 和各自 notification availability、原协议分支身份|只有已可见事实进入观测状态；不可把一个边界代替另一个|
|stream query|event/stream/device identity、host query bracket、record bracket、true/false、实际 availability、scope|只限定记录 stream 事件；false 只说明 query 时刻，不证明 cutoff 时仍未完成；无跨流证明不构造全局cursor|
|延迟与丢失模型|联合 BEGIN/END/RESULT/READY 延迟、producer FIFO、parent queue、loss/detection domain、capture block|仅 training 拟合；complete clean capture 不能单独识别未收尾/永久丢失概率|
|动态响应|attained response virtual age、已知可见 covariate path、latent joint state、原累计权重/随机数状态|后验模型状态，明确不是实际 SM/GPU 观测；缺支持则 UNKNOWN|
|六项 prefix|原完整 cohort、原 deadline/drain、已知 result 明细、原 prefix 能耗/峰值、startup摊销是否已计|Q/T/L/P/M/E 原口径；缺 prefix 资源保持 UNKNOWN，不重复积分/摊销|

连续 END-notice absence 只有在 registry 在所选 endpoint 下完整、通知协议有支持时才有信息。只从成功收尾样本估计 delay，再把未收尾病例删除，会把通知缺失与长 response 混淆。应同时记录 capture结束/监控结束、完整 producer event 序号、reader退出/丢包/管道错误、未收尾 stage 分母和独立 loss 探测；不具备时将 loss-sensitive native filtering 标为 UNKNOWN。

## 可实现顺序与接口

1. 新建 immutable `VisibleProcessingPrefix`，验证源身份、完整序号、cutoff、已消费前缀不变与所有使用 body 的 processing<=cutoff。不同 endpoint 单独 schema。先做 prefix reducer，无主引擎 mutation。
2. response/provider 给出同域阶段先验和未见未来阶段／shape的支持。notification/provider 给出联合延迟/队列与显式丢失支持。两者必须共享同一联合状态或有验证过的条件因子化，不能各自独立抽完整时长再叠加。
3. `posterior(prior_state, new_visible_prefix, cutoff)` 输出带支持／权重的 **潜在状态集合**：已结束但未通知、仍进行、通知丢失（仅有支持时）。保存原 BEGIN 和 attained age。零支持返回 UNKNOWN，不把最后节点无限延期、不把缺 END 的节点强制复活，也不重置已采样 hazard/RNG。
4. 在每个 posterior world 中分设 actual-stage 和 notification 两条事件队列，按原完整 future arrivals/FIFO/真实 host 发布顺序生成未来。latent stage END 可以先触发该 worker 后续 host stage；parent-visible peer 特征只在相应 notification 实际可用后改变。按原执行协议决定下一请求 START；阶段 END 不能释放整个 handler，也不能仅因 READY通知没到就臆造 worker 尚未 READY。
5. stream-query 循环、post_sync、host gap、response flush/READY尾部联合生成。sync wall 若已包含设备等待不能再次加 GPU wait；host wall 与 stream elapsed 不是相加关系。host 137/464 反例不能直接拿来训练 device service。
6. 先给新仿真器接纯 CPU reference 的 posterior branching/序列化，证明 cutoff causality、latent-vs-observed分离、资源／六项 prefix守恒。原 v59 只作模拟恢复对照，不能直接把其旧 holder/rate cursor当native实际状态。
7. 三 capture 起的 training-only拟合与 whole-capture留出，固定 joint family/超参/visibility与cutoff后，再独立验证 response residual、END availability、后验 mixture校准和完整队列/P95/ΔU 排序。此前不接主预测 gate、不报告分流收益。

## 最小 CPU 反例及交付边界

有限 joint reference 给两个各占 1/2 的合成世界：B=0、BEGIN在30可见；世界A W=10、END延迟40，在50可见；世界B W=50、END延迟10，在60可见。在 cutoff35 只有 BEGIN 可见。真实后验支持两个世界，已完成质量1/2，host residual 均值7.5；两个世界距离END通知分别15和25。错误的 W>age 强制只剩世界B，改变 residual 且错误声称实际未完工。

另一个用同样 `(W, DE)` 配对的例子，BEGIN在5可见：正确联合的已结束质量为1/2；把两边际独立交叉，四配对中一项 END20 已可见而与 prefix 矛盾，剩余归一化的已结束质量变成1/3。这个区别无需任何 GPU 实验即可精确核对。

`response_notification_posterior_reference_v1.py` 只实现上述有限合成世界的 conditioning、availability先过滤、明确loss/no-support UNKNOWN和reader/processing差异。7项CPU反例不是拟合概率、真实native adapter、条件独立性证明或主provider资格。完整合同仍须真实联合训练和独立验证。

## 与真实 carry-in gap 对齐的 macro 响应域

root 已实现 `composable_response_macros_v1.py`，natural training 输出在 `artifacts/predictor-profile-training-evidence-v1-natural-20261028-review02/response-macros-v1-03.json`：50请求、334个互斥macro，完整覆盖worker START→READY。这里作为训练域合同的最小可执行结构，不是完整cohort轨迹查表或已训练provider。root/recovery 的 `native_gap_milestone_contract_20261028.md` 和 evidence-v3 证明 prefill→decode1 gap 大部分有既往已发射kernel carry-in与cudaStreamSynchronize：G2 profiled gap5.003s、owner kernel overlap union4.923s、sync API wall4.844s；text#0为3.022/2.928/2.962，text#1为1.970/1.888/1.937。三列重叠，不能相加或换算SM占用，自然域也不能直接采用这些profiled纳秒。

第一版联合响应模型应建立在这个互斥macro域：worker START→first forward BEGIN为entry；每个forward BEGIN→next forward BEGIN为一个macro；最后一个forward BEGIN→native END为final macro；native END→worker READY为tail。OCR保留明确粗macro。按phase/shape/input与自身不断变化的已处理peer covariate组合，不按整cohort身份查询整段耗时。

macro内原forward host END仅为内部里程碑H，**不是**macro响应结束E。query true只对记录的stream event提供证据，既不是macro END，也不是全部GPU完成。macro可能尚有selection/stop、CPU/设备交互、下一个输入准备或model-specific finalization，不能在H/query true时将其资源或handler置空。原BEGIN起的attained age继续累计，遇H或通知时不能reset。此前联合后验公式中的E必须附带 `response_domain`：对host_stage是该call END；对forward macro是next BEGIN或native END。两者通知不能互相替换。

实际profile:0 prefill truequery发生在6117520504448747，nextBEGIN发生在6117520504525361，但query processed在6117520504635803。查询的发生属于旧macro，通知已跨入下一macro。联合状态必须保留全request的pending notification队列；不能切换macro时删除旧query，不能把其true结果回填到旧macro时刻的可见covariate。macro extractor已保留这类occurrence/processing分离，新的8项独立CPU反例验证hostEND不能提前完成macro、latequery跨macro、无terminal通知的双分支、禁止query充当macroEND，以及G2最后forward truequery后仍保留native尾部、旧query不清除下一macro、future body不可访问、raw指针无损。

G2最后macro显式包含TTS-bound CUDA scalar扫描：full sequence `N=input+generated`，核心扫描工作为2N次比较/潜在scalar读取。recovery在18个profiled训练请求中核对到短N394、长N4670，完整exit的DtoH/sync=2N+6、CompareEq=2N+5、all kernels=2N+35。这些额外常数是该执行身份的诊断计数关系，不是通用时间模型，不能用固定exit常数或在macro外再加一份等待。前瞻使用N需要已知input长度与有资格的冻结生成模板，或由联合生成/stop分支产生；不能从未来RESULT取generated长度，forward count也不能冒充token数。modeling源码当前receipt与缓存一致不等于有原执行时的历史源码快照，资格限制随证据保留。

后续训练表至少增加 `macro_id/domain/terminal_kind`、当前及下一phase/forward index、qualified current shape、internal milestone原始身份/host bracket、pending跨macro通知、generation/stop来源和qualification、last macro的full-N工作声明/receipt与UNKNOWN原因。macro03现已无损保留每个内部milestone的raw event和capture SHA/request/event ordinal指针，event_id/stream/device/host bracket依原event证据保留；原raw缺失的作用域证据不能由指针补出。macro covariate已改为专门的processed macro prefix：query true和host END不清除最后forward，下一forward BEGIN更新phase，只有native END通知已处理才清除。这个可见状态只是最后处理的macro身份，不证明实际SM活跃或物理完成；旧stream pending path单独保留为diagnostic。仍没有已取得资格的G2 prospective TTS模板，不能因当前training label覆盖完整就声称该模板已取得资格。

可实现的下一步是先以这334个互斥macro为training label域，联合建模macro响应、内部H/query及通知；外部entry/tail仍在同一worker顺序中。virtual response clock按当前processed peer状态积分，未来milestones由同一个联合模拟器生成。成功的macro拟合也不证明SM service、资源free、native assimilation、P95或最终ΔU排序资格；仍按上文的capture留出与完整future-arrival队列验证推进。
