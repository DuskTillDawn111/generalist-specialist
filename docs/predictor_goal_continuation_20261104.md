# 预测器 goal 接续：全池重复、运行配置与 OCR 边界

## 全池主机诊断完成，API定位探针已启动（2026-11-04T14:55:38.617069+00:00，优先于下文）

- five-model-fullpool-host-diagnostic-v1 四轮全部完成，1280请求，跨模式/跨轮输出一致。旧controller session4919/PID246079已终止，其state=host_diagnostic_failed是**旧v1跨时钟分析拒绝**，不是采集失败。未重采或覆盖失败。
- 独立v2分析完成：description-v2.json、analysis-v2.log、resolution-v2.json。原10%mean/25%P95通过，8路径mean3.11%–6.46%、P956.43%–15.45%；307个CPU>wall的有符号跨时钟观测全部保留，不解释为负等待。完整捕获、原规则、v2规则和来源SHA均保留。
- two-trace-brackets-v2.json涵盖31698调用、1152 bands和576重复输入band groups；7组raw constant minimax>.2（ASR、G2/asr、G2/image、image）。CPU系统态停顿解释了部分ASR/image波动，但不是所有G2变化的充分解释。此为development诊断，不授予训练/独立预测资格。
- **当前live：session57541 / controller PID249811**；日志 /tmp/gsdeploy-asr-cupti-host-probe-v1.log；目录 five-model-asr-cupti-host-probe-v1。单次320请求，准确重放trace-round1的完整五模型顺序，仅ASR worker运行在Nsight cuda/nvtx/osrt下，GC/host计数继续。已经确认Nsight子进程249896运行。GPU仍经原scheduler+collector双锁，不重启。
- 探针split=development_cupti_host_diagnostic，not_for_training，保留profiling扰动；自动收集→导出SQLite→按request/role/ordinal+同线程NVTX关联CUDA API。API嵌套用区间并集，不把sum当总CPU/GPU时间，不把CUPTI时钟直接映射为perf时钟。2项相关软件测试通过，所有源码冻结。等待真实收据，不能先断言CUDA事件分配导致长停顿。
- Nsight旧pilot API概览和实际SQLite SHA核对保存在host-mechanism-reference-v1/legacy-cuda-api-summary.json；旧域中Memcpy/launch等也有长调用，不能迁移为当前原因或训练数据。
- goal active：阶段重复性尚未解决，phase-pair未采，完整FIFO/energy/负载/副本/独立资格及优化器比较仍未完成；确认6110491/92未用，总效用不改。


## 最新全池主机诊断（2026-11-04T13:55:00.821512+00:00，优先于下方历史记录）

- passive-GC 四轮诊断已完成，696 正式请求，输出跨模式/跨轮/重复输入全部一致，GC coverage完整；原 observer 10%mean/25%P95门槛通过（mean1.37%–7.13%，P952.75%–21.02%）。见 five-model-passive-gc-diagnostic-v1/description.json。
- 两个trace中 ASR first-forward 仍有无法用同输入常数覆盖的波动：1272-141231-0018 为111.815 vs53.715ms，GC重叠均约0.54ms；251-118436-0011最大92.449ms那次GC=0。前者 prepare host59.495ms/device5.010ms且整请求system CPU59.712ms；后者prepare host36.156/device7.942ms且system CPU39.359ms。这些观测不支持以GC解释全部异常，尚未定位具体系统调用。
- 同一小诊断中G2长异常出现在未记录GC的control，trace尚未复现；短cohort缺少原32个G2 OCR等完整历史，不据此排除全池GC影响。完整band报告保留494个mode/input/band groups，4个raw constant minimax>.2；该数字未做root-order修正，不替代资格gate。
- 新流水线已实际启动：**session4919 / controller PID246079**，日志 /tmp/gsdeploy-fullpool-host-diagnostic-v1.log。目录 five-model-fullpool-host-diagnostic-v1，ABBA四轮各320请求，准确重放旧solo两轮顺序（原6111181/82，仅development重放，不宣称独立验证）；control/trace均GC on，仅trace增加thread CPU/rusage外层调用括号，无额外同步。7项相关检查通过。等待本流水线，不重复启动。每轮预算2400秒，GPU继续双锁串行。
- goal active。旧主session18835、99705和新GC session24505均已终止；phase-pair仍未采集。完整预测器/FIFO/energy/负载副本域及独立资格仍待完成，总效用改进延后，确认6110491/92未用。


## 当前接续状态（优先于以下历史记录）

Scoped 两轮全池已完成，输出一致，但 ASR first-forward 与 G2/image 的两个 band 未通过原 20% 基线门槛。旧主控制器与 phase-pair 等待器均已退出，后者在门禁处停止，未采集共跑训练。现已启动 four-arm passive GC 诊断（five-model-passive-gc-diagnostic-v1，session24505 / PID244016）：同轮顺序匹配的 off/on/on/off，各174请求，13项软件检查通过。该实验仅检查 GC 与主机停顿的关联，保持实际 GC 策略不变。实时状态以 NEXT.md 与 state.json 为准；总效用改进继续等待完整预测资格。


2026-11-04。用户暂停原聊天后，明确要求在当前聊天接续原 goal。当前 goal 保持 active；预测器验收之前不改总效用、质量锚点、safe_factor、rho 或准确性门槛。

## 已完成

复用原后台采集进程，没有重启或重复采集。五模型全池两轮 solo 全部完成，640 请求、320 个路径×输入组合，跨轮输出全部一致。自动重复诊断与实际质量评分均完成。

四个专用模型的聚合质量保持旧目录值。G2 的 OCR 原生损失为 0.0391723584，旧目录为 0.0389212535；另外三项 G2 聚合质量与旧值相同。质量锚点没有改变，新运行配置不能无条件继承旧输出质量。

两轮完整 handler 的逐路径最大比在 1.058–1.450 之间；text 最大比1.135。OCR detection 原分段有4/32输入的最大/最小比超过1.5。仅靠每输入一个固定时长，无法让这4个输入的两次该阶段观测同时满足 `abs(t/D-1)<=0.2`。这不证明更丰富状态模型不可能成功。

## 新运行配置对照（进行中）

原 reconstruct worker 在加载前对全部模型启用严格确定性；新全池 text 平均 handler 约891ms，旧四S约573–577ms，但旧数据的驻留组合和历史不同，不能归因。

新增隔离 worker 与四轮诊断，五模型驻留、权重、生成规则、输入和同轮顺序一致，只比较 strict 设置作用范围：

1. all-strict round1
2. G2-only-strict round1
3. G2-only-strict round2
4. all-strict round2

每轮120请求，5个固定均匀索引输入/路径、两次solo重复及少量共跑；种子6111161/62，使用原GPU scheduler锁，等待旧solo完成后启动。G2始终使用strict和加载后ExLlama threshold1；专用模型在scoped条件使用原fresh-process默认。READY保存加载后的torch确定性、cuDNN、TF32、CUBLAS实际值并校验；不是仅靠计划声明。

该对照只能评价设置组合的影响，不能分别解释单个flag因果作用。数据标记development_runtime_scope_diagnostic，不用于训练或独立资格。若采用新配置，还需匹配观察器开销检验、全池训练、共驻与独立检验。源代码/计划已冻结，队列期间不修改。

## OCR 因果边界候选

保留的OCR pipeline在det.forward返回后，还需后处理、将检测框`.cpu().tolist()`、排序裁剪并准备识别输入。因此det.forward的异步host返回不是检测工作全部被消费的边界。

在已有完整阶段和语义事件上定义候选分界：第一次ocr_recognition的host进入时刻。第一段包含检测至首次crop准备，第二段包含第一次识别至尾部；不新增同步、不改原PHASE、不把CUDA event相对时钟伪装为host绝对时钟，两段均仍是host/transfer/CUDA混合工作。

在两轮训练数据上：

|分段|最大跨轮比|固定基线无法同时满足20%误差的输入|
|---|---:|---:|
|原检测段|2.107|4/32|
|新检测至首次crop段|1.458|0/32|
|新识别及尾部|1.309|0/32|

这是训练后的模型设计证据，不是独立准确性结果。旧root-order修正对原检测段的最大work残差仍为34.25%，没有消除其问题。候选边界与完整native-call对照都应在最终运行配置下检验完整共驻/FIFO/work/energy，不能只选一个更稳定的阶段就授予预测资格。

新增5项运行配置测试、相关旧5项测试和3项OCR边界测试通过；测试确保G2不被放宽、S不继承strict、加载后漂移拒绝、边界连续且不裁剪非法时刻。这些是软件证据。

## 文件与接续

- 运行状态与下一步：[NEXT.md](../artifacts/activation-gpu-20261104/NEXT.md)。
- 接管及原数据哈希：[handoff-to-utility-chat-20261104.json](../artifacts/activation-gpu-20261104/handoff-to-utility-chat-20261104.json)。
- 全池重复：[description.json](../artifacts/activation-gpu-20261104/five-model-reconstruct-solo-v1/description.json)。
- 实际质量：[quality-description.json](../artifacts/activation-gpu-20261104/five-model-reconstruct-solo-v1/quality-description.json)。
- host/CUDA与时长稳定性：[boundary-stability.json](../artifacts/activation-gpu-20261104/five-model-reconstruct-solo-v1/boundary-stability.json)。
- 新OCR候选：[completion-boundary-diagnostic.json](../artifacts/activation-gpu-20261104/five-model-reconstruct-solo-v1/completion-boundary-diagnostic.json)。
- 四轮运行设置计划：[design.json](../artifacts/activation-gpu-20261104/runtime-scope-diagnostic-v1/design.json)。

下一步先读取四轮配置对照的实际结果，再决定运行配置和训练设计。当前仍无完整物理预测资格、正式总效用收益或稳定非网格优势。

## OCR 候选接线与后续采集门禁

已新增隔离的 `activation_completion_provider.py`、`fit_activation_completion_bands.py` 和 `validate_activation_completion_bands.py`。它们保留原共享 FIFO、阶段干扰回归、独立重复、完整支持和能耗门槛，仅切换 OCR 观测边界并要求显式边界策略。原始 PHASE、旧拟合及原验证入口保持原样。预测仍不读取实际未来阶段完成时刻或最终 token 数；实际阶段仅用于评分。

`completion-template-solo-feasibility-v2.json` 用可重跑脚本审计已有 all-strict 两轮全池基线。在原 root-order 修正后，15 个 family/task/band 路径的最大 solo work 残差均 <=20%；OCR 第一段最大19.20%、P95约18.88%，并非稳健裕量或独立验收。该结果不能迁移到另一个运行配置。

11 项完成边界/原相关检查及3项新观察器计划/门禁检查通过。`completion-band-software-v1/` 保存源码快照、成功日志与早期测试fixture失败日志。新增观察器入口要求先完成四轮 runtime 诊断；对照与semantic两组只能改变观察模式，其余上下文及同轮请求顺序一致。任意观察器门槛失败会阻止后续全池训练采集。

新流水线 `continue_activation_scoped_candidate.py` 可以在指定候选完成观察器检查后，自动采集两轮全输入solo、检查实际输出质量与完成边界基线；它不会自动授权共跑预测资格或正式效用。当前需先读完四轮runtime对照，才选择运行配置。

共跑设计仍需遵循：使用选定上下文的全新baseline；三次独立重启覆盖同一批交叉条件；实际阶段暴露和不可达条件全部保留；在固定输入下改变peer组成、阶段及数量，避免输入与干扰条件混淆。五模型单副本有22种不同owner的路径对、28种三owner路径组合、17种四owner和4种五owner组合（G2的四任务不会当作四个并行owner）。先按训练solo确定合法的阶段错位，再审计真实支持顶点与数值秩，针对缺支持补采；不能为降低采集量丢弃偶发顶点或放宽work门槛。尚未启动新五模型大共跑训练。

## 最新运行选择（2026-11-04T12:21:42.478330+00:00）

四轮runtime对照已完成，全部480请求输出一致。选择g2_strict待验候选的依据和限制见 `runtime-scope-diagnostic-v1/candidate-selection.json`。文本和ASR的solo加速在两组匹配对照中重现；OCR的大幅跨轮变化在两种设置下都出现，不能归因于放宽严格设置。新四轮观察器及门禁后的全池两轮pipeline已经实际启动，session18835；读取NEXT.md获取当前状态，不重启。共跑训练、独立预测资格、完整episode资格和优化器比较仍未完成。

## 支持审计与双模型训练探针

`specialist-base-forward-band-fit-v2/support-audit-v1.json`已从原raw重新推导并逐项匹配原fit的normalized_runs。旧四项缺重复来自同一实际四owner阶段组合：ASR first + image native + OCR detection + text continuation，在overlap-round2的full:image:25里只出现11.399784ms。各初始Jacobian均满列秩，不能将其表述为已通过非线性拟合或工作量验收；实际旧text/OCR work残差失败仍成立。

新增支持审计保留每个实际顶点，包括极短暴露，报告独立run数、请求例子与持续时间；结构允许但未观测的状态只表示缺测，不声称物理可达。软件检查覆盖1ns稀有状态不被丢弃，以及同一G2 owner不能同时处于不同任务/阶段。

新增 `activation_phase_crossed_design.py` 根据固定training mean与预排owner请求序号计算阶段中点错位时间，实际交集仍须从观测检验。五owner阶段/任务组合数为双owner74、三owner164、四owner168、五owner64。没有将G2四任务拆成四owner。

`five-model-scoped-phase-pair-v1` 已启动等待器（exec session99705），仅在新observer与两轮full-pool基线完成且满足门禁后准备并采集74种双owner阶段组合、三次独立重启，各153请求（148探针+5个idle前anchor），种子6111191/92/93。每轮保留1/3/12/30/90s空闲。输入是每路径training native时长的中位输入，先检验阶段law再扩大输入和owner域。会自动拟合、输出精确支持审计；只有fit全通过才回放。该部分不会替代后续三至五owner、交叉输入、FIFO突发、独立验证和完整episode能耗资格。日志/tmp/gsdeploy-scoped-phase-pair-v1.log；state与pending-plan有冻结SHA。不要重复启动。

8项新软件检查全部通过；源快照与日志见phase-support-software-v1/。这仍不授GPU预测资格。

## Scoped观察器已通过（2026-11-04T12:42:41.632126+00:00）

四轮656请求完成，匹配输出和跨轮输出全部一致；逐路径mean偏差1.57%–4.59%，P95 4.77%–15.14%，原门槛保持不变。单条较大偏差保留，不把P95门槛通过当成所有请求稳定。主控制器已实际开始scoped全池solo第一轮，两轮完成后下游阶段探针等待器按门禁继续，过程仍不授予完整预测或效用资格。两条live session为18835与99705，读取NEXT.md和进程后续跑，勿重启。

## Scoped全池首轮实测（2026-11-04T12:58:14.444580+00:00）

320请求完成，所有输出与旧all-strict完整首轮对应profile一致。四S质量仍同目录；G2 OCR保持已记录的新reconstruct损失.0391723584，并没有因S恢复默认设置再发生变化；其余G2质量一致。此处只报告完整首轮输出质量，次序未匹配，不将全池耗时差当作配置因果效果。第二轮正在运行；独立重复、baseline work、共跑及FIFO/energy资格仍待完成。runtime-scope-diagnostic-v1/plots-v1保存匹配对照图和数据来源。
