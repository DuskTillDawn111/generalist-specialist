# 共驻画像与预测验证：2026-10-27

本轮按“真实画像 → 冻结后的独立预测验证 → 干扰机制 → 最终分流”推进。GPU 已获用户授权。**统计模型已通过已见输入的局部服务探针验证；全任务、全过程队列与 ΔU/排序尚待验收，不能据此声称最终分流收益。** Goal 继续跟踪，以下为可复查的中间事实。

## 固定环境与证据

- RTX 4090，49140 MiB；固定 `text#0 + text#1 + generalist_2#0 + ocr#0` 四个独立常驻进程。
- 沿用冻结的原生模型、输入池、生成设置和缓存回收顺序。显式设置 OMP/MKL/OpenBLAS 线程为 1。
- 采集真实 worker START/PHASE/RESULT 单调时钟。NVML 20 ms 轮询板级功率、全部拥有进程显存及外来 PID。保留采样峰值的测量限制。
- 四进程启动及 19 个预检请求全部成功。此预检使用旧源绑定，只作启动调试，不作为训练画像。
- CUDA/NVTX trace 成功；Nsight Compute 返回 `ERR_NVGPUCTRPERM`，SM／DRAM／L2 需求均保持未知。不能据此声明计算／带宽互补已成立。

证据目录：`artifacts/predictor-gpu-20261027/`。各正式采集 context 绑定不可变源码快照与输入/worker 摘要；trace 与自然计时分开，不池化拟合。

## 第一轮训练与独立验证

`native-training-v3` 含 144 个训练批次、354 个请求，全部成功。覆盖 S0、S1、G、O、SS、SG、SSG、SSGO，输入长度分层、0/50/200 ms 双向启动错位、每条件两个训练重复。画像中的输入仅含 public identity/size，预测不读取答案。

`native-validation-v3` 为另外启动、在模型与预测封存之后执行的 144 个验证批次、354 个请求，全部成功。验证同时包含已画像输入的新重复及完全未画像输入，二者分别报告。原始验证数据未用于训练模型或调整拟合门槛。

正式报告 `native-validation-v3/report-v2/report.json` 为 **INCOMPLETE**：354 个请求全部 UNKNOWN，其中已见输入 177 个、未见输入 177 个；两个任务的预测 P95 均未知。已见输入失败原因是缺少完整可识别的阶段上下文，未见输入另外缺少 solo profile。该结果不是耗时误差为零，也不是验证通过。

短程 `native-trace-v2` 含 8 个正式请求、119944 个真实 CUPTI kernel；32 个阶段锚全部匹配，最大边界映射残差 88.504 微秒，无未归属 kernel。四 worker 共同 kernel 重叠约 93.044 ms；按映射残差侵蚀区间后仍约 31.390 ms。这只是时钟敏感性检查，不是统计置信界。宏调用区间包括主机发射、传输和同步，不等于纯 kernel 时长。

## 当前预测器的实际瓶颈

精确细阶段模型已见 549 种上下文，仅保留 7 种完整支持上下文，且这 7 种全部缺少对应短区间显存样本。12 组 GPU 阶段全部欠秩，没有一组通过。选定 profile 域的上下文组合计数约 39.3 万。该模型的首要问题是可识别性和覆盖，不能靠把 UNKNOWN 当零解决。

训练前声明的两宏阶段备选，将上下文合并为完整 family/phase 多重集；所有五组虽满秩，训练最大工作重构误差仍超过原 15% 门槛：G2 主阶段约 75.1%、OCR 34.8%、text 主阶段 56.1%。没有支持上下文。未提高门槛迎合数据。

进一步仅在训练集开发的一段 handler、逐输入独立模型也未解决问题：九组仅一组通过原残差条件，而该组条件数约 26795、留一系数最大相对变化约 61.8%。这说明维度过高与状态过粗同时存在，整个请求的一条常速系数不足以解释真实干扰。

训练观测中，G2 在 SG/SSG/SSGO 的 handler 减速中位数约为 2.045/2.751/3.045 倍；OCR 在 SSGO 约为 2.117 倍。这些是固定样本与错位组合的训练描述，不能替代独立预测精度或因果资源分类。

## 后续开发及禁止混淆的口径

已新增独立语义观察器，使用原生 PyTorch hooks 观察第一个 language forward 及后续 forward，记录维度、cache 对象存在与单调时钟，添加 NVTX；不替换 forward/generate、不读取 token 值或标签、不新增逐 token CUDA 同步。新的语义 worker 和 v4 数据具有独立执行身份，不混入 v3。

v4 短预检发现 text 的 PEFT 外层 `generate` 绕过外层 module hooks，两个专用 worker 的语义记录为空，G2 正常。已停止尚未产生请求的 v4 训练，保留失败工件，不从 handler 猜测 prefill。下一版先修正实际 language module 所有者并通过真实 GPU 小预检，再开始训练。

后续模型应以真实 prefill/decode 进度和完整多路交互为状态，降低细控制阶段的组合数。阶段标签不证明 GPU 实际执行；仍须用 CUPTI 关联验证。未见输入缺少 solo 基线仍为 UNKNOWN。

完整在线队列/P95、六项 ΔU 符号与候选排序、默认共享/整请求准入、最终四组分流对照均尚未通过物理验收。最终比较继续使用六项效用及原稳定性规则；不能在当前失败门槛上宣称最终分流收益。

CPU 同域 DFS 与有证明的保守界在已声明的小型构造域一致，最优 ties 全部一致；弱界未发生剪枝，不声称提速，也不声称全模型部署域物理最优。

## 后续真实进展：v5—v8

PEFT hooks 已在真实 v5 trace 修复，解析到实际 LlamaForCausalLM；G2 为 Qwen3ForCausalLM。四请求预检观测到 prefill/decode，31 锚全部匹配，最大边界映射残差 78.726 微秒。Python forward 返回不代表 kernel 完成：一个 G2 prefill 的 GPU 工作延伸至返回后约 51.6 ms。语义阶段只能作为混合墙钟统计状态，不能假装纯 GPU 阶段。

`semantic-training-v5` 144 批次、354 请求成功。细阶段及严格可识别语义模型仍未通过；另行开发固定 alpha=.1、数值速度范围 [.05,8] 的正则化统计模型。其系数是预测参数，不是唯一可识别的物理干扰参数；p90/置信界未校准。

`statistical-validation-v6/report/report.json` 为真正封存后独立运行：已见输入 **177/177 覆盖，服务时间 MARE 3.90088%，分任务 P95 平均相对误差 0.77691%**，通过预声明的已知输入局部门槛。9 个请求包含阶段插值。另 177 个未见输入全部 UNKNOWN，整体覆盖率 50%，全域仍 INCOMPLETE。该 P95 终点为 worker END，未包含父进程收到 RESULT 的尾部。

资源独立报告 `statistical-resources/validation-v6-report.json`：mean 整窗能量误差 8.22%、显存采样峰误差 9.34%；max 场景能量误差 10.70% 未达 10% 描述门槛。经验 max 显存 31666 MiB 仍低于独立运行 33726 MiB，不能视为安全上界。沿用原安全因子且只应用一次。

`background-training-v7` 固定原四驻留、264 训练批次及 690 请求成功，包含 G2 text/image/asr 与 text×2、OCR 的共跑。新服务模型独立拟合完整 family/task 与 family/task/phase 组合；资源模型也直接使用完整任务组合的功率均值和经验显存峰，不再合并不同任务。资源训练含完整 24 个活动组合（含 idle），15412 NVML 样本，最大采样间隔约 70.7 ms；经验峰仍非连续安全界。`background-validation-v7` 已封存独立服务预测并开始单次验证，未使用其观测调参。

原生全过程采集器 `native_flow.py` 与独立 `flow_validation.py` 已通过 CPU 路径/口径检查：固定到达和输入、持续 enqueue、不逐批排空；等待用真实 worker START，完整 E2E 用父进程 completion；Q 用每任务错误总量/参考单位总量，E 包含原启动摊销且排除 drain，M 含所有 owned resident 的采样峰。仍需实际流量检验。

`flow-pilot-v8` 在 GPU 启动前的封存预演，四个候选全部 UNKNOWN。只读训练域审计定位到四个 target-context 缺项：text 的 `e0ff7b…` 与 OCR 的 S+O、SS+O；OCR 的 `X51005442341` 同样两个组合。不是未见输入，也未以验证误差选择补域。新 `domain-training-v8` 预声明三已知输入、23 非空任务组合、两个重复共 138 个训练批次，补采后另封新模型和新队列验证，旧模型与失败工件保留。

真实模型的 CPU 同域 DFS/保守界核验 `cpu-search/flow-pilot-v8/summary.json` 同样保留 4/4 UNKNOWN，无最优解、无认证；空 ties 一致不能算通过。尚未启动四组最终收益确认。

## v9：完整流量暴露了决策误判

补域 v8 合并 402 个训练批次、1014 请求，15 个 target 的任务组合均完整；11/51 个语义组仍欠秩，继续按正则化预测参数解释。另行封存的 v9 周期流仅使用该已知输入闭合域，4 候选各 181 请求、共 724 请求均可预测。CPU plain/bounded 搜索重新模拟各候选，唯一模型最优均为 `shared_spill_high`，不声称物理最优或剪枝提速。

v9 四组真实请求全部完成且质量结果完整，但独立 `flow-pilot-v9/gate.json` 为 **FAIL，15 项失败、0 项 UNKNOWN**。ΔU 符号仅 1/3 正确，候选 pair 排序 3/6。相对 balanced：

| 候选 | 预测 ΔU | 实测 ΔU | 服务 MARE | 完整 E2E MARE |
|---|---:|---:|---:|---:|
| skewed | 见封存报告 | 见封存报告 | 8.37% | 15.22% |
| spill_low | -0.07527 | +0.14743 | 14.94% | 31.23% |
| spill_high | +0.21819 | -0.23350 | 22.16% | 30.14% |

balanced 服务/E2E MARE 为 5.84%/6.51%；skewed queue MAE 为 1.772 请求，超预声明 1 请求门槛，多项逐任务 P95 误差超 20%。只读分项诊断表明两次分流反号主要来自 P：low 的 ΔU 预测误差中 P 贡献约 -0.1998；high 约 +0.4197，而 E/M 差值误差远小。良好的受控服务探针拟合不能代替连续队列和效用排序验证。

原稳定性独立重构均为 recoverable_congested 且完整恢复。旧 catalog 的 rho 算术满足原 0.8 阈值，但缺当前执行/控制模式的容量画像适用性证据，正式资格保留 UNKNOWN，不宣布最终净收益。

下一步已预声明并启动 `continuous-training-v10`：两个全新训练 seed（627401、627402）×四种静态路线，共 8 个 90s 连续到达训练流。原 v9 不用于拟合或调门槛；新训练使用提前写入的 phase/split/created_ns 与 plan SHA 派生训练来源。将从新训练数据解释持续到达下的服务状态和相位反馈，并另开新独立验证，当前不进入最终四组确认。

## v10 连续训练与 v11 前瞻协议

八条连续训练流继续按预声明顺序单次执行，不自动重试或替换seed。严格 mapper 仅根据启动前冻结的 training plan 派生角色，原raw不修改。v9只读回放724请求及2848阶段全部匹配原seal：observed-context替换后G2低/高分流服务MARE仍22.84%/19.87%，说明条件服务失配与轨迹反馈并存；父接收尾部不足以解释效用反号。v9永不用于拟合。

新训练协议 `background-statistical/training-v11-fit-protocol.json` 冻结请求行权重及8个整流留出、2个整seed留出。整seed用于开发检查；两者均为回顾性训练交叉检查，绝非新的独立验证。长flow按请求贡献回归行，旧目标函数高方差下的size-bias单独审计，不把统计系数视为物理参数。备用joint residual provider已有9项CPU测试及独立审阅，当前不启用：两seed不足以校准run级/跨请求依赖。

`flow-validation-v11/plan.json` 已预生成新seed627501、四候选各181到达；沿用原完整覆盖、20%服务/完整E2E/逐任务P95、queue MAE≤1、全部ΔU符号/排序正确门槛。新模型证据树将先由 `flow_trained_validation.py` 独立入口递归重验，再调用冻结的seal/report/gate。尚未启动v11 GPU。容量迁移保持原rho/g(n)/阈值，目标seed无measured key时显式保留原resolver UNKNOWN，统计转移不能伪装目标实测；H90不得自动迁移H600。

八条连续训练已全部成功，新增1448 measurement请求。v11完整模型使用410 traces/2462请求，15输入/51语义组，直接与传递来源均无验证观测。8整流+2整seed留出全部完成，整seed报告1448/1448可预测，两个seed的ΔU符号及pair排序均100%；56项服务/E2E/queue/逐任务P95检查仍12项超原门槛，故训练CV不通过，更不能称独立验证通过。

新duration-weighted目标的训练对照协议已封存：w_i=baseline/T_i，不重新归一化权重、alpha=.1与正速度数值范围不变。其alpha0单context解预测算术均值；实际固定ridge仍可能有prior bias，多context为条件统计拟合而非因果物理模型。仅新增两个whole-seed模型，使用固定字典序选开发模型（决策错数→原时间门槛失败数→归一误差均值，完全tie保留旧）；所有逐项tradeoff保留，胜出不等于过门槛。旧失败验证不进对照。

四条件新容量训练画像数值保留：balanced最大rho .84543超原.8；skewed .77638、low .72121、high .75752。新seed迁移已封存，但原target resolver无实测key仍UNKNOWN，不能宣布正式物理容量通过。CPU同域新模型plain/bounded各4 fresh forecasts、0剪枝、同一最优high，通过算术核验，不声称物理最优或提速。

独立前选择结果已落 `duration-v12-selection.json`：duration-WLS胜出，决策错数保持0，时间项失败12→8，归一误差均值.71589→.66667；仍保留training CV FAIL。授权的新完整duration模型重拟合全部训练，内部SHA2e4571d936cc9a621c74bd9354ec785ba18ebdbede0e5a953104e547f83fa21b；无验证源，full verifier与447个递归选择来源全核验。

新v11四候选各181请求的预测全部已知且封存；预测比较U依次balanced−.90604、skewed−3.14106、low−.93657、high−.83190。`flow-validation-v11/execution/suite.json` 独立执行已启动，不重试、不替换seed。新 `capacity-qualification-protocol.json` 提前声明活跃family/task均值转移误差≤20%、完整原生资源/时钟/质量/原稳定性及原rho门槛；这是统计画像规划准入，不要求物理capacity上界，旧target resolver无measured key仍独立标识。

训练开发分支仅基于v10/旧training证据准备OCR粗context与semantic更强收缩对照，候选及选择规则需先封存；该分支不读取v11 raw/log/report。不会把新独立验证结果用于同一目标的再拟合或门槛选择。


## v11 独立验证完成与 v13 训练内选择

v11 四候选 724/724 请求完成，真实质量和资源证据齐全；ΔU 符号 3/3、pair 排序 6/6 正确。但原 gate 为 FAIL（3 项失败、0 UNKNOWN）：skewed queue MAE=1.51485、skewed text P95 误差约23.1%、low image P95 误差约37.3%。不宣布最终收益。独立容量转移四条件均未满足预声明统计适用性；原 rho/stability 不改，旧失败记录保留。

提前冻结的训练内三候选比较已完成：baseline/OCR handler/phase ridge1 的决策错误均0，时间失败数8/7/9，归一误差均值 .666674/.618704/.616850。按原字典序选 OCR handler；7/56 时间检查仍失败。该选择不读取v11观测用于拟合，不等于CV PASS。后续先为所选方案生成完整training模型，并检查残余误差与输入条件容量；尚未开始干扰机制和最终四组确认。


输入条件容量的纯训练整seed检查已新增 `capacity-input-weighted-v1/report.json`：36 family/task 组中32已知、4 UNKNOWN，配对宏平均service误差从 pooled 24.26%降至按公开目标input权重计算的14.79%（27组改善、5组退步）。缺输入组不回退/不重归一，不声明capacity上界或新目标资格。原g(n)、rho、.8+1e-8不变。

独立训练准备度审阅建议先新增训练seed和相同轨迹重复，增加低率任务P95样本，而非反复用同两seed调参。所选OCR full fitter的合成源审计tests通过，已前瞻冻结full protocol并启动CPU full fit。H90 seed627502独立plan仅准备、未seal未GPU；拟另开匹配最终H600的training campaign，严格区分training/validation。


H600 campaign 短预检（30s、seed627600、low）61/61请求/质量完成、无采集问题、四进程正常BYE/exit0；不纳H600数值训练。主采集 `continuous-training-v14-h600/campaign.json` 已实际启动，627601/627602×4主流加2条同trace重复，共12000 offered requests，H600/drain300，失败停止不重试。重复仅用于run variation诊断，数值拟合保留16主流（H90v10八条+新H600八条）及v7/v8旧profile，H600整seed留出，重复同seed不能当独立replicate。

OCR full fit已完成，strict-variant-full-audit.json独立通过传递源/选择重算/实际OCR数值身份；此前通用full verifier缺显式OCR绑定，已用新增strict wrapper补齐且5tests通过，旧冻源保持。完整模型仍是训练CV FAIL的开发产物，不用于强行启动独立/最终收益。


首条H600（627602 skewed）1200/1200及质量逐请求重算审计通过、四类错误计数0，原稳定性recoverable_congested且恢复通过，drain98.840s。按公开到达三等分，text服务均值831.5/890.6/863.1ms，scheduled→worker等待均值5.638/26.920/59.711s。输入构成随时间段不同，不能由这些均值直接断言热漂移；它们确实显示完整到达下长期排队积累，不能用短窗口探针合格代替。相同trace重复将独立检查run variation。


训练回顾性的H90 frozen OCR full model→首H600 skewed对照已计算（不是前瞻独立test、没有拟合）：1200/1200覆盖、serviceMARE8.626%、text完整P95误差11.54%，但queue depth MAE5.584req>原1req，ASR P95误差28.40%。模型及数值来源先于新训练，CPU预测在采集后生成且仅公开trace进入predict；旧H90资源是未验证迁移假设，绝不用于资格/最终收益。这进一步显示平均service误差与长期queue gate之间的差别，后续H600整seedCV不改门槛。

第一对low同trace重复：两次1200/Q1200全clean、稳定性通过、drain2.213/2.201s。整体service均值1032.16/1024.03ms、E2E P95 4111.27/4016.04ms。每任务与每alias差异方向不统一，单对仅描述，不校准总体方差，不修改固定模型。


H600主采集已完成并逐流审计6/10（7200请求/质量）：第二seed balanced同样1200完整、错误计数0、原稳定性通过，drain0.586s。第7条主流已进入启动。服务/资源整seedCV等候器仍只等待整个campaign完成；新增独立容量CPU等候器只在8条primary全部完成后执行inventory/CV/逐数值重算，不启动GPU或seal，也不使用两个diagnostic repeats。

纯训练质量条件诊断`quality-input-conditioned-v1/results.md`及`audit.md`已完成，9 CPU tests通过。按输入合计错误量/参考单位的预测，在H90整seed留出8case中Q绝对误差均值0.00001195，旧固定代理0.00570138。原评分和逐task来源审计一致至1e-15；每seed四路由实际Q相同，因此只支持绝对校准改善，不支持新ΔQ/排序收益。当前冻定CV/v15验证及旧Q数值不替换；后续若集成须新版本和协议。


第7条H600（627601 skewed）1200/Q1200完整、独立四审计通过、所有采集/映射/质量错误为0，原稳定性recoverable_congested且恢复通过，drain96.630s。两skewed主流均呈持续积压，后者textE2E P95约94.651s、offered peak backlog129；未由此调参数。累计8400完整请求，第8 balanced primary运行。所有已完成流的实际等待/服务散点图保留在campaign/figures/下，仅训练观测图，不是预测误差或独立收益。

准入接口只读审阅见`docs/h600_admission_minimum_protocol_review_20261027.md`：真实sender-ticket FIFO/parent RESULT release与模拟alias排序/handlerEND release近似不同；现shared-only资源和容量不能自动授予admission资格。新增CPU `h600_search_check.py`已独立审阅与20项回归，严格full16/H600资源/v15公有plan审计，在候选内存copy覆盖资源值，两次fresh DFS保留UNKNOWN和全部ties；还未运行最终模型，原rho/物理最优不认证。


H600主采集正式10/10完成且CPU独立审计10/10通过（见completed-main.json、completed-independent-training-audits.json），12000/Q12000完整、8primary9600新训练请求+2repeat2400诊断。两组repeat六项U重算见campaign/repeat-utility-diagnostics-v1：balanced repeat-primary -0.01756387，low +0.02094185；Q/T/M差0、L/P/E波动。仅两pair描述，非总体方差/CI/噪声门槛，不进拟合。

H600容量training CV已完成并重算源/全部数值，capacity-h600-v1/training-cv-auto-v1/results.md：36/36输入条件组已知、0UNKNOWN；observed-denominator宏均误差weighted4.7783%、pooled5.1337%；独立原prospective predicted-denominator20%算术训练诊断36/36均满足。balanced两fold原rho不满足，其余三路由均满足；这是training适用性结果，不授予独立target资格、旧catalog resolver或物理bound。自动service/resources两CPU fit已启动，未重复；full16/seal/独立GPU仍待报告。

H600整seed原服务报告已完成并保留FAIL：7200/9600已知，627602 low/high在长时钟的浮点残差上耗尽事件预算。独立最小复现表明剩余时长小于clock半ULP，clock+duration==clock而剩余分数尚高于完成阈值。新增版本化numerical-v2只按已选完成事件结算，不增事件预算、不放宽误差门槛；47项相关CPU检查通过。新报告`background-statistical/run-aware-v14-H600-numerical-v2-report/report.json`覆盖9600/9600，原已知六条预测逐项完全相同，ΔU符号6/6、pair排序12/12正确。仍有原三项时间失败：627601 skewed queue MAE2.613861386、627601 low image P95 ARE.284213646与OCR P95 ARE.377306189；因此training CV依旧FAIL，不能称独立验证通过。

原报告的残差诊断`run-aware-v14-H600-residual-diagnostics-v1`保留全部9600请求分母和两个UNKNOWN队列。父接收尾部约1ms，无法解释数百ms/秒级残差；观察context的工作重构仅为诊断，不能用真实未来轨迹回填预测。H600资源整seed模型已完成；实际context能量回放7/8已知，其缺项来自一fold的G2 ASR+OCR仅.571ms暴露，缺域不回退；新前瞻CV未遇该缺项。

在原正速度范围、alpha=.1、OCR handler约束和整seed划分不变下，已预声明一次Gamma reciprocal-mean训练目标对照。v1入口在拟合前因误用中间direct_source字段失败，记录保留；v2直接绑定原生raw和严格mapper来源，针对性检查通过，协议SHA3221bc96c878e0ee41bb438b0a68c675a815efc0466f05ca578d5ef18622ad8a。CPU交叉验证已启动，仅按预声明字典序选方案；不读取独立观测、不扫描超参数、不改门槛。新数值引擎与所选估计器须由新验证入口封存，旧v15方案不能悄然替换。

固定Gamma v2训练CV现已完成：9600/9600覆盖、决策错误0、时间失败3、归一时间均误差.444739099193，按预声明字典序优于WLS的[0,3,.460677448919]。三失败仍是627601 skewed queue2.613861386、low image P95 ARE.245728305、low OCR P95 ARE.486423987；OCR比WLS退步，不能只报告平均改善。selection.json选Gamma但original_thresholds_passed=false。已启动严格full16 CPU拟合，尚未seal或启动独立GPU；新selected numerical-v2验证/CPU搜索入口的23项CPU检查通过，独立子对话继续审阅选择来源与完整域绑定。

Gamma full16已完成且从原生来源重新数值拟合完全相同（model SHA a322044f243fd7f308526f3dbbe73281c3881cc2935ac398ffa1e439935cad34）；真实selection经固定9600域/两报告原评分/paired实际观测审计通过。新`flow-validation-h600-selected-v16/protocol.json`绑定所选Gamma、numerical-v2与原reserved plan/门槛；独立GPU仍未启动，正在seal。

新CPU同域DFS已完成`cpu-search/h600-selected-v16/summary.json`：四候选fresh plain/bounded结果及全部ties一致，唯一模型最优shared_spill_high、U=-.9955898278262456，0剪枝。原rho资格四项UNKNOWN单列，不认证物理/原gate最优或提速，更不是实际分流收益。额外真实provider/长时钟/UNKNOWN回归与独立只读diff审阅完成，见`docs/h600_selected_search_v2_review_20261027.md`；通用core的MPS域外限制保留，正式本次固定域仅shared。

seed627701四路预测已封存4800/4800完整；预测U依次balanced -1.342199670641、skewed -21.745183081227、low -1.237313368360、high -.995589827826，与fresh DFS逐项完全相同。执行suite已完成prospective-input-audit PASS，四候选/逐请求alias-task-input/无重试/qualification chronology均通过。显式execute-and-assess入口已启动，首次运行前数值来源审计仍在执行；GPU启动以原生startup为准，不能把入口启动误报为GPU已经运行。

阶段三新`h600-admission-pilot-v1/plan.json`已CPU准备且29项相关测试通过，固定seed627801、三mode同路线/到达/输入、完整3600分母。独立审阅已补原native_stability、cap2饱和等待与来源闭包审计cache；不借shared E/M/startup/容量。新continuation只在独立完整assessment落盘后封预测并显式diagnostic执行，保留原FAIL和不可行mode的skip，不绕过阶段顺序。公平四组的有限域/强禁分流包络及资格缺口见`docs/h600_four_group_fair_domain_preparation_20261027.md`；最终四组仍未运行。

独立首条shared_spill_high已实际完成并由新CPU采集审计通过：1200/Q1200、原stable recoverable_congested且passed、drain1.088s。封存U=-.995589827826，实测U=-1.216263742579。只读逐case描述对照（不是完整四路gate）显示service MARE17.08%、E2E13.95%、queue MAE.53465，但text/image/ASR/OCR的P95 ARE分别22.21%/28.09%/55.61%/24.54%，均超原20%；因此新独立预测已经出现明确尾部失败，绝不能因平均误差合格声称PASS。剩余三路按冻结协议继续、不会用这条观测拟合或调参。完整Δsign/rank/qualification尚未评定。

第二路shared_skewed已实际进入测量。另在CPU仅以已冻结训练模型及未来public机制plan生成`h600-admission-pilot-v1/cpu-planning-precheck.json`，明确未seal、不读取独立观测、不授权GPU：shared/max1/max2均预测1200完整，最后结束601.283/845.864/609.429s，text P95约4.186/236.091/17.365s。max1虽在原300s drain内仍有巨大等待风险，max2也未显示无代价降低干扰；这些仅是待机制独立检验的shared服务迁移假设，不构成实际收益。正式mechanism seal仍等待完整独立assessment。


### selected-v16 第二条独立流与后续 CPU 实现

shared_skewed 已完成1200/Q1200、原生采集和质量完整，但 collection audit=FAIL_ORIGINAL_STABILITY：原规则判为 unstable，原因 sustained_positive_tail_growth_with_nontrivial_end_backlog；仍于原drain300预算内排空，实际drain107.753809876s。不能将完成排空写成稳定性通过。只读逐流预测描述：service MARE0.0788934387、E2E MARE0.1060010379，queue depth MAE9.217821782（原门槛1，失败）；四任务P95 ARE为text0.139634498/image0.021438428/asr0.046829786/ocr0.013019824。预测U−21.7451830812，实际原六项U−25.1530713239。完整四候选Δsign/rank/gate仍待主进程自动汇总；未拟合独立结果、未改阈值或重试。来源 flow-validation-h600-selected-v16/collection-audits/shared_skewed{,-prediction-descriptive}.json。

新增纯CPU h600_four_group_statistics_v1.py，7项测试通过：锁定策略差值、强禁分流有限域包络、事后allow包络严格区分；缺配置保留9600完整分母与UNKNOWN，六项缺失不重归一，原资格空域与UNKNOWN分开，不认证source/部署最优，不给单block虚构区间。尚未冻结最终seed/策略或启动最终GPU。两子对话继续新增prospective admission2训练mapper和模式专属资源/容量模块，现有冻结源未修改。


### 后续有限角色预留与新准入训练入口已实际准备

root已创建 admission-next-stage-roles-v1.json，固定新T=628101/628102×4 admission2路线（8runs/9600），V=628201×4（4runs/4800），F=628301×shared/admission2×4（8runs/9600，R1只描述，不作统计显著性声明）。这是角色预算预留，尚非模型/预测/最终route与order seal。服务law固定原Gamma full16，独立检验准入迁移，本轮不再service重拟合/验证后修正；新T只补各route/controller模式专属资源/容量/质量。原shared预测/稳定性失败保留，最终qualified正收益资格为false，可做如实负结果/完整八配置诊断，不能换seed/阈值追逐正结果。

profile子对话已真实CPU prepare admission-training-v1/campaign.json（SHA19a7b1e5b999f6ff96fa37e867e2485bd76f8d534a7e3c7e07d9d24bfc3a4504），root独立调用verify通过。新mapper源码SHA4be279e1a0491046a04af5b9dd9976432167f8abed1da0c250b694b4920d91e5，29项CPU兼容与反例测试通过，campaign目录cpu-test-report.json保留来源与命令。尚无新训练GPU/runs；执行仍须完整stage2 assessment与stage3 mechanism报告后，root source-bound训练review。原mapper明示quality未审查，profile子对话另补训练quality模块；mode资源/容量新模块14项CPU测试通过并补pre-training protocol时序，即将CPU冻结协议。当前selected-v16第三条balanced已进入测量，原254个冻结来源回执root再检查未变。


### selected-v16 第三条balanced已完成，模式专属三训练协议已准备

shared_balanced 1200/Q1200、collection audit PASS、原stability recoverable_congested/passed，drain1.692098454s。只读描述 service MARE0.0600062602、E2E MARE0.1119153488、queue MAE0.4752475248；P95 ARE text0.090663746/image0.001846870/asr0.078101782/ocr0.012969651，均过原门槛。预测U−1.3421996706，实际U−1.4638237024。此路线通过不覆盖其他路线失败，完整gate仍待第4条。来源collection-audits/shared_balanced{,-prediction-descriptive}.json。

模式专属资源/容量已CPU prepare并verify：admission-capacity-v1/protocol.json（SHA976a4da69cb801fad7a7e638a2883a5a0db88049aae7d5d010a6f479d8318b2f）、admission-resources-v1/protocol.json（SHA1aad2201127a2c66ac83b5c411e341b023822f7b609a297982cfe840ec918a9d）；14项新CPU测试通过，协议在新训练startup前。quality模块7项真实strict native fixture+原质量算术测试通过，admission-quality-v1/protocol.json已准备（SHA9b9288f4e873b189a4dfb21dbbde1aeb088e63eff8a45f904726e4c9851a6fb0）。未启动新T GPU、未缺数据拟合。root新最终diagnostic prepare/actual/report仍未冻结/真实prepare，独立审查已修精确归一化与catalog/profile来源、quality原生来源、审计中途raw变更TOCTOU，独立5tests已PASS；完整8config与失败保留报告测试在做最终核验。


### 四路独立采集完成与有限接续封存

seed627701四路现已全部1200/Q1200，合计4800/Q4800；独立采集审计三条PASS，skewed为FAIL_ORIGINAL_STABILITY。完整预测误差/决策gate/容量评估仍由原已运行execute-and-assess进程计算，不重复启动GPU或另拟合。

T8 continuation已由主任务独立源审阅和条件授权，PID85446，仅在原stage2与三模式stage3完成并重新审核后，运行预声明628101/628102×4条admission2训练，拟合模式资源/容量/质量；固定shared Gamma服务模型不重拟合。具体协议、review及等候状态保存在admission-training-continuation-v1。

V4运行入口已独立核对并完成5项CPU控制测试，公共plan级claim防止换seal/output重试。F8运行入口完成11项主/独立CPU测试：完整T/V来源闭包在各job前后重哈希；gate原状态不能被重hash伪改；预测UNKNOWN只在真实V4完整4800且实际六项可评分时允许纯诊断接续。最终固定balanced/high策略、两条禁分流路线的强基线、完整八配置/9600分母及R1无显著性声明保持不变。上述均是执行准备，不是新的GPU证据。


### selected-v16 完整独立验证结果

四路4800/Q4800完整，ΔU符号3/3、候选pair排序6/6正确。原预测gate仍FAIL，6项失败、0UNKNOWN：skewed queue MAE9.21782；low image P95 ARE23.045%；high四任务P95 ARE22.21%/28.09%/55.61%/24.54%。balanced全部时间检查通过。四路service MARE依次6.00%/7.89%/9.00%/17.08%，因此平均拟合通过不能覆盖全过程队列及尾部失败。skewed原稳定性另FAIL，其他三条PASS。

实际比较U balanced−1.463823702409、skewed−25.153071323875、low−1.388863643418、high−1.216263742579；相对balanced两分流ΔU +.074960058991/+.247559959829。单一独立运行描述且预测gate失败，不能将这些数直接称最终可部署收益或统计显著收益。原始报告、gate、capacity-report及completed均已落盘，不用验证观测再拟合。原stage3接续已进入CPU机制seal，仍等待完整seal后按固定三模式顺序采集。


主线程已独立阅读V/F条件接续源、复跑V25项CPU tests并审阅F12项tests与交叉复核，分别写source-bound root review。V原PID87518/start603003048、F原PID87937/start603025364已显式启动等候；V仅等原T8，F等原T8/V4，不启动/替换前序、不改固定服务模型。两个waiter不代表新GPU已开始；原stage3 CPU seal仍进行。stage2完整独立误差图PNG/PDF已从冻结报告生成在report/figures-v1。


三模式pilot已完成shared与admission1两组native采集（各1200），第三admission2仍GPU测量。两组原始描述：mean worker 1088.58→733.04ms，但text E2E P95 4.7998→269.4760s，drain1.5525→281.6931s；尚不替代最终三模式审计/报告。新admission CPU search经主线程源阅读和16项CPU回归通过，等待V4完整报告后执行；两个子对话服务错误由主线程接手，不重复已有GPU任务。


### 三模式机制诊断完成

pilot正式DIAGNOSTIC_COMPLETE，3600/Q3600、配对3600完整，所有原生audit通过，三者原稳定性均recoverable_congested/PASS。max1/max2实际许可峰值1/2，饱和期间等候1198/675条，激活已证实；不是只有配置字段。实测shared/max1/max2 mean worker 1088.58/733.04/936.53ms，text完整P95 4.7998/269.4760/32.9162s，drain1.5525/281.6931/30.3634s。该固定low路线整请求准入减少平均执行干扰，但显著增加端到端尾部，不能宣布净收益。

冻shared服务迁移在pilot三模式service MARE11.31%/4.27%/9.17%，但max1/max2 E2E MARE20.99%/40.38%，queue MAE16.376/11.752；max2 textP95误差47.24%。pilot未预声明独立accuracy gate，故这里只给数值，不事后改成正式gate；模式完整六项/ΔU仍UNKNOWN。原T8接续已开始重审stage2与stage3完整证据，然后才执行预声明训练。后续V4仍是另一个固定独立检验，不用pilot结果重拟合服务。

### 续跑控制恢复（2026-10-27 06:13 UTC）

V1/F1 在任何 V/F 封存和 GPU 尝试之前，因扫描无关进程 `/proc/<pid>/cwd` 权限失败而退出；旧日志与失败证据保留。独立审查及 33 项 CPU 测试通过后，使用先按确切 `-m` 模块筛选的版本化 V2/F2 控制器。原 T8 进程 85446 保持不变，V2 为 91553，F2 为 91952；均绑定原 PID/start ticks/argv/cwd 与源摘要。V2/F2 仅等待原上游完成；正式 GPU 全局 attempt 名称、种子、请求、执行器、六项评分和阈值均不变，没有重做 GPU 实验。证据见 `orchestration-process-scan-recovery-v1.json` 及版本化控制目录。

当前 T8 正重算原独立验证 gate 和三模式机制报告；该 CPU 审计静默期间，GPU 空闲属预期。完成后按已冻结的 T8 → V4 → F8 有限链运行。同域准入 CPU 搜索已独立复核，等待 V4 完整原生报告。

### 已有微画像的输入匹配减速描述（2026-10-27 06:28 UTC）

新增独立只读分析 `microprofile-slowdown-descriptive-v1/report.md`：只用 v7 的264实际training jobs /690请求，18个精确(alias,task,pool) solo基准键、每键两次；未执行的264 validation jobs全部排除。G2 text 的 SG/SSG/SSGO worker墙钟中位比值为2.028/2.691/2.996；SSGO对应prefill墙钟2.293、decode forward墙钟合计1.187。输入与实际重叠暴露有明显差异；OCR无语义prefill/decode保持UNKNOWN。比值为经验墙钟描述，不是GPU busy或物理互补归因，不用于拟合。脚本仅核对两份直接plan/raw SHA；未执行昂贵闭包或GPU。

新增最终只读汇总 `campaign_final_summary_v2.py`（5项CPU测试通过），分别投影T8 manifest/CV、V4预测配对与实际Q完成、F8八格原六项与强禁分流基线，所有缺失项PENDING。`prediction_report_figures_v2.py`保留UNKNOWN图格，`final_four_group_figures_v1.py`输出PNG/PDF；各完成已知/未知两种临时fixture检查，合成图仅用于程序检查。现有v1源码及快照保留。

### 原 T8 已进入 GPU 采集（2026-10-27 06:30 UTC）

原 PID85446 完成完整上游 gate/pilot 重算与来源闭包，产出 `admission-training-continuation-v1/upstream-audit.json`、`training-review.json`。阶段二 FAIL 和阶段三 DIAGNOSTIC_COMPLETE/OBSERVED_ACTIVATION_ONLY 保留。第一条固定任务为 `seed-628101__admission2_balanced__primary`，四驻留真实worker已启动，未重启训练控制或替换请求。V2 PID91553、F2 PID91952依赖等待；CPU DFS等待PID92597；描述报告后处理等待PID93354，须原F8完成包及CPU搜索完成后才生成新的最终汇总/PNG/PDF。该后处理不调用goal完成接口，最终仍须主对话检查实际完整性、原资格和负结果边界。

### 已知全过程到达下的实际队列误差（2026-10-27 06:39 UTC）

新增纯标准库只读 `pilot-queue-error-decomposition-v1/report.md`，直接封存预测及三模式pilot报告的3600请求按ID/task/pool/alias配对，恢复offer一致；两份直接JSON SHA不变，不读raw/不拟合/不运行昂贵验证。max2累计worker墙钟低估20.44s、1.82%，平均service差17.03ms却对应平均wait差6302.91ms，前/后半wait差2684.99/9920.82ms、最后完成差20.93s；max1分别为3.84%和16.69s平均等待差。Δe2e=Δservice+Δwait+Δ(e2e−service−wait)逐请求成立，剩余仅约毫秒量级。全程到达已知不能固定服务律/许可时序，当前单块描述不能因果分离各误差来源或代替V4。报告保留各alias抵消、MARE、101点队列及尾部边界。

原T85446首条训练日志已出现offered1001/1200；该字段只表示已投递，不表示已完成。V2/F2及CPU/后处理原等待进程仍存活，未重试GPU。报告后处理新5项None/UNKNOWN真实schema临时检查已由子对话完成，证据 `final-report-postprocessing-independent-review-v1.json`。

### T8 第一条完整训练产物（2026-10-27 06:44 UTC）

`seed-628101__admission2_balanced__primary` 已生成原生raw、scored-raw及training-complete，原控制器有job-01-complete收据并进入第二条 `seed-628102__admission2_balanced__primary`。主线程直接小检查确认1200唯一measurement请求全部completed、1200质量totals覆盖且denominator>0、quality/runtime/telemetry错误为空、四worker正常BYE/exit0。共35366板级采样，startup至measurement为82.684s，deadline至observation_end为56.530s，原owned sampled peak30824MiB。此为训练检查点，非验证/收益/物理峰界；尚未产生mode资源/容量/质量全量fit。直接文件收据与检查范围见 `admission-training-first-job-checkpoint-v1.json`，完整源闭包与原生审计仍由冻结T8 mapper逐条及fit入口负责。

### T8 两个 balanced 种子完成（2026-10-27 06:55 UTC）

原控制器完成 `seed-628101/628102__admission2_balanced__primary`，2/8条、2400原生请求及2400质量totals完整，质量/runtime/telemetry错误为空，四worker各正常BYE/exit0；job-01/job-02及精确campaign/native SHA均保留。两条owned sampled peak均30824MiB，deadline至observation_end分别56.530/44.148s；尚未执行CV、资源/容量/质量拟合或独立验证。主线程新直接检查点为 `admission-training-first-two-checkpoint-v1.json`；这不替代冻结mapper/fit的原生闭包验证。原PID85446已开始第三条 `seed-628101__admission2_spill_low__primary` 加载worker，V2/F2与CPU搜索/报告等待进程存活；不重复采集、不更换种子。

### T8 第三条 spill_low 完成（2026-10-27 07:07 UTC）

原控制器新增完成 `seed-628101__admission2_spill_low__primary`，合计3/8条、3600原生请求及3600质量totals。第三条1200唯一measurement请求全部completed，quality/runtime/telemetry错误为空，四worker正常BYE/exit0；33611板级采样、owned sampled peak31208MiB、deadline至observation_end16.908s。精确job/native/scored收据及主线程直接检查点见 `admission-training-third-job-checkpoint-v1.json`，不代替完整冻结mapper/fit审计。第四条固定 `seed-628101__admission2_skewed__primary` 已加载worker；模型/阈值、seed、执行器和原六项效用未改，尚未使用训练中间结果提前验证或宣称净收益。

### T8 第四条 skewed 完成（2026-10-27 07:22 UTC）

`seed-628101__admission2_skewed__primary` 原生/质量1200完整、错误为空、四worker正常BYE/exit0，训练合计4/8条及4800请求/Q。39178板级采样，owned sampled peak30824MiB；deadline至observation_end135.048s，保留测量后收尾不截断，不能把完整排空自动称稳定性或原rho通过。直接检查点 `admission-training-fourth-job-checkpoint-v1.json` 绑定原native/scored/completion字节；主检查点不代替冻结mapper/fit全闭包审计。原T85446开始第五条固定 `seed-628101__admission2_spill_high__primary`，其余V2/F2/CPU及后处理等待不变。

### T8 第五条 spill_high 完成（2026-10-27 07:34 UTC）

`seed-628101__admission2_spill_high__primary` 原生/质量1200完整、quality/runtime/telemetry错误为空、四worker正常退出。训练合计5/8条及6000请求/Q；32633板级采样、owned sampled peak31240MiB、deadline至observation_end0.358s。独立直接检查点 `admission-training-fifth-job-checkpoint-v1.json` 保留确切job/native/scored/completion字节和检查边界；完整画像及整seed CV仍须八条齐备后由冻结fit入口生成。原T85446开始第六条 `seed-628102__admission2_spill_low__primary`，训练/验证角色和固定后续链保持不变。

### T8 第六条 spill_low 完成（2026-10-27 07:46 UTC）

`seed-628102__admission2_spill_low__primary` 原生/质量1200完整、quality/runtime/telemetry错误为空、四worker正常退出。训练合计6/8条及7200请求/Q；33132板级采样、owned sampled peak31240MiB、deadline至observation_end7.310s。直接检查点 `admission-training-sixth-job-checkpoint-v1.json` 保留确切job/native/scored/completion字节及检查边界，不代替冻结mapper/fit全闭包。原T85446开始第七条 `seed-628102__admission2_skewed__primary`；原V2/F2及CPU搜索/后处理进程仍等待完整训练产物。

2026-10-27T08:01:04.102589+00:00 — T8第7组 seed628102 skewed 原生1200/Q1200完成，执行/质量/遥测无错误、4进程正常退出；直接checkpoint不代替冻结完整审计。第8组spill_high原任务加载，V/F和CPU报告等待链保留。

2026-10-27T08:11:49.154310+00:00 — T8全部8组原生采集9600/Q9600完成；第8组spill_high无执行/质量/遥测错误，4进程正常退出。直接checkpoint不代替冻结完整审计。原始T85446继续CPU三画像拟合和审计，V91553/F91952及搜索/报告等待链保留；未重跑拟合或改冻结源。

2026-10-27T08:22:10.299747+00:00 — 原T8最终completed.json生成，8job顺序和3manifest直接SHA一致，9600/Q9600；服务未重拟合，未使用V/F，保留上游shared FAIL。T进程已正常退出；原V91553进入CPU完整认证。主线程未重复closure/fit。

2026-10-27T08:36:25.266611+00:00 — 原V2 training-audit.json已生成，并封存4路线4800预测/容量；直接收据一致、unknown_reason均None、未用未来真实trace。注意V2训练认证产物正式名为training-audit.json（不是upstream-audit.json）；前面状态观察已改正。仍CPU执行前认证，未创建原生V4attempt。

2026-10-27T08:40:18.904179+00:00 — 子对话新增v7同域功率/显存描述：264训练job/690严格配对请求、排除18priming和264未执行验证job；job区间14579样本，postpriming14610样本，268.79W、owned sampledpeak31634MiB。BoardMonitor在allREADY后启动，原微画像startup峰UNKNOWN，不伪造。root已读报告/目视PNG。V2已封存4800预测，正在执行前CPU核验，原链均存活，无重试/改源/加GPU。

2026-10-27T08:53:43.105712+00:00 — 原V2完成CPU训练/封存认证，独占原生V4 claim正式创建，4固定路线共4800请求；第一条balanced原生采集中，不重试/不改门槛，预测准确度待全部测量判定。新增中文证据索引docs/predictor_validation_evidence_index_20261027.md，链接检查通过。

2026-10-27T09:06:25.611816+00:00 — 原V4第1条balanced原生1200/Q1200完成，执行/质量/遥测无错误、4worker正常退出；排空66.924s，owned sampled peak30824MiB。直接收据与原runstate一致，完整gate仍待4条。第2条skewed原流程加载，未重试/改源/调阈值。

2026-10-27T09:19:54.280514+00:00 — 原V4第2条skewed原生1200/Q1200完成，累计2400/Q2400；执行/质量/遥测无错误、4worker正常退出；排空158.059s。原流程第3条spill_low加载，完整gate未判定。新整seed训练CV描述8cell、rho5/8通过、资源1未知context保持UNKNOWN；仅小报告直接SHA，不新增阈值/fit/GPU。

2026-10-27T09:32:26.154538+00:00 — 原V4第3条spill_low原生1200/Q1200完成，累计3600/Q3600；执行/质量/遥测无错误、4worker正常退出；排空26.625s。原流程第4条spill_high加载，完整gate未判定；不重试/改源/调阈值。

2026-10-27T09:45:48.874558+00:00 — 原V4第4条spill_high原生1200/Q1200完成，累计4800/Q4800、runstate COLLECTED_DIAGNOSTIC；执行/质量/遥测无错误、4worker正常退出；排空0.347s。原V2转CPU报告/独立重算审计，完整gate待产出。F/search/report原等待链保留，未重试或改源。

2026-10-27T09:54:23.370642+00:00 — 原V4 report/gate/capacity已生成，gate FAIL10/UNKNOWN0；4800配对，ΔU符号3/3、排序6/6正确。四路线queueMAE12.2178/2.2871/2.7327/2.0000均>1；其余失败为balanced/high E2E及low image/high text,image,asr P95。服务MARE6.25/6.17/9.44/11.07%均过20%。原V2仍CPU完整重算，finalcompleted未出，不能提前称审计完成。六项U直接算术一致、容量原resolver/global permit仍UNKNOWN；不训练回流/改门槛。

2026-10-27T10:17:47.496002+00:00 — 原V4最终completed.json生成，DIAGNOSTIC_COLLECTED/collection_error=None，4次4800请求/Q4800，原report-verification verified=True且数值/原生重算通过；原V2正常退出。gate仍FAIL10/unknown0，不训练回流/改源。主线程直接关键收据SHA一致，不重复closure。原CPU搜索子进程104542启动（CUDA空、三线程变量1），原F91952认证T/V后接F8，尚无F8原生attempt。

2026-10-27T10:28:08.114046+00:00 — 原准入同域CPU搜索正常完成，plain/bounded各4次fresh forecast，4候选逐项六项U及最优/全部并列一致，PASS；模型内high唯一最优U=-2.7305572882982583。原资格域UNKNOWN（low/high资格未证，balanced/skewed统计rho排除），无原门槛受限或物理最优证明；0剪枝/0节省调用，无搜索加速声明。原F91952继续T/V认证，F8原生attempt未出；不重跑搜索/实验。直接核对见admission-search-final-direct-checkpoint-v1.json。

2026-10-27T10:49:55.229789+00:00 — 原F91952产出upstream-audit.json，完成确切T8/V4原生/数值认证；原独立gate FAIL与verified凭证一致。主线程仅直接核对小完成/报告/gate收据，无重复closure。原流程继续seal，F8原生attempt仍未生成。见four-group-upstream-direct-checkpoint-v1.json。

2026-10-27T11:24:13.830812+00:00 — 原F91952生成最终sealed/seal.json，8条服务预检各known1200/PREDICTED_COMPLETE，9600请求、既定顺序及balanced/high策略封存；independent gate保持FAIL。此封存仅服务可完成性，full_six_term_prediction_known=False，不授准确度或容量。直接收据核对见four-group-seal-direct-checkpoint-v1.json；原流程run入口仍认证，原生F8attempt未生成。

2026-10-27T11:46:11.059775+00:00 — 原F91952完成运行入口认证并创建正式F8-attempt.json及execution/started.json；8格9600固定请求，第一格shared_spill_high原生启动，禁止重试/换seed。直接收据核对见four-group-native-start-direct-checkpoint-v1.json。先前CPU continuation claim不当作原生采集；原gate FAIL及资格UNKNOWN保留。

2026-10-27T11:57:44.735684+00:00 — F8第1格shared_spill_high原生1200/Q1200完成，原actual-audit已生成；独立标准库核对六项/各task P95/启动及测量能量/显存峰与原score一致，U=-1.3960355955926342。测量期完成1197，T=0.0025保留；排空0.8576s，owned sampledpeak31640MiB，执行/质量/遥测无错误、4worker正常退出，原稳定性True。核对使用原annotation，不独立重评质量/不替代闭包/不授资格，见four-group-first-cell-direct-checkpoint-v1.json。原F进入第2格admission2_spill_high；尚不能比较完整分流净收益。

2026-10-27T12:09:38.012779+00:00 — F8第2格admission2_spill_high原生1200/Q1200完成，累计2400/Q2400；标准库直接六项/P95/能量/显存核对通过，U=-2.3962048055881957。测量期1197，T0.0025保留；排空0.8276s、owned sampledpeak31240MiB，错误为空、4worker正常退出、原稳定性True。host准入observed permit peak2/上限2，记录区间非真实全acquire/release，不授容量证明。同high执行方式差admission2−shared约-1.00017，仅该输入块描述，非分流相对禁分流收益。原任务第3格admission2_balanced继续。见four-group-second-cell-direct-checkpoint-v1.json。

2026-10-27T12:21:48.999378+00:00 — F8第3格admission2_balanced原生1200/Q1200完成，累计3600/Q3600；独立标准库六项/P95/功率积分/显存核对通过，U=-11.718832505613356。测量期完成1114，T=0.0716667保留；排空48.5815s，owned sampledpeak30824MiB，textP95=50.0736s；无执行/质量/遥测错误，4worker正常退出，原稳定性True。准入observed peak2。锁定high减balanced约+9.322628，只是预锁定基线差，最强禁分流主比较仍待skewed，原资格UNKNOWN与预测FAIL不变。见four-group-third-cell-direct-checkpoint-v1.json。原流程第4格shared_balanced已启动。

2026-10-27T12:33:31.815527+00:00 — F8第4格shared_balanced原生1200/Q1200完成，累计4800/Q4800；标准库六项/P95/能量/显存交叉核对通过，U=-1.5096334388021828。测量期完成1197/T0.0025，排空0.8370s，owned sampledpeak30824MiB；错误为空、4worker正常退出、原稳定性True。预锁定balanced/high的2×2四格已齐，但两模式最强禁分流须待skewed，不能宣布完整F8或净优化收益。见four-group-fourth-cell-direct-checkpoint-v1.json；原流程第5格shared_skewed继续。

2026-10-27T12:48:18.068080+00:00 — F8第5格shared_skewed原生1200/Q1200完成，累计6000/Q6000；独立标准库六项/P95/能量/显存核对通过，U=-21.810341314075856。测量期完成1056，T0.12保留；排空97.2146s、owned sampledpeak30824MiB、textP95=98.0066s；错误为空、4worker正常退出、原稳定性True。共享模式声明两禁分流路线中balanced较优，锁定high减该强N约+0.113598，单块描述、非资格/重复性证明。原流程第6格admission2_skewed继续，核心准入strongN主比较仍待它，原gate FAIL/容量UNKNOWN不变。见four-group-fifth-cell-direct-checkpoint-v1.json。

2026-10-27T13:03:00.667883+00:00 — F8第6格admission2_skewed原生1200/Q1200完成，累计7200/Q7200；独立标准库六项/P95/能量/显存核对通过，U=-28.563243347970566。测量期完成1023/T0.1475保留；排空127.3326s，owned sampledpeak30824MiB、textP95=127.8577s；无执行/质量/遥测错误、4worker正常退出、原稳定性True。准入模式声明两禁分流路线中balanced较优，锁定high减strongN约+9.3226277000；单块描述，不授容量/原预测准确度/全域最优/可重复收益。两low格及最终原报告/认证仍待完成。见four-group-sixth-cell-direct-checkpoint-v1.json。原流程第7格admission2_spill_low继续。

2026-10-27T13:12:06.595584+00:00 — F8第7格admission2_spill_low原生1200/Q1200完成，累计8400/Q8400；独立标准库六项/P95/能量/显存核对通过，U=-6.498967041629402。测量期1172，T0.0233333保留；排空15.2553s，owned sampledpeak31240MiB、textP95=28.1445s；错误为空、4worker正常退出、原稳定性True。准入四路线测量齐备，预锁定high主策略保持，不因low实测事后替换。见four-group-seventh-cell-direct-checkpoint-v1.json；原任务第8格shared_spill_low已启动，完整报告/最终认证待结束。


### 2026-10-27T13:24:50.596352+00:00 — F8 八格采集完成

最后 shared_spill_low 1,200/Q1,200；主线程标准库六项独立核对 PASS，U=-1.3311813637525391。累计八格 9,600/Q9,600，原 report DESCRIPTIVE_COMPLETE；executor/wrapper 最终认证与后处理尚待，不提前宣布完整验收。原预测门槛 FAIL、容量 UNKNOWN、R=1 保留。


### 2026-10-27T13:45:56.187948+00:00 — 冻结有限方案原认证与最终报告完成

F8 executor 与 wrapper 已完成 DESCRIPTIVE_COMPLETE、9,600/9,600、execution_error=null、final_seal_authenticated=true；原PID91952及后处理PID93354均正常结束并留下完成凭证。自动汇总及两类PNG/PDF齐全、主线程目视完成，最终直接收据核对PASS。准确度FAIL、容量UNKNOWN、R1及条件后端缺口保持原结论。最终结论见 predictor_validation_final_results_20261027.md。附加核对脚本v1误将包含实际时间的semantic摘要当输入摘要，断言失败且未写PASS；v2按原audit与八个共同计划独立核对记录绑定修正，已通过，未修改原实验/封存/协议。
