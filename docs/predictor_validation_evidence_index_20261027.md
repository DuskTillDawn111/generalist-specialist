# 共驻画像与预测验证：证据索引

当前状态以 `artifacts/predictor-gpu-20261027/progress.json` 和原任务的完成凭证为准。本索引记录已存在的报告；实验采集完成、预测准确度通过、容量合格、分流获益分别判断，互不替代。

2026-10-27 13:45 UTC 当前进度：本轮冻结有限方案已完成。T8 8/8（9,600 请求）、V4 4/4（4,800 请求）、F8 8/8（9,600 请求）和两模式同域 CPU DFS 均有原完成凭证。F8 原 report/executor/wrapper 为 DESCRIPTIVE_COMPLETE，seal 认证 true、无执行错误；八格独立六项算术及主线程最终收据核对 PASS，最终中文汇总和 PNG/PDF 已生成并目视检查。V4 FAIL10、shared FAIL6、容量 UNKNOWN 与未实现机制保持原结论；以下早期快照保留历史状态。

[训练及 V4 报告已产出阶段汇总](../artifacts/predictor-gpu-20261027/campaign-summary-v2-V4-report-present/results.zh.md)：2026-10-27 10:07 UTC 的只读快照，整体仍为 PENDING_CAMPAIGN。原 V4 外层严格复核是否完成，以原 completed.json 为准；该汇总不重做闭包验证。

## 1. 四模型驻留与真实干扰画像

- [已有请求与画像范围盘点](request_evidence_inventory_20261027.md)：固定 text×2＋G2＋OCR 驻留及实际采集范围。
- [微画像减速与阶段描述](../artifacts/predictor-gpu-20261027/microprofile-slowdown-descriptive-v1/report.md)：264 个实际训练 job、690 个测量请求，按输入、组合和启动错位配对 solo 基线；未执行验证 job 不计入结果。
- [微画像板级功率与拥有进程显存](../artifacts/predictor-gpu-20261027/microprofile-resource-descriptive-v1/report.md)：同一批 job，区分名义组合与真实活动重叠；含 PNG/PDF。该早期监控在 all READY 后启动，因此启动峰值未知，不能与后续 startup-through-drain 峰值混用。

微画像的墙钟阶段并非 GPU busy 时间。SM/DRAM/L2 计数器权限受限，计算型/显存型互补的物理归因仍未知；板级功率与全部拥有进程显存不归因到单个任务。

## 2. 训练画像与独立预测

- [shared 独立验证报告](../artifacts/predictor-gpu-20261027/flow-validation-h600-selected-v16/report/results.zh.md)：4 条路线、4,800 个测量请求；原 gate 为 FAIL，失败 6 项；ΔU 符号 3/3、排序 6/6 正确，不能覆盖队列/P95 失败。
- [准入训练完成阶段汇总](../artifacts/predictor-gpu-20261027/campaign-summary-v2-T8-complete/results.zh.md)：T8 两个训练 seed、8 个 job、9,600 个请求及容量/资源/质量三份 manifest。每项各 8 条整 seed 留出记录；资源留出是真实上下文回放，不是前瞻队列验证。
- [三类训练画像的整 seed 留出误差](../artifacts/predictor-gpu-20261027/admission-training-cv-descriptive-v1/report.zh.md)：逐单元列出原服务适用性、rho、启动/回放能量、显存峰和 Q 误差；未知资源上下文保留 UNKNOWN，不新增接受门槛。
- [T8 完成的直接核对记录](../artifacts/predictor-gpu-20261027/admission-training-final-direct-checkpoint-v1.json)：任务顺序、完成收据、三份 manifest SHA；完整闭包认证由原流程执行，本记录不替代它。
- [V4 预测封存的直接核对记录](../artifacts/predictor-gpu-20261027/admission-validation-seal-direct-checkpoint-v1.json)：seed 628201 的 4,800 条事前预测，未读取未来真实结果、不用于拟合；封存不等于准确度通过。原进程随后已进入 GPU 测量，具体进度见 progress.json。
- [V4 首条 balanced 完成核对](../artifacts/predictor-gpu-20261027/admission-validation-first-job-checkpoint-v1.json)：1,200 个请求和质量计数一致、无执行/质量/遥测错误、四个 worker 正常退出；不代替四条路线的完整准确度门槛。
- [V4 第四条及采集完成核对](../artifacts/predictor-gpu-20261027/admission-validation-fourth-job-checkpoint-v1.json)：四条路线原生与质量共 4,800 个请求，原流程状态为 COLLECTED_DIAGNOSTIC；完整误差报告和独立重算审计仍由原进程生成，采集完成不代表预测通过。
- [V4 原门槛报告的直接核对](../artifacts/predictor-gpu-20261027/admission-validation-gate-direct-checkpoint-v1.json)：原 report/gate 已写出，FAIL 10 项、未知 0 项；ΔU 符号 3/3、排序 6/6 正确，但四条路线队列误差及部分 E2E/P95 不通过。原进程仍在完整重算，最终审计完成凭证以 progress.json 和原 completed.json 为准。
- [V4 中文描述与独立小报告复核](../artifacts/predictor-gpu-20261027/admission-validation-v4-descriptive-review-v1.md)：逐路线误差、P95、六项实际效用及原容量标签，37 项门槛和效用算术自洽；保持 FAIL，不替代原生记录重算。
- [V4 最终完成核对](../artifacts/predictor-gpu-20261027/admission-validation-final-direct-checkpoint-v1.json)：原 completed.json 已生成，4 次测量、4,800 原生/Q 请求完整；原验证器报告 verified=true、numerical_and_native_recomputed=true。原进程正常退出，门槛仍为 FAIL 10 项、未知 0 项。以上较早快照中的“复核待完成”不再代表当前进度。

服务 Gamma 模型保持冻结，T8 仅拟合模式/路线对应的容量、资源和质量画像。V/F 数据不回流训练，不事后改 seed、门槛或效用。

## 3. 共享与整请求准入机制

- [准入机制就绪条件](h600_execution_mechanism_readiness_20261027.md)：硬件资源份额与执行后端能力边界。
- [pilot 完整机器报告](../artifacts/predictor-gpu-20261027/h600-admission-pilot-v1/report/report.json)：shared/max1/max2 各 1,200 请求，真实准入激活已观测，状态为 DIAGNOSTIC_COMPLETE；执行时间减少与等待、尾延迟恶化同时出现。
- [pilot 排队误差分解](../artifacts/predictor-gpu-20261027/pilot-queue-error-decomposition-v1/report.md)：严格配对 3,600 请求。max2 累计服务时长有符号低估约 1.82%，平均等待低估约 6.30 秒；1.82% 不能当逐请求拟合误差率。
- [剩余预测瓶颈的只读解释](../artifacts/predictor-gpu-20261027/predictor_remaining_bottlenecks_readonly_v1.md)：区分已观测误差、真实 release/ticket/handoff 与模拟的代码差异、待验证服务假设。尚未识别各因素的因果贡献，不在本轮改模型。

当前执行机制为 host 整请求准入。MPS/资源份额未验证生效；共同 batch、公共权重子矩阵、跨请求 cache 复用后端未实现，预测器不添加虚构收益。这些机制须待相应后端完成后独立核验等待、额外操作、显存和质量开销。

## 4. 四组对照、搜索与最终验收

- [逐项验收清单](../artifacts/predictor-gpu-20261027/completion-audit-checklist-v1.md)：覆盖 T8、V4、CPU DFS、F8 的实际字段及原门槛，快照中的待完成项不随未来产物自动更新。
- [最终就绪独立审阅快照](../artifacts/predictor-gpu-20261027/final-readiness-review-v1.md)：10:30 UTC，T8/V4 完成及 CPU 搜索一致性 PASS；F8 未出，完整预测门槛仍 FAIL，原容量与条件后端尚未证实。该快照不代替后续最终验收。
- [F8 上游认证直接核对](../artifacts/predictor-gpu-20261027/four-group-upstream-direct-checkpoint-v1.json)：原任务完成 T8/V4 原生及数值认证，保持独立 gate FAIL；尚不代表最终封存或 GPU 采集开始。
- [F8 最终封存直接核对](../artifacts/predictor-gpu-20261027/four-group-seal-direct-checkpoint-v1.json)：八配置共 9,600 请求、服务可完成性预检已封存，原 gate 保持 FAIL；full_six_term_prediction_known=false，不授预测或容量资格。
- [F8 原生启动直接核对](../artifacts/predictor-gpu-20261027/four-group-native-start-direct-checkpoint-v1.json)：正式 F8-attempt 及 execution/started 已生成，封存和计划直接摘要一致；第一格 shared_spill_high 原生启动，尚不代表任何测量格完成。
- [F8 第一格原生与六项直接核对](../artifacts/predictor-gpu-20261027/four-group-first-cell-direct-checkpoint-v1.json)：shared_spill_high 1,200/Q1,200；原始到达、完成、P95、质量计数、启动/测量功率积分及 owned 显存峰算术与原审计一致。1197 在截止内完成，T=0.0025 保留；单格不形成分流收益结论。
- [F8 第二格原生与六项直接核对](../artifacts/predictor-gpu-20261027/four-group-second-cell-direct-checkpoint-v1.json)：admission2_spill_high 1,200/Q1,200，六项核对一致、真实准入观测峰 2；仍待本模式两条禁分流基线，不能由两格同高分流执行方式对比宣布分流收益。
- [F8 首对执行方式差异独立复核](../artifacts/predictor-gpu-20261027/four-group-first-pair-descriptive-review-v1.md)：相同 high 路线下 admission2−shared 的 U 为 −1.000169，主要算术贡献来自 L/P 增大；该对比不代表分流相对禁分流收益，不作因果或重复性推断。
- [F8 第三格禁分流基线直接核对](../artifacts/predictor-gpu-20261027/four-group-third-cell-direct-checkpoint-v1.json)：admission2_balanced 1,200/Q1,200，六项核对通过；测量期完成1114、排空48.581秒、U≈−11.718833。high 相对该预锁定基线差约+9.322628；最强禁分流主比较仍待另一条 skewed。
- [F8 第四格共享禁分流直接核对](../artifacts/predictor-gpu-20261027/four-group-fourth-cell-direct-checkpoint-v1.json)：shared_balanced 1,200/Q1,200、六项核对通过，U≈−1.509633。预锁定 balanced/high 的 2×2 已齐；两条 skewed 与两条 low 仍待固定流程完成。
- [F8 预锁定四格独立描述复核](../artifacts/predictor-gpu-20261027/four-group-locked-four-cells-descriptive-review-v1.md)：shared high−balanced≈+0.113598、admission2≈+9.322628；较大的准入内部差值伴随更差的禁分流基线，不代表准入整体净优化成功。最强禁分流仍待 skewed，完整八格仍未齐。
- [F8 第五格共享 skewed 直接核对](../artifacts/predictor-gpu-20261027/four-group-fifth-cell-direct-checkpoint-v1.json)：1,200/Q1,200，六项核对通过，U≈−21.810341。共享模式声明禁分流域中 balanced 较优，锁定 high 相对强基线 ΔU≈+0.113598；核心准入模式比较仍待下一格。
- [F8 第六格准入 skewed 直接核对](../artifacts/predictor-gpu-20261027/four-group-sixth-cell-direct-checkpoint-v1.json)：1,200/Q1,200，六项核对通过，U≈−28.563243。准入模式声明禁分流域中 balanced 较优，预锁定 high 相对该强基线 ΔU≈+9.322628；两 low 格及最终完整报告认证仍待，单块描述不授资格或重复性。
- [F8 前六格主比较独立描述复核](../artifacts/predictor-gpu-20261027/four-group-primary-six-cells-descriptive-review-v1.md)：两模式的声明强禁分流均为 balanced，admission2 主差≈+9.322628、shared≈+0.113598。仅有限路线单块描述，原完整八格报告仍待认证；预测 FAIL、资格 UNKNOWN 保留。
- [F8 第七格准入 low 直接核对](../artifacts/predictor-gpu-20261027/four-group-seventh-cell-direct-checkpoint-v1.json)：1,200/Q1,200，六项核对通过，U≈−6.498967，测量期完成1172；准入四路线已齐，主策略仍为事前锁定 high。最后一格及原完整报告尚待完成。
- [F8 第八格共享 low 直接核对](../artifacts/predictor-gpu-20261027/four-group-eighth-cell-direct-checkpoint-v1.json)：1,200/Q1,200，六项核对通过，U≈−1.331181；八格采集与直接算术均齐，完整原认证仍待。共享 low 的事后较好结果不替换锁定 high 主比较。
- [F8 完整八格独立描述复核](../artifacts/predictor-gpu-20261027/four-group-final-eight-cells-descriptive-review-v1.md)：八格9,600/Q9,600、六项与P95、主比较和锁定策略一致。该快照的原最终认证仍待，不代替executor/wrapper完成。
- [冻结任务范围验收意见](../artifacts/predictor-gpu-20261027/final-scope-acceptance-review-v2.md)：区分验证程序完成与预测门槛通过，逐项列出条件后端和容量证据缺口。
- [准入同域 CPU 搜索说明](admission_same_domain_search_check_v1_20261027.md)：原任务已完成，fresh plain/bounded 各 4 次预测，有限四路线结果一致，PASS；0 剪枝、0 节省调用。
- [CPU 搜索独立描述复核](../artifacts/predictor-gpu-20261027/admission-search-descriptive-review-v1.md)：逐候选六项效用及两种搜索结果一致，模型内 high 唯一最优 U=-2.7305572882982583；原资格域 UNKNOWN，不认证真实部署最优或加速。
- [CPU 搜索主线程直接核对](../artifacts/predictor-gpu-20261027/admission-search-final-direct-checkpoint-v1.json)：保存四份小结果的直接 SHA，未重做预测或闭包。

最终对照由原 F8 任务接续已完成的 V4，并已全部完成：同一到达与输入，shared/admission2 × balanced/skewed/spill_low/spill_high，共 8 格。主比较为 admission2 预锁定 spill_high 减本块 balanced/skewed 中实测较优的禁分流基线；强基线仍只覆盖这两条声明路线。单独立块只报告描述差值，不声称显著性、置信区间、部署全域最优或可重复正收益。

沿用 `U=-(Q+T+L+P+M+E)/6`、原 g(n)、rho 和稳定性规则。原最终流程与既定等待任务均已完成。本轮有限验证交付完整，准确度 FAIL、资格 UNKNOWN 和条件机制缺口保持独立；不将其改写为算法全部科学目标达成。


## 5. 本轮最终交付

- [最终结论与后续修改优先级](predictor_validation_final_results_20261027.md)：优先阅读；解释验证失败、分流正差及准入绝对表现的区别。
- [最终完整中文汇总](../artifacts/predictor-gpu-20261027/campaign-final-summary-v2-final/results.zh.md)及[机器结果](../artifacts/predictor-gpu-20261027/campaign-final-summary-v2-final/summary.json)：保留原门槛与完整八格六项。
- [最终主线程直接收据核对](../artifacts/predictor-gpu-20261027/campaign-final-root-direct-checkpoint-v1.json)：原 executor/wrapper 完成、八格既有独立验算及共同计划绑定；不另行重验全来源闭包。
- [独立误差图 PNG](../artifacts/predictor-gpu-20261027/admission-validation-v1/report/figures-v2/prediction-errors.png) / [PDF](../artifacts/predictor-gpu-20261027/admission-validation-v1/report/figures-v2/prediction-errors.pdf)。
- [最终对照图 PNG](../artifacts/predictor-gpu-20261027/campaign-final-summary-v2-final/figures-v1/four-group-comparison.png) / [PDF](../artifacts/predictor-gpu-20261027/campaign-final-summary-v2-final/figures-v1/four-group-comparison.pdf)。

- [最终子对话验收 v3](../artifacts/predictor-gpu-20261027/final-scope-acceptance-review-v3.md)：完整原凭证、机器汇总、八格算术、后处理及主线程收据一致；有限交付可验收，原科学缺口保留。
# 2026-10-28 只读occupancy证据更正

旧tiny probe并非缺所有UUID：`TARGET_INFO_CUDA_DEVICE.uuid` 为NULL，但
`TARGET_INFO_GPU.uuid` 与只读设备查询完整唯一匹配。早期无法绑定结论已由
`artifacts/predictor-occupancy-declaration-old-probe-correction-v2-20261028/audit.json`
更正，旧产物保持不变。新arch-aware resolver依据官方header分支与成功launch
严格等价证明，对原prepared01 smoke的全部55,891个launch给出静态上界；47个
CPU敏感性检查见
`artifacts/predictor-occupancy-declaration-audit-v2-20261028-final/audit.json`。
真实函数maxThreads/maxDynamic和请求carveout仍未测得，smoke不进入fit，当前
实际baseline训练行仍为0；不据此宣称achieved occupancy或吞吐验证。详见
`docs/occupancy_declaration_resolution_v2_20261028.md`。

