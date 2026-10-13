# Module 11A — Multi-Hotspot / Second-Generalist Structural Audit

CPU-only 审计已完成并停止；新 GPU runs = 0。Module 9、模型池、quality/capability、s0、memory、empirical g(x)、worker/runtime 未修改。

**正式结论：证据不足以认证 2G 必要区域，也不足以宣布当前模型池没有 2G 区域。** gate 记录为 `UNDETERMINED_REPLICA_EVIDENCE_REQUIRED`，而不是强行落入 SUPPORTED / 1G_ONLY / NOT_SUPPORTED。B1 有认证的 S→1G 扩展；B6 有条件性的 2G window，但可容纳未测 Specialist replicas。

## 范围与容量定义

Eligible tasks：ASR、Image、LID、OCR、Text；VQA 无 admitted Specialist，不纳入热点组合。机械枚举 10 pairs + 10 triples × 6 budgets = 120 contexts；3 envelopes/context = 360。background 为 frozen mix_03/I1。方向直接复用逐任务 operational headroom，没有结果驱动搜索。

所有 LP/MILP capacity witness 均满足 admitted path、routing conservation、safe memory 和 D≤0.8g。N_G 计 replica 数，Qwen×2 也是 2G；alpha_S 仅禁止热点→G，并未禁止 background 使用 G，也未人为限制 Specialist replicas。

缺少 g(x)、M(x) 的 count 未外推。证据支持的 space 为 6912 个 count vectors；该 space 的 optimum 是诊断值，不能将其 evidence bound 当作物理上限。missing-count memory screen 使用已测 prefix + 原单 replica residency，并明确标为 screen，未扩展 calibration table。

## 用户问题逐项回答

1. **哪些预算允许 2G？** B4/B5/B6 有已测 count 的 2G residency，且有 rho≤0.8 background witness。最低 admitted 全任务 coverage 的已测 2G safe memory 为 33678.870215 MiB（Phi+Qwen+PaddleOCR；这只是 coverage，不是该最小状态的 operational witness）。B1/B2 在 frozen residency screen 下放不下 2G。B3 的 Phi×2 screen 可放下（最低 30084.146816 MiB），但 Phi×2 的 memory/scaling 没有测量，必须保持 unresolved，不能把它判为物理不可能。

| Budget | Safe budget MiB | 2G classification | Replica certificate |
|---|---:|---|---|
| B1 | 17007.174055 | TWO_G_STRUCTURALLY_IMPOSSIBLE | PASS |
| B2 | 27980.189425 | TWO_G_STRUCTURALLY_IMPOSSIBLE | EVIDENCE REQUIRED |
| B3 | 32019.254406 | TWO_G_MEMORY_SCREEN_POSSIBLE_REPLICA_EVIDENCE_MISSING | EVIDENCE REQUIRED |
| B4 | 34567.553380 | TWO_G_EMPIRICAL_RESIDENCY_FEASIBLE | EVIDENCE REQUIRED |
| B5 | 43736.541926 | TWO_G_EMPIRICAL_RESIDENCY_FEASIBLE | EVIDENCE REQUIRED |
| B6 | 49140.000000 | TWO_G_EMPIRICAL_RESIDENCY_FEASIBLE | EVIDENCE REQUIRED |

2. **最大 W_1G：0.085012612662**，B1 ASR+LID（另有同值 triples）。alpha_S=0.474340146733，alpha_1G=0.559352759395。该 B1 window 通过 replica residency screen；共 4 个 B1 contexts 支持 S→1G operational planning 扩展。没有新 GPU stability 验证。

3. **最大 W_2G：0.042334462039**，B6 OCR+Text，alpha_1G=1.000000000000，alpha_2G=1.042334462039。这是 measured-count-space 条件值；未测 PaddleOCR/文本 Specialist replicas 可容纳，不能认证真正 Specialist exhaustion。

4. **真正 1G infeasible / 2G feasible？** 已测空间中有 20 个 contexts，全部在 B6；每个 window midpoint 经 fixed LP 验证 2G feasible，且高于全空间 1G MILP upper bound。但物理范围内未测 replicas 的 upper bound 尚不完整，正式 `SECOND_GENERALIST_CAPACITY_NECESSARY=true` 数量为 0。

5. **1G feasible 但 2G 的 J 更低？** 有 26 个未被 solver censoring 影响的条件性比较，跨 12 个 contexts。最高未 censor 的收益为 B6 lid+ocr，alpha=0.800000000000：J 从 37.605932820 ms 降到 37.205185380 ms，DeltaJ=0.400747440 ms。两路线 max rho 为 0.680467701/0.671431550。这些是 surrogate、非真实 E2E 标签；仍受缺失 replica evidence 限制。

6. **没有 2G 是否因为 Specialist 太快/太便宜？** 不能得出这一单一因果结论。B1/B2 首先是 memory；B3 是 G replica evidence；B4/B5 的已测空间中，2G residency 与背景/热点 Specialist capacity 竞争，同样的 alpha 常只能由 1G+S 承载；B6 已出现条件性 2G 扩展。未知 S scaling 仍是核心 proof gap。

7. **Specialist replicas 是否继续压倒 2G？** 低点到近 alpha_1G 的 minimum-memory witness 中，75/120 contexts 增加 Specialist counts，15/120 改变 S identity。27/120 中同一 workload 的 1G+S minimum feasible memory 比 exact 2G 更低。但 minimum-memory audit 不等于 frozen J 的选择，更不证明所有未测 S replicas 都优于 2G；B6 的正 window 反证了已测空间内的‘总是压倒’。

8. **当前六任务池值得进入多热点 GPU 吗？** 有值得继续核验的 CPU 结构，不能宣布六任务池不合适。但当前不应直接作为严格的 Specialist-exhaustion / 2G-necessity GPU benchmark：所有 2G capacity candidates 未通过完整 replica evidence certificate。此次没有 GPU 执行。

9. **冻结哪个 context 和 A0–A5？** 正式 GPU primary 与 A0–A5 均未冻结。仅按预先冻结的 R2/W2/budget/task tuple 规则保存诊断候选：B6 asr+image+lid，mix_03/I1；alpha_S=0.688544595342，alpha_1G=0.738865356902，alpha_2G=0.775126353936，R2=0.049076596562。下表只是机械计算的未冻结值，不可作为已获认证的 GPU points。

| Diagnostic point | Alpha |
|---|---:|
| A0 | 0.550835676274 |
| A1 | 0.674773703435 |
| A2 | 0.703640823810 |
| A3 | 0.723769128434 |
| A4 | 0.749743656012 |
| A5 | 0.764248054826 |

该诊断候选的 capacity maximizing 1G/2G 状态分别为 Qwen+S 与 Phi+Qwen+S；target ASR 的 G fraction 为 0.092793413/0.152418568，Image/LID 保持 S。原 1G selected state 已含 ResNet×2、Whisper-small×2、ConvNeXt、Distil-Whisper、Whisper-turbo，second G 并不是靠人为禁止这些 measured S 扩展得到的。但在 near-boundary memory witness 中，再增加 Distil-Whisper、Whisper-small 或 Whisper-turbo 的 replica memory screen 仍可行，而 g 未测；这正是不能将 conditional window 冻结为 exhaustion benchmark 的原因。

10. **Task-Rich 还是真实 multi-resource accounting？** 先处理 replica evidence 与真实 multi-resource accounting。Module10 的 operational certificate 合法状态仍出现 runtime instability，说明规划 proxy 的 capacity 不能自动等价于稳定 runtime capacity；而 Module11 已有潜在 2G 结构，因此尚无证据表明必须加 task。此处只给后续建议，不开始任何 profiling/accounting 实验。

## J audit 的可复核边界

冻结 landmark rule 生成 500 个 workload 比较；严格枚举所有 memory/coverage-feasible measured states，使用 task-wise convex relaxation lower bound 安全剪枝。未修改 frozen J objective、初始化、solver 或数值阈值。

冻结 solver 实际使用 rho<1 的域，而 operational admission 要求 rho≤0.8。原 solver route 通过独立 60-digit Decimal certificate 后才计入 operationally accepted state；真实超限不当作 floating-point 问题，也不通过改约束/换 route 修复。

唯一 CPU state/workload 检查共 15838 个：{'SOLVED': 2491, 'OPERATIONAL_PLANNING_INFEASIBLE': 11457, 'NUMERICAL_SOLVER_FAILURE': 1889, 'NUMERICAL_SOLVER_EXCEPTION': 1}；其中 889 个已收敛 route 超过 operational ceiling。另有一个 conservation exception，记录为 NUMERICAL_SOLVER_EXCEPTION，非 physical infeasibility。没有 random restart、没有改 Module9、没有 GPU 重跑。

287/500 个比较有 potential-better numerical/operational censoring；这些不能声称已经取得受约束的全局 J*。CSV 的 J_1G/J_2G 是 best accepted frozen-router outputs，伴随 unresolved count 和完整 witness。53 个 observed positive witness comparisons 中，只有 26 个未受这种 censoring 影响。Benefit 判定只使用现有 numerical certificate interval 是否仍严格正，没有人为性能 threshold。

## 验证与停止

360 个容量 witnesses 已校验所有约束；105 个 envelope 通过独立逐 deployment LP 枚举交叉检查；18 个重复 MILP 的 deployment、routing、alpha 完全一致；20 个条件性 2G region midpoint 全部验证。completion_verification.json 记录已有 Module10（含 invalid archive）及 Module9/source/runtime 文件哈希的最终复核。

该模块执行完成。正式 gate 未被强行分配，不生成虚假的 no-2G 结论，不启动 GPU，不进入 Task-Rich。
