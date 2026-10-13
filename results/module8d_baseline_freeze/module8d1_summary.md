# Module 8D-1 — Baseline Definition Freeze + Measurement Reuse Audit

BASELINE_FREEZE_PASS = **true**. MODULE8D2_READY = **true**. 本阶段新增 GPU experiment = **0**；完成后停止，尚未启动 8D-2。

## 冻结的 baseline

所有以下列出的非零模型均为 **x=1**，其余正式十二维向量均为 0；完整向量、路由和质量见 JSON/CSV。S4 = DistilBERT + ResNet50 + PaddleOCR + Whisper-small。

| Baseline | demand-balanced / speech-heavy / vision-heavy / text/OCR-heavy / reasoning/VQA-heavy | no-VQA |
|---|---|---|
| B0 Seed-only | Phi + S4，N_G=1,N_S=4 | S4，N_G=0,N_S=4 |
| B1 Qwen-centric + Quality Repair | Qwen + PaddleOCR，N_G=1,N_S=1 | 同一个 always-on deployment |
| B2 Specialist-greedy + Minimal Generalist | Qwen + S4 + Distil-Whisper，N_G=1,N_S=5 | S4 + Distil-Whisper，N_G=0,N_S=5 |
| B3 Static Hybrid | Phi + S4，N_G=1,N_S=4 | 同一个 static deployment，Phi 保留但无 VQA 流量 |

B0 六个 seed 和 canonical routing 经原 empirical integer MILP/LP 精确复现，与 8C 搜索前 witness 相同。B1 使用固定六任务 repair universe：OCR→PaddleOCR，text/image/ASR/VQA/LID→Qwen；不因零 λ 剪掉模型。B2 task→model：text→DistilBERT，image→ResNet50，OCR→PaddleOCR，ASR→Distil-Whisper，LID→Whisper-small，active VQA→Qwen。所有选择仅由 frozen admission 和真实 s0 得到。

B3 aggregate λ 是六个原向量逐任务算术平均，λ digest=`a958f9e03d5d50d11c9143a9de0563db356fa4a2c4fc9ee96b05672d94677990`。先冻结并 hash，再使用原 MILP seeder+canonical LP 生成 static deployment；所有六 workload 只重新计算 routing，不改变 deployment。

B4 random seeds = **1103, 2207, 47**。项目历史四任务 campaign 的两个 RNG seeds 1103/2207 被保留，第三个使用本次用户示例 47；不存在全项目三 seed policy。RNG seed 与测量 seeds 7101/7202/7303 分开。首轮选择在复用/质量结果审计前固定，由原 random selector 生成，CPU tripwire 证明 LP ranking 调用数 0；后续 trajectory 和 final deployment 未伪造或模拟。

## 严格 measurement reuse

扫描 **2954** 个历史 JSON artifact，找到 **581** 份多任务 raw measurement，包括 Module 8C 的 96 份、7C 与更早 real results。12 个身份条件必须全部相等，缺少证据也禁止复用；不使用浮点 rounding/tolerance。额外要求测量 phase/seed 匹配、原 config 完全一致及有效完整窗口。模型权重/adapter/环境/driver/核心源码与 8C 565 artifact 已重新验证，详见 source_identity_validation.json。

| Baseline | 已有状态可复用 | 缺失状态 | 新 GPU timed runs |
|---|---:|---:|---:|
| B0 | 6/6 | 0 | 0 |
| B1 | 0/6 | 6 | 30 |
| B2 | 0/6 | 6 | 30 |
| B3 | 5/6 | no-VQA 1 | 5 |

固定 baseline 每个 state 使用 3 search-protocol repetitions + 2 independent confirmations。B0 复用每个 8C seed 的三次 search raw 及两次 seed confirmation，共 30 份原始测量。B3 前五组与 B0 λ/vector/routing 完全相同，复用相同 25 份测量；这些不是额外独立样本。

B2 no-VQA 的 deployment 与 8C final 相同，但冻结 greedy routing 将 ASR 全部交给 Distil-Whisper，8C canonical routing 在两 speech 模型间分流；**routing identity 不相等，REUSE_FORBIDDEN**。B3 no-VQA 保留 Phi，其 replica vector 与原 8C seed/final 不同，也禁止复用。Module 7C 与其他旧 workload 不满足 exact λ/quality-admission/protocol identity，不能借用同名 deployment。

B4 18 个 RNG/workload job 的 seed 都可复用。首轮 36 个候选出现次数中 **30** 个有完整三次 search measurement；剩余 **6** 个出现次数对应 **4** 个不同 measurement state。235 个已选复用引用指向 **66** 份不同原始 raw；每项原 path、SHA256、12 条 equality checks、decision 及 measurement phase/seed 已保存。未知后续状态与 final 只可在实际选择后重新通过同样审计；seed/final confirmation donor 不得在同一个 Random job 内重叠。

## GPU campaign budget（仅规划）

B4 每 job：top_b=2、max_iterations=3、patience=1、≤6 valid unique states **含 seed**、每 state 3 repetitions、seed/final 各 2 confirmations；predictor=None。最多 18 jobs×6 = **108** 个 logical state evaluations，未复用前最多 **396** 个 timed runs。仅复用 seed 可降为 306；计入已冻结随机首轮 exact donors 后，新 state evaluation 出现次数最多 **60**，新增 physical runs 保守上限 **216**（含 36 次未知 final confirmations）。这不是承诺执行满预算，也不是 60 个保证不同的全局 deployment。

B0–B3 合计必须新增 **65** 次 timed run；总 campaign 保守新增上限 **281**。若对未来新产生的证据执行相同严格跨 job 去重，总体另一个更紧上限为 **272**。本阶段已知缺失 measurement 为 83 个引用、精确去重后 74 个不同 run；后续反馈轨迹尚未知，不能报虚构精确成本。

每 run warmup+measurement = 220 秒，保守 ceiling 对应 **17.17 GPU 小时**，另加 startup/load/drain。以 8C 实测 process 中位数 260.15 秒仅做成本 diagnostic，约 **20.31 小时**；非工期保证。失败保持原证据并停止该 run，不以 retry 扩大预算。

## Quality 与 fairness gate

全部正流量 routing 仅使用冻结的 **21** 条 admitted paths；正式排除的 6 条路径无流量。每个 baseline/workload 及随机初始候选均保存 effective_task_quality_summary：按路由比例加权 frozen audit metric，OCR 保留 secondary CER，零流量 task 标为 inactive。此值是 reporting diagnostic；WER/CER 的加权点估计不是新的 pooled transcript measurement，runtime objective 不变。B4 final quality 必须在未来 final routing 确定后按同一个规则填写。

不存在根据 8C final outcome 反向定义 baseline，也不存在 post-hoc best-of-baselines/seed selection。B1/B2 决策可由 frozen rule 自动复现，B3 六组 deployment 相同。Proposed 与 Random **每个 workload×RNG seed** 的搜索/确认预算严格相同；用户要求的三次 Random replicate 总预算显式为 Proposed 单次的 **3×**，不可合并成 18-state search，也不可只报最优 seed。三个 seed 全部报告，并预声明等权 mean/median/range。Exact donor 被使用时仍消耗 logical unique-state slot；只在候选已经随机选中后 lazy import，不能预填未访问状态。

全部 fairness checks PASS，remaining blockers = []。**可以进入 Module 8D-2 Real Baseline Campaign，但当前没有启动。**
