# Module 12 Specialist Efficiency Sensitivity Audit

已完成 CPU-only counterfactual audit；GPU 未启动。当前真实 Specialist pool 继续冻结为 `EFFICIENCY_OPTIMAL_SPECIALIST_REGIME`。原始数据、Module 9、Module 11B 均未修改。

覆盖 9 个 development workloads × B1/B3/B6；global frozen grid 1512 points。Per-task 仅 latency-only / memory-only，共 1890 行，包含共享 baseline；未运行 per-task 二维网格。

优化语义：在已测 replica counts 上全局最小化原 J，保持 g、admission、capability 不变，hard rho≤0.8。属于 CPU 结构参考，不是 frozen V3 bounded-search 的真实 outcome。N_G/N_S 是 replica 数；identity 数单独记录。

## 必答结果

1. (1,1)：27 个 context 的 base phase 全部 S_DOMINANT；primary phase 为 {'S_DOMINANT': 20, 'MULTI_G': 7}。其中 7 个 B6 context 已存在 ≥2 个 G replicas，不能算 scaling 新触发第二个 G。
2. Latency-only 首次 observed structure/routing change 为 pressured mix_03/I2/B1 的 ASR split：gamma bracket (1.2125,1.25]，其余比例不变。这里是 route transition，不是减少 Specialist identity。Latency-only 首次 T1 的 refined bracket 为 (1.9375,2.0]（mix_03/I3/B6，13→12 S replicas）。
3. Memory-only 最早 T1 refined bracket 为 (1.00,1.05]：B3/B6 部分 context 减少 Specialist replicas，baseline deployment 在高端已 memory-infeasible。5% 是 refinement 的区间精度，不是性能门槛；不能说精确阈值就是 1.05。
4. 二维 frontier 是带阶梯和 nonmonotone reentries 的离散边界，不是一条统一 gamma×mu 曲线。B1 的多数 mu>1 点直接 infeasible；B3/B6 多为 replica reduction。所有 event-specific Pareto-minimal 点和 reentries 见 transition_frontiers.json。
5. Latency 驱动主要来自 ASR/LID overlapping speech families，Text 次之；memory 对所有 family 都可触发 replica reduction。类别相关，不可将 ASR/LID 效应相加。
6. OCR/Image/Text 的真实最快 admitted G/S speedup 为 26.43×/25.39×/11.84×；OCR/Image latency-only 扫到 4× 无结构变化，支持其效率优势维持 Specialist-heavy。该关联不是 GPU 因果证据。
7. 所有 global/per-task points 均未产生新增 T2/T3。第二 G 在部分 B6 baseline 已有；scaling 后没有新 context 从 <2 G 转为 ≥2 G。
8. 首次减少 N_S：memory bracket (1,1.05]；latency refined bracket (1.9375,2.0]。此处减少主要是 replicas，不能一概写成减少 Specialist identities。
9. 在本范围内 T6=0：没有 low-load task 完全撤掉其所有 dedicated Specialist paths 的 observed region。J 不包含实测 fixed-cost，不据此作经济结论。
10. 同一 mix_03 的 I1→I2→I3 上，G→S→S+G 序列 0 个；已有 S-dominant→split 或 infeasible，未验证完整三阶段序列。
11. 描述性 HYBRID 仅在 global 15/1512 points 出现：例如 mix_03/I1/B1 gamma=4,mu=1；mix_03/I1/B3 gamma=3,mu=2.5；低-load B3 gamma=4,mu=2.5。多为 ASR 向 G 迁移，OCR/Image 等仍走 S，不能称为所有任务 balanced，也不是证明真实 Specialist exhaustion。
12. 各 task 的 REAL_MODEL_SEARCH_TARGET 如下。Memory 的首个区间代表 replica transition；latency censored 区间只是搜索下界，不能保证更大比例必然 transition。

| Task | Latency-only target | Memory-only target |
|---|---|---|
| OCR | >4.0（right-censored） | [1.0, 1.1] |
| IMAGE | >4.0（right-censored） | [1.0, 1.1] |
| TEXT | [1.5, 2.0] | [1.0, 1.1] |
| LID | [1.1, 1.25] | [1.0, 1.1] |
| ASR | [1.1, 1.25] | [1.0, 1.1] |

13. 下一步真实模型搜索优先 ASR/LID，其次 Text；Image/OCR 在当前 latency 范围仍未 transition。后续候选必须通过 frozen quality admission，实测 latency/memory，构造 Pareto frontier，再按已冻结的 region distance + quality/memory/model-ID tie-break 选择；本模块未检索或下载任何模型。

## 数值与证据限制

10 ns global J certificate 使用原数值标准。Baseline deployment 在 counterfactual 下的 memory/capacity 或独立 convex lower-bound loss，及旧 route 的 objective loss，均单独审计。`NUMERICAL_TIE_OR_UNRESOLVED` 不被当作已证实的 composition threshold；route-only transition 的支持独立记录。global ANY bracket 的二分仅在第一个 coarse bracket 内、relative width≤5% 或≤6次；不细化 per-task axes。
保护文件 SHA 核对 3329 个，PASS；全部 feasible cache certificate PASS；最大 J global gap 9.278e-09 s。完整 frontier 仍限于已测 counts；未测 counts 未补齐。

phase labels 用于描述，未参与 objective。G/S split 不等于 Specialist physical exhaustion。没有重开 Module 11B，没有 fixed-cost/low-QPS/multi-hotspot/Task-Rich runs。

## 主要文件

- `global_sensitivity_grid.csv`, `phase_map.csv`, `global_phase_map.svg`
- `transition_frontiers.json`, `transition_refinement.csv`, `transition_witnesses/`
- `per_task_sensitivity.csv`, `per_task_transition_brackets.csv`
- `speed_vs_memory_attribution.json`, `real_model_search_targets.json`
- `specialist_selection_rule.json`, `cpu_validation.json`

Module 12 完成后停止；未启动下一阶段真实模型选择。
