# Module 8F-3 — Generalist–Specialist Structural Dominance Audit

Execution PASS. Diagnosis: **BOTH**. STRUCTURAL_AUDIT_ONLY=true; GPU_RUNS=0.

1. Frozen task-level advantage and marginal residency cost (adding S does not subtract resident G memory):

| Task | fastest admitted G | fastest admitted S | speedup | additional safe MiB |
|---|---|---|---:|---:|
| text | qwen | distilbert | 11.842677658 | 833.140466441 |
| image | qwen | resnet50 | 25.394273384 | 670.955788974 |
| ocr | minicpm | paddleocr | 26.431121058 | 630.965046585 |
| asr | qwen | distil_whisper | 4.473136767 | 4079.055723696 |
| visual_qa | qwen | NONE | — | — |
| lid | qwen | whisper-small | 5.165733927 | 1795.139991692 |

2. ResNet50、PaddleOCR、DistilBERT 同时呈现排序最突出的速度/低显存组合；不定义任意“巨大”阈值。Whisper-small 对 LID 是最快 admitted Specialist，并同时覆盖 ASR。Distil-Whisper 对 ASR 最快，但 residency cost 更高。所有 12 cells 的 lambda-weighted demand saving 和 value/MiB 见 task_level_dominance.csv。

3. Primary reference = Phi + DistilBERT/ResNet50/PaddleOCR/Whisper-small/Distil-Whisper。各 budget 下 reference 是否 fits 都明确记录；没有将预算外 reference 作为 serving baseline。Phi 不 admitted text/image/OCR/LID，因此移除前四个分别失去文本/图像/OCR/LID coverage。移除 Distil-Whisper 保留 Whisper-ASR fallback，增加 pressure。额外 Qwen/MiniCPM references 展示合法 G fallback；全部 task flows、demand multipliers 保存在 specialist_removal_cost.csv。

| C12 reference G | removed S | safe MiB saved | status | delta demand (s/s) | delta z |
|---|---|---:|---|---:|---:|
| minicpm | distil_whisper | 4079.055724 | FEASIBLE | 0.12939986540738113 | 0.06365525240534042 |
| minicpm | distilbert | 833.140466 | COVERAGE_INFEASIBLE | undefined | undefined |
| minicpm | paddleocr | 630.965047 | CAPACITY_INFEASIBLE | 11.034888868437388 | 11.384285416423321 |
| minicpm | resnet50 | 670.955789 | CAPACITY_INFEASIBLE | 7.174030467374616 | 7.273069123173319 |
| minicpm | whisper-small | 1795.139992 | COVERAGE_INFEASIBLE | undefined | undefined |
| phi4mm | distil_whisper | 4079.055724 | FEASIBLE | 0.23086037397616987 | 0.17981321631195557 |
| phi4mm | distilbert | 833.140466 | COVERAGE_INFEASIBLE | undefined | undefined |
| phi4mm | paddleocr | 630.965047 | COVERAGE_INFEASIBLE | undefined | undefined |
| phi4mm | resnet50 | 670.955789 | COVERAGE_INFEASIBLE | undefined | undefined |
| phi4mm | whisper-small | 1795.139992 | COVERAGE_INFEASIBLE | undefined | undefined |
| qwen | distil_whisper | 4079.055724 | FEASIBLE | 0.22831038408332005 | 0.20724621961385337 |
| qwen | distilbert | 833.140466 | FEASIBLE | 0.2232183766524105 | 0.24380539248914357 |
| qwen | paddleocr | 630.965047 | COVERAGE_INFEASIBLE | undefined | undefined |
| qwen | resnet50 | 670.955789 | CAPACITY_INFEASIBLE | 4.481003956092111 | 4.580042611890814 |
| qwen | whisper-small | 1795.139992 | CAPACITY_INFEASIBLE | 1.8626566173575558 | 2.3097943006096746 |

4. 完整 13,824 向量；4,326 quality/capability-covered LP states。9 个全 bound-valid Pareto states，其中 8 个 <=49140 MiB。物理 frontier 的 G 数分布 {2: 4, 1: 4}；S 数分布 {4: 4, 5: 4}。全部满足 N_S>N_G。第 9 个状态含 3G，safe memory 超过物理预算，不能作为可执行解。

5. 没有 Generalist-only 或 N_G>N_S 的 Pareto region。2G+4S/5S 解是实际 Hybrid（冻结 hybrid 与 Specialist-heavy predicates 重叠），不能将其误称 Generalist-heavy。

6. I3 下 M1→M6 oracle compositions：

| Cell | budget MiB | Oracle deployment | N_G/N_S | z* | RC gap | v2 gap |
|---|---:|---|---|---:|---:|---:|
| C03 | 17007.174054952 | {"distilbert": 1, "paddleocr": 1, "phi4mm": 1, "resnet50": 1, "whisper-small": 1} | 1/4 | 0.768783698 | 0.000000000 | 0.000000000 |
| C04 | 27980.189424960 | {"distil_whisper": 1, "distilbert": 1, "paddleocr": 1, "qwen": 1, "resnet50": 1, "whisper-small": 1} | 1/5 | 0.531554485 | 0.108015261 | 0.446293314 |
| C05 | 32019.254406267 | {"distil_whisper": 1, "distilbert": 1, "paddleocr": 1, "qwen": 1, "resnet50": 1, "whisper-small": 1} | 1/5 | 0.531554485 | 0.108015261 | 0.446293314 |
| C08 | 34567.553379621 | {"distil_whisper": 1, "distilbert": 1, "paddleocr": 1, "qwen": 1, "resnet50": 1, "whisper-small": 1} | 1/5 | 0.531554485 | 0.108015261 | 0.446293314 |
| C09 | 43736.541926295 | {"distilbert": 1, "minicpm": 1, "paddleocr": 1, "phi4mm": 1, "resnet50": 1, "whisper-small": 1, "whisper_turbo": 1} | 2/5 | 0.422433449 | 0.394232591 | 0.819893052 |
| C12 | 49140.000000000 | {"distilbert": 1, "minicpm": 1, "paddleocr": 1, "qwen": 1, "resnet50": 1, "whisper-small": 1, "whisper_turbo": 1} | 2/5 | 0.420496106 | 0.400656209 | 0.828277806 |

7. 8F-2 没有呈现类别转变有两个原因：结构最优本身全程 Specialist-heavy；同时 v2 在 I3 的全部 memory points 返回 Phi+4S seed。正式 runtime objective 在同一非 steady tier 内不会因更低 mean latency 自动接受；高 intensity candidates 的 unstable/rollback 证据和 oracle 是否进入 Top-B/被测量保存于 search_structural_gap_audit.csv。所有隐藏 2G oracle 未被 GPU evaluated。不能从低 z 推断它一定改善真实 latency。

8. RC-SG：oracle deployment 相同 {RC["equal_oracle_deployment_cells"]}/12；Pareto {RC["Pareto_cells"]}/12；严格正 structural gap {RC["strict_positive_gap_cells"]}/12。gap mean/median/max={RC["mean"]:.12f}/{RC["median"]:.12f}/{RC["max"]:.12f}。不定义“近最优”阈值；完整分布在 JSON/CSV。

9. v2：oracle deployment 相同 {V["equal_oracle_deployment_cells"]}/12；Pareto {V["Pareto_cells"]}/12；严格正 gap {V["strict_positive_gap_cells"]}/12。gap mean/median/max={V["mean"]:.12f}/{V["median"]:.12f}/{V["max"]:.12f}。

10. Uniform load scaling: r(alpha)=alpha*r_base and z(alpha)=alpha*z_base. At fixed budget, structural-oracle composition cannot change with alpha while it remains feasible; if its minimum z exceeds rho_max no deployment is feasible. RC-SG base demand and all greedy values scale by the same alpha, so its choices are unchanged. Specialist throughput advantage is retained; absolute pressure difference grows with load, relative ratio does not. More load is not a mechanism for naturally hurting Specialists here.

| C12 method | z | feasible memory | feasible capacity | alpha capacity boundary |
|---|---:|---|---|---:|
| Generalist_only_fallback | 11.384285416 | True | False | 1.095855244 |
| Frozen_Generalist_centric | 8.875622487 | True | False | 1.405594806 |
| RC_SG | 0.588970482 | True | True | 21.181925506 |
| structural_oracle | 0.420496106 | True | True | 29.668595474 |
| v2 | 0.768783698 | True | True | 16.227618902 |

11. Memory counterfactual events are exact affine residency/Pareto events, not selected coarse multipliers. Main event table:

| Budget | fastest-S residency lost above c_M | first oracle S count reduction | first forced G multi-task sharing |
|---|---:|---:|---:|
| 17007.174054952 | 0.490707350902 (ALREADY_FALSE_AT_REAL_POINT) | None | None |
| 27980.189424960 | 1.860748959778 (LOSS_IMMEDIATELY_ABOVE_BREAKPOINT) | 1.8607489597780855 | 3.7919728660260033 |
| 32019.254406267 | 2.365048543689 (LOSS_IMMEDIATELY_ABOVE_BREAKPOINT) | 2.36504854368932 | 4.819672131147541 |
| 34567.553379621 | 2.683217753121 (LOSS_IMMEDIATELY_ABOVE_BREAKPOINT) | 2.6832177531206645 | 5.468061051441492 |
| 43736.541926295 | 3.828016643551 (LOSS_IMMEDIATELY_ABOVE_BREAKPOINT) | 1.2409703504043121 | 1.2409703504043121 |
| 49140.000000000 | 4.502668245063 (LOSS_IMMEDIATELY_ABOVE_BREAKPOINT) | 1.8965280386663355 | 1.8965280386663355 |

At M1, c_M>1 immediately loses minimum coverage residency: this is infeasibility, not a successful Generalist trade-off. M6 first identity switch is c_M=1.0601398985 (2G+5S remains); first S count reduction/forced sharing is c_M=1.8965280387 (2G+4S remains Specialist-heavy). Later Hybrid/Generalist-heavy counterfactual regions must be checked against the exported capacity flag, not presented as executable wins.

12. Speed counterfactual uses the fastest quality-admitted G per task and alpha multiplies the actual service-time difference. alpha=0 equalizes service time, but cheap additional Specialist workers still add capacity. Observed transition brackets:

| Budget | first identity change alpha bracket | first G/S count change alpha bracket |
|---|---|---|
| 17007.174054952 | No observed change | No observed change |
| 27980.189424960 | [0.9855521693825722, 0.9855521768331528] | [0.976999893784523, 0.9769999012351036] |
| 32019.254406267 | [0.9855521693825722, 0.9855521768331528] | [0.976999893784523, 0.9769999012351036] |
| 34567.553379621 | [0.9855521693825722, 0.9855521768331528] | [0.976999893784523, 0.9769999012351036] |
| 43736.541926295 | [0.996906965970993, 0.9969069734215736] | [0.9720572680234909, 0.9720572754740715] |
| 49140.000000000 | [0.9966225028038025, 0.9966225102543831] | [0.9720572680234909, 0.9720572754740715] |

257-point screen and <=1e-8 bisection brackets are documented. Continuous sub-grid nonmonotone islands are not exhaustively excluded; no unsupported event-completeness claim. Counterfactuals are explanatory diagnostics only.

13. RC-SG starts from Phi whose admission forces four fast Specialists as capability repairs; only Distil-Whisper is subsequently added in loose cells. Specialist final identity set matches oracle in 8/12 cells. Text/image/OCR choices are largely separate, but speech paths overlap (ASR/LID), Generalists share tasks, and min-max pressure couples the routing problem. Greedy demand-saving is not a pressure objective; zero-demand-gain parallel workers can still lower z. Both the near-separable local repair structure and the remaining global interactions are explicitly recorded.

14. Diagnosis={diagnosis}. Specialist-heavy residency is structurally favored; RC-SG is not exactly pressure-optimal in most cells. Bounded v2 misses sustained loose-budget multi-Generalist alternatives. No Generalist-heavy Pareto region exists, so this is not evidence that hidden Generalist-heavy deployments beat Specialist-heavy ones.

15. Implication: interference/congestion prediction、routing、robustness 与 bounded-search quality 是主要发展方向；也应测试隐藏 multi-Generalist Hybrid 的压力优势是否转化为真实 runtime 优势。完整 9A/9B/9C 不能凭空改变 frozen s0/memory 的结构偏向。后续独立设计真实 G/S trade-off regime 可能有必要，8F-2 永久保留。此处未开始完整 9A 或修改任何实验输入，完成后停止。

Execution notes: all work CPU-only. New audit code initially encountered a frozen-file field-name compatibility error, heterogeneous CSV-schema error, and missing Workload import; original error logs retained. No original file/source/data was repaired or regenerated. All frozen references reverified at closeout.

## Counterfactual feasibility clarification

The original coverage-space memory events are retained and now include post-event pressure/capacity flags. Additional FIRST_CAPACITY_FEASIBLE_* rows apply the unchanged rho_max gate. A coverage-optimal diagnostic with z>rho_max is not a feasible serving oracle.

| I3 budget | first capacity-feasible S count reduction | first capacity-feasible forced G sharing |
|---|---:|---:|
| 17007.174054952 | None | None |
| 27980.189424960 | 1.8607489597780855 | None |
| 32019.254406267 | 2.36504854368932 | None |
| 34567.553379621 | 2.6832177531206645 | None |
| 43736.541926295 | 1.2409703504043121 | 1.2409703504043121 |
| 49140.000000000 | 1.8965280386663355 | 1.8965280386663355 |

For I3 at M2/M3/M4, the first coverage-space forced-sharing event coincides with loss of every capacity-feasible deployment; it must not be promoted to evidence of a feasible Generalist-sharing regime. At M5/M6 the first forced-sharing transition is capacity-feasible. The pure/generalist-heavy regions of the I3 counterfactual map remain pressure-infeasible.

## Greedy interaction evidence at the existing loose budget

Whisper-turbo has zero current-assignment demand-saving value after Distil-Whisper/Whisper-small are resident. Adding it to the RC-SG Phi deployment alone leaves z unchanged. After adding MiniCPM, adding the same Turbo lowers z. All four deployment vectors are already in the frozen 13,824-vector space and fit M6; no routing rule, model set or benchmark was revised.

| Cell | RC z | +MiniCPM z | +Turbo z | +both z | Turbo benefit after MiniCPM |
|---|---:|---:|---:|---:|---:|
| C10 | 0.116288897 | 0.088284812 | 0.116288897 | 0.083407100 | 0.004877712 |
| C11 | 0.449783834 | 0.341469237 | 0.449783834 | 0.322603156 | 0.018866082 |
| C12 | 0.588970482 | 0.447137683 | 0.588970482 | 0.422433449 | 0.024704235 |

This is a concrete Generalist/Specialist capacity interaction. Most local quality repairs remain simple; the complete min-max placement problem is not independent per task. No runtime advantage is inferred from this CPU pressure comparison.

RC-SG 的 1G/no-replica/no-eviction 限制属于冻结 baseline 定义；其 oracle gap 不等于实现错误。v2 的 structural gap 说明 bounded search 对低压力候选的覆盖不足，不证明其 runtime objective 错误或 oracle 是真实 E2E 最优。
