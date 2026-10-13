# Module 6A — Final Evaluation Audit & Paper Experiment Planner

审计日期：2026-10-01（用户时区 Asia/Shanghai）。阶段状态：**审计完成，停止；未启动后续实验**。

只读取既有实验并生成数据、matplotlib 图和规划文档。新增 GPU experiment=0；新增 synthetic algorithm run=0；算法修改=0；workload/预算/搜索决策修改=0。原始结果文件的完整 SHA-256 快照覆盖 1941 个文件；构建结束后逐项核验没有变化。模型权重使用历史 hash manifest 的记录值，本阶段没有重新加载或重测权重。

## 统一实验索引与执行状态口径

`experiment_manifest.json` 覆盖 23 个现有 results/ 目录，包括所要求的全部 16 个阶段、corrected 分析层及 legacy stages。每项记录 module、purpose、workload、algorithm、deployment、routing、run 数、GPU/synthetic 口径、metric source、decision、source hashes 和 report path。`new_physical_attempts` 与 `referenced_completed_runs` 分开；同一个物理 run 被多个分析引用不能再计数。`input_inventory_sha256.json` 是所有旧结果的完整文件索引；manifest 也保留历史 source/model hash manifests。

| Experiment | Module | Owned physical attempts | Referenced completed runs | Decision |
| --- | --- | --- | --- | --- |
| data_quality_audit | Audit | 0 | 129 | PASS |
| deployment_search_foundation | 3A | 0 | 0 | PASS |
| deployment_search_multiworkload | 3C | 144 | 144 | PARTIAL SUPPORT |
| deployment_search_real_pilot | 3B | 16 | 30 | PILOT OBSERVATION ONLY |
| deployment_seed | 1 | 0 | 0 | PASS |
| dynamic_adaptation_foundation | 5A | 0 | 0 | PASS |
| dynamic_adaptation_real_pilot | 5B | 87 | 87 | PASS |
| performance_model_active_sampling | 2C | 18 | 69 | MEASUREMENT COMPLETE |
| performance_model_comparable_groups | 2G | 45 | 45 | INSUFFICIENT COMPARABLE GROUPS |
| performance_model_comparable_groups_corrected | 2G corrected | 0 | 45 | INSUFFICIENT COMPARABLE GROUPS |
| performance_model_foundation | 2A | 0 | 18 | FOUNDATION |
| performance_model_generalization | 2D | 0 | 63 | PARTIAL |
| performance_model_generalization_corrected | 2D corrected | 0 | 69 | PARTIAL |
| performance_model_local_calibration | 2E | 0 | 69 | FAIL |
| performance_model_local_calibration_corrected | 2E corrected | 0 | 69 | FAIL |
| performance_model_profiling | 2B | 33 | 33 | MEASUREMENT COMPLETE |
| performance_model_ranking | 2F | 0 | 69 | INSUFFICIENT COMPARABLE GROUPS |
| performance_model_ranking_corrected | 2F corrected | 0 | 69 | INSUFFICIENT COMPARABLE GROUPS |
| routing_search_foundation | 4A | 0 | 0 | PASS |
| routing_search_real_pilot | 4B | 9 | 9 | NEUTRAL |
| stage1_5 | Legacy Stage 1.5 | 0 | 0 | HISTORICAL |
| stage2a | Legacy Stage 2A | 2 | 2 | HISTORICAL |
| stage2b | Legacy Stage 2B | 16 | 16 | HISTORICAL |


Data Quality Audit 是执行状态语义的唯一来源。旧 `results/data_quality_audit/audit_counts.json` 仅覆盖 130 次早期尝试，并不是当前总量。本阶段读取其唯一物理 run ledger，并按后续 CampaignRuntime 已采用的同一 DQA v1 规则，重核 3C/4B/5B 的完整 window、drain、harness footer、raw hashes 和 completed stability，生成 `execution_status_ledger.jsonl` 补充层；没有覆盖旧 audit。`gpu_experiment_scale.json` 提供统一全项目 **results/ timed-serving** 口径。

| Phase | Physical attempts | Completed | External interruption | Genuine unstable |
| --- | --- | --- | --- | --- |
| Module 2B/2C profiling | 33 | 33 | 0 | 6 |
| Module 2D/2E/2F source | 18 | 18 | 0 | 0 |
| Module 2G | 45 | 45 | 0 | 11 |
| Module 3B | 16 | 15 | 1 | 0 |
| Module 3C | 144 | 144 | 0 | 0 |
| Module 4B | 9 | 9 | 0 | 0 |
| Module 5B | 87 | 87 | 0 | 0 |
| Stage 2A | 2 | 2 | 0 | 0 |
| Stage 2B | 16 | 16 | 0 | 0 |


- Total physical timed-serving attempts：**370**。
- Valid completed runs：**369**；valid latency runs：**369**。
- External interruptions：**1**；genuine completed unstable：**17**；recoverable：**12**；steady：**340**。
- 实际 workload 到达率向量：**24**；论文主 benchmark 命名 workload：**3**。
- Unique deployment states：**13**；unique routing fraction states：**32**；absolute routing-rps states：**70**。
- 后续新 ledger 记录 search physical attempts：**168**；post-search confirmation attempts：**72**。
- 另有 **13** 次 isolated accepted-path reconfiguration operations。它们与 setup/teardown 是不同操作，不能混入 timed-serving repetition 总数；候选 profiling 的重复也不能冒充 switch 次数。

去重按 active model counts、positive arrival vectors、routing fractions 计算，数值保留 12 位小数，忽略零项。GPU run 总量仅覆盖 results/ 下可追溯的 timed-serving 尝试，不声称穷尽 benchmark/ 等目录中历史 standalone/NVML/Nsight 的全部 GPU 活动。`raw_inventory_coverage.json` 单独核验所有原始 harness artifacts 是否进入 ledger，以及重复副本；无外部原因证据的缺失结果只能标 uncertain，不能猜测为外部中断。Completed unstable 保留有效观察，不因 raw failed 就重标中断。

## RQ1 — Scalability / Non-enumeration

Can the method search Generalist–Specialist deployments without exhaustive deployment enumeration?

| Models | Total generated candidates | Max candidates / round | LP solves | Synthetic runtime evaluations | Search enumeration calls |
| --- | --- | --- | --- | --- | --- |
| 5 | 17 | 5 | 11 | 8 | 0 |
| 10 | 37 | 10 | 26 | 9 | 0 |
| 20 | 77 | 20 | 56 | 9 | 0 |
| 30 | 117 | 30 | 86 | 9 | 0 |


支持结论：tested search 没有 exhaustive deployment enumeration，单轮候选至多 M，runtime evaluations 受预算限制。5-model 的独立 oracle 枚举 32 个 binary states，其中 31 个可行 objective evaluations；J*=20.833333 ms、J_search=22.727273 ms，gap=9.090909%。Oracle 不在正式 guided search 内调用。10/20/30-model 没有 exhaustive gap。所有这些 latency 是人工 synthetic objective，不能描述为真实 GPU scalability 或全局最优；O(M) 是单轮邻域候选上界，不是 MILP 时间复杂度。缺少 guided search 总 wall-clock series；现有 Module 1 MILP solve timings不能替代。

数据：`rq1_scalability.json` / `table_scalability.csv`。

## RQ2 — Static Deployment Search

Does LP-guided bounded-feedback search improve serving performance over the MILP feasible seed and simple search baselines?

比较 B0 MILP feasible seed、两个 B1 random-neighbor + runtime-feedback trial、B2 LP-guided + runtime-feedback、B3 LP-only。Main proposed predictor=None，routing=canonical min-max。三种 W 的首个 seed 都是 qwen。B0 优化最小可行内存，因此是明确且较弱的 latency baseline；不把它称为 runtime-optimal seed。

| Workload | Method | Final deployment | Search mean ms | Confirmation mean ms | State evals | LP solves | Accepted | Rollbacks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 | B0 | qwen | 301.480 | 273.045 | 1 | 1 | 0 | 0 |
| W1 | B1_seed_1103 | distilbert+paddleocr+qwen+whisper | 67.469 | 65.346 | 6 | 10 | 3 | 2 |
| W1 | B1_seed_2207 | qwen+resnet50+whisper | 82.944 | 69.349 | 6 | 10 | 2 | 3 |
| W1 | B2 | paddleocr+qwen+resnet50+whisper | 59.373 | 54.770 | 6 | 10 | 3 | 2 |
| W1 | B3 | paddleocr+qwen+resnet50+whisper | 59.373 | 55.674 | 6 | 10 | 3 | 0 |
| W2 | B0 | qwen | 290.747 | 274.008 | 1 | 1 | 0 | 0 |
| W2 | B1_seed_1103 | distilbert+qwen+resnet50+whisper | 57.174 | 54.664 | 6 | 10 | 3 | 2 |
| W2 | B1_seed_2207 | qwen+resnet50+whisper | 76.771 | 73.977 | 6 | 10 | 2 | 3 |
| W2 | B2 | distilbert+qwen+resnet50+whisper | 57.174 | 54.430 | 6 | 10 | 3 | 2 |
| W2 | B3 | distilbert+qwen+resnet50+whisper | 57.174 | 54.543 | 6 | 10 | 3 | 0 |
| W3 | B0 | qwen | 219.892 | 249.541 | 1 | 1 | 0 | 0 |
| W3 | B1_seed_1103 | distilbert+qwen+resnet50+whisper | 54.289 | 55.753 | 6 | 10 | 3 | 2 |
| W3 | B1_seed_2207 | qwen+resnet50+whisper | 76.606 | 71.250 | 6 | 10 | 2 | 3 |
| W3 | B2 | paddleocr+qwen+resnet50+whisper | 52.597 | 54.186 | 6 | 10 | 3 | 2 |
| W3 | B3 | paddleocr+qwen+resnet50+whisper | 52.597 | 53.138 | 6 | 10 | 3 | 0 |


B2 相对**自身 seed**的两类证据分别为：

| Workload | Search seed ms | Search final ms | Search reduction | Confirmation seed ms | Confirmation final ms | Confirmation reduction |
| --- | --- | --- | --- | --- | --- | --- |
| W1 | 301.480 | 59.373 | 80.31% | 261.274 | 54.770 | 79.04% |
| W2 | 290.747 | 57.174 | 80.34% | 269.719 | 54.430 | 79.82% |
| W3 | 219.892 | 52.597 | 76.08% | 234.015 | 54.186 | 76.84% |


3/3 改善方向在独立 confirmation 保持。搜索每状态 3 repetitions；每策略 seed/final 各有 2 个 independent confirmation seeds。B0 的 seed 与 final 是同一状态，本 audit 使用 search final 的 3 次和 confirmation final 的 2 次分别统计，没有再把 confirmation seed 的另 2 次并入 main final row。

严格 frozen noise-aware objective 比较：B2 胜出 **4/6** 个真实 random trials（W1 2/2、W2 1/2、W3 1/2）；不等于所有 mean 比较都显著胜出。共享 exact cache 的策略可能使用完全相同的物理搜索重复；equal six-state budgets 不能描述为各自独立六次 GPU 实验。B3 使用 runtime measurement 作报告，但 acceptance criterion 只依赖 LP；其六次 measurement 不等于 LP-only 决策必须支付这些 GPU 成本。B3 的 unchosen Top-B 不作为 runtime rollback。

原 `per_workload_metrics.json` 的部分描述均值合并 search 和 confirmation，3B/3C summary中的旧数字保留用于历史核对。新 Table A 不合并：15 个 policy records × 两类来源 = **30 行**。每行标记 Measurement Stage、Repetitions、Raw Sources；搜索成本在 confirmation 行重复只作上下文，禁止累加。当前 3C frozen decision 为 **PARTIAL SUPPORT**，不能通过改变本审计的汇总口径升格。

数据：`rq2_static_search.json` / `table_static_search.csv`。图：`fig_static_search.png`；Figure 2 为 `fig_search_efficiency.png`，x=state evaluations，y=independent confirmation mean。搜索策略均使用 6 次预算，图可描述 endpoint tradeoff，但不能宣称已描出完整 efficiency frontier。

## RQ3 — Analytical Guidance

Is LP pressure relief informative for identifying useful deployment mutations?

Δz=current canonical max pressure−candidate max pressure；ΔJ=current weighted mean E2E−candidate weighted mean E2E（ms），正值表示改善。保留 60 条 policy transitions，同时按 workload/current/candidate deployment 去重得到 **31** 条 transition。全部来自 search-time measurement，confirmation 没有用于该关系。

Policy transition Spearman：W1 0.784、W2 0.872、W3 0.716。Unique transition sensitivity：W1 0.644（n=11）、W2 0.780（n=9）、W3 0.621（n=11）。只作描述：selected Top-B population、cache reuse 和 dependent transitions 均限制解释。Figure 3 使用去重数据、workload-specific marker，不拟合，不声称因果。

**B2 与 B3 在三个 workload 的最终 deployment 相同**；positive LP relief but worse real runtime=0；runtime feedback correcting such misjudgment=0。普通 rollback labels还包含“改善但不是本轮最优”的候选，不能全部当作 LP 错判纠正。安全表述必须是：

> runtime feedback provides an online correction / safety mechanism, while the current static benchmark does not isolate an independent final-performance gain from feedback.

不允许将 `runtime feedback improves static search` 包装成 confirmed contribution。数据：`rq3_guidance.json`；图：`fig_lp_vs_runtime.png`。

## RQ4 — Dynamic Adaptation

When workload changes, can warm-start adaptation reduce search and reconfiguration overhead relative to cold restart?

D0=保持 actual old deployment 的控制；D1=同一 actual old 出发、显式 reset 到 MILP seed 再 cold search；D2=primary warm sequence，仅 D2 final 驱动后续阶段。

| Transition | Method | Action | State evals | Search mean ms | Search weighted P95 ms | Confirmation mean ms | Accepted switches | Loaded/unloaded | Isolated path time s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1→W2 | D0 | NO_ADAPT | 1 | 58.752 | 94.823 | N/A: no independent D0 confirmation | 0 | 0/0 | null: no switch |
| W1→W2 | D1 | RUNTIME_EVALUATION_BUDGET_REACHED | 6 | 55.911 | 93.855 | 53.544 | 4 | 3/3 | 10.948 |
| W1→W2 | D2 | SWITCHED | 5 | 43.181 | 73.418 | 42.115 | 1 | 1/0 | 3.344 |
| W2→W3 | D0 | NO_ADAPT | 1 | 39.780 | 70.920 | N/A: no independent D0 confirmation | 0 | 0/0 | null: no switch |
| W2→W3 | D1 | RUNTIME_EVALUATION_BUDGET_REACHED | 6 | 51.801 | 74.897 | 52.143 | 4 | 3/4 | 10.558 |
| W2→W3 | D2 | NO_SWITCH | 3 | 39.780 | 70.920 | 39.635 | 0 | 0/0 | null: no switch |
| W3→W1 | D0 | NO_ADAPT | 1 | 45.168 | 79.759 | N/A: no independent D0 confirmation | 0 | 0/0 | null: no switch |
| W3→W1 | D1 | RUNTIME_EVALUATION_BUDGET_REACHED | 6 | 59.988 | 96.460 | 54.492 | 4 | 3/4 | 11.337 |
| W3→W1 | D2 | NO_SWITCH | 3 | 45.168 | 79.759 | 42.503 | 0 | 0/0 | null: no switch |


Cold/warm 共 18/11 次 state evaluations，节省 **7 次 = 38.89%**；accepted switches 12/1；models loaded 9/1，unloaded 11/0。**32.843 s / 3.344 s** 必须标为 **isolated accepted-path reconfiguration measurement，includes cold seed reset**。不将 38.89% 描述为 wall-clock adaptation speedup，也不由结构 VRAM 大小估计秒数。

Independent final confirmations：cold/warm mean 53.544/42.115、52.143/39.635、54.492/42.503 ms；3/3 direction maintained。Confirmation 不重新选择部署、不修改接受路径。D0 没有额外 independent confirmation，新 Table B不伪造 D0 confirmation 行，共 9 search 行 + 6 confirmation 行 = **15 行**。Reconfiguration measurements 是冻结搜索后的独立实际 accepted-path activation，不是 search repetition latency。

Warm phase 1 从 paddleocr+qwen+resnet50+whisper ADD distilbert，到全部五模型；phase 2/3 都 NO_SWITCH，没有 unload。两阶段 D0与D2相同，表明保留当前良好部署已足够，不证明每个阶段都需要不同 specialist 集合。Warm 相对 bounded cold 的结果不是相对全局 cold optimum。Endpoint-only sensitivity：cold 3 switches / 1 load / 3 unloads，warm 1 / 1 / 0；alternative path的 measured time=null，不把它包装成另一组实测。

Accepted-path timing排除 profiling/warmup/confirmation/setup/teardown；activation发生于search freeze之后。Pilot 没有持续到达请求的零停机证明。Exact cache accesses 与 fresh physical GPU measurements 分开；5B新 search physical attempts=75，confirmation=12，共87。

数据：`rq4_dynamic_adaptation.json` / `table_dynamic_adaptation.csv`。Figure 4 同图三个 panel分别展示 evaluations、isolated path time、search/confirmation E2E；NO_SWITCH保留null，不伪造 measured zero。Figure 5展示离散phase的resident-model矩阵、accepted switch和NO_SWITCH，没有虚构连续时间轨迹。

## Module 2 与 Module 4 的最终定位

Module 2 corrected gates：**cross-deployment predictor=PARTIAL；bias-only calibration=FAIL；ranking evidence=INSUFFICIENT**。最终角色是 **optional coarse performance scorer**，main proposed=**predictor=None**。只作为 design exploration / motivation，解释选择 analytical guidance + bounded runtime feedback；accurate cross-deployment prediction不是核心贡献。旧和corrected文件同时索引，以corrected判定为准。

Module 4B=**NEUTRAL**：canonical=59.216 ms；10% OCR→Qwen=61.408 ms；10% image→Qwen=63.166 ms。默认 canonical min-max routing。只允许说 evaluated specialist-rich W1 deployment中的pilot没有发现优于canonical的方案。不能说 routing optimization一般没有必要，也不能声称canonical全局最优。

Module 3B的 **99.35%** 受seed第三次约38.4 s极端延迟影响，仅保留pilot observation，不进入abstract/contribution/main acceleration。优先使用3C和5B独立confirmation；本audit将两种来源明确分开。

## 最终六个问题

1. **可靠证据是什么？** 实测3C三种workload均改善minimum-memory feasible seed；LP-guided搜索在frozen objective下胜出4/6 random trials；LP relief与已测改善存在描述性正相关；3A验证5/10/20/30-model bounded non-enumeration；5B单序列warm少7次state evaluations、少accepted switches及isolated accepted-path时间，并有3/3独立confirmation方向保持。
2. **可以安全写什么？** 限定到tested cases的non-enumerative search、3/3 seed improvement、同预算4/6 random-trial wins、descriptive analytical guidance、单序列38.89% evaluation reduction、含cold reset的isolated reconfiguration measurement、NO_SWITCH reuse。准确用词见 `contribution_evidence_matrix.md`。
3. **当前不能写什么？** 独立runtime-feedback final-performance gain、准确cross-deployment prediction、routing optimization核心贡献、一般性routing无价值、全球最优或MILP polynomial guarantee、真实30-model GPU scalability、统计显著性、99.35%稳定加速、38.89% wall-clock加速、零停机适配。
4. **是否有必须补的实验？** 为现有有限事实或开始写稿，没有逻辑上必须的新实验；为作为final systems benchmark提交，具体缺口是full-five static baseline control、独立cold/warm完整序列重复，以及若保留一般LP-versus-random贡献所需的更多random seeds与独立B2 search campaign。不是默认再加所有state repetitions。CPU scalability wall-clock只对扩展planning-latency claim有必要。分类和最小规划见 `missing_experiments.md`。
5. **现在投稿最大的弱点？** 核心对照仍容易被“多放specialists即可改善弱memory seed”解释：缺少身份匹配的full-pool静态control，warm最终保持全五模型；此外只有一次完整dynamic sequence、每W两个random seeds、共享cache和较少独立search campaigns，外部有效性仍限于三W/五模型。现有confirmation增强final-state latency可信度，不能填补上述control与search-repeatability缺口。
6. **下一步补实验还是写论文？** 现在可开始限定claim的结构、方法和证据初稿；在final submission前优先补明确control与独立campaign重复。没有理由恢复算法扩展或强行提升Module2/4。所有实验只做规划，本阶段完成后停止。

## 产物与复现

- 主目录：[experiment_manifest.json](/root/autodl-tmp/generalist-specialist/results/final_evaluation_audit/experiment_manifest.json)；完整输入hash：[input_inventory_sha256.json](/root/autodl-tmp/generalist-specialist/results/final_evaluation_audit/input_inventory_sha256.json)。
- RQ1–RQ4 JSON、Table A/B/C CSV、five figure data CSV、five PNG与vector PDF均在本目录。
- Evidence matrix：[contribution_evidence_matrix.md](/root/autodl-tmp/generalist-specialist/results/final_evaluation_audit/contribution_evidence_matrix.md)；contribution outline：[provisional_contributions.md](/root/autodl-tmp/generalist-specialist/results/final_evaluation_audit/provisional_contributions.md)；gap plan：[missing_experiments.md](/root/autodl-tmp/generalist-specialist/results/final_evaluation_audit/missing_experiments.md)。
- `build_audit.py`只用既有CPU运行环境读档与审计；`plot_figures.py`使用现有base Python的matplotlib；`write_reports.py`只写规划文档。所有写入限定在此目录。没有安装包或新增预测器。
- 复现顺序：从repo root运行 `/root/miniconda3/envs/gsdeploy/bin/python results/final_evaluation_audit/build_audit.py`，然后 `python results/final_evaluation_audit/plot_figures.py` 与 `python results/final_evaluation_audit/write_reports.py`。源输入变化时完整hash核验会阻止“旧结果不变”的判定。

最终检查记录：`audit_validation.json`、`raw_inventory_coverage.json`、`figure_manifest.json`。本阶段只审计/汇总/作图/规划，完成后停止。
