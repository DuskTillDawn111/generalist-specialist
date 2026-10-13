"""Generate audit and planning documents from the audited data, not paper prose."""
from pathlib import Path
import json

O=Path(__file__).resolve().parent
ROOT=O.parents[1]
def read(n):return json.loads((O/n).read_text())
def write(n,s):(O/n).write_text(s.strip()+'\n')
def fmt(x):return f'{x:.3f}'
def pct(x):return f'{x:.2%}'
def active(d):return '+'.join(sorted(k for k,v in d.items() if v))
def mdtable(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def link(p):return f'[{Path(p).name}]({p})'

rq1,rq2,rq3,rq4=[read(f'rq{i}_{s}.json') for i,s in [(1,'scalability'),(2,'static_search'),(3,'guidance'),(4,'dynamic_adaptation')]]
gpu=read('gpu_experiment_scale.json');static=rq2['records'];dynamic=rq4['records']
cols=['Claim','Supporting experiment','Metric','Evidence strength','Known limitation','Safe wording','Unsafe wording']
matrix=[
 ['Non-enumerative deployment search','Module 1 + Module 3A 5/10/20/30-model synthetic',
  'Guided search enumeration=0; candidates=17/37/77/117; LP=11/26/56/86; evals=8/9/9/9',
  'Strong implementation/count evidence within tested cases; synthetic scale only',
  'No large real GPU pool; deterministic artificial objective; 5-model oracle gap=9.09%; MILP worst-case complexity not bounded',
  'The method searches bounded one-hop neighborhoods without exhaustive deployment enumeration in the evaluated cases.',
  'Polynomial-time global optimization; real 30-model serving scalability established; globally optimal search'],
 ['LP-guided search improves feasible seed','Module 3C; 3/3 workloads; independent seed/final confirmations',
  'B2 search means 59.373/57.174/52.597 ms; independent confirmation means 54.770/54.430/54.186 ms; confirmation reduction 79.04/79.82/76.84%',
  'Clear descriptive evidence versus specified seed; 3/3 direction confirmations',
  'Seed minimizes feasible memory, not latency; three workloads, five models, one static campaign; missing all-five static control; no global optimum',
  'Under the frozen benchmark, LP-guided bounded-feedback search improved the minimum-memory feasible seed on all three workloads, with the direction preserved in independent confirmations.',
  'Universally optimal deployment; guaranteed speedup; 99.35% main-benchmark acceleration'],
 ['LP guidance outperforms random search','Module 3C B1 seeds 1103/2207 vs B2',
  '4/6 real random trials under frozen noise-aware runtime objective; W1 2/2, W2 1/2, W3 1/2; 6 state evaluations each',
  'Limited comparative support',
  'Only two random seeds per workload; cached measurements shared; mean ranking alone differs from frozen objective; W2 same endpoint for one trial; W3 close',
  'At the same six-state evaluation budget, the proposed method beat four of six random trials under the frozen runtime objective.',
  'LP guidance always beats random; statistically significant superiority; six independent workload campaigns'],
 ['LP pressure relief is informative','Module 3C delta_z vs measured delta_J',
  '60 policy transitions; 31 unique deployment transitions; unique Spearman 0.644/0.780/0.621',
  'Descriptive association among measured candidates',
  'Selected Top-B population; cache reuse; dependent transitions; no intervention or fit',
  'LP relief was positively associated with measured latency improvement among the evaluated deployment transitions.',
  'LP relief causes improvement; accurate latency predictor; calibrated deployment-wide performance prediction'],
 ['Runtime feedback improves static search final performance','Module 3C B2 vs B3',
  'NOT established independently; B2/B3 final deployment identical in 3/3; real positive-LP/worse-runtime corrections=0',
  'Unestablished as an independent performance contribution',
  'Safety mechanism exists but the current static benchmark does not isolate a final-performance gain; rollback labels also include unchosen batch candidates',
  rq3['safe_feedback_wording'],
  'Runtime feedback independently improves final deployment; real feedback-ablation performance gain confirmed'],
 ['Warm adaptation reduces evaluation cost','Module 5B three transitions, D1/D2',
  'Cold/warm evaluations 18/11; 7 fewer = 38.89%; per transition 6/5, 6/3, 6/3',
  'Clear descriptive pilot evidence; confirmation does not change counters',
  'One full workload sequence; counts include exact-cache accesses; warm inherits a better starting deployment; final all-five retained',
  'In the evaluated sequence, warm adaptation used 38.89% fewer runtime state evaluations than the bounded cold restart baseline.',
  '38.89% wall-clock adaptation speedup; 38.89% fewer independent physical GPU runs; guaranteed cheaper adaptation'],
 ['Warm accepted-path reconfiguration cost is lower','Module 5B isolated accepted-path execution',
  'Accepted switches 12/1; loaded 9/1; unloaded 11/0; measured cumulative 32.843/3.344 s',
  'Measured isolated-path evidence for specified cold policy',
  'Includes cold seed reset; activation after search freezes; excludes profiling/warmup/confirmation/setup; endpoint-only cold alternative untimed',
  'Warm incurred less isolated accepted-path reconfiguration time in this pilot; the cold total includes reset to the MILP seed.',
  'End-to-end wall-clock gain; general cold implementation must reset; zero-downtime online adaptation proven'],
 ['Warm preserves final serving performance','Module 5B search and independent confirmations',
  'Warm mean lower in 3/3; independent direction 3/3; confirmation cold/warm 53.544/42.115, 52.143/39.635, 54.492/42.503 ms',
  'Descriptive one-sequence confirmation',
  'Limited cold budget; no global cold optimum; D0 equals warm in last two phases; no phase-specific turnover claim',
  'Warm retained or improved final latency relative to the evaluated bounded cold restarts, with the mean direction preserved in all three independent confirmations.',
  'Warm always achieves the cold optimum; every workload phase needs a different specialist combination'],
 ['Canonical routing is the default design choice','Module 4B specialist-rich W1 deployment',
  'Canonical 59.216 ms; 10% OCR→Qwen 61.408; 10% image→Qwen 63.166; 9 completed repetitions',
  'NEUTRAL routing pilot; supports local default choice',
  'One deployment/workload; two measured perturbations; not a routing optimum proof; no extra confirmation because final=canonical',
  'In the evaluated specialist-rich W1 deployment, the routing pilot did not identify an improvement over canonical routing.',
  'Routing optimization is unnecessary in general; routing optimization is a verified core contribution'],
 ['Module 2 motivates optional scorer design','Corrected DQA + Module 2D/2E/2F/2G',
  'Cross-deployment predictor PARTIAL; bias-only calibration FAIL; ranking INSUFFICIENT COMPARABLE GROUPS',
  'Audited design exploration; not positive predictor contribution',
  'Strict GBT passes one subset but full gate does not; sparse comparable groups cannot establish robust ranking',
  'Prediction remains an optional coarse scorer; the evaluated main method uses predictor=None and analytical guidance with bounded runtime feedback.',
  'Accurate cross-deployment performance prediction; validated ranking predictor core contribution']]
write('contribution_evidence_matrix.md','# Contribution evidence matrix — provisional, bounded by current evidence\n\n'+
      mdtable(cols,matrix)+'\nSearch counters and search objectives refer to frozen decisions. Post-search confirmation is separate corroboration and never a decision input. No entry asserts statistical significance or novelty relative to prior literature.\n')

write('provisional_contributions.md','''
# Provisional contributions — evidence-backed scope only

This is an experiment-backed contribution outline, not paper prose or a literature novelty assessment.

1. **Generalist–Specialist serving formulation and non-enumerative deployment search.** A minimum-memory feasible MILP seed and bounded one-hop deployment search support heterogeneous model capabilities, memory constraints and canonical min-max routing. Module 1 validates feasibility/objective correctness; Module 3A validates enumeration-free search at 5/10/20/30 synthetic models. The one-hop candidate bound is per iteration, with no polynomial-time MILP or global optimality claim. The five-model synthetic search has a 9.09% oracle gap.

2. **LP bottleneck-guided candidate generation before bounded runtime evaluation.** Cheap canonical LP pressure feedback ranks deployment mutations before Top-B runtime measurement. Module 3C improves its feasible seed on all three frozen workloads and wins against four of six real random trials under the frozen objective. The evaluated transition data support descriptive analytical guidance. Runtime feedback is an online correction/safety mechanism; because B2 and B3 have identical final deployments in all three static workloads, an independent final-performance gain from feedback is not established.

3. **Warm-start dynamic adaptation by reusing the current deployment.** Across the single W1→W2→W3→W1 pilot, warm adaptation uses 11 rather than 18 state evaluations, while accepted switches are 1 rather than 12 and independent confirmations retain the latency direction in all three transitions. Isolated accepted-path reconfiguration totals are 3.344 rather than 32.843 seconds, with cold seed reset included in the cold total. After adding DistilBERT once, the warm sequence retains all five models and produces two NO_SWITCH phases. These observations support reuse under a bounded search budget, not wall-clock speedup or continuous-serving guarantees.

Module 2 remains **optional coarse performance scorer / design exploration**: predictor PARTIAL, bias calibration FAIL, ranking evidence INSUFFICIENT. Proposed experiments use `predictor=None`. Accurate cross-deployment prediction is not a contribution.

Module 4 remains **design choice evidence**. Default routing is canonical min-max routing; the single specialist-rich W1 pilot is NEUTRAL. Routing optimization is not a verified core contribution.

The 3B 99.35% value is a pilot observation influenced by an extreme seed repetition (~38.4 s); it must not be used in the abstract, contribution list or main acceleration claim.
''')

write('missing_experiments.md','''
# Final benchmark planner — no experiments executed in Module 6A

Classification depends on the intended claim. There is no mandatory new experiment to preserve the scoped factual statements in the evidence matrix or to start a paper draft. There are concrete gaps before presenting a convincing final systems benchmark beyond a pilot. Existing data should be reused only when workload, hardware, model/source identity, routing and measurement protocol match; otherwise separate their evidence.

## MUST HAVE — for a final paper centered on the three proposed contributions

| Gap | Evidence that makes it necessary | Smallest useful resolution to plan | What it would establish |
| --- | --- | --- | --- |
| Static control with the feasible full five-model pool and canonical routing | 3C compares against a minimum-memory Qwen seed and bounded neighbor searches. It does not include a fixed full-pool control. The artifact audit found 16 historical Stage 2B full-pool attempts and 15 Module 5B full-pool attempts, with none in 3C or 2G. Stage 2B uses a different workload/protocol (a 20 s smoke example versus 3C's 210 s); 5B uses GPU UUID 011dad14 while 3C uses 3da0d2e4. These cannot silently replace a matched 3C control. | Reuse 5B full-pool/D0 comparisons as limited within-pilot evidence. Plan an identity-matched fixed full-pool control under the same frozen W1/W2/W3 and confirmation protocol if keeping the comparative static-search contribution. No workload or algorithm tuning. | Whether searching provides a useful latency/memory tradeoff relative to the obvious deploy-all-feasible baseline; otherwise limit claims to improvement over the specified memory seed. Do not claim LP search is globally superior. |
| Independent full cold/warm sequence repetitions | 5B has one end-to-end sequence. Three repetitions per state and two independent final confirmations estimate state performance, but do not independently repeat decisions, warm state inheritance or isolated accepted-path timing. | Plan independent complete sequences with fresh measurement seeds/cache namespaces and the same frozen policies/inputs. At least two additional sequences would reveal repeatability beyond the single pilot; this is not a power-derived sufficiency threshold. Report each sequence rather than treating phases as independent samples. | Whether the observed evaluation and switch reductions recur. Required if describing dynamic savings as reliable final-benchmark behavior beyond this single sequence. |
| Broader random seed coverage plus independently repeated proposed search | 3C has only two B1 seeds per workload, wins 4/6 under the frozen objective, and several strategies share exact cache evidence; close outcomes occur in W2/W3. | If retaining a general comparative LP-versus-random contribution, plan extra independent B1 seeds and repeated B2 decisions at the identical six-state budget; paired protocol blocks; retain all outcomes. An initial plan of three additional B1 seeds per workload expands coverage but does not justify significance on its own. | Distribution and sensitivity of LP/random performance; otherwise retain only the exact “4/6 trials” statement and treat it as limited evidence. |

These are justified control/repeatability gaps, not requests to start them now. No extra generic per-state repetitions are automatically required: 3C and 5B already have three search repetitions and two separate confirmation repetitions per final state. A fixed full-pool control should also report memory from the frozen catalog so that a latency-only comparison does not obscure the seed's different objective.

## NICE TO HAVE — only if the paper's scope needs them

| Item | Current coverage / reason | Scope |
| --- | --- | --- |
| CPU scalability wall-clock measurements | 3A contains 5/10/20/30-model candidate, LP and runtime-evaluation counts, but no full-search wall-clock series. Module 1's recorded MILP solve timings are not complete guided-search timings. | Plan CPU-only timing if claiming practical planning latency at larger pools. Counts already support non-enumeration; wall-clock is not mandatory for that narrower claim. |
| Additional workload intensities or tighter existing-model memory budgets | Current real core benchmark has three frozen rate vectors in one five-model pool; synthetic scaling is a single artificial family. | Broaden external validity only after the necessary controls and repetitions; do not tune workloads retrospectively. Not executed or authorized here. |
| A real LP/runtime disagreement case | 3C shows zero positive-LP/worse-runtime candidates; B2/B3 end states are identical. | Needed only to promote runtime feedback to an independent real-performance contribution. Otherwise leave it as a design safety mechanism. Do not hunt or construct a favorable benchmark after the fact. |
| Untimed cold endpoint alternative | 5B structural endpoint sensitivity is cold 3 switches / 1 load / 3 unloads vs warm 1 / 1 / 0, but alternative switch time is null. | Useful if discussing sensitivity to seed-reset policy; existing isolated-path timing is already valid when labeled precisely. |
| Service quality accounting from existing model task evaluations | Current claims concern serving latency and structural feasibility. | Reuse existing quality evidence if the paper discusses task accuracy; do not imply quality equivalence from capability coverage alone. |

## NOT NEEDED — for the current supported contribution scope

- More predictor training, predictor models or bias calibration: Module 2 is optional, and the proposed main method uses predictor=None.
- More routing optimization experiments: 4B already explains the evaluated default; no broad routing claim is made.
- Forecasting, change detection, prewarming, multi-replica or new algorithm modules.
- Reproducing 3B's 99.35% extreme-value claim.
- New GPU runs to validate the offline enumeration tripwire or synthetic count tables.
- Pooling confirmations into search statistics, treating cache hits as physical runs, or selecting a favorable subset to improve conclusions.

## Recommended next step

Start the paper's structure and methods/evidence draft now with the safe wording. Before a final submission asserting general comparative and dynamic performance, resolve the specific controls and independent-campaign gaps above in a separately authorized experiment phase. Prioritize the full-pool static control and independent cold/warm sequences; no additional algorithm development is indicated by this audit. Module 6A stops here.
''')

searchrows=[]
for r in static:
    searchrows.append([r['workload'].split('_')[0],r['algorithm'],active(r['final_deployment']),
        fmt(r['search_final']['mean']),fmt(r['confirmation_final']['mean']),r['runtime_evaluations'],r['lp_solves'],r['accepted_mutations'],r['rollbacks']])
b2rows=[]
for r in static:
    if r['algorithm']=='B2':
        b2rows.append([r['workload'].split('_')[0],fmt(r['search_seed']['mean']),fmt(r['search_final']['mean']),
            pct(r['search_improvement_fraction']),fmt(r['confirmation_seed']['mean']),fmt(r['confirmation_final']['mean']),pct(r['confirmation_improvement_fraction'])])
dynrows=[]
for r in dynamic:
    rc=r['reconfiguration'];c=r['confirmation_final']
    dynrows.append([r['transition'].replace('transition_','').replace('_','→').upper(),r['method'],r['action'],
        r['search_cost']['valid_runtime_evaluations'],fmt(r['search_final']['mean']),fmt(r['search_final']['weighted_p95']),
        fmt(c['mean']) if c else 'N/A: no independent D0 confirmation',
        rc['accepted_switches'],f'{rc["models_loaded"]}/{rc["models_unloaded"]}',
        fmt(rc['measured_switch_time_ms']/1000) if rc['measured_switch_time_ms'] is not None else 'null: no switch'])
scalerows=[[r['num_models'],r['num_candidates_generated'],r['max_candidates_in_iteration'],r['lp_solve_count'],r['runtime_evaluation_count'],r['enumeration_call_count']] for r in rq1['runs']]
phaserows=[[p,m['physical_attempts'],m['execution_status'].get('completed',0),m['execution_status'].get('external_interruption',0),m['completed_stability'].get('unstable',0)] for p,m in gpu['by_phase'].items()]
manifest=read('experiment_manifest.json')
indexrows=[[e['experiment'],e['module'],e['GPU_runs']['new_physical_attempts'],e['GPU_runs']['referenced_completed_runs'],e['decision']] for e in manifest['experiments']]

write('final_evaluation_audit.md',f'''
# Module 6A — Final Evaluation Audit & Paper Experiment Planner

审计日期：2026-10-01（用户时区 Asia/Shanghai）。阶段状态：**审计完成，停止；未启动后续实验**。

只读取既有实验并生成数据、matplotlib 图和规划文档。新增 GPU experiment=0；新增 synthetic algorithm run=0；算法修改=0；workload/预算/搜索决策修改=0。原始结果文件的完整 SHA-256 快照覆盖 {len(read('input_inventory_sha256.json'))} 个文件；构建结束后逐项核验没有变化。模型权重使用历史 hash manifest 的记录值，本阶段没有重新加载或重测权重。

## 统一实验索引与执行状态口径

`experiment_manifest.json` 覆盖 {manifest['total_experiment_directories']} 个现有 results/ 目录，包括所要求的全部 16 个阶段、corrected 分析层及 legacy stages。每项记录 module、purpose、workload、algorithm、deployment、routing、run 数、GPU/synthetic 口径、metric source、decision、source hashes 和 report path。`new_physical_attempts` 与 `referenced_completed_runs` 分开；同一个物理 run 被多个分析引用不能再计数。`input_inventory_sha256.json` 是所有旧结果的完整文件索引；manifest 也保留历史 source/model hash manifests。

{mdtable(['Experiment','Module','Owned physical attempts','Referenced completed runs','Decision'],indexrows)}

Data Quality Audit 是执行状态语义的唯一来源。旧 `results/data_quality_audit/audit_counts.json` 仅覆盖 130 次早期尝试，并不是当前总量。本阶段读取其唯一物理 run ledger，并按后续 CampaignRuntime 已采用的同一 DQA v1 规则，重核 3C/4B/5B 的完整 window、drain、harness footer、raw hashes 和 completed stability，生成 `execution_status_ledger.jsonl` 补充层；没有覆盖旧 audit。`gpu_experiment_scale.json` 提供统一全项目 **results/ timed-serving** 口径。

{mdtable(['Phase','Physical attempts','Completed','External interruption','Genuine unstable'],phaserows)}

- Total physical timed-serving attempts：**{gpu['total_physical_attempts']}**。
- Valid completed runs：**{gpu['valid_completed_runs']}**；valid latency runs：**{gpu['valid_latency_runs']}**。
- External interruptions：**{gpu['external_interruptions']}**；genuine completed unstable：**{gpu['genuine_unstable_runs']}**；recoverable：**{gpu['completed_stability_counts']['recoverable_congested']}**；steady：**{gpu['completed_stability_counts']['steady']}**。
- 实际 workload 到达率向量：**{gpu['total_workloads']}**；论文主 benchmark 命名 workload：**3**。
- Unique deployment states：**{gpu['total_unique_deployment_states']}**；unique routing fraction states：**{gpu['total_unique_routing_states']}**；absolute routing-rps states：**{gpu['total_unique_routing_rps_states']}**。
- 后续新 ledger 记录 search physical attempts：**{gpu['search_confirmation_new_physical_attempts']['search-time measurements']}**；post-search confirmation attempts：**{gpu['search_confirmation_new_physical_attempts']['post-search confirmation']}**。
- 另有 **13** 次 isolated accepted-path reconfiguration operations。它们与 setup/teardown 是不同操作，不能混入 timed-serving repetition 总数；候选 profiling 的重复也不能冒充 switch 次数。

去重按 active model counts、positive arrival vectors、routing fractions 计算，数值保留 12 位小数，忽略零项。GPU run 总量仅覆盖 results/ 下可追溯的 timed-serving 尝试，不声称穷尽 benchmark/ 等目录中历史 standalone/NVML/Nsight 的全部 GPU 活动。`raw_inventory_coverage.json` 单独核验所有原始 harness artifacts 是否进入 ledger，以及重复副本；无外部原因证据的缺失结果只能标 uncertain，不能猜测为外部中断。Completed unstable 保留有效观察，不因 raw failed 就重标中断。

## RQ1 — Scalability / Non-enumeration

Can the method search Generalist–Specialist deployments without exhaustive deployment enumeration?

{mdtable(['Models','Total generated candidates','Max candidates / round','LP solves','Synthetic runtime evaluations','Search enumeration calls'],scalerows)}

支持结论：tested search 没有 exhaustive deployment enumeration，单轮候选至多 M，runtime evaluations 受预算限制。5-model 的独立 oracle 枚举 32 个 binary states，其中 31 个可行 objective evaluations；J*=20.833333 ms、J_search=22.727273 ms，gap=9.090909%。Oracle 不在正式 guided search 内调用。10/20/30-model 没有 exhaustive gap。所有这些 latency 是人工 synthetic objective，不能描述为真实 GPU scalability 或全局最优；O(M) 是单轮邻域候选上界，不是 MILP 时间复杂度。缺少 guided search 总 wall-clock series；现有 Module 1 MILP solve timings不能替代。

数据：`rq1_scalability.json` / `table_scalability.csv`。

## RQ2 — Static Deployment Search

Does LP-guided bounded-feedback search improve serving performance over the MILP feasible seed and simple search baselines?

比较 B0 MILP feasible seed、两个 B1 random-neighbor + runtime-feedback trial、B2 LP-guided + runtime-feedback、B3 LP-only。Main proposed predictor=None，routing=canonical min-max。三种 W 的首个 seed 都是 qwen。B0 优化最小可行内存，因此是明确且较弱的 latency baseline；不把它称为 runtime-optimal seed。

{mdtable(['Workload','Method','Final deployment','Search mean ms','Confirmation mean ms','State evals','LP solves','Accepted','Rollbacks'],searchrows)}

B2 相对**自身 seed**的两类证据分别为：

{mdtable(['Workload','Search seed ms','Search final ms','Search reduction','Confirmation seed ms','Confirmation final ms','Confirmation reduction'],b2rows)}

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

{mdtable(['Transition','Method','Action','State evals','Search mean ms','Search weighted P95 ms','Confirmation mean ms','Accepted switches','Loaded/unloaded','Isolated path time s'],dynrows)}

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

- 主目录：{link(O/'experiment_manifest.json')}；完整输入hash：{link(O/'input_inventory_sha256.json')}。
- RQ1–RQ4 JSON、Table A/B/C CSV、five figure data CSV、five PNG与vector PDF均在本目录。
- Evidence matrix：{link(O/'contribution_evidence_matrix.md')}；contribution outline：{link(O/'provisional_contributions.md')}；gap plan：{link(O/'missing_experiments.md')}。
- `build_audit.py`只用既有CPU运行环境读档与审计；`plot_figures.py`使用现有base Python的matplotlib；`write_reports.py`只写规划文档。所有写入限定在此目录。没有安装包或新增预测器。
- 复现顺序：从repo root运行 `/root/miniconda3/envs/gsdeploy/bin/python results/final_evaluation_audit/build_audit.py`，然后 `python results/final_evaluation_audit/plot_figures.py` 与 `python results/final_evaluation_audit/write_reports.py`。源输入变化时完整hash核验会阻止“旧结果不变”的判定。

最终检查记录：`audit_validation.json`、`raw_inventory_coverage.json`、`figure_manifest.json`。本阶段只审计/汇总/作图/规划，完成后停止。
''')
print('Generated evidence matrix, provisional contribution outline, experiment gap plan and final audit report.')
