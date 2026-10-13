# Module 8D-3 — B2 dominance and search diagnosis

Frozen algorithms/λ/quality/pool/revisions are unchanged. No new deployment search, ablation, dynamic adaptation or second hardware.

## Distance and actual generated-graph reachability

| Workload | seed→B2 L1 | final→B2 L1 | seed pressure-gated shortest | final pressure-gated shortest |
|---|---:|---:|---:|---:|
| demand_balanced | 3 | 3 | 5 | 5 |
| speech_heavy | 3 | 2 | 3 | 2 |
| vision_heavy | 3 | 3 | 5 | 5 |
| text_ocr_heavy | 3 | 3 | 5 | 5 |
| reasoning_vqa_heavy | 3 | 2 | 5 | 4 |
| no_vqa_phase | 1 | 0 | 1 | 0 |

All six B2 vectors are feasible and graph-reachable, but structural distance is not generated-neighborhood distance and neither is accepted runtime reachability. Four VQA regimes require removing/readding Whisper-small to expose ASR pressure and allow +Distil; shortest seed paths need5 steps, more than max3 iterations. Path witnesses and lattice-parity lower bounds are in reachability_paths.json.

## LP/runtime alignment

69 evaluated transition events retained;30 exact distinct transitions. Pooled event Spearman(Δz,Δmean)=0.584, Kendall=0.477, Spearman(Δz,ΔP95)=0.320. Deduplicated sensitivity: 0.470/0.362/0.326. Shared raw samples are not independent.
Balanced/vision/text each have only2 unique transitions, so independent per-workload correlation is INSUFFICIENT; repeated-event correlation1.0 cannot be treated as robust evidence. Generalist and Specialist scopes, pairwise ordering and Top1/Top2 hit rates with measurement coverage are reported in lp_runtime_alignment_summary.json.

Exact known misses below Top2: speech +Whisper-small replica (rank5) is a better-than-seed candidate, though Proposed evaluates a stronger Distil candidate; reasoning -Phi (rank4) is a decisive missed beneficial REMOVE. Do not call every unknown unmeasured candidate a missed winner or claim pervasive Top-B failure from two cases.

## Random path analysis

Balanced/vision/text have only2 feasible generated alternatives; all random seeds evaluate the same2 candidates as Proposed, and all retain the same seed. Speech alternatives differ but final objectives tie. Reasoning-2207 randomly selects -Phi after +Qwen and escapes the poor retained Phi+Qwen state. no-VQA1103/2207 omit +Distil and stop atseed, while47 matches Proposed. Fifteen objective ties are not fifteen independent replications; exact cache/state overlap and per-seed path sets are preserved.

## Routing/deployment crossover

New fixed-state timed runs=9. A=B2/greedy, B=B2/canonical, C=Proposed/canonical, D=Proposed/greedy-compatible. Every reused point passed all12 identity checks; every new state uses3 frozen repetitions. Missing cells remain missing.
| Workload | A mean | B mean | C mean | D mean |
|---|---:|---:|---:|---:|
| demand_balanced | 37.412 | MISSING | 44.702 | 44.702 |
| speech_heavy | 50.051 | 51.324 | 55.685 | MISSING |
| vision_heavy | 29.436 | MISSING | 32.325 | 32.325 |
| text_ocr_heavy | 19.934 | MISSING | 21.319 | 21.319 |
| reasoning_vqa_heavy | 85.983 | 86.267 | 95.491 | 88.855 |
| no_vqa_phase | 21.003 | 21.355 | 21.355 | 21.003 |

A−B measures routing at fixed B2 deployment; D−C measures routing at fixed Proposed deployment. B−C changes deployment and recomputes canonical routing, so it is not a pure worker-presence effect. Interactions/generalization remain unidentified whereD/B is missing. Full P95/std/source-path contrasts are in crossover_decomposition.json.

Reasoning fixed Proposed deployment: C→D mean 95.491→88.855ms; Phi remains deployed but VQA goes entirely to Qwen. B/D exact routing equality=True. This directly separates a routing contrast from the B/D deployment contrast; n=3 descriptive evidence and observed std do not establish an exact causal allocation.

Speech A/B/C mean=50.051/51.324/55.685ms, P95=73.973/86.693/82.475ms. B2 canonical lowers mean versus Proposed but raises P95; greedy routing lowers the B2 tail. D was not triggered by the frozen mean-sign gate, so the speech interaction remains unidentified.

## Greedy strength and resource constraint

Fastest admitted Specialist vs fastest Generalist s0 ratios: text11.84×, image25.39×, OCR26.43×, ASR4.47×, LID5.17×; VQA has no Specialist. All choices satisfy frozen quality admission. B2 combines five intrinsically faster paths and the fastest admitted VQA Generalist; it is a naturally strong latency heuristic, not a weak baseline.
B2 sampled peak headroom is 22730–41935MiB. Its first-five peaks are about26400MiB (~54% of49140MiB capacity); no-VQA peak7205MiB. All-pool memory exceeds GPU, but useful B2 subsets fit comfortably. No GPU budget reduced.

The frozen empirical MILP seed minimizes feasible memory first, then replica count. It can prefer Phi and omit extra Distil despite ample headroom; this is a seed objective choice, not evidence that the 49GB hard limit binds. The runtime search then optimizes a different E2E objective, through a pressure-gated neighborhood.

## Stopping and classification

All Proposed jobs terminate NO_RUNTIME_IMPROVEMENT, at3/6 or5/6 states, never by hard state cap/max_iterations. Speech/reasoning/noVQA have remaining unvisited feasible choices; balanced/vision/text do not under unchanged generation+tabu. Patience amplifies missed candidates, but budget expansion cannot by itself fix generator gating, rejected ADD intermediates or routing misalignment.

DOMINANT_CAUSE=MULTIPLE: pressure-gated neighborhood, pressure-objective ranking/routing mismatch and two-step acceptance barriers; stopping/budget amplify selected cases. Resource capacity does not exclude B2. no-VQA is NONE under the formal objective, with a small routing-only primary-metric difference.

Recommend method revision design and minimal separately frozen validation before original ablation; no revision or ablation implemented. SWAP, ranking and routing changes are recommendations only.
