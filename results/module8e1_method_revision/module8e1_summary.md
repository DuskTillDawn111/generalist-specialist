# Module 8E-1 — Minimal method revision and offline validation

OFFLINE_VALIDATION_PASS=true. Six workloads are DEVELOPMENT WORKLOADS; no unbiased final-validation claim. Zero new GPU experiments.

v1 remains runnable via method_v1.create()/method_v1.config(), using unchanged original packages, with 76 archived source files, source hashes and 24 unchanged Proposed/Random traces. v2 lives independently in method_v2; no original file was overwritten.

## Frozen revision

Only neighborhood and Top-B selection change. Targeted swaps use admitted capability overlap, prospective NEW-model ADD replacement closure and same-task Specialist replacement. Resulting coverage/memory/bounds and original canonical LP capacity are checked. Integer counts change exactly -1/+1. Closure means a previously nonredundant model becomes replaceable by equal-or-faster frozen-s0 paths after ADD; already-redundant removals and replica-ups are not mislabeled closures.

Class P receives slot1 by delta_z. Class R receives slot2, prioritizing exact closure, service-dominated family removal, lower/equal memory, then largest delta_z (including negatives), followed by deterministic ties. Feasible candidates only; no runtime-label/model-name preferences. Seeder, empirical evidence, canonical formulation, RuntimeEvaluator and NoiseAwareRuntimeObjective are unchanged.

top_b=2; max_iterations=3; max_unique_runtime_states=6 including seed; patience=1; predictor=None. Search repetitions=3 and independent seed/final confirmations=2 remain frozen; none newly executed.

## Reachability and reasoning

| Workload | v1 shortest | v2 shortest | reduction |
|---|---:|---:|---:|
| demand_balanced | 5 | 3 | 2 |
| speech_heavy | 3 | 2 | 1 |
| vision_heavy | 5 | 3 | 2 |
| text_ocr_heavy | 5 | 3 | 2 |
| reasoning_vqa_heavy | 5 | 3 | 2 |
| no_vqa_phase | 1 | 1 | 0 |

All six B2 vectors are graph-reachable within three v2 mutations, using bounded goal-directed local path search. This is not a claim those intermediates will pass runtime acceptance or actually be selected in three iterations. Path witnesses are saved.

NEIGHBORHOOD_FIX_CONFIRMED_OFFLINE=True: reasoning seed Phi+S4 generates Qwen+S4 in the same round, and places it in Top2. Near old Phi+Qwen+S4 final, -Phi also enters Top2. Cached reasoning replay selects Qwen+S4 directly: mean86.406ms versus old final95.491ms. This is development reuse, not new or unbiased validation.

## Candidate complexity

| Workload/state | base | swaps | total |
|---|---:|---:|---:|
| demand_balanced/seed | 7 | 10 | 17 |
| demand_balanced/v1_final | 7 | 10 | 17 |
| speech_heavy/seed | 10 | 10 | 20 |
| speech_heavy/v1_final | 10 | 14 | 24 |
| vision_heavy/seed | 7 | 10 | 17 |
| vision_heavy/v1_final | 7 | 10 | 17 |
| text_ocr_heavy/seed | 7 | 10 | 17 |
| text_ocr_heavy/v1_final | 7 | 10 | 17 |
| reasoning_vqa_heavy/seed | 7 | 10 | 17 |
| reasoning_vqa_heavy/v1_final | 8 | 7 | 15 |
| no_vqa_phase/seed | 10 | 8 | 18 |
| no_vqa_phase/v1_final | 10 | 13 | 23 |

Formal pool M=12 (ordered swap upper bound132); quality-admitted effective catalog M=11 (upper bound110). Actual snapshots have7–14 targeted swaps,15–24 total generated candidates. Base candidates remain unchanged; no exhaustive deployment-vector enumeration.

| Synthetic M | density | base | swaps | total | LP screen ms |
|---|---|---:|---:|---:|---:|
| 12 | one_deployed | 12 | 11 | 23 | 35.9 |
| 12 | third_deployed | 12 | 32 | 44 | 77.4 |
| 20 | one_deployed | 20 | 19 | 39 | 63.5 |
| 20 | third_deployed | 20 | 84 | 104 | 187.2 |
| 30 | one_deployed | 30 | 29 | 59 | 102.4 |
| 30 | third_deployed | 30 | 200 | 230 | 445.1 |

Dense overlap can grow quadratically; no strict O(M) claim. The largest tested30-model case has230 candidates and under0.5s LP screening on this host, versus2^30 possible binary vectors; only generated local candidates are screened. Runtime slots remain2. Legacy bounded test oracles are regression fixtures, not v2 production enumeration.

## Historical replay and budget

| Workload | hits | misses | last resolved mean ms | status |
|---|---:|---:|---:|---|
| demand_balanced | 2 | 1 | 44.702 | CACHE_MISS_UNRESOLVED_ITERATION |
| speech_heavy | 2 | 1 | 79.678 | CACHE_MISS_UNRESOLVED_ITERATION |
| vision_heavy | 2 | 1 | 32.325 | CACHE_MISS_UNRESOLVED_ITERATION |
| text_ocr_heavy | 2 | 1 | 21.319 | CACHE_MISS_UNRESOLVED_ITERATION |
| reasoning_vqa_heavy | 4 | 1 | 86.406 | CACHE_MISS_UNRESOLVED_ITERATION |
| no_vqa_phase | 2 | 1 | 22.931 | CACHE_MISS_UNRESOLVED_ITERATION |

CACHE_MISS occurrences=6, exact unique identities=6; no predicted/synthetic runtime fills. All six replays are incomplete, so the table lists last resolved incumbents, never v2 final-deployment claims. Other selected slots in an unresolved iteration are probed within the original budget, but no winner is committed while one selected state remains unknown.

One new-to-v1 beneficial cached state is reached: reasoning Qwen+S4. Speech and no-VQA also select their already-known beneficial Distil ADD, but an unknown second-slot replacement prevents committing that iteration. This is limited evidence of better access, not six-workload dominance. All replay logical attempts are<=6.

Conditional shortest-path slot audit: speech and no-VQA witness paths fit the unchanged selection/budget if each intermediate is accepted. The other four shortest witnesses lose their second SWAP at Top2. This does not rule out other paths, but compression alone has not established full budgeted B2 reachability. Three iterations using both slots would require seven states; the last iteration has only one slot at budget6.

## Regression and gate

257 tests pass: all27 old modules plus10 new v2 contracts. The two existing sklearn modules use the existing gsdeploy environment; no packages installed or predictor retrained. Initial wrong-environment import errors are preserved, then resolved by correct existing test environment. AST audit confirms unchanged seeder/routing/feasibility and identical runtime-evaluation/accept/rollback/stopping suffix. V1 hashes/traces unchanged.

ROUTING_REVISION_PENDING. Routing need not change before a separately frozen held-out test of these two revisions; it remains a known limitation and may prevent latency dominance.

MODULE8E2_ALLOWED=true for held-out protocol preparation and then validation after preregistration. HELDOUT_GPU_CAMPAIGN_READY=false until the fresh workload matrix/protocol is frozen. No held-out campaign is launched here.
