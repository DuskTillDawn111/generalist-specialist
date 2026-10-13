# Module 4A: Non-enumerative Routing Search Foundation

Scope: fixed deployment, canonical LP start, one-step workload transfers, pressure/delta_z ranking, Top-2 pruning, shared Module 3 RuntimeEvaluator and RuntimeObjective. No GPU measurement was run.

## Synthetic validation

| Case | Initial mean (ms) | Final mean (ms) | Accepted | Evaluated | Accepted Δz |
| --- | ---: | ---: | ---: | ---: | --- |
| case1_specialist | 113.333 | 80.000 | 4 | 6 | -0.03000, -0.03000, -0.03000, -0.01000 |
| case2_runtime_correction | 94.222 | 80.222 | 3 | 5 | -0.06000, -0.06000, -0.06000 |
| case3_four_capable_models | 103.709 | 94.123 | 1 | 6 | -0.01500 |

Case 2 deliberately has a runtime optimum away from standalone pressure balance. At the initial state, the analytically ranked first move has Δz = -0.03 and worsens synthetic mean latency; the ranked second move has Δz = -0.06 and improves latency, so runtime feedback rejects the first and accepts the second. This is an offline correction example, not a real serving claim.

## Small grid oracle (baseline only)

| Case | Grid oracle (ms) | Search (ms) | Relative gap |
| --- | ---: | ---: | ---: |
| case1_specialist | 80.000 | 80.000 | 0.00% |
| case2_runtime_correction | 80.000 | 80.222 | 0.28% |

The 10% grid exists only in `small_grid_oracle`; `GuidedRoutingSearch.search` never calls it.

## Scaling

| Models | One-step transfers | Ordered-pair bound | Top-B | Runtime calls |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 90 | 90 | 2 | 0 |
| 20 | 380 | 380 | 2 | 0 |
| 30 | 870 | 870 | 2 | 0 |

## Current pool dry-run

| Workload | Tasks with >=2 deployed capable models | Transfers | Top-B |
| --- | --- | ---: | --- |
| W1_balanced | image, ocr, asr, lid | 4 | image:resnet50→qwen (Δz=-0.00558), ocr:paddleocr→qwen (Δz=-0.00379) |
| W2_generalist_heavy | text, image, asr, lid | 4 | text:distilbert→qwen (Δz=-0.00469), image:resnet50→qwen (Δz=-0.00684) |
| W3_specialist_friendly | image, ocr, asr, lid | 4 | ocr:paddleocr→qwen (Δz=-0.00379), lid:whisper→qwen (Δz=-0.00336) |

Each workload has 6 active tasks: 4 have two deployed capable models and 2 have only one. The canonical routing assigns all 4 flexible tasks to one model initially, yielding 4 one-step transfers. Every initial real-pool Δz is negative. Because canonical routing globally minimizes max pressure under the Stage-1 constraints, a strictly positive Δz is impossible at the canonical start apart from numerical noise. Pressure ranking therefore limits surrogate damage; only runtime feedback can justify accepting a pressure-worsening transfer. No runtime objective was measured for any real workload, so no real routing improvement claim is made.

## Interfaces and limits

`GuidedRoutingSearch` accepts any existing `RuntimeEvaluator` instance, including an adapter with a shared measurement cache and Data Quality Audit status semantics. The search adds no serving adapter or cache of its own. Runtime accept/rollback uses the Module 3 `RuntimeObjective` and fixed deployment memory. The optional predictor can be `None`.

Termination supports max iterations, no improvement patience, runtime evaluation budget, and no unvisited candidate. The routing state hash uses normalized 9-digit shares to avoid floating-point cycling. No real routing search or joint search was run.
