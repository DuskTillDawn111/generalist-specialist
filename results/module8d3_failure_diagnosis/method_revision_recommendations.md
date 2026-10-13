# Method revision recommendations — design only

1. Reconcile min-max pressure routing/ranking with the measured E2E objective. Δz is useful for capacity relief but can favor slower paths at low utilization and penalize beneficial removals. Standalone latency is an LP input, but minimizing maximum pressure is not minimizing weighted mean E2E/P95. The empirical seed separately minimizes memory, then replica count; a cheaper feasible seed need not have lower latency. Report objective-specific gaps; do not train a predictor or patch ranking in this module.

2. Audit candidate admission beyond only bottleneck tasks: quality-admitted ASR savings can remain invisible when VQA sets z. Consider a bounded additional specialist-offload candidate set as a future design, independently frozen before testing.

3. Consider SWAP for coverage-preserving Generalist replacement. For effective M=11, at most M(M-1)=110 ordered cross-family down/up pairs before filtering, versus at most2M=22 one-hop mutations; if restricted to one deployed Generalist and two absent Generalists, only2 alternatives. In reasoning there are2 deployed Generalists and1 absent one, at most2 Generalist swaps; a same-family pair is a no-op and should be excluded. Measure complexity after feasibility filtering.

SWAP addresses rejected ADD-before-REMOVE intermediates. It does not by itself fix positive-only pressure preference, missing non-bottleneck +Distil, or poor canonical routing; do not claim it solves B2 dominance. No SWAP implemented.

4. Minimal future budget sensitivity: freeze 6 versus7 unique states and patience1 versus2 with same max_iterations3 in reasoning. More patience alone under budget6 cannot reach rank2 -Phi in the next iteration. Balanced/vision/text have no remaining generated feasible unvisited candidate, so increasing budget alone is not a supported global cure. No sensitivity run executed.

5. Establish revised method validity before continuing the original ablation campaign. B2 remains the strongest latency baseline and Random found a better replacement state. Preserve current method/results as frozen evidence; any revision needs a new version and independently frozen protocol. No new predictor, quality threshold, workload, model, routing optimizer or deployment search implemented.
