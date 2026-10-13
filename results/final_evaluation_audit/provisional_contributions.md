# Provisional contributions — evidence-backed scope only

This is an experiment-backed contribution outline, not paper prose or a literature novelty assessment.

1. **Generalist–Specialist serving formulation and non-enumerative deployment search.** A minimum-memory feasible MILP seed and bounded one-hop deployment search support heterogeneous model capabilities, memory constraints and canonical min-max routing. Module 1 validates feasibility/objective correctness; Module 3A validates enumeration-free search at 5/10/20/30 synthetic models. The one-hop candidate bound is per iteration, with no polynomial-time MILP or global optimality claim. The five-model synthetic search has a 9.09% oracle gap.

2. **LP bottleneck-guided candidate generation before bounded runtime evaluation.** Cheap canonical LP pressure feedback ranks deployment mutations before Top-B runtime measurement. Module 3C improves its feasible seed on all three frozen workloads and wins against four of six real random trials under the frozen objective. The evaluated transition data support descriptive analytical guidance. Runtime feedback is an online correction/safety mechanism; because B2 and B3 have identical final deployments in all three static workloads, an independent final-performance gain from feedback is not established.

3. **Warm-start dynamic adaptation by reusing the current deployment.** Across the single W1→W2→W3→W1 pilot, warm adaptation uses 11 rather than 18 state evaluations, while accepted switches are 1 rather than 12 and independent confirmations retain the latency direction in all three transitions. Isolated accepted-path reconfiguration totals are 3.344 rather than 32.843 seconds, with cold seed reset included in the cold total. After adding DistilBERT once, the warm sequence retains all five models and produces two NO_SWITCH phases. These observations support reuse under a bounded search budget, not wall-clock speedup or continuous-serving guarantees.

Module 2 remains **optional coarse performance scorer / design exploration**: predictor PARTIAL, bias calibration FAIL, ranking evidence INSUFFICIENT. Proposed experiments use `predictor=None`. Accurate cross-deployment prediction is not a contribution.

Module 4 remains **design choice evidence**. Default routing is canonical min-max routing; the single specialist-rich W1 pilot is NEUTRAL. Routing optimization is not a verified core contribution.

The 3B 99.35% value is a pilot observation influenced by an extreme seed repetition (~38.4 s); it must not be used in the abstract, contribution list or main acceleration claim.
