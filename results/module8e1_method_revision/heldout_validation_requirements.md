# Held-out validation requirements

The six Module 8C/8D workloads are DEVELOPMENT WORKLOADS. They informed this revision; neither replay nor later GPU runs on those same workloads support an unbiased final-validation claim. They remain frozen and unchanged.

Before a Module 8E-2 timed run, independently define a fresh held-out demand/load matrix and deterministic seeds under a preregistered sampling/design principle. Freeze its lambda vectors and hashes BEFORE inspecting its planning/runtime results. Do not select workloads to obtain different deployments or favorable outcomes. Do not fill development cache misses and call that held-out evidence.

Freeze method_v2 source hashes, candidate rules, Class R order, canonical routing, predictor=None, quality admission, model revisions, all 27 service-demand entries, empirical capacity/memory and replica bounds. Keep top_b=2, max_iterations=3, unique runtime states<=6 including seed, patience=1, three search repetitions and two independent seed/final confirmations. Freeze task sources, preprocessing, arrival generation, warmup/window, precision/backend and hardware identities. Confirmations must not affect search decisions.

Preregister analysis, runtime objective, failures/retries, cache reuse identity checks and any requested comparisons before measurements. Report every held-out condition, unsuccessful/unstable state, cache miss and resource failure; no best-of workload/seed selection. Existing baseline definitions must be frozen if comparison is later authorized; none is run here.

Keep canonical routing unchanged for this version's attribution. ROUTING_REVISION_PENDING remains a limitation, not permission to tune routing on held-out outcomes. A routing/ranking revision after inspecting held-out results needs a new method version and fresh held-out validation. Only this module's structural correctness is established; latency dominance/generalization is not.

Module 8E-1 stops after offline validation. No new GPU search, baseline, ablation, dynamic adaptation or second hardware has been started.
