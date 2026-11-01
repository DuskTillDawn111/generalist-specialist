# H600 selected numerical-v2 search review

This CPU review covers the fixed four-route shared-executor search. It does not consume the independent target's outcomes, execute a production search, launch GPU work, or certify physical gain.

## Source comparison

The diff from [flow_search_check.py](../tools/predictor_validation_20261027/flow_search_check.py) to [h600_search_core_v2.py](../tools/predictor_validation_20261027/h600_search_core_v2.py) contains only a docstring clarification and replacement of the `FullHorizonModel` import with the explicit numerical-v2 simulator. Candidate enumeration, quality proxy, six-term scoring, model-memory/drain constraints, bounds, UNKNOWN propagation and all-tie handling are unchanged.

The diff from [h600_search_check.py](../tools/predictor_validation_20261027/h600_search_check.py) to [h600_selected_search_v2.py](../tools/predictor_validation_20261027/h600_selected_search_v2.py) selects the new validation/core/source modules, binds model auditing to the estimator selection, and labels the selected estimator and numerical version. The existing deep-copy resource overlay changes only startup joules and prephase sampled owned-memory peak. It does not alter the public plan, candidate identities, routes, catalog or quality objective.

Both searches run fresh forecasts over the same public candidate domain. Strict `upper_bound < incumbent - 1e-12` pruning and all ties within `1e-12` remain unchanged. Missing model evidence remains UNKNOWN; it is not treated as an infeasible or zero-cost candidate.

The imported [original_rho](../tools/predictor_validation_20261027/flow_eligibility.py) still calls the original service resolver and divides original planned demand by the original catalog capacity factor `g(n)`. Its threshold remains `0.8 + 1e-8`. The separately reported finite forecast service/window/resident proxy uses its existing diagnostic tolerance and cannot grant original admission. Search ranks model-eligible candidates; the summary explicitly does not certify an optimum constrained by the original rho gate.

## CPU verification

[test_h600_search_core_v2.py](../tests/test_h600_search_core_v2.py) adds five tests using real statistical-provider and resource-provider fixtures. No objective/evaluate results are mocked. The tests check prediction-entrypoint versus search equality for six-term metrics, complete-arrival FIFO waiting, fresh plain/bounded rollouts and all ties, the actual long-clock two-stage completion, UNKNOWN for joint batch/weight reuse without an executor, and preservation of unseen inputs in the full UNKNOWN domain. Only the engine constructor is wrapped for counting fresh calls; it executes the real engine.

Command, with GPU devices hidden and CPU numerical threads limited to one:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m unittest tests.test_h600_search_core_v2 tests.test_h600_selected_search_v2 tests.test_h600_event_simulator_v2
```

Result: **15 tests passed**. The long-clock fixture completes at rounded `571791700004.1077 ns` in fewer than five transitions with the original `100000` event budget; prediction and search yield identical metrics. These are CPU regression fixtures, not measured GPU evidence.

## Scope restriction for resource controls

The selected wrapper's audited public plan is restricted to the four shared routes. The inherited general simulator validates a structurally legal MPS/static-SM/green-context configuration but only applies admission semantics when mode is `request_admission`; it does not itself establish mode-specific speed/power/memory evidence or fail closed merely because another legal mode is supplied. Therefore a direct generic-core call with an unqualified resource mode must not be presented as a qualified resource-control prediction. This review leaves the frozen simulator/core unchanged. Any future expansion beyond the audited shared domain needs explicit mode applicability checks and matching measurements before search execution.

Joint batch and merged-weight execution already fail through the real UNKNOWN path and are covered above. No MPS, exclusive SM, cache-reuse, physical stability, end-to-end gain, or search speedup claim follows from this review. The selected wrapper records physical eligibility as null and physical/original-gate optimality and final-gain eligibility as false.
