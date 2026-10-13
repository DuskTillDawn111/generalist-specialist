# Module 4B real-runtime routing pilot

Run from `/root/autodl-tmp/generalist-specialist` with the existing deployment
environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m deployment_search.routing_real_pilot --prepare-only
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m deployment_search.routing_real_pilot
```

The pilot reads W1 directly from Module 3C's frozen `benchmark_workloads.json`
and the B2 final deployment directly from its `run_result.json`. It calls
`solve_canonical_routing` to initialize, then reuses Module 4A's
`GuidedRoutingSearch` and Module 3C's `CampaignRuntimeEvaluator`, which extends
`MeasurementRuntimeEvaluator` with audited execution-status handling.

The configuration is fixed at 10% task-rate transfers, Top-2, at most three
iterations, one no-improvement iteration, and six unique evaluated states
including canonical. Every state requires three completed repetitions. The
search physical safety cap is frozen at 18; it cannot increase on a rerun.
Only completed repetitions receive serving labels and enter aggregation;
genuine completed unstable results remain valid evidence. Interrupted attempts
remain in the physical ledger.

Exact canonical cache import authenticates routing, workload, deployment,
measurement settings/seeds, sources/models/inputs/interpreters, hardware identity
and audited aggregation semantics. Different GPU UUIDs or routing values require
a fresh measurement. The legacy Module 3 routing-source annotation is retained
for measurement compatibility; explicit routing values determine the state.

Only the canonical routing is a baseline. `predictor=None`. There is no
deployment mutation or routing enumeration. Enumeration tripwires surround the
search. The objective is the same Module 3C stability-first comparison with a
5% relative mean acceptance gate.

If the final routing changes, four fresh confirmation runs are interleaved
canonical/final/final/canonical using two seeds distinct from search. These
cannot modify search decisions. Their frozen physical safety cap is eight,
allowing bounded invalid-attempt replacement. Otherwise no extra confirmation
is launched. Search evaluation and repetition budgets are never expanded.

Results are in `results/routing_search_real_pilot/`. Check `module4b_summary.md`,
`decision_audit.json`, `post_run_verification.json`, full state identities,
attempt ledgers and raw request records. The independent post-run audit can be
run from the repository root without running models:

```bash
PYTHONPATH=. /root/miniconda3/envs/gsdeploy/bin/python results/routing_search_real_pilot/post_run_audit.py
```

The saved serialization repair audit records a correction to pilot evidence
writing after canonical measurement, before any candidate measurement. It did
not change routing, workload, runtime/cache identity, worker implementation,
objective or search configuration; the existing three canonical repetitions
were resumed from the exact cache.

This entry point ends at Module 4B; it has no Module 4C dispatch.
