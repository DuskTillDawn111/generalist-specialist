# Module 7A-2 standalone demand completion

This module registers frozen v2 launchers with the unchanged historical
`run_active_single_profile.main`. It profiles only prepared verified capabilities
with missing Stage-1 values. Every path has three fresh standalone runs, one
first-target warmup and the same 30 historical task-pool rows per run. The
quantity is handler `worker_latency_ms`, with the original sample-count-weighted
mean in milliseconds. Queue/E2E latency, smoke-test timing and synthetic values
cannot substitute for it.

The original catalog, v2 capability/VRAM/environment/model/adapter files and
Module 7B source stay protected by hashes. `updated_service_demand_catalog.json`
adds real missing values without replacing original values. Raw evidence,
standalone isolation and unload gates, per-repetition means/std and aggregate
sample std are retained. Failed inference remains pending; capability problems
request revalidation without changing the frozen matrix.

Commands (do not rerun completed profiling to unify old/new data):

```bash
envs/data_env/bin/python -m candidate_pool_service_demand.prepare
envs/data_env/bin/python -m candidate_pool_service_demand.campaign
envs/data_env/bin/python -m candidate_pool_service_demand.finalize
envs/data_env/bin/python -m candidate_pool_service_demand.dry_run
envs/data_env/bin/python -m candidate_pool_service_demand.cpu_regression
envs/data_env/bin/python -m candidate_pool_service_demand.report
```

Preparation refuses to overwrite before hashes; campaign resumes saved attempts
without remeasuring. Finalization never replaces a completed frozen catalog with
different values. The completed catalog loader checks frozen presence, task
capabilities, measured idle VRAM, positive real service demands, original values
and raw artifact hashes. W1/W2/W3 planning opens only at 100% completeness.

No Module 7B algorithm is changed. The restored runs use
`IntegerGuidedDeploymentSearch(..., SearchConfig(dry_run=True))`, with zero runtime
evaluations. Synthetic A–E and integer scalability campaigns are not rerun. The
CPU regression command explicitly retains four unchanged prior campaign/scalability test-wrapper
results while running the remaining existing CPU tests and new demand gates.

All measurements use exactly one worker, never a real deployment count >1.
Replica capacity and additive idle-memory assumptions remain planning surrogates;
real scaling and co-residency await Module 7C.
