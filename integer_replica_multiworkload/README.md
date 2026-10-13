# Module 7C-4

This module reuses the frozen 7C-3 replica adapter, evaluator and objective.
Only W1/W3 receive new GPU runs; W2 is an authenticated immutable reference.
There are no additional smoke/scaling runs, new workloads or optimizer changes.

```bash
OMP_NUM_THREADS=1 envs/data_env/bin/python -m unittest integer_replica_multiworkload.tests -v
OMP_NUM_THREADS=1 envs/data_env/bin/python -m integer_replica_multiworkload.campaign --prepare-only
OMP_NUM_THREADS=1 envs/data_env/bin/python -m integer_replica_multiworkload.campaign
```

Each new workload: empirical capacity/memory, unchanged 3C rates and timings,
independent MILP seed, predictor=None, Top-B=2, max_iterations=3, <=6 unique
search states, three repetitions/state, patience=1 and the unchanged 5% gate.
There are two independent seed and final confirmations each, which cannot
alter decisions. At most 18 search + 4 confirmation attempts per workload;
no automatic retries or budget increase. Source/evidence bounds are checked
before GPU use. Completed unstable runs are retained; external/model/resource
failures stop the campaign without invented objective values or substitution.

The experiment freeze authenticates old inputs, W2 records and all serving
code. Results preserve full integer vectors, rejected/pruned/rollback states,
per-replica metrics and raw GPU telemetry. CPU linear counterfactuals only
compare planning at actual incumbents with matching visited history; no linear
GPU campaign or exhaustive oracle is executed. The report computes SUPPORT /
STRONG SUPPORT / PARTIAL from recorded evidence, without retuning workloads.

Outputs: `results/integer_replica_search_multiworkload/`. Stop after 7C-4.
