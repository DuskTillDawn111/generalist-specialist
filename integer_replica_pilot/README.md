# Module 7C-3

A replica-aware adapter expands family counts into independent `family#i`
workers using the existing frozen handlers and process transport. Family routing
preserves Module 3C's deterministic weighted round-robin. A second deterministic
round-robin distributes each family stream across its replicas. Routing stays
aggregate in planning; there is no new optimizer.

```bash
OMP_NUM_THREADS=1 envs/data_env/bin/python -m unittest integer_replica_pilot.tests -v
OMP_NUM_THREADS=1 envs/data_env/bin/python -m integer_replica_pilot.campaign --prepare-only
OMP_NUM_THREADS=1 envs/data_env/bin/python -m integer_replica_pilot.campaign
```

The authorized campaign uses frozen W2 rates, 10 s warmup / 210 s timed window,
empirical capacity/memory, three repetitions/state, Top-B=2, predictor=None,
max_iterations=3, max_unique_states=6, patience=1 and the unchanged Module 3
noise-aware objective (5% mean gate). Same-nonsteady states retain the incumbent.
Budget: at most three minimal correctness smoke attempts, 18 physical search
attempts and four independent confirmation attempts. No retry or budget growth.
Confirmation never changes decisions, including when seed and final are equal.

Prelaunch gates use frozen evidence bounds and the discrete memory table. Actual
NVML idle/peak and independent process identities are recorded; OOM or budget
violation is preserved as RUNTIME_RESOURCE_INFEASIBLE and stops the pilot without
silent substitution. Full integer vectors, including zeros, enter exact cache
identity. Completed genuine unstable measurements remain valid; uncompleted
external/model/harness failures have no invented serving metrics.

Outputs: results/integer_replica_search_real_pilot/. Older result directories,
planning algorithms, native model handlers, weights and environments are kept
unchanged. This is one W2 pilot, not a full integer benchmark or final evaluation.
