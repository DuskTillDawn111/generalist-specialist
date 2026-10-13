# Module 20A CPU-only utility optimizer

Scientific choices are immutable in `results/module20_utility_optimizer/utility_definition.json`
and `utility_reference_manifest.json`. No GPU runtime/worker is launched.

Initial freeze is `python -m module20.freeze`; it intentionally refuses to replace
an existing freeze. To reproduce analysis against the existing freeze, from the
repository root run each command in sequence:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 envs/data_env/bin/python -m module20.history
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 envs/data_env/bin/python -m module20.search
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 envs/data_env/bin/python -m module20.test_optimizer
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 envs/data_env/bin/python -m module20.finalize
```

The finalizer reads the persisted `cpu_tests.log`; capture the test invocation to
that file when reproducing. Timing is a single CPU observation. Finite candidate
coverage is heuristic; uncertainty scenarios are not fitted confidence bounds.
Every search cell has its full input, generated deployment/routing alternatives,
and infeasible/unresolved MILP attempts under `results/module20_utility_optimizer/cells/`.

Module 19 is development data. The grouped archival Module 16 holdout has no
matched alternative deployment per workload and cannot validate ranking. The
Module 17 scoped calibration is recorded only and never applied to new conditions.
