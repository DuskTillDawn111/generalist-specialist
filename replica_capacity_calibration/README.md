# Module 7C-2 CPU calibration

Read frozen Module 7C-1 full-precision high-load paired gains and measured idle
memory; verify evidence hashes, preserve the candidate pool and old results.
Create empirical discrete capacity/memory policies, evidence bounds, discrete
MILP validation, exact binary compatibility and W1/W2/W3 one-hop dry-runs.

```bash
OMP_NUM_THREADS=1 envs/data_env/bin/python -m replica_capacity_calibration.foundation
envs/data_env/bin/python -m unittest replica_capacity_calibration.tests -v
```

The full command runs CPU-only regressions, including existing synthetic
scalability tests. It does not import or run GPU campaigns, start model workers,
or execute real deployment search. Results are saved only under
`results/replica_capacity_calibration/`; prior frozen results stay untouched.

The primary linear/empirical comparison holds evidence bounds and discrete
memory fixed. A separate original linear/additive-memory baseline is included.
Additional fixed-state one-hop diagnostics expose first/second Specialist
choices even if current pressure-task guidance correctly prunes zero-relief
mutations. These are diagnostic CPU LPs, not search transitions.

See `deployment_core/replica_calibration.py` for measured-source loading and
policy validation, `deployment_core/empirical_milp.py` for one-hot formulation,
and `deployment_search/integer_search.py` for evidence-bounded planning.
