# Module 7B integer replica planning

`IntegerGuidedDeploymentSearch` accepts deployment states as `dict[str, int]`.
Counts must be nonnegative integers; omitted families have count zero. Runtime
evaluation remains behind `RuntimeEvaluator`; the integer foundation uses CPU
synthetic function/table evaluation only. Existing real serving adapters retain
their single-instance guard. No multi-replica workers are implemented here.

For a measured per-model idle memory `V_m`, the default bound is
`R_m = floor(V_GPU / V_m)`. An explicitly supplied real worker/process limit can
reduce it. `replica_bounds` is an optional smaller experimental bound, recorded
as such; it cannot exceed the physical/process bound. Frozen v2 bounds use
`measured_idle_vram_mib`, never weight file sizes. The legacy v1 catalog retains
its historical memory estimates solely for reproducibility.

The integer MILP uses integer counts `x_m`, binary presence `y_m`, and aggregate
family routing `r[k,m]`:

```text
0 <= x_m <= R_m; y_m <= x_m <= R_m*y_m
sum_m V_m*x_m <= V_GPU
sum_m c[k,m]*y_m >= 1                         (lambda_k > 0)
sum_m r[k,m] = lambda_k
0 <= r[k,m] <= lambda_k*c[k,m]*y_m
D_m = sum_k r[k,m]*s0[k,m] <= rho_max*x_m
```

The feasibility seed minimizes memory first, then minimizes replica count while
fixing the primary memory optimum. Both MILP calls and the presence/count
witness are serialized. This is a minimum-memory feasible seed, with no runtime
performance optimum claim.

Memory is a **planning memory surrogate**: additive per-replica idle VRAM.
Capacity is an analytical planning surrogate, **not a claim of linear real-GPU
scaling**. Module 7C must validate memory co-residency and real capacity scaling.

For fixed counts, canonical routing minimizes `z` with `D_m <= z*x_m`. Bottleneck
feedback uses `D_m/x_m` for active families, preserving total task contributions
`D[k,m]`. Inactive families have zero placeholders in serialized feedback; their
per-replica pressure is not defined. Conservation and routing upper bounds refer
to the whole family stream, so adding replicas never multiplies the arrival rate.

Each round generates at most `2*M` direct `x_m +/- 1` mutations. Scale-up includes
the bottleneck family and alternative models capable of its pressure tasks;
scale-down is considered for every positive count. Bounds, memory, coverage and
canonical LP capacity screen candidates before `delta_z = z_current-z_candidate`
ranking and Top-B runtime selection. The existing runtime objective, budgets,
tabu states, rollback and memory tie-break remain in use. State keys include
counts. There is no exhaustive replica-vector generation in production search.

```python
from deployment_core.replicas import replica_key
from deployment_search import IntegerGuidedDeploymentSearch, SearchConfig
from deployment_search.runtime import SyntheticRuntimeEvaluator

# catalog must contain frozen or explicitly synthetic Stage-1 demand values.
result = IntegerGuidedDeploymentSearch(evaluator).search(
    workload, catalog, hardware, SearchConfig(),
    roles={m: model_role[m] for m in catalog},
)
# Table evaluation uses replica_key({"qwen": 1, "whisper": 2}),
# with SyntheticRuntimeEvaluator(table, integer_mode=True).
```

With roles supplied, results include active Generalist/Specialist type counts,
`N_G`, `N_S`, per-round candidate counts and future IDs such as `qwen#0`,
`qwen#1`. These IDs reserve a schema only; routing remains family-level.
An integer warm start must already pass coverage/capacity/memory; infeasible
starts receive no runtime evaluation. Omit the warm start to obtain a MILP seed.
Integer dynamic repair and per-replica routing are outside Module 7B.

Legacy `GuidedDeploymentSearch`, canonical routing calls without replica bounds,
and seeder calls without integer options keep binary behavior. Use
`MilpDeploymentSeeder.solve(..., binary_mode=False)` for physical integer bounds,
or supply explicit `replica_bounds`. With every `R_m=1`, integer search reuses
the legacy seed, ADD/REMOVE candidates and exact search path. This compatibility
branch retains the original single-objective seed and original result schema;
the integer two-stage objective applies outside that branch. Five pre-change
trace comparisons verify this reduction, excluding only solver wall time.

Frozen v2 capability verification does not imply Stage-1 demand availability.
The loader preserves missing demands as `None`, and refuses to create a complete
planning profile from them. At the original Module 7B freeze, all three workloads had 14 missing
task/model demands across seven types and their full 12-model dry-run reports
were blocked. Module 7A-2 later completed all paths using real standalone
measurements and restored these dry-runs. Historical blocked evidence is retained. No load-test latency or different-protocol E2E timing substitutes for s0.
All three workloads require VQA; its sole capable model Qwen has mandatory
presence due to coverage, restricting Generalist composition freedom.

CPU validation (no GPU benchmark):

```bash
envs/data_env/bin/python -m unittest deployment_search.tests.test_integer_replicas -v
envs/data_env/bin/python -m deployment_search.integer_foundation
```

Artifacts are in `results/integer_replica_search_foundation/`. Synthetic A–D
cover mandatory replicas, Generalist/Specialist choice, scale-down and memory
competition. 12/20/30-type runs enforce enumeration tripwires and a fixed runtime
budget. The separate validation oracle permits only <=3 types and R<=2 (27
vectors); search never imports or calls it. Its measured gap is a heuristic
diagnostic for one artificial objective, not an optimality guarantee.

## Module 7C-2 empirical mode

For a completed real candidate-pool v2 catalog, `IntegerGuidedDeploymentSearch`
now defaults to `capacity_mode="empirical"`, `memory_mode="empirical"`. Old
binary/synthetic catalogs retain the linear default. Specify linear explicitly
for a baseline. Seeder and standalone canonical-routing APIs retain their
legacy default; pass the desired mode explicitly when using those APIs.

```python
from candidate_pool_service_demand.dry_run import load_completed_catalog
from deployment_core.replica_calibration import load_replica_calibration
catalog, hardware, roles, records, data = load_completed_catalog()
calibration = load_replica_calibration(catalog)
result = IntegerGuidedDeploymentSearch().search(
    workload, catalog, hardware, SearchConfig(dry_run=True),
    roles=roles, calibration=calibration, capacity_mode="empirical",
)
```

Capacity is `D_m<=rho_max*g_m(x_m)`. Canonical routing minimizes effective
normalized pressure `z` subject to `D_m<=z*g_m(x_m)`; ranking still uses
`delta_z=z_current-z_candidate`. Family demand and task contributions remain
aggregate, while `D/x` is only an additional diagnostic in empirical reports.
The capacity mode flows consistently through seeding, feasibility, bottleneck
feedback, candidate LPs and ranking. The runtime objective is unchanged.

The discrete MILP uses O(sum R_m) per-count one-hot selection variables, never
an enumeration of replica vectors. Memory uses the measured discrete `M(x)`
table when selected; scale-down savings and runtime memory tie-breaks use the
same policy. Linear capacity with empirical memory provides a controlled
comparison that changes only capacity. Frozen v2 single idle memory anchors
M(1), preserving exact binary behavior; M(2) uses frozen 7C-1 double idle mean.
Observed 7C-1 single memory and peak memory are retained separately.

The three measured types have evidence bound 2; nine unmeasured types have
bound 1. Physical memory bounds remain separately recorded, and actual process
limits or smaller explicit bounds can further restrict planning. Missing g(n)
or M(n) fails validation; no synthetic fallback or count extrapolation occurs.

7C-1 factors for Qwen/Whisper are ASR-specific. Applying them to family demand
for different tasks/mixes is a transfer surrogate awaiting validation. M(x)
is an idle-memory planning surrogate, not a heterogeneous joint peak guarantee.
This stage does not wire replica IDs into the generic real RuntimeEvaluator:
`DeploymentRuntime.start` still rejects family counts above one. The separate
7C-1 scoped worker launcher is evidence, not a completed generic real search
adapter. Future small-budget GPU validation must first connect count states to
independent replica workers and freeze its validation plan.

CPU-only calibration and required W1/W2/W3 reports:

```bash
OMP_NUM_THREADS=1 envs/data_env/bin/python -m replica_capacity_calibration.foundation
envs/data_env/bin/python -m unittest replica_capacity_calibration.tests -v
```

Outputs: `results/replica_capacity_calibration/`. A validation-only tiny
27-vector oracle checks discrete MILP feasibility and the memory/count optimum;
production search does not import it. Supplemental fixed-state reports inspect
all bounded one-hop neighbors, including zero-relief Specialist second replicas
pruned by production pressure-task guidance. They never transition states or
call runtime evaluation.
