# Module 3A deployment search

Module 7B adds [integer replica planning/search](INTEGER_REPLICAS.md), including
CPU synthetic validation and the explicitly blocked full v2 workload dry-runs.
The binary Module 3/5 APIs and behavior documented below remain compatible.

`GuidedDeploymentSearch.search(workload, model_catalog, hardware, config)` starts
with `MilpDeploymentSeeder.solve`. An infeasible seed returns `NO_FEASIBLE_SEED`.
All planning feedback comes from a fresh canonical min-max routing LP, using
standalone service times and preserving `rho_max`. The seed's arbitrary routing
witness is retained for audit only.

```python
from deployment_core.models import load_model_catalog, load_hardware
from deployment_core.workload import Workload
from deployment_search import GuidedDeploymentSearch, SearchConfig

result = GuidedDeploymentSearch(predictor=None).search(
    Workload({"text": .8, "image": .4, "ocr": .4, "asr": .15,
              "visual_qa": .2, "lid": .4}),
    load_model_catalog(), load_hardware(), SearchConfig(dry_run=True),
)
result.save("results/deployment_search_foundation/manual_dry_run.json")
```

Dry run stops after one seed neighborhood and Top-B selection. It does not
accept planning-only improvements as runtime improvements. Normal search
requires an explicitly supplied `RuntimeEvaluator`. This module contains no GPU
implementation, worker startup or profiling. A future real adapter must reuse
`joint_deploy/measurement/`, including stability_state_v2 labeling.

`SyntheticRuntimeEvaluator` accepts a function `(deployment, workload, routing)`
or a table keyed by sorted tuples of active model names. Values are
`RuntimeEvaluation` objects. Routing maps `(task, model)` to requests/second;
latencies are milliseconds. `weighted_task_latency` implements arrival-rate
weighting of task means or task P95s. Weighted task P95 is not pooled system P95.
Failed/nonsteady runs can retain missing latency metrics; steady runs must have
finite nonnegative mean and P95. Evaluator/solver infrastructure exceptions
propagate; they are not silently relabeled as capacity infeasibility or runtime
instability.

Search defaults: Top-B=2, max iterations=8, patience=2, total runtime budget=17
(including seed), rho_max=.8, REMOVE pressure threshold=.2, mean tolerance=1 ms.
ADD is limited to models serving pressure tasks. REMOVE requires low absolute
standalone pressure and remaining coverage; the candidate LP still checks
capacity. `delta_z` is the primary ranking key, even when it is zero or negative.
No LP-improvement or prediction threshold rejects feasible candidates.
`relief_per_memory` and memory freed are diagnostics.

An optional `PerformanceCoarseScorer.predict` returns predicted latency or None.
Only candidates with exactly equal delta_z and available predictions reorder by
prediction. Missing slots retain deterministic model order. Predictor exceptions
or invalid values are recorded as unavailable. Module 2's predictor remains
optional given PARTIAL regression, FAIL local calibration, and INSUFFICIENT
COMPARABLE GROUPS ranking evidence.

Runtime comparison orders steady > recoverable_congested > unstable. Equal
nonsteady states do not improve. Two steady states compare mean first; absolute
differences strictly below tolerance (or exactly equal means) compare weighted
P95, then memory. Within Top-B, every acceptable candidate must beat the actual
iteration-start incumbent. At most one is accepted after evaluations; all other
measured candidates are retained as ROLLBACK. Tolerance comparisons can be
nontransitive, so the fixed LP order deterministically resolves selection among
several improving candidates; this is a heuristic, not an optimum guarantee.

`visited_deployments` includes the seed and all runtime-evaluated states, including
rollbacks. Those states are not revisited during a search. Rejected directed
mutations are additionally recorded. LP-only candidates are cached but may be
considered in later rounds after earlier Top-B states become tabu. The cache is
local to one call, with a fixed workload/catalog/rho_max. Both accepted and cached
candidate routing are canonical LP solutions. Tied LP optima need not have a
unique routing witness.

Results include the MILP seed, all per-round candidates, canonical routings and
feedback, ranking, selected/pruned states, predictions, measurements, decisions,
and termination status. Tuple-keyed routing and demand maps serialize as
`task::model`. Budget conventions:

- `lp_solve_count`: actual canonical LP calls, including seed re-routing and
  infeasible candidate solves; excludes cache hits and MILP internal relaxations.
- `milp_solve_count`: always one per search.
- `num_candidates_lp_evaluated`: candidate LP calls only.
- `runtime_evaluation_count`: seed plus runtime candidates.
- `num_runtime_candidates`: excludes seed; may be below Top-B on the final budget.
- `visited_deployment_count`: seed plus distinct runtime-evaluated states; dry
  run visits only the seed. `search_iterations` counts generated neighborhoods.

Each neighborhood has at most M candidates. Across T rounds the framework makes
at most 1+TM canonical LP calls and 1+T*Top-B runtime evaluations, further bounded
by the explicit runtime budget. This describes candidate counts, not the internal
worst-case complexity of the MILP solver. No exhaustive helper is imported by
the search implementation.

CPU-only validation:

```bash
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m unittest deployment_search.tests.test_search -v
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m deployment_search.validate
```

The validator produces the six requested artifacts plus a test log under
`results/deployment_search_foundation/`. Enumeration tripwires cover repository
helpers, `all_combinations`, and itertools. Only
`tests/small_space_baseline.py` deliberately enumerates, with a hard five-model
limit; the validator calls it separately after guarded search finishes. The
synthetic oracle gap measures one artificial objective and cannot establish
global optimality or real serving quality. Real-pool dry runs use the three
frozen Module 2G workloads and existing profiles, with zero GPU evaluations.

## Module 4A fixed-deployment routing search

`GuidedRoutingSearch.search(deployment, workload, catalog, config)` starts from
`solve_canonical_routing` and keeps `deployment` fixed. Each mutation moves at
most `routing_step_fraction * lambda_k` requests/second between one ordered
pair of deployed capable models for one active task. It checks conservation,
nonnegativity, capability, deployment membership and Stage-1 capacity before
ranking feasible candidates by pressure reduction `delta_z`. The optional
predictor only resolves ties after pressure preference. Top-B limits runtime
evaluations; `RuntimeObjective` handles accept/rollback and visited routing
states prevent cycles. Pass the same `RuntimeEvaluator` instance used by
deployment search to share its measurement cache and audit semantics.

```python
from deployment_search import GuidedRoutingSearch, RoutingSearchConfig

result = GuidedRoutingSearch(predictor=None).search(
    deployment, workload, catalog, RoutingSearchConfig(dry_run=True)
)
```

The offline validation command produces the Module 4A artifacts and never
starts serving measurements:

```bash
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m unittest deployment_search.tests.test_routing_search -v
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m deployment_search.routing_foundation
```

The coarse 10% grid oracle is confined to `routing_foundation.small_grid_oracle`
for two-model synthetic cases and is not imported by the search implementation.
