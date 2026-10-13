# Module 5A dynamic adaptation foundation

`DynamicState` represents `(deployment, canonical_routing, workload_rps)`. An
external caller provides an explicit `WORKLOAD_CHANGE`, old/new workloads, and a
phase ID. This module does not detect or forecast workload changes.

The existing search entry point now accepts an optional warm start:

```python
# Original Module 3 behavior; one MILP seed, unchanged cold result schema.
cold = GuidedDeploymentSearch(evaluator, predictor=None).search(
    new_workload, catalog, hardware, cold_config,
    initial_deployment=None,
)

# Inspect/re-route the running deployment for the new workload first.
warm = GuidedDeploymentSearch(evaluator, predictor=None).search(
    new_workload, catalog, hardware, warm_search_config,
    initial_deployment=current_deployment,
)
```

Use the dynamic wrapper to retain phase history and switching accounting:

```python
adapter = DynamicAdaptation(runtime_evaluator=evaluator, predictor=None)
result = adapter.adapt(
    current_deployment, old_workload, new_workload, catalog, hardware,
    AdaptationConfig(top_b=2, max_adaptation_iterations=3,
                     max_runtime_evaluations=4),
    phase_id="phase_2",
)
```

Each new workload phase needs a distinct explicit ID. Within a phase, history
retains evaluated/rejected states, preventing an immediate return cycle across
calls. A new phase can revisit an earlier deployment. The wrapper verifies that
the supplied running deployment agrees with `history.last_deployment`.

Warm initialization validates binary counts and catalog membership, checks
memory, then calls the existing canonical LP. Invalid counts/unknown models or
a deployment exceeding hardware memory are rejected. Missing capability or
capacity infeasibility enters ADD-only repair instead of discarding the state.

Repair generates at most one ADD for each absent model supporting an unserved
or overloaded task. It ranks restoration of feasibility first. Partial repair
reduces unserved arrival rate, then relaxed max pressure. The relaxed LP has no
pressure cap and is explicitly diagnostic; it is never a valid runtime routing.
No REMOVE or runtime evaluation occurs before feasibility is restored.

Repair consumes the same adaptation iteration budget as normal search. Only
when bounded local repair cannot finish does the search invoke one MILP fallback,
recording `warm_start_fallback=true`. Fallback does not reset or enlarge either
budget. Its initialization can change multiple models and is recorded separately
from one-hop local mutations. A globally infeasible workload returns an
`INFEASIBLE` result without evaluating latency.

Once feasible, warm search reuses Module 3 bottleneck feedback, ADD/REMOVE
generation, delta-z ranking, Top-B, runtime comparison, rollback, and tabu logic.
The running deployment is the baseline under the **new** workload. An accepted
switch is optional; unchanged deployment returns `NO_SWITCH`. Scorers remain
optional and `predictor=None` works throughout.

`ReconfigurationCostModel` provides the cost interface. The default
`StructuralReconfigurationCost` records load/unload counts and VRAM moved using
the catalog memory profiles. It never estimates time:
`measured_switch_time_ms=None`. Costs accumulate along accepted transitions,
including repair/fallback, rather than just the net old/final difference.
Synthetic candidate evaluations are not reported as physical switching.

Cold/warm comparisons charge cold initialization from the same running
deployment to its MILP seed. `deployment_mutations` counts local mutations;
`committed_transitions` additionally counts cold initialization. Loads/unloads
and VRAM movement include both. The offline report includes cold budget 6,
warm budget 4, and a separate cold budget 4 comparison using the same synthetic
objective and planning settings. It does not assume warm is better.

Planning with `dry_run=True` returns `PLAN_ONLY`, leaves history and the running
deployment unchanged, and does not evaluate runtime. In repair mode it produces
one local repair plan without applying it or falling back to MILP. The real
W1/W2/W3 plans use only new-workload LP/feedback/one-hop planning. A single
conditional next neighborhood illustrates what could follow acceptance of the
first selected ADD. Historical deployments and recorded cold seeds are attached
after planning, solely to describe reference distances.

Run the offline validation in the existing CPU controller environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m deployment_search.dynamic_foundation
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m unittest discover -s deployment_search/tests
```

Outputs are in `results/dynamic_adaptation_foundation/`. Synthetic feedback is
explicitly artificial, small exhaustive oracles are limited to synthetic pools
of at most five models, and production search is protected by enumeration
tripwires. Real planning is protected against both runtime measurements and
MILP restarts. No model worker is launched by the validation entry point.

This foundation does not measure dynamic performance or switching latency and
does not dispatch a real dynamic experiment, forecasting, prewarming, replicas,
joint routing search or a paper benchmark.
