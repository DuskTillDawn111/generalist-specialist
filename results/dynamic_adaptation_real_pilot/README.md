# Module 5B real dynamic transition pilot

Frozen sequence: `W1_balanced → W2_generalist_heavy → W3_specialist_friendly → W1_balanced`.
`dynamic_sequence.json` records the unchanged Module 3C rates, source path and
SHA-256; `dynamic_sequence.sha256` hashes the saved sequence itself. Module 3C W1
B2 supplies phase 0 only. Each subsequent actual deployment is exclusively the
previous D2 final deployment. D0 and D1 are counterfactuals of that same actual
old state. All routing is freshly solved canonical min-max LP routing.

Run/resume with the existing environment:

```bash
cd /root/autodl-tmp/generalist-specialist
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/miniconda3/envs/gsdeploy/bin/python -u \
  -m deployment_search.dynamic_real_pilot
```

The pilot has a process lock, durable measurement attempts, state access logs,
search checkpoints, frozen search results and completed reconfiguration events.
Resume preserves completed measurements and searches. An incomplete live switch
has a durable `active_switch.json`; only fully applied READY-verified events have
a measured duration. Setup reinstates the last completed deployment. An external
failure during an in-flight switch requires review of that record before making
any online availability claim.

## Runtime and search accounting

D1 uses `initial_deployment=None`, Module 3 MILP seeding and guided search. D2
uses Module 5A inspection, ADD repair and bounded MILP fallback with the actual
old deployment. Both have Top-B 2, at most 3 total iterations, patience 1, at
most 6 valid unique state evaluations including the starting/seed state, and
3 completed repetitions per state. Predictor=None. The unmodified Module 3
`NoiseAwareRuntimeObjective(1 ms, 5%)` prioritizes steady, then recoverable
congested, then unstable; latency and memory comparisons remain unchanged.

`valid_runtime_evaluations` counts algorithm state accesses with a valid
three-repetition aggregate, including strict exact cache hits. Physical search
attempts count newly started harness processes, including interrupted attempts.
Each cache access happens only after that strategy selects the state. Historical
imports look up the exact identity key, validate the aggregate and individual
raw/config hashes, and do not disclose unvisited runtime values to search. The
identity includes deployment, unchanged workload rates, exact routing floats,
measurement settings and repetition seeds, measurement source hashes, model
content hashes, sample input hash, interpreter/package identity, GPU UUID/name/
driver/memory, environment and stability/aggregation policy. Algorithm source
hashes are frozen separately: identical measurements may be reused by different
algorithms.

Search LP solves follow Module 3's established call accounting, including warm
feasibility/repair calls. The extra canonical-routing validation LP performed
before each cache access is separately counted. The phase 0 baseline and each
confirmation final also solve canonical routing outside the search budget.

Completed measurements with incomplete requests, persistent backlog or drain
failure retain genuine serving-instability labels under Data Quality Audit.
Absent completed measurement evidence has no serving label. Proven host loss,
scheduler kill or infrastructure failure is external interruption; an unknown
cause remains uncertain. Both are physical attempts, excluded from performance
aggregates. No external cause is inferred solely from a missing raw file.

## Fixed switch-time definition

Candidate profiling runs initialize their own workers and shut them down. Their
startup, measurement, repetitions, drain and teardown are **search cost**, never
system switch latency. Once a search result is frozen, a separate real-worker
session applies only its committed accepted path. Existing Worker transport and
models are reused. No new model/runtime implementation is introduced.

For each accepted `x_a → x_b` event:

- `switch_start_timestamp`: recorded immediately before the first accepted
  deployment change operation, after the old session is already READY.
- `models_to_unload` / `models_to_load`: exact set differences of accepted
  deployments. Removed workers receive shutdown and are confirmed exited before
  loading the additions; unchanged worker processes and PIDs are retained.
- `workers_ready_timestamp`: one observed worker READY record per final model,
  including wall-clock and monotonic timestamps, capabilities and process PID.
  Retained workers keep their original READY timestamps.
- `switch_ready_timestamp`: recorded after all required final workers have
  emitted READY, their capabilities have been checked, removed processes have
  exited, and every retained/final worker process is alive.
- `measured_switch_time_ms`: `(switch_ready.monotonic_ns -
  switch_start.monotonic_ns) / 1e6`. Wall-clock timestamps are also preserved.
  This includes shutdown/unload plus initialization/load operations. It excludes
  benchmark warmup, profiling and confirmation. READY means model initialization
  is complete and its request loop can accept a formal request; it does not
  guarantee warmed CUDA kernels or a specified first-request latency.

No accepted mutation means no measured switch: duration is null, not an inferred
zero measurement. Aggregate sums over actually measured events are separately
reported. Missing/failed readiness evidence leaves the active event's duration
null; no estimates replace it. Forced worker exits, if any, are recorded.

The initial session setup and final teardown are deployment-isolation overhead,
recorded separately (`residency_setups.jsonl`, worker logs and final residency
verification) and excluded from accepted-path costs. They must not be confused
with the physical total of every GPU model launch in the campaign.

For a fair starting-state comparison, D1's committed path starts from the same
actual old deployment as D2: it explicitly resets to the MILP seed and then
applies each accepted mutation. `cold_seed_resets` and `local_search_switches`
are separate fields. Since a different execution policy could perform only one
direct old→cold-final activation, `endpoint_structural_sensitivity` additionally
reports that alternative's load/unload/VRAM set differences. This endpoint path
is not executed, so its switch time remains null. The primary switch saving
uses measured accepted-path switch counts, not probe startup counts.

This architecture isolates search probes, cold counterfactuals and actual warm
activation to avoid co-resident profiling interference. Logical current state
propagates through the actual READY-verified D2 finals. Workers remain resident
within each accepted path, while isolation closes them before other GPU probes.
It is an explicit-phase real-feedback pilot, not a continuous-traffic, zero-
downtime experiment. Reliable online wall-clock adaptation latency is unavailable
and is null; orchestration elapsed time is not substituted for it.

## Structural costs and metrics

`models_added`, `models_removed`, `VRAM_added_mib`, `VRAM_removed_mib`, and
`VRAM_moved_mib` use the frozen catalog's model peak VRAM values. They are
structural accounting, not measured PCIe transfer bytes or actual instantaneous
VRAM allocation. `N_load` and `N_unload` are counts on committed paths.

`Gap_warm_cold = (mean_warm - mean_cold)/mean_cold`. Stability is reported
alongside this mean-based gap; a lower mean does not override worse stability.
`Saving_eval = 1 - warm_evals/cold_evals` and
`Saving_switch = 1 - warm_accepted_switches/cold_accepted_switches`.
Zero denominators yield null. First steady state counts evaluated state accesses,
including the initial state at index 1. The final state's first profile index and
its search acceptance index after the complete Top-B batch are separate fields.
Actual accepted-path activation follows the frozen search: the number of
evaluations already completed before physical activation is another field,
not an online wall-clock time-to-ready. D0 establishes whether the actual
transition starting deployment itself is steady under the new workload.

Final D1/D2 confirmation uses exactly two independent repetition seeds 8101 and
8202, distinct from search seeds 7101/7202/7303. Only exactly compatible
confirmation measurements are reused, and only after the final state is frozen.
Confirmation identities are a separate cache namespace. Identical D1/D2 final
states can share confirmation; they are not presented as independent strategy
samples. Confirmation never changes the search, accepted path or subsequent
actual state. Means, medians, sample standard deviations and individual
repetitions are descriptive only; no statistical significance claim.

Before runtime results, close performance was fixed as equal/better stability
and warm mean at most 5% above cold. PASS requires at least two transitions
close and cheaper in state evaluations or committed switches. Mixed evidence
is PARTIAL, similar cost/performance is NEUTRAL, consistently worse performance
or a propagation/accounting audit failure is FAIL. Exact-cache reuse means the
comparison is a real-feedback pilot, not a new independent remeasurement of
every cold state. Confirmation direction reversals and endpoint-cost sensitivity
must be considered when interpreting a PASS.

The fixed execution order D0→D1→D2 permits within-pilot exact cache sharing.
Physical attempts depend on which strategy first measured a state: D2 may reuse
its incumbent from D0 and its selected candidates from D1. Main evaluation
savings therefore count all algorithm-selected valid states, including exact
hits. Physical-attempt differences are also reported but do not establish
independent GPU-run savings for three separately executed strategies.

Outputs and scope follow Module 5B only. No forecasting, change detection,
prewarming, multi-replica, routing search or final paper benchmark is performed.
