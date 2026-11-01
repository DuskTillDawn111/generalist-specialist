# Explicit actual-worker world forecast v2

`tools/predictor_validation_20261027/actual_worker_world_forecast_v2.py`
adds an isolated integration entry point without editing frozen components:

1. Validate the complete public cohort, order requests by scheduled arrival and
   original input position for ties, and assign an ordinal within each
   `(owner, task, pool_id)` key.
2. Select the explicitly supplied shared `RequestPlan` or ordinal-specific plan.
   Exactly sum every macro's `reference_work_ns` before simulation starts.
3. Run the unchanged v4 parent-sender / actual-worker-stdin FIFO through the
   original fixed drain with supplied protocol slacks, transport draws, rates and
   initial stations.
4. Apply the v1 whole-world original six-term score/resource integration.
5. Apply the independently reviewed queue diagnostic wrapper using the work
   mapping derived in step 2. Retain all public arrivals, including future arrivals
   in earlier event slices.

There is no external work-cost override, alternate cutoff or global admission-cap
argument. Work is bound to public arrival ordinal, so enqueue reordering and public
identifier renaming cannot silently reassign it. It is the sum of macro
response-reference coordinates, not solo wall service, SM demand or physical
throughput. Changing supplied response rates can change generated wall latency
without changing offered reference work.

The resource provider is optional. Missing energy/memory support and incomplete
latency/P95 cohorts retain their unknown terms and utility. Resource execution
binding remains caller declared; even a complete synthetic score grants no
physical, native, provider, risk, stability or final-routing-gain qualification.
Original integer public arrival/scoring boundaries and the bridge's declared
rounding of fractional generated timestamps are preserved. Queue diagnostics use
exact rational clocks and the complete-cohort linear empirical P95.

The returned object includes the generated world, original score/resource view,
queue diagnostics, shared public fingerprint, and explicit per-key/ordinal macro
work breakdown. Current native risk features and physical resource demand remain
`None`; all qualification flags remain false. This is a CPU composition path for
reviewing accounting and integration, not a fitted response or transport law.

Eight tests cover full public identity/future-work conservation, fractional
multimacro sums and input reordering, enqueue reordering, reference-work versus
wall-time separation, optional resources/unknown startup, original-drain
incompleteness, required explicit inputs and override rejection, and shared plans
with renamed public IDs. Seven synthetic report scenarios include complete burst,
resource-unknown and explicit CPU resource worlds, unknown startup, faster response
with unchanged offered work, fractional heterogeneous plans, original-drain
incompleteness and sender enqueue reordering. No heldout labels or GPU work enter.

Report:
`artifacts/predictor-actual-worker-world-forecast-v2-20261028-cpu01/report.json`

SHA256:
`8acb0ad161794964ae8aa002a30c0d3b2882b884423656538190044a1b44811b`
