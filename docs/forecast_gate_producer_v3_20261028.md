# Executed trained-hypothesis forecast producer v3

`forecast_gate_v3` constructs the actual new providers and event engine, calls
original `score_forecast`, and derives prediction cells before sealing. It
accepts no prefilled service, queue, E2E, P95, six-term utility or timing cells.
This is implemented CPU prediction machinery; present real inputs still have
explicit coverage gaps, so it is not yet a successful accuracy or gain result.

## Model and execution binding

The bundle schema is `gsdeploy.forecast_gate_model_bundle.v3`, with explicit
`kind=trained_hypothesis` and `kernel_duration_mode=trained_kernel_baseline_v3`.
The original natural collector execution identity remains the candidate ID;
the adapted composite model identity is separate. No original context hash is
rewritten to manufacture compatibility.

`launch_supply_composite_v2` instantiates actual fitted kernel regressions,
independently fitted supply residuals, strict handoff, and original
`FullHorizonModel`. Missing kernel support makes the complete request UNKNOWN;
measured target times are never a fallback or endpoint-retuning target.

`cuda_dag_v2` instantiates `DagFullHorizonModel`, preserving actual stream
FIFO/event/wait/blocking-sync dependencies and multiple ready nodes per owner.
Fitted activity duration and host release are separate requirements. Production
rejects observed API-issue offsets and the observed handler-end release anchor;
missing trained copy/memset duration or host-release models stays UNKNOWN.
The separate explicit profiler replay diagnostic may use observed timing.

## Handoff pooling and composite derivation

The optional `lifecycle_provider_kind=pooled_joint_mean_hypothesis_v4` requires
`allow_pooling_hypothesis=true`, the actual trained pooled lifecycle model and
its own derived contract. `pooled_composite_v4` preserves the old contract and
original component/target execution IDs, replaces the lifecycle artifact in a
new contract, records the pooling assumption and computes a new composite ID.
The existing frozen composite implementation is not edited.

Pooling uses the joint training mean by family/task/permit scope, with two
actual captures and two input IDs. It is explicitly an exchangeability
hypothesis for unseen inputs. Missing ready/receipt fields remain unknown; it
is not evidence of input independence or load-independent tails.

## Source lineage and prospective sealing

A production run requires the full declared run/candidate Cartesian product,
whole offered cohort, same public arrival/input sequence and identical original
normalizers. QTLPME and their six weights are unchanged, as are the latency
references, memory budget, power reference, memory safety factor and startup
amortization. Missing E/M/startup evidence keeps absolute utility incomplete.
The phase-start request consumes the same deterministic routing policy state
used for future arrivals; future routing does not restart at choice one.

Every current producer/package/tool source must be frozen by its current exact
receipt. An optional `artifacts.historical_resolution` manifest permits only
byte-identical historical snapshots for old lineage receipts, via
`historical_evidence_v4.resolve_closure`. Freeze the returned materialized
sources and manifest, and retain every substitution in the production receipt.
This does not claim the old direct-path closure passes or permit snapshots to
replace current executing code. All frozen protocol receipts are rechecked
before making a seal.

```
python -m tools.predictor_validation_20261027.forecast_gate_v3 \
  --protocol /absolute/frozen-protocol.json \
  --jobs /absolute/jobs.json \
  --output /absolute/exclusive-output \
  --allow-trained-hypothesis
```

## Current real diagnostic

The current draft is
`artifacts/predictor-gpu-20261027/prospective-disjoint-validation-v3-20261028-prepared02/design.json`.
It reserves 16 real public inputs disjoint by pool ID and exact content from
13 indexed actual training captures/73 requests plus declared training input
exclusions. Its four original shared candidates receive the same 181 requests
at publicly specified 0.5x / 2x / 0.5x rates across three 30s intervals, retaining
original average rates and objective normalizers. Queues are sampled every
0.9s. Original service/E2E/P95 MARE 0.2, queue-depth MAE 1 request and
sign/rank accuracy 1 thresholds are unchanged; no queue-ms gate was invented.

The older `prepared01` draft predates new modality training and has an image
pool now used for that training. It is retained as history and must not be
frozen with the newer model. `prepared02` replaces that validation input and
explicitly excludes the new 963-token ChatQA training input.

`forecast_gate_diagnostic_v3` actually instantiated the new baseline, pooled
lifecycle and DAG path for all four candidates. Diagnostic02 retains all
724 candidate/request rows, with four complete UNKNOWN cells because no
unseen-input executable DAG template is trained yet. There is no prediction
seal, GPU validation, accuracy result or utility-gain claim.
