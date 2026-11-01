# Trainable case-level sampled-memory candidate, version 2 design

**Superseded before any target fit.** The feature-only first-shard audit found
all three cumulative H columns constant after priming and C/F identical. The
eight-column proposal below is preserved as design history, not a frozen fit
specification. Use `twoFIFO_case_memory_regression_candidate_v3_20261028.md` for
the corrected, smaller identifiable candidate and the source-bound audit.

The v1 shape-sidecar/absolute-interval prototype is a strict causal reference.
Its exact duration and full poll-geometry keys are deliberately restrictive and
will rarely repeat across native captures. It must not prevent a statistical
model or serve as the final solution to memory support. The following new
candidate predicts the original case's startup-through-fixed-drain owned sampled
peak directly. It is a pre-fit design; no native fitting or provider replacement
is authorized by this document alone.

## Target and two prediction scopes

Every one of the 150 case attempts supplies the original absolute target
`Y_case = max(owned NVML samples from this attempt's startup to this case's fixed drain)`.
Keep all 150 targets, the twelve independent startup identities and all three
whole blocks. Individual rapid macros need not contain a poll. A case that has
hundreds of measured samples remains usable even if most short macro intervals
contain no samples. The target is the original monitor's sampled peak, not an
unobserved continuous physical peak or capacity certificate.

There are two distinct input scopes:

* **Already resident conditional forecast.** `P0` is the largest owned-memory
  sample that finished before forecast origin. Its sample time and complete
  causal prefix are evidence. No later target-memory samples update the model.
  This is conditional on the existing deployment and its observed prior history.
* **Pre-start new-deployment forecast.** No target startup peak exists as an
  input. Select a compatible training startup/priming donor jointly with the
  request-vector whole-block latent. Initialize `P0`, shape history and startup
  resources from that donor once. Evolve predicted earlier cases before the
  target case; do not reset to priming independently at each case. No held-out
  startup or preceding-case peak can correct this prospective trajectory.

Both scopes output one cumulative absolute M. The recurrence remains a maximum,
and the existing downstream scorer applies its original memory safety factor
once. Do not add startup or per-owner peaks as independent cumulative terms.

## Fixed eight-parameter conditional mean model

Freeze these eight columns before any native target fit. One row describes the
complete causal prior shape envelope and generated future case trajectory:

| Column | Definition |
| --- | --- |
| 0 | Intercept |
| 1 | `P0 / 1024`, the available or generated prior cumulative sampled peak |
| 2 | `log1p(H_text0)` |
| 3 | `log1p(H_text1)` |
| 4 | `log1p(H_G2)` |
| 5 | `C`, maximum concurrent processed-active resident owners in the case |
| 6 | `F`, maximum concurrent processed-active prefill/OCR entry phases |
| 7 | `R / horizon`, phase/concurrency exposure defined below |

For owner `o`, `H_o` is the maximum visible token-shape proxy from the complete
prior envelope plus generated case: `batch * (latest prefill sequence length +
decode forward_index)`. At prefill it is `batch * sequence_length`. A decode
forward index counts model forward calls; it does not prove retained KV length.
The quantity is explicitly a KV-size proxy, never measured KV bytes. If a shape
axis required for an actual text forward is absent, the row lacks that feature
and stays UNKNOWN; do not impute zero. A task that never invokes a text forward
can contribute an explicitly identified no-text state. OCR input identity and
resident checkpoint remain in the fixed physical/input domain; the current
native OCR marker has no tensor shape, so it does not receive invented H bytes.

`C` counts simultaneous processed-active owners, not actual SM occupancy. `F`
counts phases currently visible as prefill or OCR inference, with entry retained
under its declared semantic label. Compute both from the same event sidecar
that generates the queue world. For every generated structural interval, let
`n` be concurrent owners and `f` the prefill/OCR count; define
`R = integral[choose(n,2) + f] dt`. The horizon is the original 10 s measurement
plus 60 s fixed drain; thus this exposure has a fixed, interpretable scale. Idle
and drain intervals stay in the denominator. Unprocessed future shape is latent
until its generated processing time. Forecasting a complete future path may
use the generated future trajectory to summarize H/C/F/R; it must not use the
realized held-out trajectory.

Use ordinary least squares with an SVD solve and report all coefficients,
singular values, exact frozen columns and numerical rank. Do not drop a column,
choose ridge strength, select bins or remove a sample based on error. The
provisional absolute prediction is
`M_hat = max(P0, 1024 * (X beta))`.
Clipping to P0 enforces the known cumulative prefix; it does not create a
capacity or confidence bound. If the frozen eight columns lack rank in a full
or required fold, that model is unidentifiable and must report UNKNOWN. A future
reduced specification would require a separately versioned, justified design,
not a favorable result selected from this campaign.

Weight each whole block equally, each of its four independent startup attempts
equally, then each case in an attempt equally. There are 150 case rows but only
12 startups and three independent blocks; do not describe them as 150
independent trials. Do not introduce a fitted per-case ID, raw request ID,
capture hash or block-specific intercept as a generalizable feature.

## Sampling coverage is a domain, not a floating-point lookup key

Save the original measurement/drain horizon, full case poll count, maximum gap,
median cadence, boundary bracketing and fraction of time bracketed by valid live
owned samples. These are sampling-support covariates and diagnostics, not exact
hash axes. The current fixed design already requests a nominal 20 ms cadence.
Keep original raw polls; do not generate extra pseudo observations by
interpolation, pick a favorable sample phase or discard exceptional intervals.

Before fitting, freeze the intended observation domain as this unchanged live
ownership monitor, its nominal cadence, the original case horizon, and the
existing reviewed gap/clock checks. A future generated monitor plan can declare
the same domain, while its realized coverage cannot be supplied in advance as
a known future feature. If a case lacks valid power or memory coverage, retain
it and expose UNKNOWN for the affected validation; do not train a claimed
continuous peak from incomplete telemetry. Report error against the actual
sampled target jointly with coverage diagnostics and the coarse support domain.
If one later changes to a truly fixed 20 ms measurement schedule, that is a new
observation design requiring separate data, not a silent resampling of this one.

For feature support, require the exact physical/input/checkpoint domain, actual
training coverage of each discrete task/phase category, and no continuous
feature outside the donor training range in its fold. Report distance to the
donor design span, leverage, rank and all out-of-domain rows. Marginal range
support alone does not establish reliable multivariate extrapolation. Do not
train support thresholds or feature partitions from held-out error.

## Whole-block fit and generated-path evaluation

Only the complete twelve-native intake can enable target extraction and fitting.
The target adapter rederives original resource facts and preserves all 150 rows,
including those with unknown features or coverage. Each of the three folds
trains on all eight donor attempts in two whole blocks. Root and all folds must
report rank, feature support, unknowns and startup scope, without choosing a
successful fold.

Separate two checks:

1. An **actual-trajectory conditional diagnostic** may fit/evaluate H/C/F/R from
   native processed events to check whether these proxies explain sampled M.
   Its held-out actual future features are labels-side diagnostic inputs; this
   result cannot qualify prospective queue-plus-memory prediction.
2. The **prospective generated-path test** constructs H/C/F/R and, for pre-start
   scope, P0 from donor-only request/shape/resource paths. All held-out request
   vectors, macros, future processing times and memory labels are excluded from
   generation. The resident scope can use only the P0/past descriptor available
   before origin. Save predictions before evaluation labels. Errors reflect
   both generated-shape error and conditional-memory error, as required by the
   intended deployment decision.

For every fold, case, draw and input scope, retain signed M error, absolute error,
underprediction count and magnitude, known/UNKNOWN denominator and original
startup-through-drain target. Compare against the simple causal `P0` predictor
on the same rows, retaining its misses. Report dependency at startup and whole
block granularity. With three blocks do not claim calibrated tail coverage,
formal underprediction probability or a universal allocator law. The current
finite draw quadrature is not a fitted probability distribution; report its
scope rather than interpreting its empirical mean as a proven probability.

## What this model can and cannot identify

The candidate can learn the conditional association between cumulative prior
sampled peak, text-shape envelopes, concurrency/phase co-occurrence, exposure
and the next original cumulative sampled peak within the observed domain. Its
eight parameters and all-block rank checks are feasible to inspect with twelve
attempts; they may still fail because P0 dominates the target or columns are
collinear. That failure is an empirical result, not permission to hide columns.

It does not directly identify allocator free bins/segments, retained reserved
blocks, pending stream releases, actual KV allocation, kernel workspaces or
fragmentation. Different histories with identical eight features can retain
different memory states. Evaluate residuals against earlier public case order,
time since priming, retained-memory level and missing allocator observables as
diagnostics; do not add the most favorable residual feature from these data.
No amount of a good in-domain OLS fit alone turns sampled M into a physical
capacity bound. The model can be useful for ranking empirical deployment
candidates only after the independent generated-path errors and underprediction
are acceptable under the unchanged original decision rules.
