# Case-level sampled-memory candidate v3: four parameters after structural audit

This design replaces the unfit eight-column v2 proposal. The correction follows
a feature-only structural check, without using board memory or power targets.
It does not change the frozen v1 reference, providers, runner or GPU campaign.

## Observed identifiability correction

The first complete native shard contains 18 priming requests and 13 cases. Its
priming proxy maxima are text#0 = 4518, text#1 = 4518 and G2 = 4669. All thirteen
subsequent cases stay inside those owner envelopes. Therefore all three
`log1p(complete prior + case H)` columns are constant; the intercept-plus-three-H
matrix has rank 1 of 4. The proposed positive excess over prior shape envelope
is zero in every case. They cannot be fitted independently in this shard.

The same no-target audit found C = F in every case, so those two peak concurrency
columns are also redundant. In contrast, current-case token demand A has five
distinct values, C has three and normalized exposure R has thirteen. The matrix
`[1, A, C, R]` has rank 4 in the observed shard. This establishes variation, not
predictive value or guaranteed rank in all later folds.

Audit source: `twoFIFO_case_memory_feature_audit_v1.py`. Receipt:
`artifacts/predictor-twoFIFO-case-memory-feature-audit-v1-20261028-cpu01/report.json`,
SHA256 `fc2cfe52ab69cb266681d3fb78a2c3717008ed5b173c63ab1aa088cc9ae3ab33`.
The helper whitelists request/event fields and never accesses `board_samples`;
it uses actual future events only to check feature structure. It is not a
generated-path prediction or a native-memory fit.

All 150 public case calendars also have a source-bound **structural witness**.
For each exact owner/task/input, the first first-shard priming-or-measured phase
profile is copied without fitting and shifted to the public arrivals, respecting
per-owner FIFO. This deliberate profile transplant produces `[1,A,C,R]` rank 4
for all 150 rows and for each 100-row donor/50-row held-out calendar split.
It shows that the public case matrix does not force these four columns to be
degenerate under that explicit timing witness. It is not a response-rate model
or an independent cross-validation trajectory; its first-shard source is reused
even in the block-1 calendar partition. Therefore the eventual full/fold
**donor-generated numeric ranks remain UNKNOWN until generation is possible**.
Those actual matrices must still pass rank four before any qualified fit.
The witness must not be cited as complete observed identifiability.

Witness receipt:
`artifacts/predictor-twoFIFO-case-memory-public-rank-v1-20261028-cpu01/report.json`,
SHA256 `53a2c0b7e9971dea15177d9624318f04906d8fb9a42f80762c55c75cbb8c0643`.
All case rows, singular values, source profile choices and all three calendar
partitions are retained. No board sample, memory or power target is accessed.

## Four fixed parameters and P0 as an offset

Keep the original absolute target Y: cumulative owned sampled peak from the
attempt's startup through the current case's fixed drain. P0 is the causally
available or generated prior sampled peak. Its known prefix contribution has
coefficient one, fixed by the cumulative-maximum definition, not estimated from
an almost constant startup peak.

Fit four coefficients with the fixed-offset model

`Y = P0 + 1024 * (beta0 + betaA*A + betaC*C + betaR*R) + residual`.

The output is `M_hat = max(P0, P0 + 1024 * X beta)`. This is one case-level
absolute forecast, not repeated addition of a poll increment. If later cases
are generated sequentially, carry the predicted prefix through each original
case once. Re-querying a case or splitting its reporting clock never adds a
term. Keep the original memory safety factor unapplied until its existing
downstream scoring boundary.

The four columns are:

* **1:** intercept, representing unmodeled in-domain conditional excess.
* **A:** sum across text#0, text#1 and G2 of `log1p(current-case maximum token-shape
  proxy)`. Each request's proxy is prefill `batch * sequence_length`, then
  `batch * (that request's prefill length + decode forward_index)`; take its
  case maximum per owner. A task with no text forward contributes an explicit
  no-text zero. Missing axes on a text forward remain UNKNOWN. Do not max A
  with the prior priming envelope. A measures current demand and can vary even
  when all shapes have previously been warmed. One pooled coefficient is a
  small empirical effect inside this exact checkpoint/input domain, not an
  assertion that different models have equal bytes per token.
* **C:** maximum concurrent processed-active owners during the generated case.
  It describes the possibility of simultaneously live demand; it is not SM
  occupancy or measured allocation.
* **R:** integral over the original measurement-plus-fixed-drain horizon of
  `choose(n_active,2) + n_prefill_or_OCR`, divided by that horizon. It supplies
  duration and phase/overlap exposure without fitting F separately. Idle and
  drain intervals remain in the denominator.

Prior per-owner shape envelopes, their zero/nonzero excess, current task/input
composition, P0 and priming provenance remain support descriptors and diagnostics.
They are not extra fitted columns in this version. If an input moves outside
the prior envelope or exact task/checkpoint domain, expose that domain departure
rather than silently assuming the same allocator plateau. The trained corpus
may support some envelope departures, but their absence cannot be imputed.

## Causal origins and prospective features

For an already resident conditional forecast, P0 and the prior descriptor may
come only from samples/events completely available before origin. No subsequent
held-out memory sample corrects the predicted path. For a pre-start candidate,
the same selected training whole-block latent supplies startup/priming P0,
shape history and request vectors. Generate all preceding cases if the target
is later in the attempt. Never initialize a later case with its actual future
startup-through-prior-case peak, and never reset every case to fresh priming.

In prospective leaveout evaluation, A/C/R must be summaries of the donor-only
generated trajectory using the processing-time shape sidecar. No held-out
future shape, macro duration or overlap enters those features. Actual-trajectory
features may be used in a separately labeled conditional explanatory diagnostic.
The latter cannot qualify the complete queue-plus-memory forecast.

## Fit, support, observation and evaluation

The twelve-complete-native gate, all 150 case targets, original sampled target
and three whole-block folds are retained. Fit weighted OLS by SVD: equal block
weights, equal four-startup-attempt weights within each block, equal case weights
inside each attempt. Report the four coefficients, singular values, rank,
residuals and all unsupported rows. Eight donor attempts train each fold; four
whole held-out attempts evaluate it. Neither features, bins, transformations nor
successful folds may be selected from memory errors. Full and all required fold
matrices must have rank four; report failure if later data reveal degeneracy.

Sampling domain is the unchanged live-ownership monitor with nominal 20 ms
cadence, original 10 s measurement and 60 s fixed drain. Save poll count,
cadence, maximum gap, endpoint coverage and ownership validity as support and
diagnostics; do not require exact duration/poll-position hashes. A short macro
without a poll does not invalidate an otherwise covered case target. Preserve
all raw samples and unknown cases; interpolation cannot invent a continuous
peak. A future monitor domain is a declared design, while its realized gaps are
unknown until observed.

Require the exact physical/input/task domain and donor coverage of categorical
states. Report donor feature ranges, extrapolation, design-span distance and
leverage. Fix support policy before errors; do not optimize a leverage cutoff
for a favorable result. Range membership is a necessary diagnostic, not proof
of reliable joint support.

For every case/fold/draw and both origin scopes, retain absolute Y, P0,
prediction, signed/absolute error, underprediction count and magnitude,
unsupported reason and the original denominator. Compare with the causal P0
baseline on the same cases. Preserve all finite predeclared draws and their
unknown mass; the current quadrature is not a fitted probability law.

## Remaining physical uncertainty

The model is intended to learn in-domain conditional sampled peaks with a few
identifiable parameters. The feature audit does not show that A/C/R explain Y.
The eventual all-row fits and donor-only generated-path errors must determine
whether it improves the P0 baseline and candidate rankings.

Allocator bins, reserved segments, pending stream releases, actual KV bytes,
workspace retention and fragmentation remain unobserved. A and R are proxies.
Different histories can share the four features and P0 but differ physically.
Residual checks against prior order and history may expose that limitation;
they must not be used to add the best-looking feature to this same experiment.
No result with three independent whole blocks alone establishes calibrated tail
coverage, a physical capacity bound or a universal sufficient allocator state.

## Isolated implementation boundary

`tools/predictor_validation_20261027/twoFIFO_case_memory_candidate_v3.py`
implements `actual_rows_from_verified`, `fit_rows`,
`evaluate_explanatory_rows`, `generated_case_features` and
`predict_generated_case`. Its public `run` first calls the unchanged strict
twelve-attempt resource loader, then checks each attempt's complete original
ordered case IDs, all 150 rows and all three whole-block leaveouts. The pure
table helpers intentionally permit synthetic CPU fixtures and do not establish
native authenticity. Partial known-row fits retain every original row and its
original weight denominator, and say explicitly that the fit is conditional on
the known rows; their public/input and observation support derives only from
those known rows.

Generated features use the unchanged processing-time shape sidecar and engine.
They reject nested future native request/memory labels, require complete
generated latent/reader/processed notices for the trace, and do not repair a
missing generated event from actual execution. Original-priming startup donors
must be in the training fold and share the selected request vectors' whole
block. A settled queue seed supplies no memory origin. Later-case generated
prefix propagation is deliberately not a new automatic facility in this file;
it requires an explicit causal/generated prior descriptor and P0.

The support context stores the prior owner shape proxy envelopes, exact
owner/task presence, owners with text forwards, and the per-owner envelope
departure pattern. Predictions outside the donor envelope ranges or without
the exact categorical pattern remain UNKNOWN. Feature/P0 ranges and the exact
public input and sampling domains are checked as well. Saved weighted design
geometry reports span distance and leverage, with no error-selected threshold.
These are empirical domain diagnostics, not a sufficient allocator-state proof.

The implementation has no main provider, runner, coordinator or scorer hook.
No twelve-attempt native fit or donor-generated full/fold forecast has yet been
run as part of the implementation checks. All qualification flags remain false.
The structural profile-transplant rank witness is not used as generated-rank
evidence. Repeated queries copy one absolute case result; they do not carry a
new memory increment or apply the scoring safety factor.

The independent CPU review found that finite coefficients can still overflow
after the 1024 scaling. The corrected candidate returns
`UNKNOWN_NONFINITE_MEMORY_ARITHMETIC` for any nonfinite unclipped or final peak,
including negative overflow that would otherwise be hidden by max(P0, value).
Weighted normalization, SVD/linear algebra and design-geometry failures likewise
produce an explicit numerical UNKNOWN while retaining every original training
row and weight denominator. Numerical failure never supplies a baseline peak as
if it were a supported fitted result. Author cpu01 records the earlier source;
cpu02 records this correction and does not overwrite that evidence.
