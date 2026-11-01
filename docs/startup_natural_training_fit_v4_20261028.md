# Startup-aware natural training and handoff adapter

The new `natural_training_fit_v4.py` consumes passing v4 natural strict reviews
without rewriting the older v2 source, inputs, or receipts. It checks unchanged
review sources, complete declared jobs, training labels, native completion,
startup boundaries, telemetry, the actual four-owner shared execution identity,
and an explicit source-bound local host clock assertion. CUDA clock alignment
is outside this assertion.

Each family/task/input needs at least two distinct raw captures. Multiple trials
or the two text replicas inside one raw do not create independent captures.
Unsupported profiles stay UNKNOWN and are omitted from emitted lookup models;
supported profiles retain the original joint handoff observations. Missing
worker readiness remains null, and permit release is not applicable in shared
execution. Quality stores separate expected per-request numerators and
denominators using the original task scorer. Its fallback assumes the same
training input distribution and does not predict the actual held-out answer.

Six CPU tests cover independent capture support, partial support, validation
exclusion, receipt drift, original quality units, and missing readiness. Together
with the five startup fit tests, 11 tests pass. They use synthetic files and do
not establish physical prediction accuracy.

## First actual v4 natural fit

Artifacts are under
`artifacts/predictor-gpu-20261027/small-startup-training-patch-v4-20261028-prepared01/natural-training-fit-v4-final01/`.

The modality natural and shared-SSG natural captures share execution context
`0fa031280f202b568e9e49409b6bbcb425976ce0d777b355331c2eb244fb47f1`
and the same 12 actual priming requests. Both strict reviews pass. Their real
startup observations are:

| Capture | Startup seconds | Startup board J | Prephase owned sampled peak MiB |
|---|---:|---:|---:|
| Modalities | 66.661238355 | 4085.3054847613375 | 29766 |
| Shared SSG | 66.691745527 | 4044.9697156354323 | 29766 |

The startup fit is FITTED, with training means 4065.137600198385 J and 29766 MiB.
Startup energy excludes priming; the prephase peak includes priming and excludes
active requests. No safety factor has been applied; amortization remains 3600 s.
These values do not replace active/drain memory or measurement board energy.
They are sampled training means, with no continuous peak or confidence claim.

The handoff/quality fit is UNKNOWN: these two captures have disjoint measured
job cohorts. Every family/task/input has only one distinct capture even where
there are several repeated trials or replicas. The adapter emits observations,
quality totals, a complete training index, and explicit support counts, but no
handoff or quality prediction model. This is the expected result until matching
independent captures are added. The older two-capture v2 natural models remain
separate and retain their original execution context.

No independent validation, new mechanism gain, or final routing gain is claimed.
