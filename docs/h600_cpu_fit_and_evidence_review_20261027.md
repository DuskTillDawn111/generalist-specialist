# H600 training: independent CPU and evidence review

This review reads training sources, prospective campaign inputs and H30 training
preflight evidence only. It does not inspect v9 or v11 runtime outcomes and does
not launch GPU inference.

## CPU feasibility

The existing numerical fitter scans phase boundary cuts against all phase
segments, then builds per-target exposures. For an intact flow with S segments,
this can approach quadratic work. H600 has 1200 requests per flow, versus 181 in
the historical H90 flows: about 6.63 times as many requests and potentially about
44 times as much Python scanning per flow. This is an algorithmic comparison,
not a measured production-runtime forecast.

A synthetic one-thread CPU benchmark with 1200 requests and two aliases took
0.456 seconds for two aligned phases and 2.788 seconds for four phases with
noncoincident boundaries. These are simpler than actual source verification,
multi-target regression construction and repeated semantic remapping; they do
not bound real fit time. The data size is reasonable for CPU memory. Strict
mapping, JSON copying and transitive provenance reconstruction may dominate
the small bounded regressions.

Use one BLAS thread, two whole-H600-seed folds and one full fit. Preserve intact
cohorts; neither arbitrary request subsampling nor a per-flow hyperparameter
search is needed. Any future event-sweep optimization should be implemented in
a separate version with equivalence fixtures rather than editing frozen code.

## Historical data and repeated runs

The final primary estimator uses the eight historical H90 flows, eight new H600
flows and the existing v7/v8 phase-profile collections. Pooling a conditional
phase law assumes stationarity within the shared execution context; it does not
establish stationarity of queue tails. Horizon, drain, seed, candidate, run ID
and source role remain explicit metadata. Capacity conditions remain exact:
H90/drain120 evidence does not qualify H600/drain300 admission.

The two identical-trace repeats are diagnostics only. They do not enter baseline
means or coefficient rows, avoiding extra weight on the first seed. They are not
new independent input realizations. H600 cross-validation excludes all four
primary routes of one new seed, retains all eight historical H90 flows and the
other new seed, and evaluates only the four excluded H600 flows.

## Read-only implementation review

`run_aware_training.py` preserves the original immutable mapping and numerical
components. Prepare reads prospective inputs without opening future H600 raws.
Its combined suite has 16 unique primary flow identities and 18 distinct raw
paths including the two separately labelled repeats. Its fit path has 12
continuous training flows per seed fold, or 16 in the explicitly requested full
fit. The verifier independently remaps original raw sources and checks direct
and transitive receipts, complete evidence groups, held-out seed membership,
run metadata and excluded-repeat policy. Four real CPU fixture tests passed.

The new manifest's `training_verifier` declaration does not automatically select
that verifier in older reporting programs. A new report adapter must explicitly
call `verify_run_aware_model`; the old source-bound reporters should remain
unchanged. This interface requirement was reported to the implementation owner
and parent before actual protocol preparation.

## Collection audit

The native runner already writes immutable `raw.json`, then creates a separate
`scored-raw.json` with measured quality totals and the native receipt. The new
CPU-only `training_run_audit.py` independently recomputes every quality numerator
and denominator from the frozen catalog, checks full offered-cohort coverage,
checks unchanged native fields and annotation source receipts, and calls the
strict original semantic mapper. Two tamper/incomplete/duplicate/cohort tests
passed.

The H30 preflight audit passed: 61/61 requests completed, 61 quality records,
zero quality/collection/runtime/telemetry errors, four clean BYE shutdowns and
5675 telemetry samples. This is a collection-readiness result and is explicitly
excluded from H600 fitting. Its audit is
`artifacts/predictor-gpu-20261027/continuous-training-v14-h600/preflight/postcollection-audit.json`.
It is neither an independent prediction validation nor final spillover evidence.
