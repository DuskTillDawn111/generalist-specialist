# Source-bound phase12 fitter/coordinator

`twoFIFO_source_rebound_phase12_fit_v1.py` prepares a CPU-only manifest binding
the explicit final cpu11 executor manifest, its expected `run01/report.json`,
the current coordinator/intake/adapter implementation closure, and source-bound
independent review receipts covering all four entry sources. No executor manifest
SHA is copied into code. Preparation and audit read source/public metadata only.

`run` returns `WAITING_FOR_EXACT12_CONTINUATION_FINAL` when the expected final is
absent. Failed, partial, foreign or contaminated finals return HOLD before any
native review, raw, adapter or numerical access. A complete final without `--fit`
returns `READY_EXPLICIT_FIT_REQUIRED`. Explicit fitting creates an exclusive
no-retry claim, rederives the exact12 intake and adapter, then uses the unchanged
`twoFIFO_phase12_fit_v2.fitted_rows` and evaluator for the full fit and three
whole-block folds. Every fold excludes all four heldout shards: eight training
attempts versus four evaluation attempts. No feature, threshold or column is
changed. The original numerical-root, nuisance-support and OCR support gates all
remain necessary for response-law compilation.

The denominator is **546 measured requests and 216 history-only priming
requests**. Macro row count is derived from all retained entry, forward, OCR and
READY-tail macros. It is not fixed to 546. The CPU scheduling fixture uses two
macros per request and checks all 1,092 rows; this is not a measured native macro
count. UNKNOWN nuisance groups remain in the full heldout denominator. A failed
root or support gate exposes `ALL_UNKNOWN_RETAIN_COMPLETE_COHORT_AND_PREDECLARED_DRAW_MASS`.

`with_verified_bundle(manifest_path, continuation_path, callback)` calls
`callback(bundle, intake_document)` inside the same source checks and rederives
the native sources once. A new source-bound training runner can fit once and
generate from the same in-memory bundle. The intake document is detached; its
temporary file path is not a persistent receipt. `audit_fit(report_receipt)`
checks saved manifest/final/source identity without repeating numerical fitting;
it explicitly reports `numeric_recomputed=False` and does not independently
authenticate fitted coefficient accuracy.

Example preparation after the final coordinator review exists:

```bash
python -m tools.predictor_validation_20261027.twoFIFO_source_rebound_phase12_fit_v1 prepare \
  --executor-manifest artifacts/predictor-twoFIFO-source-rebound-executor-v1-20261028-cpu11/manifest.json \
  --continuation artifacts/predictor-twoFIFO-source-rebound-executor-v1-20261028-cpu11/run01/report.json \
  --independent-reviews EXECUTOR_REVIEW.json INTAKE_ADAPTER_COORDINATOR_REVIEW.json \
  --output NEW_COORDINATOR_PREPARED_DIRECTORY
```

Use `run --manifest ... --output ...` to inspect WAITING/READY. Actual native
`run --fit` requires the complete exact12 final and final source review. No native
fitting, GPU work or validation-label access was performed during this delivery.
Conditional response fit qualification still supplies no USHER physical service,
prospective queue accuracy, calibrated risk or final routing gain qualification.

Validation: `python -m unittest tests.test_twoFIFO_source_rebound_phase12_fit_v1
tests.test_twoFIFO_source_rebound_intake_filesystem_v1`. All numerical functions
in these CPU fixtures are replaced by explicit synthetic responses; the tests
never invoke the optimizer on native or fixture data.
