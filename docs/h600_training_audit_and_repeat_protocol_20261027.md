# CPU audit and repeat diagnostics for the H600 training campaign

`training_campaign_audit_watch.py` is a finite CPU-only watcher. It does not
launch native workers, retry inference, fit a model or change acceptance rules.
It polls this campaign's completed scored files once per minute. Each completed
run receives four exclusive reports; existing reports are verified and reused.

1. `run-audits/<run_id>.json` verifies unchanged native/scored binding, strictly
   remaps source-bound actual START/PHASE/FIFO/owned telemetry/shutdown evidence,
   and independently recomputes every observed quality numerator/denominator.
2. `diagnostics/<run_id>.json` uses the original six terms and stability rules,
   and summarizes service, public-arrival wait, parent E2E and response tail by
   input and by public-arrival third. Drained requests remain in the cohort.
3. `diagnostics/<run_id>-input-composition.json` preserves each owning alias,
   task and input. Each arrival third has observed n, minimum, maximum, mean and
   P95 service. Within an alias/task, descriptive standardized means use that
   alias/task's full-cohort input frequencies. Missing input support produces
   UNKNOWN rather than substituted input means.
4. `diagnostics/<run_id>-idle-history.json` records each measurement request's
   actual previous handler on its owning alias, including priming and warmup
   predecessors. Idle gap is previous worker END to current worker START. It
   is not request arrival wait or a GPU idle counter. Fixed gap bins end at
   1, 10, 100 and 1000 ms. The fixed minimum cell n is 5; sparse bins retain
   their counts and observed ranges but are explicitly marked insufficient.
   Same-input log1p(idle)-service correlations are UNKNOWN below n=5 or without
   observed variation. Input/alias/task coverage in every arrival third is
   separately counted against n=5; there is no sparse-cell fallback.

The watcher atomically updates only `training_v14_h600` in the mutable root
progress file. It retains per-run source receipts, quality/request counts,
explicit collection/source/quality failure counts and diagnostic references.
It creates `completed-independent-training-audits.json` only after all ten
declared runs have passed audit. A collection or audit error terminates the
watcher and preserves existing outputs.

Each identical-trace repeat is compared with its declared primary after both
have passed audit, even when collection order places the repeat first. The
comparison independently checks the entire input/arrival/route/resource trace,
preserves paired per-request service/wait/E2E values and differences, and
reports observed means, P95s and paired absolute differences per alias/task/input.
Repeats remain diagnostics only and never add fit rows or independent seeds.

All ranges and comparisons are descriptive. One run pair does not calibrate
population variance or confidence intervals. Input mix, owning alias, idle
history, observed competing phases and queueing remain distinct possible causes;
these reports do not identify a causal age or cache effect. The fixed OCR model
and its whole-seed CV plan remain unchanged.

Seven focused CPU tests cover quality tampering, incomplete/duplicate cohorts,
input-composition reweighting and missing support, real priming predecessors,
sparse idle cells and overlapping history, paired traces and service/queue
differences, and atomic progress updates that preserve unrelated fields.

The active watcher command is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/root/miniconda3/bin/python -u -m tools.predictor_validation_20261027.training_campaign_audit_watch \
  --campaign artifacts/predictor-gpu-20261027/continuous-training-v14-h600/campaign.json \
  --progress artifacts/predictor-gpu-20261027/progress.json
```

Do not start another watcher against the same exclusive reports. The active
session is 31402; its log is `/tmp/predictor-training-v14-cpu-audit-watch.log`.
The native GPU campaign is owned by the parent agent, separately from this
watcher.
