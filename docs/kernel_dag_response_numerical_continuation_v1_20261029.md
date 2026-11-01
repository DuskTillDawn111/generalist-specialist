# CPU numerical continuation after the exact kernel capture

The new `kernel_dag_response_numerical_continuation_v1.py` waits for the already
running kernel continuation PID 690454 with the exact command, manifest and
start ticks. Its canonical claim and future complete final are pinned in the
prepared manifest. Partial JSON is accepted only as a wait condition while that
same writer is alive. A reused PID, stopped/incomplete final, source change or
writer exit without the complete final stops the CPU continuation without retry.

After all three native captures and twelve CPU diagnostics are complete, this
continuation performs exactly two CPU subprocesses: the complete source-bound
data seal, then the explicit full-cohort numerical fit/HOLD. Both run with empty
`CUDA_VISIBLE_DEVICES` and BLAS/OMP/MKL threads set to one. The resource capture
chain is neither a predecessor nor a success dependency. This continuation
contains no native execution action and does not modify any GPU closure.

All original immutable inputs are watched throughout. The published data seal
is additionally watched during fitting, and each completed CPU output remains
watched until the final guard checks. A guard failure after a qualified solver
return explicitly revokes qualification and retains only an unauthenticated
result diagnostic. Native/template UNKNOWN and numerical rank/root failure are
valid completed HOLD outcomes with `numerical_qualification=false`. A stopped
numerical subprocess remains a stopped continuation, with its prefix preserved.
All outcomes remain training diagnostics; prospective prediction stays false.

Before temporary reconstruction, the continuation waits for actual free disk
space to cover `max(single native-intake temporary rebuild scenario, actual
complete repeated export bytes)`. The unchanged adapter closes intake temp
exports before writing repeated temp exports. No arbitrary extra CPU reserve is
added. The numerical fits preserve all sixty responses and the frozen six-column
solver protocol described in the numerical pipeline document.

The prepared manifest is
`artifacts/predictor-kernel-dag-response-numerical-continuation-v1-20261029-cpu01-prepared/manifest.json`.
Execution requires a separate exact root independent review with schema
`gsdeploy.kernel_dag_response_numerical_continuation.v1.independent_review`,
status `PASS_CPU_WAIT_SEAL_FIT_INTERFACES_NOT_NUMERICAL_QUALIFICATION`, exact
`manifest` and `numerical_plan` receipts, empty `unresolved_blocking_findings`
and `test_summary.status=PASS`. This review covers the repaired numerical cpu02
plan and the new CPU continuation together. The runner has not been started.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/bin/python -m tools.predictor_validation_20261027.kernel_dag_response_numerical_continuation_v1 run --manifest artifacts/predictor-kernel-dag-response-numerical-continuation-v1-20261029-cpu01-prepared/manifest.json --independent-review ROOT_REVIEW_PATH --execute
```

The canonical outputs are `<prepared>/run-v1/claim.json` and
`<prepared>/run-v1/report.json`. Final statuses are
`COMPLETE_CPU_NUMERICAL_TRAINING_DIAGNOSTIC`, `COMPLETE_CPU_NUMERICAL_HOLD`, or
`STOPPED_NO_RETRY_PREFIX_PRESERVED`. Nine CPU tests include actual inotify
change-and-restore checks; the fourteen repaired numerical pipeline tests
include the root review's qualified-solver/final-guard counterexample.
