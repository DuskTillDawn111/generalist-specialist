# Native queue-training strict review v8

The CPU-only reviewer is `tools/predictor_validation_20261027/native_queue_training_review_v8.py`.
Run it against a supervised execution directory:

```sh
python -m tools.predictor_validation_20261027.native_queue_training_review_v8 \
  --plan /absolute/prepared/plan.json \
  --run /absolute/supervised-run \
  --output /absolute/new-review-directory
```

`--run` contains `execution.json`, `execution-start.json`, and the child
`collection/raw.json` and `collection/scored-raw.json`. The output directory is
exclusive. A missing or contradictory receipt produces a failed report.

The reviewer requires the unchanged training plan, parent context, source and
weight-stat receipts, the complete original 16-second trace and routes, four
distinct READY owners with matching specifications, and exactly twelve original
primers. All primer RESULT and next-request-ready reader/collector boundaries
must finish before measurement begins. Every primer and measured request passes
the existing START/PHASE/RESULT/next-ready clock and identity verifier. Output is
reparsed with each frozen CPU adapter; OCR retains its existing CTC output check.
Per-owner handler and next-ready ordering must remain FIFO.

The supervised controller must report a clean exit, no live owned descendants,
the declared 1200-second and disk budgets, and a successful final source audit.
Native telemetry is checked in its original order using the inherited owned-PID
resource helper and additional board/PID-mapping checks. Samples are never sorted
or repaired. Quality totals are recomputed from actual outputs and the declared
training references. The original six-term diagnostic calculation retains the
full measurement window, startup amortization, and single memory safety factor.

A report with `status: PASS` qualifies this training capture only. It does not
establish sustained stability, independent prediction accuracy, deployment gain,
GPU cursor position, owned scheduler attribution, or simultaneous use of an SM.
The short burst trace and sampled memory scope remain explicit limitations.

The collector still transmits the inherited complete sample payload and reads
the original catalog before inference. Training feature extraction must exclude
answers, references, quality scores, and future feedback; the collector's source
identity does not establish that later feature isolation by itself.

CPU verification:

```sh
python -m unittest \
  tests.test_native_queue_training_v8_independent_review \
  tests.test_native_queue_training_review_v8 -q
```

Seventeen synthetic CPU checks pass. They cover source/weight/context drift,
cleanup failures, task-rate identity, validation relabel prevention, and strict
review mutations of routes, outputs, quality totals, phase anchors, telemetry,
priming, reader-ready timing, final source audit, and owned cleanup. Hardware is
mocked in collector failure-path tests; the reviewer fixture uses an invented CPU
protocol adapter. These checks provide no GPU collection or model-accuracy evidence.
