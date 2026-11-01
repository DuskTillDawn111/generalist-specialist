# Active G2/OCR background collector v3

`tools/predictor_validation_20261027/batch_profile_collect_v3.py` implements a new
collector around the existing single-weight ChatQA batch owner. It keeps the
three physical owners `shared_text`, `generalist_2#0`, and `ocr#0` resident.
`text#0` and `text#1` remain logical request aliases for the one shared text
weight owner. G2 and OCR have actual declared requests, rather than idle residency.
The original v2 collector and all running v14/v15 predictor dependencies remain
unchanged by this work.

## Public input and arrival contract

Preparation requires mixed-task `samples.json` and `arrivals.json`, both with
`training` and `validation` keys. Each arrival declares `request_id`, `sample_id`,
`alias`, and nonnegative `arrival_ns`; `task` and `pool_id` are checked or filled
from its frozen sample. Every measured arrival must fall before the explicit
measurement-window boundary. Both background owners must have active arrivals
in each split. Stable public list order breaks arrival ties. Each offered request
keeps its original global arrival index in the merged report.

The separate `priming.json` has the same split keys, each containing `samples`
and `arrivals`. Priming has distinct request identities and covers every measured
physical-owner/task path. The complete frozen priming table executes on the
same resident owners before measurement. This is explicit priming evidence, not
a claim to have qualified the original largest-three-input startup policy.
Every text priming output must report success; a budget-exhausted or otherwise
unsuccessful output prevents the measurement epoch from being created.
Within a split, the same sample ID cannot denote different priming/model inputs.
Across splits, sample IDs, request IDs, text context/question content, and actual
nontext media bytes must be disjoint. Model media files are hashed and checked
again before loading. Different paths or annotation metadata cannot conceal
identical media bytes across splits.

All three owners finish loading and split priming before the main text runtime
emits `OWNER_READY`, establishing the common measurement epoch. The background
dispatcher runs in its own thread and uses this epoch for both G2 and OCR.
Text preparation or generation therefore cannot postpone their public arrival
clocks. Scheduling and OS delay remain observable: records preserve scheduled
time, arrival observation, enqueue, actual transport submission, worker START,
handler end, parent response-reader enqueue, and next-handler readiness.

Text results retain the original batch-owner timestamps, including prepare,
formation, batch dispatch, full-batch parent receipt and owner-next-ready.
Background end-to-end latency is measured from scheduled arrival to the original
parent reader's completion timestamp. It is not replaced by handler duration or
successful-response timing alone.

## Complete cohort and physical accounting

The report and per-request profile index retain every offered request, including
planned, failed, incomplete, or missing outcomes. The denominator is always the
frozen offered table. Each text batch has a separate receipt referencing its
actual member request receipts. Failures preserve prefix results, background
records, priming records, telemetry and shutdown observations. No retries or
request substitution are performed.

Real GPU collection runs inside a separate supervised process/session. The
external supervisor holds the GPU experiment lock, with an explicit timeout
covering startup, priming, measurement, drain and cleanup. It sends TERM/KILL to
the whole process group on timeout, verifies no live group members remain, and
reaps its child. An ordinary child exit also triggers a check for abandoned
workers. If a model process remains in uninterruptible kernel state, the parent
keeps the experiment lock while waiting for actual exit. Interrupts during wait,
stop or checkpoint handling cannot skip this ownership boundary. A failed or
timed-out run never claims clean shutdown simply because a thread was joined
for one second.

An atomic progress checkpoint runs every 100 ms. It preserves the complete
measured and priming plans, epoch, text/priming events, background rows and sampled
board observations even when synchronous text generation is stuck. A killed
run's checkpoint is partial evidence; missing responses or resource coverage
are not fabricated. Normal reports retain dispatcher/checkpoint thread exit
failures. The supervisor's final process-group check provides the external
execution boundary for those failures.

Real collection requires explicit `--execute`. Import, prepare and injected CPU
tests never load GPU models. The real path uses the shared GPU experiment lock;
background registry bindings must use the source-bound observed/semantic worker
and shared execution. No global one/two-request admission cap is introduced.
Each command's `--spec` must equal its frozen registry spec metadata; real READY
identifies the same spec hash and a distinct owned background PID.
The background transport queue can hold the entire declared cohort; its size is
not an execution concurrency cap.

The text backend loads once and the same object is reused for priming and
measurement. A retained reference keeps its weights alive after text requests
drain, while background requests or the declared measurement window continue.
The resource monitor includes the in-process text PID and both owned background
PIDs. Startup energy covers new owner loading plus explicit priming. Measurement
energy covers only the declared measurement window, excluding drain. Sampled
owned memory covers startup through drain. The memory safety factor is not
applied by this collector; telemetry can miss transient peaks. Missing telemetry
remains unknown rather than zero.

This new deployment does **not** inherit four-owner memory, startup, board-energy,
capacity `g(n)`, co-residency predictors, host/preparation release profiles, or
score qualification. All outputs remain `fit_eligible=false` and
`final_gain_eligible=false`. Ordered kernels and SM/DRAM/L2 resource demand are
still unknown. Active-background host preparation and parent-receipt profiles
must be collected and qualified for the new three-owner deployment before use
as a prediction model. This work prepares the mechanism ablation path; it does
not establish interference reduction or utility improvement.

## CPU artifacts and checks

The real-input/source-bound preparation demo is
`artifacts/batch-active-background-v3-20261028-prepared02/prepare-summary.json`,
SHA256 `cd63b4f86c884e88621a354ebb4a0ce90943d1f1396ec1a5335ec3dadbb0a180`.
Each split has eight offered requests: four shared-text, two G2 and two OCR.
The plan has actual source/worker bindings and a declared 1800-second whole-run
supervisor timeout, but no GPU collection has started. This is a one-second
finite profile schedule with a separately declared 300-second drain; it is not
an end-to-end throughput result or a final utility experiment.

The injected CPU execution demo is
`artifacts/batch-active-background-v3-20261028-cpu-contract02/summary.json`,
SHA256 `64c29968f376c66a7e034b04d654ce7a48ff6ade06326a7ff6b6d40ea5fda521`.
It completes all four fake offered requests and keeps physical telemetry unknown.
Its fake responses are not physical profiles or training evidence.

`tests/test_batch_profile_collect_v3.py` has fifteen tests. One blocks text execution
until a measured background request is independently dispatched, proving actual
thread overlap rather than simulating it by relabeling sequential execution.
Other checks cover all-offered failure accounting and request indices, source
and media tampering, split disjointness by media bytes, explicit priming identities,
shared text residency after text completion, E versus drain, M over the full
new deployment, and rejection of idle backgrounds/out-of-window arrivals.
Additional tests use real CPU child/grandchild processes that ignore TERM and
block until the external watchdog kills their group, check abandoned-descendant
cleanup on ordinary parent exit, and interrupt stop/checkpoint cleanup to verify
that process ownership remains guarded. No fake GPU telemetry is used.
The combined collector/batch-owner/backend and three repaired independent-review
tests have 45 passing tests, including ResourceWarning promoted to an error:

```bash
OPENBLAS_NUM_THREADS=1 python -W error::ResourceWarning -m unittest \
  tests.test_batch_profile_collect_v3 tests.test_batch_profile_collect_v2 \
  tests.test_batch_owner_runtime_v2 tests.test_shared_text_batch_v2 \
  tests.test_batch_profile_collect_v3_repaired_independent_review
```

Current collector source SHA256:
`6976145ec7c59ff848cef1bfc8e17b745df635f117f0b56bf8dea3c1d7c22dfa`.

The earlier prepared01 and cpu-contract01 artifacts are preserved. An independent
review of their source snapshot reproduced three limitations: unsuccessful text
priming could enter measurement, synchronous text execution lacked an external
hard stop, and a one-second join could leave a dispatcher alive. Preferred 02
artifacts bind the repaired source and the independent review rechecked these
boundaries. None of these CPU checks establishes real GPU performance, counter
permission, independent queue prediction accuracy, or utility improvement.
