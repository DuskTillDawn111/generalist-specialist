# Unified co-run collector v8_1

The new exclusive collector fixes a telemetry concurrency defect observed in the first unified capture. The old `unified-corun-v8-20261028-prepared01/unified-light` raw and strict FAIL remain unchanged. This preparation starts no GPU work and fits no parameters.

The v8 capture produced one common Nsight session with real measured kernel/NVTX coverage for all four owners and 4,968 GPU context-switch events. It failed strict qualification because adjacent board samples 3420/3421 had reversed midpoint timestamps by 2,056,614 ns. The inferred query intervals overlap: sample 3421 began first, took 12.573 ms and completed after sample 3420's 8.289 ms query. The base monitor appended in query-completion order without synchronizing the background sampler and a main-thread boundary sample. Sorting the failed raw would hide the collection defect and is not a repair.

`unified_corun_collect_v8_1.py` subclasses the old immutable `IdentityMonitor`. It installs an RLock before calling its parent initializer, which does not sample. The lock covers the complete inherited sample, including NVML queries, append and identity-event side effects. New local lifecycle boundaries use the same reentrant lock for phase changes and their sample. Shutdown phase/boundary assignment also takes that lock. The old runtime, semantic worker and collector are unchanged. Original four owners, twelve priming requests, two measured jobs/four requests and the native protocols stay fixed.

`unified_corun_review_v8_1.py` reads the actual diagnostic severity enum: Error is 3, Verbose is 4. It rejects actual Error events and data-loss diagnostics while retaining verbose entries. Combined stderr is classified by source: lines prefixed with an owned worker alias are not declared Nsight loss, explicit Nsight/nsys/CUPTI/quadd loss messages reject qualification, and unattributed loss candidates still require review. SQLite DIAGNOSTIC_EVENT remains authoritative profiler-origin evidence. This avoids accepting uncertain loss while distinguishing tokenizer/model warnings from profiler evidence.

The new context binds receipts of the old FAIL raw and report and declares serialized telemetry. It has a new schema/domain/source identity. The outer command explicitly runs the v8_1 cohort module; frozen sources include both new modules and the unchanged old IdentityMonitor implementation. No old timestamps or strict conclusions are rewritten.

The CPU-prepared packet is `artifacts/predictor-gpu-20261027/unified-corun-v8_1-20261028-prepared01/`. Seventeen focused CPU tests PASS, including two-thread full-sample serialization and reentrant boundary checks, real CPU subprocess-tree cleanup, exact original priming, common-clock four-owner correlation, missing-kernel and dropped-event rejection, and severity/worker-warning distinctions. Both prepared domains pass source/resource audit. This is preparation evidence; actual v8_1 capture success still needs strict review.

Execute only after parent review, from the project root:

```bash
python -m tools.predictor_validation_20261027.unified_corun_collect_v8_1 execute --run artifacts/predictor-gpu-20261027/unified-corun-v8_1-20261028-prepared01/unified-light-v8_1 --execute
python -m tools.predictor_validation_20261027.unified_corun_review_v8_1 review --run artifacts/predictor-gpu-20261027/unified-corun-v8_1-20261028-prepared01/unified-light-v8_1 --output artifacts/predictor-gpu-20261027/unified-corun-v8_1-20261028-prepared01/unified-light-v8_1/strict-review-v8_1
```

After a strict light PASS, execute/review the new `natural` domain and run the new review module's `pair` command. The provided `execute-paired-smoke.sh` performs that sequence and stops on failure. No retry occurs within an existing destination. The outer executor owns the GPU lock, so do not hold the same lock externally.

Whole-lifetime profiling, the declared 6 GiB incremental storage scenario, 12 GiB disk floor, 1 GiB early-abort margin, 0.2-second watchdog and 1200-second timeout remain as in v8. They include loading, original priming, active work, shutdown and trace flushing. Sampling and cleanup have finite response time; the recorded observed floor and any failure remain explicit. The complete SQLite is retained, with only one measured-phase kernel serialization.

Scheduler ownership stays UNKNOWN. In the first capture, NSpid exposed one container PID per owner and both NVML and nvidia-smi returned those same container PIDs; the scheduler table used four different PID values. Those views do not supply a reliable namespace bridge. A common session enables descriptive cross-owner timing, but neither its scheduler events nor overlapping kernel lifetimes prove same-SM execution, resource throughput demand or equal time sharing. NCU remains unused; current validation labels remain unused.
