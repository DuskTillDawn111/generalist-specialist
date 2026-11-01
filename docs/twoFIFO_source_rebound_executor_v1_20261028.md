# Source-rebound execution boundary v1

The input is the independently reviewed cpu02 campaign SHA `5bad96124a6fd61fce1d7c7aabe705288c8d582b21416eb2801388191eb6ae13`, prepared by SHA `2b84f3438eee14d6e7ce8a00ea32a3d3699eae863672a390fd52381362a340cf`.

This CPU implementation adds three separate source files:

- `twoFIFO_source_rebound_execute_v1.py`: independent full audit, explicit execution/child boundary and fixed-order continuation.
- `twoFIFO_source_rebound_review_v1.py`: explicit new-schema native review with the original cohort, chronology, history, raw-output, shutdown, ownership, sample and checkpoint checks.
- `twoFIFO_source_rebound_macros_v1.py`: new-schema macro extraction which rederives the exact new review before using labels.

No old collector, runner, coordinator, artifact or module global is modified. The new collector function has the frozen v3 body, with a new local audit, new raw schema and an explicit executor closure receipt. Dispatch, checkpoint ledger, monitor, process cleanup and runtime configuration helpers are the unchanged original functions. The old v3 reviewer/intake deliberately rejects the new raw and review schemas.

The independent audit authenticates the exact cpu02/preparation receipts and old campaign; reconstructs base/OCR public cases from actual input content; checks actual media SHA/size, the complete sixteen-input pool, eighteen priming requests, each balanced shard and runtime policy; compares every plan against its old plan with only the declared rebound differences; checks every direct source, canonical frozen file, context source and complete collector import closure. The execution manifest additionally freezes the new executor, reviewer and macro adapter import closure and all campaign/context/plan/pool/media receipts.

Every execution has the new bound metadata and existing fresh cpu02 execution ID. A canonical attempt claim is exclusively persisted before launch. Existing collection, failed execution or orphaned claim cannot be retried. Only the existing supervisor owns the GPU lease; its 2400-second timeout, 18 GiB preflight, disk reserve, cleanup, descendant reaping and source gates are reused unchanged. Linux inotify watches authenticated file names, every path ancestor and canonical source inodes through the run. Existing baseline hardlinks are accepted with their exact authenticated nlink; new alias changes are rejected by inode events and final metadata. Direct write/restore, atomic replacement, ancestor swap/restore and hardlink-alias write/restore all invalidate the run without polling during measurement. Watch overflow/loss fails closed. The final source audit occurs while the original supervisor holds its experiment lock.

The continuation uses fixed twelve-attempt order. Each attempt must pass exact new execution binding, new native review, complete startup and per-case measurement-energy observability and new macro rederivation before the following attempt starts. Full offered cohorts, public FIFO arrivals/routes, 10-second measurement, 60-second drain, scheduled-arrival-to-RESULT-reader E2E, startup-to-drain owned sampled memory and measurement-only energy remain unchanged. It adds no admission cap or resource controls. Missing/incomplete evidence stops the continuation and remains on disk.

After exactly twelve reviewed attempts the status is `COMPLETE_ALL_TWELVE_REBOUND_NATIVE_ATTEMPTS_REVIEWED_NO_FIT`. Numerical fit is disabled. A new source-bound intake and completion coordinator remain required; old coordinators must not consume these new receipts.

Prepare and audit CLI paths are CPU-only. `run` defaults to no execution; native `execute` and child commands require `--execute`. All current tests and preparation are CPU-only. No claim of native predictor accuracy, kernel-level interference prediction, or final routing gain follows from this work.

Validation: `python -m unittest tests.test_twoFIFO_source_rebound_execute_v1 -v` (16 tests), covering real full source audit, unchanged shared helpers, CPU fake collection and complete original timing, old-schema refusal, full-cohort/overlap rejection, source change-and-restore, replacement/deletion, ancestor swap/restore and hardlink-alias detection, no-retry/default-off behavior, execution receipt rejection and exact twelve-step control order.

Draft executor manifests `predictor-twoFIFO-source-rebound-executor-v1-20261028-cpu01` and `cpu02` predate final source hardening and are superseded. No cpu03 manifest was created. The final prepared/audited CPU manifest is `artifacts/predictor-twoFIFO-source-rebound-executor-v1-20261028-cpu07/manifest.json`, SHA `8db769eabc7130e4633f045824a84d1722b3e03ce6b80db6c027fb7e41004cab` (fresh campaign `a2ce934620bd0a88c3ce28b34287a67b40a87d90661baa84df27df946c287559`).

Frozen implementation SHAs: executor `9b04941e4341a2bb71de953b0878f38daec9606abf3ff2709f09de9f0b8c7388`, review `39a42c9f310eb5c3a1a2f885a569163070c1822abbd8b4301ef0d7282cb87a0c`, macros `48fb74001cb95f1e6a4e1102e310277915724b43f42710d6d213ea8dba6282ae`. Independent source-bound review is required before native execution.
