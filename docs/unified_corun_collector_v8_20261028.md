# Unified co-run collector v8

Implementation status: implemented as new exclusive modules after the parent released its source-stable window. Twelve focused CPU tests and twenty-three existing related checks PASS. No GPU collection was started by this implementation task.

The first collection will be one Nsight Systems session over a parent cohort and plain original semantic workers. It will preserve four original physical owners, twelve original priming requests, exact training inputs and protocols, CPU output reparse, lifecycle boundaries and sampled board/owned-process telemetry. Its context and raw schema will explicitly declare profiling and its new source identity. The original collector and reviewers remain unchanged.

The new modules are `tools/predictor_validation_20261027/unified_corun_collect_v8.py` and `unified_corun_review_v8.py`, plus focused CPU tests. The preparation CLI creates an exclusive paired cohort with new frozen source/context receipts. Default jobs are the bounded Stage B cohort: short simultaneous SSG followed by isolated short OCR, four measured requests per domain. Original long/short priming remains fixed. Natural and unified-light domains share exact jobs; natural executes without Nsight.

## Capture coordination

The initial mode uses `--capture-range=none`, explicitly verified in local 2025.6.3 CLI help. Nsight then ignores CUDA profiler start/stop calls and NVTX ranges as capture triggers. The collector sends no profiler-control requests. This prevents one owner's stop from truncating peers. Startup, priming, measured jobs and shutdown remain in one trace. Every measured owner's semantic NVTX and CUPTI kernel coverage is a strict gate.

The outer options are `--trace=cuda,nvtx --cuda-trace-scope=process-tree --gpuctxsw=true --sample=none --cpuctxsw=none --trace-fork-before-exec=false --wait=all --cuda-memory-usage=false --force-overwrite=false`. Native children execute normally without individual Nsight wrappers. Local help warns that pre-exec fork tracing with `true` relies on undefined behavior and may crash or deadlock; default process-tree includes children after exec.

An active-only range emitted once by the parent might reduce startup/priming trace overhead. It is a future verified mode, not an assumed capability. The initial bounded smoke uses whole-lifetime capture and a conservative storage guard.

## Supervision and cleanup

The outer executor alone holds the GPU experiment lock. It audits sources before launching, writes an exclusive execution-intent receipt so a failed attempt cannot be silently retried, and starts the profiled cohort in a new process session/group. A continuous wall-time/disk watchdog covers worker loading, requests, shutdown and Nsight trace flushing. A disk margin above the 12 GiB floor triggers cleanup early. The whole-lifetime storage scenario must be declared before execution and remeasured afterward.

On timeout, low disk or failure, terminate only the owned launched process group, escalate if needed and retain exit/status/watchdog evidence. A successful process exit still requires no surviving owned workers, a completed cohort raw, clean shutdown and a final source audit. Existing per-worker shutdown reports and lifecycle telemetry remain mandatory. No old artifact is deleted to make room.

## PID and scheduler attribution

The existing runtime already starts one worker at a time and waits for READY. New monitoring records each launched PID, process start ticks, parent/session/group, NSpid vector, PID namespace inode and an execution nonce while processes are alive. Raw NVML compute-PID views and nvidia-smi views are retained with timestamps during transitions.

Prior training's raw NVML PIDs exactly matched container READY PIDs. Thus NVML startup deltas alone do not bridge scheduler host PIDs. Any controlled mapping requires stable no-unknown baseline, one startup at a time, exactly one stable new identity, unchanged previous identities, nonce/ancestry/start-time evidence and a demonstrated namespace difference between actual views. Churn or multiple additions invalidate the attribution. If both views expose container PIDs, scheduler ownership stays UNKNOWN. CUPTI context IDs and driver scheduler context IDs are distinct domains and are never joined by numerical equality.

The unified trace remains useful if ownership cannot be identified: common-session kernel/phase timing can pass while owned scheduler attribution is UNKNOWN. Scheduler events do not prove same-SM concurrency, throughput demand or uninterrupted execution within a kernel lifetime.

## Strict review

The new reviewer binds raw, plan, context, executed jobs, controller execution and all evaluator/source receipts. It reuses established exact-request coverage, startup telemetry, CPU output reparse and worker phase checks without changing those implementations. It separately verifies exactly twelve original priming rows and original four READY/shutdown owners.

Unified-light exports one report to one SQLite exactly once. It parses measured anchors once, keeps native Nsight-relative timestamps for all owners and records one session origin. Per-owner filtered views are passed to the existing kernel/handler-boundary validator; its median worker mapping remains descriptive and is not used to realign inter-owner intervals. Only measured phases are serialized as canonical parsed JSON, avoiding duplication through full ranges and unattributed startup kernels. The complete raw SQLite retains startup/priming, runtime APIs, copy/memset, sync and scheduler activities.

Diagnostic events and stderr are retained. Dropped/lost/overflow/truncation diagnostics, incomplete collection or failed coverage reject qualification. Expected warnings about the non-CUDA parent are reported alongside per-child actual coverage; they are not by themselves evidence of child capture failure. A companion paired-work review requires identical actual output/token/semantic-forward and priming work across natural and light, with timings retained separately.

Focused CPU checks cover plain child commands and honest domain labels, exact original twelve-request priming, one-session parsing with process-scoped correlation IDs, all-owner coverage, dropped-event rejection, PID ambiguity remaining UNKNOWN, source/context tampering, timeout/low-disk supervision and cleanup of a real CPU-only subprocess tree.

## Prepared handoff

The CPU-prepared packet is `artifacts/predictor-gpu-20261027/unified-corun-v8-20261028-prepared01/`. Both domains pass source/resource audits under their new frozen identities. `execute-paired-smoke.sh` runs unified-light, strict review, natural, strict review and the actual-work pair review. It has passed shell syntax checking and has not been executed by this implementation task. The outer execute writes an exclusive intent and refuses reuse; a failed attempt requires a new prepared destination.

Commands from the project root:

```bash
python -m tools.predictor_validation_20261027.unified_corun_collect_v8 prepare --output EXCLUSIVE_PACKET
python -m tools.predictor_validation_20261027.unified_corun_collect_v8 audit --run EXCLUSIVE_PACKET/unified-light
python -m tools.predictor_validation_20261027.unified_corun_collect_v8 execute --run EXCLUSIVE_PACKET/unified-light --execute
python -m tools.predictor_validation_20261027.unified_corun_review_v8 review --run EXCLUSIVE_PACKET/unified-light --output EXCLUSIVE_PACKET/unified-light/strict-review-v8
# Repeat execute/review with natural, then:
python -m tools.predictor_validation_20261027.unified_corun_review_v8 pair --light-report EXCLUSIVE_PACKET/unified-light/strict-review-v8/report.json --natural-report EXCLUSIVE_PACKET/natural/strict-review-v8/report.json --output EXCLUSIVE_PACKET/paired-actual-work-v8
```

The default preparation declares a **6 GiB incremental storage scenario**, including whole-lifetime startup/priming trace overhead, not a size bound. It requires 12 GiB reserve plus this scenario before launch. The continuous watchdog polls every 0.2 seconds, aborts below 12 GiB + 1 GiB margin or when the scenario is exceeded, and limits the full outer process lifetime to 1200 seconds. Cleanup escalates within the owned process group only. Sampling and termination have finite response time; these controls are not an instantaneous disk-write guarantee. Raw and watchdog records disclose the observed floor and any failure. The complete captured SQLite is retained; measured-phase JSON avoids serializing the full session twice.

The first GPU result is still a capability experiment. Source audit and CPU tests do not demonstrate four-owner unified capture success, profiler perturbation size, scheduler ownership, same-SM concurrency or a resource bottleneck. Those remain subject to actual strict review, with scheduler ownership UNKNOWN in this version even when collection qualification passes.
