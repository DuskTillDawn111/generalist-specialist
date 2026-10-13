# Closed-loop service semantics audit

The frozen worker entrypoint is `joint_deploy/worker_protocol.py:run_worker`.
It reads one stdin request, synchronously executes `handler(request)`, emits
`RESULT`, then reads the next request. The versioned Phi/MiniCPM adapter wraps
this same function; added candidate launchers import `candidate_pool_audit/new_worker.py`,
which calls the same protocol. Existing model generation, image tiles, audio
chunks, beam/token operations, or framework internal batches are within one
handler invocation. No cross-request batch collector or concurrent request
executor is in this serving path.

Consequently one unfinished client request per independent replica matches the
frozen request-serving concurrency. Each family replica is a separate process
with its original launcher; no runtime/worker patch, MPS, stream policy change,
or cross-request batching is introduced.

The new observer replenishes a replica after its completion is collected.
The measured rate includes the unchanged transport and host refill delay.
Refill delay is separately recorded; this is service capacity through the worker
interface, not a kernel-only device maximum. Measurement uses successful
responses received within the timed window divided by its actual elapsed time;
after-window drain completions are excluded.

Closed-loop raw request timings are not periodic workload E2E latency labels.
Only periodic phases are passed to the frozen stability classifier. Lower-load
periodic validation follows each closed-loop capacity observation. Existing
periodic raw observations and the initial periodic measurement plan are not
modified.

Three independent fresh-process repetitions are used for each adjacent-count
comparison. Count order alternates, seeds/samples are paired, and all observations
remain in the raw ledger. Mixed-sign gains cannot terminate a frontier. A true
measured terminal closes higher-count evidence under the user's stated stopping
rule; their g/M values are not extrapolated.

Exact source references and the immutable pre-data protocol are in
`closed_loop_protocol.json`; CPU controller validation is in
`closed_loop_CPU_validation.json`. CUDA context count is not directly observed:
NVML compute-process counts are only a proxy.

Before any closed-loop GPU measurement, source revision v2 fixed request sampling
to one shared seeded family SamplePool, matching the aggregate original periodic
sample sequence. This avoids each replica duplicating the same initial sample
prefix. The previous source remains in `source_revisions/v1/`; the effective
source hashes and reason are in `closed_loop_source_revision_v2.json`, with the
sample-sequence CPU check in `closed_loop_CPU_validation_v2.json`. No duration,
seed, repetition, input set, normalization, stopping rule, worker, or periodic
observation changed.
