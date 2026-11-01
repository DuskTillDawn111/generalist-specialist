# Original sender/stdin FIFO training collector v2

The new CPU preparation path combines the unchanged v1 natural matrix with
the unchanged v2 OCR-entry grid. Each of three training blocks contains all
50 cases and 182 offered requests. Deterministic interleaving and rotation
change positions across blocks without changing any public arrival condition.
Four shards per block contain 13/13/13/11 cases, so there are 12 independently
supervised startup attempts. Every attempt repeats the original 18 priming
requests; 216 priming requests are separate from 546 measured offered requests.
Every shard must complete before its block can qualify. No automatic retries,
manual sender holds, profiler or counter collection are part of this domain.

`twoFIFO_training_semantic_preflight_v2.validate(base, OCR_entry)` rebuilds
exact public inputs from the frozen source context, original pool and actual
media content, then compares every intended case and arrival. It independently
checks the prepared pool receipt. Rehashing an altered case cannot make a
missing condition pass. `load_frozen()` additionally verifies the exact two
frozen manifest receipts.

`twoFIFO_owned_collect_v2.prepare(output)` freezes its complete implementation
closure, copies only the unchanged sixteen declared training inputs, and gives
each shard a distinct execution ID. `audit_plan(path)` authenticates its exact
shard, public arrivals, original runtime configuration, 18 priming rows and source
freeze. Preparation does not create a runtime or monitor.

Every measured offered row exists before startup. Before each started case,
the collector snapshots all prior priming/measured rows, pending IDs, READY
state, queued notification events and runtime errors under `runtime.lock`.
The event queue is copied under its own mutex. Completed RESULT rows do not
stand for an empty worker or sender: the actual recorded READY/events remain
available as causal history, including an unobserved READY when it has not yet
been processed. Unknown transport/read/continuation timing stays unknown.

Each case gets an actual epoch, a fixed 10 second measurement window, and a fixed
60 second drain. Enqueue follows public time/ordinal order without a RESULT or
READY gate. Early completion does not shorten the case and late dispatch does
not extend it. A failed case stops subsequent cases, retaining their complete
`not_started` offered rows with no fabricated arrival clocks. A late response
keeps the original runtime's incomplete/late-response fields.

Snapshots use a lock-protected deep copy and an fsynced temporary file followed
by atomic replacement. The reviewed `StartupBoardMonitor`, live ownership
logic and original runtime cleanup are reused. Monitoring stops and takes its
last live-worker sample before worker close. The new final-save wrapper keeps
that order while allowing previously checkpointed files to be replaced.

The raw schema is `gsdeploy.original_twoFIFO_owned_training_capture.v2`.
It has explicit case measurement windows, complete offered rows, 18 separate
priming rows, prior histories, startup/drain samples and shutdown evidence.
Injected CPU controls are marked `CPU_TEST_ONLY`; native feature eligibility
remains false until independent raw and execution review qualifies it.

Execution reuses the original supervisor's 2400 second per-attempt timeout,
18 GiB filesystem preflight, lock lease, descendant tracking and cleanup.
The filesystem preflight must pass immediately before each future launch;
a preparation-time free-space observation is not a launch approval.

The OCR grid targets processed OCR-entry peer count 0 and positive peer count
for each exact OCR input in every complete block. Its 1/20/50/100 ms offsets with
long language peers primarily target prefill. Decode coverage is unknown until
observed; scheduled intent is not phase evidence. Full and every whole-block
leaveout response rank, score and curvature must be rebuilt after collection.
Missing support remains UNKNOWN, including natural sender-busy/worker-idle
states and structurally unavailable same-phase peers.
