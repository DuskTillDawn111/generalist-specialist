# Context-switch capability and startup-aware training patch

## Actual context-switch capability probe

`artifacts/predictor-gpu-20261027/context-switch-probe-v4-20261028-prepared01`
is an actual small GPU capture. One system-level Nsight session launched two
independent CUDA processes. Owned PIDs 172026 and 172027 emitted READY/DONE,
performed respectively 4,232 and 4,224 matrix multiplies, and returned the
expected scalar 1,024. The same session contains their two NVTX ranges and
4,235 / 4,227 CUDA kernels (including preparation).

The strict CPU audit at `strict-review-v4/report.json` finds 2,130 actual
`GPU_CONTEXT_SWITCH_EVENTS`, with contiguous sequence numbers and 1,065 each
of RESTORE_START and SAVE_END. No permission, dropped-event, or error diagnostic
is present. Parent-only no-CUDA/no-NVTX warnings remain recorded rather than
discarded. This confirms the ability to collect GPU context-switch events.

The switch events decode to PIDs 557026 / 557027, while owned CUDA children
decode to 172026 / 172027. The SQL export does not provide a PID-namespace map.
CUDA context IDs are local value 1 in each process; the driver switch context
IDs are 4635722 / 4635741, with no exported bridge between these ID spaces.
Subtracting the apparent PID difference would be a guess. The audit therefore
reports `CAPABILITY_CONFIRMED_OWNED_SWITCH_UNKNOWN`. It does not qualify the
original four-owner scheduling mode, timeslicing, MPS, or same-SM concurrency.
Kernel lifetime overlap alone would not establish simultaneous SM execution.
No repeated probe is required merely to remove this UNKNOWN.

## New startup telemetry

`startup_collect_v4.py` is a new collector; previous captures and their frozen
source files remain intact. Its independently importable `StartupBoardMonitor`
keeps the original monitor interface and adds:

- Board used/total/free memory and sampled actual per-process memory.
- Actual descendant ownership before READY, including profiler launcher trees.
- Per-owner loading state and explicit phase labels. Missing process observations
  are not filled with fictitious zero allocations or instantaneous peak claims.
- A pre-startup sample, followed by startup_begin before runtime construction.
  This makes samples envelope the startup integral without extrapolation.
- Explicit startup, priming, active, and shutdown boundaries in the original raw
  artifact, with monitoring continuing through shutdown.

The old monitor already used the launcher PID as a fallback before READY for
direct Python workers. The new ownership logic additionally handles actual
descendants of wrappers such as Nsight.

`startup_training_collection_review_v4.py` checks schema
`gsdeploy.startup_training_collection_strict_review.v4` with the usual status,
raw, plan, context, evaluator receipts, and actual capture identity. It requires
the exact lifecycle boundary sequence, startup-to-shutdown telemetry coverage,
valid owned/board samples during loading, and all four READY PIDs during the
active observation. It checks measured and priming outputs against the frozen
CPU adapter and generation policy. Inactive residents remain resident and are
not assigned fictitious measured kernels. Metadata operator/actual-CUPTI links
are audited only for owners with measured requests.

## Small training patch

Executable preparation:
`artifacts/predictor-gpu-20261027/small-startup-training-patch-v4-20261028-prepared01`.
All five run manifests passed source/resource audits before execution. The
initial `prepared00` is an unused CPU preparation; the valid entry is prepared01.

The modality shard has four solo G2 requests: image/ASR, each short/long, in
metadata, light-kernel, and natural domains. The shared SSG shard has six jobs:
short/long text, simultaneous and 50 ms staggered in both owner orders, in light
and natural domains. This is 48 measured requests across the five runs. Every
run loads all four original owners and primes only training inputs; no held-out
validation input is used. Metadata need not be collected for every shared job:
later label reuse must prove exact source, work, and ordered-kernel structure
correspondence to an actual solo observation, otherwise remain UNKNOWN.

The runner retains the GPU experiment lock, exclusive outputs and no retries.
It checks the 12 GiB disk reserve and 2,400 second collection budget at phase
boundaries and before jobs; it does not have a continuously enforced watchdog
during model loading or a long request. New modality trace
size is unknown; each small shard must be reviewed before wider collection.
Actual execution status is recorded only by each run's raw and strict review.
No successful execution or predictive accuracy is inferred from preparation.

CPU verification: 34 relevant collector, telemetry, bridge, strict coverage,
operator, repeated-capture, and design checks passed. These tests cover loading
ownership, foreign PIDs, startup sampling before model construction, integral
coverage, cleanup and raw preservation after a failed startup. They do not
replace GPU validation.

Long text decode, broader SS/SG/SSGO/background coverage, held-out slowdown/P95/ΔU
ranking, admission/resource-share comparison, shared batching/weights, and final
four-group end-to-end benefit remain separate pending evidence.

## Actual G2 modality patch results

The four-request modality shard has now completed in all three domains. Its
`modalities/three-domain-bridge-review-v4/report.json` is PASS, SHA256
`4efb8e212d077de061587c734120b15e13baf245df9c128e6025a2efc3eadb9e`.
All actual input/output, policy, token work and semantic forward shapes match.
All 87 kernel groups, totaling 184,776 kernels, have exactly equal observed
launch signatures and order; 22,151 affine kernels have exact metadata-to-light
identity links. This new modality set has only one independent capture.

| G2 request | generated tokens | metadata ms | light ms | natural ms |
| --- | ---: | ---: | ---: | ---: |
| short image | 4 | 249.92 | 200.47 | 157.38 |
| long image | 3 | 676.94 | 705.63 | 622.94 |
| short ASR | 21 | 990.20 | 705.81 | 488.73 |
| long ASR | 55 | 2835.39 | 1779.10 | 1237.46 |

ASR therefore adds real 21/55-step generation coverage. Original text requests
still only have five generated tokens; no claim of long text decode is made.

The original metadata strict review remains FAIL: a single NVTX range has a
196.9 ms tail after a 33.4 μs affine host call, rejecting the original median
residual clock heuristic. The frozen source places PHASE emission between the
host end timestamp and NVTX pop, so this tail is not itself a clock-drift
measurement. A separate conditional session-origin review preserved every
anchor and all CUPTI timestamps. All 21,102 causal brackets have intersection
width 8,421 ns, containing SQLite's actual session systemClockNs. This new
lineage passes; it does not claim calibrated cross-process precision. The
original failed report and parsed artifact were not modified.

The original light strict review also remains FAIL for one review-rule defect:
NVML explicitly reported 0.0 MiB for an owned PID during shutdown. This is an
actual zero observation, not a missing PID filled with zero. The new independent
v4.1 host review allows nonnegative observed process usage outside active work;
the original strict active verifier still requires all READY PIDs with positive
usage. The new passing host review is joined with the already passing original
trace checks without re-export, sample removal, or raw changes. Natural's full
v4.1 strict review passes directly. Independent reports retain each prior
artifact and evaluator receipt.

## Actual shared SSG training observations

The six shared jobs completed in both light-kernel and natural domains, with
all four original owners resident. Both v4.1 strict reviews pass. The independent
`shared-SSG/paired-actual-work-review-v4/report.json` also passes, SHA256
`68380956f6a1b94150834fa3d9a72a173bcf941bc6ead21a3057d80ae8135225`:
all 18 measured requests and the 12 priming requests have matching actual work
across the two domains. This does not transfer profiler timings to natural
execution or qualify cross-process GPU clocks.

Natural handler durations in milliseconds are:

| input | launch order and offsets | text#0 | text#1 | G2 |
| --- | --- | ---: | ---: | ---: |
| short | simultaneous | 430.05 | 445.72 | 449.73 |
| short | text#0, text#1, G2 at 0/50/100 ms | 320.32 | 363.32 | 331.11 |
| short | G2, text#1, text#0 at 0/50/100 ms | 397.10 | 359.70 | 483.22 |
| long | simultaneous | 2101.86 | 2101.03 | 2208.14 |
| long | text#0, text#1, G2 at 0/50/100 ms | 1978.57 | 2048.69 | 2135.51 |
| long | G2, text#1, text#0 at 0/50/100 ms | 1996.60 | 1980.25 | 2228.05 |

These are training observations from one independent shared capture, not
held-out slowdown, queue, ΔU, or ranking validation. They show dependence on
launch order in this observed cohort; they do not establish a causal resource
complementarity mechanism without the corresponding counter evidence.

A second independent SSG natural startup capture subsequently completed at
`small-startup-training-patch-v4-20261028-capture02-prepared01/shared-SSG/natural`.
Its 18 measured requests pass strict v4.1 review, report SHA256
`abb1cd1fff68dab988675921bece3de27df1a5b3539bb8c93132a57529c2f615`.
This repeats only the natural domain for resource/generation training support;
there is no second shared light capture or new cross-domain timing claim.

## Second independent modality capture

The repeat uses
`small-startup-training-patch-v4-20261028-capture02-prepared01/modalities`.
Preparation reuses the unchanged frozen source/input references and verifies
that each domain's context identity matches its first capture. Old raw data and
large source snapshots are not copied into the repeat.

The second metadata capture completed with raw SHA256
`a7e243a9c2138ee21cc087934cff3e63d9c1597192b64482e74cee60141fbe4b`.
Its original v4.1 strict review remains FAIL: median boundary residual
199.66 ms causes clock and affine-binding failures. Independent host v4.1
review passes. The separate conditional causal/session-origin review passes,
SHA256 `716bc5a9948d70a5cf1d63a04ab1926de2fd1a2bbb1ceb96530068eb47b1ebef`.
Every one of the 21,102 original phase anchors remains. Their source-ordered
causal intersection is [6074687123584130, 6074687123592808] ns; the SQLite
session origin 6074687123592406 lies inside. The 8,678 ns intersection width is
not a calibrated GPU timestamp error bound. This is a new conditional
constant-offset lineage, with the original failed review and parsed file
preserved.

The repeat light-kernel and natural v4.1 strict reviews both pass directly,
with report SHA256 respectively
`1a2469e3c611d13e40fa6309b28d22021a6a8cd9cbb1718787b39930743d9c02` and
`b1535c9c2cfb6bd8544dac748df8431ea4cc7146187d13b6000fdf1193d7a209`.
The natural short/long image handlers take 158.56/630.33 ms and short/long ASR
handlers 493.86/1236.48 ms. Their actual generated-token counts are again
4/3/21/55. These remain the second training capture, separate from independent
prediction validation.

The second three-domain bridge also passes, report SHA256
`b38ee62de62cdf29c93881c193c0b8879071ee7ff44c506e6526c8bf01f1988b`.
All four requests match actual work; all 87 semantic kernel groups, 184,776
light kernels and 22,151 affine links match exact observed launch signatures
and order. Thus two independent modality training captures are available.
Neither their fit acceptance nor held-out accuracy is inferred merely from
having two captures.

## Additional ChatQA shape training

`chatqa-new-shape-training-v4-20261028-prepared00` adds one solo text#0 input
per domain while retaining the four original residents and the unchanged
12-request priming cohort. The selected input is explicitly additional
training, not held-out validation. Its new wrapper/selection sources enter the
execution context honestly, rather than pretending to share the original
context identity.

Both metadata and light strict reviews pass directly. The GPU observations
confirm 963 input tokens, six actual generated tokens, and 23,312 kernels in
each domain. Metadata's boundary residual is 1.183 ms; light's is 49.1 μs.
Their handler durations are respectively 747.45 and 492.18 ms. Report hashes:

- metadata: `f8923d510b9fff80b41a969221c102cb14fd3ff9f01e42db7cda4d98233b9412`
- light: `902b6707a6c2d01589a89a71d177a60ea05fbff7a45086f6208bae32b5af0e81`

This is one new-shape capture. It can test prior structural/grid hypotheses
against actual kernels; it does not itself satisfy a two-capture timing fit or
demonstrate independent prediction accuracy.

Natural strict review also passes directly, SHA256
`6260e50d894cb6fb2fd199d2a22a92e4f6cf86c576969799673dc2be58dac827`;
its actual handler duration is 399.75 ms with the same 963/6 token work. The
three-domain bridge passes, SHA256
`13c671aa3911da52651f34b148b1415d6986f8fe126be8ed32afc3be84437fa2`:
all seven semantic groups, 23,312 kernels and 10,758 affine links match exact
observed work, launch signatures and order.

The separate CPU template comparison uses the actual 3,829-kernel prefill.
All 3,220 stable-grid slots and the one boolean-reduction formula match the
predeclared structural rules. Nearest/lower template policies match 3,381 of
3,829 slots; the upper policy matches 3,284. All 448 nearest/lower mismatches
belong to explicit dispatch-unknown domains, chiefly actual split-K 12/new
Ampere 128x32 or 128x64 paths and 64 CUTLASS switches to the long template.
Artifacts are
`predictor-chatqa-template-actual963-comparison-v4-20261028-final01/report.json`
and `predictor-chatqa-template-actual963-pattern-review-v4-20261028-final01/report.json`.
This is a changed-context structural diagnostic with no durations or refit;
unknown dispatch is not promoted to a successful latency prediction.
