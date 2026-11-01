# USHER / Roomie physical-profile preparation

This is a CPU-only preparation packet. It starts no GPU collection, changes no
permissions, fits no predictor and reads no held-out execution outcomes. The
machine-readable report is [report.json](../artifacts/usher-roomie-profile-readiness-20261028-cpu01/report.json).
The main v54 result remains three complete forecast recipes and an unsupported
high-load same-context reacquisition prediction. This packet does not repair or
qualify that forecast.

## Evidence available without new hardware counters

The original four-owner balanced light training SQLite was opened with
`mode=ro`. It contains 1,408,534 CUPTI kernel rows and 34,305 GPU context-switch
events, plus CUDA runtime correlations, NVTX, copies, streams, grid/block sizes,
registers and shared-memory launch attributes. It has no sampled metrics table.
The [SQLite inventory](../artifacts/usher-roomie-profile-readiness-20261028-cpu01/existing-sqlite-capability.json)
binds the actual file hash, schema, row counts and GPU UUID. The recorded board
is the RTX 4090, UUID `533a287b-0e15-a7c8-8e58-2b21bdcbb13a`; its static memory
bandwidth field is a device specification, not measured request DRAM traffic.

These sources support kernel lifetime, matching, launch gaps, phase alignment,
exact union of observed kernel intervals and cross-owner lifetime overlap.
Kernel lifetime overlap does not prove simultaneous execution on the same SM.
GPU-context residency is a scheduler envelope, not active SM service. Static
occupancy upper bounds and FLOP/byte roofline estimates can propose complementary
operators, but cannot certify measured compute-versus-DRAM bottlenecks.
NVML power and owned-process memory provide physical energy/memory evidence;
coarse utilization is neither occupancy, DRAM GB/s nor L2 hit rate.

Installed `nsys profile --help=gpu` advertises GPU metrics device/set/frequency
switches. This is CLI support only. No metrics device query, capture or counter
permission reprobe was run. Historical NCU failed with `ERR_NVGPUCTRPERM`; current
access remains `UNKNOWN_NOT_REPROBED`. Nsys metrics require a real capability
check and must not be presented as a permission bypass. No existing controlled
compute/memory stressor collector was found. Such a backend could supply resource
sensitivity evidence, but is an implementation gap rather than an existing
executable alternative.

## Original four-owner profile entry

`kernel_profile_campaign_v2` preserves `text#0`, `text#1`, `generalist_2#0`,
`ocr#0` as four distinct resident physical owners. The historical prepared00
plan now fails its source audit because `gsdeploy/forecasting/__main__.py`
changed. It must not be executed. A fresh CPU-only source freeze is available at
`artifacts/usher-roomie-profile-readiness-20261028-cpu01/original-four-owner`.
All three split audits passed at preparation: one smoke job, 488 training jobs,
488 validation jobs. This is a full prepared matrix, not a recommendation to
launch all jobs without a bounded staging decision.

The matrix covers single text/G2, both specialists, either specialist plus G2,
both specialists plus G2, and all four; G2 text/image/ASR paths; short and long
public input strata; zero, 50 ms and 200 ms offsets, with reversed owner order.
Training uses two repetitions. Validation contains new trials on known inputs
and exact-content-disjoint inputs. Actual token/crop/stage dimensions must be
recorded; file size or text characters do not stand for GPU resource demand.

CPU command (safe to repeat while the frozen sources still match):

```bash
python -m tools.predictor_validation_20261027.kernel_profile_campaign_v2 audit \
  --run artifacts/usher-roomie-profile-readiness-20261028-cpu01/original-four-owner/smoke
```

Exact existing GPU entry, **not executed by this packet**:

```bash
python -m tools.predictor_validation_20261027.kernel_profile_campaign_v2 execute \
  --run artifacts/usher-roomie-profile-readiness-20261028-cpu01/original-four-owner/smoke --execute
python -m tools.predictor_validation_20261027.kernel_profile_campaign_v2 verify-smoke \
  --run artifacts/usher-roomie-profile-readiness-20261028-cpu01/original-four-owner/smoke
```

Replace `smoke` with `training`, then `validation` for subsequent existing entry
points. Their gates require smoke success and completed training respectively;
the external workflow must also freeze the fitted predictor before validation.
These commands produce profiler-perturbed timings. The underlying
`background_collect.collect(..., trace=False)` supports natural timing, but the
kernel-campaign CLI hardcodes profiling and supplies no separately qualified
paired natural campaign. A new paired preparation/execution gate is needed;
calling the underlying collector directly would bypass the campaign gate.

This old collector also needs the reviewed disk budget and outer process-group
supervision before execution. A new source freeze does not supply those controls.
The four-owner matrix is default shared execution, with no admission permit;
it cannot claim whole-request admission interference reduction.

## Actual shared-text batch backend and prepared controls

`gsdeploy.backends.shared_text_batch.SharedTextBatchExecutor.execute_prepared`
actually executes a single HF `generate` over padded rows, with one ChatQA model
weight owner and exact greedy generation compatibility. G2 and OCR use their own
observed worker processes and real scheduled requests under batch collector v3.
The correct text interpreter is
`/root/autodl-tmp/gsdeploy-assets/envs/compact-standard/bin/python`; G2 uses
`compact-mini45/bin/python`, OCR uses `chandra-native/bin/python`, as bound by the
registry. Text environment requires `CUDA_VISIBLE_DEVICES=0` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Source/spec audits pass without
loading weights. This verifies prerequisites, not GPU compatibility or quality.

The prior prepared02 inputs are unsuitable as a direct batching efficacy test.
The exact frozen prompt builder plus local tokenizer, with CUDA hidden and
`torch.cuda.is_initialized()==False`, produced:

| Split | Input token counts | Existing bucket assignment |
| --- | --- | --- |
| Training | 414, 1054, 1744, 3940 | 1024, 2048, 2048, 4096 |
| Validation | 664, 1474, 2472, 4502 | 1024, 2048, 4096, overflow |

All four old validation inputs have different buckets, so FIFO cannot batch any
pair. Two 4502-token rows would exceed its old 8192 padded-token budget anyway.
The [CPU tokenizer receipt](../artifacts/usher-roomie-profile-readiness-20261028-cpu01/cpu-token-lengths.json)
and executable script preserve this finding. Tokenization does not load a model.

Three fresh CPU plans use the same public samples, arrivals, G2/OCR background,
priming, buckets and 16384 padded-token ceiling:

| Plan directory | Maximum batch | Formation wait |
| --- | --- | --- |
| `single` | 1 | 0 |
| `batch2-wait10ms` | 2 | 10 ms |
| `batch2-wait100ms` | 2 | 100 ms |

Each split has eight text requests, two G2 and two OCR requests. Each text input
has an explicit pair of distinct request identities, enabling a compatible pair
at every length. Repeating an input is an offered-workload design; the backend
still starts a fresh generation cache and performs no prefix-cache reuse.
Training/validation contents remain disjoint. The collector audits all three
plans successfully. Actual batching depends on serial preparation time and
deadline lateness; require actual batch membership receipts, rather than treating
`max_batch_size=2` as proof that a batch happened.

Exact existing training entry, **not executed by this packet**:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/root/autodl-tmp/gsdeploy-assets/envs/compact-standard/bin/python \
  -m tools.predictor_validation_20261027.batch_profile_collect_v3 collect \
  --plan artifacts/usher-roomie-profile-readiness-20261028-cpu01/single/plan.json \
  --split training --execute
```

The report stores structured argv/environment for each variant; change the plan
directory for the other controls. Validation must follow predictor freeze, not
be used to choose waiting parameters. These are 1-second, twelve-request finite
pilots with a separate 300-second drain and 2400-second supervisor. They cannot
estimate steady-state P95 or final six-term utility.

Batch v3 has exclusive output, source binding, complete cohort accounting,
checkpointing, experiment lock and whole-process-group timeout. It still lacks
the required 18 GiB preflight, 13 GiB free-space floor and 6 GiB incremental disk
watchdog. The main-gate v2 supervisor accepts v55 prediction campaigns only;
it cannot be reused directly for this batch experiment. A reviewed wrapper is a
real execution prerequisite, not a user authorization question.

## Prediction qualification and fair mechanism comparison

First qualify original four-owner slowdown and stage/phase alignment on training
captures and matched natural runs. Fit only training data, freeze exact models,
source closure, thresholds and candidate domain, then validate complete arrival
timelines, queue occupancy, per-request completion/P95, DeltaU sign, rank error
and selected-policy regret. Preserve all offered failures/censoring. Explanatory
operator labels without physical counters remain structural hypotheses; paired
coexecution outcomes can still verify whether their selected candidates help.

Next compare shared execution versus actual whole-request admission with exact
permit grant/release timestamps and matched arrivals. Resource shares require
hardware support and measured effective enforcement. A plan setting alone is
not a resource allocation experiment. Kernel offsets can test Roomie-style phase
alignment, but require matched unprofiled evidence before attributing throughput
or latency gains to alignment.

The batch mechanism changes four physical weight/process owners to three. The
batch-size-one control isolates that ownership/residency change from batching;
max-two then measures the incremental batching mechanism, including formation,
padding, shared tail, extra host work, startup, energy and memory costs. It must
collect its own startup, memory, energy, capacity and predictive host/phase
profiles. None can inherit the four-owner predictor or `g(n)` calibration.

The final four groups still require the same full57 offered cohort, arrivals,
inputs, output-quality rules, six utility terms and original stability rules:
original execution with/without diversion and optimized execution with/without
diversion. The optimized no-diversion comparator must be searched in its own
fair feasible domain. No full57 route/scoring/lifecycle collector for the new
three-owner batch deployment is qualified yet. The finite pilot is not one of
these final four groups.

The existing maximum-common-rectangle `shared_linear` implementation is CPU
NumPy only. ChatQA INT8/BF16 plus adapter and G2 GPTQ4/FP16 have different bases
and quantization semantics; no cross-model shared GPU matrix execution has been
certified. This remains a separate backend task. Ordinary same-model batching,
shared weight residency, prefix KV reuse and measured L2 cache reuse must remain
distinct claims, with the latter two currently unimplemented/unmeasured here.
