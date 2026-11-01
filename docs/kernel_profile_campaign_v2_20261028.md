# Original four-owner kernel profiling campaign v2

This campaign retains **four separate physical owners**: `text#0`, `text#1`,
`generalist_2#0`, `ocr#0`. It does not substitute the new three-owner text batch
deployment. All commands below must run from the repository root. Preparation,
source audit, tests, and trace export do not load a model.

The exclusive campaign path is
`artifacts/predictor-gpu-20261027/kernel-profile-campaign-v2-20261028-prepared00`.
Its `campaign.json` binds the execution identity and the three run manifests.
All current package Python sources, retained native sources/specs, public pool,
and selected media payloads are copied and hashed. Any executable or input drift
blocks collection. A new preparation directory is necessary after intended
source changes; existing outputs must never be overwritten or retried.

## First GPU command: correctness and clock smoke

```bash
python -m tools.predictor_validation_20261027.kernel_profile_campaign_v2 audit --run artifacts/predictor-gpu-20261027/kernel-profile-campaign-v2-20261028-prepared00/smoke
python -m tools.predictor_validation_20261027.kernel_profile_campaign_v2 execute --run artifacts/predictor-gpu-20261027/kernel-profile-campaign-v2-20261028-prepared00/smoke --execute
python -m tools.predictor_validation_20261027.kernel_profile_campaign_v2 verify-smoke --run artifacts/predictor-gpu-20261027/kernel-profile-campaign-v2-20261028-prepared00/smoke
```

The smoke loads all four original models, primes each owner once on a short
training input, then schedules one four-owner simultaneous job: both text owners
and G2 process text, and OCR processes its own short image. Thus it has four
priming requests plus four measured requests. Generation policy is unchanged.
No smoke observation may enter fitting or predictor validation.

Each worker runs under its own `nsys profile --trace=cuda,nvtx` with capture
starting only after all residents load and prime. The collector uses the common
GPU experiment lock. The verifier exports four SQLite traces and requires:

- Four distinct READY worker PIDs, actual descendant ownership, and successful
  requests without collector/telemetry failures.
- Paired native/semantic host phase anchors, real correlated CUPTI kernels for
  every request, and valid independent per-process median clock mappings with
  maximum boundary residual at most 5 ms. Residuals remain in the raw report;
  this tolerance does not establish sub-millisecond cross-process order.
- Actual handler START/END, parent reader receipt, collector processing, and
  WORKER_READY after response flush. Handler completion precedes parent receipt;
  WORKER_READY precedes its parent receipt. Parent RESULT receipt and worker
  next-ready may occur in either order. Neither substitutes for the other.
- Board power and NVML samples that include all four actual worker PIDs and no
  foreign GPU process. Peak memory is explicitly sampled owned-process memory,
  not an instantaneous peak or the sum of independently measured peaks.

This is default sharing: there is **no admission permit**, so permit release is
explicitly NOT_APPLICABLE. Subsequent admission trials must supply actual gate
grant/release events. Nothing sets these absent events to zero.

## Training and independent validation

Two training repetitions cover single-owner activity, both specialists, either
specialist with G2, both specialists with G2, and all four owners. G2 text,
image, and ASR activity is represented; OCR background remains a separate owner.
Each task has four training inputs and four exact-content-disjoint validation
inputs spanning short and long inputs. Text strata use context-plus-question
character length, ASR uses duration, and image/OCR use file size; actual token,
crop, and execution-stage observations remain necessary and these proxies are
not labelled compute demand.

Paired/triple/four-owner jobs use simultaneous starts and 50/200 ms staggered
starts in both owner orders. Training and validation use independent fixed
shuffle seeds. Validation has two explicit views: new trials on training input
identities, and new exact input content. Validation trials are not fitting data.
Only training inputs are used in priming, including before validation. No
unseen-task/generalization accuracy is claimed.

Training cannot execute until the smoke report is PASS. Validation cannot
execute until training collection completes. The main workflow must additionally
freeze fitted predictor/model receipts before validation is inspected; this
collector produces raw evidence and does not itself fit or qualify a predictor.

```bash
python -m tools.predictor_validation_20261027.kernel_profile_campaign_v2 execute --run artifacts/predictor-gpu-20261027/kernel-profile-campaign-v2-20261028-prepared00/training --execute
python -m tools.predictor_validation_20261027.kernel_profile_campaign_v2 execute --run artifacts/predictor-gpu-20261027/kernel-profile-campaign-v2-20261028-prepared00/validation --execute
```

These are profiler-perturbed kernel observations. They cannot establish natural
service time, P95 or end-to-end ΔU without separately matched unprofiled
execution evidence. SM/DRAM/L2 counters remain UNKNOWN due to the previously
observed ERR_NVGPUCTRPERM; kernel timing/NVTX labels do not establish resource
demand, complementary compute/memory behavior, or cache reuse. G2 OCR is outside
this initial campaign; the OCR background here is the resident OCR specialist.

CPU contract verification: `python -m unittest tests.test_kernel_profile_campaign_v2 tests.test_native_profile_collection tests.test_nsys_trace_export -q`.
