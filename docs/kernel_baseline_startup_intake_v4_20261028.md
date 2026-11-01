# Startup-aware real kernel baseline intake v4

`tools/predictor_validation_20261027/kernel_baseline_intake_v4.py` is a new CPU
adapter for passing `small_startup_three_domain_bridge_review.v4` reports. The
v3 source and all prior artifacts are preserved. It uses the same original
five-feature estimator and exact observed metadata/light launch mapping.

A G2 image request has user-request batch 1 and may have observed language
forward batch 3 because its actual generation uses three beams. The adapter
uses **semantic forward batch 3** as the affine feature, preserves the original
request execution batch 1, and records both meanings plus the effective beam
count in its new execution key. It requires actual semantic metadata, actual
input/output leading dimensions and the observed effective generation setting;
tensor shape alone cannot invent batch semantics. Unexplained batch expansion
is rejected. Original context IDs are not rewritten.

`G2-solo-image` and `G2-solo-asr` are accepted only with a single exact alias,
task, input and zero start offset, plus disjoint actual worker intervals across
measured **and priming** requests. Foreign-process telemetry still causes
rejection; sampled absence is not an absolute exclusion guarantee.

The first real modality capture produced 22,151 accepted affine observations
and zero adapter rejections: 2,275 at semantic batch 3 and 19,876 at batch 1.
The four measured requests contain 184,776 kernels, so this affine coverage is
a strict subset. Uncovered kernels remain unknown. A single-capture fit gave
2,022 groups and **zero supported duration groups**, as required by the original
minimum of three observations and two independent raw captures. Maximum
relative training residual remains 0.25; a training residual is not a confidence
bound or independent prediction error.

Current artifacts:

- `artifacts/predictor-real-kernel-baseline-intake-v4-20261028-modalities-capture01/intake.json`
- `artifacts/predictor-real-kernel-baseline-fit-v4-20261028-modalities-capture01/fit.json`

The fit retains the existing v3 statistical document schema, with explicit
`intake_schema=gsdeploy.real_kernel_baseline_intake.v4` and current source
receipts. Every row retains its actual raw run identity, original complete
light-domain context, profiler version, SQLite source and isolation evidence.
There is no measured-target fallback, natural timing rescaling, output-derived
validation feature or per-kernel peak-memory target.

The second independent modality capture is now ingested. The combined fit uses
44,302 actual observations: 1,137 of 2,022 groups have a supported duration
model; 868 retain insufficient-observation/run UNKNOWN and 17 fail the unchanged
0.25 training-residual gate. Supported groups contain 40,050 of the accepted
affine observations. This denominator excludes the many non-affine request
kernels and is not whole-request coverage or prediction accuracy.

- `artifacts/predictor-real-kernel-baseline-intake-v4-20261028-modalities-capture02/intake.json`
- `artifacts/predictor-real-kernel-baseline-fit-v4-20261028-modalities-two-captures/fit.json`

The existing exact-source baseline query bridge can hydrate this statistical
fit without modifying v3 source or any context hash. A CPU query smoke sampled
every 73rd accepted launch from the two-capture index: 607 exact queries,
548 regression predictions and 59 UNKNOWN, spanning semantic batch 1 and 3,
with no measured-duration fallback. This is training-profile support only;
it does not establish an unseen-input feature/sequence mapper.
