# Isolated kernel baseline estimator v2

`gsdeploy/forecasting/kernel_baseline.py` implements a CPU-only affine baseline
for an USHER-inspired feature set: batch size, weight bytes, input tensor bytes,
output tensor bytes, and FLOPs. This is a new offline module; the frozen runtime
provider and prepared profiling worker are unchanged. No GPU API or model load
is called by this module or its audit tool.

## Timing and evidence domains

Both `profiled_isolated` and `unprofiled_isolated` training are allowed. Profiled
data requires a profiler identity, `profiler_perturbed=True`, and nonempty
perturbation flags. Unprofiled data requires no profiler identity or perturbation
flags. CUDA-event or per-kernel instrumentation must be declared as profiling;
the absence of Nsight alone does not make a timing natural. Isolation requires
a source receipt describing the actual controlled execution and peer activity.

Groups are exact matches of hardware identity, operator, kernel identity,
precision, backend/execution settings, execution-context SHA-256, feature-schema
identity, timing basis, profiler identity, and perturbation flags. Thus logical
weight bytes cannot silently share coefficients with unique storage bytes;
quantization/backend settings belong in the explicit execution key. Hardware
identity must come from capture evidence, not a current-device ordinal fallback.

Every observation has unique `id`, `run_id`, `split`, `evidence_kind`, and
`measurement_source={path,sha256}`. `evidence_kind` is `measured`, or `cpu_fixture`
with explicit estimator opt-in. Each of the five features requires its own
nonempty list of source receipts. `isolation_evidence` is also a source receipt.
Feature definitions are the caller's explicit `feature_schema_id` contract.
The estimator validates receipt structure; the file audit additionally verifies
the actual source trace bytes. Receipts are provenance declarations, not proof
that an experimental isolation assertion is true.

Duration is a positive measured kernel interval in ns. Intermediate peak bytes
is a separate optional measured target with its own source receipt. No peak is
derived from weight/input/output bytes; those bytes also do not establish DRAM
traffic or throughput. A missing peak does not block duration training.

## Real input adapters

`adapt_raw_kernel_profiles(raw_rows, manifest)` accepts raw rows containing
`id`, `config_id`, `duration_ns`, optional `intermediate_peak_bytes`, and, when
peak exists, `peak_measurement_source`. The manifest schema is
`gsdeploy.kernel_baseline_capture.v2`; capture-level fields are the provenance,
hardware/context and timing-domain fields above. `configurations[config_id]`
supplies `operator`, `kernel_id`, `precision`, `execution_key`,
`feature_schema_id`, `features`, and `feature_evidence`. Missing bindings or
evidence return explicit rejected UNKNOWN rows, without guessed zeros.

`adapt_nsys_trace(parsed_trace, manifest)` uses genuine CUPTI end-minus-start
intervals. The manifest must bind `measurement_source` to an exact source receipt
in the existing `gsdeploy.nsys_observed_phases.v1` export and attest isolation.
It accepts only `profiled_isolated`. `kernel_bindings` maps decimal canonical
launch indices to `{config_id, launch_identity}`. `trace_kernel_order(trace)`
provides that deterministic order; each identity repeats `nsys_start_ns`,
`nsys_end_ns`, `pid`, `stream`, and `correlation_id`. The bound configuration's
kernel identity must match the observed CUPTI kernel name. Runtime/backend
implementation identity belongs in the configuration execution key/context.
All NVTX-owned and unattributed launches are included once; the export's
`phases` repeat some NVTX kernels and are deliberately not concatenated.
This adapter creates no memory peak target.

`adapt_affine_feature_bindings(document, operator_bindings, observer_source)`
accepts the real operator observer schema or CombinedObservation's nested
`affine_operator_observations`. Each binding keyed by observer record ID must
explicitly supply `kernel_id`, `precision`, `execution_key`, `batch_size`,
`batch_evidence`, `weight_tensor_name`, and `operator_to_kernel_evidence`.
The two evidence fields are source receipts. It selects exactly one named
weight tensor's actual logical bytes; it never sums aliases/scales as physical
storage. Input/output bytes come from actual tensor metadata. Its FLOPs feature
means the enclosing operator's declared dense GEMM arithmetic, not measured
kernel FLOPs; its fixed feature-schema identity preserves that distinction.
Module host timestamps are never kernel targets. Missing exception outputs,
batch semantics, attribution, or FLOPs remain UNKNOWN. Attention, fused,
functional, vision/ASR frontend, and unobserved operations gain no coverage.

Typical API use after real training evidence is available:

```python
from gsdeploy.forecasting.kernel_baseline import (
    KernelBaselineEstimator, adapt_nsys_trace,
)

adapted = adapt_nsys_trace(parsed_trace, training_capture_manifest)
# Review rejected_rows and preserve them in the intake report.
fit = KernelBaselineEstimator(adapted['rows'])
prediction = fit.predict(query_configuration_with_provenance)
heldout_report = fit.evaluate(independent_validation_rows)
```

`predict` requires the configuration/timing-domain/features/evidence fields from
an observation, including intended isolated-execution evidence; it requires no
measured target, measurement source, ID, run, or split. The result keeps profiler
identity/flags and reports duration and peak independently.

## Fit, support, and independent checks

The default minimum is three target observations across two runs. Numeric
features are centered/scaled and fit with SVD least squares. The maximum relative
training residual must be at most the configured limit (default 0.25).
These are explicit intake/fit criteria, not a statistical accuracy guarantee.

Collinearity does not force fabricated unique coefficients. The report exposes
the affine design rank and whether coefficients are individually identifiable.
Predictions require membership in the observed affine subspace and convex hull;
the hull check uses a nonnegative convex-combination linear program. A query
inside each feature's min/max can still be unsupported. Out-of-domain, affine
subspace, hull, or target coverage remains UNKNOWN; no ridge or extrapolation
fallback fills gaps. Memory uses its own observations and supported hull.

`evaluate` accepts only validation rows, with disjoint observation IDs, run IDs,
and raw timing/peak measurement-source SHAs from training. Shared graph/feature
sources are allowed. Evaluation never updates coefficients or support and
reports unsupported cases alongside error/coverage. `training_sha256` freezes
the input/configuration fingerprint; it is not a deployed model artifact or an
independent-validation certificate. Training residuals are not confidence bounds.

Every prediction/report leaves `natural_timing_validated=False` and
`deployment_qualified=False`. The CPU unit suite uses explicitly opted-in
synthetic fixtures and establishes no target-model prediction accuracy.

## Calibration boundary and next real evidence

Matched unprofiled stage/request anchors from training can calibrate a separate
stage response model for a fixed profiler/context and matched input signature.
Its input would be the profiled whole-stage prediction and observed stage
structure, and its target the matched unprofiled whole-stage duration. Keep
calibration anchors, repetitions, fit residuals and supported feature hull
separate from heldout natural chronological validation. A whole-stage ratio
must not be applied blindly to every kernel: host work, gaps, launches, waits
and concurrency make individual natural kernel times unidentifiable from that
ratio. Stage calibration is designed here, not implemented or claimed validated.

For occupancy metadata, the shortest future collection path is to instrument
the existing profiler process after its normal module/function load: record
actual function/module/launch identity and `cuFuncGetAttribute` values for max
threads, registers, static shared bytes, maximum dynamic shared bytes and
preferred carveout; also record actual function cache/shared configuration
calls and requested dynamic shared bytes. Read API-exposed values and retained
configuration records rather than host-constructor defaults. Partitioned-GC,
shared-limit mode, and barrier fields still require actual configuration or
compiler/module evidence where APIs do not expose them; keep missing fields
UNKNOWN. No target module is loaded just to fabricate a handle. The existing
host calculator document describes its strict input/assumption boundary.

## Actual read-only audit

The existing six-launch probe was passed through the adapter using its original
unchanged SQLite SHA. Output:
`artifacts/predictor-kernel-baseline-input-audit-v2-20261028-final/audit.json`.
All six launches remain UNKNOWN for missing isolation and feature/configuration
bindings. No rows were trained, no natural targets reconstructed, and no GPU
work performed. A later read-only occupancy review found the old probe's exact
device UUID in `TARGET_INFO_GPU`, despite NULL `TARGET_INFO_CUDA_DEVICE.uuid`;
the earlier identity-only UNKNOWN conclusion is superseded. Isolation and
operator-feature evidence are still missing. See
`docs/occupancy_declaration_resolution_v2_20261028.md`. The separate prepared01
smoke is not training evidence.

Reproduce the CPU adapter audit in a **new** directory:

```bash
python -m tools.predictor_validation_20261027.baseline_input_audit_v2 \
  --trace artifacts/predictor-capability-audit-v2-20261027/reparsed-kernels.json \
  --output-dir artifacts/NEW-EXCLUSIVE-baseline-audit
```

Supply `--manifest` for actual bound profiles; alternatively `--raw` accepts a
raw measurement JSON with a manifest receipt exactly bound to that JSON. The
tool writes a new directory and output manifest and never executes a profiler.

Validation: `python -m unittest tests.test_kernel_baseline_v2 -q` (21 CPU tests).
