# Admission2 training and quality preparation

The CPU preparation reserves two training seeds, 628101 and 628102, for each
of balanced, skewed, spill_low and spill_high under the fixed global max_active
of 2 and per-alias max_inflight of 1. Each run has H600, drain300, warmup0 and
1200 public requests. The fixed residents remain text#0, text#1, generalist_2#0
and ocr#0. The campaign contains exactly eight jobs and 9600 offered requests.
No GPU job was launched by this preparation.

The new admission_training_v1 mapper retains the original worker-context
identity and separately binds the admission controller/runtime configuration.
It verifies every training role and native source against the complete frozen
campaign; it does not call a shared-only collection mapper or relabel validation
outputs. Service observations, full phase history, owned telemetry, startup,
measurement and drain energy remain distinct. Complete collection does not
override the original stability checks. The mapper explicitly leaves quality
unaudited and six-term evidence incomplete.

The independent admission_quality_v1 consumer closes that specific future
quality-evidence path. Its load_runs first invokes the strict admission loader,
which maps all eight native training runs. Each scored-raw.json must bind its
original raw.json by source receipt and reproduce every native field unchanged.
The unchanged verify_quality_annotation independently scores original output
against the frozen catalog and rejects any altered or incomplete annotation.
Native raw is never changed. An explicit validation request is rejected, as are
pilot/final/validation job wrappers, foreign sources and changed controllers.

For every exact route and controller, the quality model sums native error
numerators and reference denominators by family/task, then divides those sums.
OCR and ASR do not average individual request error rates. Full models use both
training seeds. A whole-seed fold excludes the heldout seed across all four
routes and uses the remaining seed of that exact route only. There are twelve
model documents: four full and eight heldout. No shared or cross-route fallback
is available. Prediction reproduces the original family/task aggregate proxy
within each public routed task and then the original offered task-rate weighted
Q. Missing positive-weight support makes the task and overall Q UNKNOWN.
This is a training-distribution proxy, not a claim that stochastic outputs are
concurrency invariant. Repeated inputs do not become independent examples.

Public API:

- admission_training_v1.prepare(output), verify(campaign_path), and
  map_training(job_or_job_id, campaign_path).
- admission_quality_v1.prepare(campaign_path, protocol_file),
  verify_protocol(protocol_file), fit(protocol_file, output_dir),
  verify_manifest(manifest_file, recompute=True), and
  predict(model_dict, public_plan, public_candidate).
- A fitted quality manifest exposes full.route_quality[candidate_id] and
  folds with omitted_seed, training_seeds and route_quality source receipts.
- A quality prediction contains status, Q, per_task, missing_support and the
  exact model/controller/route receipts. It accepts no target outcome input.

Artifacts under artifacts/predictor-gpu-20261027:

- admission-training-v1/campaign.json and protocol.json freeze acquisition.
- admission-training-v1/cpu-test-report.json records 29 passing CPU tests.
- admission-quality-v1/protocol.json freezes quality fitting prospectively.
- admission-quality-v1/cpu-test-report.json records seven passing quality CPU
  tests, including the real native mapper/quality scorer on eight complete
  1200-request CPU fixtures, annotation tampering, source/role isolation,
  whole-seed separation, ratio-of-sums CER and offered-rate weighted Q.

The real campaign has not been fitted because its native training outcomes
have not yet been collected. The quality protocol is prepared without reading
real training or independent outcomes. All future training startups must occur
after the protocol timestamp. No execution approval or forecast seal is
implied by a preparation artifact.

This preparation is compatible with admission-next-stage-roles-v1.json:
training T uses seeds 628101/628102, independent V uses 628201, and the final
diagnostic block F uses 628301 with both execution modes and four routes.
Training fits mode/route resources, capacity and quality only. The previously
fixed Gamma shared-service transfer remains a frozen hypothesis; it is not
refitted using these T data or any V outcomes. Independent failure remains a
failure. One final block is descriptive, without a significance claim. Final
candidate/order and prospective forecasts need their own later seals.
