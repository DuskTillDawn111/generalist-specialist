# Admission training continuation draft

Status: the reviewed design is implemented by the new independent module
tools/predictor_validation_20261027/admission_training_continuation_v1.py.
This document does not launch jobs, produce an execution review, seal
predictions or replace the main thread's source review.
The user has already authorized the finite study. The review below is an
internal evidence dependency, not a request for additional user permission.

## Frozen inputs

All paths below are relative to artifacts/predictor-gpu-20261027:

- admission-training-v1/campaign.json: T seeds 628101 and 628102, each with
  admission2 balanced, skewed, spill_low and spill_high; exactly eight jobs.
- admission-capacity-v1/protocol.json, admission-resources-v1/protocol.json and
  admission-quality-v1/protocol.json: all three bind that exact campaign.
- admission-next-stage-roles-v1.json: finite disjoint T8/V4/F8 roles; fixed
  shared Gamma service-law transfer hypothesis and unchanged thresholds.
- The complete selected-v16 stage2 report and its report-bound gate.
- The all-three-mode H600 admission-pilot stage3 report.

The current campaign SHA is
19a7b1e5b999f6ff96fa37e867e2485bd76f8d534a7e3c7e07d9d24bfc3a4504.
Exact receipts must be recomputed and verified when authoring the review;
this prose is not a replacement for source-bound artifacts.

## Main-thread review artifact

The implemented continuation exposes prepare(output), audit_protocol(path),
and the explicit run(protocol_path, root_review_path). Root conditional
authorization uses schema gsdeploy.admission_T8_continuation.v1.main_thread_review,
decision APPROVED_FOR_CONDITIONAL_TRAINING_ONLY, reviewer_role main_thread,
the protocol/campaign/three mode protocol/role_reservation/source receipts,
created_ns after the continuation protocol, a nonempty
stage3_incomplete_training_reason, and final_gain_eligible False. The wrapper
waits for original PIDs 79375/79980, verifies their Linux start ticks, cwd and
argv, rejects duplicate upstream processes, and never launches predecessors.
After both completion artifacts, it independently recomputes the selected
gate and pilot report once, checks exact source-bound equality, captures their
transitive JSON source closure, and materializes the training review described
below. Each job checks that immutable materialized review and complete source
closure again. The three fit stages repeat closure checks before and after.
The wrapper exposes no V/F launch.

Wait for the complete stage2 report and its gate before issuing a training
review. admission_training_v1._review requires four observed candidates with
1200 known requests each and a report-bound PASS or FAIL gate. It rejects an
incomplete gate. A genuine FAIL remains FAIL and can authorize diagnostic
training only.

The stage3 report must explicitly contain shared_spill_low,
admission1_spill_low and admission2_spill_low. Its DIAGNOSTIC_COMPLETE or
DIAGNOSTIC_INCOMPLETE result remains unchanged. If incomplete, the main-thread
review must include a nonempty stage3_incomplete_training_reason explaining
why the finite training acquisition remains justified and preserving each
failure or missing item. A partial run does not become a complete run.

After inspecting both reports, the main thread may write an exclusive review
artifact with the exact fields consumed by the frozen mapper:

```python
{
    'schema': training.SCHEMA + '.main_thread_review',
    'campaign': receipt(campaign_path),
    'decision': 'APPROVED_FOR_TRAINING_ONLY',
    'reviewer_role': 'main_thread',
    'created_ns': time.perf_counter_ns(),
    'stage2_report': receipt(stage2_report_path),
    'stage2_gate': receipt(stage2_gate_path),
    'stage3_report': receipt(stage3_report_path),
    'final_gain_eligible': False,
    # Required if stage3 is incomplete:
    'stage3_incomplete_training_reason': 'Concrete reviewed reason',
    # Additional source-bound continuation requirements:
    'capacity_protocol': receipt(capacity_protocol_path),
    'resource_protocol': receipt(resource_protocol_path),
    'quality_protocol': receipt(quality_protocol_path),
    'role_reservation': receipt(role_reservation_path),
    'source_receipts': [...],
}
```

The eventual continuation must verify all three extra protocols using their
own verify_protocol APIs, verify they bind the review campaign, and verify the
exact role reservation and fixed service-model receipt. The frozen
training._review checks the original required report/campaign fields; it does
not itself validate the additional protocol/reservation fields. Therefore the
new continuation must validate these explicitly before any execute call.
Its source must also be included in the main-thread review and frozen before
launch. No frozen training mapper change is necessary.

## Sequential finite collection

CPU readiness checks must validate the campaign, all protocol/source receipts,
the review and its referenced reports, and the exclusive worker-lock path.
Every protocol timestamp must precede every training startup. Retain exact
receipt identities throughout collection, not merely their current contents.

Use campaign.jobs in its already frozen order. Never sort by route, seed,
predicted benefit or observed outcome. For each new job, invoke only
training.execute(campaign_path, job_id, review_path). That function checks all
earlier jobs, exclusively creates its attempt marker before launching the
unchanged native collector, then strictly maps the completed native raw and
writes training-complete.json. No independent validation or final collection
is launched during T8. Each run retains largest-three priming, warmup0,
H600/drain300, the public full trace and fixed residents.

If any launch, native collection, shutdown, telemetry or strict mapping fails,
preserve the attempt marker and all outputs and stop this finite continuation.
Do not remove outputs, retry, replace seeds or silently skip a job. The later
finite diagnostic path must report the missing support. A successfully drained
run with an original stability failure can still remain a measured training
observation; its failed stability status remains in mapping_report and does
not grant capacity or deployment qualification.

A continuation resumed after a process interruption may skip only jobs with
verified existing training-complete.json packets, exact campaign/job/receipt
identity and a fresh successful strict map. An attempt without a verified
completion packet is terminal for that job, not an implicit retry allowance.
Partial fit outputs likewise must never be overwritten.

## CPU fits and cross-validation after all eight runs

Do not fit from a partial campaign. Each consumer independently requires all
eight original training native raws through the strict admission mapper. Save
fits into new exclusive directories, for example fitted-v1 under each protocol
directory, using these APIs:

```python
capacity.fit(capacity_protocol_path, capacity_output)
resources.fit(resource_protocol_path, resource_output)
quality.fit(quality_protocol_path, quality_output)
capacity.verify_manifest(capacity_output / 'manifest.json')
resources.verify_manifest(resource_output / 'manifest.json')
quality.verify_manifest(quality_output / 'manifest.json')
```

capacity and resources are admission_capacity_v1 and admission_resources_v1;
quality is admission_quality_v1. Run these CPU stages with GPU disabled and
bounded BLAS/OpenMP threads. Record all three manifests, tests, training native
raw and scored-raw receipts, and completion/stability/source audit reports.

Resources and quality each provide four route-specific full models and eight
whole-seed fold models. Capacity preserves the original g(n)/n and family rho
arithmetic; max_active=2 never substitutes for resident model count. Fitting
capacity does not prove global-permit capacity. Every missing service input,
resource context or quality family/task remains UNKNOWN without shared or
cross-route fallback. Quality recomputes every scored output from native raw,
uses ratio-of-sums error units and preserves original offered-rate weighting.

No service model is fitted in this continuation. The fixed full16 Gamma model
specified by the reservation remains the transfer hypothesis. Heldout training
seed observations may score their fold only; they may not enter its resource,
capacity or quality fit. Fold errors cannot select a new service law, tune
thresholds or authorize another reserved independent seed.

## Handoff to the next phase

The handoff reports complete/failed/missing training support and the actual CV
results, plus exact manifests and source receipts. All acquisition and fit
artifacts retain final_gain_eligible=False. Preserve the stage2/stage3 failures
alongside them.

Only after CPU source review and all supported fits are available should the
separate admission_validation_v1 interface seal prospective predictions for
seed 628201, then collect its fixed four routes. The validation law is frozen
before its first startup and V outcomes never enter these T fits. A V failure
remains a failure. Final seed 628301 then supplies one descriptive eight-config
block under its separately frozen final policy/order and execution seal. The
locked balanced/high contrast and strongest observed no-spill baseline remain
distinct from the post-observation allow envelope; no envelope is called a
deployable gain or a globally optimal search result.
