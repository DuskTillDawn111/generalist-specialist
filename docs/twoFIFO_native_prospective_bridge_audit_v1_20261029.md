# Native prospective bridge: reusable evidence and remaining boundary

This audit is CPU/public/training-source only. No target future execution labels,
new GPU work, fit or change to frozen producers is involved. A different capture
ID does not by itself imply a different execution backend. Matching backend
identity does not by itself establish predictive accuracy under a changed load.

## Existing reusable functions

| Concern | Existing implementation | What it establishes |
| --- | --- | --- |
| Exact input content | `kernel_profile_campaign_v2.input_identity`; `twoFIFO_training_semantic_preflight_v2.source_catalog`; `twoFIFO_prospective_contract_v1.public_input_manifest` | Text context/question or media bytes, task/pool binding; a supported task or an identical filename does not substitute for matching content. The training `source_catalog` intentionally accepts only its original16 catalog. |
| Public plan and frozen execution | `twoFIFO_source_rebound_execute_v1.audit_campaign` / `audit_plan`; matching producer's separate audit | Native worker commands/specs, declared inputs, source closure/frozen bytes, runtime policy/config, residents, priming and plan identity are rederived. A planned executable binding does not prove that a target run actually launched those processes. |
| Physical axes across captures | `twoFIFO_resource_provider_v3.physical_execution_payload` / `physical_execution_identity` | Retains whole context, physical runtime knobs, policy, residents, priming, sources/freeze. Excludes offered rates, routing, seed, windows and the plan/capture/attempt/execution IDs. Extra runtime knobs cannot silently escape this identity. |
| Executable-content hash | `native_collect.context_identity` | Retains context plus original source paths and frozen content hashes; excludes the redundant receipt lists. It is not a general backend-equivalence normalizer. |
| Actual priming content | `natural_macro_resources_candidate_v1.priming_identity` | Hashes ordered actual successful18 `(model,task,pool_id,exact_input_sha256)` entries, excluding request IDs. Requires authenticated native raw/context; planned18 alone cannot supply this proof. |
| Causal settled state | `twoFIFO_settled_history_initial_v1.from_snapshot` / `checked_seed`; `extract` for its supported producer | Checks complete resident/producer roster, owned lifetime and all clocks <= cutoff <= origin, processed READY, registry and no queued messages/errors. Relative sequence counters remain explicitly conditional; `from_snapshot` does not authenticate the capture source. Its `extract` accepts the original v3 schema, not a new matching schema relabeled as v3. |
| Native macro/fit provenance | Source-rebound/recovery intake and `with_verified_bundle`; `twoFIFO_source_rebound_joint_law_v1.build_joint_law` | Full exact12 source review, whole-request paths, all macros, donor block membership, phase12 root/folds and native provenance. Current exact12 producer cannot ingest the separate15-attempt matching campaign. |
| Public request-vector selection | `twoFIFO_generated_wholeblock_evaluation_v1.compile_case_draws`, `twoFIFO_joint_request_law_v4.choose_vector` / `compile_draw_v4` | Uses owner/task/exact input and public arrival ordinal, joint vectors with common block latent, paired transport feasible sets and all world weights. Repeated public requests select from donor vectors, not future target traces. |

## Direct CPU findings

The existing `audit_campaign` passes for the original12 CPU-prepared source-bound
plans. Their physical execution identity is identical across all three captures:

`62d4b7319d78a84e2cfd812b9862853dc71a6526574f3a49a1bb2e7dadf393e3`.

This proves equality of the audited **planned physical axes**, without requiring
capture/attempt/execution IDs to match. It neither asserts that all12 runs have
completed nor supplies a fitted law. Existing independent native reviews remain
the authority for completed training data.

The original181 historical public flow has arrivals/routes and a historical
context hash, but no full fresh executable plan binding its runtime config,
context receipt, exact priming, owned worker launch and frozen source closure.
It cannot be passed off as the richer training plan expected by physical identity.

The matching candidate context differs from its parent only in declared inputs
and three explicit context/provenance fields (`matching_context_sha256`,
`matching_macro_supplement`, removal of inherited `source_rebound_context_sha256`).
The worker bindings, threads, device and protocol bytes are unchanged. Its input
catalog expands16 to22. This establishes equality of those listed fields, not
equality of the entire context or an automatic statistical transfer certificate.

The matching execution producer inserts attempt-specific `request_id` into its
planned priming rows. Existing physical identity retains whole planned priming,
so it can treat these provenance-only differences as unequal physical payloads.
Do not edit the frozen resource-v3 identity in place or silently strip arbitrary
keys. A producer-specific bridge needs an explicit priming-content projection,
checked against its own plan schema and, once available, actual reviewed priming.
The existing actual `priming_identity` already shows the intended content axes.

## Minimal next verifier interface

A narrow `verify_native_forecast_source_bridge` should take immutable receipts
for the audited donor training bundle/fit/law, an explicit fresh executable plan
and context, the original public input manifest/calendar mapping, and an owned
immutable pre-cohort prefix. It should return **separate** results rather than
one caller-supplied qualification Boolean:

1. `executable_semantics`: source content/spec/checkpoint/device/threads/protocol,
   runtime knobs, policy, full resident roster and declared priming content.
   Cross-capture IDs stay provenance. Only fields explicitly reviewed for that
   producer schema may be projected; no generic removal of differing metadata.
2. `exact_input_template_support`: rederive exact contents, authenticate every
   donor vector and block, require each routed `(owner,task,input)` in every
   selected donor block, and verify full causal request macro paths. Planned
   coverage, priming and `tasks_supported` do not count as measured templates.
3. `causal_prefix_binding`: authenticate the prefix from the actual target owned
   runtime before the first forecasted arrival, rederive a settled seed and bind
   producer lifetimes. If the target is busy, do not coerce it into a settled
   seed; current native busy continuation has separate UNKNOWN limitations.
4. `predeclared_model_hypotheses`: preserve declared donor weighting, finite
   slots, phase-rate/template stationarity assumptions and changed-schedule
   protocol/transport assumptions. A paired source feasible set is not a fitted
   latent probability law; ambiguous generated START retains its interval or
   UNKNOWN. This entry can expose an **uncalibrated** forecast hypothesis after
   source binding; independent accuracy is not a prerequisite for producing it.
5. `resource_support`: authenticate original startup/measurement/drain scopes,
   owned peak and board energy, full resident states and ordered prior history.
   An old heldout provider cannot be relabeled to accept fresh target IDs.
   Missing state/history or unsupported horizon remains UNKNOWN for M/E/fullU.

The verifier must not read target future START/READY/RESULT/output-quality labels
to release source gates. It must not accept caller `transfer_qualified=True` as
evidence. It also must not confuse lack of independent accuracy calibration with
lack of source identity: a sourced predeclared forecast is what validation tests.
Routing/deployment qualification and final gain still require the later
independent timing, per-task P95, delta-U sign/rank, actual quality, full drain
and original stability/rho/g(n) checks.

## Blocking evidence, in execution order

1. Baseline exact12 recovery is not yet complete, so no complete new native
   root/fold law is available to this bridge. No law is fabricated for CPU audit.
2. Original181 needs16 routed owner/task/input keys; the existing prepared
   support audit finds12 missing native measured keys. Matching supplementation
   is planned but not executed; union22 and intent overlap are not measured data.
3. The new matching producer needs its own verified macro/fit/joint-law adapter
   preserving15 attempts/three whole blocks and390 measured requests; exact12
   intake cannot be bypassed by relabeling the new documents.
4. A fresh181 executable plan/context and actual causal prefix are absent. The
   historical public reference and CPU draft fresh IDs are not those bindings.
5. Even after identity is proven, 90+120 horizon, changed load/phase composition,
   transport, allocator history and resource fullstates are explicit model
   extrapolations to assess. They are not resolved by matching executable hashes.

No production bridge is added here: the available evidence supports the existing
planned-axis/source audit, not a new native prospective law release. The CPU
receipt accompanies a replay script that reuses existing auditors and records
the above observed public metadata without reading native target outcomes.
