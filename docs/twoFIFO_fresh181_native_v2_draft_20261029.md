# Native-v2 typed target and collector draft

This is a new CPU-reviewed draft boundary. The previous fresh181 and typed-v1
sources remain frozen. Public `native_v2.collect` and `execute` unconditionally
raise HOLD: no real GPU or validation request can be launched by this draft.
`collect_cpu` requires injected transport and clock and marks evidence CPU only.

The two new modules are `twoFIFO_fresh181_native_v2.py` and
`twoFIFO_matching_prospective_bridge_v2.py`. They solve the explicit target-schema
boundary without patching the original `fresh.audit_plan` global or asking the
old typed-v1 module to accept an arbitrary new plan.

Native-v2 preparation requires an actual immutable saved matching fit reference.
It reuses the public original181 and matching executable-plan builder, then
strictly adds the new target schema, full native-v2/typed-v2 source freeze,
saved-fit binding and a fixed300-second predeclared live barrier. Its own
manifest/auditor deterministically reconstruct every field of all four plans.
Missing actual fit or incomplete source review cannot produce an eligible
native run. No persistent native-v2 prepared artifact is claimed by CPU tests;
the tests explicitly mock the saved-native-fit boundary inside temporary dirs.

Typed-v2 has one fixed auditor registry entry:
`tools.predictor_validation_20261027.twoFIFO_fresh181_native_v2`, its exact source
receipt and native-v2 attempt schema. A caller cannot supply an auditor or
qualification Boolean. `open_target_session` audits the canonical whole target
manifest/all four plans, authenticates the saved matching fit and all15 native
attempts through the reviewed typed-v1 law session, and holds a source watcher.
Its cached target plan is sealed by canonical content and expires with its
context. A future native child must open separate producer and auditor target
sessions before starting workers and keep them alive through forecast/seal.

Live processing reads the newly persisted actual prefix/gate, rederives strict
settled18 prior history and owned ancestry, and reconstructs every prefix/gate
field against that exact target. Previously authenticated immutable training
sources stay under active change/restore watchers rather than being reread or
reparsed during the live budget. Both producer and auditor still propagate
every181×12 numerical world; no world, unknown mass, queue horizon or six-term
score is dropped to accelerate the path.

The collector derivative preserves original runtime enqueue/drain, fixed90+120,
ordered priming, owned monitor, checkpoints outside protected windows and cleanup.
It persists prefix/gate, calls the typed producer, invokes a separately opened
auditor session to fully replay and durably seal, checks the canonical typed
seal and finishes the final checkpoint before origin. Every deadline crossing
before that point stops with all181 targets unstarted. The seal accepts only
the typed-v2 full replay artifact, never the generic ID/completion CPU schema.

Typed-v1 CPU timing measured about37.8 seconds for one synthetic single-macro
181×12 pass. Two passes already exceed60 seconds. The300-second v2 lead is a
**predeclared budget, not an established capacity result**. Actual matching
multi-macro fit, source/context prefix audit, hashing, producer pass, independent
auditor pass, persistence and margin remain to be benchmarked on CPU. Plans
explicitly record `UNKNOWN_NOT_BENCHMARKED`; the public launch remains HOLD.
Preauthentication can be excluded from live lead only by completing it before
worker startup while preserving both active sessions and source watchers.

Remaining before native enablement: independent review of both new modules,
the real saved matching fit outcome, a source-bound real-fit timing/budget
report, supervised native-v2 child/claim wiring and a newly prepared artifact
binding those reviews. A later source version must make that release explicit;
this draft neither hides the HOLD nor claims native predictive accuracy.
