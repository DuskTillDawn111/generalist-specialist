# CPU handoff after the exact matching and public181 resource training

`twoFIFO_public181_full_score_continuation_v1` prepares a source-bound CPU-only
handoff from two already-running writers. It does not start or retry training,
fit a model, change a prepared campaign, prepare a native target, or launch GPU
work. Preparation and author checks do not start the waiting controller. Its
`run` subcommand requires explicit `--execute` after independent review.

The matching dependency is the existing four-stage CPU driver's exact command,
PID/start ticks, working directory, exclusive claim, current reviewed phase12
fit manifest and exact original four actions. The future fit is accepted only
at that driver's canonical `fit/report.json` together with its `fit-complete`
marker and the original source fit's exclusive `phase12-fit-claim.json`.
Completion of the unrelated later generated/resources/rescore CPU stages is
not required once that real fit publication is complete. A driver that exits
without a completed fit produces HOLD. Rank/root flags remain unchanged.

The second dependency is the existing resource-after-kernel queue's exact
command, PID/start ticks, working directory, manifest and claim. Its fixed
kernel continuation must refer to the same matching-native campaign and queue
claim as the fit source. Acceptance requires the exact three-capture/twelve-CPU-
stage kernel final, all eight resource-native attempt IDs in the original
order, and the corresponding complete native tuple intake with the same eight
review receipts. Nothing is selected away because its quality label is absent.

The controller polls at thirty seconds by default and checks the pinned writer
identity before accepting even an already-published final. PID reuse fails
closed. Partial JSON permits waiting only while the original writer remains
live. A dead original writer is acceptable only when the complete canonical
publication exists. Immutable preparations are authenticated before waiting;
polling does not repeatedly run a native review, optimizer, or law compilation.
The first completed dependency gains a source-change watch while the second
remains pending, and that watch stays active through qualification, CPU budget
execution and final handling. Edit-and-restore counterexamples are rejected.

After both publications, qualification authenticates the saved matching fit
without redoing its numerics and preserves its original root/fold/nuisance/OCR/
law-compilation gates. Missing resource tuples or actual native Q cause a
complete HOLD with zero benchmark invocations. When those gates pass, the exact
saved law is authenticated/compiled, every public181 donor-block key must be
supported, and the eight resource reviews/intake are rederived. No future
validation label is read.

Only a supported complete input reaches the one predeclared CPU command:
`twoFIFO_public181_full_score_budget_v1` with that exact real fit, frozen public
contract, resource manifest and eight reviews. The environment hides CUDA and
sets CPU BLAS thread counts to one. An exclusive continuation output and a
persisted benchmark-started claim prevent retries. Failed commands, unsupported
inputs and a full-score CPU budget beyond the original fixed 300 seconds are
explicit final HOLD outcomes. A completed budget only means readiness for a
separate native4 review; predictor accuracy, original stability and final gain
remain unqualified. No validation or deployment proceeds automatically.

The twelve CPU author tests use temporary process/JSON fixtures and explicitly
mock native fit/intake authentication in orchestration branches. They exercise
exact command/claim identities, start ticks, identity-before-final order,
partial publications, dead writers, continuous edit-and-restore detection,
exclusive fit claims, kernel/native8 denominators, rank/Q/public-key HOLD,
exactly-one CPU command, fixed-budget HOLD and no retry. These fixtures do not
produce a real timing budget or native accuracy result. The actual prepared
manifest separately authenticates the currently running writers and immutable
training preparations; its run directory remains absent until reviewed
explicit execution.
