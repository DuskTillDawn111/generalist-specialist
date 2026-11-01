# Complete-block FIFO training intake v3

The intake consumes the frozen v2 campaign and only new owned-training reviews.
Its schema is `gsdeploy.twoFIFO_complete_block_training_intake.v3`; it never
relabels a v2 macro document as old v1 training evidence. The collector and its
frozen source closure are unchanged.

`run(campaign_path, review_paths, output)` first audits the complete planned
campaign. It binds each submitted review to an exact planned attempt. A missing
or ineligible attempt returns `WAITING_FOR_ALL_TWELVE_COMPLETE_NATIVE_ATTEMPTS`
with all twelve expected plan receipts, planned offered counts and completion
counts. This path reads review metadata and plans, and does not open partial raw
responses or extract macros. Duplicate reviews and foreign campaign identities
are rejected.

Only twelve complete native reviews permit `twoFIFO_composable_macros_v2.extract`.
That existing extractor rederives each native review. The intake then verifies
every planned request in original case/arrival order, all public input/owner/task
identities, eighteen priming exclusions per attempt, and every macro count.
The denominator is 546 measured requests and 216 separate priming requests from
twelve startups. Priming informs causal history but never supplies response
labels. Entry, OCR inference, forward and READY-tail macros all remain present.

`prepared_rows` adapts the new schema directly to the existing phase design and
numerical row interface. It preserves the original owner/input/task/macro/phase/
kind/forward-shape nuisance group. `capture` is the whole training block;
`raw_sha256` and `attempt_id` are provenance. There are three replication units
and twelve startup attempts. Each leaveout excludes all four shards of one block
and trains on the eight shards of the other two. Unseen heldout nuisance groups
are counted and listed as unknown without dropping their rows.

The fixed designs are reused from `phase_heterogeneous_response_training_v1`:
global phase-aware, family/phase count plus global same-phase count, and full
family/phase count/same-phase count. No candidate or column is added or removed.
At zero response coefficients the audit reports joint nuisance/response rank,
nuisance-adjusted derivative rank, exact piecewise zero/identical columns, and
the original analytic score and curvature. These are necessary structural
diagnostics at an unfitted point. No optimizer runs, and the report never calls
them fitted-root numerical qualification. A zero residual-variance likelihood
stays explicitly unknown rather than receiving a variance floor.

OCR entry support is computed from positive-duration processed-prefix macro
paths for each exact targeted OCR input in each whole block. It retains actual
peer request/phase identities and reports n=0 exposure, n>0 exposure with a
processed language BEGIN, single-peer prefill/decode exposure, and observed
two-language phase composition. Planned offsets are never observed phase labels.
Missing conditions remain UNKNOWN. No fitted response provider, independent
prediction accuracy or final utility gain is qualified by this intake.
