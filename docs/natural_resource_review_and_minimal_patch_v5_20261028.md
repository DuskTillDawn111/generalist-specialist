# Natural resource review and minimal patch, 2026-10-28

Independent review of `natural_resources_v5.py` and the seven-capture model
finds that it implements a direct, conditional resource model. It does not
predict service slowdown. It uses real simultaneous handler activity to label
whole board-power/owned-memory states; it does not add solo or pair estimates.
Startup, priming and drain are excluded from active-state fitting. A state needs
actual interior telemetry observations in at least two independent raw captures.
NVML peaks remain sampled, and no safety factor is applied inside this model.

The existing model is
`artifacts/predictor-gpu-20261027/natural-resource-training-v5-20261028-seven-captures-final01/model.json`.
Its nine states are:

| active family/task state | independent captures | sampled intervals | interior samples | total seconds |
| --- | ---: | ---: | ---: | ---: |
| idle | 7 | 19 | 192 | 4.2734 |
| S | 5 | 15 | 248 | 5.2282 |
| G2 text | 4 | 16 | 134 | 2.8594 |
| OCR | 2 | 4 | 68 | 1.5166 |
| G2 image | 2 | 4 | 68 | 1.5692 |
| G2 ASR | 2 | 4 | 150 | 3.4565 |
| SS + G2 text | 2 | 12 | 592 | 13.9651 |
| SS | 2 | 4 | 8 | 0.2007 |
| S + G2 text | 2 | 12 | 35 | 0.8417 |

S means one specialist text owner and SS means both. The SS and SG2 text
states meet the capture-count rule but have little sampled duration. The
idle state's mean is 182.31 W, with maximum interval-average 389.32 W: recent
activity and thermal/power inertia remain in the measurement. It is an
inter-handler idle training distribution, not a pure idle baseline. `max.watts`
is a maximum interval average, not an instantaneous power bound. Input lengths,
execution stages, predecessor activity, and S0/S1 interchangeability are pooled
assumptions rather than validated invariances. Training memory means/maxima do
not establish capacity bounds.

One provenance hardening gap was reported to the parent: fit-time loaders
verify strict-review evaluator receipts, but provider construction currently
checks only the top-level model sources. It does not recursively revalidate
the nested strict evaluator sources or the separately referenced training
index. The old source and receipts are retained; any fix needs an explicit new
lineage and model rather than mutating that evaluator in place.

The original four owners give 24 possible resource states when counted by the
current family/task state key:

`3 specialist counts (0/1/2) × 4 G2 modes (idle/text/image/ASR) × 2 OCR modes`.

The 15 missing states are:

- S+OCR and SS+OCR.
- G2 text+OCR, S+G2 text+OCR and SS+G2 text+OCR.
- For each of G2 image and G2 ASR: S+G2, SS+G2, G2+OCR, S+G2+OCR and SS+G2+OCR.

The minimal new packet is
`artifacts/predictor-gpu-20261027/resource-full-state-training-v5-20261028-prepared00`.
Its `coverage-design.json` records all 24 states and exactly which existing
states are already supported. `campaign.json` records two independent fresh
natural captures. Each has exactly 30 jobs: each of the 15 missing states with
the original short and long training inputs, totaling 84 measured requests.
No held-out input enters training. Zero planned launch offsets maximize possible
full-state overlap; actual handler overlap and interior samples determine the
observed state, not the planned job label. This packet addresses resource-state
coverage and does not attempt phase, offset, background-rate or slowdown sweeps.

Both captures preserve all four original resident owners, original eight
priming inputs and their actual 12 priming requests, native worker execution,
and the frozen startup collector. They use no Nsight trace. New wrapper sources
are frozen separately; existing snapshots and old raw are not copied. The
native natural projection remains
`b2687b3d82de7832678b3a410cb9f40cf7a67296cf6ac5fbf083a07ab59ca54a`.
The new full context honestly changes to
`8c7fa532c5cc23cc0ad329943f9c14aaac4bfbe0dc7de9d3debfe715e23e21ab`.

`resource_training_campaign_v5.py` uses the new explicit generic plan schema
`gsdeploy.generic_full_state_resource_training_plan.v5`. It does not disguise
30 jobs as the old fixed six-job SSG patch. Its source/resource audit reconstructs
the exact planned missing-state jobs, validates digests, source/stat receipts,
four owners, unchanged budgets and natural projection. The separate
`resource_training_collection_review_v5.py` checks strict natural lifecycle,
telemetry, exact measured/priming output contracts and all generic job/request
bindings, producing the existing full strict result schema with its own
evaluator receipts. The old strict evaluators are untouched.

At preparation, free disk was 34,160,852,992 bytes. Keeping the original 12 GiB
reserve and a two-run 64 MiB size scenario leaves 21,208,842,240 bytes. Previous
natural startup took about 66.7 seconds and priming about 8.6 seconds; the
declared per-capture scenario is 180 seconds, not a bound. The unchanged collector
checks its 2,400-second budget and disk reserve at phase/job boundaries, not with
a continuous watchdog. Exclusive output, no retries and the original GPU lock
remain in force. A target with no interior sample in either capture stays
UNKNOWN; the plan does not promise success or automatically expand collection.

Five focused CPU tests pass, covering the complete 24-state universe, exact
15-state gap, two strata, original primers, direct targets, no synthetic owners,
and exact generic strict request coverage including missing/validation/duplicate
requests. Both prepared runs passed CPU source/resource audits before execution.
Actual execution and review status belongs to the fresh raw/strict reports.

The parent subsequently added `natural_resources_v6.py`, preserving v5 and all
old estimates. Independent source review confirms that it seals the base model,
recursively resolves nested receipts, verifies the training index exactly
against all measured training requests, and repeats those checks at provider
construction. Historical replacements require an explicit manifest and exact
original bytes. Every original estimate/evidence field is compared against the
immutable v5 base before invoking its original provider. This addresses the
reported provenance gap; physical pooling and sampled-peak limitations remain.

Capture01 actually completed with raw SHA256
`50a23e88e5f69a4e05f59c3399b130b1ced9353a61bb7ee730773a17a88f4f01`.
The new generic strict adapter reports PASS, SHA256
`ccd45803d23b37a9d0d9c6fa4966803be9a37308a6491b3b5159c987fb61a775`.
All 84 measured and 12 priming requests complete, with clean owned shutdown.
The resource coverage review finds actual simultaneous handlers and interior
NVML samples for all 15 targeted missing states; all 24 possible states appear
with interior samples in this capture. That is one independent capture, not yet
the required repeated support. Raw size is 7,489,858 bytes and free disk after
review remains 34,152,198,144 bytes. Capture02 starts as a fresh independent
startup in its predeclared exclusive directory.

Capture02 also completed, raw SHA256
`b00c4a7c8fb6eef4152b3cf0abeeffda20b827eb2f58e0705e85861689b07a44`,
strict PASS SHA256
`96521ec9431bf0680f516aeca642ab48aff9ae079d0620b799dfad52546e34a0`.
It independently provides actual handler activity and interior NVML samples
for all 24 states, including all 15 targets, across its 84 measured requests.
The combined two-fresh-capture coverage review is PASS, SHA256
`31769456fa20b9e4c4c7f730d8a956b1546cd9c8a145dc41dccc16679d6aed4f`.
Every one of the 24 family/task states has direct sampled support in both
captures; no state was synthesized. Final free disk is 34,102,005,760 bytes.
Both GPU processes exited cleanly and released the original experiment lock.
The parent owns subsequent nine-capture resource refitting and v6 sealing.
