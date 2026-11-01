# Training run-out risk diagnostic and multimodal context audit

The additive v10 risk diagnostic uses the original risk03 training rows and
exactly reproduces its frozen model with 40 stumps, learning rate 0.2 and
minimum leaf size 2. Four folds each fit three original runs (51 cutoffs,
both arrival families) and score the remaining training run (all 17 cutoffs).
Rows retain `split=training`; `evaluate_holdout` is not called. Old models,
unit tests, utility, admission and stability rules remain unchanged.

`artifacts/predictor-native-queue-risk-loro-v10-20261028-training01/report.json`
retains all 68 cutoffs. Feature support coverage is 68/68; each fold contains
4 negative and 13 positive labels. Each fold has ROC AUC 0.8846153846 and
Brier score error 0.1064986691. These are descriptive training run-out metrics,
not independent validation or probability calibration. All 40 stumps select
`queue_work_ms` in every fold, explaining the same coarse score separation.
The separate `interpretation.json` records score/label counts and this limitation.

Existing training has only `burst_A` and `burst_B`, two runs per family. Leaving
one family out would leave two runs and one family, below the fixed fitter's
three-run/two-family support requirement. New family C is reserved in
`artifacts/predictor-native-queue-risk-family-C-v10-20261028-prepared01/plan.json`.
It changes public arrival timing while preserving all 24 owner/task/input
identities, all 17 cutoff offsets and the original window, drain and scoring
constants. Its run and group are disjoint from risk03. This is preparation only:
the v8 training collector rejects validation and C/D families. A validation
execution adapter, effective-context verification and protocol/prediction seals
must be completed before collecting a genuinely independent C run. No data has
been collected or relabeled for C.

The two multimodal packets each contain six scheduler contexts and four
scheduler global PIDs. Both have two extra contexts sharing the global PID of
the OCR primary candidate. Their independent envelope/kernel endpoint evidence
is material: packet01 extras contain 19,994 and 28 measured OCR endpoints;
packet02 extras contain 18,302 and 140. The larger extra participates in all
four measured OCR requests in each packet. No extra context is dropped.

`artifacts/predictor-multimodal-context-audit-v10-20261028-summary01/report.json`
binds complete all-context reports and separate startup/priming/measurement
endpoint audits. The evidence supports temporal and scheduler-process
correspondence. Direct native context identity remains UNKNOWN: scheduler
context IDs/global PIDs do not join to the four CUDA metadata contexts or the
CUPTI/PROCESSES namespace. The SQLite exports lack native context creation and
destruction records; RESTORE_START and SAVE_END are scheduling events and cannot
substitute for lifecycle endpoints.

Unresolved evidence is retained. Packet01 has an unpaired OCR primary SAVE and
402 measured OCR endpoints outside all reconstructed brackets, plus seven
uncovered endpoints from other workers. Packet02 has four uncovered G2 measured
endpoints and two G2 endpoints inside the OCR primary envelope. No timing
tolerance, clipping, event filling or fitting is introduced to remove them.
Strict and paired-work PASS remain separate from context-attribution evidence.
All work described here is CPU-only and establishes neither physical SM service
nor independent end-to-end predictor accuracy.
