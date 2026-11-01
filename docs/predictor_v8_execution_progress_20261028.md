# Predictor v8 execution progress

The active goal remains unfinished. The order is paper mechanisms → matching training profiles → frozen independent prediction validation → mechanism ablations → fair four-group routing comparison. The original sealed03 validation remains FAIL; no current CPU result supersedes it. There is no default global cap1/cap2.

## Completed in this continuation

- Exact-input host v7 is reconstructed from the original isolated training evidence at provider construction. Eight owner/input cells have two independent raw captures each. Training diagnostic03 records the unchanged, predeclared two host scenarios and both old/new providers. Issue-depth isolated endpoint MARE changes from 40.68/38.63% to 1.80/1.65%; those endpoints fitted the model. SSG component-only errors change from 32.30/27.82% to 19.29/22.60%; other components used shared training evidence, so this is not independent full-pipeline accuracy.
- Explicit component host selection v8 and four-task gate v7 complete an actual twelve-template CPU run: eight text/OCR cells use exact-input v7, four image/ASR cells use their original independent v5 component. No UNKNOWN triggers another provider. Generation, startup, quality, resources and original equal-weight six-term scoring remain bound to their original contexts. The completed run has 354217 nodes and 441943 events, with 369 production sources unchanged. Training utility is −1.5452280656643664, a prediction value rather than measured gain. See `artifacts/predictor-component-host-fourtask-cpu-diagnostic-v8-20261028-final02/report.json`.
- Atomic simulated checkpoint v7 conserves original numerical results and cumulative budgets. Final independent review covers 477 original and 198 additional boundary continuations; all 17 preserved counterexamples are rejected. These include provider configuration and inconsistent resource totals/history. This does not recover an observed GPU cursor. See `artifacts/predictor-dag-checkpoint-independent-v7-20261028-final05/report.json` and `artifacts/predictor-dag-checkpoint-counterexamples-recheck-v7-20261028-final02/report.json`.
- A second independent shared-SSG profiler capture and a paired natural capture actually completed. Each has the original six jobs, eighteen measured requests, four resident owners and twelve priming requests. Both strict reviews and the eighteen-request actual-work pairing pass. Raw data are retained in `artifacts/predictor-corun-training-evidence-plan-v8-20261028-final01/repeat-original-SSG/`.
- A source-bound read-only repeated-phase review confirms equal work and the full launch/signature order for 108 semantic groups and 328044 kernels. Phase sums are broadly reproducible, but individual decode groups can vary substantially (G2 repeat/original ratios span about 0.54–1.86). These are kernel lifetimes containing scheduling effects, not uninterrupted SM execution or a calibrated slowdown law. See `artifacts/predictor-SSG-repeated-phase-review-v8-20261028-final01/report.json`.

## Work in progress

1. A new unified Nsight collector and review will capture the parent/worker process tree on one timeline, including GPU context switches, with a continuous budget watchdog. Existing profiler captures used one session per worker. Historical capability already produced 2130 real switch events, but driver PID/context IDs are not joined to CUDA worker identities. Current NVML evidence uses container PIDs, so a guessed PID offset or NVML delta alone cannot establish the missing host-PID bridge. The first new capture must preserve UNKNOWN if attribution cannot be demonstrated.
2. A DAG-visible observer will adapt reader-visible native facts at a declared cutoff while retaining the complete known arrival horizon. It must keep simulated progress separate, preserve START/result/readiness uncertainty, and decline busy-state continuation without a supported residual model. An idle-cutoff suffix adapter is a first executable boundary, not completion of rolling correction.
3. A phase-aware scheduling diagnostic will distinguish host supply, sync, kernel lifetime and peer phase using original solo and repeated shared evidence. It must not fit an overall SSG coefficient, invent a timeslice quantum, or turn occupancy into measured throughput. Any new scheduling hypothesis requires frozen independent validation.

Future visible-state residual calibration, recovery-risk evaluation, true GPU batching/common-weight cache mechanisms, unseen-content validation and fair routing comparisons remain open. Original rho, g(n), stability, full offered cohort, arrival→reader receipt, startup-through-drain memory and once-only 3600-second startup-energy amortization remain fixed.

## Subsequent execution in this continuation

- The first whole-lifetime unified v8 light and natural collections completed, but both strict reviews FAIL. Each has one reversed sample midpoint caused by two overlapping NVML queries appending in completion order. Original raws, order and FAIL reports are preserved. No sorting was used to qualify them. The four-owner CUDA/NVTX and 4968 context switches in the light capture remain descriptive capability evidence only.
- The new v8_1 monitor serializes complete queries, appends and identity observations. Fresh light and natural captures, both strict reviews and their actual-work pair now PASS. They contain four measured requests and twelve original priming requests per domain. Evidence: `artifacts/predictor-gpu-20261027/unified-corun-v8_1-20261028-prepared01/paired-actual-work-v8_1/report.json`, SHA `76900198bbd3fe0f3bfac42cade2afa41b58b8e94dbf4fc7c72e00b205d8f7bc`.
- Controlled temporal context attribution is now possible in the common trace. Unique context first appearances during sequential owner startup, plus all twelve owner-specific priming traces, exclude all 23 incorrect permutations. This is controlled temporal attribution, not an OS namespace PID identity or equality between driver and CUPTI context IDs. RESTORE_START→SAVE_END remains an availability envelope containing unmeasured save/restore overhead, not uninterrupted SM execution. Qualified evidence: `artifacts/predictor-dag-phase-common-session-identifiability-v8-20261028-qualified03/report.json`.
- Actual training idle-cutoff replay retains fifteen completed prefix requests and all three future requests, executes 69459 DAG events and scores the original eighteen-request cohort. Two busy cutoffs decline without invoking the engine. This profile schedule was formed after previous jobs completed, so the report explicitly remains a retrospective training wiring diagnostic; it is not prospective accuracy. Missing prefix energy/memory preserve null E/M/full U. Evidence: `artifacts/predictor-visible-dag-cutoff-replay-v8-20261028-final01/report.json`.
- New v9 request-start supply preserves public arrival and owner FIFO while allowing independently supplied absolute not-before START. Forty-seven focused CPU checks cover default v7 compatibility, dispatch waiting, HOL, checkpoint/provider identity and joint reader-survival scenarios. A missing RESULT can retain both an active handler and an ended handler whose notification is in transit. Scenario declarations alone still do not establish measured training qualification; a reconstructing fit/loader is being added. Evidence: `artifacts/predictor-visible-busy-dag-v9-20261028-final01/report.json`.
- Checkpoint callable identity now hashes a recursive typed code-value representation. Marshal's reference/intern tags can change despite unchanged code values. New tests reproduce unequal marshal bytes for semantically equal constant sharing while requiring changed instructions/constants/nested code/exception metadata to change identity. The checkpoint/numerical suite passes 37 tests after this fix. This is an implementation correction, not predictive accuracy.
- Two larger paired common-session training packets are prepared: exact prior solo/SS/SG controls, an original mixed long/short SSG job and isolated OCR, fourteen jobs/twenty-two measured requests plus twelve priming requests per domain. Packet01 light is executing first. A separate natural collector for genuinely predeclared complete arrival traces and cutoff grids is under CPU review, to support busy-state and recovery-risk training without reusing retrospective profile arrivals or sealed validation labels.

The formal sealed03 accuracy gate remains FAIL. None of these collection, attribution or CPU results supersedes it.

## Further collected training and queue implementation

Both expanded common-session packets now pass light strict review, natural
strict review and actual-work pairing. Each domain retains 22 measured and 12
priming requests, including original solo/SS/SG controls, mixed-length SSG and
OCR. Pair reports are under
`artifacts/predictor-gpu-20261027/phase-scheduler-training-v8_1-20261028-packet01/paired-actual-work-v8_1/`
and the corresponding `packet02` directory (SHA `37938f960a3a63e24a04b1cde7e1df905189e92343734e5e64f0696112cc9892`
and `3f248ea92fa90362f78ffe0438da1403e1241bd754693a038916c6398d57d5e1`).
Capture PASS does not mean every scheduling state is identifiable. Unpaired
switches and unresolved intervals remain excluded/explicitly unknown.

The phase scheduler's independent counterexamples exposed and fixed an
extra handoff before same-clock READY dispatch, missing completions exactly
at the drain boundary and zero-work closure bypassing the event budget.
The v9 continuation separately enforces prior owner READY with every start
provider, exact visible START/END/READY times, original owner FIFO identity,
and lifecycle reconstruction of pending and completed receipts. The frozen
63-check CPU report is
`artifacts/predictor-visible-busy-dag-v9-20261028-final03/report.json`, SHA
`fe3547b157118b246f2dcf7976a85e3f45f75929803d8fd8a70ec190a32bf847`.
The earlier failed source-window and counterexample artifacts remain intact.

The first prospectively declared native queue run actually completed and
passed independent strict review. The four-run campaign uses the same fixed
24-request registry, two predeclared burst patterns, 16-second offered window,
four resident owners and 12 primers per run. Plans are under
`artifacts/predictor-gpu-20261027/native-queue-training-v8-20261028-prepared01/`;
actual captures and strict reviews are under the corresponding `collected01/`.
The second run is in progress. This campaign is training only, with no new
accuracy, stability or routing-gain qualification.

`native_queue_risk_rows_v9.py` now connects the Sinan-inspired library's input
contract to native captures: the current feature path receives only public
arrival/input identities and reader-visible START/READY facts. It sums full
reference work for visible unstarted requests, oldest waiting age and owner
HOL age, and computes multiscale work envelopes from the entire declared
future suffix. It does not guess active GPU residual work. The separate
offline path labels original task-latency-reference violations and actual
queue recovery, requiring a complete future horizon and conservative complete
cohort START coverage. Missing work or censored labels explicitly omit rows;
they never become zero work or negative labels. The strict adapter reruns the
original capture review. Twenty-eight focused library, adapter and independent
checks pass. Actual source-bound four-task feature costs and real risk fitting
remain to be connected; no physical risk calibration is claimed.

The execution order and original sealed03 FAIL remain unchanged. Next are
completion of the four native training runs, reconstructed joint-support and
risk training diagnostics, a supported full-pipeline scheduling predictor,
then fresh frozen independent prediction validation. Mechanism ablations and
fair routing comparisons must wait for that gate.

## Four native runs and real optional risk fitting completed

All four predeclared native queue runs now pass independent strict review:
96 measurement requests plus 48 original priming requests in one execution
context, with complete original six-term diagnostics. Campaign evidence is
`artifacts/predictor-native-queue-training-campaign-v8-20261028-final01/report.json`
(SHA `c5851533105c60ddb6d19ab6803d72e0e12fe8be4f74b608ad3690acd4662795`).
Run03's accidental review before raw was written remains preserved separately;
its completed-capture review is `strict-review-after-collection/report.json`.

The optional risk fitting is now actually connected to those four captures.
The qualified training-coordinate version is
`artifacts/predictor-native-queue-risk-fit-v9-20261028-training03/report.json`
(SHA `89fa3cde9b165d3703e162099522ce6ccb54d4ca83796ed7792bb040da2eb0ac`).
It rechecks original capture reviews and reconstructs all twelve exact-input
reference costs from two isolated, nonoverlapping, same-work/same-ordered-kernel
captures per component. Text/OCR and image/ASR retain their original source
contexts. Only GPU lifetime reference sums enter these feature coordinates;
host, copy, lifecycle and active residual GPU work are not silently included.
The explicit natural-to-queue collector-domain bridge preserves separate full
context identities, matches original native sources/weights/input catalog
content, and remains a timing-transfer hypothesis.

All 68 predeclared cutoff rows are usable: 16 target negatives and 52 positives
across four runs/two burst families. The fitted frozen model was actually called
on all 68 training feature vectors; the sidecar records each output. There is
no held-out risk accuracy or probability calibration claim. Scores do not affect
the original six-term utility, rho, stability or execution admission.

The earlier risk training02 numeric-coordinate fit is retained with a semantic
review hold: its mixed solo/SSG fitted duration sums were incorrectly named solo
activity work. It cannot qualify the solo-work feature contract. Training01 was
interrupted after input-content/numeric-receipt review gaps were found. Neither
is substituted for the corrected training03 result.

The old-duration busy whole-DAG diagnostic also exposed a real failure: observed
training PHASE-relative emissions can lie beyond a predicted handler END. Its
failure and traceback are retained, with no clipping or endpoint stretching.
The next version screens actual DAG prefixes before attempting a full suffix,
uses supported exact observed START anchors and reports each inconsistent joint
scenario as rejected. Updated phase/availability service prediction remains
necessary before a useful complete busy forecast or new accuracy qualification.

## Whole-DAG diagnosis and first-activity supply correction

The first actually executed complete availability-model diagnostic is
`artifacts/predictor-dag-availability-whole-DAG-v8-20261028-diagnostic02/`.
All 28 jobs/44 original requests remain declared. Fourteen solo/OCR jobs
produce numerical predictions; fourteen co-run jobs are entirely UNKNOWN at
their first unsupported contention-entry state. The roughly 2.4% error of the
numeric subset therefore describes solo/OCR only. It does not establish co-run
accuracy. Independent source/model reconstruction and scheduler boundary tests
pass, but cannot turn these rejected jobs into supported predictions.

The resulting host-supply investigation identified a concrete missing axis:
the old endpoint-fitted issue-depth supply has no separately fitted first
activity preparation. For example, G2 long input's first GPU issue is predicted
near 0.120ms versus 11.911ms in its original isolated API capture; long OCR is
near 0.135ms versus 112.484ms. These are profiler-domain comparisons, not direct
natural-domain first-CUDA measurements. Overly early supply creates initial
joint frontiers that the real training did not observe. A new version is
separating initial sync, first-activity preparation and later issue supply,
subtracting preparation before refitting the remaining host budget. Simply
adding a delay to v7 would double-count part of that budget.

Two additional multimodal common-session training packets have been prepared
and independently reviewed: 14 jobs/26 measured requests plus 12 primers per
domain, preserving original inputs/residents. They cover solo G2 image/ASR,
those modalities with both long-text owners, those modalities with long OCR,
and text solo controls. Evidence is under
`artifacts/predictor-gpu-20261027/multimodal-availability-training-v8_1-20261028-packet01/`
and `packet02/`. Packet01's two strict reviews and actual-work pairing PASS;
packet02 light has completed and its review is in progress. This closes a
training coverage need, not the host-supply error itself.

The optional risk cutoff API is now executable directly from public trace and
reader-visible records via `native_queue_risk_inference_v9.py`. All 68 original
training cutoffs reproduce frozen risk outputs and preserve the original
forecast/status/utility/stability fields when attaching the sidecar. This API
diagnostic explicitly does not execute a main DAG continuation. The independent
risk review rederived all 68 rows, refit the frozen model identically, checked
the twelve exact solo cells/source receipts and reproduced all 68 provider
calls. Its report is
`artifacts/predictor-native-queue-risk-independent-review-v9-20261028-cpu01/report.json`.

### Continued execution: multimodal training complete and fresh gate adapter

Both multimodal availability packets now pass strict light and natural reviews
and actual-work pairing. Each domain has 52 measurement requests and 24 primers
across the two captures; errors are empty. Packet02 natural strict report SHA
`e3c8918e2cd4eb5cede365125b5d4e6fb3ecaf6f6c9aa228bed2269c34e5d36c`,
and actual-work pair SHA
`ce3880f91ceab9a4fe1b18b3baae3f86f1c8753ddb2fe0462e00ed94f06025e4`.
The campaign report is
`artifacts/predictor-multimodal-availability-training-v8_1-20261028-complete01/report.json`.
These are training completion checks, not independent co-run accuracy results.

The new `prospective_flow_contract_v11.py` binds arbitrary full declared cohorts
and exact fixed routes, including all future arrivals and per-owner FIFO. It
scores full parent-receipt forecasts using original six equal terms, startup
energy once and memory safety factor once; its gate projection never adds the
old adapter's second startup charge. Six CPU regressions cover omitted/reordered
future requests, foreign routes, incomplete completions/resources, exact queue
START boundaries, handler versus parent latency, chronology and startup
accounting. They pass via unittest (system Python has no pytest). No new model
or validation seal has been created.

The old v9 busy-prefix experiment was interrupted after at least 940 seconds
wall time/939 seconds CPU and about 7.3 GiB RSS without completing even the first
50ms-cutoff run. The recorded stack was repeatedly encoding static DAG plans
for the checkpoint contract. Evidence and the interruption report remain under
`artifacts/predictor-native-queue-busy-prefix-v10-20261028-training01/`.
This is an observed implementation cost; it is not an online-usable prediction
or a failed physical prediction. The independent phase-aware v10 engine uses
immutable verified static model storage and compact dynamic checkpoints.

### Actual full-cohort CPU adapter and context-state diagnosis

The fresh v11 collector, startup verifier and prospective adapter have passed
independent CPU review. Original training normalizers are checked against the
trained bundle (not against a copy made from the new plan). Receipt at the exact
drain endpoint is allowed; +1ns is rejected. A new training guard explicitly
rejects validation measurement rows or labels before availability fitting;
original unsplit priming rows are separately checked as completed priming.
The actual two source captures pass with 22 measurement requests each.

Fresh public preparation retains four fixed candidates ×57 requests, offsets,
inputs, normalizers and thresholds, with new IDs and actual v11 source/context
snapshots. It is not a protocol/model/prediction seal. Actual full-cohort
preflight is complete under
`artifacts/predictor-prospective-availability-v11-20261028-preflight02/report.json`:
all four candidates are UNKNOWN at unsupported contention-entry frontiers.
The strict completeness check therefore rejects seal and validation collection.
No new validation GPU run was performed and the old independent FAIL remains.
Cold provider rederivation took 241.31s; four candidate checks took about 30s each
including fresh startup source verification. This does not establish online
latency. All core/public source receipts remained unchanged during this run.

Source analysis identifies a state-semantics error beyond insufficient table
coverage: v8's fitted current owner denotes the observed held context, while
its execution scheduler can immediately replace that owner with the sole ready
owner, without a restore event. The real first-5ms source replay contains 40
such disagreements among 172 checked points. Independent per-owner mean host
preparation also reverses the first-GPU order seen in the two joint captures.
A new v12_1 state machine preserves held context, pending readiness, absolute
SAVE/RESTORE/SERVICE events and restore age independently; its CPU/source replay
checks do not identify a complete dispatch oracle.

Age-conditioned empirical survival work preserves unsupported tails and
explicit delayed entry/right censoring. The initial v12/v12_1 extraction used
upstream complete-envelope-selected samples, so its independent arithmetic
review does not establish an unbiased context population. Separate v12_2–v12_4
versions retain that history while reconstructing raw context envelopes and
clipping complete job observation windows for every resident holder, including
holders with no request in the job. Original restore time remains the age
origin; request/frontier boundaries censor rather than invent a SAVE. The
current v12_4 model has 329093 episodes across TWO captures, not that many
independent experiments. A predeclared leave-training-capture-out age/horizon
grid is in progress. It remains an empirical availability-envelope hypothesis,
not physical SM service, calibrated probability or a complete forecast policy.

### Continued execution: trainable context models and joint host observations

The complete training index is now `predictor-complete-training-index-v12-20261028-built02`:
34 distinct raw SHA captures, 621 measurements and 384 separately retained primers.
The original 297-row index is a full subset. The four freshly prepared 57-request
cohorts pass run/request-ID disjointness checks; no new validation labels exist.
Lineage boundaries remain explicit rather than claiming an exhaustive history audit.

The v12_4 predeclared leave-training-capture-out grid finished: 2720 queries,
16320 horizons, 5570 common-support empirical CDF comparisons and 10750 UNKNOWN.
Supported-cell mean absolute CDF disagreement is 0.0271007. This is two training
captures, not independent calibration or population accuracy. Independent review
found a floating-point median boundary defect; v12_5 keeps the complete hazard,
survival and median calculation rational until output conversion. Its actual
source rederivation and eight tests pass. Repeating the same declared grid with
v12_5 preserves coverage and the aggregate CDF disagreement; old artifacts remain.

The actual compact v10 phase continuation plus risk sidecar also finished under
`predictor-native-queue-phase-busy-v10-20261028-training02`. Cold verification was
266.407s, actual DAG work 18.617s and same-50ms-cutoff risk query 0.04016s after
47.179s work reconstruction. All 24 requests remain; the main forecast is UNKNOWN
at unsupported availability/START/work cells. Risk score 0.477156 is diagnostic,
not a calibrated probability. This establishes actual DAG-sidecar integration,
not online readiness. Four-run risk LORO retains all 68 cutoffs but shares the A/B
groups; its AUC 0.884615 cannot establish new-family generalization. All trees
select queue_work_ms. Family C is a public reservation only, with no GPU run or seal.

`joint_host_scenarios_v13.py` now extracts complete same-job, same-capture
observations. Training03 retains 56 jobs/88 requests across two natural and two
light captures. An independent audit rechecked 784974 unique kernel/API joins,
462 prefix APIs, common-origin translations and all source cohorts. First API
submission and first GPU activity are kept separate across streams; no per-owner
clock maps or independent means compose a synthetic joint scenario. Natural
and light domains remain separate and have no assigned coupling probabilities.
OCR's absent semantic/pre-sync boundaries remain explicit missing values.
This bank is supervision, not a counterfactual host-release provider: observed
synchronization/API waits have not been removed from its prefix observations.

New full-source initial/transition supervision retains 28 initial states
(27 paired, one actual gap) and 35737 adjacent-context marks. It includes 3517
holders and 715 next owners outside the kernel-pending frontier. A fixed L2
multinomial next-owner candidate now predicts new frontier combinations instead
of treating exact-cell absence as permanent UNKNOWN. Whole-training-capture
diagnostics are strong, but a workload-conditioned lookup baseline is already
very strong and independent review is checking a full-coverage fallback baseline.
The two OCR target marks are sparse support, not zero support or evidence of
reliable OCR generalization. None of these mark predictions supply SAVE timing
or an exact service-ready timestamp.

The age-conditioned SAVE candidate uses a predeclared piecewise-exponential
Poisson exposure likelihood with delayed entry and frontier-change censoring.
It retains all 329093 episodes, 35666 SAVE events and 293427 censored intervals
from two captures; it does not call them independent experiments. Its two whole
capture diagnostic exposure losses improve over constant/holder-only baselines.
This is a trainable statistical hypothesis with explicit new-category/tail flags,
not physical SM service or independent probability calibration.

The v14 held-context scheduler adds an explicit model-only initialization entry
point for joint holder/restore-age/readiness hypotheses. Ordinary source replay
stays disabled. An initial readiness delay cannot be renewed by host calls, and
pending SAVE/RESTORE/SERVICE events advance through empty ready sets. Interface
presence does not qualify a complete oracle. Five targeted CPU checks pass;
real initialization/timing/mark components still need to be composed and tested.

The actual USHER/Roomie wiring audit is documented in
`docs/usher_roomie_actual_wiring_audit_20261028.md`. Whole-request batching and its
runtime have CPU implementations, but the present v11/v10 main DAG route search
does not call them. Complementarity helpers are not on the main search path;
shared-linear/cache reuse is a NumPy prototype without the native GPU backend.
Those are real integration/evidence gaps. They must not be reported as measured
mechanism benefits. The old independent prediction gate remains FAIL and the
fresh four-candidate full-cohort gate remains unsealed/incomplete.

### Composed learned candidate v14: actual CPU integration in progress

The next-owner v14 candidate removes complete future job counts and offsets
from its features. The queue simulator still consumes every declared future
arrival chronologically. Changes to the public 4-by-57 future suffix leave
next-owner features, probabilities and OOD flags unchanged. Training02 uses the
same fixed L2 objective; the optimizer iteration budget increased solely to
achieve convergence. Whole-capture diagnostics are not independent accuracy.
The actual model is `predictor-context-next-owner-v14-20261028-training02`;
the source/reconstruction verification is `...-verification01`.

The prequeued handoff producer keeps all 35737 transitions: 30763 have an
eligible original kernel witness and 4974 remain missing. Independent review
joined every witness to its original owner/job, kernel readiness certificate
and next-context envelope. The observed RESTORE-to-first-prequeued-kernel
interval is not exact service readiness or pure switching cost. The candidate
fits SAVE-to-RESTORE gap separately from witness presence and the upper-witness
proxy; OCR pairs remain sparse and flagged. Training02 is a statistical
hypothesis, including its explicitly named reference-readiness rule.

`learned_context_oracle_v14.py` now composes the joint initial-state prior,
conditional cumulative SAVE hazard, causal current-frontier next-owner model
and handoff model. No future mark is selected before its predicted SAVE clock.
Original restore age is preserved; hazard starts at the observation's delayed
entry and its remaining budget does not reset at host/frontier boundaries.
Unknown initial-holder probability is never removed by renormalization.

Actual learned-component small-DAG smoke `predictor-real-learned-small-dag-v14-
20261028-cpu02` finishes single-request, dual-request and host-gap cases with
the explicitly supported text#1-short first frontier. Every reference GPU
node conserves its integrated work. The other three initial-frontier cases
remain UNKNOWN due to unidentified initial-holder probability. These are CPU
mechanical checks, not forecast-accuracy or physical-exclusive-service proof.

The first actual composed full-cohort preflight is running under
`predictor-learned-availability-v14-20261028-preflight01`, with the original
four routes and all 57 public requests per candidate. Provider reconstruction
finished in 237.993 seconds. No fresh validation outcome, seal or GPU request
has been produced. The host provider is still preparation-conditioned solo
means: the joint host bank has not become a counterfactual release provider.

Full-cohort v14 preflight01 completed all four executions but every forecast
was UNKNOWN at `CONTEXT_DEADLINE_EXTENSION_WITHOUT_RESTORE`; its completeness
gate correctly prohibits seal/collection. The preserved pre-repair source is
in that artifact's `source-snapshot-before-repair`. A frontier change exactly
at SERVICE queried the new context's SAVE hazard before the already scheduled
SERVICE+1ns review expired. The oracle now retains the earlier scheduler review
without changing the hazard's own remaining budget or SAVE prediction. The
regression failed before the fix; 27 related tests pass after it. Independent
actual-controller reconstruction reproduces the old failure for all four
candidates and verifies the fix (`predictor-context-deadline-independent-
review-v14-20261028-cpu01`).

Preflight02 passed that first error and continues the full DAG, but is slow.
A separate actual-controller cProfile probe identifies quadratic copying and
hashing of the growing empty-interval list: 100/300/1000 intervals took
0.240/1.829/19.160 seconds under profiling, with 19.422 of 21.188 profile seconds
in deepcopy. `predictor-cumulative-hazard-history-cost-v14-20261028-cpu01` retains
the numbers, source identity and concurrent-run limitation. A new v15 compact
state plus separately retained full audit is being implemented; numeric
equivalence must be checked before its full-cohort preflight. This is a CPU
representation optimization, not new predictive evidence or an online result.

The slow v14 preflight02 was interrupted with SIGINT/exit130; its supervisor
report records zero complete candidate forecasts, unchanged core/public sources
and an explicit seal prohibition. Nothing from that attempt is accuracy evidence.

The v15 joint initial model source-rederives all 28 observations and identifies
the former missing-holder sample as an observed scheduler handoff gap: previous
text#0 SAVE is 12287ns before query, next G2 RESTORE is 31873ns after query.
That next G2 envelope has no kernel readiness upper witness, which remains null.
The prior preserves original scenario IDs/order/capture-balanced probabilities;
its prospective query never reads source future events. The v15 scheduler keeps
holder/RESTORE/age absent until the modelled future RESTORE, with readiness
point realization allowed only by the explicitly named hypothesis. All original
three UNKNOWN tiny cases and an additional actual-gap scenario finish under
that hypothesis; bounds-only correctly refuses a point forecast. This is CPU
mechanical integration, not independent initial-state accuracy.

`cumulative_save_hazard_v15.py` retains the frozen v14 numerical solver but
moves complete empty intervals to a scoped append-only audit ledger. Its sealed
recursive state stores count, total duration and hash commitments instead of
copying the complete history. The main v15 adapter exports every forecast's
full interval audit. Actual-model verification compares 2803 states exactly;
all interval rows and chains agree. Non-profiled 100/300/1000-segment v14-to-v15
costs are 0.07284/0.53284/5.54334s versus 0.02817/0.08565/0.30150s; the largest
case improves 18.39x. The 25 controller checks pass. These timings measure
controller computation only, not whole-predictor or online latency.

`predictor-learned-availability-v15-20261028-preflight01` is now running all four
original 57-request cohorts with the explicit initial-state and compact-controller
versions. Work/host/phase/resources/startup/six-score contracts remain the same.
The v15 seal tool has eight CPU fixture checks, requires complete numeric
preflight, and after freezing independently recomputes every forecast; it never
seals copied preflight cells. No new production seal or validation GPU run yet.

### CPU preflight continuation: collector repair and static provider compilation

The active-background batch collector's repaired source is
`6976145ec7c59ff848cef1bfc8e17b745df635f117f0b56bf8dea3c1d7c22dfa`.
Independent review `batch-active-background-independent-review-v3-20261028-cpu02/report.json`
(SHA `4a9d57c0732f1954ca0a018336245e87f14e6098c7b32b80e923160ea2c71c53`)
passes 18 CPU checks, including false-prime epoch refusal, independent G2/OCR
arrivals during blocked text generation, partial evidence persistence, and
supervisor TERM/KILL with complete process-group exit before releasing the GPU
lock. This is an execution-contract result, not GPU batch performance.

The v15 full four-by-57 preflight continues without source edits; at 16 minutes
wall time it has not yet produced the first candidate. A separate v16 CPU
representation materializes the original coordinate-only work/host/phase
providers into immutable source-bound tables. It preserves every node/edge,
original profile identity, lifecycle/resources and learned v15 scheduler. Six
unit checks pass, comparing full and drain-limited numerical trajectories,
immutability and source-change rejection. Real-source diagnostic prefix
equivalence/profiling is now running under
`predictor-static-compilation-v16-20261028-cpu01`. No complete-cohort speedup,
independent accuracy, protocol seal or GPU result is claimed.

### Full preflight process failure and v16 retry

The v15 preflight process exited with native segmentation fault (139) before
writing any candidate. No cgroup OOM event was recorded; no usable core/native
fault stack was available. The final periodic Python stack is not a diagnosis.
Original sources are unchanged. Failure evidence is retained in
`predictor-learned-availability-v15-20261028-preflight01/supervisor-report.json`
(SHA `e4d0aa39997984279d4b5f392835058b3284d0cb2c6c4a1922ba9aed82ec0bb5`).

Actual v16 static-provider diagnostic preserves all 57 public rows and exactly
compares numerical trajectories through 1s (9348 events/3346 nodes) and 3s
(58554 events/42493 nodes). Independent rereading also finds complete scheduler
JSON equality, with no fields removed. Profiled times improve only
6.9588→6.6541s and 29.2018→28.8508s: compilation does not resolve the dominant
learned context computation. After that run, immutable provenance and explicit
implementation dependencies were strengthened; eleven independent/current
unit checks pass. Independent review receipt is
`predictor-static-compilation-independent-review-v16-20261028-cpu01/report.json`
(SHA `223c047b98a81134f1b89de85100f6977b99cf90aa406273e4d8a7870edd83c8`).
The actual-prefix sources predate the provenance-only repair; the report
explicitly distinguishes those evidence versions.

The first parallel v16 runner was interrupted during initialization after
independent review reproduced a progress-observer bug: native_collect.save
uses exclusive creation and cannot update one checkpoint path. All four direct
CPU workers were reaped, zero numeric candidates were produced, and the failed
runner remains unmodified. Failure receipt
`predictor-learned-availability-v16-20261028-preflight01/supervisor-report.json`
has SHA `f0c94e6420bc3613e7d59f5573e4058ba8f8200bf155e0b5f70051ec36c116d2`.
A separate preflight02 uses atomic checkpoint replacement (two-update
regression passes), retains supervisor failure reports, and now runs the full
four original 57-request forecasts in independent CPU processes. No periodic
faulthandler signal is used. No GPU validation, seal, accuracy or gain is claimed.

Joint host evidence review is now frozen in
`predictor-joint-host-provider-v17-20261028-training01/report.json`. It preserves
56 scenarios/88 members and identifies two missing OCR semantic checkpoints.
Fourteen multi-request natural jobs contain seven cases where solo-derived
semantic-start order differs; one case remains different even with joint
preparation P alone because handler-start offset H matters. These are training
descriptions, not independent accuracy. The new strict checkpoint/vector
provider is not a counterfactual host-release model; see
`docs/joint_host_preparation_candidate_v17_20261028.md` for H/P equations,
non-double-counting constraints, workload-support limits and data requirements.

### Full v16 result, queue integration and lifecycle diagnosis

The full v16 preflight02 has now finished all four independent CPU workers
(exit 0), each with the original 57-request cohort and original 30s+120s
window. All four return UNKNOWN with
`COMPLETE_DECLARED_FLOW_DAG_AND_PARENT_RECEIPTS_REQUIRED`. The source receipts
are unchanged. The runner explicitly records
`INCOMPLETE_PROHIBITS_SEAL_AND_COLLECTION`; successful process execution is
not a numerical prediction success. Its wall time is 1411.690s. No production
seal, independent GPU validation or utility-gain result follows from this run.
The original v11 score-rejection path discarded its normally returned engine
forecast, so these files cannot identify missing finishes retrospectively.

Read-only training-evidence closure06 passes for the declared operational
scope: 5868 materialized sources, 926 normal JSON objects and 17 declared
historical boundaries; all sources are unchanged. This does not claim full
transitive reconstruction of all failed historical campaigns. Independent
resolver review is recorded in
`predictor-scoped-training-independent-review-v15-20261028-cpu01/report.json`.

The v19 queue sidecar now enters the actual v16 engine at forecast(rows, now),
precomputes offered work for all declared future inputs, and separately tracks
arrived FIFO waiting, remaining reference activity work, oldest age and recovery
episodes. Unmeasured SM/DRAM/L2 demand remains unknown. Independent CPU review
`predictor-queue-diagnostics-independent-review-v19-20261028-cpu01/report.json`
(SHA `4782d0e89ab7c432ee50e6080c462b0fe6a050a1cde18b9e352884b8756819ce`)
passes all six tests; two-task branched trajectories and hand-calculated six
terms are unchanged. This proves instrumentation mechanics, not queue accuracy.

The separate v20 wrapper retains only an actual normally returned engine
forecast when the unchanged original scorer returns UNKNOWN; it adds a
diagnostic-only field, never metrics or gate eligibility. Engine exceptions do
not invent an engine return. Four focused CPU checks pass. An actual-source
3s diagnostic with all 57 public rows is running in
`predictor-queue-retention-v20-20261028-diagnostic01`; its cutoff is explicitly
not a full-horizon preflight and will never be sealed.

Read-only lifecycle diagnosis
`predictor-idle-owner-selection-v20-20261028-cpu01/report.json` finds a concrete
model error: in the 1s prefix, 993 of 1985 modelled switches select G2, although
the balanced public cohort has no G2 requests. At least 429.167ms then has
ready text GPU work with zero modelled GPU progress. The 3s prefix certifies
842.585ms of such intervals. This is a CPU model pathology, not a measured
physical GPU slowdown. Conversely all 715 training not-frontier targets have
an open owner NVTX phase, and 579 have an already-returned launch API witness;
filtering all not-ready owners would destroy valid training examples. The next
version is therefore separating handler/issued/ready lifecycle state, retaining
physical context labels and identifying unsupported states explicitly.

The actual-source v20 3s diagnostic has now completed: original 57 public rows,
58554 events, three STARTs and two parent receipts. All original numerical
engine fields match the plain v16 trajectory exactly after separating the new
queue diagnostics. Initialization took 263.017s; plain/diagnostic execution
took 14.416/13.303s. This single timing pair is not a speedup claim. All declared
future input profiles are accounted for, while only three arrived requests
enter live accounting. The unchanged scorer rejects the incomplete prefix and
the wrapper retains its actual returned forecast as diagnostic evidence.
All 184 listed source hashes were checked by reading bytes
before/after. The executed pre-repair v20 source and author test are archived
with their original path/SHA in `executed-source-snapshots.json`.

Independent review of that v20 source separately found two real boundary
defects: the inherited stat-only cache can miss same-length byte edits when
mtime/ctime also remain unchanged on this filesystem, and the outer guard did
not include compiled training references. The failing review is preserved in
`predictor-unscored-retention-independent-review-v20-20261028-cpu01/report-before-exact-hash-fix.json`
(SHA `227af7bb4c23147f08b9d5319e708c4cedb01f3b85db8e24d89c5573343c21a7`).
After the actual run fully exited and its old source was archived, v20 was
repaired to stream SHA over actual bytes and include the inner engine's complete
source receipts. The original frozen core cache was not changed. Four author
checks pass; independent repair review is pending. The earlier prefix result
does not certify this later source-guard version.

The full v16 UNKNOWN evidence has also been independently streamed and hashed,
one file at a time (9GB, peak RSS 70356KiB). All four complete cohorts, saved
worker exits, 36 frozen source receipts and final UNKNOWN states agree. The
independent report is
`predictor-v16-preflight02-independent-readonly-20261028-cpu01/report.json`
(SHA `4b8d010f12672f089674af9c5cb51dae7a7f32a2fbf4074de8eed0a30fd0324a`).
Counts labelled without-offered-work in that diagnostic mean absence from the
recorded ready frontier, not absence of an active handler or physical work.
The narrower confirmed no-request G2 finding remains distinct.

A separate v22 reference feasibility diagnostic is running all four original
57-request routes with per-node rate one and no switch overhead, preserving
original DAG/work/host/FIFO/lifecycle/deadlines. Resources are disabled only in
that diagnostic. This distinguishes failure within the frozen reference-model
class from the learned selection pathology; it is not a physical GPU capacity
bound, predictor candidate, six-term score, independent gate or gain claim.

Actual weight-sharing feasibility is documented in
`docs/actual_weight_sharing_feasibility_v18_20261028.md`: text and G2 are
different bases, layouts and quantization formats; no large common arithmetic
weight block is certified. Same-weight text batching has a real backend path.
The NumPy shared-linear prototype is not represented as HF INT8/LoRA sharing.
Batch collector CPU contract repairs remain separate from GPU performance.

## Lifecycle integration and full-cohort result, 2026-10-28 08:27 UTC

The v22 relaxed reference diagnostic has completed: all four original routes
finish 57/57, with last receipts at 29.765272202s, 33.082860847s,
29.765272202s and 29.765272202s. It disables physical switching and resources;
these values are not physical GPU predictions or scored routing gains.

The final v21 mark and v24 full-risk hazard training models are frozen as
training03. The hazard includes 559130 risk intervals, 22383203487ns exposure,
35799 SAVE_END events, all 35737 marked events and 62 additional physical SAVE
endpoints lacking next-owner labels. Empty-frontier SAVE events are retained.
The final independent lifecycle review has 45 passing CPU checks and is
`predictor-lifecycle-v24-v25-independent-review-20261028-cpu01/report-after-closure-audit.json`
(SHA `2d3a900b5cda9ec9fdeb57266cc5113456a8ca96716df761c555d35874377559`).
Capture holdout fit is training evidence, not independent GPU validation.

The actual v25 balanced full-57 run has exited. It returns UNKNOWN with
`ENGINE_SKIPPED_PREDICTED_SAVE_BOUNDARY_NO_REWIND` at 10.748970283s. The full
result is retained losslessly in `predictor-lifecycle-v25-20261028-preflight01/forecast.json.gz`,
with complete decompression SHA `c6af8735271cf589272aa4d9d18e643edeee571bcd5479b6f03648ab133876f0`.
There are no six-term metrics and this run cannot authorize validation.
Its original report's implementation-guard claim was too broad: the declared
list omitted v20 retention and idle-owner audit implementations. Exact executed
snapshots were saved, and `source-claim-correction.json` explicitly limits the
claim without overwriting the original report.

The failure is reproduced on a small causal fixture: at 100000ns, a new active
handler changes lifecycle but not the GPU frontier. The oracle moves SAVE from
693148ns to 159315ns; v25's inherited scheduler still returns the old boundary.
New v27 explicitly accepts the earlier review into scheduler state before
advancing, including empty-frontier intervals. A truly missed SAVE still fails
closed; immutable handoff and the original budget remain intact. Its source
closure starts at the actual v27 factory and includes all transitive local
imports. Four targeted regression checks pass; independent review is underway.

Whole57 training collector prepared03 has passed eight run audits plus eleven
combined author/independent CPU checks. It preserves all original arrivals,
30s measurement, 120s drain, four resident owners and uncapped execution, using
new training request IDs. Actual collection will start after storage is ready.
The exact four historical v16 UNKNOWN JSONs have now been losslessly archived:
all 9,573,456,832 original bytes were completely decompressed and SHA verified;
the archive occupies about 173MiB. A separate read-only verification has passed.
Retirement of only those redundant originals is running with durable recovery
instructions. Historical reports and original byte hashes are unchanged.

The independent GPU gate remains FAIL. No independent validation, mechanism
gain or end-to-end routing benefit is claimed by these CPU and storage steps.

## Whole57 training and v27 actual result, 2026-10-28

The first actual `whole57-joint-light-v23/balanced` training run has completed:
57 measurement requests and 12 priming requests, no execution/telemetry errors,
clean four-owner shutdown and source audit PASS. The original 30s arrival
window is preserved; observation ends at 34.167682113s. A dispatch pipe write
takes up to 1.289979s despite maximum enqueue lateness of only 0.340ms. These
are host-boundary training observations, not evidence of a pure GPU delay.
The original review is PASS (`b16ef5367798c8406f1c9cd1b4d464f1b99a9fe965d17984adced29269611a3b`).
Independent read-only verification rehashes 13 bound files, including the
2,059,465,474B derived parsed JSON, raw, NSYS and SQLite; it recomputes all
57 host lifecycle rows exactly. Its report is
`predictor-whole57-joint-light-balanced-independent-review-20261028-cpu01/report.json`
(SHA `9a43ea350ea8e198a85813d4f328a34f4e5567e0033142c8dcbdb9e894fbd684`).

The actual v27 full-57 balanced preflight has exited after 484.557s wall time.
It passes the previous missed-SAVE point, then returns UNKNOWN at 14.819039758s:
`UNSUPPORTED_SAME_CONTEXT_REACQUISITION_PREDICTION`, with OCR currently held.
The latest lifecycle/hazard and exact scheduler evidence are retained in the
lossless result (`forecast.json.gz`, decompressed SHA
`8d628d1d87870df6e75e904dd6769a58489431880468a0d136bdc60749fb780c`).
All 191 static implementation refs and compiled training refs were checked
by bytes; the earlier v25 closure defect is not inherited. There are still
no six-term metrics or validation eligibility. The v27 independent review
has 22 passing checks and no must-fix source/causal defect; that does not
certify the numerical/physical model. OCR class/support and physical
same-owner transitions must be diagnosed before replacing the rejection.

Independent endpoint work also establishes 378 kernel START/END counterexamples
to deterministic binary owner gating in the old training captures. The v27/v28
endpoint regressions worsen tails; v29 conditional signature/issued-queue KM
still gives only 92.64%/92.01% coverage at the nominal 95% END quantile on the
common support, and is NOT QUALIFIED. These candidates remain separate from
the actual forecast. See `physical_progress_endpoint_predictor_v28_20261028.md`
and `kernel_conditional_survival_v29_20261028.md`.

Further training remains ordered light/natural within each original route.
The first derived parsed JSON is scheduled for exact lossless compression,
full decompression SHA/CRC verification and recoverable retirement; original
NSYS/raw/SQLite and immutable reports remain. Consumers must explicitly bind
the gzip receipt and original decompressed byte identity, or restore the
original file. A compressed path never impersonates an old JSON receipt.

## Whole57 continuation: observable context and horizon recovery

Light/natural balanced and skewed have now all completed and passed their frozen
reviews, retaining 57 measurement plus 12 priming requests per run. Natural
balanced report SHA is `7c991196f6fcae5a5d06d5613b3fa5db5c1c402e53073ec6c6dce09816d0ead8`;
light skewed is `2615cc6aa69db6d4fd0e49d9501e43d746decbb7180ce92744eafb6bc3779536`;
natural skewed is `59eb3628178525960615c9a6615d72ebdb5d832bbbcf5f1a3777c4042cf11043`.
Independent natural-skewed review is
`predictor-whole57-joint-natural-skewed-independent-review-20261028-cpu01/report.json`
(SHA `3f6f4fcce1b35eca0a505314c157f054417cf53afa5a3b83e19aeca147086151`).

The light balanced and skewed derived parsed JSONs are now losslessly archived
with full original-byte SHA/length/CRC checks and durable restoration commands.
Raw, NSYS, SQLite and completed reports remain. Skewed archive manifest SHA is
`13cafa4c1d9488e37087efc43e90bed995bad06cdbc0bab49ca1b399b1d0e65f`.
Two older completed multimodal derived parsed copies were similarly archived
and recoverably retired to preserve the original disk guards for remaining
collections. Their v37 manifests are
`83a61944588218ed2b0ecaf82edde112c8bd871ab7684b67876f0ea925c70dbf`
and `2deeb60d9ba859c01b7874c5fe776c64446b2c7c64c57f5a45e407109c35aba9`.

New v34 reads native transition/exposure ledgers and checks complete event,
adjacency, driver-context, route, raw/context, and exposure-plus-gap accounting.
14 author/independent CPU checks pass. Native stage presence and engine handler
activity are not treated as equal features: both remain diagnostic, while the
first candidate fits attained age and held/previous holder only. The new v38
bridge/factory preserves original cumulative timing, source closure, owned-work
hypothesis and same-holder rejection. Four-route fitting is still pending the
remaining actual light captures. See `observable_context_training_v34_20261028.md`.

Queue recovery work now has actual positive-backlog reader-visible training
prefixes for two natural captures, explicit horizon labels with correct
administrative censoring, and a separate source-bound risk-domain bridge. The
bridge remains unqualified until all four actual natural members pass. The
observed-risk sidecar consumes actual reader prefixes and separates prediction
from delayed-label evaluation; it is not an executable DAG checkpoint or a
completed rolling feedback controller. Original six scores and rho remain.

The independent GPU gate remains FAIL. None of these collection, training-data
or CPU-interface results establishes physical prediction accuracy or routing
gain. Spill collection, final training, full CPU preflight and new independent
prediction validation remain required before mechanism ablations.

Actual light/natural spill_low have now passed collection review (light SHA
`3192fdec0b9afeff8e3cf49d51d819969d9d1a903b906ff39d736c091dfe2f4e`,
natural SHA `81b43f2ace6797d95ba55be70ca674f9ad43adbf855c3b1eae89bbfbac6c5b7c`).
The light derived parsed JSON was exactly archived and recoverably retired;
manifest SHA `8d9225b7f8562669d16bf8a2382e14522081831dd9d6210814ed7307b5ed6d0b`.

The spill capture exposes a real v36 shared-adapter defect: sequence numbers
repeat, and indexing SAVE by sequence alone creates 6,162 negative restoration
ages. The rejected original output remains with
`predictor-whole57-spill-low-SAVE-identity-rejection-20261028-cpu01/report.json`
(SHA `c6e9ebde72c0c5d4f2bd8cb07f5b38e000dc64ddbd5d276ad8c14ed6a2f7813e`).
New v39 uses exact timestamp/context/sequence identity and reuses the correct
native extraction without a new GPU run. Corrected shared report SHA is
`10a11924fc49ec17ec24a664a687f4ee898a4a2876770ea0d004dc8fd84fe5b5`.
The v34 consuming accounting now uses the same key; its earlier executed CPU
source/test bytes were preserved first. The updated combined suite passes 28
checks. Corrected data retains 46,089 marks and all 46,219 physical SAVE ledger
events, including 130 without next marks. Last-route fitting remains pending.

## Completed eight-run training and current prediction gates

Eight actual whole57 training captures are now complete: 456 measurement and
96 priming requests, eight PASS collection/source/independent reviews, original
30s plus 120s drain, no request cap and clean GPU process shutdown. The final
actual inventory is
`artifacts/predictor-whole57-training-eight-run-actual-inventory-v23-20261028-cpu03/report.json`.
The v34 four-light native model was fitted with complete event identity and
exposure accounting. Actual balanced/skewed preflights both return
`UNSUPPORTED_SAME_CONTEXT_REACQUISITION_PREDICTION` in only-held-handler states.
The v38 runner then encounters its second exclusive progress-file create;
historical results remain. New v43 reruns the complete four-candidate CPU batch
with independent per-candidate progress persistence, changing no model/guard.

v41 explicitly projects original worker handler boundaries under an
uncalibrated median-clock mapping. Independent reconstruction preserves all
physical SAVE/mark/exposure/gap ledgers and finds zero SAVE in 67.751908232s
of only-held-handler exposure across four captures. It does not impose zero
hazard. v42 has completed an actual four-capture handler-conditioned fit,
with projection uncertainty always retained; v44 supplies a separate explicit
unvalidated bridge for full-cohort preflight. See
`docs/observable_context_training_v34_20261028.md` for exact evidence.

The queue-risk branch has qualified all four natural training contexts and
actual READY payloads. The v33 campaign is performing 12 whole-capture
leave-out fits and sealed prefix replays (four captures times three horizons),
plus fixed joint-horizon monotonic projection and three final training models.
Available support is heterogeneous; skewed 1s and 5s training-diagnostic Brier
scores are worse than their training class-rate baselines. These results are
not evidence of uniformly improved risk, calibrated probabilities or completed
rolling feedback. Fresh validation uses a distinct evidence schema; its
execution-domain expansion and actual READY must qualify before any score.
It must export visible features, seal frozen predictions, and only then export
and join labels. Reusing a training collector nonce as a validation run is
insufficient. Preparation is underway; no fresh GPU validation has started.

The independent service/queue/P95/Delta-U/rank gate remains FAIL. Mechanism
ablations and fair final routing comparisons are still downstream of that
gate. Original six equal scores, cohort/arrivals/routes, rho, owned memory and
energy boundaries are unchanged. No empirical or occupancy quantity is
relabelled as physical SM/DRAM throughput.


## Four fresh risk validations and expired-review scheduling diagnosis

The v39 independent queue-risk branch has completed all four new native
executions, each retaining the original 57 measurement and 12 priming requests.
Predictions were frozen before labels were exported. The delivery is
`artifacts/predictor-independent-risk-validation-v39-20261028-delivery01/report.json`
(SHA `30844517faf73e28e5b040cb1eea6a49737845425d0bfd6be34dac3e146ed183`).
Across 5,528 causal reader prefixes, 4,388 have positive backlog; 4,363 are
jointly supported and 25 skewed prefixes remain UNKNOWN. On that same support,
1/5/10s raw Brier is 0.036949/0.077647/0.081940 versus fixed training-rate
baselines 0.091614/0.188215/0.248801. The predeclared equal-weight monotone
projection gives 0.042206/0.074226/0.076226. Both variants remain reported;
validation labels do not select a winner or refit the model. This is a fresh
same-recipe/content causal receipt replay result, not online rolling forecast,
GPU service identification, calibrated task P95 or Delta-U/rank qualification.

A separate v45/v46 candidate models observable reader START-to-RESULT remaining
wall time, preserving worker READY and physical GPU boundaries as different
facts. Four whole-capture holdouts and 852 surviving-prefix landmarks were
independently recomputed from saved coefficients. Every nominal95 quantile
requires extrapolation outside current reader-state exposure; some common-
support short-horizon comparisons are worse than the baseline. This candidate
is not promoted into the main service or utility model. Its actual independent
review is
`artifacts/predictor-reader-completion-hazard-v46-actual-independent-review-20261028-cpu01/report.json`
(SHA `ca1ed89b8c2af27e310012aa280963064d480e7dc02d0d91993d776dd1b78405`).

The v44 full-cohort preflight completes balanced with all 57 starts/receipts and
six numeric terms (U=-2.015589362402846), but this is still unvalidated output.
Skewed returns `ENGINE_SKIPPED_PREDICTED_SAVE_BOUNDARY_NO_REWIND`. An independent
minimal CPU fixture reproduces the precise defect: an expired scheduler review
is not replaced by the oracle's slightly later future review when the GPU
frontier is empty. The inherited scheduler drops the expired deadline, skips
the newly predicted boundary, and the original no-rewind guard correctly
rejects. This is an event propagation defect, not evidence for a same-holder
fallback or relaxed timing tolerance. The original v44 run/source/results are
preserved. A separate v49 scheduler and v50 factory/preflight are being prepared
for correction and a complete new four-candidate prediction. The v50 CPU runner
uses four private forked model instances after immutable provider compilation;
no candidate inherits another candidate's event clock or runtime state.

The v48 artifact-binding preparation explicitly permits reuse of exact complete
preflight artifacts, with producer timing disclosed and independent original
six-term reprojection. Any UNKNOWN or incomplete cohort prohibits sealing.
Neither v44 balanced completion nor risk-validation success overrides the
original service/E2E/queue/P95/Delta-U/rank gate. Mechanism ablations and the fair
four-group routing comparison still follow that gate.


## All-idle SAVE extrapolation and next statistical correction

The original v44 run has now finished all four candidates with unchanged source
bytes. Final report SHA is
`1956802a8bddd4fbbd67cfaedeafdcf7407c5128b24c9f8d65109709fd245212`.
Balanced and spill_low complete all 57 requests (U=-2.015589362402846 and
-2.870815292882536 respectively); skewed is the expired-review defect and
spill_high is `NO_VISIBLE_ACTIVE_REFERENCE_TARGET`. No protocol was sealed.
The independently reviewed v49 repair and v50 private-process wiring pass 28
CPU tests; the new full v50 preflight is running under a separate source freeze.

For spill_high, the simulated SAVE occurs exactly at 6,145,742,758ns, with all
12 lifecycle fields explicitly false and an empty GPU frontier. It is not a
missed event, observed physical SAVE, missing request or unknown boolean. The
next declared request has not yet arrived. The old bridge requires an active
reference target even in this all-idle state; its conditional distribution has
zero mass. The raw physical mark instead favors text#0, but that mark is flagged
as an unseen held/previous/worker-handler state. A separate v51/v52 bridge has
been written for possible explicit all-idle physical extrapolation, with no
active reference target and no request progress; it is not promoted to a
validated service mechanism.

More fundamentally, an independent stream count of the original four v41
projected-handler training captures finds 91 all-handler-idle risk intervals,
14,697,318,765ns exposure, zero physical SAVE and zero physical next-owner marks.
These are training projections with their original clock uncertainty, not a
proof that idle SAVE cannot occur. A constant positive pooled log-linear rate
can nevertheless extrapolate a short idle lifetime unsupported by those data.
The next candidate will explicitly model the all-idle counting-process exposure
and retain nonzero rate uncertainty (fixed Jeffreys Gamma prior, posterior mean
point hazard, source-bound counts/exposure, whole-capture leaveout). No exposure
means UNKNOWN; zero observed events will not be converted into an impossible
transition or a zero hazard. This statistical candidate must preserve the
original cumulative budget and exact event clock and pass a new full-cohort
preflight before any independent GPU validation. Its constant state-hazard and
projection assumptions remain unvalidated; zero-event holdouts cannot establish
positive-event timing accuracy.


## v50 full preflight completion and v53/v54 candidate

The repaired v50 run has completed all four original cohorts in 1425.856s.
Balanced/skewed/spill_low each have 57 numeric starts/receipts and six terms;
U is -2.015589362402846/-2.2720277206885/-2.870815292882536. Balanced and low
metrics equal their previous v44 metrics exactly. Skewed now passes the old
missed-SAVE boundary. High still returns the predeclared all-idle target UNKNOWN.
All producer sources remain unchanged. This is an implementation correction,
not an independently established accuracy or improvement result; sealing is
still forbidden for this candidate batch.

The v53 idle exposure fit has completed from the four original training captures
only. Holder-specific posterior mean SAVE rates are text#0=0.115482269/s,
text#1=0.0584109115/s, G2=0.276609059/s. OCR has no exposure and remains UNKNOWN,
with no pooling. Equal-tailed posterior credible bounds and separate zero-event
frequentist one-sided upper limits are both reported with different meanings.
The model is
`artifacts/predictor-idle-physical-save-v53-20261028-training01/model.json`
(SHA `ca82687c16f883844a91b2c7ab9b83a3ab4fa3bc5cb8312d2ffd8e32969a9590`).
Six meaningful model/controller tests and two actual-candidate factory connection
tests pass. New v54 binds the same four capture identities, both source model
hashes and unchanged v42 active mark. It has no v51 physical-idle mark fallback;
if an idle SAVE nevertheless occurs without a supported active reference target,
it still returns UNKNOWN. No training support or guard is fabricated.


## Independent v53 confirmation and immutable v54 run

The v53 actual independent reviewer reconstructs counts, exposure and ages from
all original risk rows and independently recomputes 345 exact/numeric quantities
using NormalDist-square Gamma(.5) quantiles and a negative-binomial recurrence,
without calling the fitted posterior helpers or SciPy. Ten tests pass. Report:
`artifacts/predictor-idle-save-v53-actual-independent-review-20261028-cpu01/report.json`
(SHA `3b6313e9bc3d1471d354f0f61db756a8b85499efd0bce24b0e82420ceb4d692b`).
The v54 factory separately passes seven actual-connection and independent wiring
tests; its report is
`artifacts/predictor-v54-factory-independent-review-20261028-cpu01/report.json`
(SHA `4d317b75b7f8e92baf4bf93085f1644d74e9a54380f028a653157f083bf3cca9`).
The full v54 CPU run is now in progress, with 358 source receipts frozen and
verified unchanged. All four routes have passed the former high idle failure
clock. This does not establish that the final cohorts or accuracy gate pass.

Independent preparation review found two separate integrity defects before
production: v48 did not verify each numeric START value against the completion
record; supervisor v1 did not require the exact review/preparation in protocol
sources or enforce plan threshold identity. These frozen versions remain intact.
New v55 verifies exact integer START, the full producer implementation closure,
source-before equality and honest preflight artifact reuse. New supervisor v2
binds v55 exclusively, exact original thresholds/tie tolerance, prepared cells,
training index, review and complete evidence closure. Actual source/gzip/six-term
reprojection checks and independent real CPU process cleanup tests pass. Neither
code review is substituted for the forthcoming actual production-preparation
review. No protocol, prediction seal or new main GPU run has been produced.

## v54 completed: physical mark versus request eligibility mismatch

The final v54 four-cohort CPU preflight completed in 1429.449913s with all
358 frozen sources unchanged. Balanced/skewed/spill_low each completed all
57 starts and receipts; high is UNKNOWN, not a numeric comparison. The final
report is `artifacts/predictor-projected-handler-v54-20261028-preflight01/report.json`.
No production preparation, seal or new main GPU validation is eligible.

The independently reproduced high failure occurs at exact SAVE time
31,485,677,331ns, held G2, restore 29,971,125,980ns, age 1,514,551,351ns.
Only G2 currently owns active work. The raw physical transition model actually
prefers text#0 with probability 0.9999977443173469; held G2 has only
0.0000022547182354174517. The v44 bridge conditions the physical distribution
on request-work eligibility, turning that tiny held mass into probability one.
The same-holder guard correctly rejects this changed statistical target.
The 105,028 qualified training physical marks contain zero same-owner and
zero same-driver-context transitions; inventing a self-restore is unsupported.

Removing eligibility conditioning is a semantic correction, but does not
by itself identify a supported transition or GPU service. In the original
projected-handler training data, the sole-held-active domain has 67.751908232s
of exposure and zero terminal SAVE events. Thus the pooled additive hazard can
also extrapolate a short lifetime in a zero-event domain. A separately named
candidate will preserve raw physical mark probabilities and fit an explicit
holder-by-handler-mask counting-process hazard across all 64 strata, with a
fixed Jeffreys prior, retained zero-event uncertainty and UNKNOWN when exposure
is absent. It must report whole-capture diagnostics and projection/age/constant
intensity limitations before any new full-cohort run. Original guards, hazard
budget continuity, offered cohort, score and windows remain required.

## v57 arithmetic confirmed, constant-state model rejected before preflight

The all64 Gamma-Poisson candidate was fitted once on the original four v41
training captures. Its actual independent audit reconstructed 107,685 risk rows,
all per-capture/state age unions and 130,426,645,735ns / 105,141 physical SAVE
events. There are 2,880 exact and 1,388 numerical comparisons, plus a separate
metadata supplement checking all four unique holdouts, original manifest,
parameters and full coverage. These arithmetic checks pass; they are not a
prediction qualification.

The subsequent adequacy check rejects the constant-with-age hypothesis:
nominal 95% whole-capture count intervals cover only 45/71 supported cells
(63.38%). A G2 state has 12,672 SAVE events at attained ages 20–50us, compared
with constant-model expectation 870; at ages 1–2ms it has 24 versus 4,496.
The event-time rescaling distribution also departs strongly from Exp(1).
See `artifacts/predictor-projected-state-v57-adequacy-20261028-cpu01/report.json`.
Therefore v58 has only a prepared configuration/factory and CPU wiring tests;
no full v58 preflight, production preparation, seal or GPU execution has run.
Next diagnostics retain the original age bins and compare conditional exposure
on common training support, without selecting new bins from held-out outcomes.

The physical mark-domain semantic correction is implemented separately in
`physical_context_mark_bridge_v56.py`: raw resident probabilities survive,
request eligibility remains diagnostic, and the inherited same-holder guard
still rejects a raw self-transition. Actual scheduler tests show restoration to
an inactive resident cannot grant progress to another owner's request, and
pending transitions/age/budget remain immutable. This fixes target semantics,
not the unvalidated physical-context-to-reference-service assumption.

The isolated rolling-visible interface now executes two real CPU suffixes in
the generic segment simulator, using explicit current progress receipts and all
remaining public arrivals. It recomputes queue and full-cohort P95; missing
progress remains UNKNOWN. The main lifecycle engine still has no compatible
resume contract, and no native progress producer or prefix energy accounting
is asserted. Six tests pass; details are in
`docs/rolling_visible_queue_v1_20261028.md`.

A current read-only `nsys profile --gpu-metrics-devices=help` listing confirms
the RTX4090 still reports `ERR_NVGPUCTRPERM`. No workload or permission change
was performed. The exact output is retained at
`artifacts/predictor-gpu-metrics-capability-20261028-readonly01/report.json`.
Kernel activities remain available; they do not identify SM/DRAM throughput.


## Device progress observer v1: CPU review complete, GPU smoke pending

The new opt-in `device_progress_observer_v1.py` and isolated
`semantic_progress_worker_v1.py` passed 24 author, independent and original
semantic tests. Root repeated the same 24 tests successfully. The independent
review is `artifacts/predictor-device-progress-observer-v1-independent-review-20261028-cpu01/report.json`
(SHA `381959d6f30c30164b7de5804d1a08e0e09a71c488500f0cb11dc9761b8c144d`).
Observer SHA is `7b5e9466824c277e2923eea216bcf4fa3325f52caa6e9c2cf989b784f281d4f0`;
worker SHA is `53faf12524c44cb495337b55a639292da1160297f3651c5f7adf5aa0630bffb6`.

The probe records CUDA events around language forwards on the current stream,
queries without adding synchronization, and retains host brackets and actual
parent receipt times through PHASE observation events. Device identity is checked
along with stream pointer; sink receives a deep copy. Errors never become
completion evidence. CPU tests did not initialize CUDA and cannot establish
physical API compatibility or negligible observer overhead.

Coverage is exactly the three language owners. The retained OCR adapter has no
new device event probe and remains `UNKNOWN_NOT_INSTRUMENTED` for this feature.
Single-stream event completion is not an all-stream GPU frontier, fractional
remaining work, exact device completion timestamp or effective SM service.
Natural and profiler-perturbed runs will use the same observer identity and
offered jobs. Smoke results are excluded from predictor fitting and validation.
The new profile supervisor/collector are still under CPU process-boundary review;
no new progress-observer GPU workload has yet been launched.


## Paired progress-observer collection: first real natural smoke complete

Final profile supervision CPU review passed 30 tests with 76 unchanged source
receipts: `artifacts/predictor-profile-training-independent-review-v1-20261028-cpu02/report.json`
(SHA `4be11231a7ecfef6d4afdc33d7e04cf0e5f5a209811e77995b9b8b4ac5625c13`).
It covers already-locked inherited lease, cross-session descendants and reaping,
prelaunch disk/time baseline, dynamic worker closure, no inherited hidden MPS,
and serialized NVML samples. Earlier reports and packets remain preserved.
The prepared training01 was retired before GPU after the monitor serialization
fix; training02 contains the final independently checked sources and all four
progress-enabled natural/kernel smoke/training manifests.

Natural smoke actually ran under the supervisor in
`artifacts/predictor-device-progress-paired-v4-20261028-training02/`.
The child completed in a 76.009911834-second supervised boundary with source
recheck PASS, no surviving owned descendants, four successful requests and four
clean worker exits. The strict paired smoke report is PASS. The three language
requests each have five forwards and ten queries, all recorded stream events
completed and instrumentation errors empty. OCR is not device-event instrumented.
The 3,229 board samples are time-ordered and contain no foreign/unresolved PID;
owned sampled peak is 28,986 MiB, with startup/priming/measurement/shutdown phases.

The first extra raw verifier rejects the normalized empty prediction on all
three successful raw `<No Answer>` outputs. This is a verifier/protocol mismatch,
not a device-event failure: original source-qa protocol explicitly maps this
marker to an empty QA answer. The original failed review remains; a separately
versioned verifier must CPU-reconstruct exact original output normalization
before accepting this case. No model output, request or measurement is changed.
Kernel smoke has now been launched serially; no training run or accuracy/gain
claim is made yet.

To preserve the original disk reserve, identical immutable source snapshots
were losslessly deduplicated using exact SHA-256 checks, retaining every original
path and bytes. No live executable source was linked or removed. The 5,593
replacements reclaimed 702,345,216 bytes. Full before/after per-file hashes and
paths are in `artifacts/predictor-source-snapshot-lossless-dedup-20261028-cpu01/report.json`
(SHA `5b077c6a82a8f4338f1c4281174221089fce07b56d654be2aa21b17866b1a21d`).
Original 18/13/6 GiB and 2400-second supervision budgets remain unchanged.


## Kernel smoke failed on ownership identity before measured requests

The actual training02 kernel smoke returned FAILED_NO_RETRY, with zero measured
requests and four priming records. All four owners reached READY and exited
cleanly; final source audit passed and all owned descendants were reaped.
The frozen collector v1 requires NVML PID process_group equal to collector group.
Nsight legitimately creates separate process groups, so every GPU worker was
misclassified foreign (maximum recorded owned memory zero). This is a real
collector defect, not evidence of kernel-prediction accuracy or memory pressure.
All failure raw, traces, execution and original sources remain unchanged.
New isolated paired campaign v5, collector v2 and supervisor v2 are being
reviewed using real cross-process-group ancestry/PID identity counterexamples.
No trained model, threshold, foreign-process rejection or budget is relaxed.
Training remains blocked until the corrected natural/kernel smoke pair passes.

The new `visible_device_event_ledger_v1.py` bridges the real observation events
to the existing host observer without misreading `event=observation` as a phase
boundary. Seven CPU tests pass. A reader-receipt-by-reader-receipt replay over the
natural smoke accepts all 30 queries across four host requests, while keeping
kernel_cursor null and main_lifecycle_resume_eligible false. The exact CPU
replay report is `artifacts/predictor-visible-device-event-ledger-v1-20261028-natural-smoke-replay01/report.json`
(SHA `0ffdb0f2f7162134204fc15fa63416697e64ae1571c9f31f55eb89705a3bef3d`).
Independent review is pending. No future unreceived event body is used, and
already visible host/device prefixes may not be rewritten. This is causal
evidence ingestion, not a main-engine rolling suffix forecast implementation.


## Visible device ledger independent review and v5 natural shutdown finding

The causal ledger independent review passed 18 tests (11 independent + 7 author)
and replayed the real natural smoke at 104 reader-receipt cutoffs, retaining
30 queries / four requests and no progress cursor or main resume eligibility.
Report: `artifacts/predictor-visible-device-event-ledger-v1-independent-review-20261028-cpu01/report.json`,
SHA `929a3654e99827e1aa974baa86bafcb5b450f356f8a26cb9157261f72420e383`.
The original natural smoke separately passed exact protocol reconstruction for
8/8 priming+measured requests and limited provenance/ownership review; the latter
report SHA is `45ece32fc57584ae36f54d73705cda0c463aeda2993caa487d247fca07eef104`.
No CUDA event API authentication or full-stream device frontier is inferred.

The cross-PGID v5/collector-v2 fix independently passed 42 CPU tests before fresh
preparation/execution. Its review is
`artifacts/predictor-profile-ownership-v2-independent-review-20261028-cpu01/report.json`
(SHA `31386656d19ff62fdb369ea934e32e65d42fa19be2615a980ddaf54c72b027ce`).
Fresh v5 natural smoke completed all four requests, 15 language forwards and
30 device queries with clean exits and unchanged sources. However, its raw
contains 20 unresolved ownership samples, all in shutdown after worker exit;
startup/priming/measurement contain none. NVML temporarily retains dead/zombie
PIDs after process exit, which cannot satisfy the live-ancestry checker. The
strict raw review failed, remains preserved, and no v5 kernel or training run
is authorized by this result.

## v6 full paired collection and stage response diagnostic continuation

The earlier preparation entries are historical. Both v6 smoke modes and both
first training captures have now completed without retries. Natural and kernel
training each retain the planned 26 jobs, 50 measured requests and 18 priming
requests, and clean supervised descendant exits. Natural has 4762 owned board
samples and sampled peak 34428 MiB; kernel has 4857 and 34540 MiB. Neither is a
steady-state P95 or final spillover experiment.

The complete training evidence reviewer is separate from the four-request smoke
reviewer. Independent counterexamples exposed three omissions: successful pipe
transport did not imply successful priming; startup/priming alias lists did not
have to match live ancestry; and dispatch/write/handler chronology was incomplete.
The new reviewer explicitly checks successful worker responses, every sample's
exact alias set, and branched send/write/handler/receipt chronology. Frozen GPU
sources and original raw data were not changed. Natural review01 remains an old
superseded artifact; use review02 and its 19-test independent review:

- `artifacts/predictor-profile-training-evidence-v1-natural-20261028-review02/report.json`
  SHA `f8ea8140acca709fcd7bf6d73ce96bb5a9a6eb599be8ba4ed9110cfba8dc04db`.
- `artifacts/predictor-profile-training-evidence-v1-natural-20261028-independentreview01/report.json`
  SHA `b627297dcf93f1c5242febcdd49ad7f9aed5364e7fa5c8c172f9edfd33843b79`.
- `artifacts/predictor-profile-training-evidence-v1-kernel-20261028-review01/report.json`
  SHA `accfebcc3f4000b596f31f1ef170b2a08fbcc67a2edb752069034ab7f2fb23d8`.

Kernel export and independent SQLite denominators show 869142 unique canonical
kernels, all attributed; nested phase copies were not summed. All 50 measured
and 18 priming pairs have identical actual work across modes. Dual NVTX clock
brackets admit conditional offsets, not calibrated device completion timestamps.
Native query returnValue remains zero for all records despite 230 raw false
queries; elapsed APIs are absent and event timestamps zero. Numerical event
authentication stays UNKNOWN. See paired evidence report
`artifacts/predictor-device-progress-v6-paired-training-20261028-evidence01/report.json`,
SHA `fd0eb5a4021989f3ecaafa0b9d0352ffb8968b87686a781506b61bd6e17485df`.

Independent stage-feature review found that BEGIN-notice-to-host-END paths must
not fit full host wall labels or stream elapsed. New `stage_response_training_v2`
requires the complete unchanged training review, reconstructs host exposures from
worker BEGIN to END using only each cutoff's processed prefix, and disables
stream fitting. `stage_host_partition_v1` partitions nested calls and gaps into
1028 nonoverlapping pieces across all 50 requests, exactly reconstructing
START-to-READY; response receipt and READY availability remain separate. The
14-test independent report is
`artifacts/predictor-stage-response-training-v2-natural-20261028-independentreview01/report.json`,
SHA `ff93bd4eb8938937e36a1ed540f91be16566b1800b76d07ff20183873b947ba9`.
This decomposition does not yet supply a conditional joint execution provider.

`stage_response_fit_v1` now runs a training-only lognormal reference-work
time-change likelihood, jointly estimating per-forward baseline nuisance means
and unconstrained interference coefficients. It includes the endpoint density
Jacobian, checks the integrated response Jacobian with all baseline means, and
keeps natural/profiler timing separate. Before fitting, features are rederived
from the complete current review closure to reject stale or changed evidence.
The initial one-capture fit is only diagnostic: baseline/count/phase-aware NLL
are about 839.95/798.16/798.16, and the extra phase effect is very small. The two
interference optimizers retain precision-loss/convergence flags. Full numerical
rank does not establish capture uncertainty, generalization, P95 or DeltaU.
No candidate is selected. Two additional independent whole captures per mode
are being prepared using the original 26-job design and unchanged budgets.

The original main prediction gate is still FAIL; no new main preflight, seal,
independent prediction experiment, mechanism gain or final four-group gain is
authorized by these training diagnostics. The goal stays active. Scheduler
snapshot components and the typed device ledger do not by themselves establish
main-engine rolling continuation; a separate v59 atomic-resume implementation
and independent equivalence review are in progress. No global cap1/cap2, scoring
weight, cohort, drain, stability rule or validation-label isolation was changed.

Completed historical duplicate JSON evidence was losslessly hardlinked only
after byte-identical SHA checks, preserving every original path and content.
The two receipts are `predictor-historical-evidence-lossless-dedup-20261028-cpu01`
and `predictor-historical-kernel-evidence-lossless-dedup-20261028-cpu01`; reclaimed
bytes total 1901211648. No raw evidence was deleted, no live source was linked,
and the 18 GiB launch gate / 13 GiB abort floor / 6 GiB increment remain intact.

The next isolated v6/collector-v3 candidate explicitly aligns the telemetry
window with the original startup-through-drain M definition: stop/join/final
sample while all workers are still live, after the drain-end boundary and before
worker close. Retain every sampled error and reject every foreign/unknown PID
within that complete window. Record monitoring_end before shutdown_begin; clean
shutdown/descendant reaping remain supervised separately. This does not promise
shutdown GPU-memory coverage, change original M, discard measured samples or
make the failed v5 raw pass. New CPU ordering review and fresh paired smoke are
required before fitting any new training data.


## v6 prepared domain checked against the original known-input validation

The new v6 packet is prepared at
`artifacts/predictor-device-progress-paired-v6-20261028-training01/`.
All four run plans and four supervisor-v3 manifests pass CPU audit. Natural and
kernel training jobs and inputs are exactly equal: 26 jobs, nine contexts
G/O/S0/S1/S1G/SG/SS/SSG/SSGO, two text contents and two OCR contents.
Both text identities (`tydiqa-new-native-02ea54b33b0492f20de11e0f` and
`tydiqa-new-native-e0ff7b578f9c742a6697469a`) match the original v5 sealed
protocol's known contents used in the 57-request evaluation, and OCR includes
its X51005442341. This checks input-domain overlap explicitly; it does not
transfer old timings or validation labels into the new fit. New v6 scope CPU
review is still completing; no v6 GPU run has yet started at this entry.

## Continuation: three paired captures, disjoint response clocks and queue composition

This entry supersedes the pending collection status above. All six complete
training captures (natural/kernel, independent capture01/02/03) finished with
26 jobs, 50 measured and 18 priming requests each. The four additional runs
were natural02, kernel02, natural03, kernel03, with no retries. The immutable
paired review is
`artifacts/predictor-three-capture-paired-host-evidence-v1-20261028-summary01/report.json`:
150 measured and 54 priming pairs, 657 files / 90 JSON source checks pass.
Natural and profiler-perturbed wall clocks remain separate. The execution-order
report preserves the unbalanced order and absence of temperature measurements.

Natural host partition independently reconstructs all 50 START→READY intervals,
46.578154846 seconds. Native inference outside the observed forward calls is
15.129050305 seconds (32.481%); it includes outstanding device execution and
synchronization, so it is neither added CPU service nor evidence of SM idle.
The model therefore uses disjoint entry, forward-to-next-forward/final,
native-END-to-READY tail and coarse OCR response macros. There are 334 per
capture / 1002 over the three natural captures. Stream elapsed and sync waits
are not added to these already inclusive response intervals.

The capture01 macro03 and native-prefix joint-contract independent review is
`artifacts/predictor-native-prefix-joint-contract-v1-20261028-independentreview01/report.json`.
It independently checks 1388 raw milestone identities, 3226 processing-visible
exposures and 184 query notices crossing a physical macro boundary. Host END
already occurred before BEGIN processing for 137 host stages (98 before reader
receipt, 39 during reader-to-processing delay). This is not a forward mixture
probability. Absence of END must permit completion with its notification still
in transit; query true never clears the remaining response macro.

The G2 finalization gap has a source/trace mechanism explanation: a Python loop
compares CUDA scalar IDs while locating TTS boundaries, including for text-only
requests. For these training traces, full sequence N=input+generated gives
DtoH/sync=2N+6, CompareEq=2N+5, exit kernels=2N+35. This is a diagnostic count
relationship, not a universal time model. The isolated bulk CPU-conversion
candidate is not installed in the worker and has no GPU/audio semantic
qualification. The modeling file's current receipt cannot establish a missing
historical execution snapshot. New optimized execution needs a new identity.

### Training-only whole-capture conditional response results

`artifacts/predictor-macro-joint-response-fit-v1-natural-20261028-review01/report.json`
preserves the original BFGS precision-loss flags. The independent analytic
Hessian / trust minimization / gradient-root solver and its result are separate:
`artifacts/predictor-macro-joint-response-fit-v1-natural-20261028-review02/report.json`.
All full and train-two/leave-one fits satisfy the fixed stationarity threshold,
positive curvature, multi-start agreement and nuisance-adjusted rank checks.
Original failing optimizer flags remain failing, explicitly scoped to the old
BFGS/trust attempt; they are not rewritten as successful runs.

|candidate|pooled heldout conditional mean NLL|conditional macro median MAE ms|
|---|---:|---:|
|baseline|3.841226414|36.303418597|
|visible count|3.676489515|25.343659129|
|visible phase aware|3.668804704|25.755666980|

The fixed equal-capture heldout-NLL rule selects phase-aware. MAE is conditional
on the observed heldout peer path, with endpoint-state continuation explicitly
marked when inversion exceeds that observed path. It is not a prospective
queue/P95 error, proof of independent macro residuals or a causal SM/DRAM model.
Whole captures remain the replication unit. New root tests independently check
density normalization including the endpoint Jacobian, scalar finite-difference
Hessians, unit changes and exposure splitting.

### Resource structure and the actual/visible queue split

`artifacts/predictor-static-language-work-v1-20261028-cpu01/report.json` binds
model configs, installed seven-projection declarations and safetensors headers.
Six captures each have 230 source-bound forward BEGIN structure features.
The features are nominal dense-equivalent arithmetic and declared logical
operand bytes, with text LoRA separate. They are not DRAM traffic, measured
cache reuse, hardware bottleneck classifications or a validated compute/memory
pairing rule. Attention/KV length, quantized runtime work and the missing
precision-specific hardware roof remain explicit limitations. A separate
training candidate is being built to test partial projection intensity
contrast with missingness indicators and all 1002 macros retained.

`macro_response_queue_reference_v1.py` now composes explicitly supplied joint
worlds over the complete future public arrivals and per-owner FIFO. Actual
response transitions and processed notifications are separate queues. Internal
query marks may occur in a later tail and retain their source event identity;
query success does not release a macro. Pending notifications survive the next
macro and request. CLEAR carries request/macro scope; persistent processed
issuance watermarks reject stale phase resurrection, including equal-time
messages. Native raw does not yet certify the generated issuance ordinals.

Per-public-key immutable joint-draw tuples are consumed by dispatch ordinal;
request IDs do not index historical durations. Source/caller containers stay
unchanged. Explicit transport FIFO contradictions return UNKNOWN, without
clamping notice timestamps. Four owners can proceed concurrently: no global
cap1/cap2 was introduced. The fixed drain retains every unfinished offered
request. READY processing and RESULT receipt remain distinct, with independently
declared transports; same-worker protocol FIFO still constrains their order.

The reference's latest independent report is
`artifacts/predictor-macro-queue-independent-review-v1-20261028-pass02/report.json`:
14 author and 15 independent tests pass. Earlier failures and pass01 remain.
It uses exact rational reference timestamps and replay from start, not a native
integer-clock adapter or resumable native posterior. Resource/utility outputs
remain UNKNOWN without their own provider. It cannot replace the unchanged
six-item scoring or make the old prediction gate pass.

v59 simulated checkpoint equivalence also passed its own 47 tests and root
three-scenario check, but its old held-context service rates remain scientifically
distinct from these new macro response clocks. Neither component alone is an
end-to-end native rolling predictor.

### Remaining work and unchanged gate

The empirical local joint candidate retains correlated response/H/query and
three notification-delay components, including final true queries in READY
tails. Exact continuous response matching has zero heldout support and is
reported as such. A separate virtual-coordinate joint transport candidate is
being implemented: jointly draw reference residuals and internal marks, then
let the response model generate wall timing under changed visible peer paths.
This is a new, testable transport assumption, not an independently calibrated
law or a license to replay a heldout trajectory.

The old main prediction gate remains FAIL (Delta-U sign 2/3, rank 2/6). No new
main gate seal, mechanism ablation or final four-group benefit experiment has
been authorized by predictive evidence. The next step is complete joint
generation plus queue/resource/scorer integration, training-only diagnostics,
independent source review and then a fresh independent prediction gate. The
equal-weight six-item utility, stability/rho/g(n), offered cohort, original
arrival windows and drain, startup-through-drain owned M, and original-window
board E plus one 3600-second startup amortization remain unchanged.

A directory-quota check during this continuation found 17.9494 GiB free. This
is below the unchanged 18 GiB GPU launch gate; check the directory again before
any future GPU run. Ordinary-file filesystem views remain unsuitable for this
quota. No evidence was deleted to bypass that gate.

## Follow-up: transport failures retained, original sender contract and scoring endpoints

The partial decoder-projection contrast experiment is complete. Its first
own-unsupported main-effect model loses nuisance-adjusted rank (51/52 or 52/53)
and tends toward an effectively flat large coefficient; that report is retained.
The subsequent interaction-only training design, recorded as a follow-up design
rather than independent validation, is
`artifacts/predictor-macro-projection-contrast-v1-natural-20261028-review02/report.json`.
All 1002 macros / 9697 exposure pieces remain; known peer projection exposure
is 74.814%, with the remaining exposure explicitly UNKNOWN. Phase+unknown-peer
count versus phase+unknown-peer+known contrast gives heldout NLL
3.624938136→3.622455663 and conditional MAE 25.580→25.127 ms. The contrast
increment improves two capture folds and worsens one. It is not evidence of
hardware compute/memory complementarity. Actual observed batch is one; larger
batch support is currently only structural arithmetic/synthetic tests.

Root independently rederived all 12 baseline/count/phase full/fold scalar
likelihoods, nuisance means and variances and checked scalar finite-difference
Hessians against the analytic solver:
`artifacts/predictor-macro-response-scalar-review-v1-20261028-rootreview01/report.json`.
Density normalization, unit invariance and exposure subdivision also passed
four independent tests. This is arithmetic/split validation, not physical
prediction validation.

The first complete joint request residual-vector transport is now retained at
`artifacts/predictor-joint-request-response-transport-v2-natural-20261028-training03/report.json`.
It uses three training captures, 150 request vectors, all 1002 response macros
and 1380 stream-query marks. Its whole-training-capture forward scenarios use
train-two-capture coefficients and donors, never heldout donors. Of 600 offered
scenario requests, only worlds containing 170 requests can compose; worlds
containing 430 requests have FIFO conflicts and remain UNKNOWN. The known-only
reader E2E MAE 62.58 ms is not an all-request accuracy claim. The independent
review (56 tests plus exact raw/review/macro/law reconstruction) is
`artifacts/predictor-joint-request-transport-v2-independent-review-20261028-pass01/report.json`.

Fixed donor message lags do not transport safely to different dynamic response
paths. A separate explicit protocol-station model is being implemented;
the failing v2 law, source and entire conflict denominator remain immutable.
The new model must include START as well as all PHASE, RESULT and WORKER_READY
messages, per-worker reader ordering and shared collector queue timing. Slack
inferred from timestamp recursions is protocol timing slack, not measured CPU
service; missing background/control traffic remains a qualification limit.

### Correct E2E endpoint and original two-FIFO execution

The reference now separates RESULT reader receipt from collector processing.
`RESULT_receipts` and task latency arrays use scheduled arrival→reader enqueue;
`RESULT_processed` is a separate field. Phase/query visibility remains
collector-processing based. Reader/processing FIFO and nonnegative brackets
are checked separately. Parent dispatch and actual worker START also have
separate events; cutoff preserves future worker START and in-transit RESULT.
The latest reference independent delta report is
`artifacts/predictor-macro-queue-independent-review-v1-20261028-pass03/report.json`:
17 author plus 21 independent tests pass.

The original deployment sender does NOT wait for parent RESULT/READY before
sending the next request. Per-owner sender FIFO waits only for the previous
write/flush to return; the worker stdin loop serializes execution. Thus the
reference's explicitly named after-READY-processing strategy is an alternative
execution policy and now has `dispatch_contract_matches_original_sender=False`.
It must not be used as an original-execution queue predictor. The isolated
`worker_stdin_fifo_contract_v1.py` represents sender, latent line availability,
worker read/pre-START and actual READY separately.

All 150 measured training requests were submitted after the previous owned
worker READY occurrence. This supports an idle-worker submit→START donor lag
only, not its unique decomposition or transfer to a busy pipe. In 46/150 cases,
worker START precedes the parent's write-completed timestamp, so that timestamp
cannot be imposed as a worker-start predecessor. READY occurrence is sampled
before emitting/flushing READY, so its stamp is not itself completion of the
worker's continuation to the next read. These missing intervals must be
explicit latent/support terms or UNKNOWN, rather than invented zero delays.

### New-domain resources remain separate from the old resource model

`artifacts/predictor-macro-resource-interface-gap-v1-20261028-review01/report.json`
checks the old resource interface against the new execution. Old training uses
semantic_worker and 12 priming; new training uses semantic_progress_worker and
18 priming. Shared execution options do not establish matching source identity,
so the old power/memory/startup estimates cannot silently be reused.

The new `macro_resource_timeline_v1.py` supplies one global clock of actual
macro-handler states, all resident owners, idle time and original-window
overlap; four tests pass. It does not sum parallel owner exposure into board
energy or set inactive resident memory to zero. No resource estimate or six-item
utility is invented. New-domain resource training extraction is in progress.
The profile observation end combines measurement and drain, without a separate
scoring deadline: full observation-window joules can train resource response,
but are not final six-item E. Original scored-window energy remains UNKNOWN
until the scoring workload supplies its own original deadline.

## Continuation: actual two-FIFO, joint protocol and original six-term integration

The protocol-station v3 replaces transported message lags by explicit per-worker
reader and shared collector timing slacks. All 4,614 source messages reconstruct
their original three timestamps exactly. Its full training leaveout now composes
600/600 offered draw requests, without dropping the v2 conflicts. This establishes
composition only: reader-availability collector order, empty per-job initial state
and omitted background/control traffic remain explicit assumptions.

The four-draw ensemble has reader E2E mean-prediction MAE 153.806 ms and empirical
CRPS 119.850 ms. The earlier 160.038 ms is a random-world absolute deviation from
one observed outcome, not ensemble point-prediction MAE. Errors predominantly
come from response duration, not START or reader notification delay. Conditional
macro likelihood improvement does not establish generated-path queue accuracy.
See `predictor-protocol-transport-ensemble-v3-20261028-cpu02` and
`predictor-protocol-transport-error-decomposition-v3-20261028-cpu02`.

The isolated `actual_worker_twoFIFO_reference_v4.py` now uses the original sender
FIFO and actual-worker READY release. Sender write completion does not gate the
current worker START; parent READY processing does not gate the next START. Every
transport component and initial station is explicit. The source-order-match flag
does not certify busy-pipe transfer, internal future work, native state or physical
prediction. `actual_worker_twoFIFO_query_adapter_v4_1.py` additionally binds the
complete ordered public cohort, input/route identities and original window/drain.

The actual two-FIFO training diagnostic retained 624 worlds / 1,200 offered draw
requests over all folds and both paired idle-transport decomposition endpoints.
All same-donor START and RESULT reader endpoints exactly equal frozen v3. This
zero difference is expected because every original diagnostic job has at most one
request per owner. It supplies no busy-queue or sustained-load validation. The
diagnostic and independent review are in
`predictor-actual-worker-twoFIFO-training-v4-natural-20261028-diagnostics01` and
`predictor-actual-twoFIFO-diagnostics-v4-independent-review-20261028-pass01`.

Independent v3/v4/six-term review at
`predictor-protocol-v3-v4-six-term-independent-review-20261028-pass01` passes
42 tests and rederives the joint training sources/messages. Its retained failure01
found startup-energy UNKNOWN silently becoming zero in the original scorer's
common-startup comparison mode. The whole-world bridge now withholds E unless
both raw startup and measurement energy are known. The original scorer is unchanged;
`unknown_resource_fields` records the missing contributions. Reader latency,
incomplete drain cohorts, rational timestamp rounding, memory maximum and the
single safety/amortization application remain covered by independent tests.

### Resource extraction and whole-world score path

The new-domain resource training is complete at
`predictor-natural-macro-resources-v1-20261028-training04`: 150 requests, 1,002
macros and all 14,293 polls. Of 144 exact owner/input/actual-phase states, only 48
have at least two-capture power support and 58 have memory support. Missing states
remain UNKNOWN. Whole-capture replay supplies observed state durations, so its
known-duration energy errors are conditional diagnostics, not future prediction.
All full-fold observation E/M remain UNKNOWN. The observation end combines
measurement and drain and cannot be used as a final scoring measurement deadline.

Root's independent resource review at
`predictor-natural-macro-resources-v1-20261028-rootreview01` verifies 602 source
files, rebuilds each raw macro partition and every poll bracket/state assignment,
reintegrates power using rational decimal arithmetic, and rechecks full-state
capture means and leaveout support. It does not call the candidate extraction or
integration functions. It passes arithmetic/provenance checks without granting
resource accuracy or a continuous memory capacity bound.

`actual_worker_world_score_v1.py` now connects the query-bound actual two-FIFO
world to one global resource timeline, the explicit new resource provider and
the unchanged original six-term scorer. Startup and phase-start-through-drain
peaks stay separate until the original scorer takes their maximum. Energy uses
only the original measurement overlap. Unsupported resource terms and censored
latency remain UNKNOWN; the adapter confers no stability or native eligibility.
Five integration tests pass, including incomplete fixed-drain cohorts, source
scope declarations, resource UNKNOWNs and no mutation of the generated world.

The next response candidate uses fixed family/phase-specific freely fitted
peer-count slopes, retaining all entry/OCR/tail nodes. It must first pass rank
and numerical checks, then improve complete generated-path ensembles. New
training preparation will cover sustained same-owner queues and source-visible
transport markers; current idle-only donors cannot identify those mechanisms.
No independent main gate, mechanism ablation or final benefit run has started.
The original main gate remains FAIL, and the goal remains active.

### Response candidate outcome and connected queue explanations

The fixed family/phase response experiment is finished at
`predictor-phase-heterogeneous-response-v2-natural-20261028-training01`.
The 12-feature full fit is rank 60/60, but one leaveout training fold is 59/60:
all eight OCR-entry macros in its remaining two captures have exactly zero peer
count on every exposure piece. Three equally good training roots consequently
give incompatible OCR-entry coefficients. That candidate is not promoted, and
its poor heldout conditional error is retained. The 22-feature candidate has four
exactly zero columns and an identical OCR-entry count/same pair; its structural
failure is recorded without deleting columns or spending further optimization
on an unidentifiable model. Global2 alone remains numerically qualified, with
the same 153.806 ms generated ensemble error. This is a diagnosed training-design
deficiency, not an improved interference predictor.

The old response residuals correlate with peer-prefill exposure for text/G2
prefill and OCR inference. This motivates crossed same-input/peer-phase/start-order
training within each independent capture, rather than using job/context IDs as
features. The new preparation adds OCR-entry peer-first/own-first offsets; planned
case names do not count as achieved phase coverage. Observed positive-duration
support and complete leaveout rank must be checked after collection. All outputs
remain training-only; the earlier independent validation labels were not fitted.

`actual_twoFIFO_queue_diagnostics_v1.py` now supplies full-arrival count/reference
work envelopes, sender and worker FIFO heads, serial convoy waiting, backlog
recovery and fixed-drain censoring. Actual handler progress and RESULT delivery
are separate axes. Work is an explicitly supplied macro response coordinate,
not solo execution time, SM service, DRAM or L2 demand. Resource-feasible admission
HOL and current native risk features remain UNKNOWN. Root independently verifies
queue area equals the sum of individual waiting intervals under six cutoffs,
sender reordering and parallel lanes, plus nonzero-origin invariance. All twelve
author/root tests pass at
`predictor-actual-twoFIFO-queue-diagnostics-v1-20261028-rootreview01`.

`actual_worker_world_forecast_v2.py` connects explicit public plans to actual v4
execution, the original-score bridge and this queue diagnostic. Every future
request's full reference work is computed before simulation from its plan, keyed
by public arrival ordinal rather than historical identity or sender order. There
is no cap, alternate cutoff or cost override. Eight author and two root integration
tests pass; the independent four-lane case runs all sixteen offered requests with
four concurrent handlers and no global cap. The exact-rational diagnostic versus
integer-score ceiling difference stays below one nanosecond. The independent
report is `predictor-actual-worker-world-forecast-v2-20261028-rootreview01`.

This is a concrete CPU accounting path. Its rates, joint work, native posterior,
batch/cache execution and busy transport still require matching evidence before
it can replace the main predictor. Library mechanisms and static projection
proxies are not silently promoted into USHER/Roomie physical mechanisms.

The new training-manifest independent review found duplicate capture/case and
input/window integrity holes in CPU-only validators. Frozen v1/v2 evidence stays
unchanged. A new semantic preflight must rederive the declared matrix, exact public
content and unchanged original runtime policy before a new collector can run.
Training is sharded into at most sixteen fixed 10s+60s cases per owned attempt to
respect the existing 2400s supervisor budget. Each attempt keeps all eighteen
priming requests and its own startup record. Incomplete shards cannot be counted
as complete training blocks. No GPU collection has started in this continuation.

### Repeated-owner audit and macro intake preparation

Root added `twoFIFO_owned_training_review_v2.py` and
`twoFIFO_composable_macros_v2.py` for the new repeated-owner schema. The cohort
audit checks every planned row, public owner/task/input/arrival, original fixed
measurement/drain and the separate sender and worker serial orders. Incomplete
and NOT_STARTED rows retain their full offered denominator and prevent macro
extraction. The macro adapter uses all priming and measured processing events
from one entire attempt; priming informs the visible history without becoming
response-fitting labels. It never calls the old 26-job role gate.

The direct original-runtime CPU counterexample establishes that a RESULT can be
read and processed before the sender acquires its post-send write-record lock.
The new clock audit allows this legal branch order, while retaining submit before
write and submit before START. Five root contract tests pass, and the combined
source-clock/macro/v4/forecast suite passes 33 tests. These are CPU correctness
checks. Native plan/supervisor binding and independent review remain required
before this new intake can admit GPU evidence.

Historical frozen JSON/Python hardlink deduplication is recorded at
`predictor-lossless-frozen-dedup-20261028-plan04`. All original paths, file bytes
and modes remain; no live source was hardlinked to frozen evidence. The report
records 4,224 paths and 416,993,280 allocated bytes recovered. Directory-level
free space is 18.274 GiB at the latest read and must be checked again under the
original supervisor. The original failed main-gate report still hashes to
`dbf25cedbbe63fd4bb256bb2131c7689fecfcd08aa1a45db56c5df4b0d7881a2`.

### Native repeated-owner training started after CPU02 review

The new source-frozen campaign is
`predictor-twoFIFO-owned-training-design-v2-20261028-cpu01/campaign.json`
(SHA `f6bec2e4c8c75d982e078ae4c8679a518a7eb56a9040b1878c760d6df266310d`).
It retains 150 cases, 546 measured requests and 216 priming requests across
twelve separate owned startups. Every statistical block comprises four shards.
The final prelaunch review is CPU02, not CPU01: CPU02 includes the correction
that full-raw disk checkpoints run before choosing the epoch and after the last
arrival, outside the timed arrival dispatch sequence. Its independent 500 ms
slow-disk counterexample preserves all scheduled offsets and the original fixed
drain. All 17 CPU checks pass, and a separate actual-campaign audit verifies all
twelve unique attempt/execution identities and source freezes.

The independent downstream review passes 29 CPU tests, including eligibility
forgery rejection, exact native-review rederivation and sidecar future-shape
poisoning. `twoFIFO_static_resource_sidecar_v1.py` now attaches source-bound seven
projection descriptors from the entire attempt's processed BEGIN history.
It reuses existing feature/pair-state functions, retains unsupported peers and
all macro rows, and does not change rates. Attention/KV, actual quantized costs,
DRAM, L2, physical resource class and cache gains remain UNKNOWN. The collector
has no reverse dependency on the sidecar, so its frozen identity is unchanged.

A fresh read-only environment receipt verifies worker interpreters, registry,
declared specs and checkpoint resources. Current driver settings show
`RmProfilingAdminOnly=1` and the container lacks effective/bounding
`CAP_SYS_ADMIN`. No new counter probe was run. This is a profiling capability
restriction, not a CUDA execution prohibition. The original counter failures
are not rebranded as successful physical demand measurements.

Additional byte/mode/owner/xattr-preserving historical hardlink deduplication
recovers approximately 0.521 GiB while retaining every original evidence path.
The large-JSON rescan finds no further exact duplicate group. To cover the
estimated 3.20 GiB campaign storage including an atomic-write temporary copy,
ten unused downloaded wheel archives were copied with SHA checks to the system
cache and their old cache paths retained as symlinks. This changes no installed
runtime file, weight, input or raw evidence. The relocation is reversible and
recorded at `predictor-install-cache-relocation-20261028-execution01`; data quota
free space rises to approximately 21.24 GiB while system free remains 2.24 GiB.
The original 18 GiB preflight, 13 GiB abort and 2400 s per-attempt budget remain.

Root starts `training-block-1-shard-1` through the reviewed supervisor. The first
checkpoint prefix passes actual cohort/history checking and contains all 46
offered requests plus all eighteen priming requests, including explicit future
NOT_STARTED rows. This is an active, incomplete training attempt: no native
training qualification, whole-block rank, independent gate or final benefit is
claimed. Completion must first pass the new raw review and exact macro intake.

### First native shard accepted; second shard underway

`training-block-1-shard-1` finishes all thirteen cases, 46 measured and eighteen
priming requests. Its owned supervisor exits cleanly with unchanged source audit.
`native-review01/report.json` is
`PASS_COMPLETE_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION`, SHA
`455ce3d7404e6233f8aebc3e12ec69be09d6c6d6a429df0de3c660d293fd21b9`.
It checks 40,188 board polls and an owned sampled startup-through-drain peak of
32,976 MiB. Two actual measured requests have the sender write record after
RESULT reader receipt; this validates the need for separate clock branches.

The real native macro entry rederives the entire review exactly and emits all
46 requests and 266 disjoint macros. The new sidecar's native entry also works:
365 actual processed BEGIN descriptors (including priming history) are bound;
known projection-pair exposure is approximately 91.48%, with unsupported peer
exposure retained rather than imputed. No rate, physical class or cache gain is
qualified by that coverage figure.

The real busy-transport diagnostic entry succeeds. Of all 46 measured requests,
13 are submitted while the prior handler has not reached READY; three arrive
before the prior sender write record. Fourteen START before their own write
record, and eighteen START before the previous READY is processed by the parent.
These are observed clocks under original source FIFO semantics. They do not
identify pipe line availability, CPU service demand or a transferable busy
transport law. The tool retains these unknowns and every offered request.

Root starts `training-block-1-shard-2` under the same audited supervisor. No
partial-shard fitting is performed; the next support/rank intake still waits for
all twelve completed attempts and groups leaveout by the three whole blocks.


### Owner-aware queue engine and original score bridge implemented

The isolated `actual_worker_twoFIFO_reference_v5.py` uses exact
(owner, semantic phase, peer count, same semantic phase count) rates and restores
complete generated FIFO, active-work and protocol state. Ten CPU engine tests
include more than thirty split points; JSON resume gives the identical full
world. This is explicit generated-state continuation, not native-history
inference. The v5 query/queue and v3 forecast adapters preserve complete offered
arrivals, original windows/drain and RESULT reader endpoints. Five author and
six independent integration tests pass, including unequal owner rates without
corrupting semantic phase, incomplete P95, measurement-only energy, one memory
factor and explicit prior memory peak. An independently found timeline-hash
serialization mismatch was fixed by reusing the provider digest.

### Native v2 collection scientific HOLD; supervised stop, no retry

Both first two attempts finish and rederive through the unchanged native review.
The continuation begins shard3, then root identifies a serious measurement
path problem: v2 `dispatch_case` calls checkpoint immediately after its last
arrival, inside the measurement window. Checkpoint holds runtime and sample
locks while copying all cumulative raw/history state. In firstshard the largest
measurement-intersecting board gap grows from 0.54 s to 4.38 s across cases;
8/13 measurement energies are UNKNOWN under the original 2 s gap rule. Maximum
worker occurrence-to-parent processing delay reaches approximately 8.39 s.
The code path and increasing gaps establish a collector-induced interference
risk; no direct checkpoint spans were captured, so per-event causal delay is
not identified. This is not a physical GPU interference measurement.

After independent source confirmation, root sends SIGINT only to the audited
parent controller/supervisor (PID 532058). The original supervisor performs
owned cleanup, writes final FAILED_NO_RETRY for shard3, and continuation writes
STOPPED_NO_RETRY. No later attempt starts. GPU is then 1 MiB used / 0% activity.
All raw attempts, source freezes, reviews and reports remain at original paths.
The old v2 campaign must not automatically qualify a response/notice/resource
law even if supplied additional attempts; new training entry points will guard
this HOLD before reading labels. No case is selectively deleted and the gap
threshold is not widened.

Recovery implements an isolated v3 collector: persist before selecting each
case epoch and after its fixed drain only; no full snapshot/deepcopy/serialization
between epoch and fixed drain. Exact timing spans and concurrent CPU tests will
verify that contract before a new source-bound campaign. The original utility,
request population, 10 s measurement + 60 s drain, uncapped runtime, priming,
2400 s supervisor and storage preflight remain the constraints.

### New source-bound v3 campaign and generated queue uncertainty

The new v3 campaign is `predictor-twoFIFO-owned-training-design-v3-20261028-cpu01`
(campaign SHA256 `ed60a9787ae9e7e339c79576405a95e3d695aabe74fbbb0bf6ec05c9e4243973`).
It retains twelve startups, three whole blocks, 150 cases, 546 measured and 216
priming requests. Concurrent collector and source-adaptation reviews pass;
the first attempt is running, not yet a completed native training receipt.
Its first nine completed measurement windows have maximum intersecting board
poll gaps of approximately 29–59 ms; startup through priming has a 639 ms maximum.
These are partial collection diagnostics, not fitted accuracy or causal proof
of zero observer overhead. The old v2 HOLD is permanent for that dataset.

The independently reviewed v5_2 paired-transport engine and joint-law v4 retain
the exact feasible transport envelope. A target START is a point only when all
feasible source decompositions agree; otherwise UnknownTransportInterval retains
the full public cohort and interval evidence. Lifetime sender, worker and notice
FIFO offsets survive settled-history seeds and JSON continuation. Compiler marks
now use public owner/task/input/ordinal identities to prevent equal-plan aliasing.
The final independent engine/compiler report is
`predictor-twoFIFO-paired-transport-v5_2-independent-review-20261028-cpu02/report.json`,
SHA256 `ce4bf6c153ccd5bc3881a6c4ea14bdc2c387283aceeb998ae8276f7833ea2cb1`.

The forecast v4 adapter integrates generated full-public-arrival queue worlds,
original six-term scoring and explicit resource history. Its independent report
is `predictor-actual-worker-world-forecast-v4-independent-review-20261028-pass01/report.json`,
SHA256 `c9f6930993080e79c421ed62fe857aebc979a1000ae445dae69cb2574f72845f`.
The new finite ensemble preserves UNKNOWN draw probability without renormalizing,
returns full-cohort fixed-drain risk bounds, and leaves absolute U and paired
expected delta UNKNOWN whenever their support is incomplete. Independent review
found and prompted explicit comparison bindings for all original score
normalizers and shared quality evidence, plus serialized mass/denominator checks.
Compilation missing evidence uses a typed UNKNOWN draw, never an empty template.
This is a finite predefined training law, not calibrated long-term risk or a new
main-gate pass. The original failed independent gate remains unchanged.

Ensemble final independent review: 16 CPU tests PASS, including actual joint-law
v4 compile-to-ensemble integration, nonintegral probabilities, incomplete and
UNKNOWN worlds, original score bindings and malformed serialized input rejection.
Report `predictor-twoFIFO-forecast-ensemble-v1-independent-review-20261028-cpu01/report.json`,
SHA256 `fb23f3988e99789c8de7a4ac867378588010e390d986e0417629626a6c2aa08e`.

Memory compression candidate v1 is isolated from the accepted resource provider.
It evaluates complete-block filtered replay using causal visible shape/concurrency
high-water states and previous sampled peak. It requires all twelve native
attempts and preserves every unsupported/underpredicted held-out observation.
It is not a prospective generated-memory model and cannot fill UNKNOWN M in U.

### First v3 native shard accepted; remaining fixed campaign launched

The first v3 attempt ends cleanly under the original supervisor, with final
source audit PASS and owned workers reaped. Root native review SHA256
`3c6e5e3dc22669f8bc6612080541c374a623d8ea8063058ec75930717a6ac3bb`
passes all 46 offered measured requests, eighteen priming and thirteen cases.
Exact native extraction yields 266 macros. The independent full rederivation
is identical, in `predictor-twoFIFO-first-v3-native-independent-review-v1-20261028-cpu02/report.json`
(SHA256 `4650aaa498953cd2a894ce1721c9b51f47d8ea15731271cf274141c2adbde80f`).

Separate timing review
`predictor-twoFIFO-v3-firstshard-timing-independent-20261028-readonly01/report.json`
(SHA256 `6c3bf468b3f98717741e198caa6633d9bb908c08f04b14d9fbbe647d6ec34b5d`)
checks 44,288 polls and all 100 checkpoint spans. No span overlaps any case's
epoch-through-fixed-drain protection interval. Maximum measurement-intersecting
poll gap is 85.668 ms; drain maximum is 205.647 ms; startup maximum is 639.315 ms.
All thirteen measurement energies and startup energy are observable under the
unchanged 2 s rule. Startup energy is 7520.249779 J; owned sampled peak is
33036 MiB. Maximum per-case reader-to-processing delay is at most 0.371937 ms.
The 18.434 s maximum checkpoint and 5.601 s global poll gap occur between cases;
they remain recorded and do not establish zero observer overhead.

Root launches the reviewed continuation v2 at
`predictor-twoFIFO-campaign-continuation-v2-20261028-run01`. It rederives firstshard
before starting the remaining eleven in fixed order. Any incomplete receipt,
failed native review or UNKNOWN startup/measurement energy stops further
collection with no retry. Whole-block response/resource fitting still waits for
all twelve. This native acceptance establishes no prediction accuracy, risk
calibration, physical SM/DRAM/L2 law or final routing gain.

### Pre-fit generated leaveout design and isolated resource folds

The generated whole-block runner is now independently reviewed: 17 CPU tests
PASS, report `predictor-twoFIFO-generated-wholeblock-runner-v1-independent-review-20261028-cpu01/report.json`,
SHA256 `fff8a565366f87ce15c6eb322dc27ec220bd125b4d4b00f0730497bb43515fea`.
Its generation interface takes only a public case/calendar, immutable native
past snapshot and the two donor-block joint law. Changing future native RESULT
labels changes evaluation errors but leaves prediction bytes identical. Entire
unsupported draws, fixed-drain misses and all 546 offered requests survive.
The offline native review and full/fold fit have already inspected training
outcomes: this is training leaveout, not prospective independent validation.
Prediction fsync precedes each case's evaluation-label extraction, not the first
ever read of its training labels. Q in this timing diagnostic comes from frozen
training expectations, not actual validation quality; M/E/full U remain UNKNOWN.

Root prepares `predictor-twoFIFO-generated-wholeblock-design-v1-20261028-cpu01/design.json`,
SHA256 `7d6ac41755917359c22a878e057f41b393bf7ff46ef6c47d77a883b1482ccb3a`.
Its four finite slots per donor block, equal block weights and public-key/ordinal
rotation are fixed before any native numerical fit. The original equal weights,
latency/memory/energy references and 3600 s startup amortization are retained;
ten-second case window and nominal workload rates come from the public v3 plan.
Those controlled training cases are not final routing-utility experiments.

The isolated resource fold provider rederives all twelve native tables and all
three saved folds before exposing just the selected eight donor attempts. Its
six CPU tests include all three leaveout directions in both mean/max scenarios,
and heldout power, memory and startup poisoning leaves selected predictions
unchanged. Report `predictor-twoFIFO-resource-fold-provider-v1-independent-review-20261028-cpu01/report.json`,
SHA256 `c1bde7096dc6dbb4cb98f9f4ed7dcdaaf124e7bdb5c55f0f153ccb825693952c`.
Native resource fitting has not run. The reviewed queue runner remains frozen;
this adapter does not silently replace its explicit UNKNOWN resources.

### Accepted CPU completion gate and actual v3 projection structure

The one-shot CPU completion coordinator passes twenty author/independent CPU
tests. It requires the exact final continuation status, ordered twelve native
execution/review/macro receipts and complete native rederivation. Missing final
metadata means WAITING with no review-label access; failures stop without fit.
The started output is fsynced before either numerical stage and permanently
prevents retry/overwrite. Independent report
`predictor-twoFIFO-training-completion-coordinator-v1-independent-review-20261028-cpu01/report.json`,
SHA256 `877bede85017ba23e93a451c483f5e21597913d85c07deda508db6f3b368b0a4`.
Final prepared02 manifest SHA256
`de0b33e14e3b5fc3e82d1a4c6c2bc967e4398d33ec13d077c0b45e63b4921c86`.
Root actually invokes it once before collection completion: WAITING,
reviews_opened=false, both numerical stages false and no execution output.
A separate bounded metadata-only waiter is running at
`predictor-twoFIFO-training-completion-wait-20261028-run01`: after a stable final
controller report it calls that same audited gate exactly once, without retry.
The only authorized automatic stages here are resource fitting followed by
generated whole-block training diagnostics; it launches no GPU validation/gain run.

The isolated static resource sidecar v2 genuinely accepts new v3 native macros
and review; it does not relabel old v2 documents. Thirteen CPU tests and actual
firstshard re-extraction pass independent review at
`predictor-twoFIFO-static-resource-sidecar-v2-independent-review-20261028-readonly01/report.json`,
SHA256 `2b07c74d4b03005556e3574161bbbbc8294eb4a88a274f194de5748810c1c9e7`.
The independent raw join retains 46 requests, 266 macros, 1775 pieces and all
eighteen priming requests/994 priming events. Its 365 supported BEGIN descriptors
are available only after their processing clock; seven-projection nominal FLOPs
and logical operand bytes are independently recomputed. Known pair exposure is
13.067849255 s of 37.333119583 s (35.0034%); the remaining 24.265270328 s stays
UNKNOWN. Those are structural coverage units, not GPU service, measured traffic,
physical compute/memory classes or cache benefit. No rate or score is changed.

### Second v3 shard accepted; third running

The fixed continuation accepts shard2 after complete native rederivation:
66/66 measured requests, eighteen priming, thirteen cases, zero critical
checkpoint overlap and no runtime/telemetry errors. Startup energy is
7568.251905 J; all thirteen measurement energies remain known under the original
gap rule; owned sampled peak is 33044 MiB. Exact macro extraction retains all
66 requests and 426 macros. Native review SHA256
`ad9f305daa3719848164d7fa147ad537e530731ebd2d387c6206ea2d1548cd70`.
Across the first two accepted attempts the full population is 112 measured and
36 priming requests. No complete whole block exists yet and no native fit runs.
The controller starts `training-block-1-shard-3` next in its frozen order.
The metadata-only CPU waiter (session 37785, PID 557026 at launch) remains active;
the GPU campaign controller is session 10542. Do not launch duplicate collectors
or manually consume the CPU coordinator output while its waiter owns that call.

### Prospective memory reference and pre-fit identifiability check

The strict prospective-memory v1 reference passes sixteen author/independent CPU
tests. Report
`predictor-twoFIFO-prospective-memory-v1-independent-review-20261028-cpu01/report.json`,
SHA256 `825eb52a7de2c5f22f3ef1c961153ec88d3c04f915fbbf1c05936b3c36e3d2e5`.
It evolves the cumulative absolute sampled peak by a maximum, never by summing
per-query increments. Generated event processing controls shape visibility;
future native polls are excluded. Its exact-duration and poll-geometry support
requirements make it a restrictive CPU reference, not an identified native
memory solution. Cross-boundary or unobserved intervals remain UNKNOWN.

A separate feature-only check, without memory targets, finds that priming has
already saturated all three proposed cumulative text-shape columns: on the
first shard they are constants, with rank one including the intercept. The two
proposed concurrency columns C and F are also identical. Report
`predictor-twoFIFO-case-memory-feature-audit-v1-20261028-cpu01/report.json`,
SHA256 `fc2cfe52ab69cb266681d3fb78a2c3717008ed5b173c63ab1aa088cc9ae3ab33`.
The eight-column v2 memory design is superseded before any target fit. A smaller
v3 design is being checked using public plans and explicit generated-path
requirements; no full-campaign numerical rank or native prediction accuracy
has yet been established. Earlier concerns about missing priming
`profile_job_id` were checked against all eighteen real first-shard priming
rows and withdrawn: all contain `profile_job_id='priming'`. No HOLD applies to
the accepted v3 capture on that basis.

The metadata waiter has also passed a read-only independent review, report
`predictor-twoFIFO-training-completion-wait-independent-review-20261028-readonly01/report.json`,
SHA256 `71860ada5846d2ca3cb3ecc232af3872047255812ae1b231ee2455559d122e1b`.
It has exactly one coordinator call, no retry and exclusive creation of its
started/result/error receipts. Stable final metadata is not itself predictor
qualification; the coordinator still authenticates all twelve native attempts.

### Certified single-owner transport bounds and main-chain wiring gaps

The new isolated single-owner interval module passes eleven CPU checks,
including 150 concrete feasible worlds compared with frozen v5_2. Independent
report `predictor-twoFIFO-single-owner-transport-intervals-v1-independent-review-20261028-cpu01/report.json`,
SHA256 `464392543f0a06348341d2d3e46637f792a1031c62ce8d5a63fae4c8fabb3178`.
With one owner, no external producer/peer, fixed no-peer work and protocol
slacks, START is monotone in prior READY. Independent paired feasible
coordinates yield simultaneously attainable lower/upper cohort schedules;
the stdout-reader and collector FIFO recurrences preserve that order. The
complete-cohort P95 envelope is sharp in this declared subdomain. Uncompleted
fixed-drain cohorts never become completed-subset P95. Multiple owners remain
UNKNOWN: the preserved two-owner interior-latent counterexample defeats
endpoint-only P95 bounds. This module has not replaced the frozen runner or
supplied resource bounds, latent probabilities or full utility.

A source-bound read-only audit now makes the remaining main-chain gaps
explicit in `docs/twoFIFO_literature_wiring_gaps_v1_20261028.md`, report
`predictor-twoFIFO-literature-wiring-gaps-v1-20261028-cpu01/report.json`,
SHA256 `7db06a89ef8c3d8903609cd968d85de6afcfb0a8ce2c0d09b0616f9796a4290e`.
The current phase12 fit is a conditional response model: the seven-projection
sidecar has not entered its rate matrix. A non-singleton START can invalidate a
whole point-world, but multiowner operation does not itself imply failure;
actual coverage awaits the complete generated diagnostic. Resource models and
fold adapters exist, while the frozen runner does not pass them, so it still
has UNKNOWN M/E/full U. Exact simulated continuation and a settled native seed
also do not implement busy native rolling assimilation. New independent
companions are being built for actual resource rescoring and the smaller memory
candidate without changing any source owned by the running campaign or its
automatic completion stages. None of these CPU checks overrides the old FAIL.

### Third v3 shard accepted; fourth running

The continuation accepts all 38/38 measured requests, eighteen priming and
thirteen cases from `training-block-1-shard-3`; macro extraction retains 210
macros. Review SHA256
`9bc3e7cec5161c3fd1e9faf6318d492419f08e76c4440faf65ae844c33d13a18`.
Its 44,698 board samples record an owned sampled peak of 34,368 MiB. All 100
checkpoint spans remain outside protected case intervals; energy acceptance
passes the unchanged continuation checks. The aggregate accepted population is
150 measured and 54 priming requests, still short of one complete four-shard
block. Shard4 launches in the fixed order. No numerical fit or independent
validation has run from this incomplete campaign.

### Multiowner finite-rate outer envelope independently reviewed

`twoFIFO_multiowner_outer_envelope_v1.py` adds a conservative alternative to
point-world failure, without changing the frozen main runner. Every declared
macro requires the complete positive finite rate grid. Its minimum/maximum
bound duration and internal mark offsets even when peer processing changes the
rate path. Monotone paired START propagation bounds each owner's READY;
single-producer reader FIFO then bounds RESULT and full-cohort P95. Forgetting
cross-owner dependence deliberately widens the envelope: these are not sharp
or jointly attainable extrema. No collector processing bound, physical rate
certificate, latent probability, resource bound or full utility is supplied.
Unknown model-domain support preserves all offered requests.

Independent review passes nine CPU tests and checks 171 concrete worlds against
frozen v5_2, including two phases, peer acceleration as well as slowdown,
enqueue reordering, FIFO backlogs and the original interior-P95 counterexample.
Report `predictor-twoFIFO-multiowner-outer-envelope-v1-independent-review-20261028-cpu01/report.json`,
SHA256 `c0da2d457f40f29b43c90aadc6ca3dd3e8d04f58c93132084b306de0cb1bf3cd`.
The result establishes an envelope of the declared finite model, not accuracy
against native GPU execution. Any later training report must show its width
and uninformative fraction rather than count a wide interval as a good forecast.

### Isolated memory candidate and resource rescore companion sealed

The corrected case-level memory candidate fixes the cumulative prefix `P0` as
an offset and fits only `[1, A_case, C, R]`. It keeps an explanatory
actual-trajectory adapter separate from donor-only generated features and
rejects missing processed/latent/reader evidence, unsupported domains and any
held-out memory or execution labels. Native fitting remains gated on all twelve
reviewed attempts and is still not run. The final candidate source passes 21
CPU checks (11 author plus 10 independent), including positive/negative final
overflow and strict poll-origin boundaries; overflow becomes UNKNOWN rather
than an infinite or silently clipped prediction. Independent report
`predictor-twoFIFO-case-memory-candidate-v3-independent-review-20261028-cpu01/report.json`,
SHA256 `f2c306edba59f7c0ac2afd8bbc31b09de42506841d3412f219e7112e2f8a4e70`,
binds source SHA256
`abbec9c4489d6219f6779898fd54bceb17c5e02276a451f6157392839bb730a9`.

The resource companion `twoFIFO_fold_resource_rescore_v1.py` remains an
explicit single-case continuation from a completed generated training report;
it authenticates the selected eight-donor fold, exact campaign/review receipts,
and the physical execution identity before reusing the original six-term
score. Its source-bound past-memory state is a caller-declared evidence
contract, so absent state leaves M and full U UNKNOWN. Energy covers only the
original measurement with startup amortized once at 3600 seconds; memory covers
startup through fixed drain and applies the safety factor once in the unchanged
scorer. It does not merge the case-memory candidate or invoke the coordinator.
Final author evidence has 32 CPU checks (11 author, 7 independent binding, 14
regression); the repaired source SHA256 is
`aa43f647adf63c4d27f9bc3f8449228a959707e425a5b10a113086732e950477`, report
`predictor-twoFIFO-fold-resource-rescore-v1-20261028-cpu02/report.json`,
SHA256 `b3f9df034b794d6471404e866825aac7ad958ce0679f4e766a622c76047691f8`.
This is a wiring qualification, not a completed 150-case resource rescore or
an independent GPU prediction gate.

To avoid reloading the twelve native resource attempts for every case, an
isolated batch companion now caches one authenticated eight-donor
`FoldResourceTrainingProvider` per held-out block and reuses it across that
fold's cases. Its corrected source is SHA256
`faf50d88e2c0319c197f4afb2d36cbff78c553235b35a499036e649ca67c48ea`; author
CPU report `predictor-twoFIFO-fold-resource-batch-rescore-v1-20261028-cpu02/report.json`,
SHA256 `8221674e3ad09f8733cfcb5ee7dfd384c59625087dabcec63bef7af9820b79de`.
Twenty-one CPU checks pass. The report tests the cache and contract only; it
explicitly records that no real 150-case rescore has run. Its independent
binding review is pending before any use with a completed native training
report.

An additional source-level independent review of the repaired resource
companion passes 18 checks and confirms the exact campaign/review membership,
eight/four fold split, world replay hashes, frozen score, memory drain scope and
measurement-only energy scope. Report
`predictor-twoFIFO-fold-resource-rescore-v1-independent-review-20261028-cpu03/report.json`,
SHA256 `c84ee5d6dba0883f32d07c109dba206c48ed36a36021af76846f5b1d3f51b0b8`.
The past-memory receipt remains caller-declared and the resource law remains an
empirical hypothesis; the case-memory candidate is deliberately not merged.

The fixed continuation has now accepted all four shards of training block 1:
38, 66, 38 and 32 measured requests respectively, with 72 priming requests in
the block and all native reviews PASS. Shard4's owned sampled peak is 34,564
MiB, 86 checkpoint spans remain outside protected case intervals and 192 macros
were retained. Training block 2 shard1 is currently running; the automatic
completion waiter remains the sole future coordinator caller. No fit starts
until all twelve attempts have been authenticated.

The next receipt is now accepted for `training-block-2-shard-1`: 62/62 offered
measured requests, eighteen priming, thirteen cases and 386 macros; its owned
sampled peak is 33,044 MiB and critical checkpoint overlap is zero. The fixed
controller has launched block2 shard2. Across the five accepted attempts the
measured denominator is 236 and priming history is 90; no whole-block fit is
eligible yet.

The batch resource companion's independent review exposed case-history key
ambiguity: the real 150 rows reuse fifty case IDs across blocks. The final
source requires `block|attempt|case` identity and rejects ambiguous bare case
keys. Source SHA256
`2022f0497bfeecc47b5640b93032766549ffe20de2726a3c529b778e56497370`, author
CPU03 report SHA256
`fb4c9c3ea04a25ea5dd80957ab68f14a26b4daf09cca722e37381f9e56772580`.
Independent report
`predictor-twoFIFO-fold-resource-batch-rescore-v1-20261028-cpu03-independent/report.json`,
SHA256 `a01fc53b13c649692339cd2bc631a0ad162525c28c4934cfa0cccf98dda2ce65`,
checks source/report binding, all 150/546 public denominators and exactly three
provider constructions for three folds; invalid donor IDs are rejected before
construction. Twenty-two author/prior checks pass. Actual all-case native
rescoring remains unexecuted until the complete training output exists.

That batch review is subsequently WITHDRAWN AS AN EXECUTABLE PASS. A full-entry
CPU execution finds that the cache's target-plan dictionary is unpacked as a
two-item receipt/plan pair, stopping the first fold. Per-case output also needs
its own attempt's plan identity. The prior independent report is retained as
structural evidence, not executable qualification. A new version of the
candidate must pass an actual complete 150-case CPU fixture, not only static
contract checks, before use. No native batch rescore was started.

The corrected batch source is now frozen at SHA256
`642a032a750c37093abd2c0db751980c5710841612b64c484174184a33b434ca` with
author CPU05 report SHA256
`d423b160e88ea723bb1912b665718575c207ddce82019415321d972154ed592f`.
Its 23 checks include the real 150-case/546-request full-entry fixture, three
provider constructions, per-case plan and physical identity, and unique
block/attempt/case memory keys. Earlier cpu03/cpu04 reports are superseded;
runtime is performing the final source-bound read-only check.

The controller accepts `training-block-2-shard-2`: 44/44 measured requests,
eighteen priming, thirteen cases and 268 macros, sampled owned peak 34,482 MiB,
zero protected-interval checkpoint overlap. Block2 shard3 starts next. The six
accepted attempts retain 280 measured requests and 108 priming; fitting still
waits for all twelve.

## 2026-10-28 queue-prediction continuation check

The CPU contract suite was rerun against the queue-prediction pieces in scope
for the next gate. `tests.test_full_horizon_forecast` passed 41 tests,
`tests.test_rolling_visible_queue_v1` passed 6 tests,
`tests.test_twoFIFO_multiowner_outer_envelope_v1` plus its independent review
passed 9 tests, and `tests.test_recovery_risk_v2` plus
`tests.test_rolling_native_visibility_v2` passed 13 tests. These results verify
contracts only; they do not qualify native accuracy, busy rolling, batch
padding, recovery-risk calibration, or final utility.

The queue additions remain staged. Full public arrivals, FIFO/HOL transport,
fixed drain, full-cohort P95, UNKNOWN probability mass, and paired six-term
ΔU are wired into the CPU prediction chain. Rolling visible checkpoints, batch
waiting/padding, calibrated recovery risk, CV/robust arrival scenarios, and the
multi-owner outer envelope remain isolated sidecars until native support and an
independent prediction gate qualify them. No global request cap was added. The
original GPU continuation is now stopped on the source-integrity HOLD below;
no fit or generated prediction has started.

The scoring/execution gap design is now recorded in
`docs/twoFIFO_scoring_execution_completion_design_v1_20261028.md` (SHA256
`0aac808eeab7d68e04bf274f1d16005913c6a81231bd0a5b0d43a0bcf000c921`). It
specifies a guarded multi-candidate ranking companion and execution sidecars
for solo/shared-batch/common-weight-submatrix/USHER-Roomie mechanisms. The
companion keeps complete-cohort P95, six-term Q/T/L/P/M/E, rho and stability
unchanged; any UNKNOWN M/E or world mass leaves complete utility and rank
UNKNOWN. It is a design contract pending native evidence, not a claim that
batch or cache gains are already measured.

## 2026-10-28 continuation hold

Block3 shard3 produced raw data and a watchdog exit with all owned descendants
reaped, but its final source audit failed because the campaign-frozen hashes for
`gsdeploy/milp.py` and `gsdeploy/optimizer.py` no longer matched the working
tree. The fixed continuation therefore wrote `STOPPED_NO_RETRY` and preserved
the complete attempt evidence; shard4 and the coordinator were not launched.
This is a source-integrity HOLD, not a native performance result. The drift
also exposed an invalid trailing Chinese fragment in `gsdeploy/milp.py`; that
syntax error was removed and the affected optimizer/MILP tests now pass (23
tests), but the existing campaign remains closed and cannot consume the repaired
source. A new campaign, if needed, must freeze the repaired source first and
receive a fresh independent audit.

The CPU-only source-rebound preparation is now complete at
`artifacts/predictor-twoFIFO-source-rebound-v1-20261028-cpu01/campaign.json`.
It creates twelve fresh plan identities and rebinds the current
`milp.py`/`optimizer.py` bytes without editing the stopped campaign. Its status
is `CPU_PREPARED_SOURCE_REBOUND_NOT_COLLECTED`; it has no execute entry point,
and `GPU_started`, fitting, and validation-label access are all false. The
source-rebound audit and the 23 MILP/search tests pass. A new native executor
and independent review are still required before any GPU collection can use
this manifest.

The final source-rebound preparation is `cpu02` (campaign SHA
`5bad96124a6fd61fce1d7c7aabe705288c8d582b21416eb2801388191eb6ae13`, prep
source SHA `2b84f3438eee14d6e7ce8a00ea32a3d3699eae863672a390fd52381362a340cf`).
Its author audit and a fresh local audit both pass; the earlier `cpu01` draft
is retained as superseded preparation evidence. The rebound manifest is still
CPU-only and intentionally has no execute entry point. An independent audit is
now passes at
`artifacts/predictor-twoFIFO-source-rebound-v1-20261028-cpu03-independent/report.json`
(SHA256 `aa09ce3f648d27e78d8f03707d349a4f5504022a3366ac2d3fc6949d1fb5f530`).
The manifest remains non-executable until a separate executor is implemented
and independently audited.
