# Context state supervision and first next-owner candidate

These are CPU-only training artifacts. No validation outcomes were opened, no
GPU work was started, and no protocol or prior artifact was rewritten.

## Source supervision

Use `artifacts/predictor-context-initial-transition-v13-20261028-training02/report.json`.
Its SHA256 is `7fb93852046052b5badab86639c3f8e7801b0bf38b607765f72ead1215eef96b`.
The extractor is `tools/predictor_validation_20261027/context_initial_transition_v13.py`.

The two original common-session light captures contain 28 public jobs. At the
first source queued-or-started pending-kernel witness, 27 jobs have an identified
paired context envelope and one has a genuine context-event gap. Initial state
stores the original RESTORE_START and attained age, including a context already
held before the request. It never replaces the holder with a sole pending owner.
One initial observation has a different holder and sole pending owner. For the
same SS-long public workload, the two captures begin with different observed
holders and ages: text#0 at 247223 ns versus text#1 at 985239 ns. A single fixed
initial owner cannot represent both observations.

The extractor retains 35737 observed SAVE_END to next RESTORE_START adjacencies.
It excludes 62 affected by unpaired context events. It preserves 28 original
RESTORE clocks crossing a request start. Of the retained transitions, 3517 have
a holder outside the pending-kernel frontier, 715 restore an owner outside that
frontier, and 133 have an empty pending-kernel frontier. These are real labels;
they are not dropped or replaced by a round-robin policy. Pending kernels do
not describe all driver, copy, or context-maintenance state.

For 35566 transitions, a later mapped owner kernel provides an upper bound on
service readiness. The lower bound is RESTORE_START. For 171 transitions there
is no such measured-kernel upper witness. The interval includes possible host
and kernel-supply waiting: it is not a precise restore completion time, pure
switch cost, SM demand, or an SM service quantum.

Each public workload record is `[owner, task, pool_id, declared_start_offset_ns]`.
Each pending-frontier record is `[owner, task, pool_id, semantic_phases]`.
Initial labels include holder, original restore clock, age, and service bounds.
Transition labels include outgoing holder, age at save, last frontier-segment
entry age, next owner, next restore clock, and interval-censored service bounds.
The source-ready clock is a training reconstruction, not a public arrival clock
or a claim that production can observe CUDA context events.

Exact public-workload/frontier grouping gives two-capture support to 24 of the
27 known initial states and 35690 of the 35737 transitions. In the two complete
capture exclusions, a same-state training age interval covers 17159/17662 and
17577/18075 test transition ages. These are support diagnostics. They do not
establish numerical prediction accuracy, and exact-cell matching is not the
final algorithm or a reason to forbid statistical extrapolation.

## Fixed next-owner model

Use `artifacts/predictor-context-next-owner-v13-20261028-training03/model.json`
and its `report.json`. The report SHA256 is
`595c478a7120e0700486d8f97344b6cba20b0e58a6c30619cf6eb3392b2023eb`.
The implementation is `tools/predictor_validation_20261027/context_next_owner_v13.py`.

The first candidate is a fixed L2 multinomial logistic model, regularization 1,
capture-balanced loss, L-BFGS with at most 300 iterations, and no fold-based
tuning. Features are outgoing holder, continuous log attained age and its square
with holder interactions, pending owner/task/input/phase indicators, and public
workload/input/declared-offset summaries. All four resident owners are target
classes, even when not pending. Pending waiting age and remaining reference work
are explicitly missing in this first candidate; future observed kernel lifetime
is never substituted for either feature.

| Complete training capture excluded | Logit candidate | Global majority | Holder-only majority | Holder + public workload | Holder + workload + frontier lookup |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw 26305a33 | 99.932% | 37.351% | 59.659% | 99.366% | 99.762% |
| raw 98c4c261 | 99.934% | 37.095% | 59.541% | 99.419% | 99.856% |

The last-holder baseline is 0% because these source adjacencies change owner.
The two lookup baselines have 1/0 and 32/15 unknown rows respectively, counted
against the entire transition domain in the table. The candidate predicts all
35737 transitions. Its accuracy for next owners outside the pending frontier is
97.253% and 96.866% in the two folds; all 715 such labels remain included.

These high numbers mainly show repeatable transfer structure within the same
public job cohorts. There are only two independent captures, and thousands of
transitions within a capture are dependent. The stronger baselines make clear
that the gain cannot all be credited to continuous age. This is not a new
independent gate, calibrated probability evidence, or an end-to-end gain result.
There are two observed OCR next-owner labels, one per capture, at empty-kernel
frontiers before the declared OCR job. The complete transition extraction
retains these labels, which the older dual-pending handoff filter omitted. OCR
remains a possible output class; two labels do not establish reliable OCR mark
prediction. The API reports every resident's actual training target count.

The early training01/training02 mark reports contained a stale human-written
scope sentence saying OCR had no labels; their dynamic label counts and fitted
coefficients already retained the two OCR labels. Use training03, whose scope
and API report the actual counts. These early artifacts remain preserved.

## Scheduler interface and remaining pieces

```python
candidate = NextOwnerCandidate(model)
mark = candidate.predict_given_save_now(
    held_owner=outgoing_owner,
    context_age_ns=save_clock_ns - original_restore_start_ns,
    pending_frontier=current_forecast_frontier,
    public_workload=public_active_cohort,
    pending_age_ns_by_owner=None,
    pending_reference_work_ns_by_owner=None,
)
```

The API returns four owner probabilities, the most likely owner, observed-class
gaps, and feature-range/holder-age OOD flags. It permits a known public feature
combination to be statistically extrapolated and labels that result explicitly.
Unknown public identities, unsupported semantic phases, invalid clocks, future
label fields, and untrained nonmissing age/work covariates are rejected. It does
not silently accept those extra fields as predictor input.

This is a mark model conditional on a SAVE at the query clock. A composed held
context scheduler must obtain that clock from a separate survival model, preserve
the outgoing original restore clock, and forecast the frontier at that clock.
It must also supply an initial holder/age scenario and a separately trained
restore/service-ready interval. The mark API cannot invent these timings or
stand in for a complete availability oracle. Continuous-age marked survival,
reconstructed cutoff-safe pending ages/reference work, and an interval-censored
service model are statistical improvements that can now be trained and tested.
Unpaired source events and exact physical SM service remain different limits
on what the present source labels identify.

## Complete training index

The separate conservative index is
`artifacts/predictor-complete-training-index-v12-20261028-built02/training-index.json`,
SHA256 `71b54e36df0bdc6bf9d213ad5bb0c10d59d3346037520905668e0ea64a32c02c`.
It has 34 original raw-SHA runs and 1005 four-field records: 621 measured and
384 explicit completed priming records. The old sealed 20-run/297-record index
is an exact subset. Repeated raw references do not count as independent runs.
The eight phase/multimodal captures, four queue captures, and two additional
original shared repeat captures are included.

Fresh prepared01 public requests pass run/request disjointness for all four
candidates at 57 requests each. Their task/pool/exact public content agrees with
declared training inputs. The bounded model metadata graph has 26 indexed raw
references and one inherited smoke reference appearing only in materialized
source closures/strict smoke provenance. It is excluded from training identity;
the resource capture list, startup observation list, and generation observation
targets do not fit that smoke raw. Its payload was not opened. Large derived
arrays and implementation snapshots are explicit graph traversal boundaries.
The index does not prove an exhaustive historical absence of contamination.

## Joint initial prior v14

Use `artifacts/predictor-context-initial-prior-v14-20261028-training02/model.json`
and its `report.json`, SHA256
`bc1d8afea3f1fbd31d922e322f4cae0691d867bf47e0986a893380cb4790230f`.
The implementation is `tools/predictor_validation_20261027/context_initial_prior_v14.py`.
Training completely rederives the v13 supervision from its original strict
reports and compares it before constructing the prior.

The prior retains all 28 initial observations. It conditions on the first
pending owner/task/input/phase frontier and an explicit projection of public
resident, execution, thread, device, and observer setup. Complete future job
composition or request count is never an initialization feature. Supporting
captures receive equal weight, then matching jobs within a capture receive equal
weight. Each correlated holder/attained-age/readiness observation remains a joint
scenario. Missing-holder probability is not renormalized away.

Of these source observations, 25 have an owned kernel witness no later than the
initial query clock, establishing zero additional readiness delay. Two SS-long
observations identify holder and attained age but have no service upper witness.
One source gap leaves the holder and readiness unknown. Unconditional missing
holder mass is 1/28; conditional missing mass depends on the queried frontier.

```python
prior = InitialContextPrior(model)
result = prior.predict(
    first_pending_frontier=first_frontier,
    public_execution_setup=public_setup(context),
    allow_unvalidated_frontier_pooling=False,
    allow_unvalidated_observer_transfer=False,
)
```

New input/phase combinations can use an explicitly requested coarse first-owner
pooling hypothesis, then public-setup pooling. Observer/profiler changes need an
explicit transfer hypothesis; resident/execution/thread/device changes are not
silently bridged. These are candidate scientific assumptions, not approval
requests. A missing-holder scenario cannot be realized by choosing a first-ready
owner. `realize_initial_scenario` requires a named readiness rule: witnessed point,
lower-bound hypothesis, or upper-bound hypothesis. Missing upper bounds cannot
be turned into a finite upper time. Lower-bound realization of unresolved
readiness explicitly marks an optimistic unvalidated hypothesis.

All 14 jobs in each excluded capture receive a prior. On the 13/14 observed
holder labels, mean probability assigned to the true holder is 94.87%/89.29%.
However, same-holder empirical age-range coverage is only 15.38%/21.43%, and
paired-only age-mean MAE is 25.82/15.40 ms. These means are diagnostic summaries,
not initialization decisions. Prior idle/context history makes attained age
substantially less stable than holder identity. Missingness, conditional
exchangeability, and profiler-to-native transfer remain unvalidated; no MAR
assumption is used to erase missing mass. This prior is not a complete
initialization oracle or independent accuracy evidence.

The focused tests cover training/priming split boundaries, original-raw identity
deduplication, fresh public disjointness/content mismatch, closure versus fitted
smoke references, preserved context gaps/age, service-ready interval bounds,
future-feature rejection, continuous-age learning, nonpending target retention,
OOD reporting, whole-domain UNKNOWN accounting, capture-balanced correlated
initial scenarios, missing mass, future-workload exclusion, explicit observer
transfer/pooling, and readiness interval realization. All 29 tests pass.

## Source-bound handoff candidate v14

Use `artifacts/predictor-handoff-candidate-v14-20261028-training02/model.json`
(file SHA256 `cbf1577c79bc429e572c6c970f96b18b5796caeaeaa11b2fe3b360c557676f1b`)
and its `report.json`
(SHA256 `d8422c024056671211f2052ab48bc26d60f6c05baee0ffb18cf0590df10b2432`).
The implementation is `tools/predictor_validation_20261027/handoff_candidate_v14.py`.
The source is the stricter prequeued witness producer
`artifacts/predictor-prequeued-handoff-v14-20261028-training01/labels.json`,
SHA256 `ddd09b866086a87795a744ca76efd32b94bdd192b1c814ca3edf1386b952ee3d`.
This producer completely rederives original supervision. Candidate training
verifies those source receipts rather than repeating the same extraction.

All 35737 transitions train SAVE_END to RESTORE_START envelope gaps. There are
30763 observed prequeued kernel witnesses and 4974 missing witnesses. A witness
must have known recorded dependencies and a source-certified ready bound no
later than RESTORE_START. Its later observed start gives a readiness interval
`[0, observed_start - restore_start]`; this is neither exact readiness nor an
exogenous bound guaranteed under a different counterfactual schedule. Missing
upper labels remain missing, never become zero, and train a separate witness
presence model on the complete transition domain.

The fixed candidate uses capture-balanced L2 ridge for gap and observed-witness
delay in `log1p(ns/1000)` space, and L2 binary logistic for witness presence.
Regularization is 1; logistic L-BFGS has at most 300 iterations, with no tuning
from the two capture-exclusion diagnostics. Features are outgoing holder,
continuous attained age, pending owner/task/input/phase, and already chosen
incoming owner and pair interactions. Full future public workload, future kernel
residuals, and unobserved pending-age/work labels are not handoff features.

| Complete training capture excluded | Gap MAE | Pair baseline gap MAE | Observed upper proxy MAE | Pair baseline upper proxy MAE | Witness presence Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw 26305a33 | 353 ns | 614 ns | 385 ns | 417 ns | 0.0170 |
| raw 98c4c261 | 339 ns | 619 ns | 305 ns | 325 ns | 0.0152 |

Constant-gap baseline MAE is 622/627 ns; constant-upper baseline MAE is
431/337 ns. Constant witness-presence Brier is 0.1214/0.1183. Gap P95 absolute
percentage error is 1.95%/1.94%; observed-upper proxy P95 absolute percentage
error is 19.21%/19.64%. These are training capture-exclusion diagnostics from two
captures, not independent validation or exact readiness prediction errors.
Every gap is predicted, including missing-witness rows. All optimizer runs
converged. No utility or end-to-end effect was evaluated.

```python
candidate = HandoffCandidate(model)
handoff = candidate.query(
    held_owner=outgoing_owner,
    next_owner=chosen_restore_owner,
    context_age_ns=save_clock_ns - original_restore_start_ns,
    pending_frontier=forecast_frontier_at_save,
    readiness_rule="lower_bound_reference_availability_hypothesis",
)
```

That named rule uses delay 0 after predicted restore as an explicit unvalidated
reference-availability hypothesis. The other supported rule,
`prequeued_upper_reference_availability_hypothesis`, uses the predicted observed
upper-witness delay, also as an unvalidated hypothesis. Neither claims a
calibrated readiness bound. Both return witness/missing probabilities, pair
training counts, feature-range and unsupported-pair flags. The caller must retain
these diagnostics when composing initial prior, SAVE hazard, next-owner mark,
handoff and queue forecasts. RESTORE_START does not become exact service ready.

The OCR-to-text0 and G2-to-OCR pairs each have two gap labels and zero prequeued
upper witnesses. These still receive numeric statistical extrapolations with
`NO_OBSERVED_PREQUEUED_WITNESS_FOR_OWNER_PAIR`; such extrapolation does not
establish support. For example, an initial OCR-to-text0 query has an estimated
upper proxy around 552 us and modeled missing-witness probability above 0.999999,
not a measured finite upper. A chosen mark outside the pending frontier remains
permitted. Unknown identities and invalid/noninteger attained ages are rejected.

Seven handoff CPU tests cover explicit readiness hypotheses, nonpending marks,
retained missing mass, unsupported-pair extrapolation, scalar global fallback
for a pair absent from training, capture balancing, continuous regression,
future-feature rejection, missing-upper semantics and model-array validity.
The combined focused suite now has 36 passing tests. Training01 remains
preserved; training02 fixes the unseen-pair baseline fallback and validates
finite model arrays without changing statistical hyperparameters.
