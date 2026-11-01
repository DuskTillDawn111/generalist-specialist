# Visible busy DAG conditioning v8

`dag_busy_posterior_v8.py` supplies an executable interface between the actual
reader-visible observer and complete v9 DAG scenario rollouts. It deliberately
reports empirical conditional scenarios rather than a calibrated posterior,
risk guarantee, successful predictor validation, or certified deployment gain.

Each scenario binds independent training run receipts and public-input
conditioned node-duration, request-start supply, node host-supply, lifecycle and
reader-emission evidence.
It runs the actual DAG over every original declared request and future arrival.
The original phase start, deadline, drain, route and scoring cohort are retained.
No input can provide target suffix completion, response body, generated token
count or future quality.

Reader START/PHASE/RESULT/READY patterns and worker/receipt timing support select
joint candidates. No RESULT by the cutoff means that RESULT's reader receipt
is still in the future. A valid candidate may already have ended its handler
or emitted worker READY while the corresponding reader messages are in transit.
END, RESULT and READY residuals are taken from the same actual DAG/lifecycle draw.
They are not independently averaged or combined. PHASE uses a separately declared
training observation model; it never identifies a kernel cursor. Unsupported
phase/input, too few independent runs or no surviving joint scenarios remain
UNKNOWN with a concrete additional collection plan.

The implementation locates actual atomic DAG checkpoints on either side of the
cutoff using fresh deterministic rollouts. A cutoff inside an interval remains a
bracket; it is never inserted as an artificial simulation event or restored as
observed GPU progress. Forecast suffixes come from the uninterrupted full DAG
trajectory. Before-rollout and checkpoint provider identities must agree.

Each surviving trajectory is scored before any aggregation. Real completed
prefix rows replace predicted prefix rows with their original arrivals and
actual RESULT reader receipts. Resource integration is clipped to the predicted
suffix. Missing observed prefix energy/memory leave E/M/full_U UNKNOWN. Six equal
weights and the original scoring window are enforced. No stability rule changes.

Current CPU tests demonstrate handler END and worker READY before an invisible
RESULT/READY receipt, rejection of a physically ended-and-received scenario,
visible PHASE support filtering, unchanged output after poisoned future-body
replacement, full future-cohort P95/scoring and refusal of heldout training
sources or reweighted objectives. These establish software behavior, not physical
accuracy or calibrated uncertainty.

## Explicit request-start supply and remaining real training

No measured joint scenario model is fitted by this interface. Existing adaptive
profile-job arrivals cannot become a prospective declared workload. The parent
is adding a separate native prospective training collector. That collector must
retain all requested worker/reader boundaries, full public trace/routes/window,
independent repeat-run identities and verified initial priming/READY inventory.
Training construction and support radii must use cross-fitted training runs;
the previous failed heldout validation stays untouched.

The new independent `cuda_dag_simulator_v9.py` preserves the v7 numerical result
field for field when request-start supply is absent. With an explicit pure
`request_start_provider(row) -> absolute_not_before_START_ns`, an idle owner
waits for its FIFO head's bound. Future bounds create actual engine wakeups.
Public arrival remains unchanged for queue wait, latency and the scoring cohort;
a later request cannot bypass a delayed head. Providers can optionally implement
`from_owner_ready(row, owner_ready_ns=...)`. This pure selected callable receives
the preceding explicit predicted worker READY (initially the original phase
start), so an independently trained eligible-to-START lag is added after FIFO
waiting: preceding READY=100 plus lag=5 yields START=105. An arrival-based bound
would incorrectly produce 100. The small derived last-READY cache is reconstructed
from complete finish/receipt-pending records during continuation. The v9
checkpoint wrapper retains
all v7 structural/resource checks, binds the new provider and implementation
sources, including the selected `from_owner_ready` method. Historical STARTs
must form the original owner FIFO prefix, obey each predecessor's READY and
their request-start bound. This check also applies with no provider or a provider
that returns an earlier absolute bound. All completed and receipt-pending records
also rederive RESULT and worker READY from recorded handler END and the bound
joint lifecycle provider; synchronously forging READY, wake and unready maps is
rejected.

The earlier busy adapter constructed `VisibleStartSupply` from each scenario's
`request_start_delay_ns_by_input` for every original public input, including
future requests. Explicit actual-submit and observed START facts added host lower
bounds; they never implied CUDA progress. This implementation remains only for
declarative CPU fixtures; source-rederived support uses the exact conditional
anchors described below. The scenario is still checked against
the complete worker/reader event pattern and training timing support. No future
input supply, arrived-but-unsent dispatch ambiguity, or insufficient independent
joint training stays UNKNOWN. A CPU example has arrival=100, actual submit=104
and observed START=108; execution starts at 108, queue wait remains 8 and the
future request retains arrival=200. This is actual v9 DAG execution.

The source-supported interface is now connected, while fitting a measured joint
scenario model and evaluating independent prediction accuracy remain required.
The support document declares `generation_work_scope`, `joint_scope`,
`timing_radius_basis`, `observation_tolerance_ns`, immutable training `sources`,
and scenario-specific `training_run_ids`, `prior_mass`, public-input request-start
delays and joint reader/phase `emissions`. Arrays retain successive occurrences
of the same alias/task/input within one original run. The factory accepts scenario ID, the
original phase start and the supplied request-start provider. Source receipts,
training split/context and complete input support are checked before execution.

## Source-rederived support loader

`dag_busy_support_v9.py` now consumes strict reviewed native queue-training
captures. For measured data it reruns `native_queue_training_review_v8.review`,
compares the full review, reloads bound plan/raw receipts, and reconstructs all
timing fields. The conditional host lag is exactly
`START - max(original arrival, preceding same-owner worker READY)`, which removes
FIFO waiting before fitting. Handler wall time remains a diagnostic; it is never
used as a node-service or whole-request SSG multiplier.

Each whole run remains one empirical scenario, including input occurrence order,
phase worker offsets/reader delays, START/READY reader delays and the joint END
to RESULT/worker READY vector. Equal prior masses follow the observed independent
run draws. Timing support errors leave an entire run out and compare each input
occurrence with other compatible runs. The maximum is explicitly an empirical
host/reader relative-timing support radius. It does not calibrate absolute DAG
rollout error, a tail probability or a posterior confidence bound.

The loader reconstructs the full fit from original raw and compares every field.
Changing emissions, start lags, occurrence order, radius, priors or definitions
and re-signing the document is rejected. Measured busy conditioning requires
this loader and the same source-rederived `JointLifecycleFromSupport` draw used
inside the actual DAG. Unsupported public trace/route transfer stays UNKNOWN.
CPU fixtures require explicit opt-in and can never qualify measured support.

Visible host START, END and worker READY timestamps are exact facts: candidate
worker times must match them exactly. A broad empirical timing radius cannot
relax an observed host boundary. Reader-emission compatibility and unseen-event
survival remain separate checks. When exact evidence leaves too few independent
joint training runs the answer remains UNKNOWN. The original failing case
(visible START=100 versus incorrectly accepted candidate START=102) was preserved
before correction in the visible-start counterexample artifact.

Four strict-reviewed prospectively declared native queue-training runs now
support a raw-rederived joint fit. They do not constitute independent prediction
validation. The first real old-provider whole-horizon replay failed because the
projected training PHASE/reader chronology exceeded predicted DAG boundaries.
The original failure is preserved in
`artifacts/predictor-native-queue-busy-replay-v9-20261028-training02/failure-report.json`.
PHASE offsets must never be clipped, and predicted END must not be stretched to
hide that mismatch. A bad joint scenario is rejected individually; insufficient
surviving independent runs stay UNKNOWN.

## Exact conditional START and atomic prefix screening

`dag_visible_anchor_v10.ConditionalVisibleStartSupply` conditions a visible
START on its exact worker timestamp. It computes the implied host lag relative
to the same scenario's `max(original arrival, preceding worker READY)` and
checks the residual against the maximum leave-entire-run-out **start_lag** error
for that public alias/task/input occurrence. Handler wall, PHASE and unrelated
input timing errors cannot supply this radius. Only a supported observation is
anchored. START before the scenario's preceding READY or actual submit is
inconsistent. The radius is empirical support, never a calibrated probability.
Future requests retain the scenario's READY-relative trained lag. END and READY
remain exact comparison facts; no observed host boundary creates a CUDA cursor.

The v9 engine can pause at the first existing atomic boundary at or after a
requested time, without inserting an event. Busy conditioning requests the
first boundary strictly after the cutoff, compares the actual DAG prefix to
visible events, and resumes its unchanged checkpoint only after compatibility.
The complete original cohort, event budget, measurement/deadline/drain and
future arrivals remain bound. Rejected prefixes do not consume the entire future
service schedule. `prefix_screen_only=True` produces an explicitly partial
training diagnostic with no candidates, posterior, full-horizon forecast or U.
CPU checks compare resumed and uninterrupted complete results and verify that
rejected prefixes never evaluate future request service.

The existing lifecycle `residual_handoff` supports known END and explicitly
confirmed pending boundaries only. It cannot treat a missing reader event as
proof of a physical pending boundary. The new trajectory interface preserves
that distinction and does not use whole-request SSG multipliers or an elapsed
request fraction as physical progress.
