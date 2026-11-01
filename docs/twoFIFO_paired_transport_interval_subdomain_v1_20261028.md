# Paired transport: exact single-owner intervals and multiowner limits

The frozen v5_2 engine correctly returns an UNKNOWN world as soon as one paired
transport START is non-singleton. This can discard useful finite timing support
for every request, including requests whose own clocks were already supported.
A synthetic CPU example below has zero point-world support while a complete
cohort interval remains finite. The native coverage rate has not been measured:
this work reads no native or validation labels and makes no empirical forecast
coverage claim.

The new independent module `twoFIFO_single_owner_transport_intervals_v1.py`
implements an exact, explicitly declared subdomain. It changes no frozen engine,
ensemble or whole-block runner. Its input is the full public cohort, fixed joint
plans, rates, protocol slacks, typed paired transport constraints and explicit
initial sender/worker/protocol stations. Its closed caller declaration requires
one owner in the full modeled world, no external visible or future producers,
an initially empty visible-peer set, and the Cartesian product of the per-request
transport feasible sets. This is a conditional mathematical model, not evidence
that the latent coordinates have any probability distribution or that their
transfer to a new schedule is physically qualified.

For request i let U_i=min(W_i,H_i), q_i be the source READY-minus-submit offset,
s_i the target sender submit, and R the target previous actual READY. The exact
paired source constraint and target recurrence are

```
0 <= u_i <= U_i
x_i = H_i - max(q_i,u_i)
B_i(R,u_i) = s_i + max(R-s_i,u_i) + H_i - max(q_i,u_i)
R_i = B_i + fixed_handler_duration_i
```

All target submits are fixed before worker execution: each owner's sender uses
the previous parent write record, not worker RESULT or READY. Plan ordinals are
assigned by public arrival before sender enqueue sorting, matching v5_2. For
fixed u, B_i is nondecreasing in R. For fixed R, the difference
`max(R-s_i,u)-max(q_i,u)` is monotone in u: decreasing if R-s_i>=q_i and increasing
otherwise. Thus its extrema are at 0 and U_i. The implementation evaluates both
endpoints for each of the lower and upper previous READY values. This is not a
claim that all-low u and all-high u schedules give global bounds: each extremal
schedule can choose a different endpoint at each request.

Induction gives stronger support than a generic interval hull. Every possible
START lies between the recursively constructed lower and upper STARTs; each
entire vector of START extrema is jointly attainable. At step i the earlier
extremal schedule is feasible, and u_i is a new free feasible-set coordinate.
Choosing its minimizing or maximizing endpoint preserves feasibility and extends
the extremal schedule. Hence all STARTs and READYs reach their lower bounds
together, and likewise all their upper bounds. This proof depends on fixed
handler durations and no peer-dependent response rates.

Protocol delays are not replaced with an additive RESULT lag. With one worker,
all marks and READY have a fixed issuance order, even across adjacent handlers.
Every occurrence is B_i plus its fixed macro-work/rate offset. For message m of
the single stdout producer, the implementation uses the frozen recurrences

```
reader_m = max(occurrence_m, reader_(m-1)) + reader_slack_m
dequeue_m = max(reader_m, collector_(m-1)) + dequeue_slack_m
collector_m = dequeue_m + processing_slack_m
```

Initial reader/collector backlogs are retained. Nonnegative max-plus recurrences
are monotone in every occurrence. The same two attainable START schedules
therefore simultaneously extremize every RESULT reader receipt and processed
receipt. Collector processing does not release the worker, and this owner's own
processed phase is excluded from its peer covariates; processing backlog cannot
change its rates. Multiple producers, external future messages and unsettled
pending notifications fail the subdomain gate rather than being ignored.

Subtracting fixed arrivals gives sharp complete-cohort E2E bounds. The original
linear-interpolated P95 is coordinatewise monotone. Because both latency vectors
are simultaneously attained, their two P95 values are also sharp global bounds
in this subdomain. This argument applies to all finite offered requests; no
completed subset is selected. Per-request fixed-drain completion has ALWAYS,
NEVER or transport-dependent status. The exact incomplete-fraction endpoints
are attained by the same two schedules. If any feasible world is drain censored,
the original complete-cohort fixed-drain P95 upper envelope is not reported as
a numeric metric. Untruncated model timing is separately labeled and never
substitutes for the original scorer's missing latency. Missing initial state,
missing donors/rates/slacks or a failed subdomain gate retain every request as
UNKNOWN.

Optional scoring calls the unchanged original scorer on the two schedules.
Q remains a frozen training expectation; T/L/P use the original window, fixed
drain and upward rounding of rational clocks. Monotonicity also survives that
rounding. M/E and full U remain UNKNOWN. For two candidates A and B with bounded
utilities, the universally sound difference enclosure is
`DeltaU in [U_A_low-U_B_high, U_A_high-U_B_low]`; it may be loose when the
candidates share latent coordinates. Full DeltaU requires resource intervals or
an explicit verified common-resource cancellation contract. Current common
startup alone does not cancel candidate memory/measurement energy. A weighted
Q/T/L/P difference must be labeled partial utility, never full DeltaU. Missing
or unsupported worlds retain their predefined weight, with no renormalization.
Finite donor-world timing intervals can be combined as weighted support
intervals for expected metrics; they assign no probability within a latent
feasible set and are not calibrated risk intervals.

Multiowner endpoint enumeration is unsound. The saved CPU counterexample runs
the frozen v5_2 explicit-world branch, with A and B each having work 10, solo rate
1, same-phase peer rate 1/2, and zero protocol slacks. A starts at 0. B has legal
source W=H=q=20, so its source continuation is identically zero for u in [0,20].
With target submit=previous READY=0, B START=u. The outcomes are

| B latent u | RESULT A | RESULT B | cohort P95 |
| --- | --- | --- | --- |
| 0 | 20 | 20 | 20 |
| 10 | 10 | 20 | 19.5 |
| 20 | 10 | 30 | 29 |

The interior feasible point already lies below the two endpoint P95s. These
three runs demonstrate failure of the proposed endpoint bound; the artifact
does not claim they enumerate all regimes or prove the global minimum.

A later multiowner extension can use event-order branches with affine clocks
and rational polytope guards. On a fixed branch, split both transport max terms,
every protocol FIFO max, next-event comparisons, visible-phase updates and
P95 sorting into explicit inequalities. With fixed rate-table entries, virtual
ages and candidate event times are affine in u within that branch. Branches must
retain the frozen equal-time processing/issuance policy. Optimize each output
linear form over every feasible polytope; P95 needs rank branches, and original
integer scoring additionally needs a certified ceil enclosure or integer-clock
partition. Drain-censored metric undefinedness is a separate branch. A missing
visible-rate key remains UNKNOWN, not a zero or a dropped branch. Exhausted
branch budgets must retain unexpanded support with a conservative outer bound
or explicit UNKNOWN. Taking the min/max of explored witnesses would be unsafe.
This can grow exponentially with competing producers, marks and processed
phase transitions, so a general solver is not implemented here.

The executable minimum is now the isolated single-owner evaluator and seven
CPU tests. Twenty-five feasible interior grid worlds are compared with frozen
v5_2, both simultaneous extremizing schedules are checked against its exact
START/READY/RESULT/processing clocks, and nonzero reader/collector backlog,
drain censoring, whole-cohort UNKNOWN, original scorer and the two-owner
endpoint counterexample are covered. The grid is a sanity check; the induction
and monotone protocol argument establish the continuous feasible-set bound.
No native labels, numerical training fit, GPU or final validation is involved.
