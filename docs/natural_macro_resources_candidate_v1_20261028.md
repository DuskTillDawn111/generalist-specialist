# New natural-domain resource training candidate

`tools/predictor_validation_20261027/natural_macro_resources_candidate_v1.py`
uses only the three accepted natural progress-worker training captures. It
rechecks raw protocol reviews, full training reviews and exact macro extraction.
The old semantic worker, profiler captures and old twelve-request priming
startup means are excluded.

Training output is
`artifacts/predictor-natural-macro-resources-v1-20261028-training04/report.json`.
It retains 150 measured requests, 1,002 macros and all 14,293 board polls. Three
capture tables contain one global START-to-READY phase/input timeline with idle
gaps, the four-resident roster, every original poll midpoint and query bracket,
and original observation/startup boundaries. Inactive resident memory remains
part of simultaneous owned memory.

The state key contains exact owner, task, public input and actual response
phase, plus all resident owners. This is host response activity, not observed
SM activity. Forward index, shape and history are deliberately omitted by the
initial stationary resource hypothesis. No pairwise addition or per-model
power attribution is performed.

A poll labels a state only if its entire NVML query bracket lies inside that
state. A power training pair requires consecutive polls wholly inside the same
state; its midpoint trapezoid contributes once. Boundary polls, short intervals
and absent states remain in the tables with UNKNOWN support. Memory labels are
simultaneous owned sampled peaks. State means weight captures equally; at least
two distinct captures must support a state. The complete model has 144 states,
48 with power support and 58 with memory support.

`ResourceTrainingProvider(model_path, target_context,
allow_training_resource_hypothesis=True)` verifies exact new execution identity,
source hashes, rederives original raw protocol/training reviews and exact macros,
checks that their entire source closure is retained, and recomputes
raw-to-table-to-fit values. Its `power` and
`memory` methods accept `actual_START_to_READY_handlers` and the complete
`resident_owners` roster. Unseen public inputs and unsupported full states return
UNKNOWN. It is a hypothesis interface, with no core-provider eligibility.

`predict_timeline(timeline, provider, target_context=..., execution_binding=...)`
requires explicit exact target context and a caller declaration binding the
exact timeline hash to that execution. This declaration is not independent
authentication of a real execution. The adapter accepts the root's global resource
timeline, integrates power over `measurement_overlap_ns` once and takes memory
maxima through the full drain. Startup joules remain separate. Existing scoring
alone applies startup amortization over 3,600 seconds and the original memory
safety factor, once each. No scorer, weights or safety constants change.

The profile collector exposes a combined measurement-and-drain end, without a
separate scoring measurement deadline. Therefore the table retains the complete
observation energy but final scoring measurement energy and E remain UNKNOWN.
Startup begin to observation start includes all eighteen actual priming inputs,
bound by public input bytes and the exact context. Its three-capture mean is
7,042.506 J; the premeasurement owned sampled peak is 29,866 MiB. Each full raw
monitor sampled maximum is 34,428 MiB. These are empirical sampled quantities,
not bounds or continuous maxima.

Whole-capture leaveout uses only the other two captures and supplies actual
held-out state durations. Power support covers 94.64%, 96.34% and 95.99% of the
complete observation duration. Known-duration energy errors are 0.105%, 0.383%
and 0.975%; these exclude unsupported durations, so complete observation energy
and memory error remain UNKNOWN in every fold. Startup errors are 1.501%,
0.388% and 1.090%. This is conditional resource replay, not prospective timing,
resource calibration or independent utility validation.

Sixteen candidate tests and four global-timeline tests pass. Provider
construction recomputes the published tables and fit; a changed old worker is
rejected and an unseen input stays UNKNOWN. Independent resource review and an
independent prediction gate are still required before native integration.
