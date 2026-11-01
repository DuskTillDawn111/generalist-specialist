# Conditional owner/phase twoFIFO reference v5_2

v5_2 is isolated from the frozen v5_1 engine. Its public `simulate` signature
retains full settled-prefix seeds, tracked relative producer offsets and exact
JSON continuation. `transport_draws` now accepts either an explicitly supplied
`TransportDraw`, preserving v5_1 semantics, or the immutable
`PairedTransportConstraints` from `twoFIFO_paired_transport_v3.py`.

A constraint carries the complete paired observed sender and source recurrence:
enqueue extra, sender-turn extra, write-record duration W, source submit-to-START
H, and signed source previous-READY offset q. Source validation accepts only
exact nonnegative rational sender durations and exact signed rational q, and
reconstructs a closed canonical feasible set. Float/bool clocks, extra/missing
fields and inconsistent derived bounds are rejected.

The sender branch computes enqueue, submit and write record directly. It emits
`stdin_line_available_ns=None` and `stdin_line_availability_known=False`; no
unobserved line time or continuation delay is replaced by zero. At the actual
generated worker FIFO head release, the engine projects the same complete paired
source set against generated submit and previous READY. A singleton schedules
START. A non-singleton raises `UnknownTransportInterval`, an `UnknownEvidence`
subclass, whose `evidence` contains exact START bounds, public key and ordinal,
current request ID, the full offered request ID list and denominator. The caller
must retain the affected forecast as UNKNOWN; conditioning on only successful
rows would change its denominator. Evidence is not a selected latent world.

The supported singleton cases include idle source to idle target (identified
composite H), and source/target READY both dominating write (identified
continuation H-q). Intermediate schedules use the exact feasible envelope. The
same paired W, line delay and continuation transferring to a changed schedule is
an explicit assumption; the engine does not fit or certify that transfer or any
latent split probability. Actual READY releases the worker; processed notices
still control peer visibility. Full exact resume fingerprints bind every paired
constraint and seed, so a continuation cannot silently change its latent set.

`twoFIFO_joint_request_law_v4.py` retains the new v3 collector/intake and all twelve
native attempts/three whole-block fitted-root gates. `compile_draw_v4(vector,
public_key_ordinal)` emits the entire RequestPlan, every START/phase/query/RESULT/
READY protocol slack and typed paired constraint together. It requires no dummy
latent transport argument. Equal block weights, whole-block leaveout donors,
complete vectors and UNKNOWN public-input support remain unchanged. This is a
conditional compiler, not independent validation or a fitted busy-transport law.

Twenty CPU tests pass across engine, compiler and partial-ID suites. They cover
idle/busy support and exact unsupported intervals, decision using generated READY,
full offered denominator on exceptions, all nodes and protocol marks in a real
joint fixture compiled into the engine, source START reconstruction, explicit
TransportDraw equivalence to v5_1, exact JSON continuation and settled seed offsets,
constraint fingerprint tampering, closed canonical schemas, fractional feasible
set enumeration and training-role gating. GPU execution and native predictor
qualification are outside these checks. v2 capture evidence remains on HOLD.

Generated message identities include the complete public key and arrival ordinal.
The returned generated-to-source identity map preserves each original event
ordinal. This keeps two structurally identical response plans with different
protocol timing vectors distinct in the engine's plan-keyed slack map; one
public slot cannot overwrite the other's measured joint notices.
