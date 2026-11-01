# Actual two-FIFO generated-world queue diagnostics v1

`tools/predictor_validation_20261027/actual_twoFIFO_queue_diagnostics_v1.py`
binds a v4 generated world to its complete, ordered public request cohort and
original measurement/drain boundaries using the v4.1 public-query adapter.
It does not change v4, admission, the six-term utility, or stability acceptance.

The complete public arrival trace supplies multiscale maximum arrival count and
response-reference-work envelopes in half-open `(t-scale,t]` windows. The optional
work mapping uses `(owner, task, pool_id, public-key arrival ordinal)`. Its values
are sums of macro response-reference coordinates, not solo wall service or
physical SM/DRAM/L2 demand. Missing work remains unknown even if a completed
exposure could retrospectively reveal it. Supplied work must agree with consumed
exposure and, for completed handlers, with the full integrated reference work.

Every generated event cut retains two separate partitions of the full public
cohort. The worker axis distinguishes future arrivals, arrived but unsubmitted,
submitted but not START, actual START-to-READY, and released handlers. The RESULT
axis distinguishes not occurred, in transit, and reader received. This preserves
RESULT received while a handler still runs and READY before RESULT delivery.
Sender FIFO and worker FIFO use generated enqueue/submit order, including equal
submit ties and enqueue orders that differ from scheduled arrivals.

Arrival-to-START queue recovery includes sender waiting and worker waiting.
RESULT backlog recovery includes only already arrived requests; future requests
remain in the complete offered denominator but do not become current backlog.
FIFO convoy counts pending requests behind a running handler or a reserved
waiting head. It is not a resource-feasible HOL diagnosis. Resource feasibility,
SM/DRAM/L2, power and memory remain `None`. Current native risk features remain
`None`; all snapshots and future envelopes describe explicitly supplied CPU worlds.

The output includes exact recovery/clearance events, censoring at the generated
cutoff, and original-drain unrecovered requests. Original-drain outcomes stay
unknown when simulation stopped early. E2E uses scheduled arrival to RESULT
reader receipt. A complete-cohort P95 uses the original linear empirical
percentile convention on exact unrounded clocks; it stays unknown for any missing
RESULT. The original score bridge separately rounds fractional clocks upward by
less than one ns, as already declared by that bridge.

The synthetic report contains eight CPU scenarios and nine tests: exact half-open
arrival envelopes, full-cohort conservation, overlap between worker and delivery
axes, burst recovery, sender blockage/reordering, right censoring, unknown work,
foreign or inconsistent costs, and exact rational progress/input immutability.
It uses no GPU data or heldout labels and establishes no GPU prediction accuracy,
calibration, busy transport identification, native/provider eligibility, physical
resource complementarity, stability proof, or final routing gain.

Report: `artifacts/predictor-actual-twoFIFO-queue-diagnostics-v1-20261028-cpu01/report.json`.
SHA256: `d956f31e61c1ac1f0df4a6da739d7b07be26121bc8b42b6998358aa061edc226`.
