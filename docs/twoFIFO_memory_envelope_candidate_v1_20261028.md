# Isolated memory-envelope candidate ③

`twoFIFO_memory_envelope_candidate_v1.py` is an experiment model, not a
replacement for `twoFIFO_resource_provider_v3.py`. Its current status is CPU
implementation checked; no native campaign has been fitted or qualified.

The query state uses only semantic events already processed before the beginning
of an owned-memory poll. It retains the resident roster, each owner's current
task, exact public input pool and last processed actual BEGIN shape, componentwise
past shape maxima, and current and past maximum visible concurrency. Entry and
OCR shapes and allocator/KV/workspace/release state remain explicitly unknown.
Semantic concurrency is not hardware occupancy or a live-allocation observation.

For poll `i`, the anchor `P_i` is the largest owned-memory sample from an earlier
poll that completely finished before poll `i` began, with samples restricted to
startup onward. The fitted label is `max(0, M_i - P_i)`. The current poll label and
later shapes or memory samples never enter its features. Each exact compressed
state retains the empirical increment list in every supporting whole block,
with equal weight between supporting blocks. A conditional empirical maximum
requires at least two whole blocks; missing support stays UNKNOWN.

This is a sufficiency hypothesis. Shape maxima can compress two different
ordered histories into one state, while allocator bins, retained reservations,
KV lifetimes, workspaces and releases can still differ. The empirical maximum
does not establish a probabilistic or physical capacity bound. No safety factor
is added here; the original memory safety factor must be applied once at the
original downstream decision boundary if a future resource predictor is
qualified.

The complete-campaign entry rederives all twelve native reviews and macros
before fitting. It requires three complete four-shard blocks under one physical
execution identity. The report contains the three-block model, every leave-one-
whole-block-out fit (eight training attempts and four held-out attempts), and all
case and poll evaluations, including unsupported states and underestimation.
It does not choose a successful fold or omit failed cases. Before twelve complete
attempts, the entry fails without opening raw data for extraction or fitting.

The held-out evaluation is **filtered conditional replay**: each query receives
the actual held-out past processed events and completed memory polls. It tests
the compressed state's next-sample conditional error. It cannot establish a
prospective generated-world memory forecast or full-horizon six-term utility.
That next step requires a generated shape/resource-state trajectory and a
causally evolved peak anchor, without injecting realized held-out states.

The read-only complete-campaign command is:

```bash
python -m tools.predictor_validation_20261027.twoFIFO_memory_envelope_candidate_v1 \
  --campaign CAMPAIGN_JSON \
  --training-reviews REVIEW_1_JSON REVIEW_2_JSON ... REVIEW_12_JSON \
  --output NEW_OUTPUT_DIRECTORY
```

Author CPU receipt:
`artifacts/predictor-twoFIFO-memory-envelope-candidate-v1-20261028-cpu03/report.json`,
SHA256 `ef7a4448d77a3a7c5e966fe30a0bd1a80b4b96aa1d85851e9c70e5177c893d22`.
Seven tests check the whole-block gate, retained underestimation and UNKNOWN,
future-shape and future-poll independence, commutative high-water compression,
the strict earlier-completed-poll peak anchor, and physical-domain/schema
rejection for direct evaluation. CPU fixtures remain synthetic
implementation evidence and do not qualify native prediction performance.
