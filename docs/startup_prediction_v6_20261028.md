# Flow-bound premeasurement startup prediction, 2026-10-28

`startup_prediction_v6.py` fixes the target/provenance gaps independently found
in v5. The v5 source, training model and original independent audit are retained
unchanged. The v6 API is:

```python
prediction(
    model_path_v5, flowplan_path, candidate_id,
    allow_native_priming_transfer_hypothesis=True,
)
```

It reopens every strict training review, raw, plan and context in the v5 model.
It verifies the recursive evidence closure and original implementation
receipts, unique raw capture identities, natural training-only requests,
actual priming and full/native context identities. It recomputes each original
target from raw telemetry and then recomputes both means. Observations,
contexts, capture counts, projections, priming identity, windows, means and
scoring flags must match the v5 artifact. Nonfinite or negative resource values
are rejected. Editing a model's numeric means cannot bypass training evidence
validation merely by creating a new receipt for the edited model.

The energy window is startup_begin through active_begin, including actual
loading, readiness and priming. The v5 independent audit already confirmed all
seven captures match the original flow energy integrator to less than 1e-9 J
at these boundaries, and that sampled premeasurement memory peaks match.
Prediction remains a conditional training mean, not a capacity/confidence
bound. No safety factor is applied inside this model.

The target is the actual immutable flow plan, not a loose context/input list.
Validation requires the known-input v5 plan schema and self hash, actual runner
and complete collector receipts, exact catalog bytes, zero warmup, a declared
candidate with the original shared/no-admission resource configuration, and
the original four physical owners. It verifies both target context and
separately referenced priming profile context, their complete executing-source
receipts and weight stats. Both original full identities remain explicit;
their native projections must match under the declared transfer hypothesis.

The actual runtime-input catalog must equal its frozen snapshot, be at the path
the real collector reads, and agree between target/profile contexts. The
priming profile self hash and each selected input's task, size and exact public
content identity are verified. Text hashes use the actual public
context/question; media hashes are checked against actual bytes. The twelve
ordered priming requests are compared by content with all training captures,
not merely by pool ID. A target with warmup or admission requires separate
evidence and is rejected by this API.

The actual target plan is
`artifacts/predictor-gpu-20261027/known-input-dag-co-run-v5-20261028-prepared01/plan.json`.
Its target full context is
`40dde9e69b8e9556353c8c5f649f0d84bc07af9cb5c7c627cc9287797e7c3f8b`;
the actual eight-input priming profile's full context is
`8c7fa532c5cc23cc0ad329943f9c14aaac4bfbe0dc7de9d3debfe715e23e21ab`.
They are distinct. Their explicit natural native projection and twelve actual
priming input contents match the seven-capture training evidence.

Real source revalidation produced predictions for all four shared candidates:
startup energy **5822.193265376832 J** and raw sampled premeasurement owned peak
**29766 MiB**. Artifact:
`artifacts/predictor-startup-v6-flow-bound-review-20261028-final01/predictions.json`,
SHA256 `af7805b59259ac506008127fad3b9dedca75f2003a12af783acf34e721d531ae`.
Each prediction binds its exact plan/candidate, target/profile original IDs,
actual priming contents, all training recomputations and evidence. The runtime
gate independently recomputes it before accepting an embedded prediction.

Seven focused CPU tests pass. They reject negative/positive/NaN mean edits,
changed observed targets, warmup, admission, wrong plan hashes, incomplete
collector sources, changed priming content under unchanged IDs, nested training
reviewer changes, modified live runtime catalog and changed media bytes. They
also verify explicit transfer opt-in and preservation of different complete
contexts. Tests and real source checks do not establish predictive accuracy,
future physical equivalence, instantaneous memory bounds or final utility gain.
No GPU workload was started for this task.
