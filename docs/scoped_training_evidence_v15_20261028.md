# Exact operational source closure with failed-history boundaries

The actual learned campaign sealer passed the original prepared-plan check and
the complete training-index recheck: four candidates, 57 requests, and the exact
1,005 measured-plus-priming records from 34 strict training captures. A direct
`availability_training_sources` check also confirmed that both actual availability
model fitting captures are in that index.

The original unbounded historical resolver could not finish this graph. It
reached an obsolete test-source receipt through a recorded failed predecessor
capture, not through an additional fitted-label source:

```text
packet02 strict training review
  → context.json / prior_failed_evidence[0]
  → failed unified-corun-v8 raw.json / plan
  → plan.json / parameters
  → parameters.json / prepared_repeat
  → prepared-repeat.json / source_before
  → source-before.json / sources[342]
  → tests/test_dag_checkpoint_v7.py, historical SHA 7db85167…
```

The current test and seven surviving same-name snapshots have SHA 9a49a7cb…;
none has the required original bytes. The old receipts and snapshots remain
unchanged. Full evidence, incoming JSON paths and the snapshot search are in
`artifacts/predictor-learned-preseal-readiness-v15-20261028-readonly03/`.

`scoped_training_evidence_v15.resolve_closure` defines a narrow operational
boundary. Its caller supplies exact context receipts from independently checked
strict training captures. Only references below such a context's top-level
`prior_failed_evidence` list are opaque. The context must be `context.json`,
explicitly training-only, with the resident and execution-source structure.
An arbitrary model with the same field name, an unqualified context, or a nested
same-name field receives ordinary recursive traversal.

Every opaque target is still resolved through the unchanged
`HistoricalReceiptResolver`, verified against its exact bytes, materialized in
the source set and reverified at the end. The output records each boundary's
parent, JSON path, original receipt, resolved receipt and reason. It does not
parse that target's nested historical metadata. All remaining edges use the
ordinary exact-receipt and historical-snapshot rules. No source category such
as tests is generally ignored.

A second actual run reached an old full-repository source inventory through the
current training plan's parameters, independently of failed-history contexts.
The original `prepared_repeat` preparation report has SHA `2a0e88c0…`; its
`source_before` target has SHA `2fb58b41…`. The frozen actual collector reads
only `params['stages']` to select jobs and checks the parameters file receipt;
neither collection nor DAG/availability fitting reads this retained preparation
audit to select numerical labels. Worker, collector and reviewer implementations
have independent frozen receipts in the actual training context.

After independently reviewing that incoming path and the frozen collector,
one source-bound explicit policy admits exactly this parent/JSON-path/target
triple as a byte-verified opaque audit inventory. No general `source_before` or
`prepared_repeat` field exemption exists. All other children of the preparation
report recurse normally. A normal edge to the same inventory still upgrades it
to full traversal. The policy and its implementation receipt/digest are included
in the resolution artifact. Original historical receipts remain untouched.

A separate normal edge or explicit root to an opaque target upgrades it to
normal traversal, regardless of visit order or snapshot aliases. Opaque
reachability never marks a source as parsed. An optional pre-JSON read guard
runs before every normal parse, including upgrades. The campaign guard admits
only exact indexed training raw and explicitly classified inherited runtime
smoke raw, and rejects known validation-outcome documents. A normal nonindexed
raw cannot inherit the opaque exemption from another edge.

The guard checks the original receipt, resolved snapshot and all historical
aliases declared for that snapshot before checking whether the source was
already parsed. This matters because the manifest's own snapshot field is a
normal edge and can arrive first. Rejection after parsing a renamed snapshot is
insufficient. Read-spy tests and an independent four-test review confirm that
normal raw/observed aliases are refused before any sensitive snapshot JSON is
parsed, including transitive alias chains and both visit orders.

Repeated exact receipts use a resolution cache because inherited source
inventories otherwise hash large profiler files repeatedly. This cache never
suppresses normal upgrades; final verification still rehashes every materialized
source. Tests include drift after caching, both upgrade visit orders, exact
snapshot aliases, arbitrary/nested-field rejection, per-edge boundary recording,
and validation guard execution before parsing.

This is the complete operational model/training closure under its declared
failed-history boundaries. It does not establish an exhaustive historical
closure or historical noncontamination. Numeric four-candidate preflight remains
a prerequisite for future sealing; this source review does not bypass it.

The final actual-source review is frozen at
`artifacts/predictor-scoped-training-evidence-v15-20261028-readonly06/`.
It passed in 273.867 seconds with unchanged sources: 5,868 materialized sources,
926 normally parsed JSON documents, three exact historical snapshot
substitutions and 17 effective boundaries (16 failed-history edges and the one
reviewed preparation inventory). The 41 combined resolver/producer tests and
four independent before-parse guard tests passed. Final source snapshots,
boundary policy, full resolution and verification receipts are retained.
No production protocol, prediction seal, GPU work or validation-label read
occurred in this review.
