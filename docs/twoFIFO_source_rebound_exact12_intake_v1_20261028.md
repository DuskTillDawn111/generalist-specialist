# Source-rebound exact12 training intake v1

`twoFIFO_source_rebound_intake_v1.py` is a CPU-only provenance gate for the
new rebound campaign. It consumes an executor-audited continuation final, all
12 native review receipts, and all 12 source-bound macro receipts. It requires
three whole capture blocks with four shards each, unique attempt/execution IDs,
546 measured offered requests, and 216 separate priming rows.

Before accepting an attempt, the intake verifies the executor's final receipt,
rechecks the native review from raw/plan/execution/manifest, and rederives the
macro document from that review. It rejects old review/macro schemas, duplicate
or missing attempts, failed or incomplete reviews, changed source receipts,
retry claims, and any validation-label or fit flag. Failed evidence remains a
HOLD in the continuation and cannot be replaced by a new attempt.

The emitted schema is a complete training intake only. It does not fit a model,
qualify a root, certify native CUDA/SM service, or authorize queue-provider or
routing gain. The current downstream interface gap is explicit: the new macro
rows need an adapter into the phase12 fit row schema, then a source-bound
coordinator must perform the fit; independent validation labels remain absent.
