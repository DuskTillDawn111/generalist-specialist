# Batch fold resource rescore companion v1

`twoFIFO_fold_resource_batch_rescore_v1.py` is an independent continuation wrapper around the frozen single-case resource companion. It accepts one completed generated whole-block report and one source-authenticated twelve-attempt resource model. It authenticates exactly one `FoldResourceTrainingProvider` for each of the three held-out blocks, caches those providers, and reuses the cached provider for all 50 cases in that fold. The single-case helper is not called inside the case loop, so native twelve-attempt loading and fold reconstruction do not repeat per case.

The batch path restores each compact prediction through the single-case full-world replay and verifies the archived full forecast hashes and compact envelope before scoring. It then calls the original `score_world` and six-term `whole_score` bridge with the cached provider. Every case writes an exclusive report containing its complete draw set; transport UNKNOWN worlds retain their original cohort, weights, and reason. No case labels are opened, no numerical fit runs, and no collector, controller, GPU, or waiter starts.

The completed report and resource model must carry the exact same campaign receipt. All twelve resource review receipts must occur in the completed report source closure and remain byte-stable. Each fold must contain exactly four heldout attempts and eight donor attempts. The batch checks the original case calendar, public cohort, measurement window, fixed drain, and folded-law digest for every case. Missing past memory evidence leaves M and full-U UNKNOWN; the batch never assumes a fresh state.

Past-memory receipts are keyed by `heldout_block_id|attempt_id|case_id`. A bare case_id is accepted only when it is globally unique in the completed report; if the same case_id appears in multiple attempts or blocks, a bare mapping is rejected and missing full keys leave each case UNKNOWN. This prevents causal history from being reused across attempts.

This is an explicitly authorized CPU continuation helper. Its output is a resource training hypothesis and does not qualify resource prediction, native accuracy, queue-provider eligibility, or final gain. The batch result requires 150 cases and 546 offered requests; it does not claim that any native labels were read.

```bash
python - <<'PY'
from tools.predictor_validation_20261027.twoFIFO_fold_resource_batch_rescore_v1 import run_all_completed_cases
run_all_completed_cases('/absolute/completed-report.json', '/absolute/resource-model.json',
                        output='/absolute/exclusive-batch-output')
PY
```
