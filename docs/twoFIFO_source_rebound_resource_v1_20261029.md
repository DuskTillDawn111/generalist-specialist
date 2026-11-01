# Source-rebound resources and folded scoring

`twoFIFO_source_rebound_resource_v1.py` is a new adapter. It does not change the
frozen executor, phase12 fitter, generated runner, old resource loader or scorer.
New native raw/review/macro schemas remain intact. The adapter enters through
`phase12_fit.with_verified_bundle`: true exact12 intake and macro rederivation
finish before any resource table is built. It reuses only
`resource_provider_v3.table_from_verified`, `fit_tables` and `predict_timeline`
mathematics. It never relabels native data to pass the old resource loader.

`prepare(fit_manifest_path, output)` fixes the implementation closure, expected
continuation path and full design before the exact12 final exists and before a
response-fit claim exists. `audit(design_path)` verifies those immutable bindings.
`run(design_path, output, fit_enabled=False)` returns WAITING, HOLD or READY before
native intake/resource labels. Only explicit `--fit` after the exact complete
continuation invokes verified intake and fits the resource means. This resource
operation uses no response optimizer and starts no collection. It may be
rederived from the same immutable evidence; it is not a GPU retry.

The resulting `model.json` contains `resource_model`, `whole_block_leaveout`,
detached `intake_evidence`, exact source receipts and the 546 measured / 216
history-only priming denominator. Each of three folds trains on exactly two
whole blocks and eight attempts, excluding all four heldout attempts. Startup is
recomputed from those donors. Every state requires two independent donor blocks
for power/memory support. A heldout state cannot supply missing donor support.

```python
from tools.predictor_validation_20261027.twoFIFO_source_rebound_resource_v1 import (
    FoldResourceTrainingProvider,
)

provider = FoldResourceTrainingProvider(
    model_path, target_plan, target_context,
    heldout_block_id=target_plan['capture_id'],
    allow_training_resource_hypothesis=True,
    scenario='mean',
)
```

Construction rederives true exact12 evidence, the full resource model and all
three folds before selecting the fold. The target attempt must belong to the
selected heldout four, and its physical worker/configuration/priming identity
must match. `power`, `memory` and `startup` reuse the original provider methods,
so `actual_worker_world_score_v3.score_world` accepts this provider directly.
`fold_binding` retains the full native model receipt, selected-model digest,
heldout block and attempts, donor blocks and attempts, and implementation/source
receipts. A companion may cache one authenticated provider per heldout block.

Power integrates only each original measurement window. Drain power does not
enter E. Memory retains all resident owners and its exact ordered state history,
including prior cases. Missing or unseen history leaves M and full U UNKNOWN;
the adapter never resets a case to fresh priming. Startup energy remains
unamortized and memory remains without a safety factor. The original scorer
alone applies startup/3600 and its memory factor once. Priming and startup occur
once per attempt, rather than once per case.

CPU validation uses synthetic raw/reviews and real table/fold/timeline math;
native intake and row extraction are mocked at their explicit boundaries. It
also exercises the real coordinator order `intake -> adapter -> resource table`.
No native training fitting, GPU work or validation labels are used. Passing
these contracts does not qualify resource prediction accuracy, physical SM or
USHER service, full U, final routing gains or the main queue provider.

```bash
python -m unittest -v tests.test_twoFIFO_source_rebound_resource_v1 \
  tests.test_twoFIFO_resource_provider_v3 tests.test_twoFIFO_resource_fold_provider_v1
```
