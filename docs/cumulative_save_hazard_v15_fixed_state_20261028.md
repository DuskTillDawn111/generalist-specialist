# Cumulative SAVE hazard v15: constant-size numeric state, complete external audit

The v14 full-cohort preflight exposed quadratic CPU overhead: every visible host
refresh deep-copied and hashed a growing `empty_intervals` list inside each sealed
restore state. v15 retains the frozen v14 hazard arithmetic and moves that list
to an append-only external audit. This changes CPU cost, not the predicted SAVE
budget, rates, review clocks or conditional-entry semantics.

Implementation: `tools/predictor_validation_20261027/cumulative_save_hazard_v15.py`.
Frozen SHA256: `9bef900885a350abb3ea563ac46336ffff123619867dd504270a439f8a1a26f6`.
The v14 implementation is unchanged, SHA256
`1c35b3f522c4e2565ad4e5b625f0590d2966ed0d7270a856e4d6a22d48bddfd8`.

The sealed v15 state contains `empty_audit_summary` with exactly four fields:
`count`, `total_ns`, `hashchain`, and `last_interval`. Each actual empty segment
remains a separate external event; none are merged, deleted or reset. Events
carry global offsets, scope IDs, restore keys, original RESTORE/observation clocks,
per-state interval chains and a global append-only chain. The ledger has no role
in the hazard calculation. State field count is constant; integer text length can
grow logarithmically.

Before each forecast, call `cursor = controller.begin_audit_scope(scope_key)`.
At forecast completion, call `controller.export_audit_since(cursor)` and persist
the returned JSON with the forecast. Reusing the same scope key or restore key
does not mix candidates: every scope has a distinct ordinal/ID and a sealed
start cursor; starting a new scope closes the previous one. A state from a closed
scope is rejected. Export returns deep copies and preserves closed-scope bounds.
Export is linear in the number of exported events; exporting the full growing
scope on every update would reintroduce caller-side quadratic work.

v15 binds both v14 and v15 source receipts. Seeded uniforms deliberately retain
the original v14 budget-stream identity, preserving exactly the same declared
sample when solver provenance changes. Original RESTORE age, late conditional
observation entry, the one-draw budget, empty-frontier UNKNOWN/no-query behavior,
future frontier changes, crossed age bins, terminal SAVE and old-review retention
all use frozen v14 numerical methods.

`python -m unittest tests.test_cumulative_save_hazard_v15 -v` passed all 25 tests
in 0.280 seconds. These include all 17 inherited v14 contracts and new exact
old/new stepwise comparisons, seeded equality, full audit-chain verification,
repeated restore-key scope isolation, and export-mutation independence.

The CPU benchmark used the actual frozen training-only age-additive SAVE model,
the same holder/frontier and 1 ms conditional entry as the original parent probe.
Each empty segment uses two updates: one making the frontier empty and the next
restoring it. Timings exclude construction, equivalence comparison and export;
v14/v15 calls were interleaved. The original v14 probe is retained unchanged.

| Empty segments | Update calls | v14 seconds | v15 seconds | Speedup | v15 export seconds |
| --- | --- | --- | --- | --- | --- |
| 100 | 200 | 0.07284 | 0.02817 | 2.59 | 0.00082 |
| 300 | 600 | 0.53284 | 0.08565 | 6.22 | 0.00242 |
| 1000 | 2000 | 5.54334 | 0.30150 | 18.39 | 0.00822 |

All 2,803 comparisons, including initial states, matched every numerical result
and state field after removing only schema/provenance/audit-format fields. All
100/300/1000 historical intervals matched v14 individually. No budget was
redrawn. v15 state JSON stayed 1,225/1,225/1,226 bytes; v14 grew to
7,936/21,936/70,935 bytes. Separate 1000-segment cProfile runs reduced total time
from 19.069 to 0.689 seconds, total calls from 83,459,983 to 2,121,843, and
`deepcopy` cumulative time from 17.540 to 0.384 seconds.

Reproduction:

```sh
python -m tools.predictor_validation_20261027.cumulative_history_cost_v15 \
  --model artifacts/predictor-save-hazard-v13-20261028-training01/age_additive-model.json \
  --output artifacts/predictor-cumulative-hazard-history-cost-v15-20261028-cpu02
```

Evidence is frozen under
`artifacts/predictor-cumulative-hazard-history-cost-v15-20261028-cpu01/`:
`report.json`, complete `audit-*.json`, raw `v14-1000.prof`/`v15-1000.prof`,
`tests.log`, and `verification.json`. No GPU was started. These results establish
numerical equivalence and reduced CPU history cost; they do not establish
independent prediction accuracy, probability calibration, or a physical kernel
availability gate. The previously observed cross-envelope kernel remains a
counterexample to the strict physical interpretation.
