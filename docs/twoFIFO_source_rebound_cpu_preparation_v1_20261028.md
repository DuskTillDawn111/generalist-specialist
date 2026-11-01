# TwoFIFO source rebound CPU preparation

The original `predictor-twoFIFO-owned-training-design-v3-20261028-cpu01`
campaign remains immutable evidence. Its continuation stopped at the source
integrity HOLD because the working tree had changed after the campaign freeze.
This preparation creates a separate campaign directory and does not retry or
edit any old plan, raw capture, review, manifest, or report.

The new CPU-only entry point is
`tools/predictor_validation_20261027/twoFIFO_source_rebound_prepare_v1.py`.
It copies the complete old source-freeze tree, replaces the two drifted files
with the current bytes, and rebinds every copied plan to a new context and
execution identity. It has only `prepare` and `audit` commands; there is no
collection, fit, or GPU execution path.

The current source receipts are:

| source | SHA256 |
|---|---|
| `gsdeploy/milp.py` | `f53b4883513a19c5fdfa4132400e43b58345e1b1aad279896224ecc8951baee1` |
| `gsdeploy/optimizer.py` | `dc1e2a15671579827e59983709d0b138397ec79c97a94017951d872def40d45a` |

The final source-bound prepared manifest is
`artifacts/predictor-twoFIFO-source-rebound-v1-20261028-cpu02/campaign.json`.
Its status is `CPU_PREPARED_SOURCE_REBOUND_NOT_COLLECTED`, with
`GPU_started=false`, `numerical_fit_run=false`, and
`validation_labels_read=false`. The source rebound output has twelve fresh
plan receipts and a new context receipt. No GPU experiment was started.

Validation performed:

* `python -m py_compile tools/predictor_validation_20261027/twoFIFO_source_rebound_prepare_v1.py`
* source-rebound `audit` during preparation
* `python -m unittest tests.test_milp tests.test_search_contract` (23 tests)

The new campaign is still only a preparation artifact. Any future collection
must be separately reviewed and explicitly authorized by the fixed campaign
gate; the old continuation/coordinator must not consume this manifest.
The earlier `cpu01` preparation remains as a draft receipt from before the
audit gained its immutable-workload comparison. It is not the final source-
bound preparation and is preserved without edits.
