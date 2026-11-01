# Whole-block resource-provider adapter

`twoFIFO_resource_fold_provider_v1.FoldResourceTrainingProvider` supplies the
resource side of whole-block queue evaluation. It leaves the existing v3
provider and all scoring formulas unchanged. Its constructor takes the same
model, target plan/context, hypothesis opt-in and mean/max scenario as the v3
provider, plus the required keyword `heldout_block_id`.

Before selection, it verifies source receipts, the clean v3 campaign and all
twelve complete native attempts through the original loader. It rederives the
full resource model and every one of the three saved whole-block folds, including
the folds that were not selected. Any missing, duplicated or altered fold fails
closed. The selected model must contain exactly eight attempts in two complete
donor blocks and must match the target physical worker/configuration/priming
identity. Only that selected model is installed in `provider.document` for
power, memory and startup prediction.

The original model retains per-block startup energy, while its per-block startup
memory values are not separately serialized. They remain identifiable through
the mandatory native table rederivation. Every fold's startup energy and
premeasurement sampled-memory peak are recomputed from its eight donor tables;
the held-out startup values cannot enter them. Missing donor startup values
propagate UNKNOWN through the existing aggregation.

`predict_timeline` delegates to the original v3 timeline integration and attaches
an explicit fold binding with the full audited model receipt, selected donor
IDs, held-out IDs, selected model digest and source receipts. The full model
receipt can change if held-out training observations change; selected resource
quantities and the selected-model digest cannot change from those labels.
The existing explicit timeline/physical binding and memory-initial-state
contract apply. A random seed does not establish a fresh allocator history.
Without explicit initial memory history, memory stays UNKNOWN. An unsupported
state or ordered history remains UNKNOWN even if the held-out block contains a
matching memory observation.

This adapter does not qualify empirical resource-state pooling, physical memory
sufficiency, changed-schedule transfer, independent prediction accuracy or final
routing gains. It does not implement the separate memory-envelope candidate.
The original startup amortization and memory safety factor remain unapplied
here, for their existing downstream boundaries to apply once.

Five CPU tests use an explicit synthetic replacement at the native-loader
boundary; no complete native campaign is currently available. They check
held-out power/memory/startup poisoning, nonselected-fold/full-model tampering,
physical mismatch, unsupported donor states/history, missing startup, explicit
initial history and the real incomplete-intake gate before fitting. These are
implementation contracts, not native performance validation.
