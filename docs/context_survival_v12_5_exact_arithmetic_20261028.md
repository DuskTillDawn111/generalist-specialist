# Context survival v12_5 exact arithmetic

This CPU estimator retains `Fraction` risk, hazard, conditional survival, event mass,
and restricted area through calculation. It converts mass, tail, and mean to JSON
floats only at output. Median and complete-finite-distribution decisions use exact
values. For two captures each with ten envelopes, five SAVE events at ages 1–5
and five right-censored envelopes at age 10, the median is 5: cumulative survival
is exactly one half. v12_4 incorrectly returned no median after floating point
accumulation produced 0.5000000000000001.

Newly derived v12_5 episode models use the v12_5 schema and bind the new extractor
source receipt. Existing immutable v12_4 episode models can be reused explicitly:

```python
from tools.predictor_validation_20261027.context_survival_v12_5 import ContextEnvelopeSurvival
provider = ContextEnvelopeSurvival(
    original_v12_4_model,
    allow_unvalidated_survival_hypothesis=True,
    allow_v12_4_source_model=True,
)
```

This path always invokes the original frozen v12_4 extractor on all original
training sources and compares the complete regenerated object with the supplied
model. `allow_cpu_fixture=True` is rejected on this reuse path. The original
`model_sha256` continues to identify episode evidence; query output separately
identifies `source_episode_model_schema`, `survival_computation_schema`, and the
v12_5 computation source receipt. Reusing episode evidence does not reuse the
v12_4 numerical algorithm.

Extraction retains v12_4 semantics: every resident context is observed through
each job's minimum-to-maximum measured-phase window, including owners without a
request in that job. Original RESTORE age is retained, actual SAVE is the only
failure endpoint, and window/frontier cuts censor. Empty pending frontiers do not
supply query support. Unresolved source envelopes are excluded. These are
training observations under unvalidated conditional exchangeability and
noninformative censoring assumptions. They establish no independent accuracy,
SM service quantum, or next-owner policy.

The independent CPU tests compare 32 actual cells with separate direct risk-set
Fraction arithmetic, check all original SQLite envelope and coverage bindings,
sample 128 frontier endpoints, retain four earlier statistical counterexamples,
and test exact-half, strictly-above-half, and explicit legacy-source reuse.
The strictly-above-half numerical fixture injects a rational hazard to avoid
allocating an impractical number of synthetic envelopes.
