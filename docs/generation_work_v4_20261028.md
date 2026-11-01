# Empirical generation work, 2026-10-28

`tools/predictor_validation_20261027/generation_work_v4.py` predicts a frozen
distribution of actual output lengths and language-forward work. It never uses
a future request's observed completion as a prediction input. The generation
policy maximum is a bound checked against training observations; it is not
substituted for their actual generated lengths.

The source-bound training artifact is
`artifacts/predictor-generation-work-fit-v4-20261028-final03/model.json`, SHA256
`e59812afbe4998591beb2dceab8d66d1c3e2ff8b39d4a891e6acf8356c8dad8d`.
It uses seven independent raw captures and 61 measured training requests.
Four OCR requests remain explicitly `UNKNOWN_NON_AUTOREGRESSIVE_OCR`; they
never enter an autoregressive length model. Priming requests and validation
requests never train this model. The old final01/final02 artifacts are retained.

| family/task | actual input contents | independent captures | frozen output/work scenarios |
| --- | ---: | ---: | --- |
| G2 ASR | 2 | 2 | 21 or 55 tokens/forwards, weight 1/2 each; batch 1 |
| G2 image | 2 | 2 | 3 or 4 tokens/forwards, weight 1/2 each; internal beam batch 3 |
| G2 text | 2 | 4 | 5 tokens/forwards, weight 1; batch 1 |
| ChatQA text | 3 | 5 | 5 tokens/forwards with weight 2/3; 6 with weight 1/3; batch 1 |

The six-token ChatQA outcome comes from one independent capture of the new
963-token input. Its outcome support records that fact; the scenario is not a
replicated or independently validated length prediction. At least two captures
and two distinct input contents are required at the family/task/policy group
level. Actual content hashes, rather than sample aliases, count inputs.

Weights are equal across input contents, then equal across captures for an
input, then equal across measured requests in that input/capture cell. Repeating
an identical replica within a capture neither creates another input/capture nor
changes the aggregate weight of that input. Different training inputs remain a
declared empirical population hypothesis, without a confidence or tail bound.

Each group binds the full effective generation configuration, protocol SHA,
family/task and native execution projection. The natural projection is
`b2687b3d82de7832678b3a410cb9f40cf7a67296cf6ac5fbf083a07ab59ca54a`.
The seven original full context identities and their immutable source receipts
remain separately recorded. Projection matching assumes collector invariance;
it does not claim equality of complete contexts or profiler/natural timing
equivalence. Provider construction verifies raw, plan, context, catalog,
evaluator, model implementation and native source receipts again.

`actual_work()` checks EOS membership, actual terminal token, actual generated
length and policy budget. It checks the complete observed forward sequence:
exactly one prefill followed by decode forwards, contiguous indices, observed
prefill/decode dimensions and the internal beam batch. Greedy work requires
one forward per actual generated token. Beam search may perform additional
steps after the selected sequence ends; that extra work is a separate joint
scenario field, never discarded or replaced by selected-token count. These
Python forward observations do not prove calibrated GPU execution phases or
correct KV cache reuse. ASR currently has only two distinct training durations,
so no conditional duration regression is fitted.

The provider interface is:

```python
provider = GenerationWorkProvider(
    model, native_projection_sha256,
    allow_empirical_length_hypothesis=True,
)
result = provider.scenarios({
    "family": "text", "task": "text",
    "native_execution_projection_sha256": native_projection_sha256,
    "protocol_sha256": public_protocol_sha256,
    "effective_generation": public_effective_generation,
    "input_tokens": public_input_tokens,
    "request_id": runtime_request_id,
})
```

Only those public fields are accepted. Adding `generated_tokens`,
`worker_response`, `raw_output`, `finish_reason`, or observed final forwards to
the request is rejected. Each scenario returns total prefill/decode work,
internal batch, weight and remaining forward counts. A future request whose
length is outside the frozen support is not extrapolated to the policy maximum.

Optional conditioning accepts an immutable receipt for a separate
`gsdeploy.observed_generation_prefix.v4` artifact plus a prediction time. The
prefix binds request ID, protocol, policy digest, native projection and
`host_perf_counter_ns`. Its exact keys are `schema`, `request_id`,
`protocol_sha256`, `native_execution_projection_sha256`,
`effective_generation_sha256`, `observation_time_ns`, `token_observations`,
`status` and `clock`; status must be `OBSERVED_PREFIX`. Every token carries only
`token_id` and `observed_time_ns`, in order and no later than the prefix time;
the prefix time cannot exceed prediction time. Final output fields are rejected.

For a greedy non-EOS prefix of length n, retain only empirical lengths greater
than n and renormalize their weights. No surviving scenario means UNKNOWN. An
already observed greedy EOS or exhausted budget has zero remaining forwards.
Beam-prefix conditioning remains UNKNOWN because an observed beam path does
not establish the final selected sequence or remaining beam search. Existing
collectors do not yet produce per-token prefix receipts: this interface requires
a genuine causal producer before runtime use; final worker responses are not
converted into prefix evidence.

Eight generation-work tests plus five native-projection tests pass. They cover
completion/forward inconsistencies, beam work, replica support and weights,
future-output rejection, prefix causality, unknown extrapolation, source
mutation, priming exclusion and OCR exclusion. A separate real-source provider
load review checks all four profiles after revalidating provenance:
`artifacts/predictor-generation-work-fit-v4-20261028-final03/provider-load-review.json`,
SHA256 `62e1eeffa512765f9648b546cf5483c378df05266abf74c0a0691d053b27d1f4`.
These are CPU contract checks, not held-out generation, slowdown, queue/P95,
ΔU/ranking or deployment validation. No GPU workload was started for this work.
