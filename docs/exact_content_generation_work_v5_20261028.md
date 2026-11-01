# Exact public-content generation work, 2026-10-28

`generation_work_v5.py` adds conditional generation-work scenarios for new
requests of a known public input. It retains the v4 source, models and global
empirical mixture reports unchanged. It does not claim generalization to
unseen input content and never falls back silently to the v4 family/task model.

The fit consumes the unchanged observations referenced by
`predictor-generation-work-fit-v4-20261028-final03/model.json`. Its grouping key
is the entire v4 key (native projection, family, task, protocol and complete
effective-generation digest), plus public input-content SHA and actual
tokenized input length. Every exact group needs at least two independent raw
captures. Capture weights are equal, then measurements within a capture receive
equal weight. Replicas and trial IDs do not count as new captures or inputs.
Non-autoregressive OCR remains unsupported.

The new artifact is
`artifacts/predictor-exact-content-generation-work-v5-20261028-final01/model.json`,
SHA256 `627db5e35dd1304ac30d870207f72f17725e0b23c2987cf75b2f6795e843e56c`.
Eight of nine observed exact profiles are supported:

| family/task | actual input tokens | independent captures | generated tokens/forwards | internal batch |
| --- | ---: | ---: | ---: | ---: |
| ChatQA text short | 414 | 4 | 5 | 1 |
| ChatQA text long | 4514 | 4 | 5 | 1 |
| ChatQA new text | 963 | 1 | UNKNOWN | UNKNOWN |
| G2 text short | 389 | 4 | 5 | 1 |
| G2 text long | 4665 | 4 | 5 | 1 |
| G2 image short | 100 | 2 | 4 | 3 |
| G2 image long | 629 | 2 | 3 | 3 |
| G2 ASR short | 79 | 2 | 21 | 1 |
| G2 ASR long | 248 | 2 | 55 | 1 |

Each supported profile currently has a single joint length/work outcome with
weight one. In particular, the short/long known ChatQA profiles do not include
the unrelated six-token new-input outcome. This does not turn deterministic
training observations into a proof of future determinism or accuracy.

The public hash is computed by the unchanged
`generation_work_v4.input_content_identity(public_catalog_row)`: text uses the
public context/question, and media uses the public content SHA. Sample aliases,
reference answers and future outputs do not determine this hash. Query input
tokens must be public preprocessing evidence, not copied from a future result.
Changed content, shape, protocol, policy or a group without two captures yields
UNKNOWN. Extra query fields, including `generated_tokens`, `worker_response`,
`raw_output` and final observed forwards, are rejected.

```python
provider = ExactContentGenerationWorkProvider(
    document, native_projection_sha256,
    allow_exact_content_hypothesis=True,
)
result = provider.scenarios({
    "family": "text", "task": "text",
    "native_execution_projection_sha256": native_projection_sha256,
    "protocol_sha256": public_protocol_sha256,
    "effective_generation": public_effective_generation,
    "input_tokens": public_tokenized_input_length,
    "input_content_sha256": input_content_identity(public_catalog_row),
    "request_id": new_request_id,
})
```

The result follows the v4 scenario output contract: joint selected length,
prefill/decode forward counts and internal batch, weight, remaining work,
scope and limitations. This API covers prediction before the new request
executes; it does not claim a causal prefix producer has been implemented.

The artifact binds its original v4 model receipt, original training-observation
receipt, implementation receipts and a recursive closure of 1,338 materialized
sources. Explicit historical source resolution uses existing byte-identical
snapshots; it does not rewrite historical reports. Provider construction
recomputes the fit from immutable observations, compares every fit field and
revalidates the complete closure, rejecting altered scenarios, nested evaluators
or training sources. Original full contexts stay distinct under the explicit
native-projection collector-invariance hypothesis.

Six focused CPU tests pass: conditional profiles avoid unrelated length mixing;
unknown/single-capture/changed-shape queries remain UNKNOWN; replicas do not
create capture support; public content hashes ignore aliases/answers; future
outputs/policy changes are rejected; and fit/closure mutations fail. A real
provider construction and nine-profile check also passes, with eight supported
profiles and the expected 963-token UNKNOWN:
`provider-load-review.json`, SHA256
`9975fe53fd76b93024b00c9af04b842384931a03805f7887d74bd524bcce181a`.
These checks are source/contract and known-input coverage diagnostics. They do
not establish held-out slowdown, queue/P95, ΔU ranking or deployment gains.
