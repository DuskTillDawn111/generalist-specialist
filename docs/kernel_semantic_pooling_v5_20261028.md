# Complete-launch baseline hypotheses and actual shape/state features

The new CPU tools preserve the old affine baseline and its evidence. They add
separate schemas for real non-affine launch geometry, explicit cross-layer
pooling, actual affine operator features and observed semantic execution state.
They produce executable numerical hypotheses while retaining failed fits,
missing observations and unsupported queries. None is deployment-qualified or
validated against independent natural request latency.

## Actual new-shape result

The actual additional training request uses the real963-token catalog input and
generates6tokens, whereas the earlier ChatQA requests generate5. Actual prefill
still contains3829kernels, and the complete new request contains23312kernels.
The template comparison finds all3220stable-grid slots and the bool-reduction
resource formula exact. Nearest/lower dispatch policies match3381/3829complete
signatures; upper matches3284/3829. All448nearest-policy mismatches belong to
explicitly unresolved algorithm domains:224split-K counts,96newAmpere128x32,
64newAmpere128x64and64CUTLASSalgorithm switches.

Receipts and exact differing signatures are retained in:

- `artifacts/predictor-chatqa-template-actual963-comparison-v4-20261028-final01/report.json`
  SHA256 `aa95cce3c4cc9ba1bdc9b27cd53020302f9348b42c502445420812770a8c38e2`.
- `artifacts/predictor-chatqa-template-actual963-pattern-review-v4-20261028-final01/report.json`
  SHA256 `c58aca96256c0109120dfa1b8e636b9f1054a14c38ff46b0493d096333b83d7b`.

This was extra training, and the collector changes the complete context. The
comparison therefore remains a cross-context diagnostic, not independent
same-context validation. The subsequent fit may use it only under the new
explicit native projection hypothesis below.

## Native execution projection

`native_execution_projection_v5.py` verifies all117retained native source
receipts, actual worker/task specifications and the weight-stat JSON receipt.
Its projection binds the complete worker bindings, residents, thread settings,
execution configuration, device declaration, measurement domain, profiler,
observer variant, weight identity scope and exact weight-stat content SHA.
It explicitly excludes the collector wrapper and telemetry policy. Original
full-context SHA values remain attached to every observation.

The real old two captures and963capture have identical light native projection
`948500ead7f304301c7e463ac3eaa24aae295d787d4f8102cbfd78bb25a86840`.
Their natural projection is separately
`b2687b3d82de7832678b3a410cb9f40cf7a67296cf6ac5fbf083a07ab59ca54a`.
The shared source digest is
`240a3e7b75d3da9b36ed250639430c6204ebd30cd9e9eb82ced1a1bd322f05ea`.
`artifacts/predictor-native-execution-projection-v5-20261028-final01/report.json`
records four distinct full contexts and two distinct native projections.
Projection equality is a declared collector-invariance hypothesis, not proof
of full-context equality. Natural and profiled durations never mix.

## Geometry-only pooling experiment

`kernel_launch_pooling_v4.py` fits all259090real launches in the original two
captures, including181302without the prior affine binding. Features are actual
CUDA grid.x/y/z and grid block count. Complete typed kernel name, non-grid launch
attributes, phase, family, configured precision, hardware, profiler flags and
canonical complete context remain exact grouping fields. Removing module name
and local kernel ordinal is an explicit cross-layer pooling hypothesis.

The final nonnegative regression has294groups:197pass the unchanged training
residual requirement,90fail and7lack observations. Minimum observations3,
minimum actual raw capture runs2and maximum per-observation relative error25%
are unchanged. The model never calls launch geometry FLOPs or memory bytes.

`artifacts/predictor-real-launch-pooling-fit-v4-20261028-final02/report.json`
has SHA256 `577cfb424c2b9d025f473c4cf28e4d8ed558ac97f9e79601d655bbe8887b4623`.
All3829numeric963prefill duration hypotheses are emitted under each dispatch
policy in `predictor-chatqa-pooled-duration-candidate-v4-20261028-M963-*-final02`.
One reduction resource change requires an explicit unvalidated source-conditional
transfer from an anchor model. Failed fits and geometry-hull violations remain
flagged. The sums131.76ms(lower/nearest) and254.07ms(upper) are kernel-duration
hypotheses, not request wall times; only264positions satisfy both the strict
fit and feature support conditions.

The diagnosis is concrete:123exact geometry feature points in38groups contain
31090observations whose maximum/minimum durations exceed5/3, including within
one actual capture. No deterministic model of those exact same features can
keep every relative error below25%, regardless of regression flexibility.
The proof follows from the non-overlap of `[.75*y,1.25*y]` intervals. See
`predictor-real-launch-pooling-diagnosis-v4-20261028-final01/report.json`.

## Actual shape and state model

`kernel_semantic_pooling_v5.py` uses the old two complete training bridges plus
the actual963training bridge. Every row retains its original raw capture and
full-context identity, hardware/profiler domain and exact source launch.
Native projection equality permits only the explicitly stated new pooling
hypothesis. Exact metadata/light bridge mappings attach real affine M/N/K,
named primary-weight bytes, input/output logical bytes and declared dense
operator FLOPs. These remain operator covariates and are not measured kernel
FLOPs. Non-affine rows receive no invented FLOPs or byte values.

Every row additionally has real geometry and request/semantic-phase launch
ordinal. Where actually observed, it has public input tokens, semantic input
sequence length, batch, forward index and cumulative declared input positions.
The latter is a clearly named proxy: actual KV tensor length is unknown. Missing
features are absent, with separate feature schemas; no zeros are imputed.
Typed kernel, actual tensor precision/implementation when available and all
static launch resources remain exact grouping fields.

The current source-bound result is
`artifacts/predictor-semantic-shape-kernel-fit-v5-20261028-final02/report.json`:
282402launches,88546affine-feature rows,193856non-affine rows and299groups.
205groups/181627rows meet the original25%training threshold;
82groups/100689rows fail;12groups/86rows lack enough observations or runs.
New963-only algorithms never borrow other algorithms' run counts. The numeric
models cover282316/282402training launches. Query acceptance additionally checks
the actual feature affine subspace and convex hull, never a min/max box.

Training aggregate diagnostics separately report each request, semantic forward,
and tiny(<=10microseconds)/larger kernels. Old text request kernel-sum errors are
2.77–13.69%;963covered sum error is16.97%, its covered prefill sum31.38% and
individual covered decode sums0.74–1.27%. The new request still lacks72numeric
targets, mainly64newAmpere128x64launches observed in only one capture. These are
in-sample diagnostics, not independent20%end-to-end qualification, and missing
kernel sums are not claimed complete. Kernel-duration sums do not include
natural host gaps, CUDA overlap or complete request wall time.

`kernel_semantic_candidate_v5.py` instantiates the model on new numeric prefill
templates. It requires exact constant/integer-linear public-token relations for
all existing operator/state features, verifies them against all retained
training shapes414,963,4514, and rejects unsupported relations. Actual future
KV length and generated token count remain separate scenarios. Dispatch and
source-conditional resource transfers remain explicit hypotheses.

The 1,025-token nearest-anchor candidate is recorded in
`artifacts/predictor-chatqa-semantic-duration-candidate-v5-20261028-M1025-nearest_anchor-final01/report.json`.
It has 3,829 numerical prefill hypotheses, a 151.096 ms kernel-duration sum and
only 67 positions that meet both strict fit and feature support requirements.
Overlapping failure counts are 3,116 residual failures, 586 points outside the
actual convex hull, 609 outside the feature affine subspace and one unvalidated
reduction resource transfer. These numerical results do not validate dispatch,
request wall time or unseen-input accuracy.

`kernel_semantic_query_v5.ExactSourceSemanticHypothesisResolver` exposes the
regression coefficients for retained exact-source launches. The resolver checks
the original SQLite receipt, process/context/correlation/interval identity and
static signature. Its index contains target-free feature rows; measured query
durations never provide fallback values. `numeric_hypothesis_available` permits
an explicitly selected numerical experiment to use a duration while `known`,
`strict_known` and the original failure reasons retain the strict result.
Every result remains `hypothesis_only`, with physical accuracy and deployment
qualification false. The CPU review in
`artifacts/predictor-semantic-query-resolver-v5-20261028-final01/report.json`
indexes all 282,402 source launches: 178,624/178,696 text, 63,304/63,312 generalist
and 40,388/40,394 OCR launches have numerical models. The remaining 86 launches
stay UNKNOWN because their groups lack the unchanged observation/run minimums.

A separately authorized experiment adds an explicit provisional estimand in
`kernel_two_capture_provisional_v5.py`, with artifact
`artifacts/predictor-two-capture-provisional-kernel-v5-20261028-final01/report.json`.
Seven previously insufficient groups each have exactly two training observations
from two distinct raw captures, one per capture. Their arithmetic group means
cover eight generalist and six OCR source launches. The three OCR target pairs
are [1,696, 1,728], [1,888, 1,984] and [5,792, 5,792] ns, giving provisional means
1,712, 1,936 and 5,792 ns. The artifact retains every exact training sample,
actual raw receipt, features and original rejected strict model. This is a new
provisional group-mean hypothesis with insufficient observations, without a
strict residual qualification claim.

`ExactSourceTwoCaptureProvisionalResolver(report_path,
allow_two_capture_provisional_hypothesis=True)` must be selected explicitly;
the default leaves these launches numerically UNKNOWN. The resolver delegates
all source/signature/identity checks to the original semantic resolver and
preserves its `prediction`, original rejection reasons, `known=False` and
`strict_known=False`. The mean uses actual training targets; the measured query
duration never supplies a value. Single-capture numerical hypotheses are
forbidden, leaving all 72 missing text launches UNKNOWN. Four contract tests
and the actual-source CPU review in
`artifacts/predictor-two-capture-provisional-query-review-v5-20261028-final01/report.json`
verify all 86 previously missing launches, default opt-out, exact identity
rejection, unchanged strict evidence and independence from query durations.
The original regression artifact is unchanged. No held-out, natural-timing or
deployment qualification follows from this optional numerical coverage.

## Bounded supplemental eligibility

The strict review is
`artifacts/predictor-semantic-shape-kernel-strict-review-v5-20261028-final01/report.json`.
For the actual 963-token request, one additional exact same-work capture could
make 64 new Ampere launches and five decode reductions eligible for fitting.
The three singleton launches require two additional exact repeats to reach
three observations. Eligibility does not guarantee the 25% residual criterion.
Existing 7,121 residual failures remain a separate problem, and eight exact
rich-feature points already contain mutually incompatible duration intervals
under that criterion. Repetition alone cannot resolve these omitted variables.
No supplemental capture was started by this review.

## Copy and memset numerical hypotheses

`copy_memset_baseline_v5.py` fits actual request-attributed CUPTI byte counts.
The source-bound fit is
`artifacts/predictor-copy-memset-baseline-v5-20261028-final03/report.json`
(SHA256 `ae0a6a7b0586aa2e2ce8a2f0399e81b292d4c7b2aa6bac3a41f21564f03b19a0`).
Attribution requires the owning process and the actual CUDA runtime issue
interval to lie within one request-stage NVTX range. Exact activity kind,
runtime API, actual direction/memory-kind enums, hardware, profiler, native
projection and family remain group fields. The sole regression covariate is
actual bytes; initialization or unmatched work cannot supply request training
observations. Nonnegative regression uses the same three-observation, two-run,
25% maximum relative training error and actual feature-hull requirements.

There are 28,301 observations from three raw capture runs and 15 groups. Six
groups/560 observations pass; nine groups/27,741 observations fail the residual
threshold. All 28,301 have numerical models. Copy-engine independence remains
a hypothesis; DRAM interference, natural timing and held-out accuracy are
unvalidated.

| Family/activity | Strict known | Numerical only, residual failed |
| --- | ---: | ---: |
| Text copy | 156 | 201 |
| Text memset | 0 | 3,872 |
| Generalist copy | 92 | 20,376 |
| Generalist memset | 0 | 668 |
| OCR copy | 156 | 2,624 |
| OCR memset | 156 | 0 |

`ExactSourceCopyMemsetHypothesisResolver(fit_report_path)` accepts
`resolver(dag_node, profile['source_sqlite'])`. It additionally requires exact
rowid, kind, process/device/context/stream, correlation, interval, API and memory
metadata. It preserves strict failures and qualification flags. The CPU review
`artifacts/predictor-copy-memset-query-review-v5-20261028-final01/report.json`
matches all 28,301 observations to actual `cuda_dependency_dag_v2` nodes across
nine immutable SQLite sources. Changing every query duration to an unrelated
value leaves every prediction unchanged; changing byte metadata is rejected.
Its query index contains no measured duration targets. Five copy resolver and
six semantic resolver contract tests pass. The six missing OCR numerical
kernel models are unaffected by completing copy/memset coverage.

## Minimal additional metadata design

The next observer should be a new metadata-domain-only wrapper, with no runtime
or frozen-worker edits. A scoped `TorchDispatchMode` around the actual request
can record ATen overload/schema, ordered input/output tensor shapes, strides,
storage offset, dtype, device, layout, alias/storage identity and relevant scalar
arguments. It must read metadata only: no `.item()`, `.cpu()`, tensor copies,
content hashing, explicit CUDA synchronization, profiler counters or model
weight reloading. Enter/exit NVTX markers must have unique request/operator IDs.
Nested redispatch and composite operations need an explicit stack so one CUDA
launch is not multiply attributed to overlapping ranges.

For normalization/elementwise work, retained exact tensor shapes determine numel
and dtype widths; reduction records require the actual axes/keepdim/input-output
shape; attention records require query/key/value shapes, strides, causal flag,
mask metadata and cache tensor sequence dimensions before/after each forward.
For opaque bitsandbytes/custom CUDA calls that bypass ATen dispatch, add a
metadata-only source wrapper at the public function boundary to record the same
actual arguments. Kernel workload features should be added only after a pinned
source launch equation connects those arguments to the observed launch. This
does not equate logical bytes with measured memory traffic.

The observer must run only in a separate metadata capture and bridge to an
unhooked light capture by exact full group order/signatures and exact work/output
checks. Added hooks may change launch paths; any mismatch leaves that transfer
unsupported. Preserve separate original contexts and new source receipts. The
new observer is a proposal only; no such capture or new global non-affine shape
evidence has been claimed. Existing traces already suffice for the CPU progress
above; no more GPU collection is started by these tools.
