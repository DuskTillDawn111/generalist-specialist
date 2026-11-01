# Affine metadata for the paper-inspired baseline

`tools/predictor_validation_20261027/operator_observer_v2.py` adds opt-in,
request-scoped hooks around recognized leaf `Linear`, bitsandbytes linear and
GPTQ `QuantLinear` modules. It records actual input/output shapes and dtype,
direct parameter/buffer metadata, declared affine dimensions and the dense GEMM
arithmetic feature `2 × rows × in_features × out_features`. A separate worker,
`operator_worker_v2.py`, reuses the retained semantic worker and adds these hooks
within the actual causal-language owner. Loader, inference, generation policy,
cache reclamation and model weights are unchanged.

This closes part of the USHER feature-collection gap: input size, actual packed
weight representation and configuration-dependent operation size are available
for later explicitly bound operator-to-kernel regression. It is not a complete
operator graph or a trained prediction model. Functional/fused attention, vision
and ASR frontends, and standalone OCR are outside the present hook coverage.
LoRA parent wrappers are skipped when they have children, avoiding counting a
wrapper and its executed base/low-rank affine children twice.

Logical tensor bytes retain actual packed dtype/shape. They are not unique
weight storage bytes, dense dequantized size, DRAM traffic or an intermediate
allocation peak. The FLOPs feature excludes bias, quantization/dequantization
and backend overhead. No tensor contents are read and no CUDA synchronization
is added. Kernel duration/geometry must still come from genuine CUPTI records
correlated to the per-call NVTX range. Missing kernel or allocation evidence
cannot be filled by these features. Observation budgets and instrumentation
errors are explicit; truncated coverage cannot be used as complete graph data.

The extra per-module hooks add host overhead, so the opt-in worker has its own
execution identity. It belongs in isolated structural profiling and its matched
co-running study, with a separate retained semantic worker for natural endpoint
measurements. Neither a hook wall range nor a profiling duration is relabelled
as unprofiled service time.

CPU verification:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest tests.test_operator_observer_v2 -q
```

Six checks use actual CPU torch hook semantics: exact unchanged native outputs,
GEMM feature counts and shapes, packed INT32/FP16 metadata, wrapper exclusion,
budget/nested/exception cleanup, optional sink failure, and combined
language/operator observations. They do not establish GPU accuracy or speedup.
