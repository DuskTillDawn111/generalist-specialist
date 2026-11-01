# Architecture-aware observed-launch occupancy declaration v2

The new offline tools `occupancy_declaration_v2.py` and
`occupancy_declaration_audit_v2.py` resolve the fields actually required by the
pinned CUDA 11.8 `cudaOccMaxActiveBlocksPerMultiprocessor` implementation on
CC8.9. They leave original function attributes and requested device state
separate from mathematically equivalent calculation representatives. No old
calculator/runtime/profiler source is changed, and no CUDA API, model load,
profiler/counter query, permission change, or GPU workload is executed.

The authoritative smoke result is
`artifacts/predictor-occupancy-declaration-audit-v2-20261028-final/audit.json`.
All four immutable SQLite sources match the retained hardware's exact unique
UUID, `GPU-533a287b-0e15-a7c8-8e58-2b21bdcbb13a`, through `TARGET_INFO_GPU`.
Source hashes match before and after the audit. Results:

| Owner | CUPTI launches | Static ceilings resolved | Unknown |
|---|---:|---:|---:|
| text#0 | 19,423 | 19,423 | 0 |
| text#1 | 19,423 | 19,423 | 0 |
| generalist_2#0 | 11,552 | 11,552 | 0 |
| ocr#0 | 5,493 | 5,493 | 0 |
| Total | 55,891 | 55,891 | 0 |

There are no zero-block contradictions for these completed launches. These
are static ceilings under the observed launch/resource state, not measured
achieved occupancy, cross-context coexistence, throughput, or training timing
accuracy. The smoke remains excluded from baseline fitting.

## Exact branch reductions

The occupancy header SHA is
`2abf47c8e7be865463040cdb20dc146243606db220223baf2acd3d51984c610a`.
The resolver rejects other headers or architectures until independently audited.
Its reductions follow the actual source branches, without constructor defaults:

- `partitionedGCConfig` is forced OFF by the header on8.9. Header support exists
  on5.2/5.3 and6.x except6.0, so this is not a universal GC simplification.
- `numBlockBarriers` affects the active-block result only on computeMajor>=9.
  Missing barriers on8.9 are not applicable, with a zero calculation representative.
- `maxDynamicSharedSizeBytes` is irrelevant in the observed DEFAULT shared-limit
  branch: the dynamic-limit test is disabled and DEFAULT ignores the maximum
  in its per-block-limit choice. Zero is an irrelevant representative, not a
  claim that the real function has zero dynamic-shared capacity.
- A source-bound completed exact CUPTI launch proves funcMaxThreads>=the actual
  block's thread count. The max-active-block header only checks funcMaxThreads>0
  and does not otherwise use it. The observed block count is therefore a
  `lower_bound_equivalent_occupancy` representative. It is not a measured
  function limit and does not authorize predicting arbitrary larger blocks.
- SQLite supplies the real DEFAULT/OPTIN mode, decoded using its enum table.
  It also supplies `sharedMemoryExecuted`, the driver's selected shared-memory
  capacity. The resolver finds every integer percentage that maps to that
  exact CC8.9 header capacity bucket, and chooses one as an equivalent input.
  The actual requested percentage remains unknown. Any observed capacity below
  the rounded per-block allocation or outside the pinned buckets is rejected.
- With that nonnegative effective-capacity projection, cacheConfig is not used
  by the header. The raw SQLite cache enum is preserved; no missing carveout
  setting is reconstructed from it. Underlying CUPTI cache fields can also be
  stale when an explicit carveout is requested.

## OPTIN maximum proof

Let `s` be observed static shared bytes, `d` observed dynamic shared bytes,
`r` the evidenced reserved bytes per block, and `m` the unmeasured real function
dynamic-shared maximum. The completed OPTIN launch proves `m>=d`.
The pinned `cuda.h` cuFuncSetAttribute contract bounds `s+m` by the evidenced
device opt-in limit, so the allowed interval is `[d,optin-s]`.

The header computes its rounded block allocation from `s+r+d`, independently of
`m`. Once `m>=d`, its only remaining use of `m` is selecting the per-block
DEFAULT/OPTIN limit at the threshold `m=default-s-r`. If `m=d` selects DEFAULT,
the rounded allocation already fits DEFAULT; choosing the larger OPTIN limit
cannot change the result. If `m=d` selects OPTIN, every allowed larger value
selects OPTIN too. The observed effective SM shared capacity already fixes the
per-SM resource calculation. Thus `m=d` is an observationally equivalent
representative for this completed launch, while the real maximum stays unknown.

The resolver checks the bounds/allocation conditions and pins the CUDA API
contract source SHA. It records the lower/upper bounds, threshold, branch-boundary
points, and exact completed-launch evidence in every derived OPTIN declaration.
Without that launch proof, the unknown maximum remains required. A value just
below `d` demonstrably changes a feasible result to zero and is excluded from
the proof; it cannot be treated as an alternative successful configuration.

The final artifact saves47 CPU sensitivity cases covering invariant fields,
all four cache settings under explicit carveout, both OPTIN threshold branches,
lower/upper endpoints, threshold-1/threshold/threshold+1, and ineligible controls.
Positive controls preserve the sensitive sm5.2 GC branch (21 versus20 blocks)
and sm9.0 barrier branch. Every case passes. These are synthetic formula tests,
clearly separate from the real smoke launch metadata.

## API and downstream scope

```python
read = read_sqlite_launches(sqlite_path, hardware)
row = read['rows'][0]
resolved = resolve_declaration(
    row['kernel'], hardware, row['function_attributes'], row['device_state'],
    calculator=calculator_build_receipt,
    effective_shared_bytes=row['effective_shared_bytes'],
    effective_shared_evidence=row['effective_shared_evidence'],
    completed_launch_evidence=row['completed_launch_evidence'],
)
patch = resolved['calculation']['kernel_patch']
```

Original real attributes remain in `original_function_attributes`; actual
funcMaxThreads/maxDynamic have not been measured. Calculation representatives,
their applicability and proofs are in `derivations`. The static patch includes
rounded registers/shared bytes and the header's subpartition-sensitive block
ceiling. `real_function_attributes_completed` and `requested_state_reconstructed`
remain false. Fields with actual supplied values require their original evidence.

The audit groups only identical occupancy-relevant observations, retaining
member SQLite row IDs and an exact representative launch. Consumers must verify
the source SQLite SHA, unique UUID mapping, row membership and matching declared
fields before applying a representative patch. The proof's scope is those
completed observed launches; it does not measure actual function maxima or
establish equivalent behavior for arbitrary future launch configurations.
Unsupported clusters/cooperative launches, mismatched identities, missing
effective capacity, or unproved relevant fields remain UNKNOWN.

## Correction to the earlier old-probe identity audit

The old tiny trace also contains the exact matching UUID in `TARGET_INFO_GPU`.
Earlier checks inspected only `TARGET_INFO_CUDA_DEVICE.uuid`, which is NULL;
the resulting claim that unique trace/device identity was unavailable was
incomplete. It is superseded by the new read-only correction artifact
`artifacts/predictor-occupancy-declaration-old-probe-correction-v2-20261028/audit.json`.
Old artifacts remain unchanged. This correction does not provide isolation,
operator features, kernel timing targets for the actual target models, or any
baseline training row.

Independent forecast-producer review found inconsistent paired objective
normalizers; the parent fixed them. The new independent tests verify rejection
for all five changed normalizers and changed paired arrivals, plus full-cohort
actual CPU simulator output. No further producer correctness finding remains
after the normalizer/source-closure fixes; no producer source was edited here.

Validation:12 new declaration tests,8 retained host tests and the focused
resource/provider/producer/gate suite pass (70 tests total). The declaration
tests and saved sensitivity cases make no target-model accuracy claim.
