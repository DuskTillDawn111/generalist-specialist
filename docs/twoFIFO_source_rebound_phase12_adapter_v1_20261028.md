# Source-rebound phase12 row adapter v1

The adapter consumes the complete source-rebound exact12 intake and independently
rederives every native review and macro receipt. It preserves the new schemas and
never relabels a rebound raw capture as the old training schema.

It compiles all macro rows from the 546 measured requests into the twelve-feature
`own_family_phase_count_global_same` row representation, emits one full bundle
and three whole-capture-block folds (8 donor attempts versus 4 held-out attempts),
and retains held-out rows outside each training set. It reuses only frozen phase
vector, processed-peer, zero-point diagnostic and support helpers. It performs no
optimizer, numerical root fit, validation read, GPU call, candidate selection,
or USHER physical qualification.

The output is `ROWS_READY_FOR_SEPARATE_PHASE12_FIT_NOT_FIT`. A separate
source-bound fitter must consume the in-memory row bundle and apply the existing
root, support, nuisance, gradient/curvature and fold gates. This adapter cannot
be passed directly to the old fit CLI or used as queue-provider evidence.
