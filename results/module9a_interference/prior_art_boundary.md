# Prior-art boundary — frozen before validation

Allowed claim: **TASK-PATH / ROUTING-CONDITIONED EXTENSION**.

Borrowed from [Roomie](https://arxiv.org/html/2607.16784v1): per-kernel profiling, theoretical occupancy, adjusted occupancy, occupancy-based slowdown, temporal alignment, pairwise approximation, and robust median alignment aggregation.

Interference-aware resource/placement modeling is prior art in [USHER](https://www.usenix.org/conference/osdi24/presentation/shubha) and [iGniter](https://arxiv.org/abs/2211.01713).

Our scope replaces a model execution signature by a model-task-path signature and supplies `routing -> task-path mixture -> interference`. No claim of first interference-aware scheduler, first kernel interference model, or first joint deployment-routing is permitted.

Duration reconstruction, bounded alignment sampling, and cross-context scheduling assumptions below are disclosed engineering adaptations, not novel Roomie primitives. All 8F-2 states are permanently development/failure evidence; 9A-1/2 anchors are calibration evidence. No optimizer, routing iteration, candidate ranking, task benchmark, memory budget, model pool, or frozen method is changed.
