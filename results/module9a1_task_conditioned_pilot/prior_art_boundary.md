# Module 9A-0 — Prior-Art Boundary Freeze

Status: **HYPOTHESIZED GAP**. This development pilot makes no first/novel claim and does not implement a complete v3 method.

## Established primitives and boundaries

- **Roomie** (Faye et al., arXiv:2607.16784v1, 18 July 2026): kernel-level resource profiling, occupancy-based analytical interference, separation of profiling metadata from uninstrumented service timing, and pairwise approximation are borrowed prior-art ideas. Roomie's contribution includes kernel alignment and placement; those mechanisms cannot be claimed as ours. This pilot measures task-conditioned signatures and real pair slowdown; it does not reproduce Roomie's full predictor or assert prediction accuracy. [Primary paper](https://arxiv.org/html/2607.16784v1).
- **USHER** (OSDI 2024): kernel-based resource estimation and interference-aware scheduling jointly address placement, batching, replication, and GPU resource use. Interference-aware placement/resource modeling is established prior art. [USENIX paper page](https://www.usenix.org/conference/osdi24/presentation/shubha).
- **iGniter** (Xu et al., arXiv:2211.01713; TPDS 2023): an interference-aware inference performance model guides resource provisioning and adaptive batching. Interference-aware resource prediction/provisioning is established prior art. [Primary paper](https://arxiv.org/abs/2211.01713).
- **Cascadia** (ICLR 2026 cascade-serving paper, not the unrelated AIPC project): system deployment/resource allocation and request routing are coupled in a two-level optimization. [Conference proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b76105b94a8c1286c860ad885ec588f1-Abstract-Conference.html).
- **RouterWise** (Kasnavieh et al., arXiv:2604.10907, 13 April 2026): resource allocation, request-induced load, routing fractions, and setup-specific latency are jointly modeled. Resource/deployment plus routing coupling is established prior art. [Primary paper](https://arxiv.org/abs/2604.10907).

## Hypothesis to test, not a novelty finding

`routing -> Generalist task mix -> model-task execution/kernel mix -> co-residency interference -> effective latency -> routing/deployment`

The narrow question is whether a fixed Generalist and fixed co-model exhibit stable differences in interference across every active, admitted task path drawn mechanically from frozen development evidence, and whether those paths also differ structurally at kernel level. Detecting this phenomenon would support further investigation; it would not establish that prior work never studies workload-conditioned interference, nor establish novelty of the complete feedback chain. The pilot does not revise routing, ranking, deployment selection, workloads, budgets, quality, or the candidate pool.

## Evidence boundary

Module 8F-2 is permanently frozen and used only as development/failure evidence. Its deployments, routings, workloads, labels, candidate histories and source hashes are immutable. **8F-2 is ineligible for any future v3 held-out final validation.** Future validation must have an independent, pre-runtime freeze and evidence boundary.

## Profiler/timing boundary

Kernel durations observed under instrumentation are diagnostics only. Real slowdown uses new unprofiled standalone/pair measurements with matched protocols; the existing formal s0 catalog is not replaced or refit. Hardware counter access may be unavailable; absent metrics remain absent. NVIDIA's standard theoretical occupancy calculator may use traced launch-resource metadata, without fitting an aggregate SM/DRAM/L2 interference equation. [Nsight Compute profiling documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
