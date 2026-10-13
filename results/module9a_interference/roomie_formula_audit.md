Status: **FORMULA AUDIT ONLY; PAIR PREDICTOR DEFERRED BY USER**.

# Roomie formula audit

Primary source: [Roomie v1, Sections 2.3 and 3.1–3.3](https://arxiv.org/html/2607.16784v1), reviewed before predictor implementation; [PDF](https://arxiv.org/pdf/2607.16784). Public paper and author/title repository search did not expose reproducible per-kernel isolated-duration reconstruction code. This is an availability finding, not proof that no implementation exists.

Equations: `Wb=ceil(threads/32)`, `o=Wb*b/W` (1); adjusted occupancy recomputes `b` from remaining resources; `d_e=d*o/o_tilde` (3); two phases finish the shorter kernel first and restore the survivor's isolated rate (4); `mu=(T_tilde-T)/T` (5); representative delay is the median over feasible alignments; `gamma=max(q_p/q_q,1)`.

Algorithm 1 instead divides performance drop by `T_tilde`. Its delay shortcut uses `d_i*o_i/(o_i+o_j)` and increments both kernel indices, unlike the duration-driven completion narrative. Eq.(4)'s printed survivor expression uses a self-referential `d_tilde`; the explanatory text specifies `d_e` for that term.

## Pre-audit design notes — deferred, not executable decisions

- Use formal Eq.(5), with denominator **T**, as requested; never silently mix the two definitions.
- Use event-driven completion and residual isolated work. For rates `v_i=o_tilde_i/o_i`, the next event is `min(work_i/v_i, work_j/v_j)`; subtract progressed work; advance only completed kernels; after the other sequence ends, finish residual work at rate 1. This implements the requested two-phase interpretation rather than the pseudocode shortcut.
- Device limits include maximum blocks/SM, warps/SM, threads/block, registers/SM, shared memory/SM, and launch geometry. Isolated blocks use the already installed NVIDIA occupancy calculator (allocation granularity included). Unknown function limits/barriers/carveout remain disclosed assumptions. Grid residency is separately capped by available grid blocks/SM for interference demand; theoretical occupancy stays the official unbounded-grid launch metric.
- Runtime applicability is decided in runtime_sharing_semantics_audit.md. Cross-context simultaneous SM co-residency is unavailable in this runtime. An initial FCFS-like kernel-boundary proposal has been withdrawn: driver time slicing and compute preemption do not establish FCFS or kernel-completion switching.
- No scheduler policy is selected. Context switch quantum, preemption frequency, and fairness are unobserved. No further pair predictor implementation or execution is authorized until the user decides after the sharing audit.
- A feasible starting alignment is a pair of launch boundaries. There are `q_p*q_q` possible starts in this linear-trace approximation. Enumerate fully up to 64 starts, otherwise use deterministic hash-seeded sampling without replacement; simulate each sampled suffix. This bounded offline approximation is an engineering adaptation. No alignment is chosen using measured slowdown. Report its sampling uncertainty and a 32/64/128 sensitivity.
- Use `gamma` exactly as published, even for autoregressive traces where it can amplify errors. Same-model replicas never enter these pairs.
- Profiler timing is only a relative temporal shape. Reconstruct each request to the frozen unprofiled path s0 using `d_corrected=s0*d_profile/sum(d_profile)`, adjusting the final entry for floating-point roundoff. This **PROJECT_ENGINEERING_ADAPTATION** allocates CPU/launch gaps proportionally too; it is not isolated device-kernel timing. Audit this limitation and raw/corrected pair ranking, without serving-outcome tuning.
