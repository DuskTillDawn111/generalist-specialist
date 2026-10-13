Status: **DEFERRED DESIGN SKETCH**, written before user steering; no active evaluator or predictor is delivered at this checkpoint.

# Pure routing-conditioned evaluator

## BORROWED

Offline per-kernel launch signatures, occupancy/resource constraints, alignment simulation, median delays, kernel-count overlap scaling, and additive pairwise approximation follow Roomie's modeling structure. See [formula audit](roomie_formula_audit.md) for discrepancies and engineering adaptations; this project does not reproduce Roomie's placement algorithm.

## OUR EXTENSION

All admitted `p=(m,k)` have separate traces. `routing[k,m]` is requests/second, consistent with the frozen project; workload is `lambda[k]` in requests/second and serves as a conservation check. Service times are seconds internally, milliseconds in external result files/API.

`D_m^0=sum_k r[k,m]*s0[k,m]` (worker-seconds/second).
`rho_m^0=D_m^0/g_m(x_m)`; absent models must have zero routed demand.
`pi[k|m]=r[k,m]*s0[k,m]/D_m^0`, or zero at zero demand.
`a_m=min(rho_m^0,1)`.
`Delta[p<-j]=a_j*sum_l pi[l|j]*Delta_pair[p<-(j,l)]`.
`s_eff[p]=s0[p]+sum_{j!=m} Delta[p<-j]`.

The busy probability approximation and service-demand task mixture are our routing-conditioned extension, not Roomie's original formula. Frozen empirical `g_m(x_m)` handles replica scaling: **SELF_INTERFERENCE_HANDLED_BY_G=true**. Additional co-replica counts do not multiply the pair delay again. This approximation is separately flagged when mixed-task replica transfer or 3+ models are evaluated.

The evaluator checks admitted paths, finite nonnegative rates, resident-model routes, frozen replica bounds, workload conservation, and pair-table coverage. It emits effective service times, slowdown factors, task mixture, base utilization, pair contributions, overload flags, and confidence metadata. Low confidence supplies information and never permanently prunes a candidate. It does not mutate routing or optimize a fixed point; Module 9B remains outside scope.
