# Lightweight scheduling pressure model

The earlier Runtime Sharing Semantics Audit rules out Roomie spatial occupancy simulation for this independent-process, independent-CUDA-context, no-MPS runtime. All 21 offline profiles are preserved. Only target kernel counts are consumed. No occupancy, kernel duration, cache, frequency, target susceptibility or neural-network features enter this version.

iGniter Eq. (5) decomposes scheduling delay as kernel count times baseline plus extra scheduling delay; its original experiments use MPS and its round-robin discussion is a conjecture. This implementation borrows the decomposition, not iGniter spatial provisioning or a verified scheduler policy. Source: https://arxiv.org/html/2211.01713v1 (Sec. III-A, Eq. 5-6).

Project temporal-sharing adaptation: T_hat[p<-q] = T_p + n_p delta_q. The coefficient belongs to the co-resident model-task path q. delta_q = median_p ((T_pair[p<-q] - T_p)/n_p), with calibration targets only. It is an empirical residual pressure proxy, not an isolated context-switch duration; CPU effects, clocks and temporal scheduling can contribute to the observed residual. No target-specific fitted tax is used. Each co-path has two calibration target paths and two disjoint validation target paths. No historical mixed-task measurements are relabeled as pure co-path data.

Model-level tax pools calibration residuals by co-model; task-path-level tax pools by co-path. Negative estimates are preserved and reported, not silently refitted or clipped. Inference outputs are checked for positive service times. New matched standalone measurements supply the T_p anchor for measured target paths; other admitted paths retain the frozen unprofiled s0. The small profiling sample remains a kernel-count sampling limitation.

Only TASK-PATH / ROUTING-CONDITIONED EXTENSION wording is claimed. Shared target/co-paths limit bootstrap independence; validation is a finite-domain migration test. The routing busy-probability/additive extrapolation is not yet multi-model GPU ground truth.
