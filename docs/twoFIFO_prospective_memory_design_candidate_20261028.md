# Prospective memory candidate: structure and identification gaps

The current memory candidate is filtered conditional replay. Its per-poll label
`max(0, current_sample - actual_prior_sampled_peak)` cannot be repeatedly added
inside a generated world: the result would depend on how often the world is
queried, and held-out actual past samples would silently correct prediction.
The following is a design proposal only; it does not modify the frozen provider.

1. **Generate structural states from the same request world.** Retain a sidecar
   keyed by generated public slot and macro identity that binds
   `joint_request_law_v4.vector.nodes.forward_shape` to the corresponding generated
   phase BEGIN mark. The existing compiler does not attach `forward_shape` to
   `Macro`; the sidecar must be explicit. Apply the shape/high-water update only
   when that generated notice is processed. Clear the owner on the generated
   original clear/READY lifecycle event as specified by the actual runtime.
   Do not update the state from future decode shapes merely because the complete
   latent donor vector is already available. Use all offered arrivals and the
   original two FIFO processes without an additional admission limit.

2. **Predict an absolute memory level, not repeated positive increments.** Fit an
   absolute sampled owned-memory envelope `D_j` for each supported structural
   state or whole structural interval. Draw the startup/priming peak once from
   compatible training evidence. Evolve the predicted attempt prefix peak as
   `P_(j+1) = max(P_j, D_j)`. Repeated queries or duplicated poll times leave this
   recurrence unchanged; a new structural event can only change the absolute
   demand term. Never replace `P_j` with an actual held-out prefix peak. Whole
   block/attempt sampling must retain joint startup, memory, generated-shape and
   timing dependence rather than independently drawing a memory label at each
   poll. A currently unobserved joint combination stays UNKNOWN.

3. **Keep observation uncertainty explicit in the training adapter.** A native
   poll brackets a query; a processed semantic notice does not identify device
   allocation or release. A poll crossing a state boundary cannot be assigned
   wholly to one state without justification. Keep ambiguous assignments and
   intervals with no eligible poll UNKNOWN. Interval duration and the sampling
   process affect sampled maxima, so an interval maximum cannot be transported
   to a different duration as a calibrated distribution without evidence.
   An absolute state sample envelope is simpler but still only an empirical
   hypothesis, never a bound on an unobserved between-poll peak.

4. **Test support before selecting a model.** With twelve complete native
   attempts, report all three whole-block folds, state support, ambiguous/empty
   intervals, every case's UNKNOWN, signed peak error and underestimation. Test
   that inserting extra query times, duplicating one state, and changing actual
   held-out future memory cannot change predicted peaks. Compare same compressed
   state reached after different histories. Do not select favorable folds or
   infer that a sampled plateau proves allocator renewal.

The unresolved identification gap is physical memory state. Current semantic
shapes, high-water values and NVML totals do not identify allocator live and
reserved bins, pending releases, KV residency or retained workspaces. The twelve
attempts may show repeated compressed states, but repetition alone cannot prove
that omitted history is irrelevant. A causal generated-state adapter plus the
max recurrence fixes future-label leakage and query-count artifacts; it does
not resolve this physical-state gap. Unsupported or non-identifiable memory must
remain UNKNOWN in the six-term utility, and the original memory safety factor
must be applied only once at its existing decision boundary after qualification.
