# Memory history support and testable smaller-state hypotheses

The current resource provider deliberately requires an exact ordered state-prefix
hash before it transports a state memory observation. This is a safe UNKNOWN
gate, not an established sufficient description of allocator state. It omits
elapsed time, tensor dimensions, KV length, allocated/reserved bins and stream
release state, so even a matching hash does not certify physical equivalence.

The frozen v3 matrix rotates each public case across positions/shards. A CPU
enumeration of owner/task/input/relative-arrival prefixes, excluding all case,
request and execution IDs, yields 150 distinct prefixes for 150 case attempts,
with zero complete-prefix repetition across blocks. All twelve first-case
signatures also differ. Actual compressed state hashes can still coincide
because they omit arrival duration, but the design does not guarantee two-block
memory support. Generated decode counts and owner interleavings introduce
additional branching. Once a prefix differs, its descendants cannot match.

This does not justify removing history. Three smaller-state hypotheses can be
tested without changing the original memory safety factor or utility formula.

1. **Warm, quiescent boundary as a renewal state.** Require all offered requests
   have actual processed READY, no live tensors or KV references, no queued
   send/worker work, and the same resident/checkpoint/runtime/priming identity.
   Validate across deliberately different earlier case orders that each owner's
   live allocation, reservation and allocator free-bin state agree or no longer
   predict the next case's observed peak. A quiet NVML sample alone cannot prove
   this. No cache-empty/reset action is implied; such an action would be a new
   execution mechanism and require separate measurement and startup accounting.

2. **Allocator sufficient state.** Candidate state contains per-owner current
   live allocated bytes, reserved bytes, free-block size histogram/segments,
   pinned workspaces or graph pools, active shape/KV sequence and pending stream
   releases, together with the cumulative prior owned sampled peak. A generated
   resource world must evolve these fields; realized heldout state cannot be
   supplied as prospective input. If the backend does not expose an axis, that
   axis stays UNKNOWN. Current NVML totals and semantic BEGIN markers do not
   identify this state.

3. **Observed per-owner high-water shape envelope.** Under unchanged priming,
   test whether all subsequent exact public inputs, prefill/decode lengths,
   batch/concurrency combinations and workspaces lie inside an already warmed
   envelope and whether retained memory reaches a reproducible plateau. Use
   actual start/READY overlap, per-owner allocation/reservation evidence and
   whole-block leaveout to test the hypothesis. Do not assume three largest
   single requests dominate concurrent or fragmented allocations. Prior sampled
   peak must remain in the cumulative maximum. An empirical maximum is still
   not a physical capacity bound.

For each candidate, compare identical next-case public work after different
prior histories, retain every case and unknown sample, and test whether the
omitted history still predicts residual memory error or an underprediction.
Evaluate the same candidate state in all three whole-block folds; unsupported
states remain UNKNOWN. With only three blocks, failure to detect a residual
effect is weak evidence and cannot establish a universal Markov property.

The immediate CPU deliverable should be coverage: current exact-history support
fraction per case and whole-block fold, plus each proposed state collision's
within-state memory spread across different histories. This is diagnostic work.
Promoting a smaller state requires independent measurements of the missing
allocator/release axes or a separately verified renewal/envelope condition.
The existing power/energy integration and original E/M/P scoring remain intact.
