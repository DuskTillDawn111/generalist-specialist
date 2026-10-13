# Reasoning/VQA-heavy diagnosis

Seed Phi+S4 → Proposed Phi+Qwen+S4 → Random-2207 Qwen+S4; B2=Qwen+S4+Distil-Whisper.

+Qwen accepted: mean103.243→95.491ms (7.509%), steady; frozen 5% gate passes. After this correct ADD, -Phi is generated and coverage/memory/canonical LP feasible, but analytical rank=4, delta_z=-0.08891564427; Top-B=2 excludes it. Proposed never evaluates this REMOVE.

Rank1=-DistilBERT and rank2=-ResNet50 both worsen runtime and roll back; rank3=-Whisper-small and rank4=-Phi remain untested. Removal raises min-max pressure, which ranks it poorly, but removal also sends VQA entirely to the faster Qwen path; Random evaluates and accepts it.

Random-2207 -Phi: mean86.406ms, P95=147.757ms, steady, memory21516MiB; Proposed95.491ms/157.304ms/33288MiB. Independent confirmations also show improvement. Correct ADD without completing matching REMOVE is directly observed.

A SWAP (-Phi,+Qwen) requires two x±1 incumbent transitions. In this workload +Qwen already passes acceptance, so no intermediate acceptance barrier is observed for this SWAP. In balanced/vision/text the +Qwen intermediate rolls back; in speech after Distil+1 it becomes slower, a direct barrier. Do not generalize that barrier to reasoning.

Stopping is NO_RUNTIME_IMPROVEMENT after two iterations/5 states, with one state slot remaining. Without changing ranking, one more iteration would place -Whisper-small rank1 and -Phi rank2; only the former fits the sixth slot. A future diagnostic needs budget7 plus patience2 (max_iterations3 unchanged) to isolate this path; not executed here.

B2 is another step farther structurally (+Distil), but that ADD is absent while VQA dominates. Actual pressure-gated graph needs4 steps from Proposed final to exact B2 vector, including a Whisper-small detour. Feasible graph reachability does not prove runtime-accepted reachability.

Fixed-deployment routing crossover (n=3 per new state):

C canonical=95.491ms, D greedy-compatible=88.855ms; D-C=-6.636ms, P95 difference=-7.009ms. Phi remains physically deployed in D but receives no VQA; memory vector is unchanged. This is routing evidence at fixed Proposed deployment.

B versus D exact routing equality=True; B-D mean=-2.587ms, P95=-3.112ms. With equality true, this changes deployed Phi versus Distil-Whisper while preserving all task routing. B2 greedy A=85.983ms. Do not attribute the entire C-to-A improvement to deleting Phi.

D mean sample std=3.452ms, B std=2.016ms. These small, separately timed n=3 contrasts are descriptive, without an inferential significance or exact percentage-of-causality claim.
