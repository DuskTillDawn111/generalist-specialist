# Final benchmark planner — no experiments executed in Module 6A

Classification depends on the intended claim. There is no mandatory new experiment to preserve the scoped factual statements in the evidence matrix or to start a paper draft. There are concrete gaps before presenting a convincing final systems benchmark beyond a pilot. Existing data should be reused only when workload, hardware, model/source identity, routing and measurement protocol match; otherwise separate their evidence.

## MUST HAVE — for a final paper centered on the three proposed contributions

| Gap | Evidence that makes it necessary | Smallest useful resolution to plan | What it would establish |
| --- | --- | --- | --- |
| Static control with the feasible full five-model pool and canonical routing | 3C compares against a minimum-memory Qwen seed and bounded neighbor searches. It does not include a fixed full-pool control. The artifact audit found 16 historical Stage 2B full-pool attempts and 15 Module 5B full-pool attempts, with none in 3C or 2G. Stage 2B uses a different workload/protocol (a 20 s smoke example versus 3C's 210 s); 5B uses GPU UUID 011dad14 while 3C uses 3da0d2e4. These cannot silently replace a matched 3C control. | Reuse 5B full-pool/D0 comparisons as limited within-pilot evidence. Plan an identity-matched fixed full-pool control under the same frozen W1/W2/W3 and confirmation protocol if keeping the comparative static-search contribution. No workload or algorithm tuning. | Whether searching provides a useful latency/memory tradeoff relative to the obvious deploy-all-feasible baseline; otherwise limit claims to improvement over the specified memory seed. Do not claim LP search is globally superior. |
| Independent full cold/warm sequence repetitions | 5B has one end-to-end sequence. Three repetitions per state and two independent final confirmations estimate state performance, but do not independently repeat decisions, warm state inheritance or isolated accepted-path timing. | Plan independent complete sequences with fresh measurement seeds/cache namespaces and the same frozen policies/inputs. At least two additional sequences would reveal repeatability beyond the single pilot; this is not a power-derived sufficiency threshold. Report each sequence rather than treating phases as independent samples. | Whether the observed evaluation and switch reductions recur. Required if describing dynamic savings as reliable final-benchmark behavior beyond this single sequence. |
| Broader random seed coverage plus independently repeated proposed search | 3C has only two B1 seeds per workload, wins 4/6 under the frozen objective, and several strategies share exact cache evidence; close outcomes occur in W2/W3. | If retaining a general comparative LP-versus-random contribution, plan extra independent B1 seeds and repeated B2 decisions at the identical six-state budget; paired protocol blocks; retain all outcomes. An initial plan of three additional B1 seeds per workload expands coverage but does not justify significance on its own. | Distribution and sensitivity of LP/random performance; otherwise retain only the exact “4/6 trials” statement and treat it as limited evidence. |

These are justified control/repeatability gaps, not requests to start them now. No extra generic per-state repetitions are automatically required: 3C and 5B already have three search repetitions and two separate confirmation repetitions per final state. A fixed full-pool control should also report memory from the frozen catalog so that a latency-only comparison does not obscure the seed's different objective.

## NICE TO HAVE — only if the paper's scope needs them

| Item | Current coverage / reason | Scope |
| --- | --- | --- |
| CPU scalability wall-clock measurements | 3A contains 5/10/20/30-model candidate, LP and runtime-evaluation counts, but no full-search wall-clock series. Module 1's recorded MILP solve timings are not complete guided-search timings. | Plan CPU-only timing if claiming practical planning latency at larger pools. Counts already support non-enumeration; wall-clock is not mandatory for that narrower claim. |
| Additional workload intensities or tighter existing-model memory budgets | Current real core benchmark has three frozen rate vectors in one five-model pool; synthetic scaling is a single artificial family. | Broaden external validity only after the necessary controls and repetitions; do not tune workloads retrospectively. Not executed or authorized here. |
| A real LP/runtime disagreement case | 3C shows zero positive-LP/worse-runtime candidates; B2/B3 end states are identical. | Needed only to promote runtime feedback to an independent real-performance contribution. Otherwise leave it as a design safety mechanism. Do not hunt or construct a favorable benchmark after the fact. |
| Untimed cold endpoint alternative | 5B structural endpoint sensitivity is cold 3 switches / 1 load / 3 unloads vs warm 1 / 1 / 0, but alternative switch time is null. | Useful if discussing sensitivity to seed-reset policy; existing isolated-path timing is already valid when labeled precisely. |
| Service quality accounting from existing model task evaluations | Current claims concern serving latency and structural feasibility. | Reuse existing quality evidence if the paper discusses task accuracy; do not imply quality equivalence from capability coverage alone. |

## NOT NEEDED — for the current supported contribution scope

- More predictor training, predictor models or bias calibration: Module 2 is optional, and the proposed main method uses predictor=None.
- More routing optimization experiments: 4B already explains the evaluated default; no broad routing claim is made.
- Forecasting, change detection, prewarming, multi-replica or new algorithm modules.
- Reproducing 3B's 99.35% extreme-value claim.
- New GPU runs to validate the offline enumeration tripwire or synthetic count tables.
- Pooling confirmations into search statistics, treating cache hits as physical runs, or selecting a favorable subset to improve conclusions.

## Recommended next step

Start the paper's structure and methods/evidence draft now with the safe wording. Before a final submission asserting general comparative and dynamic performance, resolve the specific controls and independent-campaign gaps above in a separately authorized experiment phase. Prioritize the full-pool static control and independent cold/warm sequences; no additional algorithm development is indicated by this audit. Module 6A stops here.
