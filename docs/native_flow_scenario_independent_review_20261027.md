# Independent review of residual scenarios and whole-trace aggregation

This is a CPU-only interface/scientific review, not new empirical evidence. It does not change a frozen model, validation output, gate, runner, or capacity catalog. The reviewed implementation is `tools/predictor_validation_20261027/residual_scenarios.py`; the parent may subsequently change that module. Findings below refer to its current interface, not an asserted permanent defect.

## Interface findings and confirmed corrections

1. `flow_forensics.reconstruct_work` writes `complete_known_vector`; `fit_vectors` requires `known is True`. Direct passing previously made every vector unusable. The parent has added `adapt_reconstructed_vectors`, explicitly mapping the field while retaining source row contents, training role and unknown records and requiring a named reconstruction basis. The downstream fitter and planner continue to check positive finite values and phase completeness. This addresses the interface mismatch; source verification remains a sealer responsibility.
2. The scenario wrapper multiplies `plan.segments[*].duration_ms` and `service.service_ms`. The current statistical provider recomputes service from its stored profile, so the CPU rollout test correctly checks one factor. The generic `gsdeploy.forecasting.profiles.ProfileProvider.service`, however, derives service from the modified `row.segment_duration_ms`: wrapping that provider would apply the factor twice. The parent has now restricted the wrapper to exact `BackgroundStatisticalProvider` instances, closing that unsupported-provider path. Any future provider expansion needs an explicit scaling contract and a nonunit-factor rollout test.
3. The provider verifies the residual document self hash and point-model/context identity. The prospective sealer must additionally verify the complete transitive source receipts before use, and bind the wrapper source itself. A self hash alone does not certify training role or immutable reconstruction inputs.

## What these scenarios do and do not identify

One sampled vector is reused across phases of one request. This preserves empirical within-request phase dependence. Requests are sampled independently: run-level clock/thermal/allocator/host state, temporal correlation, and shared fluctuations among co-resident requests are not represented. Scheduler contention still occurs in the event simulator, but that does not recreate omitted shared residual dependence. Two independent training seeds remain two run replicates; their hundreds of requests do not establish hundreds of independent operating runs.

Populations keyed only by family/task/input pool combine routing, resource-share, task-mix, alias and active-state contexts. An uncentered residual may therefore carry omitted-context bias as well as random variability. Applying it as portable work under new contexts is an additional modeling assumption. Reconstructed work uses observed training phase boundaries and service contexts; it is a retrospective mixed-wall residual, not separately measured physical GPU work or a causal attribution of interference.

Cross-fitting is recommended in the function docstring but not enforced: `reconstruction_basis` can be unspecified. Residuals reconstructed using a point model fitted to the same flows can understate variability. Prefer leaving an entire seed out across all four routes when fitting reconstruction models; splitting individual requests leaks run conditions. With only two training seeds, report this limitation and do not claim calibrated P95 intervals or probabilities. A later hierarchical run effect plus contiguous residual blocks is plausible, but requires sufficient training runs and predeclared selection rules. Do not add arbitrary noise, post-hoc scale factors or favorable quantile choices to reverse the independent v9 failure.

Using the same request/scenario uniform across candidates is a reproducible Monte Carlo coupling. Mapping that uniform to differently ordered route populations is arbitrary unless a paired or canonical ordering is defined. It preserves each population's marginal distribution; it does not identify physical counterfactual covariance. Consequently scenario `P(ΔU > 0)` is a model diagnostic, not calibrated confidence. Preserve population ordering in the frozen model and document this coupling.

## Whole-trace target to freeze before a new holdout

Each scenario must simulate the entire identical declared arrival/input trace through FIFO, admission and co-resident phases. The scenario denominator includes every offered request, including requests completing in the declared drain. Incomplete/censored scenarios must remain explicit failures or unknown coverage; never discard them to improve a mean or P95.

Recommended point target is the expected operating-run outcome under the frozen empirical scenario model:

| Quantity | Frozen point definition |
| --- | --- |
| Per-request service/wait/E2E | Mean of that request's outcome across complete scenarios |
| Queue depth at a declared grid time | Mean depth across scenario traces at that time |
| Q | Frozen training quality proxy under the declared route/input distribution; independently score actual outputs |
| T | Mean of complete scenario T terms |
| L | Mean of complete scenario weighted mean E2E/reference terms |
| P | Mean of scenario weighted task P95/reference terms, with each task P95 computed over its request cohort within that scenario |
| M | Mean of each scenario's full-trace peak estimate, applying the original safety factor exactly once |
| E | Mean of each scenario's declared-window energy term, with startup/amortization only when prospectively supported |
| U | Mean of scenario six-term utility using unchanged weights and references |
| Paired ΔU | Mean of `U_candidate,s - U_baseline,s` over the same declared scenario indices |
| Candidate ordering | Descending frozen expected U, with a predeclared deterministic tie rule |

`mean_s(P95_requests(latency_s))` is different from `P95_requests(mean_s(latency_s))` and from a P95 pooling requests and scenarios. None is interchangeable. The recommendation above estimates expected run P95; compare it to the actual holdout run P95 descriptively. Do not substitute a phase quantile, per-request quantile, or risk quantile into it. If another estimand is selected, label and freeze it explicitly before measurement.

The linear six-term objective makes mean utility equal utility of the mean *terms*, including the correctly aggregated P term. It does not make utility of mean request latencies equivalent. Report absolute U as unknown when startup is unsupported; only a predeclared common-startup comparison may omit a common contribution. Actual paired differences retain measured startup differences.

Residual sampling can change a nonlinear queue expectation even when per-request means are similar. This is a new trained predictive model and must be evaluated on fresh data. It is not a guaranteed uncertainty correction. Independent units for the final comparison remain complete operating runs, not scenario samples.

The seal should bind model/resource/quality/source hashes, exact trace and candidates, all scenario indices and namespace, sample count, population ordering, reconstruction/cross-fitting method, point aggregation and task quantile convention, drain/censoring policy, utility references/weights, baseline pairing, error/ranking gates and all support/unknown counts. Freeze it before any new holdout startup. Resource predictions require the separate empirical power/memory provider; the service wrapper's delegated methods alone do not supply missing evidence.

## Acceptance boundary

Current CPU tests show deterministic joint phase sampling, supported-provider scaling once, input separation, role/duplicate rejection, unknown preservation and source tamper detection. They do not establish realistic co-residency dependence, calibrated tail prediction, physical work identification, transfer to another condition, or correct ΔU signs. Those require the fresh independent full-trace report and unchanged scientific gates.
