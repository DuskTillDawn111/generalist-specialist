# Final deployment diversity and Generalist runtime evidence

Six complete workloads; distinct final vectors=4.
Actual served VQA requests are checked separately from model presence. Family timings below are diagnostics over only requests assigned to that family; input subsets may differ under routing, and co-deployment is not a single-Generalist replacement trial. Official acceptance still uses the unchanged whole-workload objective.

## Reasoning/VQA-heavy actual serving

|State|Model|Count|VQA λ|Observed requests (3 reps)|Family VQA mean ms|Family mean-of-P95 ms|Whole-workload mean ms|
|---|---|---:|---:|---:|---:|---:|---:|
|56c268d2b82c|phi4mm|1|1.297065|817|177.915|265.827|103.243|
|1797c0162e5d|qwen|1|0.681763|429|146.828|276.684|95.491|
|1797c0162e5d|phi4mm|1|0.615301|388|181.784|269.609|95.491|
|6a94c2d3b7a1|phi4mm|1|0.713374|450|179.325|269.146|107.236|
|6a94c2d3b7a1|minicpm|1|0.583691|367|193.186|305.237|107.236|
|c950629ea78e|qwen|1|0.599723|378|146.840|236.337|108.461|
|c950629ea78e|phi4mm|1|0.697342|439|182.528|269.318|108.461|
|d07311e68bd4|qwen|1|0.599723|378|148.529|243.717|107.515|
|d07311e68bd4|phi4mm|1|0.697342|439|180.811|267.174|107.515|

Actually observed VQA-serving Generalists in the visited reasoning phase: ['minicpm', 'phi4mm', 'qwen']. No unvisited singleton superiority is inferred.

## No-VQA phase

Final N_G=0, N_S=5; search stability=steady, independent confirmation stability=steady. Generalists remained available to the unchanged candidate generator; no N_G=0 constraint was imposed.

|Actually evaluated Generalist mutation|Mean E2E ms|Weighted task P95 ms|Stability|Decision|Reason|
|---|---:|---:|---|---|---|
|qwen x=0→1|25.747|44.963|steady|ROLLBACK|mean_worse|

These observations cover only visited states; no universal Generalist disadvantage or unvisited Phi/MiniCPM result is inferred.

## Interpretation

The frozen workload suite may lead to repeated final deployments. Those repetitions and objective rollbacks are experimental results, not a reason to retune λ. Per-task weighted contribution reconstruction matches every official state mean. Full phase/model observations, null values for unserved models, routing rates and raw hashes are retained in the accompanying JSON/CSV.
VQA/LID numeric quality thresholds remain UNSPECIFIED under the authorized semantic-only policy. This analysis adds neither quality measurements nor thresholds and does not affect admission or decisions.

## Memory surrogate diagnostic

Five phases contain at least one post-warmup idle observation beyond the historical 5% diagnostic tolerance; maximum deviation is 7.756%. All actual sampled device peaks remain within the frozen 49140 MiB hard budget (campaign maximum 40356 MiB). The tolerance is reported as a surrogate-accuracy diagnostic and is not silently converted into an acceptance gate. Frozen empirical memory tables and final decisions were not changed. Per-phase/all-run details are retained in memory_surrogate_diagnostic.json and memory_validation.json.
