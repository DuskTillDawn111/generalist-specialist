# Module 2 active profiling audit

Selected conditions: 6; saved new runs: 18/18; prior runs: 51; total runs: 69; task-model rows: 463.
**Rows are not independent deployment samples.** run_id/configuration_id/deployment_id/repetition_id are retained in rows and run manifest.

| Condition | Mutation | Expected category | Qwen u | seconds | distance to existing | observed categories |
| --- | --- | --- | ---: | ---: | ---: | --- |
| AS_01 | add distilbert | clearly_stable | 0.100 | 238 | 0.4239 | {'stable': 3} |
| AS_02 | add whisper | near_boundary | 0.250 | 120 | 0.3349 | {'stable': 3} |
| AS_03 | add paddleocr | likely_unstable | 0.350 | 120 | 0.3381 | {'stable': 3} |
| AS_04 | add resnet50 | clearly_stable | 0.100 | 300 | 0.3016 | {'stable': 3} |
| AS_05 | add distilbert | near_boundary | 0.250 | 293 | 0.3020 | {'stable': 3} |
| AS_06 | add paddleocr | likely_unstable | 0.350 | 180 | 0.2970 | {'stable': 3} |

Only 5 immediate mutations were considered; 60 bounded feature conditions, 54 not selected. No recursive expansion or exhaustive deployment enumeration.

## Audit answers

1. Feature coverage: mean distance of the finite candidate feature space to measured coverage fell 0.2717 → 0.1555 (42.8%); expanded ranges: ['mix/image', 'mix/ocr', 'pressure/kernel_fraction_conv', 'pressure/kernel_fraction_elemcopy', 'pressure/kernel_time_ms_per_request', 'pressure/offline_energy_j_per_request']. This is local feature coverage, not prediction accuracy or global coverage.
2. Observed new stable/boundary/unstable counts: {'stable': 18}; combined counts: {'stable': 61, 'boundary': 2, 'unstable': 6}. Expected bands were acquisition priors; labels come only from saved run evidence. No extra runs are added to force a desired label.
3. New deployment contexts: 2; deployment count 5 → 7. IDs only support auditing/grouping and are absent from acquisition vectors.
4. Grouped cross-deployment validation: 7 formal deployment contexts and 22 configuration groups support preparing an exploratory grouped holdout design. Unstable examples occur in 1 formal contexts; strict all-path stable-latency examples in 6. These counts do not yet establish reliable unified stability/latency/throughput training: context/load/mix may be confounded, and smoke protocols, short risk windows and input/seed variation require controls. No model was trained.

## Quality and limitations

Integrity passed: True; changed frozen sources: []; changed existing results: [].
New runs with <30 successful samples on any positive path: 6. P95 is descriptive for these paths, not a reliable tail estimate.
Latencies use successful requests only; incomplete/error records remain. Stable-region latency eligibility excludes warning/unstable runs and short samples. Labels are operational and horizon-dependent, not proof of infinite-horizon queue stability.
The acquisition schedule uses different observation horizons across risk bands. A future stability classifier must address that label-horizon confound rather than treating absence of overload in a short window as proof of stability.
Safety thresholds are serialized in selected_conditions.json, raw results and stability_label_summary.json. Backlog is detected after each bounded original-harness window; this wrapper does not claim live backlog-triggered interruption. Hard host-queue limits, drain timeouts and overall timeouts bound failures. Likely-unstable windows are capped at 180 seconds.
All new features come from the shared offline extractor; u and C are standalone/offline estimates, not measured GPU utilization. Real-time queue evidence is used as a label, not a planning input.
The feature-distance normalization is an acquisition artifact fitted before new measurement; it must not be reused as train/test preprocessing without refitting on training groups.
Completed this budget only. No predictor training, deployment search, routing optimization, forecasting or replica extension.
