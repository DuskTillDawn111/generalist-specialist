# Module 7C-1 real replica characterization

This separate experiment launcher maps each family count of one or two to distinct
`family#i` transport IDs. Each ID starts the same frozen model script in a separate
process with independently loaded weights and a CUDA context. The historical
binary measurement runner, integer search and its analytical capacity/memory
surrogates are unchanged. This launcher is not connected to deployment search.

The frozen plan selects Qwen-ASR, Whisper-small-ASR and ResNet50-image. Three
arrival rates are set from existing standalone demands and historical raw timing,
then locked before GPU loading. Equal smooth weighted round-robin balances the
total family request stream. Paired conditions share seeded arrival/sample order.
Every run loads fresh worker processes, primes each replica, warms at fixed low
load, measures for 30 s, drains for at most 60 s, shuts down, and verifies NVML
memory release. Protective backlog stop preserves the shortened raw outcome and
skips only remaining repeats of that exact condition. No retries, tuning or
additional offered-load/replica sweeps are automatic.

Within-window completions determine the primary throughput. Cohort completion
including drain is separately saved: full post-drain completion does not establish
sustainable capacity. Existing Data Quality Audit reconstruction and thresholds
classify steady, recoverable congestion and instability without changes. Failed
and unstable two-replica evidence remains visible. NVML process/device residency,
sampled workload/drain peak, utilization, bandwidth activity and power are saved.
The peak is sampled every 0.5 s and can miss shorter allocation peaks.

Commands for this one frozen campaign (never rerun saved raw attempts):

```bash
envs/data_env/bin/python -m replica_scaling.plan
envs/data_env/bin/python -m replica_scaling.campaign
envs/data_env/bin/python -m replica_scaling.analyze
envs/data_env/bin/python -m replica_scaling.validate
envs/data_env/bin/python -m replica_scaling.report
```

Three tested rates provide observed sustainable points/ranges, not exact maximum
capacity or long-duration stability proof. Exploratory `g_m(2)` and `gamma_m` are
characterization outputs. They are not inserted into Module 7B or used to optimize
Generalist/Specialist allocation. This module stops after the reports.
