# Module 3B real-runtime pilot

This is an explicitly authorized, single-workload GPU pilot. Module 3A validation
artifacts remain sealed. The search implementation gained optional objective and
checkpoint hooks; default Module 3A objective/selection behavior is unchanged.

`MeasurementRuntimeEvaluator` calls the existing
`joint_deploy.measurement.run_measurement` CLI sequentially, using the original
workers. Each exact configuration has three fixed-seed repetitions. Aggregation
uses the worst stability_state_v2, equal-weight repetition latency means, and
sample standard deviation (`ddof=1`). All raw results, logs, configurations,
per-task latencies, worker latencies, throughput/completion and queue evidence
remain available, including failures. If any repetition lacks a task latency,
the aggregate latency is missing; available-only descriptive statistics retain
their observation count and never authorize steady acceptance.

`NoiseAwareRuntimeObjective` extends the existing lexicographic comparator.
Stability improvement bypasses the noise gate. Same-steady latency-based
acceptance requires at least 5% improvement in workload-weighted mean E2E;
weighted P95 cannot circumvent this gate. Memory-only objective ties retain the
original rule. Repetition std is diagnostic, not a significance test.

`RuntimeEvaluationCache` keys deployment, workload, exact canonical routing,
measurement settings and repetition seeds, stability thresholds, and source/model
content hashes. Fingerprints cover model files, referenced task inputs, prompts,
measurement/worker source, interpreter bytes, package metadata and GPU identity.
Historical Module 2G runs lack model content hashes, so they cannot be certified
as exact cache hits. They are used only as workload-selection evidence.

The cache uses a file lock, atomic manifest writes, and a durable reservation
before each real run. Completed raw measurements are never relaunched. The
first recovery pass conservatively wrote a failure envelope for an attempt
interrupted by host power-off; it was later corrected to `interrupted_external`
after the user clarified the cause. That envelope, original log, process
metadata, old aggregate and trace are archived. The attempt remains in the
physical run ledger but is excluded from the three valid repetition labels; a
single replacement was authorized within the original 18-attempt cap. A
different in-flight run that had produced a complete raw result was restored
without remeasurement. The corrected effective configuration cap is five.
Completed cache reads verify aggregate and raw hashes. Corrupted entries cause
an explicit error rather than remeasurement. A live previous process prevents
duplicate launch. The underlying measurement process has a bounded timeout;
timeout cleanup targets only its own process group.

Frozen pilot limits are Top-B=2, max iterations=3, patience=1, at most six unique
configurations and eighteen real run attempts, including the seed and failures.
No candidate replacement, predictor, exhaustive enumeration or automatic budget
expansion is used. The existing W1_balanced rates, 210-second measurement window,
10-second warmup and three historical repetition seeds are reused. It had 15/15
historical steady repetitions but that does not guarantee current stability.

The run entrypoint is `python -m deployment_search.real_pilot`; use the existing
`/root/miniconda3/envs/gsdeploy/bin/python` environment and single CPU thread
environment settings. `--prepare-only` saves the workload and frozen config before
any GPU run. A finalized pilot is not rerun. An interrupted pilot replays search
using the persistent cache; it does not refresh previous measurements. This is
a recovery path, not authorization for another experiment.

Outputs are in `results/deployment_search_real_pilot/`. The decision audit replays
the Top-B acceptance decisions, verifies one-hop mutations, LP delta/ranks, budgets,
cache identities and raw hashes. The final state is explicitly requested from
the cache once to prove reuse adds zero real runs. The summary distinguishes
search LP solves, unique runtime evaluations, real runs and cache hits.

CPU-only tests:

```bash
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m unittest deployment_search.tests.test_real_runtime deployment_search.tests.test_search -v
```

Real exhaustive optimum is unavailable. This pilot does not establish global
optimality, statistical significance or final paper performance. No additional
workload, routing optimization, replicas, forecasts or predictor retraining is
included.
