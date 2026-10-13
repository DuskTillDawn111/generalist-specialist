# Bounded cross-deployment active profiling

This module performs data acquisition only. One existing MILP seeder call on the
six-task base workload returns a seed. Only its immediate binary add/remove
neighbors are constructed. The existing coverage/memory checks and capacity LP
screen each neighbor and each finite prescribed condition; explicit routing is
also checked by the shared feature extractor. No deployment graph recursion,
exhaustive baseline, learned uncertainty or performance ranking is used.

Six conditions are selected using block-balanced feature distances: own Qwen
load, task-demand mix, offline co-location pressure and resident memory/count.
All dimensions come from the existing extractor; IDs never enter the score.
Two conditions per expected risk band and at most two per deployment context
provide a small bounded design. Risk bands are sampling hypotheses, not labels.
The distance scaling is frozen from offline existing/candidate features and is
not train/test preprocessing.

Run with the existing gsdeploy Python environment:

```
python -m performance_model.active_sampling.plan
python -m performance_model.active_sampling.run
python -m performance_model.active_sampling.audit
```

The run entry point invokes the unchanged measurement harness, saves each of 18
repetitions independently, and automatically audits after measurement. Resume
reuses existing raw files, including failures; do not regenerate a plan with
measurements present. No automatic budget expansion occurs.

Safety policy is explicit in the plan. Windows are 120–300 seconds, risky
conditions capped at 180 seconds; drain is at most 30 seconds, total-run timeout
900 seconds, and the existing host queue is bounded. Backlog detection is
post-window; live backlog interruption is not implemented. Short successful
cohorts are flagged instead of extending overload indefinitely.

`performance_model.stability.label_stability` requires all thresholds. The
binary flag is false for operational failures/overload; the additional boundary
category captures warning-level backlog/drain on otherwise completed runs.
Strict stable-region latency eligibility excludes boundary and low-sample runs.
Missing measurement windows are retained as failed observations without invented
latency/throughput labels.

The merged dataset contains the Stage 2A/2B-1 foundation, Qwen Module 2B and new
runs. It records run/configuration/deployment/repetition IDs on every row and
provides explicit grouping maps. Task rows and repetitions are not independent
deployment samples. Evaluation must group by configuration or entire deployment,
respect protocol differences, and fit preprocessing on training groups only.
Merged evaluation configuration IDs use only x/r/lambda, so even different
measurement windows of the same input stay in one fold. The frozen plan/raw
acquisition condition ID additionally identifies its measurement protocol; it is
retained as `acquisition_condition_id` alongside the conservative grouping ID.

No stability classifier, latency regressor, final predictor, deployment search,
routing optimizer, forecast or replica optimization is trained/implemented.
