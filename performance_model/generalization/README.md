# Module 2D exploratory gate

CPU-only Leave-One-Deployment-Out evaluation of the existing unified dataset.
This package does not export a planning predictor or acquire new measurements.
Existing feature extraction, offline calibration, and measurement code are unchanged.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/gsdeploy/bin/python -m performance_model.generalization.evaluate
/root/miniconda3/envs/gsdeploy/bin/python -m unittest performance_model.generalization.test_gate
```

The fixed evaluation and decision policy is saved before fitting. Every deployment
is held out in full, and preprocessing is fit on the remaining deployment groups.
Standalone calibration is shared offline knowledge, not held-out joint-run labels.
The primary eligibility rule is the existing `stability.stable is True`; because
that boolean includes boundary cases, a second strict subset excludes boundary,
smoke runs and paths with fewer than 30 completions. Both results are retained.
No latency-threshold exclusion, random-row split, tuning or deployment-ID encoding
is performed. Missing/nonpositive standalone times and targets are explicitly
reported and excluded; unavailable resource dimensions are handled only using
training-fold medians and missingness indicators, never fabricated profiles.

A is standalone latency, alongside the existing controlled piecewise baseline
constructed exclusively from each training fold (with explicit standalone fallback).
B contains `u,q`; C adds numeric co-location, offline path and deployment/model
features. Both fixed Ridge and small Huber GBT models compare direct milliseconds
and `log(L/s0)`. The normalized variants determine the gate under the saved rule;
direct variants are reported as diagnostics, not chosen after evaluation.

Configuration-balanced training weights avoid allowing a configuration with many
runs/paths to dominate. Fold metrics retain per-row and configuration-macro values;
the deployment macro is an unweighted average of held-out folds. Repeated tasks
are not counted as independent deployment samples. No statistical significance or
causal attribution is claimed.
