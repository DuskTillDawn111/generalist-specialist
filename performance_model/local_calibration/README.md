# Module 2E offline gate

No new measurements or production model are created. All existing code, raw labels
and Module 2D results remain unchanged. This directory adds an offline executable:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/gsdeploy/bin/python -m performance_model.local_calibration.evaluate
/root/miniconda3/envs/gsdeploy/bin/python -m unittest performance_model.local_calibration.test_local_calibration -v
```

`stability_v2.py` reconstructs the submitted-request backlog from monotonic host
send-start and successful completion timestamps. It also saves scheduled-cohort
backlog separately. Full standalone demand of pending requests is not residual GPU
work. Missing times/profiles are explicit. Warning sensitivity changes magnitude
thresholds by 0.75/1/1.25; hard failure thresholds stay fixed.

Global fitting reuses the fixed Module 2D C GBT and training-only preprocessing,
with only v2 steady rows from other deployments. Both strict and extended evaluation
use that same fit. Local calibration is a median log residual only, with all
repetitions of any selected configuration kept together. No task-specific or affine
correction is fitted. Recovered congestion is evaluated separately, never used as a
normal calibration probe.

Selection is planning-only. A conservative configured load screen and deterministic
farthest-point selection operate before probe labels are revealed. Revealed non-steady
configurations are rejected without replacement; they still consume budget. A trial
reserves its entire supported nested calibration chain before assigning a fixed test
set. Unused reserved points at smaller K are neither fitted nor tested. This permits
paired K comparisons without reusing calibration repetitions as evaluation data.

Some contexts have too few independent configurations. Unsupported budgets are
reported explicitly. Metrics retain all trials, sample standard deviations and
worst cases; deployment macro scores must be compared only on matching support.
Trials reuse existing data and are not statistically independent. A single
configuration is not equivalent to a single run; measurement cost is reported.
