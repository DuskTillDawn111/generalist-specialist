# Module 2F: saved-OOF decision-focused ranking gate

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/miniconda3/envs/gsdeploy/bin/python -m performance_model.ranking.evaluate
/root/miniconda3/envs/gsdeploy/bin/python -m unittest performance_model.ranking.test_ranking -v
```

Only reads the Module 2D primary LODO predictions and Module 2E v2 labels.
No estimator is fit or exported, no inference worker is invoked, and neither
local calibration nor deployment search is performed.

Comparison requires exactly matching task arrival rates and compatible measurement
protocol. Deployed worker membership can differ, because that is a deployment
treatment; shared implementations and all other protocol fields must agree.
Measurement duration and warmup are not relaxed. Candidates are paired on common
seed/arrival-phase blocks. All repetitions of a configuration must satisfy the
analysis state restriction; unstable repetitions cannot be silently removed to
make a configuration appear steady.

Mean scores combine saved path means by planned routing fractions, then apply the
same workload weights as measured task-aggregate means. Task P95 ground truth is
recomputed from request records, never approximated by a mixture of path P95s.
There is no saved P95 predictor: predicted tail objectives remain null. Optional
tail ranking metrics use the mean score as an explicitly labelled proxy baseline.

One workload/protocol comparison is one group regardless of its repetitions,
variants or strict/extended views. Routing-only groups are reported separately
from cross-deployment groups. A two-candidate Top-2 hit is trivial and cannot
establish shortlisting quality. Insufficient cross-deployment groups leave the
PASS/PARTIAL/FAIL gate unassessed, rather than manufacture a ranking conclusion.
