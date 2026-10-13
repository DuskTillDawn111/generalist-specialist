# Module 13B execution

The original Module13A artifacts are immutable. `download.py` consumes exact
frozen SHAs and verifies all LFS files against archived SHA256. Transport uses
independent byte ranges, checked against Content-Range and final full SHA.
One local artifact directory is used per checkpoint.

1. `python module13b/download.py`
2. `envs/nvml_env/bin/python module13b/campaign.py cpu`
3. Inspect CPU failures and apply correctness-only compatibility repairs before
   performance. Record failures; do not substitute checkpoint, precision,
   backend, dataset, task, admission or candidate order.
4. `envs/nvml_env/bin/python module13b/lock.py`
5. `envs/nvml_env/bin/python module13b/campaign.py gpu`
6. `python module13b/analyze.py`
7. With original CPU optimization environment and OMP/MKL/OPENBLAS threads1:
   `/root/miniconda3/envs/gsdeploy/bin/python -m module13b.replay`
8. Generate summary and integrity validation; stop. Replica profiling is a
   decision artifact only and is not executed in13B.

GPU campaign runs exactly one model process at a time, all quality paths before
standalone performance, references first in every task/repetition. For latency,
one untimed warmup, required five-second idle memory observation, 30 sequential
original pool requests. Every result is retained. All nine paths receive
performance measurements under the user's13B instruction; quality failures
remain ineligible for selection. Sampling is external NVML observation, not an
artificial request delay. Old worker protocol and baseline workers are unchanged.

Selection is independent of CPU replay and uses the immutable13A Pareto and
log-distance rule. Complete old efficiency pool remains its own unchanged
comparator. Only tasks with selected new winners replace their old Specialist
paths in BALANCED. Other tasks and all Generalist capabilities stay fixed.
New counts greater than1 remain unknown. A separate CPU diagnostic limits both
regimes' searched-task Specialist identities to real single-replica evidence;
primary and diagnostic outputs are kept separate. No multi-replica measurements,
Generalist GPU campaign, or final deployment result enters model selection.
