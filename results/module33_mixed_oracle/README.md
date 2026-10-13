# Module33 isolated benchmark

All implementation, raw data, manifests, caches and new logs live here. Historical
Modules14F–32G, model weights, utility definitions and quality admission stay read-only.

The frozen finite domain is in `candidate_space_definition.json`; original
profiles and variable bounds are in `pools/`. `optimizer_objective_audit.json`
distinguishes the promoted formal V3 from the unpromoted20A utility algorithm.
The latter's grid adapter changes only route-variable units and integrality,
not its objective, generation schedule or ranking.

`cpu/*_all_candidates.csv` contains every route surviving route-independent
coverage/memory screening, including any nominal capacity rejects. All replica
vectors and screening reasons are in `feasible_deployments.csv`. The independent
Cartesian census additionally counts every covered route BEFORE memory screening.
Do not mistake21xx evaluated legal routes for the full79,704-route domain, or
claim continuous global optimality from grid results.

To reproduce CPU enumeration, optimizer decisions and independent verification
in a fresh subdirectory (never overwriting the recorded experiment):

```bash
cd /root/autodl-tmp/generalist-specialist
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=results/module33_mixed_oracle:. \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/miniconda3/envs/gsdeploy/bin/python -m m33.reproduce_cpu --name audit_run_1
```

The GPU input domain, all development configurations, request/input hashes and
independent seeds are fixed by `frozen_problem_manifest.json` and the GPU manifests.
Versions1–4 retain the premeasurement syntax/cache corrections and the journal
assessment correction. `raw_runs/` and `request_journals/` are original evidence;
the journal correction reconstructed native derived latency fields in memory
from identical timestamps and did not rerun that measurement. Initialization
failures remain NULL and cannot compete for empirical utility.

Do not rerun the recorded GPU campaign: its startup assertions prevent outcome
retries. A new GPU replication requires a separately frozen manifest and new
run IDs, retaining every negative/invalid observation. No automatic Module34
or algorithm tuning is authorized by this benchmark's output.

Only `module33_gate.json` after full final SHA256 verification is the final gate.
Intermediate versions deliberately mark history unverified and GPU incomplete.

In `gpu_observations/*.json`, top-level `U` and `eligible` are the authoritative
Module33 feasibility-gated outputs. `components` retains the unmodified Module26
raw-arithmetic helper output: its nested `U=null, eligible=false` are defaults
because this harness does not set Module26's `_path` archival-extractor input.
Only `components.raw_arithmetic_U` supplies arithmetic, independently verified
against the original20A component function; Module33 separately applies the
original stability classifier, complete-cohort, quality, trace and budget checks.
No invalid observation receives a top-level sortable utility.
