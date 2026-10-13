# Module 8C: Final Real Multi-workload Search

The module binds the immutable Module 8B final workload matrix, quality policy,
admission matrix, v2.1 pool and real 27-path catalog. The existing MILP,
canonical LP, integer neighborhood, analytical ranking, runtime evaluator,
noise-aware objective and replica measurement harness are imported unchanged.

The effective catalog contains the 21 admitted edges. PaddleOCR-VL has no
admitted edge and is fixed to zero; exports restore all 12 model coordinates.
Evidence bounds stay at two for Qwen, Whisper-small and ResNet50, one otherwise.
Each workload uses a cold empirical MILP seed, predictor=None, Top-B=2, at most
six unique runtime states, three search repetitions, three iterations and
patience=1. Measurement lasts 210 seconds with 10 seconds of warmup. Search
seeds are 7101/7202/7303; independent confirmations are 8101/8202, in the
unchanged seed1/final1/final2/seed2 order. Confirmations cannot alter decisions.

```bash
OMP_NUM_THREADS=1 envs/data_env/bin/python -u -m module8c_final_real_search.campaign
```

Preparation checks native weight hashes, environments, frozen artifacts,
quality masks and the six seeds against 8B before GPU use. The campaign runs
the six workloads sequentially on the same GPU. Exact-cache recovery reuses
completed repetitions; it does not launch duplicates or enlarge budgets.

One initial startup attempt failed before measurement because replica worker
names bypass the historical literal-name PaddleOCR library-path branch.
Its entire experiment is retained in
`results/module8c_final_real_search_startup_failure_001/`. The corrected campaign
inherits the existing OCR CUDA library path, without editing a model handler,
runtime harness, package or search algorithm. That recovery and its extra
invalid physical attempt are explicitly recorded in the new execution freeze
and report. Further failures stop without automatic retries or invented metrics.

Outputs live in `results/module8c_final_real_search/`. Reports retain full
vectors, Generalist/Specialist identities and counts, every candidate decision,
replica activations, raw timings, memory telemetry, confirmations and final
deployment diversity. Runtime feedback reversals distinguish a positive
analytical delta that fails the incumbent objective from another evaluated
Top-B candidate being preferred. No unvisited single-Generalist superiority is
inferred from co-deployed additions.

The campaign stops after Module 8C. It does not run a baseline, ablation,
dynamic adaptation, new standalone/scaling campaign or another hardware target.
