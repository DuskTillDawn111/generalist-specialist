# Finite H600 training acquisition

Prepared on CPU only: `artifacts/predictor-gpu-20261027/continuous-training-v14-h600/campaign.json`, SHA256 `f7199c2d14dcf173f6cc4abcccdab6163c0443d78551119656fd7435f92b8723`. No independent result or v11 execution output was read to prepare this campaign. No GPU run was launched by preparation.

The campaign contains two fresh training seeds, 627601 and 627602, each under balanced, skewed, spill_low and spill_high shared routing. H=600 s, bounded drain=300 s, warmup=0, offered rates text=1.85 and each of image/ASR/OCR=.05 rps. Every run predeclares 1,200 requests: text=1,110 and 30 each background task. The fixed text×2 + G2 + OCR residency, source-bound worker stages, input domain and resource configuration are retained. Two extra cold runs repeat exactly seed 627601's balanced and spill_low arrival/input trace, providing a first descriptive same-trace run comparison. These are new run replicates, not additional independent input seeds.

Every plan has predictor_training/prospective_training/training role and created/source/protocol receipts before startup. The ten-job order is prospectively shuffled and frozen. The unchanged native collector records the full offered cohort, observed stages, parent E2E, actual Q numerators/denominators, board energy and owned process memory, and performs original readiness/prime/drain/shutdown. The new executor requires a completed clean native collection and clean shutdown before the next job; a failed run stops the campaign without retries or replacement seeds. Exclusive start markers additionally prevent rerunning a phase after partial failure.

An isolated H30 spill_low preflight, seed 627600, checks collection plumbing before committing the main campaign. Its role explicitly excludes it from H600 numerical training. Execute from the repository root:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m tools.predictor_validation_20261027.training_campaign_v14 execute \
  --campaign artifacts/predictor-gpu-20261027/continuous-training-v14-h600/campaign.json --phase preflight
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m tools.predictor_validation_20261027.training_campaign_v14 execute \
  --campaign artifacts/predictor-gpu-20261027/continuous-training-v14-h600/campaign.json --phase main
```

Expected wall time is 100 minutes of main arrivals plus roughly 20–45 minutes for ten cold startups, priming and clean shutdowns; preflight adds roughly 2–4 minutes. The declared 300-second drain allows up to 50 additional minutes over all ten runs. These are planning estimates, not promised completion times. This is new matched-horizon training evidence, not transfer of H90 capacity or P95 qualification.

`primary-training-suite.json` contains the eight unique seed/route primary runs and preserves the frozen fitter's existing input format. `repeat-training-suite.json` contains the two repeats separately: the frozen fitter derives IDs from seed+candidate and would otherwise collide. The campaign supplies an explicit unique run_id for all ten; no raw observations are relabeled. Before integrating repeats into a new fit, freeze an adapter with run-aware identities and group every repetition of the same seed in one training-CV fold. Do not treat requests or repeated inputs as independent run evidence. The old capacity builder likewise expects one run per declared seed/condition, so use the primary suite unchanged and report repeat sensitivity separately.

The next statistical work is to summarize these new raw runs and compare repeated traces, retaining all complete observations and failures. Fit/selection policy should then remain finite and mechanism-driven rather than sweeping hyperparameters on the previous eight H90 flows. Original stability and six-term scoring rules remain separate checks; native collection success alone does not claim steady queues or final diversion gains.

Three CPU tests cover the finite two-seed-plus-repeat structure and prospective roles, tampered source rejection, required preflight, and failure preservation without retry. `verify` rechecks campaign/plan hashes and source receipts without opening runtime results:

```
python -m tools.predictor_validation_20261027.training_campaign_v14 verify \
  --campaign artifacts/predictor-gpu-20261027/continuous-training-v14-h600/campaign.json
```
