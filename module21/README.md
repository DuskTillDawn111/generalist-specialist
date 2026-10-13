# Module21 frozen Poisson validation

Results: `results/module21_positive_utility/`.

The original input protocol, Module20 utility/reference hashes, all 432 deployment
states, all eligible continuous-route competition, and coarse/fine CDF results are
retained. `acceptance_clarification.json` corrects a preliminary extra CI gate before
any GPU observations: all three finite paired gains must exceed 0.01; report CI as
small-sample uncertainty. Original scientific weights and stability thresholds stay
unchanged. `frozen_manifest.json` contains the final effective GPU acceptance rule.

CPU uses `envs/data_env/bin/python`; native GPU control uses
`/root/miniconda3/envs/gsdeploy/bin/python` and launches historical workers through
unchanged registry/runtime paths. Model workers run in their original environments.
No environment installation, global capacity penalty, historical/V3 patch, or model
search is involved.

The conditional sequence is freeze → CPU full-pool optimization → CPU tests → only
if Stage A passes, freeze_gpu → gpu → finalize. Do not remove raw files or rerun
observations. A campaign resumes only existing frozen planned slots; it must retain
all outcomes and stopping decisions. CPU queues assume Poisson thinning with
constant standalone service; real GPU request E2E and worker service remain separate.

A generated trajectory stores sample, task, relative arrival, routing uniform and
replica uniform. Paired processes consume exactly the same trajectory. Arrival RNG
streams are independent by task. Repetition seeds are fixed before optimization,
with no seed screening. Three fresh process repetitions per arm are planned; later
stages stop after repeated instability or failure of a completed stage's utility rule.

No Module22 or additional models are launched by any Module21 command.
