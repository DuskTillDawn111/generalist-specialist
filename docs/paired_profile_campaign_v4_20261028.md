# Paired natural/kernel training profiles and bounded supervisor

The new isolated entry points are `paired_profile_campaign_v4.py`,
`profile_training_supervisor_v1.py`, `profile_owned_collect_v1.py` and
`profile_process_ownership_v1.py`. They leave all historical collector sources
and evidence intact. CPU preparation, audit and tests load no GPU models.

The paired campaign freezes two distinct execution identities, `natural` and
`kernel`, referencing one exact offered-job document per split. The natural
branch launches no profiler; the kernel branch uses the retained per-worker
CUDA/NVTX capture. Both preserve four separate resident owners, the original
generation settings and the same semantic observer. The original v2 profiler
plan is rejected by the new gate; command-line mode switching is unavailable.
The new collector checks the frozen mode independently before loading workers.

The first campaign is a training pilot, rather than the full488-job campaign.
It contains one four-owner smoke job per mode, followed by26 matched training
jobs per mode: nine single/pair/triple/four-owner contexts at short and long
text lengths plus SG at50/200ms offsets in both start orders. It collects only
training input identities. Both exact source-bound smoke reports and completed
supervisor receipts must pass before training. There is deliberately no new
validation entry: independent predictor freeze and a separate validation gate
remain necessary before held-out outcomes can be collected or inspected.

Preparation example (CPU only, fresh exclusive directory):

```bash
python -m tools.predictor_validation_20261027.paired_profile_campaign_v4 prepare \
  --output artifacts/NEW-EXCLUSIVE-PAIRED-CAMPAIGN
python -m tools.predictor_validation_20261027.profile_training_supervisor_v1 prepare \
  --kind paired --target artifacts/NEW-EXCLUSIVE-PAIRED-CAMPAIGN/smoke/natural \
  --output artifacts/NEW-EXCLUSIVE-PAIRED-CAMPAIGN/supervisor-smoke-natural.json
```

The opt-in `--progress-observer` preparation flag declares a separate execution
identity using `semantic_progress_worker_v1.py`. Both modes bind the same flag,
observer and public jobs. Its CUDA-event record/query costs are explicit;
current-stream event completion is not an all-stream completion frontier or a
GPU work fraction. Default preparation retains `semantic_worker.py`. The actual
chosen worker's complete dynamic local-import closure is required during audit,
including the progress observer when enabled. All source-root Python files,
worker specs, adapter source, exact original registry, input catalog and declared
media are additionally required; deleting a worker source reference and
recomputing context hashes does not bypass this gate.

The GPU execute entry (not run during CPU implementation) is:

```bash
python -m tools.predictor_validation_20261027.profile_training_supervisor_v1 execute \
  --manifest artifacts/NEW-EXCLUSIVE-PAIRED-CAMPAIGN/supervisor-smoke-natural.json \
  --output artifacts/NEW-EXCLUSIVE-PAIRED-CAMPAIGN/execution-smoke-natural --execute
python -m tools.predictor_validation_20261027.paired_profile_campaign_v4 verify-smoke \
  --run artifacts/NEW-EXCLUSIVE-PAIRED-CAMPAIGN/smoke/natural \
  --execution artifacts/NEW-EXCLUSIVE-PAIRED-CAMPAIGN/execution-smoke-natural/execution.json
```

Prepare a separate supervisor manifest for `smoke/kernel` and repeat the same
execute/review sequence. Training manifests can be prepared in advance, but
their execution is rejected until both smoke modes pass. Natural smoke validates
the exact offered job, input/task/arrival offsets, actual START/handler/phase and
parent receipt boundaries, priming, four-owner telemetry and shutdown. Kernel
smoke also requires correlated kernels and valid clock mappings. Neither smoke
is fitting data or an accuracy result.

The supervisor supports existing batchv3 training plans through `--kind batch
--target PLAN --split training`. It uses the pinned text Python, frozen allocator
and device environment, plus actual G2/OCR bindings. It calls the existing
reserved child implementation once under the new outer supervisor, avoiding a
second experiment lock. A batch plan must declare2400s and have no source audit
blockers. Batch validation is refused until an independent frozen-predictor gate
exists. An incomplete batch child returns nonzero. Prepared sources and
environment bindings are checked before launch and after collection; an
otherwise successful child with source drift remains failed.

Every execution uses an exclusive directory, an18GiB free-space preflight,
13GiB free-space abort floor,6GiB incremental-storage ceiling and2400s timeout.
The new `wait_from_baseline` preserves the reviewed `wait_bounded` policy and
reuses its process-group cleanup helpers. It checks every200ms against the
single disk/time baseline captured before launch, including launch and
watchdog-entry growth without resetting the6GiB ceiling. Lock contention refuses immediately. Inherited
`CUDA_MPS_*` controls and a different `CUDA_VISIBLE_DEVICES` are rejected because
they change the declared default-sharing execution identity.

The outer parent holds the experiment lock until actual owned processes exit.
The child validates the parent's pre-existing exclusive FLOCK through
`/proc/locks` and both descriptors' fdinfo, including exact PID/device/inode;
validation never creates a lock on the child's behalf. The parent is a Linux
subreaper, so orphaned descendants that call `setsid` are adopted, identified by
PID/start ticks and reaped before lock release. The dedicated supervisor refuses
collection before launch if it has any pre-existing child tree: a later orphan
from that tree would make attribution ambiguous after its original parent exits.
Those existing processes are preserved. The execution nonce supplements live ancestry. Root-group
exit with a live detached child is a failed collection and triggers cleanup.
Uninterruptible owned processes keep the lock during continued cleanup instead
of allowing another GPU experiment. This entry is a single-collection CLI
supervisor; it does not run arbitrary caller commands from a manifest.

The isolated four-owner collector starts its new monitor before `runtime.start`
and stops it after shutdown. It attributes NVML PIDs through actual owned
descendant/process-group membership and launched worker ancestry, including the
nsys launcher-to-worker transition before READY. Missing PID namespace evidence
remains unknown and fails qualification. Board samples retain PID ownership,
phase and lifecycle boundaries. Memory peaks are sampled owned values, with
transient peaks still unobservable. The finite job schedule is not a throughput
or P95 campaign and does not produce six-term utility.

Current GPU metric capability is explicitly unavailable:
`artifacts/predictor-gpu-metrics-capability-20261028-readonly01/report.json`
records `ERR_NVGPUCTRPERM` from the installed nsys metric capability query.
No permission change or GPU workload was needed for that query. Kernel timing,
context events and static attributes remain useful evidence, but do not establish
SM/DRAM bottleneck causality or cache reuse. All new profile outputs remain
`fit_eligible=false` and `final_gain_eligible=false` pending independent review.

CPU regression command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -W error::ResourceWarning -m unittest tests.test_profile_training_supervisor_v1 -v
```
