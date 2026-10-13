# Module 7A-1

Isolated compatibility/resource inventory. Frozen baseline algorithms, catalog,
workers and environments are not edited. New worker launchers reuse the existing
process protocol; `registry()` can be supplied through the existing runtime's
registry injection in a later module. This module does not invoke that runtime's
workload generator or search tools.

Evidence is in `results/candidate_pool_v2`. A task capability becomes true only
with actual current adapter inference evidence. Model cards alone are insufficient.
The static adapter preflight is a schema/semantics gate, distinct from capability
verification. Missing or failed candidates remain pending and are excluded from
the formal pool; nothing is imputed from checkpoint file sizes.

Run preparation and minimal single-model tests through Python modules:

```bash
python -m candidate_pool_audit.provenance before
python -m candidate_pool_audit.prepare
python -m candidate_pool_audit.download
python -m candidate_pool_audit.load_test convnext_tiny roberta_sst2 paddleocr_vl whisper_turbo distil_whisper
python -m candidate_pool_audit.provenance after
python -m candidate_pool_audit.finalize
python -m unittest candidate_pool_audit.test_freeze
```

The downloader expects pinned official metadata saved in `sources/*.api.json`.
This first audit obtained metadata via the public HF mirror after checking the
official publisher cards. Download SHA-256 is checked against repository LFS
metadata in `download_integrity.json`. Never overwrite `provenance_before.json`
when auditing an existing run. Existing raw attempts are not rerun; preserve them
and use a separate result directory for any future audit.

Each model loads exactly once in this audit, receives one minimal warmup and
three sanity requests, then shuts down. There is no concurrency of GPU workers.
Reported idle VRAM is the maximum of three quiescent per-process NVML samples at
each load/warmup/sanity phase. It includes process contexts and allocator caches,
and is explicitly separate from weight size and transient inference peak.

The resource sum is evidence for the declared process resource model, not a joint
deployment experiment. Generalist simultaneous residency inferred from the idle
sum does not verify inference peaks or arbitrary workloads. New-model quality and
latency need later authorized experiments. Stop after pool freeze/audit.
