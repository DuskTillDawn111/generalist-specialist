# Module 8A quality + workload demand audit

Only measures quality of the frozen verified candidate paths, revalidates Phi/MiniCPM on the exact existing ScienceQA task, reads completed search evidence and freezes demand-normalized workload proposals. It does not import or run a deployment search.

Run from the project root:

```bash
envs/data_env/bin/python -m quality_workload_audit.prepare
python -m quality_workload_audit.evaluate
python -m quality_workload_audit.model_provenance
python -m quality_workload_audit.demand
python -m quality_workload_audit.finalize
python -m quality_workload_audit.plot
```

`prepare` freezes the existing 200 IDs per task, task definitions, asset/source hashes and historical quality presets before quality outcomes. It refuses to replace a differing manifest. `evaluate` starts one isolated current serving worker at a time and saves all outputs; completed model audits are reused only against the saved audit manifest. VQA revalidation uses an audit-only wrapper without registering new serving capabilities. It extracts the exact pure ScienceQA prompt/normalization functions from the frozen source; no Qwen-specific dependency is added to another model's environment. It adds no accuracy gate.

`model_provenance` hashes current local model assets. `finalize` compares them against the frozen 7C4 model file identities, validates all 200 IDs and metrics, and checks that original artifacts and algorithms were unchanged. WER/CER use corpus totals. Historical summaries lacking complete revision/preprocessing/decoding bindings are not mixed into the current quality matrix.

`demand` freezes target shares and arrival vectors algebraically, with a common reference load equal to the median of the original three workloads. It refuses to replace differing proposals. Canonical planning is optional and was not run. Proposal freeze is distinct from the eventual final benchmark/quality policy freeze. Newly verified VQA paths have no standalone s0 yet, so the v2.1 proposal cannot be silently admitted into the optimizer.

All new artifacts are in `results/quality_workload_audit/`. No final quality threshold is invented. Legacy four-task quality presets are reported separately from the unspecified current six-task optimizer policy.
