# Text capacity diagnostic independent analysis (2026-10-22)

`tools/spillover_capacity_20261022/analyze.py` is an independent CPU consumer of the immutable six-job `balanced_closed_loop_capacity_v1` packet. It does not start GPU workers, fit predictors, rank deployments, compute utility, or alter frozen production code. All `U` fields remain null.

The fixed resident context is text×2 + generalist_2×1 + ocr×1. States A/B/C activate two text aliases, only generalist_2, or all three text-capable aliases respectively. Each active alias receives the same 32-input sequence three times within a block. The six jobs are development diagnostics; two blocks and repeated inputs do not constitute independent formal validation.

Metadata-only readiness (reads packet, bindings, frozen configuration/scorer; does not read queue or raw):

```bash
python tools/spillover_capacity_20261022/analyze.py \
  --packet artifacts/static-spillover-study-20261022/text-capacity-diagnostic-v1/experiment-packet.json \
  --check-only
```

Analysis requires explicit queue progress paths and a new output directory. By default it waits until all six terminal receipts and the final queue completion status are published, then reads only registered exact `run.json` paths with matching SHA. It never scans for latest attempts, reads live snapshots, retries, or overwrites output. `--partial` explicitly permits already registered terminal attempts, with `INCOMPLETE` status and no final repeatability judgment.

```bash
python tools/spillover_capacity_20261022/analyze.py \
  --packet /absolute/path/to/experiment-packet.json \
  --queue-progress /absolute/path/to/queue-progress.json \
  --output /absolute/path/to/new-analysis-directory
```

The analyzer independently verifies packet/source bindings, each planned input/alias/order/repetition identity, resident deployment, native worker commands/environment/READY spec, frozen transport settings, GPU idle boundaries, 21 priming records, and complete drained 30-second warmup. Completed measurement records require verified runtime identity. Early failures without completed measurement records can remain diagnostic UNKNOWN with failed runtime checks disclosed.

Every planned measurement row is retained. Missing/incomplete rows have unknown service and quality; mismatched native response identity cannot acquire valid service. Native unsuccessful responses with valid timing can retain service while quality remains UNKNOWN. Native predictions are scored per record with the frozen `tasks.score_result` and `quality.sample_totals`, keyed by exact input ID. No formal quality aggregate is substituted, and no label is passed to workers. Observed accuracy is descriptive, with the actual available denominator disclosed.

Outputs include:

- `report.md`, `report.json`: completion coverage, service/handler distributions, native accuracy, input/output token counts, per-alias finite-cohort completion rates, repeated-cohort drift, errors, clean shutdown, resource and energy evidence.
- `<job_id>.json`: original measurement rows plus SHA, native responses, host/worker timing ledger, per-input/repetition/quartile summaries, exact half-open handler activity masks, and per-request activity exposure.
- `protocol.json`, `manifest.json`: packet, queue, raw and tool receipts plus frozen bindings and output SHA.

Rates use host monotonic clocks and an explicitly reported denominator. Alias finite-cohort rates end when that alias finishes all planned requests; common-window rates include the measurement tail. Incomplete cohorts retain descriptive common-window completions but do not claim a full-cohort alias completion rate. Repetition rates use that repetition's first schedule through last completion. These are finite closed-loop observations, not open-loop stable capacity.

Observed handler intervals use worker wallclock timestamps. Same-alias intervals are unioned and illegal overlaps separately flagged. A common alias lifetime span can contain idle holes and is not actual simultaneous activity. Singleton B's all-active condition is trivially satisfied during its handler. Unknown requests imply censored observed activity. Handler overlap is neither GPU utilization nor a predictor input. Same-alias/same-input comparisons A→C and B→C are reported within block; same-state block1→block2 comparisons are separate, without pooling, fitting, or significance claims.

Verification: `python -m unittest discover -s tests -p test_spillover_capacity_analysis.py` passed 21 CPU tests, including real frozen scoring, balanced input coverage, censoring, runtime tampering, half-open overlap accounting, publication SHA/newline checks, default no-raw waiting, check-only isolation, and synthetic partial/full-six CLI output/manifest validation.


`tools/spillover_capacity_20261022/plot.py` reads only `report.json` from this analyzer, then creates a compact two-panel PNG/PDF plus `plot-data.json` and an input/tool/output SHA manifest in a new directory. Panel (a) shows same-state/same-alias/same-input block2/block1 service-median ratios, with complete/partial input coverage counts. Panel (b) shows each job/alias median-service drift over the three balanced repetitions. Captions disclose retrospective handler-activity coverage, completion tails, singleton B, clustered inputs, and the absence of utility ranking. Unknown observations are not filled. All manifest paths and printed artifact paths are absolute. Matplotlib 3.9.0 was available; a fully synthetic report rendered successfully to PNG/PDF and was visually inspected without reading any actual experiment raw.

```bash
python tools/spillover_capacity_20261022/plot.py \
  --report /absolute/path/to/analysis-v1/report.json \
  --output /absolute/path/to/new-figure-directory
```
