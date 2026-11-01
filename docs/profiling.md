# Measuring model profiles

The current compact study uses QA, image classification, whole-page OCR and ASR.
Prepare its four-task manifests with `python -m gsdeploy.profiling.study`; see
[`configs/study.json`](../configs/study.json) and the [pool design](model_pool_design_48g.md).
LID/VQA are excluded from this study. The older adapters and preparation protocol
below retain historical LID compatibility; they do not define the active study.

`gsdeploy.profiling` measures pinned local checkpoints with real, frozen input
manifests. Profiling results do not automatically admit a model or overwrite the
deployment catalog. It supports UnifiedQA-v2, EVA-CLIP, olmOCR-2 and the shared
VibeVoice/fastText ASR and LID pipeline.

Every replica runs in its own process with all model parameters on the same GPU,
BF16 weights and batch size one. Startup includes loading, fixed class feature
preparation and three warmup requests. Three shuffled rounds then measure a
saturated closed loop. Each round finishes all dispatched requests, lasts at
least 30 seconds and covers every declared input; the default request minimum
is 64. Truncated generations count as unsuccessful requests, while their
latency and energy remain in the accounting.

Use a dedicated GPU and an environment providing PyTorch, compatible
Transformers, the native model implementations, NVML, and the task processors.
Model specs explicitly provide local checkpoint, tokenizer and resource paths.
The pinned VibeVoice source must be importable. No weights are fetched by the
inference adapters.

```bash
python -m gsdeploy.profiling.runner \
  --spec MODEL_SPEC.json --samples SAMPLES.jsonl \
  --output NEW_OUTPUT_DIRECTORY --replicas 1
```

The spec contains `model: {id, revision, kind, path}` and `resources`. Kinds are
`qa`, `image`, `ocr`, and `speech`. Image resources include the 1000-class index;
OCR includes the frozen native prompt; speech includes the local tokenizer and
fastText file. For QA, the native SentencePiece tokenizer's decoded no-answer
marker is normalized. For the older Transformers interface, the OCR adapter
validates duplicated config dimensions and constructs a flat in-memory config;
the original checkpoint files remain untouched.

For reproducible calibration preparation, populate the frozen dataset downloads
and metadata in the output directory, then run:

```bash
python -m gsdeploy.profiling.prepare \
  --output RUN_DIRECTORY --assets RESOURCE_DIRECTORY --archive ORIGINAL_DATA_DIRECTORY
```

The current preparation contract uses SQuAD2, SROIE pages, LibriSpeech
test-other, the existing ten VoxLingua languages cropped to declared lengths,
and public ImageNet-A. ImageNet-A is a 200-source-class stress dataset with
1000-way predictions; it is explicitly not ImageNet-1K validation. Manifest
selection is independent of model predictions. Quality diagnostics deduplicate
repeated performance requests and cannot replace independent admission tests.

Whole-board energy uses trapezoidal integration of sampled NVML board power,
requiring full sampled coverage and bounded gaps. Cumulative GPU energy counters
are retained as diagnostics, because this device's counters systematically
disagree with board-power integration. They are not substituted for missing
board coverage. Total memory and process memory are retained. CPU energy is not measured. Service and client latency tails
are distinct; a closed-loop throughput measurement does not establish an
offered-load SLO.

`gsdeploy.profiling.suite` sequentially evaluates model replicas and excludes
memory domains using the measured single-replica resident footprint and the
configured safety policy. An exclusion is a derived conservative decision,
not a claim of an observed OOM. It distinguishes failed loads from measured
profiles and supports retrying only failed configurations with a fresh output
suffix. Existing raw results are never overwritten.

```bash
python -m gsdeploy.profiling.suite --root RUN_DIRECTORY
python -m gsdeploy.profiling.suite --root RUN_DIRECTORY --retry-failed --suffix=-retry2
python -m gsdeploy.profiling.report --root RUN_DIRECTORY
```

The suite expects prepared `qa/image/ocr/speech-spec.json` files and
`text/image/ocr/speech.jsonl` manifests. Its initial entry can wait for an
already-running QA experiment via `--existing-qa`; use it only with a matching
existing experiment. Summaries include service/client tails, measured board
energy, startup cost, real same-card replica scaling, and diagnostic quality.
Mixed ASR/LID measurements retain one shared model instance per replica and
include full transcription plus text-language classification.

After a single-replica speech profile completes, the CPU classifier can also
be measured separately on its actual transcripts:

```bash
python -m gsdeploy.profiling.cpu --root RUN_DIRECTORY
```

This measures only text language classification, including empty transcript
handling. It reports three rounds, import/load/warmup time, process peak RSS,
and latency. CPU energy remains unmeasured; these component numbers do not
replace the complete audio LID pipeline. VibeVoice's native acoustic latent
sampling is retained; greedy text generation does not make that encoder
deterministic. Raw outputs remain available for checking repeated predictions.

These profiles remain separate from whole-plan energy and colocation evidence.
Energy is not assumed additive across models. Comparing new specialists to
all three generalists requires matching task, input, hardware and energy
contracts before an admission decision.
