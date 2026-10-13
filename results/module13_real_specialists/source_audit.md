# Module 13A source audit

Only CPU / WEB / METADATA work was performed. No model weights were fetched and no model was loaded or run. Public unauthenticated HF API access establishes discoverability; 13B must validate weight downloads and offline runtime.

Metadata snapshots are pinned to model SHAs, with retrieval timestamps and local SHA256 hashes in source_manifest.json. Exact files and LFS hashes are preserved in each *.api.json. Missing README files are recorded by their absence, not reconstructed.

Original frozen Text is SST-2 sentence sentiment classification; original LID is VoxLingua107 spoken-language identification from audio. Generic pretrained encoders needing a newly trained classifier, text-LID, SST-3, NLI and generative sentiment substitutes are excluded.

GPU feasibility is an inference from ready batch1 architectures and artifact sizes, not a measured allocation. Parameters, disk weights, idle VRAM, peak VRAM and latency are separate fields. No parameter ratio is used as gamma or mu. Current local host metadata reports 49140 MiB; a standard RTX4090 24GB deployment has not been tested.

LID has semantic-only admission. QUALITY_LIKELY_PASS for LID does not claim accuracy >= the old reference. Public family and full-set scores cannot be local subset ground truth.

MMS-1024 vs MMS-256 retains the same48-layer/1280-wide backbone. Artifact bytes increase by about3.15MB (0.0815%); primarily a classifier/storage difference is plausible, not established latency or idle-memory difference.

No candidate is PRELIMINARY_DOMINATED: comparable local latency/idle memory are missing. Unshortlisted candidates are retained with explicit evidence/reasons.

Search coverage: six candidate task paths each for ASR/LID/Text; 15 unique checkpoint repositories. HF search terms included roberta-large-sst2, roberta-large-finetuned-sst2, bert-large-sst2, deberta-v3-large-sst2. Few-shot SetFit results, untrained base models and incomplete dataset evidence were not promoted to same-task likely-pass checkpoints. This is a bounded public search, not an exhaustive ecological claim.

## Public evidence

- asr:openai/whisper-medium: QUALITY_LIKELY_PASS; wer=0.029 on LibriSpeech test-clean, full set. Whisper paper Appendix D.1.1 Table 8, greedy decoding. Different normalizer and sample set; not local admission. [source](https://cdn.openai.com/papers/whisper.pdf), [source](https://huggingface.co/openai/whisper-medium)

- asr:openai/whisper-medium.en: QUALITY_LIKELY_PASS; wer=0.031 on LibriSpeech test-clean, full set. Whisper paper Appendix D.1.1 Table 8, greedy decoding. Different normalizer and sample set; not local admission. [source](https://cdn.openai.com/papers/whisper.pdf), [source](https://huggingface.co/openai/whisper-medium.en)

- asr:openai/whisper-large-v2: QUALITY_LIKELY_PASS; wer=0.027 on LibriSpeech test-clean, full set. Whisper paper Appendix D.1.1 Table 8, greedy decoding. Different normalizer and sample set; not local admission. [source](https://cdn.openai.com/papers/whisper.pdf), [source](https://huggingface.co/openai/whisper-large-v2)

- asr:openai/whisper-large-v3: QUALITY_LIKELY_PASS; None=None on None. Official card reports 10–20% lower errors than large-v2 across evaluated languages; family-level inference only. No verified exact test-clean WER for this checkpoint in this audit; not a substitute for local WER. [source](https://huggingface.co/openai/whisper-large-v3)

- asr:nvidia/parakeet-tdt-0.6b-v2: QUALITY_LIKELY_PASS; wer=0.0169 on LibriSpeech test-clean, full set. Official card: greedy TDT, no external language model; test-other WER 3.19%. Different scorer/precision and sample set remain possible. [source](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)

- asr:facebook/wav2vec2-large-960h-lv60-self: QUALITY_LIKELY_PASS; wer=0.019 on LibriSpeech test-clean, full set. Official card greedy CTC argmax example, no external LM; test-other 3.9%. Dataset-specific fine-tuning is pre-existing; no user fine-tuning needed. [source](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)

- lid:openai/whisper-medium: QUALITY_LIKELY_PASS; None=None on None. LIKELY_PASS means existing semantic-only admission, not a numeric accuracy promise. Whisper family supports acoustic language-token detection; paper reports best-model FLEURS accuracy 64.5% (all 102 languages) / 80.3% (82 overlapping), not checkpoint-specific VoxLingua accuracy. Larger size alone does not imply better frozen-task quality. [source](https://cdn.openai.com/papers/whisper.pdf), [source](https://github.com/openai/whisper), [source](https://huggingface.co/openai/whisper-medium)

- lid:openai/whisper-large-v2: QUALITY_LIKELY_PASS; None=None on None. LIKELY_PASS means existing semantic-only admission, not a numeric accuracy promise. Whisper family supports acoustic language-token detection; paper reports best-model FLEURS accuracy 64.5% (all 102 languages) / 80.3% (82 overlapping), not checkpoint-specific VoxLingua accuracy. Larger size alone does not imply better frozen-task quality. [source](https://cdn.openai.com/papers/whisper.pdf), [source](https://github.com/openai/whisper), [source](https://huggingface.co/openai/whisper-large-v2)

- lid:openai/whisper-large-v3: QUALITY_LIKELY_PASS; None=None on None. LIKELY_PASS means existing semantic-only admission, not a numeric accuracy promise. Whisper family supports acoustic language-token detection; paper reports best-model FLEURS accuracy 64.5% (all 102 languages) / 80.3% (82 overlapping), not checkpoint-specific VoxLingua accuracy. Larger size alone does not imply better frozen-task quality. [source](https://cdn.openai.com/papers/whisper.pdf), [source](https://github.com/openai/whisper), [source](https://huggingface.co/openai/whisper-large-v3)

- lid:speechbrain/lang-id-voxlingua107-ecapa: QUALITY_LIKELY_PASS; error_rate=0.067 on VoxLingua107 development, 33-language full set. Exact official model card: 6.7% error, equivalent to 93.3% accuracy. All ten frozen language codes verified in pinned label_encoder.txt. This is below the old 98% local reference numerically but from a different evaluation set; no LID numeric threshold exists. [source](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)

- lid:facebook/mms-lid-256: QUALITY_LIKELY_PASS; accuracy=0.934 on VoxLingua107 evaluation, 33 languages. MMS paper Table 7, corresponding 256/1024-language system, not a cryptographic link to this HF checkpoint. Paper-system evidence only. Card/config verify ready speech classifier and all ten labels after fixed ISO mapping; semantic-only admission remains unchanged. [source](https://arxiv.org/html/2305.13516), [source](https://huggingface.co/facebook/mms-lid-256)

- lid:facebook/mms-lid-1024: QUALITY_LIKELY_PASS; accuracy=0.928 on VoxLingua107 evaluation, 33 languages. MMS paper Table 7, corresponding 256/1024-language system, not a cryptographic link to this HF checkpoint. Paper-system evidence only. Card/config verify ready speech classifier and all ten labels after fixed ISO mapping; semantic-only admission remains unchanged. [source](https://arxiv.org/html/2305.13516), [source](https://huggingface.co/facebook/mms-lid-1024)

- text:philschmid/roberta-large-sst2: QUALITY_LIKELY_PASS; accuracy=0.9644495412844036 on GLUE SST-2 evaluation set. Exact model-card model-index; different 872-sample evaluation from frozen 200. Explicit negative/positive id2label. [source](https://huggingface.co/philschmid/roberta-large-sst2)

- text:philschmid/bert-large-cased-whole-word-masking-sst2: QUALITY_LIKELY_PASS; accuracy=0.9438073394495413 on GLUE SST-2 evaluation set. Exact model-card model-index; cased whole-word-masking BERT fine-tuned for the same binary task; not a local guarantee. [source](https://huggingface.co/philschmid/bert-large-cased-whole-word-masking-sst2)

- text:textattack/facebook-bart-large-SST-2: QUALITY_LIKELY_PASS; accuracy=0.953 on SST-2 evaluation reported in paper. Published paper supplement Table 1 explicitly names this checkpoint. HF README absent; config has trained BartForSequenceClassification and finetuning_task sst-2. Classification logits, not text generation. [source](https://proceedings.mlr.press/v139/zanella-beguelin21a/zanella-beguelin21a-supp.pdf), [source](https://huggingface.co/textattack/facebook-bart-large-SST-2)

- text:textattack/xlnet-base-cased-SST-2: QUALITY_LIKELY_PASS; accuracy=0.944 on SST-2 evaluation reported in paper. Published supplement Table 1 explicitly names this checkpoint. HF README absent; config confirms XLNetForSequenceClassification / sst-2. SentencePiece dependency must be prepared before measurement. [source](https://proceedings.mlr.press/v139/zanella-beguelin21a/zanella-beguelin21a-supp.pdf), [source](https://huggingface.co/textattack/xlnet-base-cased-SST-2)

- text:textattack/bert-base-uncased-SST-2: QUALITY_UNCERTAIN; accuracy=0.924 on SST-2 evaluation reported in paper. Paper reports .924; TextAttack documentation 806/872 = .9243, slightly below .925. Different frozen subset may pass or fail; not chosen ahead of likely-pass checkpoints. Missing HF README is recorded. [source](https://proceedings.mlr.press/v139/zanella-beguelin21a/zanella-beguelin21a-supp.pdf), [source](https://textattack.readthedocs.io/en/latest/3recipes/models.html), [source](https://huggingface.co/textattack/bert-base-uncased-SST-2)

- text:vroomhf/deberta-v3-base-finetuned-sst2: QUALITY_UNCERTAIN; accuracy=0.9599 on Unspecified evaluation dataset in generated card. Checkpoint name suggests SST-2 but card explicitly says unknown dataset and generic labels. Do not treat its .9599 as a verified SST-2 result or transfer DeBERTa family GLUE scores to this checkpoint. [source](https://huggingface.co/vroomhf/deberta-v3-base-finetuned-sst2)


## Retrieval limitations

The TextAttack ReadTheDocs page was readable with the web tool, but raw urllib archival returned HTTP403; supplemental_sources.json records that failure. The primary paper supplement was archived successfully and supplies the exact BERT/XLNet/BART checkpoint-quality mapping. No missing result was invented.


## Reproducible local paths

- asr / `openai/whisper-medium@abdf7c39ab9d0397620ccaea8974cc764cd0953e`: AutoProcessor + AutoModelForSpeechSeq2Seq.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); original prepare_inputs + generate_transcription(language=en, task=transcribe, do_sample=False); frozen long-form timestamps for >30s. [pinned files](https://huggingface.co/openai/whisper-medium/tree/abdf7c39ab9d0397620ccaea8974cc764cd0953e)

- asr / `openai/whisper-medium.en@2e98eb6279edf5095af0c8dedb36bdec0acd172b`: AutoProcessor + AutoModelForSpeechSeq2Seq.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); original prepare_inputs + generate_transcription(language=en, task=transcribe, do_sample=False); frozen long-form timestamps for >30s. [pinned files](https://huggingface.co/openai/whisper-medium.en/tree/2e98eb6279edf5095af0c8dedb36bdec0acd172b)

- asr / `openai/whisper-large-v2@ae4642769ce2ad8fc292556ccea8e901f1530655`: AutoProcessor + AutoModelForSpeechSeq2Seq.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); original prepare_inputs + generate_transcription(language=en, task=transcribe, do_sample=False); frozen long-form timestamps for >30s. [pinned files](https://huggingface.co/openai/whisper-large-v2/tree/ae4642769ce2ad8fc292556ccea8e901f1530655)

- asr / `openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1`: AutoProcessor + AutoModelForSpeechSeq2Seq.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); original prepare_inputs + generate_transcription(language=en, task=transcribe, do_sample=False); frozen long-form timestamps for >30s. [pinned files](https://huggingface.co/openai/whisper-large-v3/tree/06f233fe06e710322aca913c1bc4249a0d71fce1)

- asr / `nvidia/parakeet-tdt-0.6b-v2@ae9ad07059c7c739ffaf932226a8fe64ae2620b0`: nemo.collections.asr.models.ASRModel.restore_from(LOCAL_PINNED_NEMO).eval().float().to(cuda); transcribe(local_audio_paths, batch_size=1), greedy TDT, no timestamps/external LM; normalize frozen ASR text. [pinned files](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/tree/ae9ad07059c7c739ffaf932226a8fe64ae2620b0)

- asr / `facebook/wav2vec2-large-960h-lv60-self@54074b1c16f4de6a5ad59affb4caa8f2ea03a119`: AutoProcessor + AutoModelForCTC.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); full mono 16k input; logits.argmax(-1), processor.batch_decode; frozen ASR normalization. [pinned files](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self/tree/54074b1c16f4de6a5ad59affb4caa8f2ea03a119)

- lid / `openai/whisper-medium@abdf7c39ab9d0397620ccaea8974cc764cd0953e`: AutoProcessor + AutoModelForSpeechSeq2Seq.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); model.detect_language with language=None and no forced language; ALL native language tokens; no transcript/text-LID. [pinned files](https://huggingface.co/openai/whisper-medium/tree/abdf7c39ab9d0397620ccaea8974cc764cd0953e)

- lid / `openai/whisper-large-v2@ae4642769ce2ad8fc292556ccea8e901f1530655`: AutoProcessor + AutoModelForSpeechSeq2Seq.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); model.detect_language with language=None and no forced language; ALL native language tokens; no transcript/text-LID. [pinned files](https://huggingface.co/openai/whisper-large-v2/tree/ae4642769ce2ad8fc292556ccea8e901f1530655)

- lid / `openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1`: AutoProcessor + AutoModelForSpeechSeq2Seq.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); model.detect_language with language=None and no forced language; ALL native language tokens; no transcript/text-LID. [pinned files](https://huggingface.co/openai/whisper-large-v3/tree/06f233fe06e710322aca913c1bc4249a0d71fce1)

- lid / `speechbrain/lang-id-voxlingua107-ecapa@0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`: speechbrain.inference.classifiers.EncoderClassifier.from_hparams(source=LOCAL_PINNED_DIR, savedir=LOCAL_CACHE, overrides={'pretrained_path': LOCAL_PINNED_DIR}, run_opts={'device':'cuda'}).classify_batch(full_mono_16k_waveform); decode top-1 label prefix before ':'; no online fetch. [pinned files](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9)

- lid / `facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a`: AutoFeatureExtractor + AutoModelForAudioClassification.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); raw mono 16k full waveform; logits.argmax(-1) over ALL classes; id2label and frozen ISO mapping. [pinned files](https://huggingface.co/facebook/mms-lid-256/tree/edc73fd00996e671dfc59d16436a29b12b10588a)

- lid / `facebook/mms-lid-1024@6216408c37833201358b509224b019312577dd70`: AutoFeatureExtractor + AutoModelForAudioClassification.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); raw mono 16k full waveform; logits.argmax(-1) over ALL classes; id2label and frozen ISO mapping. [pinned files](https://huggingface.co/facebook/mms-lid-1024/tree/6216408c37833201358b509224b019312577dd70)

- text / `philschmid/roberta-large-sst2@7d2599d698b7a805b6831e15e830e60a0b07bdb4`: AutoTokenizer + AutoModelForSequenceClassification.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); one sentence, truncation=True, logits.argmax(-1); frozen 0=negative,1=positive; eval/inference mode; no generation. [pinned files](https://huggingface.co/philschmid/roberta-large-sst2/tree/7d2599d698b7a805b6831e15e830e60a0b07bdb4)

- text / `philschmid/bert-large-cased-whole-word-masking-sst2@5c930994422145867b5768ba7113a2aa330b19d2`: AutoTokenizer + AutoModelForSequenceClassification.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); one sentence, truncation=True, logits.argmax(-1); frozen 0=negative,1=positive; eval/inference mode; no generation. [pinned files](https://huggingface.co/philschmid/bert-large-cased-whole-word-masking-sst2/tree/5c930994422145867b5768ba7113a2aa330b19d2)

- text / `textattack/facebook-bart-large-SST-2@8a4baa9662afa056a97a2e9441ae665a06504979`: AutoTokenizer + AutoModelForSequenceClassification.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); one sentence, truncation=True, logits.argmax(-1); frozen 0=negative,1=positive; eval/inference mode; no generation. [pinned files](https://huggingface.co/textattack/facebook-bart-large-SST-2/tree/8a4baa9662afa056a97a2e9441ae665a06504979)

- text / `textattack/xlnet-base-cased-SST-2@9ceeb077dcd5cf5ae790572b2bd6aec755a263be`: AutoTokenizer + AutoModelForSequenceClassification.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); one sentence, truncation=True, logits.argmax(-1); frozen 0=negative,1=positive; eval/inference mode; no generation. [pinned files](https://huggingface.co/textattack/xlnet-base-cased-SST-2/tree/9ceeb077dcd5cf5ae790572b2bd6aec755a263be)

- text / `textattack/bert-base-uncased-SST-2@95f0f6f859b35c8ff0863ae3cd4e2dbc702c0ae2`: AutoTokenizer + AutoModelForSequenceClassification.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); one sentence, truncation=True, logits.argmax(-1); frozen 0=negative,1=positive; eval/inference mode; no generation. [pinned files](https://huggingface.co/textattack/bert-base-uncased-SST-2/tree/95f0f6f859b35c8ff0863ae3cd4e2dbc702c0ae2)

- text / `vroomhf/deberta-v3-base-finetuned-sst2@297a5dfce6be14d776bd06e98e2b98ab3326c627`: AutoTokenizer + AutoModelForSequenceClassification.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); one sentence, truncation=True, logits.argmax(-1); frozen 0=negative,1=positive; eval/inference mode; no generation. [pinned files](https://huggingface.co/vroomhf/deberta-v3-base-finetuned-sst2/tree/297a5dfce6be14d776bd06e98e2b98ab3326c627)
