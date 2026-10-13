"""Build the Module 13A discovery/freeze from archived public and local metadata.

CPU only: stdlib, no model imports, no inference, no benchmark launcher.
"""
import csv
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SRC = OUT / 'sources'
if (OUT / 'module13a_gate.json').exists():
    raise RuntimeError('Module13A is frozen; do not overwrite it. Any amendment needs a separate versioned namespace.')

def read(p):
    return json.loads(Path(p).read_text())

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def ref(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)

def source(model, file='README.md'):
    return SRC / (model.replace('/', '--') + '.' + file)

STAMP = datetime.datetime.now(datetime.timezone.utc).isoformat()
MANIFEST = read(OUT / 'source_manifest.json')
REMOTE = {r['model_id']: r for r in MANIFEST['records']}
POLICY_PATH = ROOT / 'results/module8b_quality_pool_finalization/quality_policy.json'
POLICY = {r['task']: r for r in read(POLICY_PATH)['tasks']}
TASK_PATH = ROOT / 'results/candidate_pool_v2/task_definitions.json'
TASKS = read(TASK_PATH)
POOL_PATH = ROOT / 'results/module8b_quality_pool_finalization/candidate_pool_v2_1.json'
MODELS = {m['id']: m for m in read(POOL_PATH)['models']}
SERVICE_PATH = ROOT / 'results/module8b_quality_pool_finalization/service_demand_v2_1.json'
SERVICE = read(SERVICE_PATH)['models']
CAL_PATH = ROOT / 'results/module11_replica_and_cost/efficiency_optimal_baseline/replica_scaling.json'
CAL = read(CAL_PATH)
ADMISSION_PATH = ROOT / 'results/module8b_quality_pool_finalization/quality_admission_matrix.json'
ADMISSION = {(r['task'], r['model']): r for r in read(ADMISSION_PATH)['rows']}
TARGET_PATH = ROOT / 'results/module12_specialist_sensitivity/real_model_search_targets.json'
TARGETS = read(TARGET_PATH)['tasks']
M12RULE = ROOT / 'results/module12_specialist_sensitivity/specialist_selection_rule.json'
BASE_NAMES = {'asr': ['distil_whisper', 'whisper-small', 'whisper_turbo'],
              'lid': ['whisper-small', 'whisper_turbo'], 'text': ['distilbert', 'roberta_sst2']}
REFERENCE = {'asr': 'distil_whisper', 'lid': 'whisper-small', 'text': 'distilbert'}
BASE = {}
for task, names in BASE_NAMES.items():
    m0 = REFERENCE[task]
    baseline = []
    for name in names:
        m = MODELS[name]
        baseline.append({**m, 'task': task,
                         'quality': ADMISSION[task, name]['measured_quality'],
                         'quality_admitted': ADMISSION[task, name]['quality_admissible'],
                         'measured_service_ms': SERVICE[name]['service_time_ms'][task],
                         'measured_idle_memory_mib': CAL['memory_table_mib'][name]['1'],
                         'measured_gamma_to_reference': SERVICE[name]['service_time_ms'][task] / SERVICE[m0]['service_time_ms'][task],
                         'measured_mu_to_reference': CAL['memory_table_mib'][name]['1'] / CAL['memory_table_mib'][m0]['1'],
                         'service_sources': SERVICE[name]['service_time_sources'][task]})
    BASE[task] = {'ratio_reference': m0, 'models': baseline,
                  'reference_service_ms': SERVICE[m0]['service_time_ms'][task],
                  'reference_idle_memory_mib': CAL['memory_table_mib'][m0]['1'],
                  'reference_checkpoint': MODELS[m0]['source'],
                  'reference_revision': MODELS[m0]['source_revision']}
write('baseline_references.json', {'regime': 'EFFICIENCY_OPTIMAL_SPECIALIST_REGIME',
                                 'preserved_unchanged': True, 'tasks': BASE,
                                 'sources': [ref(p) for p in [POOL_PATH, SERVICE_PATH, CAL_PATH, ADMISSION_PATH]]})

# Claims are attributed to exact public checkpoints where possible. Family/paper
# results are explicitly labelled, never promoted to this local 200-sample score.
WHISPER_PAPER = 'https://cdn.openai.com/papers/whisper.pdf'
MMS_PAPER = 'https://arxiv.org/html/2305.13516'
TEXT_PAPER = 'https://proceedings.mlr.press/v139/zanella-beguelin21a/zanella-beguelin21a-supp.pdf'
EVIDENCE = {}

def ev(model, task, status, metric=None, value=None, dataset=None, urls=None, note=''):
    key = task + ':' + model
    EVIDENCE[key] = {'model_id': model, 'task': task, 'preliminary_quality': status,
                     'published_metric': metric, 'published_value': value,
                     'published_dataset': dataset,
                     'source_urls': urls or ['https://huggingface.co/' + model],
                     'exact_revision': REMOTE[model]['revision'],
                     'local_frozen_quality': None, 'admission_measured': False,
                     'frozen_admission': POLICY[task], 'limitations': note}

for model, wer in [('openai/whisper-medium', .029), ('openai/whisper-medium.en', .031),
                   ('openai/whisper-large-v2', .027)]:
    ev(model, 'asr', 'QUALITY_LIKELY_PASS', 'wer', wer, 'LibriSpeech test-clean, full set',
       [WHISPER_PAPER, 'https://huggingface.co/' + model],
       'Whisper paper Appendix D.1.1 Table 8, greedy decoding. Different normalizer and sample set; not local admission.')
ev('openai/whisper-large-v3', 'asr', 'QUALITY_LIKELY_PASS', urls=['https://huggingface.co/openai/whisper-large-v3'],
   note='Official card reports 10–20% lower errors than large-v2 across evaluated languages; family-level inference only. No verified exact test-clean WER for this checkpoint in this audit; not a substitute for local WER.')
ev('nvidia/parakeet-tdt-0.6b-v2', 'asr', 'QUALITY_LIKELY_PASS', 'wer', .0169,
   'LibriSpeech test-clean, full set', note='Official card: greedy TDT, no external language model; test-other WER 3.19%. Different scorer/precision and sample set remain possible.')
ev('facebook/wav2vec2-large-960h-lv60-self', 'asr', 'QUALITY_LIKELY_PASS', 'wer', .019,
   'LibriSpeech test-clean, full set', note='Official card greedy CTC argmax example, no external LM; test-other 3.9%. Dataset-specific fine-tuning is pre-existing; no user fine-tuning needed.')
for model in ['openai/whisper-medium', 'openai/whisper-large-v2', 'openai/whisper-large-v3']:
    ev(model, 'lid', 'QUALITY_LIKELY_PASS', urls=[WHISPER_PAPER, 'https://github.com/openai/whisper', 'https://huggingface.co/' + model],
       note='LIKELY_PASS means existing semantic-only admission, not a numeric accuracy promise. Whisper family supports acoustic language-token detection; paper reports best-model FLEURS accuracy 64.5% (all 102 languages) / 80.3% (82 overlapping), not checkpoint-specific VoxLingua accuracy. Larger size alone does not imply better frozen-task quality.')
ev('speechbrain/lang-id-voxlingua107-ecapa', 'lid', 'QUALITY_LIKELY_PASS', 'error_rate', .067,
   'VoxLingua107 development, 33-language full set',
   note='Exact official model card: 6.7% error, equivalent to 93.3% accuracy. All ten frozen language codes verified in pinned label_encoder.txt. This is below the old 98% local reference numerically but from a different evaluation set; no LID numeric threshold exists.')
for model, acc in [('facebook/mms-lid-256', .934), ('facebook/mms-lid-1024', .928)]:
    ev(model, 'lid', 'QUALITY_LIKELY_PASS', 'accuracy', acc, 'VoxLingua107 evaluation, 33 languages',
       [MMS_PAPER, 'https://huggingface.co/' + model],
       'MMS paper Table 7, corresponding 256/1024-language system, not a cryptographic link to this HF checkpoint. Paper-system evidence only. Card/config verify ready speech classifier and all ten labels after fixed ISO mapping; semantic-only admission remains unchanged.')
ev('philschmid/roberta-large-sst2', 'text', 'QUALITY_LIKELY_PASS', 'accuracy', .9644495412844036,
   'GLUE SST-2 evaluation set', note='Exact model-card model-index; different 872-sample evaluation from frozen 200. Explicit negative/positive id2label.')
ev('philschmid/bert-large-cased-whole-word-masking-sst2', 'text', 'QUALITY_LIKELY_PASS', 'accuracy', .9438073394495413,
   'GLUE SST-2 evaluation set', note='Exact model-card model-index; cased whole-word-masking BERT fine-tuned for the same binary task; not a local guarantee.')
ev('textattack/facebook-bart-large-SST-2', 'text', 'QUALITY_LIKELY_PASS', 'accuracy', .953,
   'SST-2 evaluation reported in paper', [TEXT_PAPER, 'https://huggingface.co/textattack/facebook-bart-large-SST-2'],
   'Published paper supplement Table 1 explicitly names this checkpoint. HF README absent; config has trained BartForSequenceClassification and finetuning_task sst-2. Classification logits, not text generation.')
ev('textattack/xlnet-base-cased-SST-2', 'text', 'QUALITY_LIKELY_PASS', 'accuracy', .944,
   'SST-2 evaluation reported in paper', [TEXT_PAPER, 'https://huggingface.co/textattack/xlnet-base-cased-SST-2'],
   'Published supplement Table 1 explicitly names this checkpoint. HF README absent; config confirms XLNetForSequenceClassification / sst-2. SentencePiece dependency must be prepared before measurement.')
ev('textattack/bert-base-uncased-SST-2', 'text', 'QUALITY_UNCERTAIN', 'accuracy', .924,
   'SST-2 evaluation reported in paper', [TEXT_PAPER, 'https://textattack.readthedocs.io/en/latest/3recipes/models.html', 'https://huggingface.co/textattack/bert-base-uncased-SST-2'],
   'Paper reports .924; TextAttack documentation 806/872 = .9243, slightly below .925. Different frozen subset may pass or fail; not chosen ahead of likely-pass checkpoints. Missing HF README is recorded.')
ev('vroomhf/deberta-v3-base-finetuned-sst2', 'text', 'QUALITY_UNCERTAIN', 'accuracy', .9599,
   'Unspecified evaluation dataset in generated card',
   note='Checkpoint name suggests SST-2 but card explicitly says unknown dataset and generic labels. Do not treat its .9599 as a verified SST-2 result or transfer DeBERTa family GLUE scores to this checkpoint.')
write('quality_evidence.json', {'evidence_kind': 'PUBLIC_PRELIMINARY_NOT_LOCAL_ADMISSION',
                              'published_WER_scale': 'fraction, not percent',
                              'LID_likely_pass_semantics': 'No numeric LID SLO; accuracy still measured and used in Pareto selection.',
                              'entries': EVIDENCE})

UNIVERSE = {
 'asr': ['openai/whisper-medium', 'openai/whisper-medium.en', 'openai/whisper-large-v2',
         'openai/whisper-large-v3', 'nvidia/parakeet-tdt-0.6b-v2', 'facebook/wav2vec2-large-960h-lv60-self'],
 'lid': ['openai/whisper-medium', 'openai/whisper-large-v2', 'openai/whisper-large-v3',
         'speechbrain/lang-id-voxlingua107-ecapa', 'facebook/mms-lid-256', 'facebook/mms-lid-1024'],
 'text': ['philschmid/roberta-large-sst2', 'philschmid/bert-large-cased-whole-word-masking-sst2',
          'textattack/facebook-bart-large-SST-2', 'textattack/xlnet-base-cased-SST-2',
          'textattack/bert-base-uncased-SST-2', 'vroomhf/deberta-v3-base-finetuned-sst2']}
SHORT = {
 'asr': ['facebook/wav2vec2-large-960h-lv60-self', 'openai/whisper-large-v3', 'openai/whisper-medium'],
 'lid': ['facebook/mms-lid-256', 'openai/whisper-medium', 'speechbrain/lang-id-voxlingua107-ecapa'],
 'text': ['philschmid/roberta-large-sst2', 'textattack/facebook-bart-large-SST-2', 'textattack/xlnet-base-cased-SST-2']}
EXCLUSION = {
 ('asr','openai/whisper-medium.en'): 'LEGAL_RETAINED: same medium architecture/size bucket; multilingual medium adds a shared ASR/LID operating point. Not rejected for latency.',
 ('asr','openai/whisper-large-v2'): 'LEGAL_RETAINED: large-v3 covers the same large family bucket with published generation improvement; large-v2 remains auditable, not classified dominated.',
 ('asr','nvidia/parakeet-tdt-0.6b-v2'): 'LEGAL_RETAINED: excellent published WER; NeMo is an additional environment/backend whereas CTC supplies alternate-family diversity on the frozen Transformers runtime. Not rejected for being fast.',
 ('lid','openai/whisper-large-v2'): 'LEGAL_RETAINED: medium represents same Whisper family; dedicated ECAPA/MMS add task-native architecture diversity. No larger/slower exclusion.',
 ('lid','openai/whisper-large-v3'): 'LEGAL_RETAINED: same Whisper family as medium; full large decoder memory does not establish a near-target LID latency.',
 ('lid','facebook/mms-lid-1024'): 'LEGAL_RETAINED: same 48-layer backbone as MMS-256 with slightly larger classifier; ten labels already covered by 256. Not declared dominated because exact quality/latency are unmeasured.',
 ('text','philschmid/bert-large-cased-whole-word-masking-sst2'): 'LEGAL_RETAINED: likely-pass, explicit labels. Shortlist covers a smaller base encoder, large RoBERTa encoder, and encoder-decoder classifier; BERT-large overlaps the large encoder size bucket. Not declared dominated.',
 ('text','textattack/bert-base-uncased-SST-2'): 'LEGAL_RETAINED: public task-compatible but quality uncertain around/below threshold; likely-pass candidates precede it.',
 ('text','vroomhf/deberta-v3-base-finetuned-sst2'): 'METADATA_UNCERTAIN: unspecified evaluation data and generic label semantics; quality/task evidence weaker than selected ready SST-2 checkpoints.'}

APPROX_PARAMS = {'facebook/wav2vec2-large-960h-lv60-self':315_000_000,
 'nvidia/parakeet-tdt-0.6b-v2':600_000_000,
 'philschmid/bert-large-cased-whole-word-masking-sst2':334_000_000,
 'philschmid/roberta-large-sst2':355_000_000,
 'textattack/bert-base-uncased-SST-2':110_000_000,
 'textattack/facebook-bart-large-SST-2':406_000_000,
 'textattack/xlnet-base-cased-SST-2':110_000_000}

def family(model):
    if model.startswith('openai/'): return 'Whisper'
    if 'mms-lid' in model: return 'MMS-Wav2Vec2 speech-LID'
    if 'speechbrain' in model: return 'ECAPA-TDNN'
    if 'parakeet' in model: return 'FastConformer-TDT'
    if 'wav2vec2' in model: return 'Wav2Vec2-CTC'
    if 'roberta' in model: return 'RoBERTa'
    if 'bart' in model: return 'BART sequence classifier'
    if 'xlnet' in model: return 'XLNet'
    if 'deberta' in model: return 'DeBERTa-v3'
    return 'BERT'

def files_for(model):
    api = read(source(model, 'api.json'))
    sib = api['siblings']
    safe = [r for r in sib if r['rfilename']=='model.safetensors']
    if safe: return safe
    names = {'parakeet-tdt-0.6b-v2.nemo', 'pytorch_model.bin'}
    if model.startswith('speechbrain/'):
        names = {'embedding_model.ckpt', 'classifier.ckpt', 'normalizer.ckpt'}
    return [r for r in sib if r['rfilename'] in names]

def local_path(model, task):
    if model.startswith('speechbrain/'):
        return "speechbrain.inference.classifiers.EncoderClassifier.from_hparams(source=LOCAL_PINNED_DIR, savedir=LOCAL_CACHE, overrides={'pretrained_path': LOCAL_PINNED_DIR}, run_opts={'device':'cuda'}).classify_batch(full_mono_16k_waveform); decode top-1 label prefix before ':'; no online fetch"
    if 'parakeet' in model:
        return 'nemo.collections.asr.models.ASRModel.restore_from(LOCAL_PINNED_NEMO).eval().float().to(cuda); transcribe(local_audio_paths, batch_size=1), greedy TDT, no timestamps/external LM; normalize frozen ASR text'
    if task=='asr' and 'wav2vec2' in model:
        return 'AutoProcessor + AutoModelForCTC.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); full mono 16k input; logits.argmax(-1), processor.batch_decode; frozen ASR normalization'
    if task=='asr':
        return 'AutoProcessor + AutoModelForSpeechSeq2Seq.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); original prepare_inputs + generate_transcription(language=en, task=transcribe, do_sample=False); frozen long-form timestamps for >30s'
    if task=='lid' and 'mms-lid' in model:
        return 'AutoFeatureExtractor + AutoModelForAudioClassification.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); raw mono 16k full waveform; logits.argmax(-1) over ALL classes; id2label and frozen ISO mapping'
    if task=='lid':
        return 'AutoProcessor + AutoModelForSpeechSeq2Seq.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); model.detect_language with language=None and no forced language; ALL native language tokens; no transcript/text-LID'
    return 'AutoTokenizer + AutoModelForSequenceClassification.from_pretrained(LOCAL_PINNED_DIR, local_files_only=True, trust_remote_code=False); one sentence, truncation=True, logits.argmax(-1); frozen 0=negative,1=positive; eval/inference mode; no generation'

ROWS = []
for task, ids in UNIVERSE.items():
    for model in ids:
        api = read(source(model, 'api.json'))
        config = read(source(model, 'config.json')) if source(model, 'config.json').exists() else {}
        weights = files_for(model)
        num = (api.get('safetensors') or {}).get('total')
        quality = EVIDENCE[task+':'+model]
        chosen = model in SHORT[task]
        language_check = 'English / 16kHz audio' if task=='asr' else 'English sentence / SST-2 binary sentiment' if task=='text' else 'All ten frozen spoken language codes; native full class space'
        if 'mms-lid' in model:
            native = set(config['id2label'].values())
            assert {'ara','deu','eng','spa','fas','fra','jpn','rus','tur','cmn'} <= native
        if task=='lid' and model.startswith('openai/'):
            g=read(source(model,'generation_config.json'))
            assert {f'<|{x}|>' for x in TASKS['lid']['label_space']} <= set(g['lang_to_id'])
        if task=='lid' and model.startswith('speechbrain/'):
            text=source(model,'label_encoder.txt').read_text()
            assert all("'" + x + ': ' in text for x in TASKS['lid']['label_space'])
        artifact_size = sum(r['size'] for r in weights)
        repo_revision = REMOTE[model]['revision']
        ROWS.append(dict(task=task,model_id=model,revision=repo_revision,family=family(model),
          parameter_count=num or APPROX_PARAMS.get(model),
          parameter_count_basis='HF safetensors metadata (serialized/tied-weight accounting)' if num else 'published approximate family/model count, not exact local count' if model in APPROX_PARAMS else 'UNKNOWN; not inferred from artifact bytes',
          weight_files=';'.join(r['rfilename'] for r in weights),
          artifact_weight_bytes=artifact_size,artifact_weight_gb=artifact_size/1e9,
          artifact_precision='float16' if model=='openai/whisper-large-v3' else 'float32 expected; confirm checkpoint tensors in 13B',
          expected_primary_precision='float32',
          license=api.get('cardData',{}).get('license') or 'UNSPECIFIED_IN_CARD',
          input_compatibility=language_check,
          legal_status='METADATA_TASK_QUALITY_UNCERTAIN' if 'vroomhf' in model else 'LEGAL_PUBLIC_LOCAL_READY_MODEL',
          public_access_verified=REMOTE[model]['access'],gated=REMOTE[model].get('gated',False),
          requires_commercial_api=False,requires_online_inference=False,requires_user_finetuning=False,
          rtx4090_feasibility='METADATA_LIKELY_BATCH1_STANDARD_24GB; no GPU validation; artifact fits with plausible inference overhead; actual host advertises 49140 MiB',
          preliminary_quality=quality['preliminary_quality'],published_metric=quality['published_metric'],
          published_quality=quality['published_value'],published_dataset=quality['published_dataset'],
          quality_evidence_id=task+':'+model,
          published_latency_evidence='OpenAI README: A100 English transcription relative speed ~medium 2x / large 1x; backend/duration/precision differs; never transferred to LID' if task=='asr' and model.startswith('openai/') else 'Card RTFx 3380 at batch128, different GPU and throughput statistic, unusable as local batch1 latency' if 'parakeet' in model else 'NO_COMPARABLE_PUBLISHED_BATCH1_LOCAL_SERVICE_LATENCY',
          local_latency_ms=None,local_idle_memory_mib=None,local_peak_memory_mib=None,
          reference_model=REFERENCE[task],reference_latency_ms=BASE[task]['reference_service_ms'],
          reference_idle_memory_mib=BASE[task]['reference_idle_memory_mib'],
          gamma_est=None,mu_est=None,
          gamma_first_split_target='UNKNOWN',gamma_stronger_transition_target='UNKNOWN',mu_target='UNKNOWN',
          size_over_reference_resident_bytes=artifact_size/(BASE[task]['reference_idle_memory_mib']*1048576),
          size_ratio_warning='DISK_BYTES/MEASURED_IDLE_VRAM_ONLY; NOT mu_est; precision/format/context/allocator overhead differ; not used for target classification',
          preliminary_pareto_status='PRELIMINARY_UNRESOLVED',preliminary_dominated=False,
          local_inference_path=local_path(model,task),
          implementation_status='DOCUMENTED_LOCAL_PATH; adapter not executed in 13A',
          shortlist=chosen,
          shortlist_reason='Likely-pass original admission; reproducible pinned local path; family/architecture and size diversity; target proximity UNKNOWN' if chosen else EXCLUSION[task,model],
          model_card_url='https://huggingface.co/'+model+'/tree/'+repo_revision,
          source_urls=';'.join(quality['source_urls'])))
assert len(ROWS)==18
for name, rows in [('candidate_universe.csv',ROWS)] + [(t+'_candidates.csv',[r for r in ROWS if r['task']==t]) for t in UNIVERSE]:
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ROWS[0]));w.writeheader();w.writerows(rows)

SERVICE_POOL = ROOT/'benchmark/task_pools/mixed_formal_1200_6task.jsonl'
pool=[json.loads(s) for s in SERVICE_POOL.read_text().splitlines()]
datasets={}
input_refs={}
for task in UNIVERSE:
    definition=TASKS[task]
    path=Path(definition['evaluation_manifest'])
    assert ref(path)['sha256']==definition['manifest_sha256']
    quality_rows=[json.loads(s) for s in path.read_text().splitlines()]
    latency_rows=[r for r in pool if r['task_type']==task][:30]
    assert len(quality_rows)==200 and len(latency_rows)==30
    raw_source = SERVICE[REFERENCE[task]]['service_time_sources'][task]
    original_raw=(raw_source['repetitions'][0]['raw']['path'] if 'repetitions' in raw_source else raw_source['source']['summaries'][0]['path'].removesuffix('.summary.json'))
    measured_rows=[json.loads(s) for s in Path(original_raw).read_text().splitlines()]
    assert [r['pool_id'] for r in measured_rows]==[r['pool_id'] for r in latency_rows]
    datasets[task]={'definition':definition['definition'],'input_type':'text' if task=='text' else 'audio',
                    'dataset':definition['dataset'],'split':definition['split'],
                    'quality_manifest':ref(path),'quality_count':200,
                    'quality_sample_ids_in_order':[r['id'] for r in quality_rows],
                    'quality_policy':POLICY[task],
                    'latency_source_pool':ref(SERVICE_POOL),'latency_count_per_repeat':30,
                    'latency_pool_ids_in_order':[r['pool_id'] for r in latency_rows],
                    'latency_sample_ids_in_order':[r['sample']['id'] for r in latency_rows],
                    'latency_sample_match_with_reference_raw':ref(original_raw),
                    'latency_duration_note':'Same frozen first-30 pool rows as measured denominator; do not substitute the separate historical [4,6)s subset.'}
    (OUT/('latency_samples_'+task+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in latency_rows))
    for sample in quality_rows+[r['sample'] for r in latency_rows]:
        if task!='text':input_refs[sample['input']]=ref(sample['input'])
write('dataset_input_hashes.json',{'file_count':len(input_refs),'files':list(input_refs.values()),
                                 'text_inputs':'Contained in hashed manifests / service pool; labels never model inputs.'})

runtime={'transformers':{'python':'3.11','torch':'2.14.0','transformers':'5.17.0','tokenizers':'0.23.2',
                         'numpy':'2.4.6','soundfile':'0.14.0','safetensors':'0.8.0',
                         'sentencepiece':'0.2.1 for XLNet (prepare isolated clone before inference)',
                         'current_env':'envs/specialist_env; keep immutable; clone for additions'},
         'speechbrain':{'python':'3.11','torch':'2.5.1+cu124','torchaudio':'2.5.1+cu124',
                        'speechbrain':'1.0.3','numpy':'1.26.4','soundfile':'0.12.1',
                        'hyperpyyaml':'1.2.2','huggingface-hub':'0.27.1',
                        'isolation':'new env; native FP32; no changes to Specialist baseline env',
                        'verification':'Versions/planned offline API fixed; compatibility not runtime-tested in 13A'},
         'NeMo_unshortlisted':{'nemo_toolkit':'2.2.1','path':'ASRModel.restore_from pinned .nemo; no service/API required',
                               'verification':'Published NeMo2.2 path; not installed or selected for 13B'}}
write('runtime_metadata_freeze.json',runtime)

short_entries=[]
order=[]
for task in ['asr','lid','text']:
    for model in sorted(SHORT[task]):
        row=next(r for r in ROWS if r['task']==task and r['model_id']==model)
        order.append({'order':len(order)+1,'task':task,'model_id':model,'revision':row['revision']})
        short_entries.append({'task':task,'model_id':model,'revision':row['revision'],
                              'primary_precision':'float32','dataset':datasets[task],
                              'quality_evidence_id':row['quality_evidence_id'],
                              'local_inference_path':row['local_inference_path'],
                              'weight_files':row['weight_files'].split(';'),
                              'runtime':'speechbrain' if model.startswith('speechbrain/') else 'transformers',
                              'selection_reason':row['shortlist_reason'],
                              'gamma_target_status':'UNKNOWN','mu_target_status':'UNKNOWN'})
write('measurement_shortlist.json',{'module':'13A','frozen':True,'frozen_at_utc':STAMP,
   'freeze_before_new_candidate_GPU_results':True,'new_candidate_GPU_results_seen':0,
   'new_candidate_weight_downloads':0,'regime_proposal':'BALANCED_SPECIALIST_REGIME',
   'regime_status':'PROPOSED_NOT_ADMITTED_NOT_OFFICIAL',
   'baseline_preserved':'EFFICIENCY_OPTIMAL_SPECIALIST_REGIME',
   'candidate_count_per_task':{t:len(SHORT[t]) for t in SHORT},
   'unique_checkpoints':len({r['model_id'] for r in short_entries}),
   'task_model_paths':len(short_entries),'entries':short_entries,
   'measurement_order':order,
   'selection_priority':['preliminary quality likely pass','deployment reproducibility',
      'diversity of real operating points','proximity to Module12 transition region',
      'higher reported quality','deterministic model ID'],
   'priority_application':{
    'quality':'All nine selected paths likely pass original admission; uncertain Text checkpoints do not displace them.',
    'reproducibility':'ASR Transformers checkpoints precede additional NeMo environment; SpeechBrain chosen for well-documented native speech-LID API.',
    'diversity':'ASR: CTC + medium/full large Whisper. LID: native ECAPA + MMS raw-waveform classifier + medium Whisper. Text: base XLNet + large RoBERTa + BART classifier.',
    'proximity':'All new local gamma/mu UNKNOWN; neither claimed near-target nor rejected based on guessed slowness.',
    'reported_quality':'large-v3 generation improvement selects the full-large representative; all shortlisted Text checkpoints have published same-task evidence. Cross-dataset values are not directly ordered.',
    'deterministic_ID':'Fixed exact IDs/revisions and lexicographic measurement order; no post-measurement replacement.'},
   'not_selected_but_retained':[{ 'task':t,'model_id':m,'reason':reason} for (t,m),reason in EXCLUSION.items()],
   'no_automatic_backfill':True,'no_final_V3_outcome_used':True})

# Inherit the already frozen unweighted log-space distance. Archive actual numeric
# target segments now; each is a search set, not a prediction for a real model.
regions={}
for task in SHORT:
    rectangles=[]
    for b in TARGETS[task]['context_brackets']:
        if not b.get('supported'):continue
        item={'gamma':[b['low'],b['high']] if b['axis']=='latency' else [1.,1.],
              'mu':[1.,1.] if b['axis']=='latency' else [b['low'],b['high']],
              'origin':'Module12 reported per-task '+b['axis']+' bracket',
              'workload':b['workload'],'budget':b['budget']}
        rectangles.append(item)
    rectangles += [
     {'gamma':[1.2125,1.25],'mu':[1.,1.],'origin':'Module12 global first routing-split guidance; whole-pool scaling, not task-local evidence'},
     {'gamma':[1.9375,2.],'mu':[1.,1.],'origin':'Module12 global first replica-reduction guidance; whole-pool scaling'},
     {'gamma':[1.,1.],'mu':[1.,1.05],'origin':'Module12 global memory replica-reduction guidance; whole-pool scaling'}]
    regions[task]=rectangles
write('specialist_selection_rule.json',{'module':'13A','frozen_at_utc':STAMP,'frozen':True,
   'inherited_rule_source':ref(M12RULE),'target_source':ref(TARGET_PATH),
   'search_guidance_only':True,'selection_order':[
     'Measure the complete frozen shortlist with unchanged task quality admission; no outcome-dependent candidate replacement.',
     'Include all measured quality-admitted existing Specialists and shortlisted candidates in the task Q-max/latency-min/idle-memory-min universe.',
     'Construct the real three-objective Pareto frontier. Dominance requires no worse on every objective and strictly better on at least one.',
     'Choose the frontier member nearest the frozen union of Module12 search target segments.',
     'Tie-break: higher frozen-task quality utility, then lower measured idle memory, then lexicographically smaller full model ID@revision.'],
   'quality_utility':{'asr':'-corpus_WER','lid':'accuracy','text':'accuracy'},
   'quality_policy':ref(POLICY_PATH),'no_numeric_LID_threshold_added':True,
   'primary_latency':'sample-count-weighted mean worker service ms of all 3x30 requests; retain all valid repetitions',
   'primary_memory':'mean single-worker post-warmup NVML process idle MiB; baseline M1 definition; peak reported separately',
   'ratio_denominators':{t:{'model':REFERENCE[t],'latency_ms':BASE[t]['reference_service_ms'],
                           'idle_memory_mib':BASE[t]['reference_idle_memory_mib']} for t in SHORT},
   'distance':'unweighted L-infinity in (log(gamma), log(mu)) to closed target set; minimum over frozen rectangles; no fitted weights',
   'distance_formula':'min_R max(max(log(R.gamma_lo)-log(gamma),0,log(gamma)-log(R.gamma_hi)), max(log(R.mu_lo)-log(mu),0,log(mu)-log(R.mu_hi)))',
   'closed_set_note':'Open lower bracket endpoints closed for distance only (infimum unchanged); does not turn a baseline endpoint into an observed transition.',
   'target_rectangles_by_task':regions,
   'tie_tolerance':{'distance_log_units':1e-12,'quality':'exact full-precision score; no rounding','memory':'full measured value'},
   'failure_policy':'Retain failed or OOM paths in results; do not synthesize or backfill. Complete the frozen quality evaluations and evaluate every admitted shortlisted path. Failed measurements leave the evidence partial; a sole quality-admitted candidate may be selected by the same rule after the full shortlist was evaluated. No admitted new frontier winner => BASELINE_RETAINED, not a manufactured alternative.',
   'missing_evidence_policy':'No candidate with missing valid latency/idle memory may be selected; failed baseline control blocks comparability. No estimates admitted.',
   'baseline_frontier_policy':'If an existing baseline wins this rule, retain it and report BASELINE_RETAINED; never discard a dominating baseline to create a transition.',
   'cross_task_identity_policy':'ASR/LID can share one checkpoint/process identity; union selected IDs and avoid double-counting resident memory. Admit only frozen measured task-model paths; no unmeasured capability transfer.',
   'replica_policy':'13B standalone measurements do not create real multi-replica g or memory curves; unmeasured counts stay unknown.',
   'freeze_selection_before_final_V3_outcome':True,'Generalist_outcome_used':False,
   'selected_to_make_Generalist_win':'NO','synthetic_slowdown_or_memory_scaling':False})

write('measurement_protocol_freeze.json',{'module':'13A','frozen':True,'frozen_at_utc':STAMP,
   'execution_boundary':'Specification only; 13B not authorized/executed by Module13A completion.',
   'measurement_shortlist':ref(OUT/'measurement_shortlist.json'),
   'selection_rule':ref(OUT/'specialist_selection_rule.json'),
   'task_definition':ref(TASK_PATH),'quality_policy':ref(POLICY_PATH),
   'datasets':datasets,'dataset_input_hashes':ref(OUT/'dataset_input_hashes.json'),
   'runtime_freeze':ref(OUT/'runtime_metadata_freeze.json'),
   'model_snapshot':{'revision':'exact SHA from shortlist, never main','download':'13B only; weight + tokenizer/feature extractor/config/generation config; verify LFS sha256 from archived API; no TensorFlow/Flax duplicate weights',
                     'offline':'local_files_only; no API or runtime remote fetch; SpeechBrain pretrained_path override points to pinned local directory'},
   'hardware':{'family':'RTX4090','expected_feasibility':'standard24GB batch1 inferred only',
               'historical_host_total_mib':49140,'source':ref(ROOT/'results/module11_replica_and_cost/efficiency_optimal_specialist_regime_frozen.json'),
               '13B_record':'Actual UUID/name/VRAM/driver/clocks/power and package versions before measurement; do not claim a standard24GB run if host remains modified49140MiB'},
   'primary_operating_point':{'batch_size':1,'replicas':1,'precision':'float32 explicit, including float16-stored large-v3 cast to float32',
      'autocast':False,'quantization':False,'compile':False,'external_LM':False,
      'attention':'Native default of each model class in the pinned Transformers runtime; use SDPA where the library selects it, eager only where native support requires it. Record and lock effective implementation during CPU preflight before any GPU performance result; no result-driven backend search.',
      'TF32':'Preserve original PyTorch runtime defaults and record effective matmul/cuDNN settings before GPU performance. Reference controls and candidates in the same runtime use identical policy; do not change settings to target a gamma. Old frozen denominators never rewritten.',
      'threads':{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','torch_num_threads':1},
      'inference':'model.eval and inference_mode; no gradients; no deliberate slowdown or memory padding'},
   'quality_phase':{'order':order,'count_per_path':200,'repeat_count':1,
       'all_paths_before_latency':True,'admission':'exact original task policy; invalid per-sample outputs count wrong; no threshold invented for LID',
       'asr':'same mono16k loading/normalization and corpus sum(errors)/sum(reference words); Whisper original long-form timestamps >30s; CTC full waveform and greedy argmax; no external LM',
       'lid':'full frozen audio files, native features; no transcripts; argmax over full native language space (not restricted to ten labels); trim/lowercase scorer; unsupported predictions wrong',
       'lid_MMS_mapping':{'ara':'ar','deu':'de','eng':'en','spa':'es','fas':'fa','fra':'fr','jpn':'ja','rus':'ru','tur':'tr','cmn':'zh'},
       'lid_MMS_unmapped':'return native ISO code and score wrong; no fuzzy aliases or relabeling after quality results',
       'lid_SpeechBrain_mapping':'take language code prefix before colon from pinned label_encoder.txt; all107 classes retained',
       'text':'one original English sentence; native tokenizer with truncation=True; 0 negative / 1 positive; argmax classifier head; no labels or prompt annotations as inputs',
       'Text_generic_label_provenance':'TextAttack XLNet/BART config finetuning_task=sst-2 plus task/model mapping in published supplement; 0/1 GLUE semantics fixed before outputs; never use validation labels to choose label permutation'},
   'service_phase':{'order':'For repeat1,2,3: task ASR,LID,Text; reference control first for each task then admitted frozen candidates in listed order.',
       'candidate_order':order,'reference_controls':REFERENCE,'repeat_count':3,'requests_per_repeat':30,
       'warmup':'exactly one untimed request of first frozen latency sample per fresh process, original active_single_profile_v1',
       'submission':'sequential, next request after prior result; no open-loop low-QPS or hotspot campaign',
       'sample_subset':'all IDs and order frozen above and matched against original denominator raw requests',
       'timing':'original worker start/end timer around native handler: file decode/front-end/H2D/model/sync/output decode/scoring included; queue wait/startup/warmup excluded',
       'aggregation':'weighted mean of all90 valid requests; also raw per-request, mean per repetition, median,p95,std; never fastest run selection',
       'reference_drift':'rerun same reference checkpoint and semantics; report historical ratio and same-session control ratio separately. Selection uses immutable Module12 denominator. Report differences; do not rescale old baseline or infer transition if comparability fails.',
       'profiler_semantics_source':ref(ROOT/'joint_deploy/run_active_single_profile.py'),
       'worker_timer_source':ref(ROOT/'joint_deploy/worker_protocol.py')},
   'memory_phase':{'primary':'single-worker post-warmup idle NVML process MiB; observe 5s at100ms intervals and average each run; average3 fresh-process runs',
       'secondary':'idle, inference peak NVML process MiB at10ms polling (sampled peak), torch allocated/reserved current/max and checkpoint artifact bytes separately',
       'peak_limitation':'10ms NVML polling may miss short peaks; report torch maxima separately, never replace idle ratio by peak silently',
       'synchronization':'synchronize before/after request and before idle window; whole checkpoint resident, no offload/extra padding/forced allocation',
       'stop':'all candidate processes exit and VRAM released before next process'},
   'environment_preflight':'13B downloads/install in separate namespace; perform CPU loading/API/label sanity and lock full environment/adapter SHA before first GPU performance result; correctness-only fixes logged, cannot change shortlist/checkpoint/metric/threshold/order/selection.',
   'failure_no_cherrypick':'Keep all failures. No immediate model substitution, tuning, best-run retry, alternate precision/backend or new campaign. Infrastructure-only retry requires an evidence-bearing preregistered amendment before GPU retry; preserve failed raw attempt.',
   'no_final_deployment_experiment':True,'no_low_QPS':True,'no_multi_hotspot':True,
   'no_Module9_mutation':True,'no_baseline_mutation':True})

audit=['# Module 13A source audit\n',
       'Only CPU / WEB / METADATA work was performed. No model weights were fetched and no model was loaded or run. Public unauthenticated HF API access establishes discoverability; 13B must validate weight downloads and offline runtime.\n',
       'Metadata snapshots are pinned to model SHAs, with retrieval timestamps and local SHA256 hashes in source_manifest.json. Exact files and LFS hashes are preserved in each *.api.json. Missing README files are recorded by their absence, not reconstructed.\n',
       'Original frozen Text is SST-2 sentence sentiment classification; original LID is VoxLingua107 spoken-language identification from audio. Generic pretrained encoders needing a newly trained classifier, text-LID, SST-3, NLI and generative sentiment substitutes are excluded.\n',
       'GPU feasibility is an inference from ready batch1 architectures and artifact sizes, not a measured allocation. Parameters, disk weights, idle VRAM, peak VRAM and latency are separate fields. No parameter ratio is used as gamma or mu. Current local host metadata reports 49140 MiB; a standard RTX4090 24GB deployment has not been tested.\n',
       'LID has semantic-only admission. QUALITY_LIKELY_PASS for LID does not claim accuracy >= the old reference. Public family and full-set scores cannot be local subset ground truth.\n',
       'MMS-1024 vs MMS-256 retains the same48-layer/1280-wide backbone. Artifact bytes increase by about3.15MB (0.0815%); primarily a classifier/storage difference is plausible, not established latency or idle-memory difference.\n',
       'No candidate is PRELIMINARY_DOMINATED: comparable local latency/idle memory are missing. Unshortlisted candidates are retained with explicit evidence/reasons.\n',
       'Search coverage: six candidate task paths each for ASR/LID/Text; 15 unique checkpoint repositories. HF search terms included roberta-large-sst2, roberta-large-finetuned-sst2, bert-large-sst2, deberta-v3-large-sst2. Few-shot SetFit results, untrained base models and incomplete dataset evidence were not promoted to same-task likely-pass checkpoints. This is a bounded public search, not an exhaustive ecological claim.\n',
       '## Public evidence\n']
for key,e in EVIDENCE.items():
    links=', '.join('[source]('+u+')' for u in e['source_urls'])
    audit.append(f"- {key}: {e['preliminary_quality']}; {e['published_metric']}={e['published_value']} on {e['published_dataset']}. {e['limitations']} {links}\n")
audit += ['\n## Retrieval limitations\n',
          'The TextAttack ReadTheDocs page was readable with the web tool, but raw urllib archival returned HTTP403; supplemental_sources.json records that failure. The primary paper supplement was archived successfully and supplies the exact BERT/XLNet/BART checkpoint-quality mapping. No missing result was invented.\n',
          '\n## Reproducible local paths\n']
for r in ROWS:
    audit.append(f"- {r['task']} / `{r['model_id']}@{r['revision']}`: {r['local_inference_path']}. [pinned files]({r['model_card_url']})\n")
(OUT/'source_audit.md').write_text('\n'.join(audit))

counts={t:sum(r['task']==t and r['legal_status']=='LEGAL_PUBLIC_LOCAL_READY_MODEL' for r in ROWS) for t in UNIVERSE}
gate='REAL_SPECIALIST_SHORTLIST_READY' if all(len(SHORT[t])>=2 for t in SHORT) else 'REAL_SPECIALIST_SHORTLIST_PARTIAL'
write('module13a_gate.json',{'module':'13A','gate':gate,'at_utc':STAMP,
   'evidence_kind':'REAL_MODEL_CANDIDATE_DISCOVERY','candidate_task_paths':18,
   'legal_candidate_count_by_task':counts,'shortlist_count_by_task':{t:len(SHORT[t]) for t in SHORT},
   'unique_shortlist_checkpoints':len({x for a in SHORT.values() for x in a}),
   'all_shortlisted_public_ungated':all(not REMOTE[m].get('gated') for a in SHORT.values() for m in a),
   'all_shortlisted_local_inference_paths_documented':True,
   'local_inference_runtime_validation':'DEFERRED_TO_13B; documented reproducibility is gate evidence, not a GPU smoke test',
   'quality_admission_frozen':True,'task_mapping_checkpoint_dataset_metric_threshold_order_selection_frozen':True,
   'all_new_gamma_mu_target_classifications':'UNKNOWN','new_operating_points_near_transition_demonstrated':False,
   'BALANCED_SPECIALIST_REGIME_status':'PROPOSED_NOT_YET_REAL_MEASURED_OR_ADMITTED',
   'selected_to_make_Generalist_win':'NO','selection_rule_evidence':ref(OUT/'specialist_selection_rule.json'),
   'GPU_runs':0,'weights_downloaded':0,'models_loaded':0,
   'counterfactual_values_used_as_real_performance':False,'Module13B_started':False,
   'completion_instruction':'Stop after Module13A. Do not execute the frozen measurement specification.'})

summary=['# Module 13A — Real Specialist Candidate Discovery\n',
  f'Gate: **{gate}**. ASR/LID/Text 各冻结3个 task-model paths，共9条、8个唯一 checkpoint。所有新候选未本机实测；BALANCED_SPECIALIST_REGIME 目前是待13B验证的 proposal，不能作为正式实验模型。\n',
  '本轮CPU/WEB/METADATA only：GPU runs=0，权重下载=0，模型加载=0。Module12是COUNTERFACTUAL SEARCH TARGET；13A是真实公开模型发现；13B才是真实本地质量/延迟/内存测量。EFFICIENCY_OPTIMAL_SPECIALIST_REGIME、Module9及全部历史结果保留。\n',
  '## 1–4. 候选和当前 references\n',
  'Text语义：SST-2英文句子negative/positive情感二分类；200条validation冻结manifest。LID语义：VoxLingua107 dev音频识别语言，十种reference labels；模型在自身完整language class space上预测。ASR：LibriSpeech test-clean英文转录、corpus WER。\n',
  '| Task | Module12比例reference | 已测service ms / idle MiB | 原quality admission |\n|---|---|---|---|']
for t in SHORT:
    policy='WER ≤ 0.042181' if t=='asr' else 'accuracy ≥ 0.925' if t=='text' else 'semantic validity only；无数值accuracy门槛'
    summary.append(f"| {t.upper()} | {BASE[t]['reference_checkpoint']} | {BASE[t]['reference_service_ms']:.9f} / {BASE[t]['reference_idle_memory_mib']} | {policy} |")
summary += ['\nASR baseline同时保留Whisper-small、Whisper large-v3-turbo、Distil-Whisper large-v3；LID保留Whisper-small和turbo；Text保留DistilBERT和RoBERTa-base SST-2。完整版本和原始来源见baseline_references.json。\n',
            '公开candidate universe（参数仅作模型metadata；GB是实际选定权重文件十进制磁盘大小；均以FP32作为13B主operating point）：\n',
            '| Task | Candidate | 参数metadata | 权重GB | 公开质量证据 | Preliminary | 13B |\n|---|---|---|---|---|---|---|']
for r in ROWS:
    params=f"{r['parameter_count']/1e6:.1f}M" if r['parameter_count'] else 'UNKNOWN'
    quality=f"{r['published_metric']}={r['published_quality']}" if r['published_quality'] is not None else 'family/card evidence；无对应frozen-set数值'
    summary.append(f"| {r['task'].upper()} | [{r['model_id']}]({r['model_card_url']}) | {params} | {r['artifact_weight_gb']:.4f} | {quality} | {r['preliminary_quality']} | {'冻结' if r['shortlist'] else '保留未入选'} |")
summary += [
 '\n## 5–7. 靶区接近度与memory/latency区别\n',
 '新候选：**目前无法确认谁最接近γ≈1.2–1.25，也无法确认谁接近γ≈1.9–2.0；全部gamma_est、mu_est为空且分类UNKNOWN**。这表示证据不足，不能声称现实生态不存在这些点。Whisper medium/full-large和base/large Text覆盖真实架构/容量变化，CTC/ECAPA等高效替代也保留；它们是测量探索点，不是已证实near-target候选。\n',
 '旧ASR baseline可作已测定位参考：turbo γ=1.1662、small γ=1.3005，相对Distil-large-v3位于first-split靶区两侧；旧Text RoBERTa-base γ≈1.4609，相对DistilBERT仍低于Text per-task[1.5,2.0]搜索区间。这些是已有模型的历史测量比例，不是新候选的13A估计，也不是新deployment outcome。\n',
 '没有证据认定某个新候选“主要是本机memory更大而latency不变”。最接近该假设的metadata对照是MMS-1024 vs MMS-256：同48层backbone，只扩分类头，文件约增0.0815%；不等于μ≈1.05。Whisper full-large-v3相对turbo扩decoder，LID只检测语言token，但仍经过decoder层，不能声称其latency不变。\n',
 '参数量不能当VRAM；FP16磁盘权重也不能当FP32本机resident memory。例如large-v3磁盘约3.087GB，13B主点会转FP32，尚未计入CUDA context、workspace、activations、KV、allocator。CSV中的size_over_reference_resident_bytes仅是显式标注的disk/idle-byte proxy，不作μ估计或靶区分类。\n',
 '## 8–9. 公开质量与“大模型不一定更适合”\n',
 '每个候选的来源、数值/缺失、数据集、checkpoint归属强弱和限制均在quality_evidence.json/source_audit.md；官方卡已固定到SHA。Whisper-medium/medium.en/large-v2的greedy test-clean WER来自论文Table8；large-v3仅用官方跨语言改善证据，不能给它套用large-v2的具体WER。MMS数字是论文对应语言规模system evidence，不能冒充HF checkpoint实测。\n',
 'BERT-base公开accuracy约0.924，紧贴且略低于0.925，标QUALITY_UNCERTAIN。DeBERTa-v3-base卡报告0.9599但evaluation dataset未注明，不能按名称确认为SST-2 likely-pass。BERT-large/RoBERTa-large/BART-large已有同任务classifier，但参数更大并不保证更接近靶区；MMS-1024更多语言对当前十个reference languages无必然收益。所有未入选候选保留；没有在缺失latency/memory时宣称真实dominance。\n',
 '## 10–12. 冻结短名单、反cherry-picking、13B下载\n',
 '短名单：ASR=Wav2Vec2 large-960h-lv60-self、Whisper large-v3、Whisper medium；LID=MMS-LID-256、Whisper medium、SpeechBrain VoxLingua107 ECAPA；Text=RoBERTa-large SST-2、BART-large SST-2、XLNet-base SST-2。每个task3个，medium共享checkpoint下载但分别测ASR/LID，不能把两task的memory重复加成两套模型。\n',
 '**有没有为了让Generalist赢而挑模型：NO。** 已冻结selection-rule evidence：原quality admission → 真实Q/latency/idle-memory Pareto（含旧baseline）→ 距离Module12靶区union最近 → higher measured quality → lower measured memory → full model ID@revision。Generalist/V3 outcome不进入选择；旧baseline若支配或按规则胜出，保留旧baseline并报告BASELINE_RETAINED，不能强行制造balanced结果。\n',
 '13B完整测量顺序：ASR→LID→Text，各task按完整model ID排序；先所有9条200样本quality，之后仅admitted paths做3×30单worker测量。每轮task先重测原reference control，再测候选；无low-QPS、hotspot或deployment campaign。历史分母原样固定，同期control比例另报；所有失败保留，不根据结果换checkpoint、调precision/backend或补选。\n',
 '13B需要下载以下8个精确版本及其必要tokenizer/processor/config文件（本轮尚未下载权重）：\n',
 '| Checkpoint | Frozen revision | Tasks |\n|---|---|---|']
for model in sorted({m for a in SHORT.values() for m in a}):
    ts=','.join(t.upper() for t in SHORT if model in SHORT[t])
    summary.append(f"| {model} | `{REMOTE[model]['revision']}` | {ts} |")
summary += ['\nFP32主点、单请求/单replica、原样本与metric/threshold已冻结。SpeechBrain需独立版本环境和本地pretrained_path override；XLNet需SentencePiece。基本推理代码路径来自公开文档，本轮未做runtime smoke；13B先CPU加载/接口核对并锁adapter/environment SHA，再出现新GPU性能结果。\n',
 '## 13. Gate与停止边界\n',
 f'**{gate}**：各task≥2个公开、无需用户fine-tuning、可本地推理的合法checkpoint且路径有证据；该gate不认证质量已过、γ/μ落靶区、标准24GB实测通过或新regime已成立。13A结果不足以建立真实Pareto frontier。13B后才允许新Specialist成为正式实验模型。\n',
 '完成Module13A后停止；未开始13B、正式GPU benchmark、最终deployment、low-QPS、multi-hotspot、Task-Rich，也未做人工sleep/slowdown或memory scaling。\n']
(OUT/'module13a_summary.md').write_text('\n'.join(summary))
print('Built',len(ROWS),'candidate task paths;',len(short_entries),'frozen paths;',gate)

if __name__ == '__main__':
    pass
