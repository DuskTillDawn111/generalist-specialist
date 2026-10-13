"""Candidate-only adapters on the existing READY/RESULT/BYE process protocol.

No benchmark loop, routing, latency aggregation, or deployment search.
"""
import copy
import json
import os
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'joint_deploy'))
sys.path.insert(0, str(ROOT / 'benchmark/scripts'))
from worker_protocol import run_worker
import eval_whisper_librispeech as asr
import eval_whisper_voxlingua as lid

name = Path(sys.argv[0]).stem.removeprefix('worker_')
if name == 'new_worker':
    raise RuntimeError('Launch a candidate wrapper from registry.py')
model_dir = ROOT / 'models/candidate_pool_v2' / name
device = torch.device('cuda:0')
if not torch.cuda.is_available():
    raise RuntimeError('Local CUDA inference required')

LABELS = ['negative', 'positive']
IMAGE_LABELS = ['tench', 'English springer', 'cassette player', 'chain saw',
                'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']

if name == 'convnext_tiny':
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    indices = [weights.meta['categories'].index(label) for label in IMAGE_LABELS]
    index_tensor = torch.tensor(indices, device=device)
    model = convnext_tiny(weights=None)
    model.load_state_dict(torch.load(model_dir / 'convnext_tiny-983f1562.pth',
                                    map_location='cpu', weights_only=True), strict=True)
    model.eval().to(device)
    tasks = ['image']
elif name == 'roberta_sst2':
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    processor = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    if model.config.finetuning_task != 'sst-2' or model.config.num_labels != 2:
        raise ValueError('SST-2 checkpoint with two output logits required')
    # This SST-2 checkpoint uses GLUE IDs, confirmed with opposing labeled sanity samples.
    model.eval().to(device)
    tasks = ['text']
elif name in ('whisper_turbo', 'distil_whisper'):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir, local_files_only=True,
                                                    torch_dtype=torch.float32).eval().to(device)
    for config in (model.config, model.generation_config):
        if hasattr(config, 'forced_decoder_ids'):
            config.forced_decoder_ids = None
    tasks = ['asr', 'lid'] if name == 'whisper_turbo' else ['asr']
elif name == 'paddleocr_vl':
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16).eval().to(device)
    tasks = ['ocr']
else:
    raise ValueError(name)
torch.cuda.synchronize()


@torch.inference_mode()
def handler(request):
    task, sample = request['task_type'], request['sample']
    if task not in tasks:
        raise ValueError(f'Unsupported task: {task}')
    if name == 'convnext_tiny':
        with Image.open(sample['input']) as image:
            tensor = preprocess(image.convert('RGB')).unsqueeze(0).to(device)
        logits = model(tensor).index_select(1, index_tensor)
        label_id = int(logits.argmax(-1).item())
        result = {'prediction': IMAGE_LABELS[label_id], 'prediction_id': label_id,
                  'reference': sample['label'], 'reference_id': int(sample['label_id']),
                  'imagenet_index': indices[label_id], 'correct': label_id == int(sample['label_id'])}
    elif name == 'roberta_sst2':
        inputs = processor(sample['input'], return_tensors='pt', truncation=True).to(device)
        label_id = int(model(**inputs).logits.argmax(-1).item())
        result = {'prediction': LABELS[label_id], 'prediction_id': label_id,
                  'reference': sample['label'].strip().lower(),
                  'correct': label_id == int(sample['label_id'])}
    elif task == 'lid':
        lid.validate_sample(sample)
        audio, sr = asr.load_audio(sample['input'])
        inputs = processor(audio, sampling_rate=sr, return_tensors='pt', return_attention_mask=True)
        config = copy.deepcopy(model.generation_config)
        config.language = None
        config.forced_decoder_ids = None
        token = model.detect_language(input_features=inputs['input_features'].to(device),
                                      generation_config=config)
        token_id = int(token.reshape(-1)[0].item())
        prediction = lid.token_id_to_language(token_id, config)
        result = {'prediction': prediction, 'reference': sample['label'],
                  'language_token_id': token_id, 'correct': prediction == sample['label'],
                  'language_detection': 'acoustic detect_language; no forced language',
                  'language_token_map': config.lang_to_id}
    elif task == 'asr':
        audio, sr = asr.load_audio(sample['input'])
        inputs, long_form = asr.prepare_inputs(processor, audio, sr, float(sample['duration_s']), device)
        ids = asr.generate_transcription(model, inputs, long_form)
        prediction = asr.normalize_asr(processor.batch_decode(ids, skip_special_tokens=True)[0])
        reference = asr.normalize_asr(sample.get('normalized_label', sample['label']))
        errors = asr.levenshtein_words(reference.split(), prediction.split())
        result = {'prediction': prediction, 'reference': reference, 'word_errors': errors,
                  'reference_words': len(reference.split()), 'wer': errors / len(reference.split()),
                  'long_form': long_form}
    else:
        with Image.open(sample['input']) as image:
            image = image.convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                               {'type': 'text', 'text': 'OCR:'}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                return_dict=True, return_tensors='pt').to(device)
        ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        raw = processor.decode(ids[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        # Same normalization as the frozen IIIT5K task; no Markdown/layout parsing.
        import re
        norm = lambda text: re.sub(r'[^a-z0-9]', '', text.lower())
        prediction = norm(raw)
        reference = norm(sample.get('normalized_label', sample['label']))
        result = {'prediction': prediction, 'reference': reference, 'raw_response': raw,
                  'correct': prediction == reference, 'recognition_mode': 'plain text OCR:'}
    torch.cuda.synchronize()
    return dict(result, model=name)


run_worker(name, handler, {'tasks': tasks, 'parameter_count': sum(p.numel() for p in model.parameters()),
                          'precision': str(next(model.parameters()).dtype),
                          'device': str(device), 'audit_only': True})
