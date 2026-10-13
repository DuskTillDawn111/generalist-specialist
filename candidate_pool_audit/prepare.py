"""Static task/adapter contract audit, completed before new-model load tests."""
import json
from pathlib import Path

from .inventory import ROOT, TASKS, EXISTING, NEW
from .load_test import MANIFESTS, rows
from .provenance import sha

OUT = ROOT / 'results/candidate_pool_v2'


def main():
    definitions = {}
    config = {
        'text': ('sst2', 'validation', 'accuracy', {'0': 'negative', '1': 'positive'},
                 'Sentence sentiment classification; no passage QA or generic text generation.',
                 ['joint_deploy/distilbert_worker.py', 'benchmark/scripts/eval_distilbert_sst2.py']),
        'image': ('imagenette', 'validation', 'accuracy', {str(i): v for i, v in enumerate(
                 ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn',
                  'garbage truck', 'gas pump', 'golf ball', 'parachute'])},
                  'Closed-set 10-class Imagenette; gather matching ImageNet-1K logits before argmax.',
                  ['joint_deploy/resnet_worker.py', 'benchmark/scripts/eval_resnet_imagenette.py']),
        'ocr': ('iiit5k', 'test', 'exact_match', None,
                'Recognize the single word in a supplied image crop. Output recognized text; lowercase and retain only a-z/0-9 for scoring.',
                ['joint_deploy/ocr_worker.py', 'benchmark/scripts/eval_paddleocr_iiit5k.py']),
        'asr': ('librispeech', 'test-clean', 'corpus_word_error_rate', None,
                'English audio transcription. Decode mono 16kHz; lowercase, retain apostrophes, remove other punctuation and collapse whitespace. WER=sum(word errors)/sum(reference words).',
                ['joint_deploy/whisper_worker.py', 'benchmark/scripts/eval_whisper_librispeech.py']),
        'visual_qa': ('scienceqa', 'validation', 'accuracy', 'Per-sample option letters A..chr(64+len(choices))',
                       'ScienceQA-IMG multiple-choice question + choices + hint + image; answer one option letter.',
                       ['joint_deploy/qwen_worker.py', 'benchmark/scripts/eval_qwen_scienceqa.py']),
        'lid': ('voxlingua107', 'dev', 'accuracy', sorted({s['label'] for s in rows('lid')}),
                'Spoken-language identification from audio, independent of transcription. Output current language code; trim and lowercase only.',
                ['joint_deploy/whisper_worker.py', 'benchmark/scripts/eval_whisper_voxlingua.py', 'benchmark/scripts/eval_qwen_voxlingua.py']),
    }
    inputs = {
        'text': {'input': 'sentence string', 'label': 'negative|positive', 'label_id': 'integer 0|1'},
        'image': {'input': 'existing local RGB-convertible image path', 'label': 'Imagenette name', 'label_id': 'integer 0..9'},
        'ocr': {'input': 'existing local word-crop image path', 'label': 'recognized reference word', 'normalized_label': 'optional normalized word'},
        'asr': {'input': 'existing local audio path', 'label': 'reference transcript', 'duration_s': 'positive float', 'sample_rate': '16000', 'normalized_label': 'optional normalized reference'},
        'visual_qa': {'input': 'existing local image path', 'question': 'string', 'choices': 'list[string]', 'hint': 'string', 'label': 'option letter', 'label_id': 'zero-based choice index'},
        'lid': {'input': 'existing local audio path', 'label': 'language code', 'duration_s': 'positive float', 'sample_rate': '16000'},
    }
    sources = {'text': 'https://huggingface.co/datasets/stanfordnlp/sst2',
               'image': 'https://github.com/fastai/imagenette', 'ocr': 'https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset',
               'asr': 'https://www.openslr.org/12', 'visual_qa': 'https://github.com/lupantech/ScienceQA',
               'lid': 'https://bark.phon.ioc.ee/voxlingua107/'}
    for task, (dataset, split, metric, labels, semantics, adapters) in config.items():
        path = ROOT / f'benchmark/manifests/{MANIFESTS[task]}.jsonl'
        samples = rows(task)
        if any(s['dataset'] != dataset or s['split'] != split for s in samples):
            raise ValueError('Mixed source/split: ' + task)
        if task in ('text', 'image') and any(s['label'] != labels[str(s['label_id'])] for s in samples):
            raise ValueError('Label mismatch: ' + task)
        for sample in samples:
            if task != 'text' and not Path(sample['input']).is_file():
                raise ValueError('Missing sample asset: ' + sample['id'])
        output = {'prediction': 'normalized recognized text or task label', 'reference': 'normalized reference'}
        if task == 'asr':
            output.update(word_errors='nonnegative integer', reference_words='positive integer', wer='word_errors/reference_words')
        else:
            output['correct'] = 'boolean'
        common_sample = {'id': 'stable sample identifier', 'task': samples[0]['task'],
                         'dataset': dataset, 'split': split, 'input_type': samples[0]['input_type']}
        definitions[task] = {'task': task, 'definition': semantics,
            'input_schema': {'transport': 'existing worker request: request_id,pool_id,arrival_index,task_type,sample,submit_ns',
                             'sample': {**common_sample, **inputs[task]}, 'annotation_policy': 'Labels are used only to score output; never included in the model input.'},
            'output_schema': output, 'dataset': dataset, 'split': split, 'source': sources[task],
            'evaluation_manifest': str(path), 'manifest_sha256': sha(path), 'sample_count': len(samples),
            'metric': metric, 'label_space': labels, 'existing_adapter': [str(ROOT / p) for p in adapters],
            'service_pool': str(ROOT / 'benchmark/task_pools/mixed_formal_1200_6task.jsonl'),
            'preprocessing': semantics,
            'scope_note': 'Historical datasets and metrics frozen. Sanity calls establish interface compatibility, not benchmark quality.'}
        if task == 'asr':
            definitions[task]['historical_latency_subset'] = '[4,6) seconds; sanity ASR selected from the same subset; service supports general durations.'
        if task == 'image':
            definitions[task]['preprocessing'] = 'Decode same RGB images and same closed-set mapping. Use each official pretrained weight transform; ConvNeXt uses its own officially prescribed resize/crop/normalization.'
        directory = OUT / 'tasks' / task
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'task_definition.json').write_text(json.dumps(definitions[task], indent=2, ensure_ascii=False))
    (OUT / 'task_definitions.json').write_text(json.dumps(definitions, indent=2, ensure_ascii=False))
    reasons = {
        'convnext_tiny': 'Official torchvision ImageNet-1K V1 weights; identical Imagenette RGB inputs, 10 target category metadata indices, gather logits before argmax, same manifest and accuracy.',
        'roberta_sst2': 'Publisher TextAttack model zoo explicitly maps checkpoint to SST-2; pinned config.finetuning_task=sst-2; 2 logits, GLUE IDs 0=negative/1=positive; opposing-label sanity gate. Runtime classifier has 124,647,170 versus DistilBERT 66,955,010 parameters (1.86x); stored checkpoint includes an unused pooler and has 125,237,762 tensor elements.',
        'paddleocr_vl': 'Official element-level OCR: prompt yields recognized text; input is the same IIIT5K word crop; decode only generated suffix and apply frozen OCR normalization. No document parser, layout, formula, or added task.',
        'whisper_turbo': 'Official multilingual ASR checkpoint; same processor/acoustic features, audio decoder and normalized WER. LID uses separate detect_language without forced English and checks the model language-token map against current codes.',
        'distil_whisper': 'Official English ASR checkpoint; same LibriSpeech inputs, transcript normalization and WER; no current LID adapter.'}
    contracts = {name: {'adapter_contract_status': 'verified', 'tasks': tasks,
                         'reason': reasons.get(name, 'Existing frozen task adapter and historical formal per-task inference evidence.'),
                         'capability_status': 'pending until successful local inference and unload; historical task evidence explicitly cited in final matrix'}
                 for name, tasks in {**EXISTING, **NEW}.items()}
    contracts['gemma3n'] = {'adapter_contract_status': 'pending', 'reason': 'Official gated weights: configuration HTTP 403; no local HF credentials; no task adapter verified.', 'tasks': []}
    contracts['got_ocr'] = {'adapter_contract_status': 'pending', 'reason': 'Optional fallback only; plain-text API exists but no local adapter/inference verified. Preferred PaddleOCR-VL tested first.', 'tasks': []}
    contracts['siebert_rejected'] = {'adapter_contract_status': 'unsupported', 'reason': '15-dataset general sentiment checkpoint; does not satisfy strict same SST-2 fine-tuned dataset requirement. Same binary labels alone are insufficient.', 'tasks': []}
    (OUT / 'adapter_preflight.json').write_text(json.dumps(contracts, indent=2))
    print(json.dumps({'tasks': list(definitions), 'preflight_models': len(contracts)}))


if __name__ == '__main__':
    main()
