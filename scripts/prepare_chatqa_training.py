"""Freeze a train-only QA adapter corpus; no model forward, weights or CUDA."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsdeploy.backends.compact import CompactAdapter
from gsdeploy.profiling.prepare import digest
from gsdeploy.profiling.runner import save


def train_questions(document, excluded_ids, excluded_articles, excluded_contexts):
    """Refuse split collisions; training references are never evaluation labels."""
    groups = {True: [], False: []}
    seen = set()
    for article in document['data']:
        if article['title'] in excluded_articles:
            raise ValueError('Training article overlaps the frozen evaluation splits')
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            if hashlib.sha256(context.encode()).hexdigest() in excluded_contexts:
                raise ValueError('Training context overlaps the frozen evaluation splits')
            for qa in paragraph['qas']:
                if qa['id'] in seen or qa['id'] in excluded_ids:
                    raise ValueError('Duplicate or overlapping training question ID')
                seen.add(qa['id'])
                impossible = qa['is_impossible']
                if type(impossible) is not bool or bool(qa['answers']) == impossible:
                    raise ValueError('Inconsistent official training answerability')
                groups[not impossible].append({'id': qa['id'], 'task': 'text', 'dataset': 'SQuAD2-train',
                    'context': context, 'question': qa['question'], 'article': article['title'],
                    'target': '<No Answer>' if impossible else qa['answers'][0]['text'],
                    'answerable': not impossible})
    return groups


def training_ids(tokenizer, prompt, target):
    prefix = tokenizer.encode(tokenizer.bos_token + prompt, add_special_tokens=False)
    answer = tokenizer.encode(' ' + target, add_special_tokens=False)
    if tokenizer.encode(tokenizer.bos_token + prompt + ' ' + target, add_special_tokens=False) != prefix + answer:
        raise ValueError('Training answer boundary changes the complete prompt tokens')
    return prefix + answer + [tokenizer.eos_token_id], [-100] * len(prefix) + answer + [tokenizer.eos_token_id]


def prepare(output, per_class=5000):
    from transformers import AutoTokenizer
    source_path = ROOT / 'artifacts/qa-adaptation-20261018/source-verification.json'
    source = json.loads(source_path.read_text())
    if digest(source['local_path']) != source['sha256']:
        raise ValueError('Changed pinned official training source')
    study_path = ROOT / 'configs/study_repaired.json'
    study = json.loads(study_path.read_text())
    excluded_ids, excluded_articles, excluded_contexts = set(), set(), set()
    input_sources = []
    for split in ['calibration_preparation', 'quality_preparation']:
        manifest = ROOT / study[split]['manifest']
        if digest(manifest) != study[split]['manifest_sha256']:
            raise ValueError('Changed frozen evaluation manifest')
        entry = json.loads(manifest.read_text())['tasks']['text']
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Changed frozen QA split')
        # Only IDs, article IDs and context hashes are used for exclusion.
        # Evaluation answers are never read into the training examples.
        for line in Path(entry['path']).read_text().splitlines():
            row = json.loads(line)
            excluded_ids.add(row['id'])
            excluded_articles.add(row['article'])
            excluded_contexts.add(hashlib.sha256(row['context'].encode()).hexdigest())
        input_sources.extend([manifest, Path(entry['path'])])
    document = json.loads(Path(source['local_path']).read_text())
    groups = train_questions(document, excluded_ids, excluded_articles, excluded_contexts)
    pointer = json.loads((ROOT / 'artifacts/qa-replacement-20261018/active-run.json').read_text())
    worker = json.loads((Path(pointer['output']) / 'calibration-packet/template-worker.json').read_text())
    policy = json.loads(Path(worker['resources']['protocol']).read_text())
    policy['tasks']['text']['chatqa_prompt_format'] = 'native_system_v2'
    train_policy = copy.deepcopy(policy)
    train_policy['tasks']['text']['dataset'] = 'SQuAD2-train'
    adapter = CompactAdapter(train_policy, json.loads(Path(worker['resources']['classmap']).read_text()))
    tokenizer = AutoTokenizer.from_pretrained(worker['resources']['checkpoint']['path'], local_files_only=True)
    selected, rejected = [], Counter()
    for answerable in [True, False]:
        pool = sorted(groups[answerable], key=lambda s: s['id'])
        random.Random(20261018 + int(answerable)).shuffle(pool)
        count = 0
        for row in pool:
            request = adapter.build(worker['candidate'], row)
            try:
                ids, labels = training_ids(tokenizer, request['native']['text'], row['target'])
            except ValueError:
                rejected['token_boundary'] += 1
                continue
            target_length = sum(label != -100 for label in labels)
            if len(ids) > 1024 or target_length > policy['tasks']['text']['max_new_tokens']:
                rejected['whole_example_exceeds_declared_training_budget'] += 1
                continue
            selected.append({**row, 'input_ids': ids, 'labels': labels})
            count += 1
            if count == per_class:
                break
        if count != per_class:
            raise ValueError('Insufficient complete train-only examples for the fixed balanced corpus')
    random.Random(20261018).shuffle(selected)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = output / 'training.jsonl'
    with data.open('w') as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    evaluation_policy_path = output / 'evaluation-protocol.json'
    save(evaluation_policy_path, policy)
    save(output / 'training-protocol.json', train_policy)
    worker['resources']['protocol'] = str(evaluation_policy_path)
    save(output / 'base-worker.json', worker)
    recipe = {'schema': 'gsdeploy.qa_train_only_adapter.v1', 'state': 'CPU_PREPARED_GPU_UNTRAINED',
              'gpu_started': False, 'models_loaded': 0, 'data': str(data), 'data_sha256': digest(data),
              'samples': len(selected), 'answerable': per_class, 'unanswerable': per_class,
              'base_worker': str(output / 'base-worker.json'), 'base_worker_sha256': digest(output / 'base-worker.json'),
              'seed': 20261018, 'max_sequence_tokens': 1024, 'truncation': False,
              'epochs': 1, 'microbatch': 2, 'gradient_accumulation': 16,
              'learning_rate': 0.0002, 'warmup_fraction': 0.03, 'max_grad_norm': 1.0,
              'lora': {'r': 16, 'alpha': 32, 'dropout': 0.05,
                       'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']},
              'whole_example_rejections': dict(rejected),
              'split_audit': {'excluded_question_ids': len(excluded_ids), 'excluded_articles': len(excluded_articles),
                              'excluded_contexts': len(excluded_contexts), 'collisions': 0,
                              'training_articles': len({row['article'] for row in selected})},
              'source_hashes': {str(p): digest(p) for p in [source_path, Path(source['local_path']), Path(__file__),
                                  study_path, output / 'training-protocol.json', evaluation_policy_path, *input_sources]},
              'scope': 'Only official train-v2.0 labels. Frozen calibration and independent quality examples never train the adapter.'}
    save(output / 'recipe.json', recipe)
    print(json.dumps({k: recipe[k] for k in ['state', 'samples', 'split_audit', 'whole_example_rejections']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.output)
