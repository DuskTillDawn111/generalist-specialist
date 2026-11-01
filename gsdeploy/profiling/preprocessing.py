"""Actual CPU tokenizer/media preprocessing audit; weights and CUDA are forbidden."""

import argparse
import ast
from contextlib import redirect_stdout
import copy
import gc
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

from ..backends.compact import CompactAdapter, FAMILIES
from ..backends.compact_native import NativeExecutor, CompleteMiniProcessor
from .evaluation import read_rows
from .prepare import digest


class Prepared(Exception):
    pass


def native_mini_prepare(directory, processor, config, torch):
    """Run the pinned chat preprocessing AST, stopping before its generate call."""
    import numpy as np
    from PIL import Image
    from copy import deepcopy
    tree = ast.parse((directory / 'modeling_minicpmo.py').read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MiniCPMO')
    chat = copy.deepcopy(next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == 'chat'))
    chat.decorator_list = []
    init = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    template = next(ast.literal_eval(n.value) for n in init.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Attribute) and t.attr == 'default_tts_chat_template' for t in n.targets))
    namespace = {'np': np, 'Image': Image, 'deepcopy': deepcopy, 'json': json, 'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[chat], type_ignores=[])), '<pinned-minicpm-chat>', 'exec'), namespace)

    def run(request, media):
        captures = []
        def stop(**kwargs):
            captures.append(kwargs)
            raise Prepared()
        fake = SimpleNamespace(config=config, processor=processor, device=torch.device('cpu'),
                               default_tts_chat_template=template, generate=stop)
        proxy = CompleteMiniProcessor(processor, 32768 - request['generation']['max_new_tokens'])
        messages = copy.deepcopy(request['native']['msgs'])
        if media is not None:
            messages[0]['content'][0] = media
        try:
            namespace['chat'](fake, msgs=messages, tokenizer=processor.tokenizer, processor=proxy,
                              max_new_tokens=request['generation']['max_new_tokens'], **request['native']['options'])
        except Prepared:
            pass
        if len(captures) != 1:
            raise ValueError('MiniCPM preprocessing did not reach the native generation boundary')
        return captures[0], proxy.last_input_tokens
    return run


def native_deepseek_prepare(directory, torch):
    """Execute only pinned pure preprocessing AST, ending before every CUDA call."""
    import importlib.util
    import sys
    from typing import List, Dict, Optional, Tuple
    from abc import ABC
    import math
    import numpy as np
    from PIL import Image, ImageOps
    from torchvision import transforms
    conversation = directory / 'conversation.py'
    spec = importlib.util.spec_from_file_location('gsdeploy_deepseek_cpu_conversation', conversation)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tree = ast.parse((directory / 'modeling_deepseekocr.py').read_text())
    names = {'load_image', 'find_closest_aspect_ratio', 'dynamic_preprocess', 'normalize_transform',
             'format_messages', 'text_encode', 'load_pil_images', 'BaseTransform', 'BasicImageTransform'}
    nodes = [copy.deepcopy(n) for n in tree.body if getattr(n, 'name', '') in names]
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'infer')
    prefix = copy.deepcopy(method)
    prefix.name = 'prepare_native'
    prefix.decorator_list = []
    body = []
    for node in prefix.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not) and isinstance(node.test.operand, ast.Name)
                and node.test.operand.id == 'eval_mode'):
            break
        body.append(node)
    prefix.body = body + ast.parse('return {"input_ids": input_ids, "images_ori": images_ori, "images_crop": images_crop, "images_seq_mask": images_seq_mask, "images_spatial_crop": images_spatial_crop}').body
    for node in ast.walk(prefix):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {'cuda', 'generate', 'from_pretrained'}:
            raise ValueError('Native CPU prefix unexpectedly contains model/GPU execution')
    namespace = {'List': List, 'Dict': Dict, 'Optional': Optional, 'Tuple': Tuple, 'ABC': ABC,
                 'torch': torch, 'nn': torch.nn, 'np': np, 'math': math, 'os': os,
                 'Image': Image, 'ImageOps': ImageOps, 'transforms': transforms,
                 'get_conv_template': module.get_conv_template}
    nodes.append(prefix)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<pinned-deepseek-cpu-prefix>', 'exec'), namespace)
    return namespace['prepare_native']


def audit(study_path, policy_path, inventory_path, classmap_path, slot, output, limit=None, manifest_path=None, resume=False):
    import torch
    def forbidden(*args, **kwargs):
        raise RuntimeError('GPU access or checkpoint weight loading is forbidden in the CPU preprocessing audit')
    torch.cuda._lazy_init = forbidden
    torch.cuda.init = forbidden
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.cuda.get_device_capability = forbidden
    torch.Tensor.cuda = forbidden
    torch.load = forbidden
    import safetensors
    safetensors.safe_open = forbidden
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer, T5Tokenizer
    torch.set_num_threads(1)
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(output)
    study = json.loads(Path(study_path).read_text())
    policy = json.loads(Path(policy_path).read_text())
    candidate = next(c for c in study['generalists'] + study['specialists'] if c['slot'] == slot)
    inventory = json.loads(Path(inventory_path).read_text())['models'][slot]
    directory = Path(inventory['path'])
    family = FAMILIES[candidate['model']]
    adapter = CompactAdapter(policy, json.loads(Path(classmap_path).read_text()))
    config = AutoConfig.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
    if family == 'unifiedqa':
        processor = T5Tokenizer.from_pretrained(directory, local_files_only=True, legacy=True)
    elif family == 'deepseek':
        processor = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=True)
    executor = NativeExecutor(adapter, candidate, SimpleNamespace(config=config), processor,
                              torch=torch, device='cpu', dtype=torch.float32)
    mini = native_mini_prepare(directory, processor, config, torch) if family == 'minicpm' else None
    deep = native_deepseek_prepare(directory, torch) if family == 'deepseek' else None
    quality = json.loads(Path(manifest_path or study['quality_preparation']['manifest']).read_text())
    output.mkdir(parents=True, exist_ok=resume)
    contract = {'candidate': candidate, 'protocol_fingerprint': adapter.policy_sha256,
                'manifest_sha256': digest(manifest_path or study['quality_preparation']['manifest']),
                'limit_per_task': limit, 'classmap_fingerprint': adapter.classmap_sha256,
                'source_hashes': {str(p): digest(p) for p in [Path(__file__),
                    Path(__file__).parents[1] / 'backends/compact.py',
                    Path(__file__).parents[1] / 'backends/compact_native.py']},
                'processor_sources': {p.name: digest(p) for p in directory.glob('*.py')}}
    contract_path = output / 'audit-contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Partial CPU audit contract changed; preserve it and use a fresh directory')
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '\n')
    observations, failures = [], []
    journal = output / 'observations.jsonl'
    if resume and journal.exists():
        for entry in read_rows(journal):
            (failures if 'error' in entry else observations).append(entry)
    done = {entry['sample_id'] for entry in observations + failures}
    if len(done) != len(observations) + len(failures):
        raise ValueError('Duplicate sample identity in CPU audit journal')
    if family == 'siglip':
        classes = adapter.build(candidate, read_rows(quality['tasks']['image']['path'])[0])['native']
        labels = processor(text=classes['class_texts'], **classes['text_processor_options'])
        if tuple(labels['input_ids'].shape) != (1000, 64):
            raise ValueError('The full SigLIP class vocabulary does not fit the frozen text budget')
    import tempfile
    for task in adapter.supported_tasks(candidate):
        rows = read_rows(quality['tasks'][task]['path'])
        if digest(quality['tasks'][task]['path']) != quality['tasks'][task]['sha256']:
            raise ValueError('Input manifest content changed')
        if limit:
            rows = rows[:limit]
        for row in rows:
            if row['id'] in done:
                continue
            try:
                if task != 'text' and digest(row['input']) != row['input_sha256']:
                    raise ValueError('Changed source media')
                request = adapter.build(candidate, row)
                with redirect_stdout(io.StringIO()):
                    if mini:
                        inputs, length = mini(request, executor.media(request))
                    elif deep:
                        with tempfile.TemporaryDirectory(dir=output) as tmp:
                            options = dict(request['native']['options'])
                            options.pop('eval_mode')
                            inputs = deep(SimpleNamespace(disable_torch_init=lambda: None), processor,
                                          output_path=tmp, eval_mode=True, **options)
                            length = len(inputs['input_ids'])
                    elif family == 'siglip':
                        inputs = processor(images=executor.media(request), return_tensors='pt')
                        if tuple(inputs['pixel_values'].shape[-2:]) != (384, 384):
                            raise ValueError('SigLIP resolution differs from the checkpoint contract')
                        length = None
                    else:
                        prepared = executor.prepare(request)
                        inputs, length = prepared['inputs'], prepared['input_tokens']
                shapes = {k: list(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')}
                entry = {'sample_id': row['id'], 'task': task, 'input_tokens': length,
                         'tensor_shapes': shapes}
                observations.append(entry)
            except Exception as error:
                entry = {'sample_id': row['id'], 'task': task, 'error': type(error).__name__, 'detail': str(error)}
                failures.append(entry)
            with journal.open('a') as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
            # The audit retains shapes only. Release the preceding full-page/audio
            # tensors before preparing the next sample in memory-limited CPU jobs.
            inputs = prepared = None
            if (len(observations) + len(failures)) % 64 == 0:
                gc.collect()
        (output / 'progress.json').write_text(json.dumps({'processed': len(observations), 'failures': failures}, indent=2))
    result = {'status': 'PASS_CPU_PREPROCESSING' if not failures else 'FAIL_CPU_PREPROCESSING',
              'model': candidate, 'protocol_fingerprint': adapter.policy_sha256,
              'samples_checked': len(observations) + len(failures), 'successful': len(observations),
              'failures': failures, 'limited_diagnostic': limit is not None,
              'cuda_initialized': torch.cuda.is_initialized(), 'weights_loaded': 0, 'gpu_started': False,
              'observations': observations,
              'processor_source_hashes': {p.name: digest(p) for p in directory.glob('*.py')},
              'input_manifest_sha256': {t: quality['tasks'][t]['sha256'] for t in adapter.supported_tasks(candidate)}}
    (output / 'preprocessing.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='configs/study.json')
    parser.add_argument('--protocol', default='configs/compact_protocol.json')
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--classmap', required=True)
    parser.add_argument('--slot', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit-per-task', type=int)
    parser.add_argument('--manifest', help='Optional frozen calibration manifest instead of held-out quality inputs')
    parser.add_argument('--resume', action='store_true', help='Resume only an identical CPU preprocessing contract')
    args = parser.parse_args()
    result = audit(args.study, args.protocol, args.inventory, args.classmap, args.slot, args.output, args.limit_per_task, args.manifest, args.resume)
    print(json.dumps({k: result[k] for k in ['status', 'samples_checked', 'successful', 'failures', 'gpu_started']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
