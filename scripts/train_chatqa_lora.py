"""One frozen train-only QLoRA run, with durable progress and local artifacts."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsdeploy.profiling.prepare import digest
from gsdeploy.profiling.runner import save
from gsdeploy.profiling.locking import gpu_lease


def collate(rows, torch, pad_id):
    size = max(len(row['input_ids']) for row in rows)
    ids, labels, masks = [], [], []
    for row in rows:
        if len(row['input_ids']) != len(row['labels']):
            raise ValueError('Training labels do not align with the complete native prompt')
        padding = size - len(row['input_ids'])
        ids.append(row['input_ids'] + [pad_id] * padding)
        labels.append(row['labels'] + [-100] * padding)
        masks.append([1] * len(row['input_ids']) + [0] * padding)
    return {'input_ids': torch.tensor(ids, dtype=torch.long), 'attention_mask': torch.tensor(masks, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)}


def run(recipe_path, output):
    recipe_path, output = Path(recipe_path).resolve(), Path(output).resolve()
    recipe = json.loads(recipe_path.read_text())
    for path, expected in recipe['source_hashes'].items():
        if digest(path) != expected:
            raise ValueError('Training source changed: ' + path)
    if digest(recipe['data']) != recipe['data_sha256'] or digest(recipe['base_worker']) != recipe['base_worker_sha256']:
        raise ValueError('Frozen training data or base worker changed')
    spec = json.loads(Path(recipe['base_worker']).read_text())
    if spec['candidate']['model'] != 'nvidia/Llama3-ChatQA-1.5-8B':
        raise ValueError('This training recipe supports only the explicit QA replacement')
    from gsdeploy.backends.compact_memory import allocator_environment
    os.environ.update(allocator_environment(spec['execution']))
    os.environ['OMP_NUM_THREADS'] = '1'
    rows = [json.loads(line) for line in Path(recipe['data']).read_text().splitlines()]
    if len(rows) != recipe['samples'] or any(len(r['input_ids']) > recipe['max_sequence_tokens'] for r in rows):
        raise ValueError('Training examples differ from the fixed complete-token cohort')
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'STARTING', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
             'optimizer_steps': 0, 'samples_seen': 0, 'pool_admitted': False, 'training_only': True,
             'recipe_sha256': digest(recipe_path)}
    state_path = output / 'state.json'
    save(state_path, state)
    sources = {str(p.resolve()): digest(p) for pattern in ['gsdeploy/**/*.py', 'scripts/train_chatqa_lora.py']
               for p in ROOT.glob(pattern)}
    save(output / 'frozen-run-sources.json', sources)
    begin = time.monotonic()
    try:
        with gpu_lease():
            occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            if occupied:
                raise RuntimeError('GPU occupied before training: ' + occupied)
            state['hardware'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,power.limit,driver_version', '--format=csv,noheader'],text=True).strip()
            # Imports can inspect CUDA (notably bitsandbytes via PEFT). Acquire
            # the lease and inspect other processes before importing them.
            import torch
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
            from transformers import get_linear_schedule_with_warmup
            from gsdeploy.backends.compact import CompactAdapter
            from gsdeploy.backends.compact_native import load_native
            random.seed(recipe['seed'])
            torch.manual_seed(recipe['seed'])
            torch.cuda.manual_seed_all(recipe['seed'])
            policy = json.loads(Path(spec['resources']['protocol']).read_text())
            adapter = CompactAdapter(policy, json.loads(Path(spec['resources']['classmap']).read_text()))
            executor = load_native(adapter, spec['candidate'], spec['execution'], spec['resources']['checkpoint'], allow_gpu=True)
            model = prepare_model_for_kbit_training(executor.model, use_gradient_checkpointing=True,
                                                    gradient_checkpointing_kwargs={'use_reentrant': False})
            options = recipe['lora']
            model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=options['r'],
                lora_alpha=options['alpha'], lora_dropout=options['dropout'], target_modules=options['target_modules']))
            model.config.use_cache = False
            trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            if not trainable or any('.lora_A.' not in name and '.lora_B.' not in name for name, _ in trainable):
                raise RuntimeError('Only the declared new QA LoRA parameters may train')
            if any(p.device.type != 'cuda' for p in model.parameters()):
                raise RuntimeError('Training CPU/disk model offload is outside this recipe')
            state.update(status='TRAINING', trainable_parameters=sum(p.numel() for _, p in trainable),
                         total_parameters=sum(p.numel() for p in model.parameters()))
            save(state_path, state)
            optimizer = torch.optim.AdamW([p for _, p in trainable], lr=recipe['learning_rate'])
            generator = torch.Generator().manual_seed(recipe['seed'])
            loader = torch.utils.data.DataLoader(rows, batch_size=recipe['microbatch'], shuffle=True, generator=generator,
                    num_workers=0, collate_fn=lambda batch: collate(batch, torch, executor.tokenizer.eos_token_id))
            accumulation = recipe['gradient_accumulation']
            total_steps = math.ceil(len(loader) / accumulation) * recipe['epochs']
            scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(total_steps * recipe['warmup_fraction'])), total_steps)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            group_loss = 0.0
            for epoch in range(recipe['epochs']):
                for index, batch in enumerate(loader):
                    group_size = min(accumulation, len(loader) - (index // accumulation) * accumulation)
                    batch_size = int(batch['input_ids'].shape[0])
                    batch = {key: value.to('cuda:0') for key, value in batch.items()}
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = model(**batch, use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite QA training loss')
                    (loss / group_size).backward()
                    group_loss += float(loss.detach().cpu()) / group_size
                    state['samples_seen'] += batch_size
                    if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                        grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], recipe['max_grad_norm'])
                        if not torch.isfinite(grad_norm):
                            raise RuntimeError('Non-finite adapter gradient norm')
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        state['optimizer_steps'] += 1
                        state.update(loss=group_loss, elapsed_seconds=time.monotonic() - begin,
                                     updated_utc=datetime.now(timezone.utc).isoformat())
                        with (output / 'losses.jsonl').open('a') as handle:
                            handle.write(json.dumps({k:state[k] for k in ['optimizer_steps','samples_seen','loss','elapsed_seconds']})+'\n')
                        if state['optimizer_steps'] % 10 == 0:
                            save(state_path, state)
                            print(json.dumps({k:state[k] for k in ['optimizer_steps','samples_seen','loss','elapsed_seconds']}), flush=True)
                        if state['optimizer_steps'] % 100 == 0:
                            checkpoint = output / ('checkpoint-' + str(state['optimizer_steps']))
                            model.save_pretrained(checkpoint, safe_serialization=True)
                        group_loss = 0.0
            torch.cuda.synchronize()
            model.eval()
            final = output / 'adapter'
            model.save_pretrained(final, safe_serialization=True)
            for path, expected in sources.items():
                if digest(path) != expected:
                    raise ValueError('Training code changed during the frozen run: ' + path)
            files = [{'name': name, 'size': (final/name).stat().st_size, 'sha256': digest(final/name)}
                     for name in ['adapter_config.json','adapter_model.safetensors']]
            adapter_manifest = {'schema': 'gsdeploy.local_qa_adapter.v1', 'model': spec['candidate']['model'],
                'revision': spec['candidate']['revision'], 'path': str(final), 'files': files,
                'recipe': str(recipe_path), 'recipe_sha256': digest(recipe_path), 'data_sha256': recipe['data_sha256'],
                'samples_seen': state['samples_seen'], 'optimizer_steps': state['optimizer_steps'],
                'scope': 'official train-only replacement adapter, not model-pool admission'}
            save(output / 'adapter-manifest.json', adapter_manifest)
            state.update(status='TRAINING_COMPLETE_NOT_ADMITTED', adapter_manifest=str(output/'adapter-manifest.json'),
                         adapter_manifest_sha256=digest(output/'adapter-manifest.json'),
                         peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as error:
        state.update(status='FAILED', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        state['elapsed_seconds'] = time.monotonic() - begin
        save(state_path, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('GPU training requires --execute')
    run(args.recipe, args.output)
