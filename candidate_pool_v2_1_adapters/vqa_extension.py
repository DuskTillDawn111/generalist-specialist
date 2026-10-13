"""Versioned v2.1 ScienceQA extension; original v2 handlers remain unchanged.
Promotes the exact Module 8A revalidated function body without changing decoding.
"""
import ast, json, os, runpy, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'joint_deploy'), str(ROOT / 'benchmark/scripts')]
import worker_protocol
original_run_worker = worker_protocol.run_worker

def versioned_run_worker(worker_name, handler, ready_payload=None):
    # Load the exact pure semantic functions without importing Qwen-specific
    # transformer classes into the Phi/MiniCPM runtime environments.
    semantic_source = ROOT / 'benchmark/scripts/eval_qwen_scienceqa.py'
    tree = ast.parse(semantic_source.read_text())
    names = {'option_letter', 'valid_options', 'normalize_option', 'validate_sample', 'build_instruction'}
    pure = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    vqa = types.ModuleType('frozen_scienceqa_semantics')
    exec(compile(pure, str(semantic_source), 'exec'), vqa.__dict__)
    # Functions defined by runpy retain the original worker globals.
    g = handler.__globals__
    task_prompt = json.loads((ROOT / 'benchmark/prompts/task_prompts.json').read_text())['visual_qa']['prompt']
    def versioned_handler(request):
        if request['task_type'] != 'visual_qa':
            return handler(request)
        sample = request['sample']
        vqa.validate_sample(sample)
        instruction = vqa.build_instruction(task_prompt, sample)
        if worker_name == 'phi4':
            module = g['image_mod']
            prompt = module.build_chat_prompt(g['processor'].tokenizer, instruction)
            result = module.generate_one(g['model'], g['processor'], prompt,
                                         sample['input'], g['device'], max_new_tokens=1)
            raw = result['response']
        elif worker_name == 'minicpm':
            result = g['image_mod'].run_one(g['model'], instruction, sample['input'], max_new_tokens=1)
            raw = result['raw_response']
        else:
            raise ValueError(worker_name)
        prediction = vqa.normalize_option(raw)
        reference = vqa.normalize_option(sample['label'])
        return {'model': worker_name, 'prediction': prediction, 'reference': reference,
                'raw_response': raw, 'correct': prediction == reference,
                'valid_closed_set_output': prediction in vqa.valid_options(sample),
                'capability_version': 'v2.1'}
    payload = dict(ready_payload or {})
    payload['tasks'] = payload.get('tasks', []) + ['visual_qa']
    payload['capability_version'] = 'v2.1'
    original_run_worker(worker_name, versioned_handler, payload)

worker_protocol.run_worker = versioned_run_worker
runpy.run_path(os.environ['AUDIT_ORIGINAL_WORKER'], run_name='__main__')
