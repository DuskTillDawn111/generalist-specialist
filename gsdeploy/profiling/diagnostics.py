"""Small frozen native baselines and checked single-factor contrasts.

These jobs cannot grant admission, run a system workload, or tune on the
independent quality set. Free visual answers remain observational text.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from ..backends.compact import CompactAdapter, fingerprint
from .compact_experiments import execute_job
from .locking import gpu_lease
from .prepare import digest
from .runner import save

ROOT = Path(__file__).resolve().parents[2]


def _without_path(value, path):
    value = copy.deepcopy(value)
    target = value
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target.pop(path[-1], None)
    return value


def validate_contrasts(cases):
    """Every non-root case changes exactly its named factor, on identical inputs."""
    seen = {}
    for case in cases:
        if case['id'] in seen:
            raise ValueError('Duplicate diagnostic case')
        if not case['samples'] or len({s['id'] for s in case['samples']}) != len(case['samples']):
            raise ValueError('Empty or duplicate diagnostic samples')
        if len({s['task'] for s in case['samples']}) != 1:
            raise ValueError('Each diagnostic contrast has one task')
        if case['policy'].get('diagnostic_only') is not True:
            raise ValueError('Diagnostic case must declare diagnostic_only=true')
        parent_id, factor = case.get('parent'), case.get('factor')
        if parent_id is None:
            if factor is not None:
                raise ValueError('Root baseline cannot declare a contrast factor')
        else:
            if parent_id not in seen:
                raise ValueError('Diagnostic parent must precede its child')
            if not isinstance(factor, list) or len(factor) < 2 or factor[0] not in {'policy', 'execution'}:
                raise ValueError('A single explicit policy/execution factor is required')
            allowed = (
                (factor == ['policy', 'phi_prompt_format'] and case['candidate']['model'] == 'microsoft/Phi-4-multimodal-instruct')
                or (len(factor) == 2 and factor[0] == 'execution' and factor[1] in {'dtype', 'quantization', 'attention_implementation'})
                or (len(factor) == 3 and factor[:2] == ['execution', 'memory_optimizations'])
                or (len(factor) == 4 and factor[:2] == ['policy', 'tasks'] and factor[2] in {'text', 'image', 'ocr', 'asr'}
                    and factor[3] in {'instruction', 'deepseek_prompt', 'generative_decoding', 'catalog_presentation',
                                     'assistant_prefix', 'class_label_surface', 'chatqa_prompt_format', 'deepseek_image_preset',
                                     'deepseek2_image_preset'})
                or (len(factor) == 5 and factor[:2] == ['policy', 'tasks'] and factor[3] == 'generation_controls'
                    and factor[4] in {'repetition_penalty', 'no_repeat_ngram_size'})
                or (len(factor) == 5 and factor[:2] == ['policy', 'model_generation_controls']
                    and factor[4] in {'num_beams', 'repetition_penalty', 'no_repeat_ngram_size'}))
            if not allowed:
                raise ValueError('Factor must name one supported setting, not a whole task/protocol')
            task = case['samples'][0]['task']
            if factor[:2] == ['policy', 'tasks'] and factor[2] != task:
                raise ValueError('Contrast factor must affect the measured task')
            if factor[:2] == ['policy', 'model_generation_controls'] and (
                    factor[2] != case['candidate']['model'] or factor[3] != task):
                raise ValueError('Native decoding factor must affect the measured model/task')
            parent = seen[parent_id]
            if case['samples'] != parent['samples']:
                raise ValueError('Contrasts must use the identical frozen input cohort')
            if any(case['candidate'][k] != parent['candidate'][k] for k in ['model', 'revision', 'slot']):
                raise ValueError('A model change requires a separate root baseline')
            a = {k: parent[k] for k in ['policy', 'execution']}
            b = {k: case[k] for k in ['policy', 'execution']}
            if a == b or _without_path(a, factor) != _without_path(b, factor):
                raise ValueError('Contrast changes more than its declared single factor')
        seen[case['id']] = case
    return seen


def prepare(cases, base_manifest_path, output, extra_sources=()):
    """Prepare only requested small calibration cases; never create hidden GPU jobs."""
    validate_contrasts(cases)
    base = json.loads(Path(base_manifest_path).read_text())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases_path = output / 'cases.json'
    save(cases_path, cases)
    manifest = {'schema': 'gsdeploy.compact_jobs.v1', 'diagnostic_only': True,
                'status': 'CPU_PREPARED_GPU_DISABLED', 'gpu_started': False,
                'runtime_catalog': None, 'utility': None, 'memory_validation': False,
                'workers': {}, 'jobs': [], 'study': base['study'], 'sources': {},
                'code_hashes': {str(p.resolve()): digest(p) for p in ROOT.glob('gsdeploy/**/*.py')}}
    for path in [Path(base_manifest_path), cases_path, *map(Path, extra_sources)]:
        manifest['sources'][str(path.resolve())] = digest(path)
    for case in cases:
        name, candidate = case['id'], copy.deepcopy(case['candidate'])
        worker = base['workers'][candidate['slot']]
        template = json.loads(Path(worker['spec']).read_text())
        original_policy = json.loads(Path(template['resources']['protocol']).read_text())
        for key in ['quality_gate', 'normalization_references', 'completion']:
            if case['policy'][key] != original_policy[key]:
                raise ValueError('Diagnostics cannot modify scientific gates or utility')
        for task in original_policy['tasks']:
            if case['policy']['tasks'][task]['max_new_tokens'] != original_policy['tasks'][task]['max_new_tokens']:
                raise ValueError('Diagnostics retain the frozen output budgets')
        candidate['precision'] = 'diagnostic execution SHA256 ' + fingerprint(case['execution'])
        folder = output / name
        policy_path = folder / 'protocol.json'
        save(policy_path, case['policy'])
        resources = copy.deepcopy(template['resources'])
        resources.update(protocol=str(policy_path), request_output=str(folder / 'request-temporary'))
        adapter = CompactAdapter(case['policy'], json.loads(Path(resources['classmap']).read_text()))
        for sample in case['samples']:
            request = adapter.build(candidate, sample)
            if 'input' in sample and digest(sample['input']) != sample['input_sha256']:
                raise ValueError('Diagnostic media hash mismatch: ' + sample['id'])
            if request['input_truncation']:
                raise ValueError('Diagnostic inputs cannot be truncated')
        spec = {**template, 'id': name, 'candidate': candidate,
                'execution': case['execution'], 'resources': resources}
        spec_path = folder / 'worker.json'
        save(spec_path, spec)
        manifest['workers'][name] = {'spec': str(spec_path), 'spec_sha256': digest(spec_path),
                                     'python': worker['python'], 'tasks': [case['samples'][0]['task']]}
        job = {'id': name, 'phase': 'diagnostic', 'slot': candidate['slot'],
               'python': worker['python'], 'spec': str(spec_path), 'samples': case['samples'],
               'warmup_samples': [], 'replicas': 1, 'rounds': 1, 'exact_cohort': True,
               'minimum_seconds': 0, 'minimum_requests': len(case['samples']), 'requires': [],
               'parent': case.get('parent'), 'factor': case.get('factor'), 'diagnostic_only': True,
               'notes': ['No warmup gate: invalid/empty native answers are retained as diagnostic observations.',
                         'Single closed-loop diagnostic round; no serving-capacity or admission claim.']}
        job_path = folder / 'job.json'
        save(job_path, job)
        manifest['jobs'].append({'id': name, 'phase': 'diagnostic', 'path': str(job_path), 'sha256': digest(job_path)})
        for path in [policy_path, Path(resources['classmap']), Path(template['resources']['protocol'])]:
            manifest['sources'][str(path.resolve())] = digest(path)
    save(output / 'experiment-manifest.json', manifest)
    return manifest


def collect(entry, output):
    """Keep collection/completion/quality failures distinct, including partial runs."""
    from ..tasks import summarize_quality
    job = json.loads(Path(entry['path']).read_text())
    folder = Path(output) / entry['id']
    profile_path = folder / 'profile.json'
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    records = []
    for path in folder.glob('*-round*.jsonl'):
        records.extend(json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n'))
    ids = [r['sample_id'] for r in records]
    complete = profile.get('status') == 'COMPLETE' and len(ids) == len(job['samples']) and set(ids) == {s['id'] for s in job['samples']}
    if complete:
        spec = json.loads(Path(job['spec']).read_text())
        policy = json.loads(Path(spec['resources']['protocol']).read_text())
        adapter = CompactAdapter(policy, json.loads(Path(spec['resources']['classmap']).read_text()))
        samples = {s['id']: s for s in job['samples']}
        verified = []
        for record in records:
            sample = samples[record['sample_id']]
            request = adapter.build(spec['candidate'], sample)
            if record.get('request_sha256') != fingerprint(request) or record.get('protocol_sha256') != adapter.policy_sha256:
                raise ValueError('Diagnostic request/protocol identity changed')
            if 'input' in sample and record.get('media_binding', {}).get('sha256') != sample['input_sha256']:
                raise ValueError('Diagnostic media binding was not verified')
            observation = {k: record[k] for k in ['finish_reason', 'generated_tokens', 'terminal_token_id', 'eos_token_ids']}
            if 'class_scores' in record:
                observation['class_scores'] = record['class_scores']
            parsed = adapter.finish(request, sample, record['raw_output'], evaluate_quality=False, **observation)
            verified.append({**record, **parsed})
        records = verified
    observational = any(r.get('quality_eligible') is False for r in records)
    result = {'collection_complete': bool(complete), 'profile_status': profile.get('status', 'NO_PROFILE'),
              'expected_requests': len(job['samples']), 'returned_requests': len(records),
              'valid_completions': sum(r.get('success') is True for r in records),
              'failure_stage': profile.get('failure_stage'), 'error': profile.get('error'),
              'quality_summary': None, 'observational_only': observational,
              'parent': job.get('parent'), 'factor': job.get('factor'),
              'pool_admitted': False, 'scope': 'calibration_diagnostic_not_admission'}
    if complete and not observational:
        result['quality_summary'] = summarize_quality(job['samples'], records)
    result['outputs'] = [{k: r.get(k) for k in ['sample_id', 'raw_output', 'prediction', 'success', 'failure_reason',
                                                  'finish_reason', 'generated_tokens', 'media_binding', 'effective_generation']} for r in records]
    save(folder / 'diagnostic-result.json', result)
    return result


def run(manifest_path, output):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('diagnostic_only') is not True or any(j['phase'] != 'diagnostic' for j in manifest['jobs']):
        raise ValueError('This runner accepts only diagnostic manifests')
    for path, expected in {**manifest['sources'], **manifest['code_hashes']}.items():
        if digest(path) != expected:
            raise ValueError('Frozen diagnostic source changed: ' + path)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'diagnostic-queue-state.json'
    if state_path.exists() or any((output / j['id']).exists() for j in manifest['jobs']):
        raise FileExistsError('Existing diagnostic evidence must be preserved')
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
             'manifest': str(manifest_path), 'manifest_sha256': digest(manifest_path),
             'current_case': None, 'completed': [], 'failed': [], 'blocked': [], 'results': {},
             'pool_admitted': False, 'runtime_catalog': None, 'utility': None}
    with gpu_lease():
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if occupied:
            raise RuntimeError('GPU is occupied: ' + occupied)
        state['hardware'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,power.limit,driver_version', '--format=csv,noheader'], text=True).strip()
        try:
            for entry in manifest['jobs']:
                job = json.loads(Path(entry['path']).read_text())
                parent = job.get('parent')
                if parent and parent not in state['completed']:
                    state['blocked'].append(entry['id'])
                    save(state_path, state)
                    continue
                state['current_case'] = entry['id']
                save(state_path, state)
                print(datetime.now(timezone.utc).isoformat(), 'START', entry['id'], flush=True)
                try:
                    with (output / (entry['id'] + '.log')).open('x') as log:
                        command = [sys.executable, '-m', 'gsdeploy.profiling.compact_experiments', 'run',
                                   '--manifest', str(manifest_path), '--job', entry['id'],
                                   '--output', str(output / entry['id']), '--execute']
                        process = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                                 env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
                    result = collect(entry, output)
                    result['returncode'] = process.returncode
                    state['results'][entry['id']] = result
                    key = 'completed' if process.returncode == 0 and result['collection_complete'] else 'failed'
                    state[key].append(entry['id'])
                    print(datetime.now(timezone.utc).isoformat(), key.upper(), entry['id'], flush=True)
                finally:
                    save(state_path, state)
            state['status'] = 'COMPLETE_WITH_COLLECTION_FAILURES' if state['failed'] or state['blocked'] else 'COMPLETE_DIAGNOSTICS_NOT_ADMITTED'
        except BaseException as error:
            state.update(status='CONTROLLER_FAILED', error=type(error).__name__ + ': ' + str(error))
            raise
        finally:
            state['current_case'] = None
            state['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('GPU execution requires --execute; importing this module is CPU-safe')
    run(args.manifest, args.output)


if __name__ == '__main__':
    main()
