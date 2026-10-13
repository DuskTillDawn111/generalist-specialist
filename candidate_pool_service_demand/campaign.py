"""Sequential 14-path x 3-repetition standalone campaign, no search/replicas."""
import json
from pathlib import Path
import subprocess
import sys

from .prepare import ROOT, OUT, POOL, read, write


def main():
    pending = read(OUT / 'pending_service_demand_paths.json')['paths']
    attempts = []
    for path in pending:
        name, task = path['model_id'], path['task']
        for rep in range(1, 4):
            output = OUT / f'raw_measurements/r{rep}/active_single_{name}_{task}_n30.jsonl'
            validation = Path(str(output) + '.validation.json')
            if validation.exists():
                attempts.append({'model_id': name, 'task': task, 'repetition': rep,
                                 'status': read(validation)['status'], 'reuse_saved_attempt': True})
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-m', 'candidate_pool_service_demand.profile',
                       '--pool', str(POOL), '--target-worker', name, '--target-task', task,
                       '--target-count', '30', '--output', str(output), '--timeout-s', '3600']
            print(json.dumps({'event': 'starting', 'model_id': name, 'task': task, 'repetition': rep}), flush=True)
            with Path(str(output) + '.profile.log').open('w') as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            result = read(validation) if validation.exists() else {'status': 'MEASUREMENT_FAILED'}
            entry = {'model_id': name, 'task': task, 'repetition': rep, 'return_code': completed.returncode,
                     'status': result['status'], 'capability_revalidation_required': result.get('capability_revalidation_required', False),
                     'output': str(output), 'command': command}
            attempts.append(entry)
            write(OUT / 'campaign_progress.json', {'attempts': attempts, 'planned_repetitions': 42})
            print(json.dumps(dict(event='completed', **entry)), flush=True)
            if not result.get('GPU_memory_released', False):
                raise RuntimeError('GPU release not verified; cannot start another model')
    write(OUT / 'campaign_progress.json', {'attempts': attempts, 'planned_repetitions': 42})


if __name__ == '__main__':
    main()
