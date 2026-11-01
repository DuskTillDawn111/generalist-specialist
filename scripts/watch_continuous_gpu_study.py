"""Quiet low-frequency checkpoints; the active goal handles experiment decisions."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/continuous-gpu-study-20261018'


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def checkpoint():
    plan = json.loads((FOLDER / 'continuation-plan.json').read_text())
    pointer = json.loads(Path(plan['active_pipeline_pointer']).read_text())
    state = json.loads(Path(pointer['state']).read_text())
    process = Path('/proc') / str(pointer['pid'])
    command = process.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode() if process.exists() else ''
    alive = 'run_chatqa_replacement.py' in command and pointer['output'] in command
    inventory = json.loads((ROOT / 'artifacts/qa-replacement-20261018/resources/resource-inventory.json').read_text())
    checkpoint = inventory['models']['text']
    completed = total = 0
    files = []
    for entry in checkpoint['files']:
        if not entry['name'].endswith('.safetensors'):
            continue
        path = Path(checkpoint['path']) / entry['name']
        total += entry['size']
        count = entry['size'] if entry['status'] == 'VERIFIED' else 0
        ranges = path.with_name(path.name + '.ranges.json')
        if count == 0 and ranges.exists():
            ranges_state = json.loads(ranges.read_text())
            count = sum(min(ranges_state['chunk'], entry['size'] - i * ranges_state['chunk'])
                        for i in ranges_state['complete'])
        completed += count
        files.append({'name': entry['name'], 'status': entry['status'], 'completed_bytes': count})
    terminal = state['status'] in {'FAILED', 'CALIBRATION_REJECTED', 'CALIBRATION_COMPLETE_NOT_ADMITTED'}
    current = plan.get('current_experiment')
    activity = None
    if current:
        status_file = current.get('queue_state') or current.get('status_path')
        if status_file and Path(status_file).exists():
            current_state = json.loads(Path(status_file).read_text())
            activity_pid = current_state.get('pid', current['pid'])
            proc = Path('/proc') / str(activity_pid)
            cmd = proc.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode() if proc.exists() else ''
            activity = {'status': current_state['status'], 'pid': activity_pid, 'process_alive': bool(cmd),
                        'state': status_file, 'current_case': current_state.get('current_case'),
                        'completed_cases': len(current_state.get('completed', [])),
                        'failed_cases': len(current_state.get('failed', [])) if isinstance(current_state.get('failed', []), list) else None}
    snapshot = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'watcher_pid': os.getpid(),
                'task_status': plan['status'], 'pipeline_status': state['status'], 'pipeline_pid': pointer['pid'],
                'pipeline_process_alive': alive, 'gpu_started': state.get('gpu_started'),
                'completed_weight_bytes': completed, 'total_weight_bytes': total, 'files': files,
                'requires_agent_review': terminal or not alive,
                'next_action': 'review_calibration_and_continue_pool_repair' if terminal else
                    'inspect_interrupted_pipeline' if not alive else 'wait_for_next_scheduled_check',
                'pool_admitted': plan.get('pool_admitted', False),
                'system_experiments_complete': plan.get('system_experiments_complete', False)}
    if activity is not None:
        snapshot['current_experiment'] = activity
        active = (activity['status'] in {'STARTING', 'TRAINING', 'GPU_CALIBRATION_RUNNING'}
                  or activity['status'].startswith(('RUNNING', 'WAITING_FOR_')))
        snapshot['requires_agent_review'] = not (active and activity['process_alive'])
        snapshot['next_action'] = ('wait_for_next_scheduled_check' if active and activity['process_alive']
                                   else 'review_current_phase_and_continue_task')
    save(FOLDER / 'latest-checkpoint.json', snapshot)
    with (FOLDER / 'checkpoints.jsonl').open('a') as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False) + '\n')
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    FOLDER.mkdir(parents=True, exist_ok=True)
    with (FOLDER / '.watcher.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                result = checkpoint()
                if args.once:
                    print(json.dumps(result, ensure_ascii=False))
                    return
                if result['task_status'] == 'COMPLETE':
                    return
            except Exception as error:
                save(FOLDER / 'watcher-error.json', {'time_utc': datetime.now(timezone.utc).isoformat(),
                     'error': type(error).__name__ + ': ' + str(error), 'pid': os.getpid()})
                if args.once:
                    raise
            # This detached process, not a model turn, performs the timer wait.
            time.sleep(900)


if __name__ == '__main__':
    main()
