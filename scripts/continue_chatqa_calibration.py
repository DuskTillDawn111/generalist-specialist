"""Wait quietly for one pinned training run, then perform fresh calibration once."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


def save(path, data):
    data['updated_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def alive(pid):
    path = Path('/proc') / str(pid) / 'stat'
    try:
        return path.read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    training = args.training.resolve()
    output = args.output.resolve()
    status_path = training / 'calibration-continuation.json'
    with (training / 'calibration-continuation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists():
            prior = json.loads(status_path.read_text())
            if prior['status'] != 'WAITING_FOR_TRAINING':
                raise RuntimeError('Continuation was already launched; inspect its evidence before retrying')
        status = {'status': 'WAITING_FOR_TRAINING', 'training': str(training),
                  'output': str(output), 'pool_admitted': False}
        save(status_path, status)
        while True:
            state = json.loads((training / 'state.json').read_text())
            if state['status'] == 'FAILED':
                status.update(status='TRAINING_FAILED', error=state.get('error'))
                save(status_path, status)
                return 1
            running = alive(state['pid'])
            if not running:
                if state['status'] != 'TRAINING_COMPLETE_NOT_ADMITTED':
                    status.update(status='TRAINING_PROCESS_LOST', training_status=state['status'])
                    save(status_path, status)
                    return 1
                break
            time.sleep(60)
        command = [args.python, str(Path(__file__).with_name('calibrate_trained_chatqa.py')),
                   '--training', str(training), '--output', str(output), '--execute']
        status.update(status='CALIBRATING', command=command)
        save(status_path, status)
        with (training / 'automatic-calibration.log').open('a') as log:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        result_path = output / 'state.json'
        status.update(status='CALIBRATION_FINISHED' if result.returncode == 0 else 'CALIBRATION_FAILED',
                      returncode=result.returncode,
                      result=json.loads(result_path.read_text()) if result_path.exists() else None)
        save(status_path, status)
        return result.returncode


if __name__ == '__main__':
    sys.exit(main())
