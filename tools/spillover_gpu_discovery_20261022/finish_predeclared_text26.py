"""Finish unattempted text2.6 conditions if the original queue stops.

Keep the original queue's stop and every attempted result. No retries or score
selection; independent conditions already in the frozen packet may continue.
An absent raw result or identity mismatch is an infrastructure stop.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/static-spillover-study-20261022/text26-development-v1'
SOURCE = ROOT / 'artifacts/static-study-gpu-20261022/execution-source-v5'
sys.path.insert(0, str(SOURCE))
from gsdeploy.static_study import validate_packet
sys.path.insert(0, str(ROOT / 'tools/spillover_discovery_20261022'))
import audit_text26_development as audit


def receipt(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def publish(path, value):
    temporary = path.with_suffix('.continuation.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def result_for(job, output, packet):
    path = output / 'run.json'
    if not path.is_file():
        raise RuntimeError('Missing raw result; never retry: ' + str(output))
    raw = json.loads(path.read_text())
    checked = audit.identity_checks(raw, job, packet)
    if not checked['verified']:
        raise ValueError('Executed identity differs: ' + job['labels'][0])
    utility = raw['evaluation']['utility']
    return dict(label=job['labels'][0], job_id=job['id'], raw=receipt(path),
                status=raw['status'], eligible=utility['eligible'], U=utility['U'],
                errors=raw['errors'], attempt=1, identity_verified=True)


def wait_for_idle_gpu():
    # The single-GPU experiment is serial. Leftover model workers must leave
    # the device, including those not named prepare_static_experiments.py.
    for _ in range(12):
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        if not result.stdout.strip():
            return
        time.sleep(5)
    raise RuntimeError('GPU still has compute processes; do not overlap experiments')


def main():
    packet_path = BASE / 'experiment-packet.json'
    packet = json.loads(packet_path.read_text())
    if not validate_packet(packet)['valid']:
        raise ValueError('Frozen packet validation failed')
    audit.verify_roster(packet)
    declaration_path = BASE / 'continuation-declaration.json'
    declaration = dict(schema='gsdeploy.text26_finish_predeclared.v1',
        declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        packet=receipt(packet_path), source=receipt(Path(__file__)),
        identity_audit_source=receipt(Path(audit.__file__)),
        policy='Continue only unattempted frozen-packet jobs after run-level invalidity; no retries, no score selection, no gate changes.',
        ordered_job_ids=[job['id'] for job in packet['jobs']], stability_rules_changed=False)
    with declaration_path.open('x') as stream:
        stream.write(json.dumps(declaration, indent=2) + '\n')
    progress = BASE / 'runs/queue-progress.json'
    output_progress = BASE / 'remaining-queue-progress.json'
    while True:
        state = json.loads(progress.read_text()) if progress.is_file() else None
        if state and state['status'] != 'RUNNING':
            break
        time.sleep(20)
    if state['status'] not in {'STOPPED_FOR_REVIEW', 'CALIBRATION_COMPLETE'}:
        raise RuntimeError('Unexpected original queue status: ' + state['status'])
    # The original parent publishes STOPPED immediately before exiting. Wait
    # until no native runner is alive before collecting any independent job.
    while True:
        processes = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True, check=True).stdout
        if not any('prepare_static_experiments.py run --packet' in line
                   for line in processes.splitlines()):
            break
        time.sleep(5)
    if not validate_packet(packet)['valid'] or receipt(packet_path) != declaration['packet']:
        raise ValueError('Packet changed while waiting')
    if receipt(Path(audit.__file__)) != declaration['identity_audit_source']:
        raise ValueError('Identity audit changed while waiting')
    snapshot = BASE / 'original-queue-completion-or-stop.json'
    with snapshot.open('x') as stream:
        stream.write(json.dumps(state, indent=2) + '\n')
    preserved = []
    remaining = []
    for job in packet['jobs']:
        output = BASE / 'runs' / (job['labels'][0] + '-attempt1')
        log = output.with_suffix('.log')
        if output.exists() or log.exists():
            preserved.append(result_for(job, output, packet))
        else:
            remaining.append(job)
    if len(preserved) != len(state['jobs']):
        raise ValueError('Original attempt roster does not match original queue')
    if [row['job_id'] for row in preserved] != [row['job_id'] for row in state['jobs']]:
        raise ValueError('Original queue identity/order mismatch')
    status = dict(schema='gsdeploy.text26_remaining_queue.v1', status='RUNNING',
        declaration=receipt(declaration_path), original_queue=receipt(snapshot),
        original_queue_status=state['status'], preserved_attempts=preserved, jobs=[],
        ordered_remaining_ids=[job['id'] for job in remaining])
    publish(output_progress, status)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.pop('CUDA_VISIBLE_DEVICES', None)
    try:
        for job in remaining:
            if not validate_packet(packet)['valid'] or receipt(packet_path) != declaration['packet']:
                raise ValueError('Frozen bindings changed')
            output = BASE / 'runs' / (job['labels'][0] + '-attempt1')
            log = output.with_suffix('.log')
            if output.exists() or log.exists():
                raise FileExistsError('Never retry/overwrite: ' + str(output))
            status['current'] = dict(label=job['labels'][0], job_id=job['id'], output=str(output))
            publish(output_progress, status)
            wait_for_idle_gpu()
            command = [sys.executable, '-u', str(SOURCE/'scripts/prepare_static_experiments.py'),
                'run', '--packet', str(packet_path), '--job', job['id'], '--output', str(output), '--execute']
            with log.open('x') as stream:
                done = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            row = result_for(job, output, packet)
            row['exit_code'] = done.returncode
            status['jobs'].append(row)
            publish(output_progress, status)
            print(json.dumps(row), flush=True)
            if done.returncode and row['status'] != 'failed':
                raise RuntimeError('Nonzero runner exit without a documented failed raw experiment')
        status.pop('current', None)
        status.update(status='ALL_PREDECLARED_ATTEMPTS_COLLECTED',
            all_attempts_eligible=all(row['eligible'] for row in preserved + status['jobs']))
        publish(output_progress, status)
    except Exception as exc:
        status.update(status='STOPPED_INFRASTRUCTURE_ERROR', error_type=type(exc).__name__, error=str(exc))
        publish(output_progress, status)
        raise


if __name__ == '__main__':
    main()
