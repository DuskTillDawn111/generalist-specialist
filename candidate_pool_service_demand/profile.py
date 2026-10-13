"""Run the unchanged Stage-1 main with frozen v2 Worker launch configurations.

Only launch registration, stdout logging, isolation monitoring and shutdown are
adapted. Historical sample selection, warmup, submission, timers, raw format and
mean summary code execute directly from run_active_single_profile.py.
"""
import argparse
import json
from pathlib import Path
import queue
import sys
import threading
import time

from candidate_pool_audit.load_test import gpu_snapshot
from candidate_pool_audit.registry import registry
from .prepare import ROOT, OUT, read, write, reference

sys.path.insert(0, str(ROOT / 'joint_deploy'))
import run_active_single_profile as stage1


def run(args):
    pending = read(OUT / 'pending_service_demand_paths.json')['paths']
    authorized = {(p['model_id'], p['task']) for p in pending}
    if (args.target_worker, args.target_task) not in authorized or args.target_count != 30:
        raise ValueError('Only prepared missing verified paths with original count=30 may be measured')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or Path(str(output) + '.validation.json').exists():
        raise ValueError('Never overwrite an independent measurement attempt')
    audit = {'model_id': args.target_worker, 'task': args.target_task, 'protocol': 'active_single_profile_v1',
             'replica_count': 1, 'standalone_only': True, 'status': 'MEASUREMENT_FAILED',
             'before_load': gpu_snapshot(), 'workers_started': 0, 'load_succeeded': False,
             'events': [], 'GPU_process_samples': [], 'foreign_GPU_processes': [], 'requests': [],
             'launcher': {k: str(v) for k, v in registry()[args.target_worker].items()},
             'original_profiler': reference(ROOT / 'joint_deploy/run_active_single_profile.py')}
    if audit['before_load']['processes']:
        write(str(output) + '.validation.json', audit)
        raise RuntimeError('GPU occupied; refusing to start any worker')
    workers = []
    stop_monitor = threading.Event()

    class StandaloneWorker(stage1.SharedWorker):
        def start(self):
            if workers:
                raise RuntimeError('At most one worker instance permitted')
            workers.append(self)
            audit['workers_started'] += 1
            super().start()
            audit['pid'] = self.process.pid

            def monitor():
                while not stop_monitor.is_set():
                    try:
                        snapshot = gpu_snapshot(self.process.pid)
                        audit['GPU_process_samples'].append(snapshot)
                        foreign = [p for p in snapshot['processes'] if p['pid'] != self.process.pid]
                        if foreign:
                            audit['foreign_GPU_processes'].append(foreign)
                            self.process.terminate()
                            return
                    except Exception as error:
                        audit['monitor_error'] = repr(error)
                        self.process.terminate()
                        return
                    stop_monitor.wait(1.)

            self.monitor = threading.Thread(target=monitor, daemon=True)
            self.monitor.start()

        def _reader(self):
            with Path(str(output) + '.worker.log').open('w') as log:
                for line in self.process.stdout:
                    log.write(line)
                    log.flush()
                    text = line.rstrip()
                    if text.startswith(('READY|', 'RESULT|', 'ERROR|', 'BYE|')):
                        kind, serialized = text.split('|', 1)
                        payload = json.loads(serialized)
                        audit['events'].append({'kind': kind, 'payload': payload})
                        if kind == 'READY':
                            audit['load_succeeded'] = True
                        self.event_queue.put((self.name, kind, payload))

        def send(self, payload):
            if payload.get('op') != 'shutdown':
                audit['requests'].append(payload)
            super().send(payload)

        def terminate(self):
            # The measured region has ended. Wait for shutdown/release before any
            # next standalone job; do not leave the historical terminate race.
            stop_monitor.set()
            if getattr(self, 'monitor', None):
                self.monitor.join(timeout=5)
            forced = False
            if self.process and self.process.poll() is None:
                self.shutdown()
                try:
                    self.process.wait(timeout=30)
                except Exception:
                    forced = True
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except Exception:
                        self.process.kill()
                        self.process.wait(timeout=10)
            audit['shutdown'] = {'exit_code': self.process.returncode if self.process else None, 'forced': forced}

    stage1.SharedWorker = StandaloneWorker
    stage1.WORKERS = registry()
    stage1.MODEL_TASKS = {m: {task for name, task in authorized if name == m}
                          for m in {name for name, _ in authorized}}
    error = None
    try:
        stage1.main()
    except BaseException as caught:
        error = caught
        audit['error'] = f'{type(caught).__name__}: {caught}'
    finally:
        stop_monitor.set()
        for worker in workers:
            if worker.process and worker.process.poll() is None:
                worker.terminate()
        audit['after_unload'] = []
        for _ in range(3):
            time.sleep(.3)
            audit['after_unload'].append(gpu_snapshot(audit.get('pid')))
        audit['GPU_memory_released'] = all(not s['processes'] and
            s['device_used_mib'] <= audit['before_load']['device_used_mib'] + 2 for s in audit['after_unload'])
        if (error is None and not audit['foreign_GPU_processes'] and not audit.get('monitor_error')
            and audit['GPU_memory_released'] and audit.get('shutdown') == {'exit_code': 0, 'forced': False}):
            audit['status'] = 'measured'
        audit['capability_revalidation_required'] = bool(audit['load_succeeded'] and
              any(event['kind'] == 'ERROR' for event in audit['events']))
        write(str(output) + '.validation.json', audit)
    if error:
        raise error
    if audit['status'] != 'measured':
        raise RuntimeError('Standalone isolation/unload validation failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', required=True)
    parser.add_argument('--target-worker', required=True)
    parser.add_argument('--target-task', required=True)
    parser.add_argument('--target-count', type=int, default=30)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout-s', type=float, default=3600.)
    run(parser.parse_args())
