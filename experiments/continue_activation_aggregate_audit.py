"""Audit each completed aggregate capture once, preserving failures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'solo-base', 'reference', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--queue-pid', required=True, type=int)
    a = p.parse_args()
    base, solo, output = a.base.resolve(), a.solo_base.resolve(), a.output.resolve()
    queue = json.loads((base/'queue.json').read_text())
    paths = [Path(__file__).resolve(), a.reference.resolve(), base/'queue.json',
             ROOT/'experiments/audit_activation_aggregate_capture.py',
             ROOT/'experiments/analyze_activation_semantic.py', ROOT/'experiments/fit_activation_native.py',
             ROOT/'experiments/analyze_activation_trim_episode_energy.py',
             *[ROOT/'gsdeploy'/f for f in ('energy.py', 'quality.py', 'tasks.py', 'normalization.py')],
             ROOT/'gsdeploy/forecasting/native_training.py']
    output.mkdir(parents=True, exist_ok=False)
    sources = []
    for i, path in enumerate(paths):
        content = path.read_bytes()
        target = output/'source-snapshot'/str(i)/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        sources.append(dict(path=str(path), snapshot=str(target), sha256=hashlib.sha256(content).hexdigest()))
    save(output/'plan.json', dict(schema='gsdeploy.aggregate_capture_audit_continuation.v1',
        source_receipts=sources, captures=queue['captures'], queue_pid=a.queue_pid,
        baseline_directories=[str(solo/f'solo-round{i}') for i in (1, 2)],
        physical_predictor_qualified=False))
    state = dict(status='waiting_for_completed_captures', reports=[], physical_predictor_qualified=False)
    save(output/'state.json', state)
    def check():
        for receipt in sources:
            if any(hashlib.sha256(Path(receipt[k]).read_bytes()).hexdigest()!=receipt['sha256'] for k in ('path', 'snapshot')):
                raise ValueError('Frozen audit source changed: '+receipt['path'])
    try:
        for directory in map(Path, queue['captures']):
            while not (directory/'hardware.json').exists():
                upstream = json.loads(Path(queue['state']).read_text())
                if upstream.get('errors') or 'failed' in upstream['status']:
                    raise RuntimeError('Upstream capture failed: '+str(directory))
                try:
                    cmd = (Path('/proc')/str(a.queue_pid)/'cmdline').read_bytes()
                except FileNotFoundError:
                    cmd = b''
                if b'run_activation_queue.py' not in cmd:
                    if (directory/'hardware.json').exists():
                        break
                    raise RuntimeError('Queue ended before hardware receipt')
                time.sleep(2)
            check()
            state.update(status='auditing_completed_capture', current=str(directory))
            save(output/'state.json', state)
            report = output/(directory.name+'.json')
            command = [sys.executable, str(ROOT/'experiments/audit_activation_aggregate_capture.py'),
                '--baseline', str(solo/'solo-round1'), '--baseline', str(solo/'solo-round2'),
                '--capture', str(directory), '--reference', str(a.reference.resolve()), '--output', str(report)]
            with (output/(directory.name+'.log')).open('x') as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            check()
            result = json.loads(report.read_text())
            state['reports'].append(dict(path=str(report), issues=result['runs'][0]['issues'],
                changed_outputs=result['runs'][0]['changed_output_count']))
            state.update(status='waiting_for_completed_captures', current=None)
            save(output/'state.json', state)
        state['status'] = 'capture_audits_completed'
        save(output/'state.json', state)
    except BaseException as exc:
        state.update(status='capture_audit_failed', error=type(exc).__name__+': '+str(exc))
        save(output/'state.json', state)
        raise


if __name__ == '__main__':
    main()
