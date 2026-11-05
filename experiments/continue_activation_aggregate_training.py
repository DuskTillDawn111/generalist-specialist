"""Wait for the live aggregate queue, then fit and replay once under receipts."""
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
    for name in ('base', 'solo-base', 'policy', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--queue-pid', required=True, type=int)
    args = p.parse_args()
    base, solo, output = args.base.resolve(), args.solo_base.resolve(), args.output.resolve()
    queue = json.loads((base/'queue.json').read_text())
    captures = [solo/f'solo-round{i}' for i in (1, 2)] + [Path(s) for s in queue['captures']]
    paths = {Path(__file__).resolve(), args.policy.resolve(), base/'queue.json', base/'design.json',
             ROOT/'experiments/fit_activation_aggregate_native.py', ROOT/'experiments/replay_activation_aggregate.py',
             ROOT/'experiments/fit_activation_native.py', ROOT/'experiments/review_activation_aggregate_accuracy.py',
             ROOT/'experiments/fit_activation_declared_idle_energy.py', ROOT/'experiments/collect_activation_training.py',
             ROOT/'gsdeploy/paper_training.py', ROOT/'gsdeploy/paper_capture.py', ROOT/'gsdeploy/energy.py'}
    paths.update((ROOT/'gsdeploy/forecasting').glob('*.py'))
    for capture in captures:
        paths.update(capture/(name+'.json') for name in ('context', 'plan', 'sources'))
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    for i, path in enumerate(sorted(paths)):
        target = output/'source-snapshot'/str(i)/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes()
        target.write_bytes(content)
        receipts.append(dict(path=str(path), snapshot=str(target), sha256=hashlib.sha256(content).hexdigest()))
    save(output/'analysis-plan.json', dict(schema='gsdeploy.aggregate_analysis_continuation.v1',
        captures=[str(p) for p in captures], source_receipts=receipts, queue_pid=args.queue_pid,
        upstream_queue=str(base/'queue.json'), local_work_error_gate=False,
        independent_validation_scheduled=False, physical_predictor_qualified=False))
    state = dict(status='waiting_for_aggregate_training', physical_predictor_qualified=False,
                 formal_utility_qualified=False)
    def update(status, **values):
        state.update(status=status, **values)
        save(output/'state.json', state)
    def check():
        for r in receipts:
            if any(hashlib.sha256(Path(r[k]).read_bytes()).hexdigest() != r['sha256'] for k in ('path', 'snapshot')):
                raise ValueError('Frozen analysis source changed: '+r['path'])
    def run(script, argv, log):
        check()
        with (output/log).open('x') as stream:
            subprocess.run([sys.executable, str(ROOT/'experiments'/script), *map(str, argv)],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        check()
    update('waiting_for_aggregate_training')
    try:
        check()
        while True:
            upstream = json.loads(Path(queue['state']).read_text())
            if upstream['status'] == 'capture_plan_completed':
                break
            if upstream.get('errors') or 'failed' in upstream['status']:
                update('upstream_capture_failed', upstream=upstream)
                return 1
            proc = Path('/proc')/str(args.queue_pid)/'cmdline'
            try:
                command = proc.read_bytes()
            except FileNotFoundError:
                command = b''
            queue_names = [str(base/'queue.json').encode(), str((base/'queue.json').relative_to(ROOT)).encode()]
            if b'run_activation_queue.py' not in command or not any(name in command for name in queue_names):
                upstream = json.loads(Path(queue['state']).read_text())
                if upstream['status'] == 'capture_plan_completed':
                    break
                raise RuntimeError('Queue handle ended without completion')
            time.sleep(2)
        update('fitting_aggregate_native_model')
        capture_args = [value for capture in captures for value in ('--capture', capture)]
        run('fit_activation_aggregate_native.py', [*capture_args, '--output', output/'fit.json'], 'fit.log')
        update('replaying_complete_planned_episodes')
        run('replay_activation_aggregate.py', ['--fit', output/'fit.json', *capture_args,
            '--policy', args.policy.resolve(), '--output', output/'training-replay.json'], 'replay.log')
        replay = json.loads((output/'training-replay.json').read_text())
        update('aggregate_training_analysis_completed',
               complete_runs=sum(r['assessment']['complete_cohort'] for r in replay['reports']),
               requested_runs=len(replay['reports']),
               independent_validation_pending=True, candidate_ranking_pending=True)
        return 0
    except BaseException as exc:
        update('aggregate_training_analysis_failed', error=type(exc).__name__+': '+str(exc))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
