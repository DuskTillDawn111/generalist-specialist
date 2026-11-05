"""Wait once for scoped acceptance, then run the bounded CPU search once.

No GPU collection, retraining, retries, or scope expansion. Failed/restricted
validation is handed back for decision-impact review, with failures intact.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.scoped_routing_objective import file_receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('validation-base', 'handoff-base', 'evidence', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--wait-seconds', type=float, default=6000.)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (a.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    command = [sys.executable, str(ROOT/'experiments/benchmark_scoped_routing.py'),
        '--validation-base', str(a.validation_base.resolve()), '--evidence', str(a.evidence.resolve()),
        '--reference', str(a.reference.resolve()), '--release', str((a.handoff_base/'release-decision.json').resolve()),
        '--output', str((a.output/'comparison').resolve())]
    paths = {Path(__file__).resolve(), ROOT/'experiments/benchmark_scoped_routing.py',
             ROOT/'experiments/scoped_routing_objective.py', a.evidence.resolve(), a.reference.resolve(),
             *sorted((ROOT/'gsdeploy').rglob('*.py'))}
    protocol = json.loads((a.validation_base/'protocol.json').read_text())
    paths.update(Path(r['path']) for r in protocol['source_receipts'])
    paths.update(a.validation_base/f'aggregate-validation-round{i}/plan.json' for i in (1, 2))
    paths.update(a.validation_base/name for name in ('protocol.json', 'fit-frozen.json', 'policy-frozen.json'))
    receipts = []
    for i, path in enumerate(sorted(paths)):
        snapshot = a.output/'source-snapshot'/str(i)/path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        receipts.append(dict(**file_receipt(path), snapshot=str(snapshot.resolve())))
    save('plan.json', dict(command=command, receipts=receipts,
        wait_seconds=a.wait_seconds, extra_gpu_runs=0, automatic_refits=0,
        on_restricted_or_failed_validation='decision-impact review, no automatic route search or recollection'))
    save('state.json', dict(status='waiting_for_scoped_release'))
    deadline = time.monotonic()+a.wait_seconds
    try:
        while True:
            state = json.loads((a.handoff_base/'handoff-state.json').read_text())
            if state['status'] == 'SCOPED_PASS':
                break
            if state['status'] != 'waiting_for_existing_validation':
                save('state.json', dict(status='decision_impact_review_required', upstream=state,
                                       automatic_extra_experiments=0))
                return
            if time.monotonic() >= deadline:
                save('state.json', dict(status='release_still_pending', automatic_extra_experiments=0))
                return
            time.sleep(min(30., max(0., deadline-time.monotonic())))
        for receipt in receipts:
            if any(file_receipt(receipt[k])['sha256'] != receipt['sha256'] for k in ('path', 'snapshot')):
                raise ValueError('Frozen optimization source/input changed: '+receipt['path'])
        save('state.json', dict(status='running_bounded_cpu_optimization', command=command))
        with (a.output/'comparison.log').open('x') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        save('state.json', dict(status='bounded_optimization_completed',
            next_action='Select at most three distinct candidates and freeze paired measured comparison',
            measured_ranking_qualified=False))
    except BaseException as exc:
        save('state.json', dict(status='failed_no_retry', error=type(exc).__name__+': '+str(exc)))
        raise


if __name__ == '__main__':
    main()
