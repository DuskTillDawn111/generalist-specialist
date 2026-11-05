"""One bounded measured comparison after the predeclared optimization.

No automatic retries, extra candidates, model fitting or enlarged scope.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import save
from experiments.prepare_scoped_confirmation import prepare
from experiments.scoped_routing_objective import file_receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('optimization', 'validation', 'evidence', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--wait-seconds', type=float, default=7200.)
    a = p.parse_args()
    for name in ('optimization', 'validation', 'evidence', 'reference', 'output'):
        setattr(a, name, getattr(a, name).resolve())
    a.output.mkdir(parents=True, exist_ok=False)
    paths = {Path(__file__).resolve(), ROOT/'experiments/prepare_scoped_confirmation.py',
             ROOT/'experiments/score_scoped_confirmation.py', ROOT/'experiments/scoped_routing_objective.py',
             a.evidence, a.reference, a.optimization/'plan.json',
             *[Path(r['path']) for r in json.loads((a.validation/'protocol.json').read_text())['source_receipts']]}
    receipts = []
    for i, path in enumerate(sorted(paths)):
        snapshot = a.output/'source-snapshot'/str(i)/path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        receipts.append(dict(**file_receipt(path), snapshot=str(snapshot)))
    save(a.output/'plan.json', dict(receipts=receipts, maximum_candidates=3, rounds=2,
        maximum_gpu_episodes=6, wait_seconds=a.wait_seconds, automatic_retries=0,
        source_optimization=str(a.optimization), confirmation_seeds=[6110491, 6110492]))
    save(a.output/'state.json', dict(status='waiting_for_bounded_optimization'))
    try:
        deadline = time.monotonic()+a.wait_seconds
        while True:
            upstream = json.loads((a.optimization/'state.json').read_text())
            if upstream['status'] == 'bounded_optimization_completed':
                break
            if upstream['status'] not in ('waiting_for_scoped_release', 'running_bounded_cpu_optimization'):
                save(a.output/'state.json', dict(status='upstream_review_required', upstream=upstream,
                                                 automatic_extra_experiments=0))
                return
            if time.monotonic() >= deadline:
                save(a.output/'state.json', dict(status='optimization_still_pending', automatic_extra_experiments=0))
                return
            time.sleep(min(30., max(0., deadline-time.monotonic())))
        for receipt in receipts:
            if any(file_receipt(receipt[k])['sha256'] != receipt['sha256'] for k in ('path', 'snapshot')):
                raise ValueError('Frozen confirmation source/input changed: '+receipt['path'])
        base = a.output/'captures'
        save(a.output/'state.json', dict(status='sealing_complete_candidate_panel'))
        summary = prepare(a.optimization/'comparison', a.validation, a.evidence, a.reference, base)
        save(a.output/'state.json', dict(status='collecting_bounded_paired_confirmation', **summary))
        with (a.output/'collection.log').open('x') as stream:
            subprocess.run([sys.executable, str(ROOT/'experiments/run_activation_queue.py'),
                            '--queue', str(base/'queue.json')], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        save(a.output/'state.json', dict(status='scoring_sealed_paired_confirmation'))
        with (a.output/'assessment.log').open('x') as stream:
            subprocess.run([sys.executable, str(ROOT/'experiments/score_scoped_confirmation.py'),
                            '--base', str(base), '--evidence', str(a.evidence), '--reference', str(a.reference),
                            '--policy', str(a.validation/'policy-frozen.json')],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = json.loads((base/'comparison.json').read_text())
        save(a.output/'state.json', dict(status='bounded_confirmation_completed',
            selection_accepted=result['assessment']['descriptive_panel_selection_accepted'],
            prediction_tolerance_failures=result['prediction_tolerance_failures'],
            next_action=result['assessment']['next_action'], extra_experiments_scheduled=0))
    except BaseException as exc:
        save(a.output/'state.json', dict(status='failed_no_retry', error=type(exc).__name__+': '+str(exc),
                                         extra_experiments_scheduled=0))
        raise


if __name__ == '__main__':
    main()
