"""Execute the existing bounded confirmation with a mandatory pure-S control.

This records the decision-use limitation instead of changing the failed gate.
The original candidate preparation, capture queue and scoring are reused.
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
from experiments.run_scoped_decision_search import check_release
from experiments.scoped_routing_objective import file_receipt


def require_pure_specialist(panel):
    controls = [r['candidate_id'] for r in panel if set(r['routing']) == {'asr', 'image', 'ocr', 'text'} and all(
        set(m for m, q in route.items() if q > 0) == {task}
        for task, route in r['routing'].items())]
    if not controls:
        raise ValueError('Known OCR limitation requires a pure-specialist measured control; do not launch or expand the panel')
    return controls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('comparison', 'validation', 'release', 'evidence', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    for name in ('comparison', 'validation', 'release', 'evidence', 'reference', 'output'):
        setattr(a, name, getattr(a, name).resolve())
    release = json.loads(a.release.read_text())
    check_release(release)
    a.output.mkdir(parents=True, exist_ok=False)
    save(a.output/'plan.json', dict(decision_use_release=file_receipt(a.release),
        source_optimization=str(a.comparison), controller=file_receipt(Path(__file__)),
        maximum_candidates=3, rounds=2, maximum_gpu_episodes=6,
        strict_latency_gate_pass=False, automatic_retries=0))
    save(a.output/'state.json', dict(status='sealing_decision_use_confirmation'))
    try:
        base = a.output/'captures'
        summary = prepare(a.comparison, a.validation, a.evidence, a.reference, base)
        protocol = json.loads((base/'protocol.json').read_text())
        controls = require_pure_specialist(protocol['candidates'])
        if summary['gpu_episodes'] > release['max_gpu_episodes']:
            raise ValueError('Frozen GPU episode budget exceeded')
        for i, path in enumerate((Path(__file__).resolve(), ROOT/'experiments/run_scoped_decision_search.py')):
            snapshot = base/'decision-source-snapshot'/str(i)/path.name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(path.read_bytes())
            protocol['source_receipts'].append(dict(**file_receipt(path), snapshot=str(snapshot)))
        protocol['input_receipts'].extend([file_receipt(a.release), file_receipt(a.output/'plan.json')])
        protocol.update(decision_use_release=file_receipt(a.release), strict_latency_gate_pass=False,
            known_limitations=release['known_limitations'], pure_specialist_controls=controls,
            sealed_unix_ns=time.time_ns())
        save(base/'protocol.json', protocol)
        save(a.output/'state.json', dict(status='collecting_bounded_paired_confirmation',
            pure_specialist_controls=controls, strict_latency_gate_pass=False, **summary))
        with (a.output/'collection.log').open('x') as stream:
            subprocess.run([sys.executable, str(ROOT/'experiments/run_activation_queue.py'), '--queue', str(base/'queue.json')],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        save(a.output/'state.json', dict(status='scoring_sealed_paired_confirmation'))
        with (a.output/'assessment.log').open('x') as stream:
            subprocess.run([sys.executable, str(ROOT/'experiments/score_scoped_confirmation.py'),
                '--base', str(base), '--evidence', str(a.evidence), '--reference', str(a.reference),
                '--policy', str(a.validation/'policy-frozen.json')],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = json.loads((base/'comparison.json').read_text())
        save(a.output/'state.json', dict(status='bounded_confirmation_completed',
            selection_accepted=result['assessment']['descriptive_panel_selection_accepted'],
            strict_latency_gate_pass=False, known_limitations=release['known_limitations'],
            prediction_tolerance_failures=result['prediction_tolerance_failures'],
            next_action=result['assessment']['next_action'], extra_experiments_scheduled=0))
    except BaseException as exc:
        save(a.output/'state.json', dict(status='failed_no_retry', error=type(exc).__name__+': '+str(exc),
                                         extra_experiments_scheduled=0))
        raise


if __name__ == '__main__':
    main()
