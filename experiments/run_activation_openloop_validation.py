"""Run frozen open-loop qualification only after its earlier heldout check passes."""
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
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prior-validation', type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    protocol = json.loads((output/'protocol.json').read_text())
    state_path = output/'continuation.json'
    if state_path.exists():
        raise FileExistsError(state_path)
    state = dict(status='waiting_for_prior_independent_gate', formal_utility_qualified=False)
    save(state_path, state)
    deadline = time.monotonic()+21600
    while not args.prior_validation.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Prior independent score was not produced')
        time.sleep(2)
    prior = json.loads(args.prior_validation.read_text())
    if prior.get('fit_sha256') != protocol['fit_sha256']:
        raise ValueError('Earlier validation tested a different frozen predictor')
    state.update(prior_validation=str(args.prior_validation.resolve()),
        prior_validation_sha256=hashlib.sha256(args.prior_validation.read_bytes()).hexdigest(),
        orchestration_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if not prior['development_accuracy_passed']:
        state.update(status='not_run_prior_independent_gate_failed', prior_reasons=prior['reasons'])
        save(state_path, state)
        return 0

    def verify():
        for source in protocol['prediction_sources']:
            if any(hashlib.sha256(Path(source[k]).read_bytes()).hexdigest() != source['sha256']
                   for k in ['path', 'snapshot']):
                raise ValueError('Frozen open-loop prediction source drift: '+source['path'])
        for directory, expected in protocol['run_plan_sha256'].items():
            if hashlib.sha256((Path(directory)/'plan.json').read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen open-loop cohort plan changed')

    for directory in map(Path, protocol['runs']):
        verify()
        state.update(status='collecting', current_capture=str(directory))
        save(state_path, state)
        with (directory/'collector.log').open('x') as out:
            r = subprocess.run([sys.executable, str(ROOT/'experiments/run_activation_capture.py'),
                                '--output', str(directory)], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        if r.returncode:
            state['status'] = 'capture_failed'
            save(state_path, state)
            return 1
    verify()
    state['status'] = 'scoring'
    save(state_path, state)
    with (output/'score.log').open('x') as out:
        r = subprocess.run([sys.executable, str(ROOT/'experiments/validate_activation_native.py'),
                           'score', '--output', str(output)], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    if r.returncode:
        state['status'] = 'score_failed'
        save(state_path, state)
        return 1
    result = json.loads((output/'validation.json').read_text())
    state.update(status='openloop_development_accuracy_passed' if result['development_accuracy_passed']
                 else 'openloop_independent_validation_failed', reasons=result['reasons'],
                 latency_errors=result['latency_errors'])
    save(state_path, state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
