"""Advance declared training domains to heldout GPU checks only after full replay passes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import POOL, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    state_path = Path(plan['state'])
    if state_path.exists():
        raise FileExistsError(state_path)
    state = dict(status='waiting', domains=[], errors=[], formal_utility_qualified=False)
    save(state_path, state)
    deadline = time.monotonic()+21600
    def wait_for(condition):
        while True:
            path = Path(condition['path'])
            upstream = json.loads(path.read_text()) if path.exists() else {}
            if upstream.get('status') == condition['status']:
                break
            if upstream.get('errors') or 'failed' in upstream.get('status', ''):
                state.update(status='upstream_failed', upstream=upstream)
                save(state_path, state)
                raise RuntimeError('Declared qualification prerequisite failed')
            if time.monotonic() > deadline:
                raise TimeoutError('Declared qualification prerequisite not completed')
            time.sleep(2)
    for condition in plan.get('after', []):
        wait_for(condition)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def run(script, argv, log):
        with Path(log).open('x') as out:
            result = subprocess.run([sys.executable, str(ROOT/'experiments'/script), *map(str, argv)],
                                    cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
        return result.returncode == 0

    for domain in plan['domains']:
        item = dict(label=domain['label'], status='checking_training_replay')
        state['domains'].append(item)
        state['status'] = 'processing_domains'
        save(state_path, state)
        for condition in domain.get('after', []):
            item['status'] = 'waiting_for_domain_receipts'
            save(state_path, state)
            wait_for(condition)
        fit, replay = Path(domain['fit']), Path(domain['replay'])
        if domain.get('captures'):
            prefix = fit.parent/domain['label']
            native = Path(str(prefix)+'-native.json')
            energy = Path(str(prefix)+'-energy.json')
            argv = []
            for capture in domain['captures']:
                argv += ['--capture', capture]
            argv += ['--pool', POOL, '--order-aware', '--output', native]
            stages = [('fit_activation_native.py', argv, native),
                ('fit_activation_energy.py', ['--fit', native, '--output', energy], energy),
                ('fit_activation_temporal.py', ['--fit', energy, '--ridge', '.1', '--output', fit], fit),
                ('replay_activation_training.py', ['--fit', fit, '--output', replay], replay)]
            for script, argv, target in stages:
                item.update(status=script, current_output=str(target))
                save(state_path, state)
                if not run(script, argv, target.with_suffix('.log')):
                    item.update(status='command_failed', command=script)
                    state['errors'].append(str(target))
                    break
            if item['status'] == 'command_failed':
                save(state_path, state)
                continue
        report = json.loads(replay.read_text())
        item.update(fit=str(fit), replay=str(replay), reasons=report['reasons'],
                    latency_errors=report['latency_errors'])
        if not report['training_replay_accuracy_within_threshold']:
            item['status'] = 'training_replay_failed_no_heldout_capture'
            save(state_path, state)
            continue
        validation = Path(domain['validation'])
        item.update(status='preparing_independent_validation', validation=str(validation))
        save(state_path, state)
        stages = [('validate_activation_native.py', ['prepare', '--fit', fit,
            '--reference', plan['reference'], '--output', validation, '--seeds', *domain['seeds']],
            validation.with_suffix('.prepare.log'))]
        if not run(*stages[0]):
            item['status'] = 'validation_prepare_failed'
            state['errors'].append(str(validation))
            save(state_path, state)
            continue
        protocol = json.loads((validation/'protocol.json').read_text())
        for capture in map(Path, protocol['runs']):
            item.update(status='collecting_independent_validation', current_capture=str(capture))
            save(state_path, state)
            if not run('run_activation_capture.py', ['--output', capture], capture/'collector.log'):
                item['status'] = 'validation_capture_failed'
                state['errors'].append(str(capture))
                break
        if item['status'] == 'validation_capture_failed':
            save(state_path, state)
            continue
        if not run('validate_activation_native.py', ['score', '--output', validation], validation/'score.log'):
            item['status'] = 'validation_score_failed'
            state['errors'].append(str(validation))
        else:
            result = json.loads((validation/'validation.json').read_text())
            item.update(status='development_accuracy_passed' if result['development_accuracy_passed']
                        else 'independent_validation_failed', validation_reasons=result['reasons'],
                        validation_latency_errors=result['latency_errors'])
        save(state_path, state)
    state['status'] = 'development_stages_evaluated'
    save(state_path, state)
    return int(bool(state['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
