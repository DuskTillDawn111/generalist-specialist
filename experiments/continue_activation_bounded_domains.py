"""Evaluate declared alternative age-basis domains and preserve every failed stage."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    base = args.plan.resolve().parent
    state_path = base/'state.json'
    if state_path.exists():
        raise FileExistsError(state_path)
    state = dict(status='running', domains=[], errors=[], formal_utility_qualified=False)
    save(state_path, state)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for domain in plan['domains']:
        directory = base/domain['label']
        directory.mkdir(exist_ok=False)
        item = dict(label=domain['label'], status='waiting_for_training_parent')
        state['domains'].append(item)
        save(state_path, state)
        parent = Path(domain['parent'])
        deadline = time.monotonic()+21600
        while not parent.exists():
            if time.monotonic() > deadline:
                item['status'] = 'parent_wait_timeout'
                state['errors'].append(domain['label'])
                save(state_path, state)
                return 1
            time.sleep(2)
        fit, replay = directory/'fit.json', directory/'training-replay.json'
        commands = [('fitting_new_basis', 'fit_activation_temporal.py',
            ['--fit', parent, '--output', fit, '--ridge', str(plan['ridge']),
             '--age-descriptor', plan['age_descriptor']]),
            ('replaying_training', 'replay_activation_training.py', ['--fit', fit, '--output', replay])]
        failed = False
        for status, script, argv in commands:
            item['status'] = status
            save(state_path, state)
            with (directory/(status+'.log')).open('x') as out:
                result = subprocess.run([sys.executable, str(ROOT/'experiments'/script), *map(str, argv)],
                    cwd=ROOT, env=environment, stdout=out, stderr=subprocess.STDOUT)
            if result.returncode:
                item['status'] = status+'_command_failed'
                state['errors'].append(domain['label'])
                failed = True
                break
        if failed:
            save(state_path, state)
            continue
        report = json.loads(replay.read_text())
        item.update(reasons=report['reasons'], latency_errors=report['latency_errors'],
                    training_replay_passed=report['training_replay_accuracy_within_threshold'])
        qualification = dict(schema='gsdeploy.native_domain_qualification_plan.v1',
            reference=plan['reference'], state=str(directory/'qualification-state.json'), after=[],
            domains=[dict(label=domain['label'], fit=str(fit), replay=str(replay),
                          validation=str(directory/'independent'), seeds=domain['seeds'])])
        save(directory/'qualification-plan.json', qualification)
        # Launching this process only queues clean GPU work; it immediately
        # records a failed replay without allocating GPU if the gate is false.
        with (directory/'qualification-controller.log').open('x') as out:
            child = subprocess.Popen([sys.executable, str(ROOT/'experiments/continue_activation_qualification.py'),
                '--plan', str(directory/'qualification-plan.json')], cwd=ROOT, env=environment,
                stdout=out, stderr=subprocess.STDOUT)
        item.update(status='qualification_controller_started', controller_pid=child.pid,
                    qualification_state=str(directory/'qualification-state.json'))
        save(state_path, state)
    state['status'] = 'domain_models_evaluated_qualification_controllers_independent'
    save(state_path, state)
    return int(bool(state['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
