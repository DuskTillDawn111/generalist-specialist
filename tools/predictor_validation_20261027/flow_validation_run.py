"""Run one complete independent flow suite after all predictions are sealed."""
import argparse
import json
from pathlib import Path
import random
import time

from .flow_trained_validation import check_sealed
from .native_flow import checked, execute, receipt, save
from .probe_validation import read


def prepare(plan_path, sealed_dir, qualification_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    check_sealed(sealed_dir)
    plan = read(plan_path); sealed = read(Path(sealed_dir) / 'predictions.json')
    if sealed['sources']['plan'] != receipt(plan_path) or plan.get('phase') != 'predictor_validation':
        raise ValueError('Independent plan differs from sealed predictions')
    declared = {c['id'] for c in plan['candidates']}
    predicted = [c['candidate_id'] for c in sealed['candidates']]
    if len(predicted) != len(set(predicted)) or set(predicted) != declared:
        raise ValueError('Every declared candidate must have one sealed prediction')
    if any(p['unknown_reason'] or not p['metrics'] or
           not p['metrics']['six_term_comparison_complete'] or
           any(r['e2e_ms'] is None for r in p['requests']) for p in sealed['candidates']):
        raise ValueError('Complete declared-domain predictions required before native independent suite')
    jobs = [{'candidate_id': c['id'], 'output': str(output / 'runs' / c['id'])} for c in plan['candidates']]
    random.Random('independent-flow-validation-order:' + str(plan['seed'])).shuffle(jobs)
    refs = {'plan': receipt(plan_path), 'predictions': receipt(Path(sealed_dir) / 'predictions.json'),
            'training_audit': receipt(Path(sealed_dir) / 'training-audit.json'),
            'qualification': receipt(qualification_path), 'source': receipt(__file__)}
    value = {'schema': 'gsdeploy.independent_native_flow_suite.v1', 'created_ns': time.perf_counter_ns(),
             'sources': refs, 'sealed_dir': str(Path(sealed_dir).resolve()), 'jobs': jobs,
             'no_retries': True, 'no_seed_replacement': True, 'GPU_started': False}
    output.mkdir(parents=True, exist_ok=False); save(output / 'suite.json', value)
    return value


def run(path):
    suite = read(path)
    if suite.get('schema') != 'gsdeploy.independent_native_flow_suite.v1':
        raise ValueError('Explicit independent validation suite required')
    for ref in suite['sources'].values():
        checked(ref)
    check_sealed(suite['sealed_dir'])
    for job in suite['jobs']:
        print(json.dumps({'event': 'starting_independent_validation_run',
                          'candidate': job['candidate_id']}), flush=True)
        raw = execute(suite['sources']['plan']['path'], job['candidate_id'], job['output'])
        if raw['status'] != 'completed' or raw['qualification_issues']:
            raise RuntimeError('Independent collection failed; preserve failure and stop without retries')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    a = sub.add_parser('prepare')
    for name in ('plan', 'sealed-dir', 'qualification', 'output'):
        a.add_argument('--' + name, required=True)
    a = sub.add_parser('run'); a.add_argument('--suite', required=True)
    args = parser.parse_args()
    if args.operation == 'prepare':
        result = prepare(args.plan, args.sealed_dir, args.qualification, args.output)
        print(json.dumps({'status': 'PREPARED', 'runs': len(result['jobs']), 'GPU_started': False}))
    else:
        run(args.suite)
