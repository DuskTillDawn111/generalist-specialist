"""Run the same bounded optimizers under an explicit decision-use release.

The original strict latency failure remains false. This entry point allows
proposal search with that documented limitation and mandatory measured checks.
No fit, validation, objective, or search implementation is changed.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_scoped_routing import run
from experiments.scoped_routing_objective import file_receipt, scoped_reference
from experiments.collect_activation_training import save


def check_release(release):
    if (release['status'] != 'FROZEN_FOR_BOUNDED_SEARCH_WITH_KNOWN_LIMITATION'
            or release['decision_search_authorized'] is not True
            or release['strict_latency_gate_pass'] is not False
            or release['no_refit'] is not True or release['no_additional_predictor_collection'] is not True
            or release['mandatory_confirmation_controls'] != ['pure_specialists']
            or release['max_candidates'] != 3 or release['max_gpu_episodes'] != 6
            or release['search_methods'] != ['Joint-K2', 'Activation-BO-MADS']
            or release['search_seeds'] != [11, 29, 47]
            or release['search_budget_per_method_seed'] != dict(seconds=300, predictions=64, routes=32, master_calls=128)):
        raise ValueError('Explicit frozen decision-use restrictions required')
    for receipt in release['source_receipts']:
        if file_receipt(receipt['path']) != receipt:
            raise ValueError('Decision-use evidence changed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('validation', 'release', 'evidence', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    release = json.loads(a.release.read_text())
    check_release(release)
    validation = json.loads((a.validation/'validation.json').read_text())
    if validation['known_input_history_accuracy_within_tolerances'] or not validation['execution_integrity_passed']:
        raise ValueError('This release applies to the documented accuracy limitation with intact captures')
    inputs = [a.validation/'fit-frozen.json', a.validation/'validation.json', a.validation/'policy-frozen.json']
    if any(file_receipt(path) not in release['source_receipts'] for path in inputs):
        raise ValueError('Release does not identify these validation/model inputs')
    system = json.loads(a.reference.read_text())
    fit = json.loads((a.validation/'fit-frozen.json').read_text())
    evidence = json.loads(a.evidence.read_text())
    if file_receipt(a.reference) not in evidence['sources'] or not set(evidence['solo_sources']) <= set(fit['source_receipts']):
        raise ValueError('Reference or solo evidence mismatch')
    for receipt in evidence['sources']:
        if file_receipt(receipt['path']) != receipt:
            raise ValueError('Training evidence changed')
    templates = []
    for i in (1, 2):
        plan = json.loads((a.validation/f'aggregate-validation-round{i}/plan.json').read_text())
        plan['jobs'] = [plan['jobs'][0]]
        templates.append(plan)
    protocol = json.loads((a.validation/'protocol.json').read_text())
    for receipt in protocol['source_receipts']:
        if file_receipt(receipt['path'])['sha256'] != receipt['sha256']:
            raise ValueError('Frozen predictor source changed')
    a.output.mkdir(parents=True, exist_ok=False)
    comparison = a.output/'comparison'
    comparison.mkdir()
    paths = {Path(__file__).resolve(), ROOT/'experiments/benchmark_scoped_routing.py',
             ROOT/'experiments/scoped_routing_objective.py', a.release.resolve(), a.evidence.resolve(),
             a.reference.resolve(), *[path.resolve() for path in inputs],
             *[Path(r['path']) for r in protocol['source_receipts']],
             *sorted((ROOT/'gsdeploy').rglob('*.py')),
             *[(a.validation/f'aggregate-validation-round{i}/plan.json').resolve() for i in (1, 2)]}
    receipts = [file_receipt(path) for path in sorted(paths)]
    snapshots = []
    for i, receipt in enumerate(receipts):
        path = Path(receipt['path'])
        snapshot = a.output/'source-snapshot'/str(i)/path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        snapshots.append(dict(**receipt, snapshot=str(snapshot.resolve())))
    save(a.output/'source-snapshots.json', snapshots)
    specification = dict(methods=release['search_methods'], seeds=release['search_seeds'],
        budget=release['search_budget_per_method_seed'], source_receipts=receipts,
        strict_latency_gate_pass=False, decision_use_release=file_receipt(a.release),
        known_limitations=release['known_limitations'], mandatory_measured_control='pure_specialists',
        reference_reuse=scoped_reference(system, templates[0])[0]['utility_normalization'],
        physical_ranking_qualified=False, extra_predictor_experiments=0)
    save(a.output/'plan.json', specification)
    save(comparison/'plan.json', specification)
    save(a.output/'state.json', dict(status='running_bounded_cpu_optimization', strict_latency_gate_pass=False))
    save(comparison/'state.json', dict(status='running_bounded_optimization'))
    try:
        summaries = []
        for seed in release['search_seeds']:
            methods = list(release['search_methods'])
            if seed % 3 == 2:
                methods.reverse()
            for method in methods:
                result = run(system, fit, evidence, templates, method, seed,
                             seconds=300., predictions=64, routes=32)
                save(comparison/f'{method}-{seed}.json', result)
                w = result.get('winner')
                row = dict(method=method, seed=seed, status=result['status'], budget=result['budget'],
                    utility=w.get('score', w.get('best_feasible_utility')) if w else None,
                    routing=w.get('routing', w.get('routing_qps')) if w else None,
                    deployment=w.get('deployment') if w else None)
                summaries.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
        check_release(release)
        if any(file_receipt(r['path']) != r for r in receipts):
            raise ValueError('Frozen search inputs/source changed during execution')
        save(comparison/'comparison.json', dict(rows=summaries, strict_latency_gate_pass=False,
            known_limitations=release['known_limitations'], measured_ranking_qualified=False,
            disposition='Candidate proposals only; original accuracy failure retained; pure-specialist measured control required'))
        save(comparison/'state.json', dict(status='bounded_search_completed', measured_ranking_qualified=False))
        save(a.output/'state.json', dict(status='bounded_optimization_completed',
            next_action='Lock at most three candidates including pure specialists for the already bounded paired comparison'))
    except BaseException as exc:
        save(a.output/'state.json', dict(status='failed_no_retry', error=type(exc).__name__+': '+str(exc)))
        raise


if __name__ == '__main__':
    main()
