"""Run exactly W1/W3 with frozen 7C-3 serving/acceptance; never launch W2."""
import argparse
import fcntl
import json
from candidate_pool_service_demand.dry_run import load_completed_catalog
from deployment_core import Workload
from deployment_core.replica_calibration import load_replica_calibration
from deployment_search import IntegerGuidedDeploymentSearch
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from deployment_search.tests.fixtures import forbid_enumeration
from deployment_search.runtime_cache import save_json, file_hash
from joint_deploy.measurement.schema import MeasurementConfig
from integer_replica_pilot.campaign import once
from integer_replica_pilot.evaluator import ReplicaRuntimeEvaluator, PilotExecutionBlocked, aggregate
from .prepare import OUT, JOBS, CONFIG, prepare, verify_freeze, read


def confirmation(directory, trace, protocol, evaluator, catalog, workload):
    decision = {'search_trace_sha256': file_hash(directory / 'search_trace.json'),
        'seed': trace['seed']['deployment'], 'final': trace['deployment'], 'decision_frozen_before_confirmation': True}
    path = directory / 'post_search_freeze.json'
    if path.exists() and read(path) != decision:
        raise RuntimeError('Decision changed before confirmation')
    save_json(path, decision)
    states = {'seed': (trace['seed']['deployment'], trace['iterations'][0]['canonical']['routing']),
              'final': (trace['deployment'], trace['canonical']['routing'])}
    rows = {'seed': [], 'final': []}
    # Same paired seeds and frozen order as 7C-3. Even identical deployments
    # receive two independently executed repetitions per label; no cache reuse.
    for which, index in protocol['confirmation_order']:
        counts, routing = states[which]
        routing = {tuple(key.split('::')): value for key, value in routing.items()}
        identity = evaluator.identity(counts, workload, routing)
        config = MeasurementConfig(identity['deployment'], identity['workload'], identity['routing'],
            random_seed=protocol['confirmation_seeds'][index - 1], **protocol['measurement_settings']).as_dict()
        rep = once(directory / 'confirmation', which + '_rep' + str(index), config,
            directory / 'runtime_context.json', directory / 'confirmation_attempts.json', 4, catalog, workload)
        rep.update(repetition=index, random_seed=protocol['confirmation_seeds'][index - 1], post_search_confirmation=True)
        rows[which].append(rep)
        save_json(directory / 'confirmation_results.json', {'complete': False, 'repetitions': rows, 'influences_search': False})
        if not rep['valid_runtime_repetition']:
            raise PilotExecutionBlocked('Confirmation incomplete; no retry or budget growth')
    aggregates = {which: aggregate(sorted(reps, key=lambda r: r['repetition']), 'independent_confirmation',
        directory / 'confirmation' / (which + '_aggregate.json'), states[which][0]) for which, reps in rows.items()}
    for which, value in aggregates.items():
        save_json(directory / 'confirmation' / (which + '_aggregate.json'), value)
    seed, final = aggregates['seed'], aggregates['final']
    gain = (seed.weighted_mean_e2e - final.weighted_mean_e2e) / seed.weighted_mean_e2e if seed.weighted_mean_e2e and final.weighted_mean_e2e is not None else None
    better, reason = NoiseAwareRuntimeObjective(1., .05).compare(final, seed, 0., 0.)
    save_json(directory / 'confirmation_results.json', {'complete': True, 'influences_search': False,
        'repetitions': rows, 'aggregates': aggregates, 'relative_mean_gain': gain,
        'mean_direction_maintained': gain is not None and gain >= 0,
        'objective_direction_maintained': better, 'objective_reason': reason,
        'search_final_equals_seed': states['seed'][0] == states['final'][0],
        'physical_attempts': len(read(directory / 'confirmation_attempts.json'))})


def run_workload(short, name):
    if (short, name) not in JOBS:
        raise ValueError('Only W1 and W3 are authorized new GPU jobs; W2 is immutable reference')
    verify_freeze()
    directory = OUT / short
    protocol = read(directory / 'protocol.json')
    if protocol['workload']['name'] != name:
        raise RuntimeError('Job/workload identity mismatch')
    catalog, hardware, roles, _, _ = load_completed_catalog()
    cal = load_replica_calibration(catalog)
    workload = Workload(protocol['workload']['arrival_rates_rps'])
    evaluator = ReplicaRuntimeEvaluator(directory / 'search_cache', catalog, directory / 'runtime_context.json',
        protocol['measurement_settings'], seeds=protocol['search_seeds'])
    final_path = directory / 'search_complete.json'
    if final_path.exists():
        trace = read(final_path)
    else:
        print(json.dumps({'event': 'workload_search_start', 'workload': name, 'new_GPU_search': True}), flush=True)
        with forbid_enumeration() as guards:
            result = IntegerGuidedDeploymentSearch(evaluator, predictor=None,
                runtime_objective=NoiseAwareRuntimeObjective(1., .05),
                trace_callback=lambda r: save_json(directory / 'search_trace.json', r.to_dict())).search(
                    workload, catalog, hardware, CONFIG, roles=roles,
                    capacity_mode='empirical', memory_mode='empirical', calibration=cal)
        trace = result.to_dict()
        trace['enumeration_calls'] = sum(g.call_count for g in guards)
        save_json(directory / 'search_trace.json', trace)
        save_json(final_path, trace)
    if trace['runtime_evaluation_count'] > 6 or trace['search_iterations'] > 3 or trace['enumeration_calls']:
        raise AssertionError('Frozen budget/enumeration guard violation')
    verify_freeze()
    confirmation(directory, trace, protocol, evaluator, catalog, workload)
    save_json(directory / 'status.json', {'status': 'complete', 'workload': name, 'W2_GPU_reuse': False})
    print(json.dumps({'event': 'workload_complete', 'workload': name, 'deployment': trace['deployment'],
        'evaluations': trace['runtime_evaluation_count']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare()
        if args.prepare_only:
            return
        try:
            for short, name in JOBS:
                run_workload(short, name)
            from .report import report
            report()
            save_json(OUT / 'campaign_status.json', {'status': 'complete', 'stopped_after_module': '7C-4', 'W2_new_GPU_runs': 0})
            from .report import artifact_manifest
            artifact_manifest()
            print(json.dumps({'event': 'module7c4_complete', 'W2_new_GPU_runs': 0}), flush=True)
        except (PilotExecutionBlocked, KeyboardInterrupt) as exc:
            save_json(OUT / 'campaign_status.json', {'status': 'blocked', 'reason': str(exc), 'no_retry_or_budget_growth': True})
            from .report import report
            report(incomplete=True)
            raise


if __name__ == '__main__':
    main()
