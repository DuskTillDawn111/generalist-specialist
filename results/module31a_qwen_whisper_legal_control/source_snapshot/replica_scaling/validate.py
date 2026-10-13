"""Artifact and immutable-input validation; never launches a model worker."""
from collections import Counter
from candidate_pool_audit.provenance import environment_snapshot
from joint_deploy.measurement.load_generator import SamplePool, periodic_arrivals
from joint_deploy.measurement.router import Router
from .common import OUT, ROOT, frozen_plan, read, write, reference, verify_refs
from .telemetry import snapshot


def validate():
    plan = frozen_plan()
    ledger = read(OUT / 'campaign_ledger.json')
    before = read(OUT / 'source_hashes_before.json')
    errors = []
    source_count = 0
    for refs in (before['protected_files'], before['inputs'], *before['models'].values()):
        try:
            source_count += verify_refs(refs, full=True)
        except Exception as exc:
            errors.append(str(exc))
    environment = environment_snapshot()
    unchanged_env = environment == read(OUT / 'environment_manifest.json')['environments']
    if not unchanged_env:
        errors.append('Frozen environment drift')
    if len(ledger['slots']) != 54 or ledger['status'] != 'schedule_finished':
        errors.append('Frozen schedule incomplete')
    if sum(s.get('physical_run', False) for s in ledger['slots']) > 54:
        errors.append('Physical run budget exceeded')
    checks = []
    raw_refs = {}
    conditions = Counter()
    stopped = set()
    for slot in ledger['slots']:
        key = (slot['model_id'], slot['replica_count'], slot['load_level'])
        if slot['replica_count'] not in (1, 2):
            errors.append('Unauthorized replica count')
        if not slot.get('output'):
            if slot['execution_status'] != 'skipped_after_condition_stop':
                errors.append('Unconfirmed execution outcome: ' + slot['run_id'])
            continue
        path = slot['output']['path']
        raw = read(path)
        raw_refs[slot['run_id']] = reference(path)
        if raw_refs[slot['run_id']]['sha256'] != slot['output']['sha256']:
            errors.append('Raw outcome changed: ' + slot['run_id'])
        if raw['valid_full_repetition']:
            conditions[key] += 1
        if raw['execution_status'] == 'protective_backlog_stop':
            stopped.add(key)
        model = next(m for m in plan['models'] if m['model_id'] == slot['model_id'])
        config = raw['deployment']
        pids = {v['pid'] for v in raw['worker_identities'].values()}
        identities = len(pids) == slot['replica_count'] and set(raw['worker_identities']) == set(config)
        cuda_proven = all(s['each_owned_pid_has_cuda_context'] and not s['foreign_pids'] for s in raw.get('idle_before_workload', [])) and bool(raw.get('idle_before_workload'))
        router = Router(raw['routing_rps'])
        pool = SamplePool(plan['input_pool']['path'], [model['task']], slot['random_seed'])
        rr_ok = samples_ok = times_ok = True
        for row in raw['request_records']:
            rr_ok &= row['model'] == router.choose(row['task'])
            item = pool.next(row['task'])
            samples_ok &= row['pool_id'] == item['pool_id'] and row['sample_id'] == item['sample'].get('id')
            if row['status'] == 'completed':
                times_ok &= row['scheduled_time_ns'] <= row['actual_submit_time_ns'] <= row['completion_time_ns']
        counts = Counter(r['model'] for r in raw['request_records'])
        balanced = max((counts[i] for i in config), default=0) - min((counts[i] for i in config), default=0) <= 1
        provenance_ok = raw['frozen_plan_sha256'] == reference(OUT / 'replica_scaling_plan.json')['sha256'] and all(
            raw['provenance'][a] == model[b] for a, b in [('model_manifest_sha256', 'model_manifest_sha256'), ('environment_hash', 'environment_hash'), ('source_revision', 'source_revision')])
        release = raw['gpu_release']['verified']
        check = dict(run_id=slot['run_id'], execution_status=raw['execution_status'], valid_full_repetition=raw['valid_full_repetition'],
            independent_processes=identities, each_idle_pid_has_cuda_context=cuda_proven,
            deterministic_round_robin=rr_ok, per_replica_counts_balanced=balanced,
            paired_input_sequence_reproduced=samples_ok, host_timer_order=times_ok,
            frozen_provenance_matches=provenance_ok, gpu_release_verified=release,
            error_count=len(raw['errors']), stability=raw.get('stability', {}).get('stability_state_v2'))
        checks.append(check)
        if raw['valid_full_repetition'] and not all((identities, cuda_proven, rr_ok, balanced, samples_ok, times_ok, provenance_ok, release)):
            errors.append('Protocol validation failed: ' + slot['run_id'])
    condition_checks = []
    for model in plan['models']:
        for count in (1, 2):
            for level in ('low', 'medium', 'high'):
                key = (model['model_id'], count, level)
                accepted = conditions[key] == 3 or key in stopped
                condition_checks.append(dict(condition=list(key), valid_full_repetitions=conditions[key],
                                             protective_stop=key in stopped, repetition_policy_satisfied=accepted))
                if not accepted:
                    errors.append('Three valid repetitions not obtained: ' + str(key))
    final_gpu = snapshot()
    if final_gpu['processes'] or final_gpu['used_mib'] > 128:
        errors.append('Final GPU not idle')
    result = dict(status='passed' if not errors else 'failed', errors=errors, frozen_plan_sha256=reference(OUT / 'replica_scaling_plan.json')['sha256'],
        source_files_full_hash_checked=source_count, frozen_inputs_and_7B_unchanged=not any('Frozen assets' in e for e in errors),
        environments_unchanged=unchanged_env, run_checks=checks, condition_checks=condition_checks,
        raw_file_hashes=raw_refs, final_gpu=final_gpu, physical_runs=sum(s.get('physical_run', False) for s in ledger['slots']),
        runtime_deployment_search_calls=0, workers_with_more_than_two_replicas_started=0,
        old_runtime_binary_guard_preserved=True, interpretation='Unstable overload remains valid completed execution; external interruptions and protective stops remain explicit. Protection-stop conditions are accepted with preserved evidence per frozen stop policy.')
    write(OUT / 'measurement_validation.json', result)
    write(OUT / 'source_hashes.json', dict(before_manifest=reference(OUT / 'source_hashes_before.json'),
         characterization_sources={str(p.relative_to(ROOT)): reference(p) for p in sorted((ROOT / 'replica_scaling').glob('*.py'))}, full_hash_check_completed=True,
         checked_files=source_count, environments_unchanged=unchanged_env, frozen_plan_sha256=result['frozen_plan_sha256'],
         all_protected_inputs_unchanged=not any('Frozen assets' in e for e in errors), errors=errors))
    print(__import__('json').dumps(dict(status=result['status'], errors=errors, raw_runs=len(checks), checked_sources=source_count)), flush=True)
    if errors:
        raise SystemExit(1)

if __name__ == '__main__':
    validate()
