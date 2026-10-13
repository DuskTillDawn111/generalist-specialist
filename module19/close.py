"""Close the amended static benchmark after all analysis, without more GPU."""
from replica_scaling.telemetry import snapshot
from .common import *


def csvread(name):
    with (OUT / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    f = read(OUT / 'frozen_benchmark_manifest.json')
    verify(f['implementation'] + f['reused_implementation'] + [f['CPU'], f['input_protocol']])
    protected = read(OUT / 'protected_before.json')['files']
    verify(protected)
    state = read(OUT / 'campaign_status.json')
    measured = set(state['completed_runs'])
    skipped = {r['run_id'] for r in state['skipped_runs']}
    assert measured.isdisjoint(skipped)
    assert measured | skipped == {r['run_id'] for r in f['slots']}
    assert {p.stem for p in (OUT / 'raw_runs').glob('*.json')} == measured
    correction = read(OUT / 'benchmark_protocol_correction.json')
    verify(correction['controller_implementation'])
    user = read(OUT / 'user_v3_scope_resolution.json')
    assert user['full_V3_GPU_feedback_search_authorized'] is False
    cpu, gpu = csvread('cpu_baseline_comparison.csv'), csvread('gpu_strategy_comparison.csv')
    full_cpu = [r for r in cpu if r['strategy'] == 'BASELINE_C_ORIGINAL_V3']
    full_gpu = [r for r in gpu if r['strategy'] == 'BASELINE_C_ORIGINAL_V3']
    assert len(full_cpu) == 12 and len(full_gpu) == 36
    assert all(r['status'] == 'UNRESOLVED_RUNTIME_FEEDBACK_REQUIRED' and not r['deployment'] for r in full_cpu)
    assert all(r['measurement_source'] == 'NOT_RUN' for r in full_gpu)
    required = ['frozen_benchmark_manifest.json', 'cpu_baseline_comparison.csv',
                'gpu_strategy_comparison.csv', 'routing_ablation.csv',
                'resource_latency_tradeoff.csv', 'worker_stability.csv',
                'module19_gate.json', 'module19_summary.md',
                'routing_system_task_metrics.csv', 'routing_system_runs.csv',
                'routing_system_pair_differences.csv', 'routing_system_analysis.json']
    assert all((OUT / name).is_file() and (OUT / name).stat().st_size > 0 for name in required)
    audit = read(OUT / 'completion_audit.json')
    assert audit['all_slots_accounted_for'] and all(x['same_all_task_sample_schedule'] is not False for x in audit['same_seed_sample_checks'])
    assert all(r['release_verified'] for r in audit['measured'])
    assert not any(r['foreign_PID_samples'] for r in audit['measured'])
    hardware = snapshot()
    assert not hardware['processes'] and hardware['used_mib'] <= 128
    save('frozen_benchmark_manifest_effective.json', {
        'at_utc': now(), 'status': 'AMENDED_STATIC_SCOPE_COMPLETE_AND_STOPPED',
        'original_manifest_preserved': ref(OUT / 'frozen_benchmark_manifest.json'),
        'protocol_label_correction': ref(OUT / 'benchmark_protocol_correction.json'),
        'user_scope_resolution': ref(OUT / 'user_v3_scope_resolution.json'),
        'user_requested_system_analysis': ref(OUT / 'system_analysis_extension_protocol.json'),
        'full_Original_V3': 'UNVERIFIED; initialization diagnostics retained separately',
        'excluded_remaining_initializer_only_configs': correction['no_further_seed_only_configurations'],
        'static_A_B_Proposed_and_ablation_models_QPS_budgets_routes_unchanged': True,
        'measured_original_slots': len(measured), 'skipped_original_slots': len(skipped),
        'no_added_GPU_slots': True, 'no_Module20': True})
    gate = read(OUT / 'module19_gate.json')
    gate['effective_manifest'] = ref(OUT / 'frozen_benchmark_manifest_effective.json')
    gate['final_accounting_audit'] = ref(OUT / 'completion_audit.json')
    # Distinguish legacy target-only gates from the added whole-system decision.
    gate['target_routing_latency_or_throughput_benefit_supported_against_legal_S_only'] = gate.pop('routing_latency_or_throughput_benefit_supported_against_legal_S_only')
    gate['target_routing_latency_negative_or_boundary_tasks_including_diagnostics'] = gate.pop('routing_negative_or_boundary_tasks_including_diagnostics')
    gate['completed_at_utc'] = now()
    save('module19_gate.json', gate)
    save('final_integrity_audit.json', {
        'at_utc': now(), 'required_outputs': required, 'protected_history_files_verified': len(protected),
        'frozen_source_inputs_verified': True, 'unique_fresh_worker_PIDs': audit['fresh_unique_worker_PIDs'],
        'all_original_slots_accounted_for': True, 'no_extra_GPU': True,
        'all_same_seed_full_window_request_schedules_match': True,
        'full_V3_not_misreported_as_executed': True,
        'no_foreign_GPU_process_samples': True, 'GPU_released': hardware,
        'source': ref(ROOT / 'module19/close.py'), 'no_Module20': True})
    artifact = read(OUT / 'artifact_manifest.json')
    artifact.update({'at_utc': now(), 'gate': gate['gate'], 'status': 'COMPLETE_AND_STOPPED',
                     'artifacts': [ref(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.json'],
                     'implementation': [ref(p) for p in sorted((ROOT / 'module19').glob('*.py'))],
                     'no_Module20': True})
    save('artifact_manifest.json', artifact)
    print('MODULE19_CLOSED', gate['gate'], 'new runs', len(measured), 'skipped', len(skipped),
          'system classifications', gate['routing_benefit_classifications'])


if __name__ == '__main__':
    main()
