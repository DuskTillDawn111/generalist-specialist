"""Verify the replica-only closeout without requiring deferred cost measurements."""
from collections import defaultdict
import hashlib
import subprocess
from .common import *
from replica_scaling.telemetry import snapshot


def main():
    freeze = read(OUT / 'input_freeze.json')
    verify(freeze['protected_references'])
    verify([freeze[k] for k in ['measurement_plan', 'resident_set_manifest', 'samples']])
    sup = read(OUT / 'closed_loop_source_freeze.json')
    verify(sup['periodic_raw_data_refs_at_supplement_freeze'])
    verify([sup['protocol']])
    revision = read(OUT / 'closed_loop_source_revision_v2.json')
    verify(revision['effective_harness'])
    archived_sources = {r['original_reference']['path']: r for r in revision['archived_old_sources']}
    for original in sup['harness']:
        if original['path'] in archived_sources:
            archived = archived_sources[original['path']]
            verify([archived['archive_reference']])
            assert original['sha256'] == archived['archive_reference']['sha256']
        else:
            verify([original])
    verify(read(OUT / 'periodic_pre_closed_loop_archive/manifest.json')['files'])
    verify(read(OUT / 'periodic_completion_manifest.json')['immutable_raw_refs'])
    for entry in read(OUT / 'fixed_cost_deferred_archive/manifest.json')['files']:
        for field in ['original_path', 'archive_path']:
            assert hashlib.sha256(Path(entry[field]).read_bytes()).hexdigest() == entry['sha256']
    assert read(OUT / 'replica_completion_status.json')['status'] == 'COMPLETE'
    capacity = read(OUT / 'updated_capacity_witnesses.json')
    verify([capacity['calibration'], capacity['directions']])
    assert len(capacity['rows']) == 360
    for fit in capacity['rows']:
        assert fit['status'] == 'OPTIMAL_EMPIRICAL_SPACE'
        assert fit['rho_limit'] == .8 and fit['replica_certificate']
        assert fit['safe_memory_mib'] <= fit['memory_budget_mib'] + 1e-6
        assert fit['max_capacity_residual'] <= 1e-8
        assert fit['max_conservation_residual_rps'] <= 1e-7
    windows = read(OUT / 'updated_windows.json')['rows']
    assert len(windows) == 120 and all(w['replica_certificate'] for w in windows)
    groups = defaultdict(list)
    identities = defaultdict(list)
    raws = []
    fixed_cost_raw = []
    for p in sorted((OUT / 'raw_runs').glob('*.json')):
        r = read(p); slot = r['slot']
        assert r['gpu_release']['verified']
        assert slot == read(OUT / 'slots' / f"{slot['run_id']}.json")
        worker_ids = tuple(sorted(w['pid'] for w in r['worker_identities'].values()))
        assert len(set(worker_ids)) == len(worker_ids)
        if slot['kind'] == 'closed_loop_profile':
            assert r['valid_session']
            phase = r['phases']['closed_loop_capacity']
            assert phase['max_outstanding_per_replica'] == 1 and 'stability' not in phase
            assert r['closed_loop_warmup']['all_requests_completed_after_drain']
            key = (slot['model'], slot['required_x'], slot['deployment'][slot['model']])
            groups[key].append(slot['seed']); identities[key].append(worker_ids)
        elif slot['kind'] == 'fixed_cost':
            fixed_cost_raw.append(ref(p))
        raws.append(ref(p))
    assert not fixed_cost_raw, 'No fixed-cost sessions had started when user deferred them; none may start now'
    seeds = sorted(read(OUT / 'measurement_plan.json')['seeds'])
    for key, observed in groups.items():
        assert sorted(observed) == seeds, (key, observed)
        assert len(set(identities[key])) == 3
    c, hw, roles, base = inputs()
    scaling = read(OUT / 'replica_scaling.json')
    specialist_models = {r['model'] for r in read(OUT / 'measurement_plan.json')['required_initial_frontier'] if roles[r['model']] == 'specialist'}
    assert specialist_models == set(scaling['terminal_frontiers'])
    for m, terminal in scaling['terminal_frontiers'].items():
        assert not any(model == m and frontier > terminal['x'] for model, frontier, count in groups)
    for m in c:
        for n, v in base.capacity_factors[m].items():
            assert scaling['capacity_factors'][m][str(n)] == v
        for n, v in base.memory_table_mib[m].items():
            assert scaling['memory_table_mib'][m][str(n)] == v
    assert not subprocess.run(['pgrep', '-af', '[n]vidia-cuda-mps'], capture_output=True, text=True).stdout.strip()
    gpu = snapshot()
    assert not gpu['processes']
    outputs = ['replica_evidence_required.csv', 'replica_profile_results.csv', 'replica_scaling.json',
        'updated_multihotspot_capacity_envelopes.csv', 'updated_second_generalist_regions.csv',
        'replica_evidence_gate.json', 'replica_scaling_impact.csv', 'replica_scaling_impact.json',
        'module11b_replica_summary.json', 'module11b_summary.md', 'fixed_cost_execution_status.json']
    assert all((OUT / n).exists() for n in outputs)
    save('replica_completion_verification.json', {'status': 'PASS', 'at_utc': now(),
        'protected_files_verified': len(freeze['protected_references']),
        'old_periodic_and_closed_loop_observations_preserved': True,
        'historical_calibration_values_unchanged': True, 'Module9_unchanged': True,
        'frozen_capacity_protocol_and_harness_verified': True,
        'independent_closed_loop_repetition_groups': len(groups), 'raw_observations': len(raws),
        'terminal_Specialist_models': sorted(specialist_models), 'higher_terminal_counts_not_artificial_missing': True,
        'CPU_capacity_rows': 360, 'CPU_window_contexts': 120,
        'fixed_cost_GPU_runs': 0, 'fixed_cost_status': 'DEFERRED_BY_USER',
        'prepared_cost_artifacts_preserved': True, 'GPU_released': gpu, 'MPS_off': True,
        'formal_hotspot_GPU_runs': 0, 'Task_Rich_started': False, 'sensitivity_audit_started': False})
    save('replica_output_manifest.json', {'at_utc': now(), 'outputs': [ref(OUT / n) for n in outputs],
        'raw_observations': raws, 'verification': ref(OUT / 'replica_completion_verification.json'),
        'final_module11b_sources': [ref(p) for p in (ROOT / 'module11b').glob('*.py')],
        'closed_loop_protocol': ref(OUT / 'closed_loop_protocol.json'),
        'closed_loop_source_revision': ref(OUT / 'closed_loop_source_revision_v2.json'),
        'deferred_cost_archive': ref(OUT / 'fixed_cost_deferred_archive/manifest.json'),
        'execution_boundary': ref(OUT / 'execution_boundary_revision_replica_only.json')})
    print(json.dumps({'verification': 'PASS', 'observations': len(raws), 'closed_loop_groups': len(groups)}), flush=True)


if __name__ == '__main__':
    main()
