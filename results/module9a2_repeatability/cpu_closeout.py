"""CPU-only closeout; preserve frozen execution code and every GPU observation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# The historical overlap utility imports its sibling as a top-level module.
sys.path.insert(0, str(ROOT / 'joint_deploy'))
from module9a2_repeatability.common import OUT, PILOT, load, save, sha, verify, now

def main():
    state = load(OUT / 'campaign_state.json')
    assert state['completed_runs'] == 50 and state['new_GPU_runs'] == 50
    if not (OUT / 'cpu_analysis_failure.json').exists():
        save('cpu_analysis_failure.json', state | {
            'phase': 'CPU historical overlap helper import',
            'remedy': 'Add existing sibling-module directory to CPU-only entrypoint sys.path',
            'GPU_retries': 0, 'frozen_sources_modified': False,
        })
    freeze = load(OUT / 'execution_freeze.json')
    protocol = load(OUT / 'protocol_identity.json')
    verify(freeze['execution_sources'] + freeze['core_files'])
    verify(protocol['frozen_refs'])
    save('campaign_state.json', {'status': 'ANALYZING', 'completed_runs': 50,
                                'new_GPU_runs': 50, 'CPU_only_closeout': True})
    from module9a2_repeatability.analyze import final
    final()
    from candidate_pool_audit.provenance import environment_snapshot
    assert environment_snapshot() == protocol['runtime_environment']
    assert sha(PILOT / 'pilot_gate.json') == protocol['source_9A1_gate_hash']
    assert load(PILOT / 'pilot_gate.json')['status'] == 'GATE_PARTIAL'
    save('validation_results.json', {
        'status': 'PASS', 'all_old_artifacts_and_sources_unchanged': True,
        'frozen_reference_count': len(protocol['frozen_refs']),
        'source_9A1_gate': 'GATE_PARTIAL', 'all_50_process_level_runs_complete': True,
        'all_exact_input_ID_and_warmup_count_checks_passed': True,
        'no_method_routing_ranking_workload_budget_revision': True,
        'full_Module9A_or_heldout_validation_started': False,
        'CPU_closeout_import_path_correction': True, 'GPU_retries': 0,
    })
    save('campaign_state.json', {'status': 'COMPLETE', 'completed_runs': 50,
                                'new_GPU_runs': 50, 'completed_at_utc': now(),
                                'CPU_only_closeout': True})
    save('artifact_hashes.json', {
        str(f.relative_to(OUT)): {'path': str(f.resolve()), 'sha256': sha(f)}
        for f in sorted(OUT.rglob('*'))
        if f.is_file() and f.name not in ['artifact_hashes.json', 'execution.log']
    })
    print(load(OUT / 'repeatability_gate.json'))

if __name__ == '__main__':
    main()
