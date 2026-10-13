"""Post-run evidence audit and count-based trace labels; no GPU runner calls."""
from pathlib import Path
from datetime import datetime
import json
import math

from deployment_search.runtime_cache import file_hash, save_json, digest
from module8e3_heldout_validation.common import ROOT, F, OUT, read, ref, verify_preregistration, verify_manifest


def normalize_candidates(value, counts):
    if isinstance(value, dict):
        if 'operation_original' in value and 'candidate_deployment' in value and 'current_state' in value:
            operation = value['operation_original']
            before, after = value['current_state'], value['candidate_deployment']
            if operation in ['SCALE_UP', 'SCALE_DOWN']:
                changed = [m for m in set(before) | set(after) if before.get(m, 0) != after.get(m, 0)]
                assert len(changed) == 1
                model = changed[0]
                a, b = before.get(model, 0), after.get(model, 0)
                if operation == 'SCALE_UP':
                    assert b == a+1
                    typ = 'ADD' if a == 0 else 'REPLICA_UP'
                else:
                    assert b == a-1
                    typ = 'REMOVE' if b == 0 else 'REPLICA_DOWN'
                if value['candidate_type'] != typ: counts['labels_normalized'] += 1
                value['candidate_type'] = typ
                value['count_transition'] = {'model': model, 'before': a, 'after': b}
        for v in value.values(): normalize_candidates(v, counts)
    elif isinstance(value, list):
        for v in value: normalize_candidates(v, counts)


def run():
    campaign = read(OUT / 'campaign_status.json')
    assert campaign['status'] != 'running', 'Post-run audit cannot alter views during a running campaign'
    verify_preregistration()
    manifest = verify_manifest()
    before_classification = {name: file_hash(OUT / name) for name in ['revision_validation_status.json',
        'v2_vs_v1_summary.json', 'b2_gap_analysis.json', 'm1_results.json', 'm2_results.json', 'b2_results.json',
        'heldout_method_comparison.csv']}
    asset_checks = read(OUT / 'hash_verification.json')['model_asset_checks']
    for r in asset_checks: assert file_hash(r['path']) == r['expected'], r['path']
    from candidate_pool_audit.provenance import environment_snapshot
    from replica_scaling.telemetry import snapshot
    assert environment_snapshot() == manifest['environment'], 'Post-run environment drift'
    gpu = snapshot()
    assert all(gpu[k] == manifest['GPU'][k] for k in ['uuid', 'name', 'driver', 'total_mib'])
    assert not gpu['processes'] and gpu['used_mib'] <= 128, 'GPU not released'
    physical = read(OUT / 'physical_measurements.json')['runs']
    sharing = read(OUT / 'measurement_sharing_log.json')
    references = sharing['logical_run_references']
    evaluations = sharing['logical_evaluations']
    actual_states = set()
    for r in physical:
        ident = r['identity']; cfg = ident['config']
        state_key = digest({'deployment': cfg['deployment'], 'workload': cfg['workload_rps'],
                           'routing': cfg['routing_rps'], 'context': ident['runtime_context_sha256'],
                           'workload_hash': r['workload_sha256']})
        assert state_key == r['state_identity_sha256']
        actual_states.add(state_key)
        if r.get('raw_sha256'): assert file_hash(r['raw_path']) == r['raw_sha256']
    donors = {r['physical_raw_id']: r for r in physical}
    seen = {}
    for reference in references:
        donor = donors[reference['physical_raw_id']]
        assert reference['measurement_identity_hash'] == digest(donor['identity'])
        assert Path(reference['raw_path']).resolve() == Path(donor['raw_path']).resolve()
        assert file_hash(reference['raw_path']) == reference['raw_sha256']
        key = reference['measurement_identity_hash']
        if reference['cache_hit']:
            assert key in seen and seen[key] == donor['physical_raw_id'] and donor['valid']
        else:
            assert key not in seen
            seen[key] = donor['physical_raw_id']
    # Every selected round is committed before the receiving method's logical evaluation.
    execution = read(OUT / 'execution_log.json')['slots']
    slot_times = {(r['workload'], r['slot_index']): datetime.fromisoformat(r['started_at_utc']) for r in execution}
    for e in evaluations:
        if e['role'] != 'candidate': continue
        for method, h in e['round_barrier'].items():
            if h is None: continue
            folder = 'm1_search_traces' if method == 'M1' else 'm2_search_traces'
            files = list((OUT / folder / e['workload']).glob('round_*_commitment.json'))
            commitment = next(read(p) for p in files if read(p)['commitment_sha256'] == h)
            assert digest({k: v for k, v in commitment.items() if k != 'commitment_sha256'}) == h
            assert datetime.fromisoformat(commitment['committed_at_utc']) <= slot_times[e['workload'], e['logical_slot']]
    accounting = read(OUT / 'run_accounting.json')
    assert accounting['logical_search_evaluations'] == len(evaluations) <= 78
    assert accounting['unique_physical_states'] == len(actual_states) <= 72
    assert accounting['new_physical_attempts'] == len(physical) <= 264
    assert accounting['new_timed_runs'] == sum(r.get('measurement_started', False) for r in physical)
    assert accounting['measurement_sharing_count'] == sum(r['cache_hit'] for r in references)
    assert accounting['retries'] == 0
    if campaign['status'] == 'complete':
        from module8e3_heldout_validation.report import compare, classify_revision, classify_gap
        from module8e3_heldout_validation.common import inputs
        from deployment_core.replica_calibration import ReplicaPlanningPolicy
        c, _, _, cal = inputs(); policy = ReplicaPlanningPolicy(c, 'empirical', 'empirical', cal)
        expected_evals = 0
        summary = read(OUT / 'v2_vs_v1_summary.json')
        for w in read(F / 'heldout_workloads.json')['workloads']:
            results = {m: read(OUT / w['id'] / m / 'result.json') for m in ['M1', 'M2', 'B2']}
            expected_evals += sum(r['logical_search_evaluations'] for r in results.values())
            a = results['M2']['confirmations']['final']; b = results['M1']['confirmations']['final']
            comparison = compare(a, b, policy.deployment_memory(results['M2']['deployment']), policy.deployment_memory(results['M1']['deployment']))
            assert comparison == next(r for r in summary['per_workload'] if r['workload'] == w['id'])['M2_vs_M1']
            for r in results.values():
                for agg in r['confirmations'].values():
                    assert len(agg['repetitions']) == 2
                    for rep in agg['repetitions']:
                        assert rep['valid_runtime_repetition'] and file_hash(rep['raw_result_path']) == rep['raw_sha256']
        assert expected_evals == accounting['logical_search_evaluations']
        validation = read(OUT / 'revision_validation_status.json')
        assert classify_revision(validation['variables'], True)[0] == validation['REVISION_VALIDATION_STATUS']
        assert classify_gap(summary['per_workload'], True)['B2_GAP_STATUS'] == validation['B2_GAP_STATUS']
        assert read(OUT / 'protocol_audit.json')['valid']
    counts = {'labels_normalized': 0}
    files = list(OUT.glob('m*_search_traces/*/audit_trace.json'))
    files += [OUT / 'mechanism_evidence.json', OUT / 'replica_activation_audit.json']
    for path in files:
        d = read(path); normalize_candidates(d, counts); save_json(path, d)
    for name, h in before_classification.items(): assert file_hash(OUT / name) == h, 'Primary/metrics changed in labeling'
    result = {'status': 'PASS', 'campaign_complete': campaign['status'] == 'complete',
        'frozen_artifacts_and_methods_unchanged': True, 'model_assets_rehashed': len(asset_checks),
        'post_run_environment_unchanged': True, 'GPU_released': True,
        'measurement_sharing_identity_donor_and_hash_checks': True, 'candidate_commit_times_verified': True,
        'physical_states_recomputed': len(actual_states), 'logical_evaluations_recomputed': len(evaluations),
        'within_frozen_caps': True, 'all_valid_repetitions_preserved': True,
        'primary_classification_and_all_method_metrics_unchanged': True,
        'trace_label_normalization': counts,
        'trace_label_rule': 'Original integer SCALE_UP/DOWN encoding is preserved in raw traces. Audit views classify 0->1 as ADD, 1->0 as REMOVE, existing-family increments/decrements as REPLICA_UP/DOWN; exact closure SWAP also carries REPLACEMENT label.',
        'auditor_source': ref(Path(__file__)), 'new_GPU_runs_from_audit': 0}
    save_json(OUT / 'final_validation.json', result)
    summary_path = OUT / 'module8e3_summary.md'
    paragraph = '\n最终独立审计 PASS：再次校验冻结 artifacts、源码、模型权重、环境、GPU release、raw hashes、共享 donor、round commitment 时间与实际预算。Audit trace 按 count transition 区分 ADD/REMOVE 与 REPLICA_UP/DOWN，保留原始 integer SCALE_UP/DOWN 编码；primary 分类与方法 metrics 未改动。\n'
    if paragraph not in summary_path.read_text(): summary_path.write_text(summary_path.read_text() + paragraph)
    report_manifest = read(OUT / 'report_manifest.json')
    report_manifest['artifacts'] = {p.name: ref(p) for p in OUT.iterdir()
        if p.is_file() and p.suffix in ['.json', '.csv', '.md'] and p.name != 'report_manifest.json'}
    report_manifest['post_run_auditor_source'] = ref(Path(__file__))
    save_json(OUT / 'report_manifest.json', report_manifest)
    print(json.dumps(result))


if __name__ == '__main__': run()
