"""Read-only, post-campaign checks. Never starts a runtime measurement."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

from candidate_pool_audit.provenance import environment_snapshot
from deployment_search.runtime_cache import file_hash, save_json
from module8d_real_baselines.common import (
    ROOT, F, C, OUT, B, read, ref, verify_inputs, validate_raw, full_vector,
)


def main():
    verify_inputs()
    assert read(OUT / 'campaign_status.json')['status'] == 'complete'
    for filename in ('campaign_manifest.json', 'analysis_policy_freeze.json'):
        assert file_hash(OUT / filename) == (OUT / (filename + '.sha256')).read_text().strip()
    manifest = read(OUT / 'campaign_manifest.json')
    compat = read(OUT / 'confirmation_donor_compatibility_freeze.json')
    for entry in compat['compatibility_sources'].values():
        assert file_hash(entry['path']) == entry['sha256']
    control = read(OUT / 'execution_control_freeze.json')
    for field in ('controller_source', 'confirmation_compatibility', 'original_frozen_campaign'):
        entry = control[field]
        assert file_hash(entry['path']) == entry['sha256']
    assert environment_snapshot() == read(C / 'environment_manifest.json')
    source_validation = read(F / 'source_identity_validation.json')
    sources = {r['path']: r['actual_sha256'] for r in source_validation['native_files'].values()}
    sources.update({r['path']: r['sha256'] for r in source_validation['model_manifest_files'].values()})
    for path, sha in sources.items():
        assert file_hash(path) == sha, path

    frozen_states = read(F / 'baseline_deployments.json')['states']
    physical = read(OUT / 'new_measurement_manifest.json')
    reused = read(OUT / 'reuse_execution_log.json')
    assert len(physical['runs']) == physical['physical_attempts'] <= manifest['physical_run_cap']
    assert len({r['raw_result_path'] for r in physical['runs']}) == len(physical['runs'])
    assert sum(r.get('valid_timed_execution', False) for r in physical['runs']) == physical['valid_timed_runs']
    assert all(r.get('valid_timed_execution', False) for r in physical['runs'])
    for run in physical['runs']:
        for field in ('raw', 'config'):
            assert file_hash(run[field]['path']) == run[field]['sha256']
    assert reused['references'] == len(reused['events'])
    assert reused['unique_raw_paths'] == len({r['original_run']['path'] for r in reused['events']})
    for event in reused['events']:
        assert file_hash(event['original_run']['path']) == event['original_run']['sha256']
        assert event['proof']['decision'] == 'REUSE_ALLOWED'
        assert all(event['proof']['equality_checks'].values())

    raw_checks = {}
    def check_runtime(rt, identity, workload, expected_n):
        assert len(rt['repetitions']) == expected_n
        assert len(set(rt['raw_result_paths'])) == expected_n
        for raw_path in rt['raw_result_paths']:
            seed = read(Path(raw_path).with_suffix('.config.json'))['random_seed']
            key = (raw_path, json.dumps(identity, sort_keys=True), seed)
            if key not in raw_checks:
                raw_checks[key] = validate_raw(raw_path, identity, seed, workload)

    fixed = []
    for job in manifest['phase_A']:
        baseline, name = job['baseline'], job['workload']
        directory = OUT / baseline.lower() / name
        assert read(directory / 'status.json')['status'] == 'complete'
        result = read(directory / 'result.json')
        state = next(r for r in frozen_states if r['baseline'] == baseline and r['workload'] == name)
        assert result['state'] == state
        assert full_vector(result['identity']['deployment']) == full_vector(state['deployment'])
        # The formal freeze includes the excluded PaddleOCR-VL family with
        # zero flow; the serving catalog omits that disabled family entirely.
        for task, routing in state['routing_rps'].items():
            actual = result['identity']['routing'][task]
            assert all(actual.get(model, 0.0) == flow for model, flow in routing.items())
            assert all(routing.get(model, 0.0) == flow for model, flow in actual.items())
        assert result['identity']['workload'] == state['lambda_rps']
        check_runtime(result['runtime'], result['identity'], name, 3)
        check_runtime(result['confirmation'], result['identity'], name, 2)
        search_raw = set(result['runtime']['raw_result_paths'])
        confirmation_raw = set(result['confirmation']['raw_result_paths'])
        assert not search_raw & confirmation_raw
        owned_runs = [r for r in physical['runs'] if r['owner'] == f'{baseline}/{name}']
        expected_new = 0 if baseline == 'B0' or (baseline == 'B3' and name != 'no_vqa_phase') else 5
        assert len(owned_runs) == expected_new
        if baseline == 'B0':
            original = read(C / name / 'search_complete.json')['seed_runtime']
            assert result['runtime']['raw_result_paths'] == original['raw_result_paths']
        if baseline == 'B2' and name == 'no_vqa_phase':
            original = read(C / name / 'search_complete.json')['runtime']
            assert not search_raw & set(original['raw_result_paths'])
        fixed.append({'baseline': baseline, 'workload': name, 'new_runs': len(owned_runs), 'status': 'PASS'})
    static = [r['deployment'] for r in frozen_states if r['baseline'] == 'B3']
    assert len({json.dumps(r, sort_keys=True) for r in static}) == 1

    random = []
    for job in manifest['phase_B']:
        name, seed = job['workload'], job['random_seed']
        directory = OUT / f'b4_seed_{seed}' / name
        assert read(directory / 'status.json')['status'] == 'complete'
        trace = read(directory / 'search_complete.json')
        assert trace['random_seed'] == seed and trace['LP_rank_calls'] == trace['enumeration_calls'] == 0
        assert trace['runtime_evaluation_count'] <= 6 and trace['search_iterations'] <= 3
        journal = read(directory / 'logical_evaluations.json')['evaluations']
        assert len(journal) == trace['runtime_evaluation_count']
        assert len({r['cache_key'] for r in journal}) == len(journal)
        frozen = next(r for r in frozen_states if r['baseline'] == 'B4_SEED' and r['workload'] == name and r['random_seed'] == seed)
        assert full_vector(trace['seed']['deployment']) == full_vector(frozen['deployment'])
        assert journal[0]['identity']['workload'] == frozen['lambda_rps']
        for row in journal:
            check_runtime(row['runtime'], row['identity'], name, 3)
        conf = read(directory / 'confirmation_results.json')
        assert conf['complete'] and not conf['influences_search']
        donors = []
        for role in ('seed', 'final'):
            rows = conf['repetitions'][role]
            assert sorted(r['repetition'] for r in rows) == [1, 2]
            identity = journal[0]['identity'] if role == 'seed' else next(r['identity'] for r in journal if r['cache_key'] == trace['runtime']['cache_key'])
            check_runtime(conf['aggregates'][role], identity, name, 2)
            donors.extend(r['raw_result_path'] for r in rows)
            if role == 'seed':
                assert {r['raw_result_path'] for r in rows} == {str(C / name / 'confirmation' / f'seed_rep{i}.json') for i in (1, 2)}
        assert len(set(donors)) == 4
        all_search_raw = {p for r in journal for p in r['runtime']['raw_result_paths']}
        assert not all_search_raw & set(donors)
        random.append({'workload': name, 'random_seed': seed, 'logical_states': len(journal), 'status': 'PASS'})

    expected_rows = {'static_baseline_comparison.csv': 30, 'random_baseline_comparison.csv': 24,
                     'deployment_comparison.csv': 48, 'search_cost_comparison.csv': 24, 'quality_compliance.csv': 48}
    tables = {}
    for filename, expected in expected_rows.items():
        with (OUT / filename).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == expected
        tables[filename] = len(rows)
        if filename == 'quality_compliance.csv':
            assert all(r['quality_admissible'] == 'True' and r['excluded_path_used'] == 'False' for r in rows)
        if filename in ('static_baseline_comparison.csv', 'random_baseline_comparison.csv'):
            proposed = {r['workload']: r for r in rows if r['method'] == 'Proposed'}
            for row in rows:
                for value, gain in (('mean_e2e_ms', 'relative_mean_improvement_Proposed_percent'),
                                    ('weighted_task_p95_ms', 'relative_P95_improvement_Proposed_percent')):
                    b, p = float(row[value]), float(proposed[row['workload']][value])
                    assert math.isclose(float(row[gain]), 100 * (b - p) / b, abs_tol=1e-10)
    audit = read(OUT / 'baseline_claim_audit.json')
    assert audit['MODULE8E_READY'] and not audit['MODULE8E_started'] and not audit['remaining_blockers']
    from replica_scaling.telemetry import snapshot
    gpu = snapshot()
    assert not gpu['processes'] and gpu['used_mib'] <= 128
    result = {'status': 'PASS', 'all_frozen_inputs_and_model_sources_unchanged': True,
              'source_files_rehashed': len(sources), 'fixed_conditions': fixed, 'random_conditions': random,
              'validated_unique_raw_identity_pairs': len(raw_checks), 'table_rows': tables,
              'new_physical_timed_runs': physical['valid_timed_runs'], 'reuse_references': reused['references'],
              'unique_reused_raw_runs': reused['unique_raw_paths'], 'all_21_admitted_paths_only': True,
              'random_LP_ranking_calls': 0, 'enumeration_calls': 0, 'same_per_job_runtime_budget': True,
              'seed_final_confirmations_disjoint': True, 'GPU_idle': gpu,
              'module8E_started': False, 'validator_source': ref(Path(__file__).resolve())}
    save_json(OUT / 'final_execution_validation.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'new_physical_timed_runs', 'reuse_references', 'table_rows')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
