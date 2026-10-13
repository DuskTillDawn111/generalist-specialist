"""Read-only independent verification of saved Module 4B GPU evidence."""
import json
import math
from collections import Counter
from pathlib import Path

from deployment_core.workload import Workload
from deployment_search.runtime_cache import digest, file_hash, save_json
from deployment_search.runtime import weighted_task_latency

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def main():
    required = ('pilot_freeze.json', 'routing_search_config.json', 'runtime_cache_manifest.json',
        'routing_search_trace.json', 'routing_search_trace.md', 'runtime_evaluations.json',
        'decision_audit.json', 'confirmation_results.json', 'module4b_summary.md')
    for name in required:
        assert (OUT / name).is_file(), name
    freeze = read(OUT / 'pilot_freeze.json')
    assert file_hash(freeze['workload_source']['path']) == freeze['workload_source']['sha256']
    assert file_hash(freeze['deployment_source']['path']) == freeze['deployment_source']['sha256']
    assert file_hash(OUT / 'routing_search_config.json') == freeze['config_sha256']
    for path, sha in freeze['implementation_sha256'].items():
        assert file_hash(ROOT / path) == sha, path
    repair = read(OUT / 'implementation_repair_audit.json')
    old_freeze = read(OUT / 'pilot_freeze_before_recording_repair.json')
    old_sources = old_freeze.pop('implementation_sha256')
    current_freeze = dict(freeze)
    current_sources = current_freeze.pop('implementation_sha256')
    assert old_freeze == current_freeze
    changed_sources = [p for p in current_sources if current_sources[p] != old_sources[p]]
    assert changed_sources == ['deployment_search/routing_real_pilot.py']
    assert current_sources[changed_sources[0]] == repair['new_runner_sha256']
    assert old_sources[changed_sources[0]] == repair['old_runner_sha256']
    trace = read(OUT / 'routing_search_trace.json')
    records = read(OUT / 'runtime_evaluations.json')
    manifest = read(OUT / 'runtime_cache_manifest.json')
    assert manifest == read(OUT / 'search_cache/runtime_cache_manifest.json')
    assert trace['deployment'] == freeze['fixed_deployment']
    assert trace['initial_routing'] == freeze['canonical_routing']
    assert read(OUT / 'decision_audit.json')['passed']
    assert 1 <= len(records) <= 6
    w = Workload(freeze['frozen_workload']['arrival_rates_rps'])
    seen = set()
    for record in records:
        identity, runtime = record['full_identity'], record['runtime']
        assert identity['deployment'] == freeze['fixed_deployment']
        assert identity['workload'] == w.arrival_rate
        assert digest(identity) == runtime['cache_key']
        assert runtime['cache_key'] not in seen
        seen.add(runtime['cache_key'])
        assert manifest['configurations'][runtime['cache_key']]['identity'] == identity
        assert len(runtime['repetitions']) == 3
        for rep in runtime['repetitions']:
            assert rep['run_status'] == 'completed'
            assert file_hash(rep['raw_result_path']) == rep['raw_sha256']
            raw = read(rep['raw_result_path'])
            assert raw['deployment'] == identity['deployment']
            assert raw['workload_rps'] == identity['workload']
            assert raw['routing_rps'] == identity['routing']
            assert raw['random_seed'] == rep['random_seed']
            expected_mean = weighted_task_latency(w, {k: raw['per_task_metrics'][k]['mean_e2e_latency_ms'] for k in w.active_tasks})
            expected_p95 = weighted_task_latency(w, {k: raw['per_task_metrics'][k]['p95_e2e_latency_ms'] for k in w.active_tasks})
            assert math.isclose(expected_mean, rep['weighted_mean_e2e'], abs_tol=1e-9)
            assert math.isclose(expected_p95, rep['weighted_p95_e2e'], abs_tol=1e-9)
            counts = Counter((r['task'], r['model']) for r in raw['request_records'])
            for task in w.active_tasks:
                for model, rate in identity['routing'][task].items():
                    assert counts[task, model] == 0 if rate == 0 else counts[task, model] > 0
    for row in trace['iterations']:
        for c in row['candidates']:
            if c['objective'] is not None:
                assert c['selected_top_b']
                assert c['objective']['cache_key'] in seen
    frozen = read(OUT / 'search_decision_frozen.json')
    assert frozen['trace_sha256'] == digest(trace)
    assert frozen['final_routing'] == trace['final_routing']
    assert frozen['audit_sha256'] == digest(read(OUT / 'decision_audit.json'))
    conf = read(OUT / 'confirmation_results.json')
    if trace['final_routing'] == trace['initial_routing']:
        assert not conf['required']
    else:
        assert conf['required'] and conf['complete']
        for which, route in [('canonical', trace['initial_routing']), ('final', trace['final_routing'])]:
            assert len(conf['repetitions'][which]) == 2
            for rep in conf['repetitions'][which]:
                assert rep['run_status'] == 'completed'
                assert rep['random_seed'] not in records[0]['full_identity']['measurement_settings']['repetition_seeds']
                assert file_hash(rep['raw_result_path']) == rep['raw_sha256']
                raw = read(rep['raw_result_path'])
                flat = {f'{k}::{m}': v for k, models in raw['routing_rps'].items() for m, v in models.items()}
                assert flat == route
    report = dict(passed=True, required_artifacts=list(required),
        frozen_experiment_inputs_unchanged=True,
        source_hashes_match_audited_serialization_repair=True, exact_identity_audit_passed=True,
        raw_runtime_metrics_recomputed=True, actual_request_routing_verified=True,
        search_decision_unchanged_by_confirmation=True,
        valid_unique_evaluations=len(records), physical_search_attempts=len(manifest['run_attempts']))
    save_json(OUT / 'post_run_verification.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
