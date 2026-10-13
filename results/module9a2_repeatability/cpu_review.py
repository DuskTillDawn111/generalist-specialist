"""Read-only measurement review and derived CSVs; no runtime or rule changes."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from module9a2_repeatability.common import *
import numpy as np

def main():
    p = load(OUT / 'protocol_identity.json')
    freeze = load(OUT / 'execution_freeze.json')
    verify(freeze['execution_sources'] + freeze['core_files'])
    runs = {}; rows = []; matched = []; all_pids = set(); overlaps = []
    peaks = []; errors = []; arrival_delays = []; measurements = 0
    for key, s in p['states'].items():
        runs[key] = []
        for rep in range(1, 6):
            r = load(OUT / 'raw_runs' / f'{key}__rep{rep}.json')
            assert r['status'] == 'COMPLETE' and r['configured_window_completed']
            assert r['gpu_release']['verified'] and not r['frozen_subset_changed']
            assert all(v['exit_code'] == 0 and not v['forced'] for v in r['shutdown'].values())
            pids = {v['pid'] for v in r['ready'].values()}
            assert not pids & all_pids
            all_pids.update(pids)
            assert r['actual_peak_vram_mib'] <= s['budget_mib']
            peaks.append(r['actual_peak_vram_mib'])
            expected = [(v['worker'], v['item']['pool_id']) for v in s['warmup_sequence']]
            actual = [(v['worker'], v['pool_id']) for v in r['submission_records'] if v['category'] == 'warmup']
            assert actual == expected
            assert [v['pool_id'] for v in r['target_records']] == [v['pool_id'] for v in s['target_items']]
            assert [v['scheduled_offset_seconds'] for v in r['target_records']] == s['target_schedule_seconds']
            if s['protocol_family'] == 'historical_full_state_replay':
                expected = [(v['worker'], v['item']['pool_id'], v['offset_seconds']) for v in s['measurement_schedule']]
                actual = [(v['worker'], v['pool_id'], v['scheduled_offset_seconds']) for v in r['submission_records'] if v['phase'] == 'measurement']
                assert actual == expected
                expected = [(v['worker'], v['item']['pool_id']) for v in s['priming_sequence']]
                actual = [(v['worker'], v['pool_id']) for v in r['submission_records'] if v['category'] == 'priming']
                assert actual == expected
            elif s['co_model']:
                assert all(v['outstanding_co_requests'] == s['co_backlog'] for v in r['co_backlog_trace'])
                assert not r['overlap_filter_applied']
                overlaps += [v['active_overlap_ratio'] for v in r['target_records']]
            for v in r['target_records']:
                arrival_delays.append((v['submit_ns'] - r['measurement_start_unix_ns']) / 1e6 - v['scheduled_offset_seconds'] * 1000)
            errors += r['telemetry_errors'] + r['diagnostic_errors']
            measurements += len(r['target_records'])
            runs[key].append(r)
        lat = np.array([np.mean([v['worker_latency_ms'] for v in r['target_records']]) for r in runs[key]])
        rows.append({'state': key, 'n': 5, 'mean_latency_ms': float(lat.mean()),
                     'latency_std_ms': float(lat.std(ddof=1)),
                     'latency_CV': float(lat.std(ddof=1) / lat.mean()),
                     'latency_MAD_ms': float(np.median(abs(lat - np.median(lat)))),
                     'standalone_S_definition': 'UNDEFINED' if not s['co_model'] else 'see run_level_measurements.csv'})
    for key, s in p['states'].items():
        if not s['co_model']: continue
        single = 'historical_phi_asr_none' if s['protocol_family'] == 'historical_full_state_replay' else f'{s["model"]}__{s["task"]}__none'
        for rep, (a, b) in enumerate(zip(runs[single], runs[key]), 1):
            for x, y in zip(a['target_records'], b['target_records']):
                assert x['pool_id'] == y['pool_id'] and x['sample_index'] == y['sample_index']
                matched.append({'state': key, 'repetition': rep, 'sample_id': x['pool_id'],
                                'sample_index': x['sample_index'], 'standalone_latency_ms': x['worker_latency_ms'],
                                'pair_latency_ms': y['worker_latency_ms'], 'S_i': y['worker_latency_ms'] / x['worker_latency_ms']})
    csvsave('physical_condition_repeatability.csv', rows)
    csvsave('matched_input_request_ratios.csv', matched)
    assert measurements == 1220 and len(matched) == 760 and len(all_pids) == 100
    required = ['anchor_selection.json', 'protocol_identity.json', 'run_level_measurements.csv',
                'request_level_measurements.csv', 'matched_input_analysis.csv', 'noise_decomposition.json',
                'task_effect_repeatability.csv', 'task_effect_bootstrap.json', 'signal_to_noise_analysis.json',
                'historical_noise_source_audit.json', 'ground_truth_protocol_decision.json',
                'repeatability_gate.json', 'module9a2_summary.md']
    assert all((OUT / name).is_file() for name in required)
    assert not errors
    save('closeout_review.json', {'status': 'PASS', 'raw_runs': 50, 'independent_worker_PIDs': len(all_pids),
         'target_request_records': measurements, 'exact_matched_input_ratios': len(matched),
         'warmup_sample_ID_and_order_checks': True, 'target_sample_ID_order_and_schedule_checks': True,
         'historical_literal_worker_routing_and_request_timeline_checks': True,
         'backlog_64_at_every_current_pair_target': True,
         'minimum_current_pair_overlap': min(overlaps), 'maximum_peak_vram_mib': max(peaks),
         'normal_shutdown_and_GPU_release_all_runs': True, 'telemetry_or_diagnostic_errors': errors,
         'max_target_submit_delay_ms': max(arrival_delays), 'required_outputs': required,
         'original_execution_sources_and_core_unchanged': True, 'new_GPU_runs_in_review': 0})
    save('artifact_hashes.json', {str(f.relative_to(OUT)): {'path': str(f.resolve()), 'sha256': sha(f)}
         for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['artifact_hashes.json', 'execution.log']})
    print(load(OUT / 'closeout_review.json'))

if __name__ == '__main__': main()
