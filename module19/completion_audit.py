"""Post-measurement accounting and historical drift audit; never controls serving."""
from collections import Counter, defaultdict
from .common import *
from .metrics import summarize


def signature(raw):
    phase = raw['phases'].get('measurement', {})
    start = phase.get('window', {}).get('start_time_ns', 0)
    return [(r['arrival_index'], r['task'], r['pool_id'],
             r['scheduled_time_ns'] - start)
            for r in phase.get('request_records', [])]


def main():
    frozen = read(OUT / 'frozen_benchmark_manifest.json')
    verify(frozen['implementation'] + frozen['reused_implementation'] +
           [frozen['CPU'], frozen['input_protocol']])
    protected = read(OUT / 'protected_before.json')['files']
    verify(protected)
    state = read(OUT / 'campaign_status.json')
    slots = {s['run_id']: s for s in frozen['slots']}
    configs = {c['config_id']: c for c in frozen['configurations']}
    cases = {c['case_id']: c for c in frozen['cases']}
    measured, pids, draws, drift, accounting, routing, decomposition = [], [], defaultdict(list), [], [], [], []
    for path in sorted((OUT / 'raw_runs').glob('*.json')):
        raw = read(path)
        rid = raw['slot']['run_id']
        assert rid in slots and raw['slot'] == slots[rid]
        slot = slots[rid]
        config = configs[slot['config_id']]
        case = cases[config['owner_case']]
        row, _, workers, paths, _ = summarize(raw, config, case)
        assert raw['transport_config']['workload_rps'] == slot['lambda_rps']
        counts = Counter(v['family'] for v in raw['worker_identities'].values())
        if raw['status'] == 'COMPLETED':
            assert dict(counts) == slot['deployment']
        assert all(v['READY']['pid'] == v['pid'] for v in raw['worker_identities'].values())
        pids.extend(v['pid'] for v in raw['worker_identities'].values())
        phase = raw['phases'].get('measurement', {})
        records = phase.get('request_records', [])
        accepted = sum(r.get('actual_submit_time_ns') is not None for r in records)
        end = phase.get('window', {}).get('end_time_ns', 0)
        sent_by_end = sum(r.get('actual_submit_time_ns') is not None and
                          r['actual_submit_time_ns'] <= end for r in records)
        outcomes = Counter(r['status'] for r in records)
        assert sum(outcomes.values()) == len(records)
        assert outcomes['completed'] <= accepted <= len(records)
        duration = phase.get('window', {}).get('duration_seconds', 0.)
        drain = phase.get('window', {}).get('drain_seconds', 0.)
        groups = [('SYSTEM', '__ALL__', records)]
        groups += [('TASK', k, [r for r in records if r['task'] == k])
                   for k in sorted({r['task'] for r in records})]
        groups += [('WORKER', k, [r for r in records if r['model'] == k])
                   for k in sorted(raw['worker_identities'])]
        for scope, identity, rr in groups:
            done = [r for r in rr if r['status'] == 'completed']
            within = sum(r['completion_time_ns'] <= end for r in done)
            accounting.append({'run_id': rid, 'scope': scope, 'identity': identity,
                               'offered': len(rr),
                               'accepted_host_send_start': sum(r.get('actual_submit_time_ns') is not None for r in rr),
                               'accepted_by_window_end': sum(r.get('actual_submit_time_ns') is not None and r['actual_submit_time_ns'] <= end for r in rr),
                               'completed': len(done), 'completed_within_window': within,
                               'status_counts': dict(Counter(r['status'] for r in rr)),
                               'window_seconds': duration, 'drain_seconds': drain,
                               'within_window_completed_QPS': within / duration if duration else None,
                               'cohort_completed_QPS_including_actual_drain': len(done) / (duration + drain) if duration + drain else None,
                               'completed_count_divided_by_window_QPS': len(done) / duration if duration else None,
                               'rate_interpretation': 'Observed workload completion, never a precise service capacity estimate.',
                               'worker_busy_interpretation': 'Completed-request handler wall time; lower bound if unfinished requests, distinct from NVML GPU utilization.'})
        for task in sorted({r['task'] for r in records}):
            tr = [r for r in records if r['task'] == task]
            sent = sum(r.get('actual_submit_time_ns') is not None for r in tr)
            completed = sum(r['status'] == 'completed' for r in tr)
            for worker in sorted({r['model'] for r in tr}):
                wr = [r for r in tr if r['model'] == worker]
                ws = sum(r.get('actual_submit_time_ns') is not None for r in wr)
                wc = sum(r['status'] == 'completed' for r in wr)
                routing.append({'run_id': rid, 'task': task, 'worker': worker,
                                'model': raw['replica_to_family'][worker],
                                'offered': len(wr), 'sent': ws, 'completed': wc,
                                'offered_fraction': len(wr) / len(tr),
                                'sent_fraction': ws / sent if sent else None,
                                'completed_fraction': wc / completed if completed else None,
                                'is_frozen_CPU_legal_control': config['operationally_legal']})
        if slot['stage'] == 'HIGH':
            catalog = read(case['catalog']['path'])
            target_done = [r for r in records if r['task'] == case['target_task'] and r['status'] == 'completed']
            for role in ['generalist', 'specialist']:
                rr = [r for r in target_done if catalog['models'][raw['replica_to_family'][r['model']]]['role'] == role]
                fraction = len(rr) / len(target_done) if target_done else None
                latency = statistics.mean(r['e2e_latency_ms'] for r in rr) if rr else None
                decomposition.append({'run_id': rid, 'config_id': slot['config_id'], 'target_task': case['target_task'],
                                      'repetition': slot['repetition'], 'role': role,
                                      'completed': len(rr), 'completed_fraction': fraction,
                                      'mean_E2E_ms': latency,
                                      'mean_execution_service_ms': statistics.mean(r['worker_latency_ms'] for r in rr) if rr else None,
                                      'mean_queue_ms': statistics.mean(r['worker_queue_wait_ms'] for r in rr) if rr else None,
                                      'contribution_to_target_mean_E2E_ms': fraction * latency if rr else None,
                                      'censored_by_incomplete_requests': len(target_done) != sum(r['task'] == case['target_task'] for r in records),
                                      'interpretation': 'Observed conditional means and weighted identity, not an interference or capacity model.'})
        measured.append({'run_id': rid, 'ref': ref(path), 'status': raw['status'],
                         'offered': len(records), 'accepted_host_send_start': accepted,
                         'accepted_by_window_end': sent_by_end,
                         'unsubmitted_at_run_end': len(records) - accepted,
                         'outcomes': dict(outcomes), 'measurement_seconds': phase.get('window', {}).get('duration_seconds'),
                         'release_verified': row['release_verified'],
                         'system': row['system_stability'],
                         'worker_nonsteady': row['worker_nonsteady'],
                         'foreign_PID_samples': sum(bool(x.get('foreign_pids')) for x in raw['telemetry']['samples'])})
        draws[slot['block_id'], slot['repetition']].append((rid, signature(raw)))
        if any(a['label'] == 'PROPOSED' for a in config['aliases']):
            old_path = case['historical_raw_files'][slot['repetition'] - 1]['path']
            old = read(old_path)
            # Historical services are diagnostic references, never paired controls.
            old_records = old['phases']['measurement']['request_records']
            for p in paths:
                same = [r for r in old_records if r['task'] == p['task'] and
                        old['replica_to_family'][r['model']] == p['model'] and
                        r['status'] == 'completed']
                old_service = statistics.mean(r['worker_latency_ms'] for r in same) if same else None
                drift.append({**{k: p[k] for k in ['run_id', 'repetition', 'task', 'model', 'mean_execution_service_ms', 'execution_service_vs_s0']},
                              'historical_raw': ref(old_path),
                              'same_scheduled_samples': signature(raw) == signature(old),
                              'same_task_family_assignment':
                              [(r['arrival_index'], r['task'], raw['replica_to_family'][r['model']]) for r in records] ==
                              [(r['arrival_index'], r['task'], old['replica_to_family'][r['model']]) for r in old_records],
                              'historical_mean_execution_service_ms': old_service,
                              'new_vs_historical_execution_ratio': p['mean_execution_service_ms'] / old_service
                              if old_service and p['mean_execution_service_ms'] is not None else None,
                              'interpretation': 'Descriptive runtime drift; not a new empirical capacity bound or a policy effect.'})
    assert len(pids) == len(set(pids))
    assert set(r['run_id'] for r in measured) == set(state['completed_runs'])
    skipped = {s['run_id'] for s in state['skipped_runs']}
    assert set(slots) == set(state['completed_runs']) | skipped
    draw_checks = []
    for (block, repetition), values in draws.items():
        complete = [(rid, sig) for rid, sig in values if
                    next(r for r in measured if r['run_id'] == rid)['measurement_seconds'] == 120.]
        matches = all(sig == complete[0][1] for _, sig in complete) if complete else None
        draw_checks.append({'block': block, 'repetition': repetition,
                            'full_window_runs': [rid for rid, _ in complete],
                            'same_all_task_sample_schedule': matches})
    save('completion_audit.json', {'at_utc': now(), 'measured': measured,
                                  'skipped_runs': state['skipped_runs'],
                                  'same_seed_sample_checks': draw_checks,
                                  'fresh_unique_worker_PIDs': len(pids),
                                  'protected_files_verified': len(protected),
                                  'all_slots_accounted_for': True,
                                  'analysis_only_source': ref(ROOT / 'module19/completion_audit.py')})
    csvsave('historical_runtime_drift.csv', drift)
    csvsave('request_accounting.csv', accounting)
    csvsave('actual_task_worker_routing.csv', routing)
    csvsave('high_route_latency_decomposition.csv', decomposition)
    print('COMPLETION_AUDIT', len(measured), 'new runs', len(pids), 'unique PIDs',
          'sample matches', all(x['same_all_task_sample_schedule'] is not False for x in draw_checks))


if __name__ == '__main__':
    main()
