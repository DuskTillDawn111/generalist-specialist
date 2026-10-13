"""Continue unchanged serving runs after correcting the seed-only V3 label."""
import time
from .common import *
from .gpu import run
from .metrics import summarize


def main():
    frozen = read(OUT / 'frozen_benchmark_manifest.json')
    correction = read(OUT / 'benchmark_protocol_correction.json')
    verify(frozen['implementation'] + frozen['reused_implementation'] +
           [frozen['CPU'], frozen['input_protocol']])
    verify(correction['controller_implementation'])
    cases = {c['case_id']: c for c in frozen['cases']}
    configs = {c['config_id']: c for c in frozen['configurations']}
    excluded = set(correction['no_further_seed_only_configurations'])
    completed, stopped, skipped = [], {}, []
    abort = None
    for slot in frozen['slots']:
        ident = slot['config_id']
        config = configs[ident]
        case = cases[config['owner_case']]
        path = OUT / 'raw_runs' / (slot['run_id'] + '.json')
        if path.exists():
            raw = read(path)
        elif ident in excluded:
            skipped.append({'run_id': slot['run_id'],
                            'reason': 'V3 dry-run is initialization only; no further diagnostic GPU measurement substitutes for full baseline.'})
            continue
        elif ident in stopped or abort:
            skipped.append({'run_id': slot['run_id'], 'reason': stopped.get(ident, abort)})
            continue
        else:
            print('START', slot['run_id'], slot['configuration_role'], flush=True)
            raw = run(slot, case['historical_plan'])
        row, _, workers, _, _ = summarize(raw, config, case)
        completed.append(slot['run_id'])
        hard = [w['worker'] for w in workers if w['worker_stability'] == 'unstable']
        print('FINISH', slot['run_id'], 'valid', row['valid_session'],
              'system', row['system_stability'],
              'workers', [(w['model'], w['worker_stability']) for w in workers],
              'memory', row['memory_passed'], 'target_E2E',
              row['target_mean_E2E_ms'], row['target_P95_E2E_ms'],
              'completed/within', row['target_completed'],
              row['target_within_window_completed'], 'errors', row['errors'], flush=True)
        if row['system_stability'] == 'unstable' or hard or not row['valid_session'] or not row['memory_passed'] or not row['release_verified']:
            stopped[ident] = {'run_id': slot['run_id'], 'system_stability': row['system_stability'],
                             'worker_unstable': hard, 'valid_session': row['valid_session'],
                             'memory_passed': row['memory_passed'], 'errors': row['errors'],
                             'reason': 'First unstable/sustainedbacklog/runtime/memory failure; no retry, laterrep skipped.'}
        if not row['release_verified']:
            abort = {'run_id': slot['run_id'], 'reason': 'GPU release failure; abort remaining experiment.'}
        save('campaign_status.json', {'at_utc': now(), 'completed_runs': completed,
                                     'stopped_configurations': stopped, 'global_abort': abort,
                                     'skipped_runs': skipped, 'frozen_config_changed': False,
                                     'controller_correction': ref(OUT / 'benchmark_protocol_correction.json'),
                                     'no_outcome_retry': True})
    save('campaign_status.json', {'at_utc': now(), 'completed_runs': completed,
                                 'stopped_configurations': stopped, 'global_abort': abort,
                                 'skipped_runs': skipped, 'frozen_config_changed': False,
                                 'controller_correction': ref(OUT / 'benchmark_protocol_correction.json'),
                                 'no_outcome_retry': True})
    print('BENCHMARK_GPU_COMPLETE', 'runs', len(completed), 'stopped', stopped,
          'skipped', len(skipped), flush=True)


if __name__ == '__main__':
    main()
