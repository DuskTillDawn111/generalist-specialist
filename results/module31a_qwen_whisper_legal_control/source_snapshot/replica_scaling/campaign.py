"""Bounded frozen experiment schedule; no optimization or automatic retries."""
import json
import os
import subprocess
import time
from .common import ROOT, OUT, frozen_plan, read, write, reference, verify_refs


def main():
    plan = frozen_plan()
    before = read(OUT / 'source_hashes_before.json')
    for refs in (before['protected_files'], before['inputs'], *before['models'].values()):
        verify_refs(refs)
    ledger_path = OUT / 'campaign_ledger.json'
    ledger = read(ledger_path) if ledger_path.exists() else dict(started_unix_ns=time.time_ns(), slots=[], status='running')
    done = {s['run_id'] for s in ledger['slots']}
    stopped = {tuple(s['condition']) for s in ledger['slots'] if s.get('stop_condition')}
    env = dict(os.environ, **plan['environment_variables'])
    for index, slot in enumerate(plan['schedule'], 1):
        if slot['run_id'] in done:
            continue
        condition = (slot['model_id'], slot['replica_count'], slot['load_level'])
        row = dict(**slot, condition=list(condition))
        if condition in stopped:
            row.update(execution_status='skipped_after_condition_stop', physical_run=False)
        else:
            output = OUT / 'raw_runs' / (slot['run_id'] + '.json')
            log = OUT / 'raw_runs' / (slot['run_id'] + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            if log.exists() and not output.exists():
                raise RuntimeError('Prior unfinished attempt requires explicit review; no automatic retry: ' + slot['run_id'])
            print(f'[{index}/54] {slot["run_id"]} load={slot["offered_load_rps"]:.6f}', flush=True)
            if not output.exists():
                with log.open('w') as stream:
                    outcome = subprocess.run([str(ROOT / 'envs/data_env/bin/python'), '-m', 'replica_scaling.runtime', '--run-id', slot['run_id']],
                                             cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                row['returncode'] = outcome.returncode
            if not output.exists():
                row.update(execution_status='uncertain_interruption', physical_run=True, stop_condition=True, log=reference(log))
                stopped.add(condition)
                ledger['slots'].append(row)
                ledger['status'] = 'halted_unconfirmed_cleanup'
                write(ledger_path, ledger)
                raise RuntimeError('No raw outcome/cleanup evidence; halt campaign without retry')
            else:
                raw = read(output)
                status = raw['execution_status']
                stop_condition = status != 'completed'
                row.update(execution_status=status, physical_run=True, valid_full_repetition=raw['valid_full_repetition'],
                    stability=raw.get('stability', {}).get('stability_state_v2'), output=reference(output),
                    log=reference(log), stop_condition=stop_condition)
                if stop_condition:
                    stopped.add(condition)
                metrics = raw.get('metrics') or {}
                print(json.dumps(dict(run=slot['run_id'], execution=status, valid=raw['valid_full_repetition'],
                    stability=row['stability'], throughput=metrics.get('completed_throughput_rps'),
                    completion_ratio=metrics.get('completion_ratio'), mean_e2e_ms=metrics.get('mean_e2e_ms'))), flush=True)
                if not raw['gpu_release']['verified']:
                    ledger['slots'].append(row)
                    ledger['status'] = 'halted_gpu_release_failure'
                    write(ledger_path, ledger)
                    raise RuntimeError('GPU release failed; halt without starting another condition')
        ledger['slots'].append(row)
        write(ledger_path, ledger)
    ledger.update(status='schedule_finished', finished_unix_ns=time.time_ns(),
                  physical_runs=sum(s['physical_run'] for s in ledger['slots']),
                  valid_full_runs=sum(s.get('valid_full_repetition', False) for s in ledger['slots']))
    write(ledger_path, ledger)
    print(json.dumps(dict(status=ledger['status'], physical_runs=ledger['physical_runs'], valid_full_runs=ledger['valid_full_runs'])), flush=True)

if __name__ == '__main__':
    main()
