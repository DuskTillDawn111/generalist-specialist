"""Serial frozen GPU jobs, with phase barriers and durable logs/state.

This controller never selects new models or generates a system experiment matrix.
Smoke failures stop before quality; quality failures stop before capacity/cost.
Already completed receipts are reused only for the identical frozen manifest.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsdeploy.profiling.prepare import digest
from gsdeploy.profiling.runner import save
from gsdeploy.profiling.locking import gpu_lease

PHASES = ('smoke', 'memory', 'quality', 'profile', 'matched_cost')


def now():
    return datetime.now(timezone.utc).isoformat()


def successful_receipt(path, manifest_hash, job_hash):
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text())
    return (receipt.get('status') == 'COMPLETE'
            and receipt.get('manifest_sha256') == manifest_hash
            and receipt.get('job_sha256') == job_hash)


def run_queue(manifest_path, output, through):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    fingerprint = digest(manifest_path)
    state = {'status': 'RUNNING', 'started_utc': now(), 'pid': os.getpid(),
             'manifest': str(manifest_path), 'manifest_sha256': fingerprint,
             'controller_sha256': digest(__file__), 'through': through, 'current_job': None,
             'completed': [], 'failed': [], 'blocked': [], 'quality_gates': {},
             'runtime_catalog': None, 'utility': None}

    def update():
        state['updated_utc'] = now()
        save(output / 'queue-state.json', state)

    # One controller for this results tree; native collectors independently
    # reject an occupied GPU before loading any model.
    with gpu_lease():
        update()
        try:
            for path, expected in {**manifest['sources'], **manifest['code_hashes']}.items():
                if digest(path) != expected:
                    raise ValueError('Frozen experiment source changed: ' + path)
            for phase in PHASES[:PHASES.index(through) + 1]:
                if not any(e['phase'] == phase for e in manifest['jobs']):
                    continue
                state['phase'] = phase
                phase_failures = []
                for entry in (e for e in manifest['jobs'] if e['phase'] == phase):
                    job_id = entry['id']
                    job = json.loads(Path(entry['path']).read_text())
                    missing = [r for r in job['requires'] if r not in state['completed']]
                    if missing:
                        state['blocked'].append({'job': job_id, 'prerequisites': missing})
                        phase_failures.append(job_id)
                        update()
                        continue
                    state['current_job'] = job_id
                    update()
                    destination = output / job_id
                    if successful_receipt(destination / 'receipt.json', fingerprint, entry['sha256']):
                        print(now(), 'REUSE', job_id, flush=True)
                    elif destination.exists():
                        state['failed'].append({'job': job_id, 'reason': 'Existing failed/partial evidence retained; use a new run directory for retry'})
                        phase_failures.append(job_id)
                        update()
                        continue
                    else:
                        print(now(), 'START', job_id, flush=True)
                        with (output / (job_id + '.log')).open('x') as log:
                            command = [sys.executable, '-m', 'gsdeploy.profiling.compact_experiments',
                                       'run', '--manifest', str(manifest_path), '--job', job_id,
                                       '--output', str(destination), '--execute']
                            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                     env={**os.environ, 'OMP_NUM_THREADS': '1',
                                                          'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'})
                        if process.returncode or not successful_receipt(destination / 'receipt.json', fingerprint, entry['sha256']):
                            state['failed'].append({'job': job_id, 'returncode': process.returncode})
                            phase_failures.append(job_id)
                            print(now(), 'FAILED', job_id, flush=True)
                            update()
                            continue
                    state['completed'].append(job_id)
                    if phase == 'quality':
                        quality = json.loads((destination / 'quality.json').read_text())
                        state['quality_gates'][job['slot']] = {t: q['gate'] for t, q in quality.items()}
                    print(now(), 'COMPLETE', job_id, flush=True)
                    update()
                # n2/n3 are conditional configurations. An infeasible replica
                # count remains failed evidence, but must not block the valid
                # n1 matched-cost experiment. A failed n1 still blocks it.
                required_failures = [job_id for job_id in phase_failures
                                     if phase != 'profile' or next(
                                         json.loads(Path(e['path']).read_text()).get('replicas', 1)
                                         for e in manifest['jobs'] if e['id'] == job_id) == 1]
                if required_failures:
                    state['status'] = 'NEEDS_' + phase.upper() + '_REPAIR'
                    break
                if phase == 'smoke' or (phase == 'memory' and manifest.get('memory_validation')):
                    from gsdeploy.profiling.memory_gate import screen_pool_memory
                    study = json.loads(Path(manifest['study']).read_text())
                    profiles = {slot: json.loads((output / f'{phase}-{slot}-n1' / 'profile.json').read_text())
                                for slot in manifest['workers']}
                    memory = screen_pool_memory(study, profiles)
                    memory.update(manifest_sha256=fingerprint, profile_sha256={slot: digest(
                        output / f'{phase}-{slot}-n1' / 'profile.json') for slot in manifest['workers']})
                    if phase == 'memory':
                        memory['scope'] = 'complete_calibration_memory_screen_only'
                        memory['remaining'] = 'Independent quality, matched-load measurements and whole-plan feasibility remain required.'
                    save(output / ('smoke-memory-screen.json' if phase == 'smoke' else 'memory-screen.json'), memory)
                    state['memory_screen'] = memory['status']
                    if not memory['passed']:
                        state['status'] = memory['status']
                        state['pool_admitted'] = False
                        break
                    if phase == 'memory':
                        from gsdeploy.profiling.compact_experiments import verify_memory_screen
                        verify_memory_screen(manifest_path, manifest, output)
                        from gsdeploy.profiling.compact_experiments import calibration_quality_screen
                        calibration = calibration_quality_screen(manifest_path, manifest, output)
                        save(output / 'calibration-quality-screen.json', calibration)
                        state['calibration_quality_screen'] = calibration['status']
                        if not calibration['passed']:
                            state.update(status=calibration['status'], pool_admitted=False)
                            break
                if phase == 'quality' and not all(gate['passed'] for tasks in state['quality_gates'].values() for gate in tasks.values()):
                    state['status'] = 'POOL_REJECTED_BY_ABSOLUTE_QUALITY_GATES'
                    break
                if phase == 'quality':
                    from gsdeploy.profiling.compact_experiments import assess
                    screening = assess(manifest_path, output, output / 'pool-screen.json')
                    study = json.loads(Path(manifest['study']).read_text())
                    expected_pairs = len(study['generalists']) * len(study['specialists'])
                    pairs = screening.get('comparisons', [])
                    if expected_pairs and (len(pairs) != expected_pairs or any(
                            not pair.get('quality') or not pair['quality']['passed'] for pair in pairs)):
                        state.update(status='POOL_REJECTED_BY_PAIRED_QUALITY_GATES', pool_admitted=False)
                        break
            else:
                state['status'] = 'COMPLETE_THROUGH_' + through.upper()
            if state['phase'] in {'quality', 'matched_cost'}:
                from gsdeploy.profiling.compact_experiments import assess
                try:
                    screening = assess(manifest_path, output, output / 'pool-screen.json')
                    state['pool_admitted'] = screening['all_requirements_passed']
                except Exception as error:
                    # Retain the collection/quality failure instead of masking
                    # it with a second failure while building a summary.
                    state.update(assessment_status='FAILED_EVIDENCE_VALIDATION',
                                 assessment_error=str(error), pool_admitted=False)
                    if state['status'].startswith('COMPLETE_'):
                        state['status'] = 'NEEDS_ASSESSMENT_REPAIR'
        except Exception as error:
            import traceback
            state.update(status='CONTROLLER_FAILED', error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            state['current_job'] = None
            state['finished_utc'] = now()
            update()
            print(now(), state['status'], flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--through', choices=PHASES, default='matched_cost')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('GPU execution requires --execute')
    state = run_queue(args.manifest, args.output, args.through)
    if not state['status'].startswith('COMPLETE_'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
