"""Index complete frozen calibration observations without granting pool admission."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsdeploy.backends.compact import fingerprint
from gsdeploy.profiling.admission import quality_gate
from gsdeploy.profiling.prepare import digest
from gsdeploy.profiling.runner import save


def summarize(output):
    study = json.loads((ROOT / 'configs/study_repaired.json').read_text())
    cohort_path = ROOT / study['calibration_preparation']['manifest']
    if digest(cohort_path) != study['calibration_preparation']['manifest_sha256']:
        raise ValueError('Changed current calibration manifest')
    cohorts = {}
    for task, entry in json.loads(cohort_path.read_text())['tasks'].items():
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Changed calibration inputs')
        rows = [json.loads(line) for line in Path(entry['path']).read_text().splitlines()]
        cohorts[task] = fingerprint(sorted(rows, key=lambda row: row['id']))
    groups, skipped = {}, []
    queues = list((ROOT / 'artifacts/compact-memory-results-20261017').glob('*/diagnostic-queue-state.json'))
    queues += list((ROOT / 'artifacts/qa-adaptation-20261018').glob('*/gpu-results/diagnostic-queue-state.json'))
    queues += list((ROOT / 'artifacts/generalist-replacement-20261018').glob('*/gpu-results/diagnostic-queue-state.json'))
    queues += list((ROOT / 'artifacts/minicpm45-adapter-stage-20261018').glob('*/gpu-results/diagnostic-queue-state.json'))
    for state_path in queues:
        state = json.loads(state_path.read_text())
        if not state.get('manifest'):
            skipped.append({'state': str(state_path), 'reason': 'legacy_schema_without_frozen_job_manifest'})
            continue
        manifest_path = Path(state['manifest'])
        if digest(manifest_path) != state['manifest_sha256']:
            raise ValueError('Changed diagnostic experiment manifest')
        manifest = json.loads(manifest_path.read_text())
        entries = {job['id']: job for job in manifest['jobs']}
        for name, observation in state.get('results', {}).items():
            if not observation.get('collection_complete') or not observation.get('quality_summary'):
                continue
            entry = entries[name]
            if digest(entry['path']) != entry['sha256']:
                raise ValueError('Changed calibration job')
            job = json.loads(Path(entry['path']).read_text())
            task = job['samples'][0]['task']
            if fingerprint(sorted(job['samples'], key=lambda row: row['id'])) != cohorts.get(task):
                continue
            spec_path = Path(job['spec'])
            if digest(spec_path) != manifest['workers'][name]['spec_sha256']:
                raise ValueError('Changed calibration worker')
            spec = json.loads(spec_path.read_text())
            policy = json.loads(Path(spec['resources']['protocol']).read_text())
            candidate, execution = spec['candidate'], spec['execution']
            identity = fingerprint({'model': candidate['model'], 'revision': candidate['revision'],
                                    'execution': execution, 'adapter_sha256': candidate.get('adapter_sha256')})
            group = groups.setdefault(identity, {'model': candidate['model'], 'revision': candidate['revision'],
                'execution': execution, 'adapter_sha256': candidate.get('adapter_sha256'), 'tasks': {},
                'pool_admitted': False})
            group['tasks'].setdefault(task, []).append({'case': name, 'summary': observation['quality_summary'],
                'gate': quality_gate(policy, task, observation['quality_summary']),
                'state': str(state_path), 'manifest': str(manifest_path),
                'worker': str(spec_path), 'worker_sha256': digest(spec_path),
                'policy_sha256': fingerprint(policy), 'code_identity': fingerprint(manifest['code_hashes'])})
    for group in groups.values():
        group['tasks_with_passing_observations'] = [task for task, rows in group['tasks'].items()
                                                   if any(row['gate']['passed'] for row in rows)]
        group['four_task_calibration_observations_available'] = len(group['tasks_with_passing_observations']) == 4
    result = {'schema': 'gsdeploy.calibration_candidate_index.v1', 'groups': groups, 'skipped': skipped,
              'pool_admitted': False, 'system_experiments_complete': False,
              'scope': 'Historical calibration index for configuration assembly. Different task policies/source versions '
                       'are shown explicitly; selecting variants requires new frozen full calibration, independent '
                       'quality and all original paired cost gates. This index is never admission evidence.'}
    save(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = summarize(args.output)
    print(json.dumps({'groups': len(data['groups']), 'pool_admitted': False}))
