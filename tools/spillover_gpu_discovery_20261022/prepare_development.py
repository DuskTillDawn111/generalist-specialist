"""Declare new-load native discovery pairs using unchanged frozen v5 runtime."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/root/autodl-tmp/generalist-specialist-project')
OLD = ROOT / 'artifacts/static-study-gpu-20261022'
OUT = ROOT / 'artifacts/static-spillover-study-20261022'
SOURCE = OLD / 'execution-source-v5'
sys.path.insert(0, str(SOURCE))
from gsdeploy.config import SystemConfig
from gsdeploy.static_study import _bindings, _file, _job, validate_packet
from gsdeploy.quality import quality_digest

def main():
    destination = OUT / 'development-asr18-v1'
    destination.mkdir(parents=True, exist_ok=False)
    original = json.loads((OLD / 'execution-packet-v2.json').read_text())
    raw = copy.deepcopy(original['config'])
    raw.update(workload={'text': .05, 'image': .05, 'ocr': .05, 'asr': 1.8},
               horizon_seconds=600, random_seed=51, route_grid=8)
    raw['catalog']['performance_mode'] = 'task_replica_proxy'
    raw['catalog']['memory_mode'] = 'standalone_proxy'
    raw['catalog']['measurement_context'] = {**raw['catalog'].get('measurement_context', {}),
        'gpu_uuid': 'GPU-8a262fe0-598c-46b7-c953-41d9da84a47e'}
    config = SystemConfig(**raw)
    protocol = {
        'schema': 'gsdeploy.spillover_discovery_development.v1',
        'purpose': 'Exploratory actual measurements, not independent final confirmation or a free-search optimum',
        'workload': raw['workload'], 'horizon_seconds': 600, 'seed': 51,
        'shares_to_asr_specialist': [1., .75],
        'deployment': {'asr': 2, 'generalist_2': 1, 'ocr': 1},
        'other_routes': {'text': 'generalist_2', 'image': 'generalist_2', 'ocr': 'ocr'},
        'all_attempts_preserved': True, 'old_validation_transferred': False,
        'independent_future_validation_seeds_reserved': [61, 62],
        'independent_future_confirmation_seeds_reserved': [71, 72, 73],
        'fitting_policy': 'Discovery may inform a separately declared new predictor; never reuse as independent validation',
        'runtime_source': str(SOURCE),
    }
    for name, value in [('system-config.json', raw), ('discovery-protocol.json', protocol)]:
        (destination / name).write_text(json.dumps(value, indent=2) + '\n')
    bindings = _bindings(original['inputs'], SOURCE)
    bindings += [_file(destination / name) for name in ['system-config.json', 'discovery-protocol.json']]
    jobs = []
    for share in protocol['shares_to_asr_specialist']:
        routing = {t: {m: raw['workload'][t]} for t, m in protocol['other_routes'].items()}
        routing['asr'] = {'asr': 1.8 * share}
        if share < 1:
            routing['asr']['generalist_2'] = 1.8 * (1-share)
        plan = {'deployment': protocol['deployment'], 'routing': routing}
        constraints = config.validate_plan(**plan)
        assert constraints['constraints_valid']
        label = 'asr18_n2_specialist_share_' + str(share).replace('.', 'p')
        job = _job(config, original['workers'], plan, label, 'calibration', 51, bindings, 30)
        job['purpose'] = 'new_load_exploratory_development_not_final_confirmation'
        job['proxy_constraints'] = constraints
        jobs.append(job)
    packet = {
        'schema': 'gsdeploy.static_experiments.v1', 'status': 'PREDECLARED_NEW_LOAD_DEVELOPMENT',
        'inputs': original['inputs'], 'config': raw, 'workers': original['workers'],
        'bindings': bindings, 'frozen_files_sha256': quality_digest(bindings),
        'gpu_started': False, 'jobs': jobs, 'discovery_protocol': str(destination / 'discovery-protocol.json'),
        'execution': {'policy': 'Serial native collection using unchanged frozen v5 source'},
    }
    assert validate_packet(packet)['valid']
    (destination / 'experiment-packet.json').write_text(json.dumps(packet, indent=2) + '\n')
    (destination / 'manifest.json').write_text(json.dumps({'files': {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file()},
        'gpu_started': False}, indent=2) + '\n')
    print(json.dumps({'packet': str(destination / 'experiment-packet.json'), 'jobs': len(jobs),
                      'validated': validate_packet(packet), 'gpu_started': False}))

if __name__ == '__main__':
    main()
