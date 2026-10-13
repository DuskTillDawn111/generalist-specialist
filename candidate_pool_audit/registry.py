from pathlib import Path
from joint_deploy.measurement.deployment_runtime import default_registry

ROOT = Path(__file__).resolve().parents[1]


def registry():
    result = default_registry()
    result['phi4mm'] = result.pop('phi4')
    result['whisper-small'] = result.pop('whisper')
    result['minicpm'] = {'python': ROOT / 'envs/minicpm_env/bin/python',
                         'script': ROOT / 'joint_deploy/minicpm_worker.py'}
    for name in ('convnext_tiny', 'roberta_sst2', 'paddleocr_vl', 'whisper_turbo', 'distil_whisper'):
        result[name] = {'python': ROOT / 'envs/specialist_env/bin/python',
                        'script': ROOT / f'candidate_pool_audit/worker_{name}.py'}
    return result
