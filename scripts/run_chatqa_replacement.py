"""Finite download -> verify -> native QA calibration pipeline; no system jobs."""
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


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = {str(p): digest(p) for pattern in ['gsdeploy/**/*.py', 'configs/*repaired.json',
                'scripts/prepare_chatqa_calibration.py', 'scripts/run_chatqa_replacement.py',
                'artifacts/qa-replacement-20261018/download-study.json',
                'artifacts/qa-replacement-20261018/metadata/*-hub.json']
               for p in ROOT.glob(pattern)}
    save(output / 'frozen-sources.json', sources)
    state = {'status': 'DOWNLOADING_WEIGHTS', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
             'pool_admitted': False, 'scope': '64_sample_QA_calibration_only', 'gpu_started': False}
    state_path = output / 'state.json'
    save(state_path, state)
    interpreter = '/root/autodl-tmp/gsdeploy-assets/envs/compact-standard/bin/python'
    environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    def command(argv, log):
        with (output / log).open('x') as handle:
            subprocess.run(argv, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
    def check_sources():
        for path, expected in sources.items():
            if digest(path) != expected:
                raise ValueError('Source changed while waiting for QA checkpoint: ' + path)
    try:
        command([interpreter, '-m', 'gsdeploy.profiling.resources', '--study',
                 'artifacts/qa-replacement-20261018/download-study.json', '--metadata-root',
                 'artifacts/qa-replacement-20261018/metadata', '--assets', '/root/autodl-tmp/gsdeploy-assets',
                 '--output', 'artifacts/qa-replacement-20261018/resources'], 'download.log')
        inventory = json.loads((ROOT / 'artifacts/qa-replacement-20261018/resources/resource-inventory.json').read_text())
        if inventory['status'] != 'COMPLETE':
            raise RuntimeError('Pinned QA weights are not completely downloaded and hash-verified')
        check_sources()
        state['status'] = 'PREPARING_CALIBRATION'
        save(state_path, state)
        packet = output / 'calibration-packet'
        command([interpreter, str(ROOT / 'scripts/prepare_chatqa_calibration.py'), '--output', str(packet)], 'prepare.log')
        check_sources()
        state.update(status='GPU_CALIBRATION_RUNNING', gpu_execution_requested=True, gpu_started=None)
        save(state_path, state)
        results = output / 'gpu-results'
        command([interpreter, '-m', 'gsdeploy.profiling.diagnostics', '--manifest',
                 str(packet / 'experiments/experiment-manifest.json'), '--output', str(results), '--execute'], 'gpu-calibration.log')
        queue = json.loads((results / 'diagnostic-queue-state.json').read_text())
        result = queue['results'].get('qa-chatqa-nf4-native-calibration', {})
        if queue['status'] != 'COMPLETE_DIAGNOSTICS_NOT_ADMITTED' or not result.get('collection_complete'):
            raise RuntimeError('QA calibration collection did not complete; inspect raw logs')
        from gsdeploy.profiling.admission import quality_gate
        policy = json.loads((ROOT / 'configs/compact_protocol_repaired.json').read_text())
        gate = quality_gate(policy, 'text', result['quality_summary'])
        state.update(status='CALIBRATION_COMPLETE_NOT_ADMITTED' if gate['passed'] else 'CALIBRATION_REJECTED', gpu_started=True,
                     quality_summary=result['quality_summary'], calibration_gate=gate,
                     results=str(results / 'diagnostic-queue-state.json'))
    except BaseException as error:
        state.update(status='FAILED', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save(state_path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('The finite GPU calibration pipeline requires --execute')
    run(args.output)
