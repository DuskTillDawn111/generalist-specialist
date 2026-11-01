"""Separate long-idle board-power calibration; no task-utility observation."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path

from tools.queue_coupled_utility_20261023 import controlled_calibration as calibration

SCHEMA = 'gsdeploy.coupled_idle_power_calibration.v1'


def execute(controlled_packet_path, output, *, execute=False):
    if not execute:
        raise ValueError('Explicit execute=True required')
    original_packet = json.loads(Path(controlled_packet_path).read_text())
    calibration.validate_packet(original_packet)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    packet = dict(schema=SCHEMA, controlled_packet=calibration.receipt(controlled_packet_path),
                  source=calibration.receipt(__file__), phases=[dict(seconds=60., role='idle_after_priming'),
                                                               dict(seconds=60., role='continued_idle')],
                  calibration_phase_label=321, arrival_seed=None, requests=0, U=None, no_retries=True,
                  claim='POWER_STATE_DURATION_CALIBRATION_ONLY_NOT_600S_UTILITY',
                  declared_before_measurement=True)
    calibration.save(output / 'experiment-packet.json', packet)
    runner = calibration.frozen_runtime()
    package = runner.__package__.split('.runtime')[0]
    static = importlib.import_module(package + '.static_study')
    config_module = importlib.import_module(package + '.config')
    base = json.loads(Path(original_packet['base_packet']['path']).read_text())
    job = copy.deepcopy(original_packet['binding_template'])
    try:
        config, manifest, workers = static.bind_native_job(base, job, output / 'native-binding')
        config['catalog']['performance_mode'] = 'task_replica_proxy'
        config['catalog']['memory_mode'] = 'standalone_proxy'
        raw = calibration.collect(config_module.SystemConfig(**config), original_packet['input_pool']['path'],
                                  original_packet['policy_contract_for_preflight_only'], [], backend='native',
                                  workers=workers, admission_path=str(manifest), idle_seconds=60.)
    except (Exception, KeyboardInterrupt) as exc:
        raw = dict(status='failed', errors=[f'{type(exc).__name__}: {exc}'], backend='native')
    raw.update(schema=SCHEMA, source_packet_digest=calibration.digest(packet), packet=calibration.receipt(output / 'experiment-packet.json'),
               claim=packet['claim'], U=None, calibration_phase_label=321, arrival_seed=None,
               original_controlled_calibration_untouched=True)
    calibration.save(output / 'idle-power.json', raw)
    return raw


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controlled-packet', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    raw = execute(a.controlled_packet, a.output, execute=a.execute)
    print(json.dumps(dict(status=raw['status'], errors=raw.get('errors'), output=str(a.output / 'idle-power.json'))))
