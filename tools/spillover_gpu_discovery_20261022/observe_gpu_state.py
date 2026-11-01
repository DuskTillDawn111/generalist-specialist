"""Supplemental read-only GPU telemetry; never used for utility/admission."""
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/static-spillover-study-20261022'
OUTPUT = BASE / 'text26-supplemental-gpu-observation-v1'
FIELDS = ['uuid', 'pstate', 'utilization.gpu', 'utilization.memory', 'memory.used',
          'power.draw', 'temperature.gpu', 'clocks.sm', 'clocks.mem', 'clocks_event_reasons.active']


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    return dict(command=command, exit_code=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())


def receipt(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    OUTPUT.mkdir(exist_ok=False)
    source = receipt(Path(__file__))
    declaration = dict(schema='gsdeploy.supplemental_gpu_observation.v1',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source=source,
        interval_seconds=10, gpu_fields=FIELDS,
        measurement_inputs_changed=False, stability_changed=False, used_for_utility=False,
        scope='Partial temporal coverage starting during second text26 condition; no retrospective telemetry for first text condition or ASR.',
        commands_are_read_only=True, overhead_limit='Additional nvidia-smi query every10s; nonzero host overhead, no GPU computation launched.')
    (OUTPUT/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    count = 0
    with (OUTPUT/'samples.jsonl').open('x') as stream:
        while True:
            begin_mono = time.perf_counter_ns()
            sample = dict(index=count, begin_monotonic_ns=begin_mono, begin_unix_ns=time.time_ns(),
                gpu=run(['nvidia-smi', '--query-gpu='+','.join(FIELDS), '--format=csv,noheader,nounits']),
                compute_apps=run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits']))
            queues = {}
            for relative in ['text26-development-v1/runs/queue-progress.json', 'text26-development-v1/remaining-queue-progress.json']:
                path = BASE/relative
                if path.is_file():
                    state = json.loads(path.read_text())
                    queues[relative] = dict(status=state['status'], current=state.get('current'), jobs_completed=len(state.get('jobs', [])))
            sample.update(queues=queues, end_monotonic_ns=time.perf_counter_ns())
            sample['query_wall_seconds'] = (sample['end_monotonic_ns']-begin_mono)/1e9
            stream.write(json.dumps(sample)+'\n')
            stream.flush()
            count += 1
            continuation = queues.get('text26-development-v1/remaining-queue-progress.json', {})
            if continuation.get('status') in {'ALL_PREDECLARED_ATTEMPTS_COLLECTED', 'STOPPED_INFRASTRUCTURE_ERROR'}:
                break
            time.sleep(max(0, 10-sample['query_wall_seconds']))
    (OUTPUT/'manifest.json').write_text(json.dumps(dict(schema='gsdeploy.supplemental_gpu_observation_completion.v1',
        samples=count, source=source, artifacts=[receipt(OUTPUT/'declaration.json'), receipt(OUTPUT/'samples.jsonl')],
        used_for_utility=False, production_math_changed=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
