"""Describe verified long resident-idle windows without qualifying a predictor."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fit_activation_native import load_bundle
from gsdeploy.energy import integrate_power


def summarize(directory):
    bundle = load_bundle(directory)
    raw, plan = bundle['raw'], bundle['plan']
    if (raw['status'] != 'completed' or raw['kind'] != 'measured'
            or raw['execution_context_sha256'] != plan['execution_context_sha256']
            or raw['resident_aliases'] != plan['aliases']
            or any(raw.get(k) for k in ('errors', 'runtime_errors', 'telemetry_errors'))
            or any(s.get('foreign_pids') for s in raw['board_samples'])
            or any(s['forced'] or s['exit_code'] != 0 or not s['bye_received']
                   for s in raw['shutdown'].values())):
        raise ValueError('Complete clean native idle capture required')
    if len(raw['batches']) != len(plan['jobs']):
        raise ValueError('Idle batch roster changed')
    windows = []
    for job, batch in zip(plan['jobs'], raw['batches']):
        if job['id'] != batch['id']:
            raise ValueError('Idle job identity changed')
        if job['requests']:
            continue
        start, end = batch['start_ns'], batch['arrival_end_ns']
        if (batch['request_ids'] or end - start != round(job['duration_ms'] * 1e6)
                or end - start < 30_000_000_000):
            raise ValueError('Unchanged declared long idle interval required')
        parts = []
        for label, lo, hi in [('full', start, end), ('first_10s', start, start + 10_000_000_000),
                              ('last_10s', end - 10_000_000_000, end)]:
            joules = integrate_power(raw['board_samples'], lo, hi, max_gap_seconds=.25)
            if joules is None:
                raise ValueError('Uncovered idle power interval')
            samples = [s for s in raw['board_samples'] if lo <= s['time_ns'] <= hi]
            parts.append(dict(part=label, duration_seconds=(hi-lo)/1e9,
                              joules=joules, mean_watts=joules/((hi-lo)/1e9),
                              minimum_watts=min(s['watts'] for s in samples),
                              maximum_watts=max(s['watts'] for s in samples),
                              sample_count=len(samples),
                              peak_owned_mib=max(s['owned_used_mib'] for s in samples)))
        windows.append(dict(id=job['id'], parts=parts))
    if not windows:
        raise ValueError('No long idle windows')
    return dict(source=bundle['source'], execution_context_sha256=raw['execution_context_sha256'],
                resident_aliases=raw['resident_aliases'], windows=windows,
                scope='resident workers after serial priming; observed board energy, not an unloaded baseline',
                energy_prediction_qualified=False, utility_qualified=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = dict(schema='gsdeploy.resident_idle_description.v1',
                  runs=[summarize(d.resolve()) for d in args.capture],
                  analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as out:
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
