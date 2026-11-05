"""Describe repeated serial precursor effects; do not fit or grant accuracy."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.activation_semantic_bands import band_layout
from experiments.analyze_activation_semantic import output_identity
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.native_training import digest, stage_layout, validate_capture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', action='append', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    runs = []
    rows = []
    contexts = set()
    for directory in args.capture:
        bundle = load_bundle(directory.resolve())
        contexts.add(validate_capture(bundle))
        raw, plan = bundle['raw'], bundle['plan']
        by_job = defaultdict(list)
        batches = {batch['id']: batch for batch in raw['batches']}
        for row in raw['request_records']:
            by_job[row['profile_job_id']].append(row)
        seen = set()
        for index, job in enumerate(plan['jobs']):
            if not job['id'].startswith('target:'):
                continue
            if len(by_job[job['id']]) != 1:
                raise ValueError('Exactly one serial target per planned block required')
            row = by_job[job['id']][0]
            seen.add(row['request_id'])
            identity = job['id'].removeprefix('target:')
            previous = plan['jobs'][index-1]
            if previous['id'].startswith('idle:'):
                if previous['requests'] or previous['id'].removeprefix('idle:') != identity:
                    raise ValueError('Idle block identity mismatch')
                precursor_job = plan['jobs'][index-2]
                idle_seconds = previous['duration_ms']/1000
                history = 'idle'+str(round(idle_seconds))
            else:
                precursor_job = previous
                idle_seconds = 0.
                history = identity.rsplit(':', 2)[1]
            if precursor_job['id'] != 'history:'+identity or len(by_job[precursor_job['id']]) != 1:
                raise ValueError('Matching completed precursor required')
            precursor = by_job[precursor_job['id']][0]
            target_start = min(s['start_ns'] for s in stage_layout(row))
            precursor_ready = max(precursor['completion_time_ns'], precursor['previous_handler_next_request_ready']['perf_ns'])
            if (precursor_ready > batches[precursor_job['id']]['end_ns']
                    or batches[precursor_job['id']]['end_ns'] > row['scheduled_time_ns']
                    or row['scheduled_time_ns'] > target_start):
                raise ValueError('History and target are not serial')
            family = raw['resident_aliases'][row['model']]
            bands = {s['native_band_role']: (s['end_ns']-s['start_ns'])/1e6
                     for s in band_layout(row, family) if s['kind'] == 'gpu'}
            events = row['worker_response']['semantic_phase_events']
            rows.append(dict(run_id=raw['run_id'], request_id=row['request_id'], family=family,
                pool_id=row['pool_id'], history=history, idle_seconds=idle_seconds,
                serial_gap_ms=(target_start-precursor_ready)/1e6,
                handler_ms=row['worker_response']['worker_latency_ms'], bands_ms=bands,
                output_identity_sha256=digest(output_identity(row['worker_response'])),
                device_elapsed_ms_by_component={component: sum(e['device_elapsed_ms'] for e in events
                    if e['component'] == component) for component in {e['component'] for e in events}},
                device_scope='summed asynchronous stream intervals; not pure compute or global time placement'))
        if len(seen) != 84:
            raise ValueError('Complete predeclared 84-target cohort required')
        runs.append(dict(run_id=raw['run_id'], source=bundle['source'], targets=len(seen)))
    if len(contexts) != 1 or len({r['run_id'] for r in runs}) != len(runs) or len(runs) < 2:
        raise ValueError('Independently repeated matching context required')
    groups = defaultdict(list)
    output_groups = defaultdict(set)
    for row in rows:
        output_groups[row['family'], row['pool_id']].add(row['output_identity_sha256'])
        for band, duration in row['bands_ms'].items():
            groups[row['family'], row['pool_id'], band, row['history']].append((row['run_id'], duration))
    comparisons = []
    for key, values in sorted(groups.items()):
        by_run = defaultdict(list)
        for run, value in values:
            by_run[run].append(value)
        all_values = [v for _, v in values]
        comparisons.append(dict(family=key[0], pool_id=key[1], band=key[2], history=key[3],
            n=len(values), mean_ms=statistics.mean(all_values), min_ms=min(all_values), max_ms=max(all_values),
            maximum_to_minimum_duration=max(all_values)/min(all_values),
            per_run={run: dict(n=len(v), mean_ms=statistics.mean(v), min_ms=min(v), max_ms=max(v))
                     for run, v in by_run.items()}))
    report = dict(schema='gsdeploy.specialist_history_description.v1', runs=runs,
        execution_context_sha256=next(iter(contexts)), target_requests=rows, band_history_comparisons=comparisons,
        output_changes=[dict(family=k[0], pool_id=k[1], identities=len(v)) for k, v in output_groups.items() if len(v)>1],
        limitations=['training-only targeted input selection', 'precursor blocks randomize order but do not identify device/cache mechanism',
                     'postcompletion timing and output are diagnostics; no future observations may enter online prediction'],
        physical_predictor_qualified=False, formal_utility_qualified=False,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), target_requests=len(rows), output_changes=report['output_changes'])))


if __name__ == '__main__':
    main()
