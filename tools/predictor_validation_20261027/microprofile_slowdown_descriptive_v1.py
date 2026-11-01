"""Describe direct v7 actual training walls; no frozen-module imports or fitting."""
from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

RESIDENTS = {'text#0': 'text', 'text#1': 'text', 'generalist_2#0': 'generalist_2', 'ocr#0': 'ocr'}
METRICS = ('worker_latency_ms', 'prefill_wall_ms', 'decode_forward_sum_wall_ms')
CONTEXTS = ['SS', 'SG', 'SSG', 'SSGO', 'SG_image', 'SSG_image', 'SSGO_image',
            'SG_asr', 'SSG_asr', 'SSGO_asr']
TARGETS = [('text#0', 'text'), ('text#1', 'text'), ('generalist_2#0', 'text'),
           ('generalist_2#0', 'image'), ('generalist_2#0', 'asr'), ('ocr#0', 'ocr')]


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def describe(values):
    values = sorted(values)
    def percentile(fraction):
        position = (len(values)-1)*fraction
        left = int(position)
        return values[left] + (values[min(left+1, len(values)-1)]-values[left])*(position-left)
    return dict(n=len(values), median=statistics.median(values) if values else None,
                p10=percentile(.1) if values else None, p90=percentile(.9) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def semantic_walls(row, start, end):
    observation = row['worker_response'].get('semantic_forward_observations')
    result = dict(prefill_wall_ms=None, decode_forward_sum_wall_ms=None,
                  prefill_forward_count=0, decode_forward_count=0)
    if not observation or not observation.get('records'):
        return result
    require(observation['request_id'] == row['request_id'] and observation['clock'] == 'time.perf_counter_ns'
            and not observation.get('instrumentation_errors'), 'Foreign/error semantic observation')
    events = row['execution_events']
    previous = start
    durations = collections.defaultdict(list)
    for forward in observation['records']:
        phase = forward['semantic_phase']; a, b = forward['start_ns'], forward['end_ns']
        require(phase in ('prefill', 'decode') and previous <= a < b <= end, 'Invalid semantic forward order')
        previous = b
        for edge, timestamp in [('start', a), ('end', b)]:
            matches = [e for e in events if e.get('segment_id') == forward['id'] and e.get('event') == edge]
            require(len(matches) == 1 and matches[0]['perf_ns'] == timestamp
                    and matches[0].get('semantic_phase') == phase, 'Semantic event/response mismatch')
        durations[phase].append((b-a)/1e6)
    for phase, metric in [('prefill', 'prefill_wall_ms'), ('decode', 'decode_forward_sum_wall_ms')]:
        result[phase+'_forward_count'] = len(durations[phase])
        if durations[phase]:
            result[metric] = sum(durations[phase])
    return result


def analyze(plan, raw, plan_ref):
    require(plan['resident_aliases'] == RESIDENTS and set(raw['ready']) == set(RESIDENTS), 'Four exact residents required')
    require(raw['schema'] == 'gsdeploy.native_phase_collection.v1' and raw['kind'] == 'measured'
            and raw['backend'] == 'native' and raw['status'] == 'completed' and raw['plan'] == plan_ref,
            'Completed original native profile and exact direct plan required')
    require(not any(raw.get(k) for k in ('errors', 'runtime_errors', 'telemetry_errors')), 'Native collection errors')
    require(raw['execution_context_sha256'] == plan['execution_context_sha256'], 'Context identity mismatch')
    for alias, family in RESIDENTS.items():
        require(raw['ready'][alias]['worker'] == family and raw['ready'][alias]['clock'] == 'time.perf_counter_ns', 'READY mismatch')
        require(raw['shutdown'][alias] == dict(exit_code=0, forced=False, bye_received=True), 'Unclean native shutdown')
    resident_pids = {str(v['pid']) for v in raw['ready'].values()}
    require(len(resident_pids) == 4 and bool(raw['board_samples'])
            and all(set(s['owned_aliases']) == set(RESIDENTS) and set(s['processes_mib']) == resident_pids
                    and not s['foreign_pids'] for s in raw['board_samples']), 'Four distinct owned resident PIDs must persist in actual telemetry')
    jobs = {j['id']: j for j in plan['jobs']}; batches = {b['job_id']: b for b in raw['batches']}
    require(len(jobs) == len(plan['jobs']) and len(batches) == len(raw['batches']), 'Duplicate job/batch')
    training_ids = {j['id'] for j in plan['jobs'] if j['split'] == 'training'}
    require(set(batches) == training_ids, 'Actual batches must equal only the declared training jobs')
    all_native = raw['request_records'] + raw['priming_request_records']
    require(len({r['request_id'] for r in all_native}) == len(all_native), 'Duplicate native request')
    require(all(r['status'] == 'completed' and r.get('worker_response', {}).get('success') is True for r in all_native), 'Incomplete actual native requests')
    intervals = [(r['request_id'], r['model'], r['worker_response']['start_perf_ns'], r['worker_response']['end_perf_ns']) for r in all_native]
    rows = []; by_job = collections.defaultdict(list)
    for row in raw['request_records']:
        job = jobs[row['profile_job_id']]; batch = batches[job['id']]; alias = row['model']; response = row['worker_response']
        context = job['context'].split('_')[0]
        context_aliases = {'S0': ['text#0'], 'S1': ['text#1'], 'G': ['generalist_2#0'], 'O': ['ocr#0'],
                           'SS': ['text#0', 'text#1'], 'SG': ['text#0', 'generalist_2#0'],
                           'SSG': ['text#0', 'text#1', 'generalist_2#0'], 'SSGO': list(RESIDENTS)}
        require(context in context_aliases and set(job['inputs']) == set(context_aliases[context]), 'Nominal context/active aliases mismatch')
        task = job.get('tasks', {}).get(alias, 'ocr' if alias == 'ocr#0' else 'text')
        if alias == 'generalist_2#0':
            require(task == (job['context'].split('_', 1)[1] if '_' in job['context'] else 'text'), 'Nominal generalist task/context suffix mismatch')
        require(row['split'] == job['split'] == 'training' and alias in job['inputs'], 'Nontraining/foreign actual request')
        require(row['pool_id'] == job['inputs'][alias] == row['sample_id'] == response['sample_id']
                and row['task'] == response['task'] == response['task_type'] == task, 'Nominal task/input mismatch')
        require(response['request_id'] == row['request_id'], 'Response request mismatch')
        scheduled = batch['start_ns'] + round(job['start_offsets_ms'][alias]*1e6)
        start, end = response['start_perf_ns'], response['end_perf_ns']
        require(row['scheduled_time_ns'] == scheduled and scheduled <= row['actual_submit_time_ns'] <= start < end <= row['completion_time_ns'] <= batch['end_ns'], 'Offset/actual worker chronology mismatch')
        require(row['observed_worker_start_time_ns'] == start, 'Missing/mismatched observed worker START')
        starts = [e for e in row['execution_events'] if 'event' not in e]
        require(len(starts) == 1 and starts[0]['perf_ns'] == start and starts[0]['pid'] == raw['ready'][alias]['pid'], 'Actual worker START event mismatch')
        require(math.isclose(row['worker_latency_ms'], (end-start)/1e6, abs_tol=1e-6, rel_tol=1e-9), 'Worker wall/latency mismatch')
        peers = [(rid, other, max(start, a), min(end, b)) for rid, other, a, b in intervals
                 if rid != row['request_id'] and max(start, a) < min(end, b)]
        require(all(other != alias for _, other, _, _ in peers), 'Actual same-worker handlers overlap')
        cuts = sorted({start, end, *[x for _, _, a, b in peers for x in (a, b)]})
        overlap = sum(b-a for a, b in zip(cuts, cuts[1:]) if any(left < b and a < right for _, _, left, right in peers))
        result = dict(request_id=row['request_id'], job_id=job['id'], nominal_context=job['context'],
                      alias=alias, task=task, pool_id=row['pool_id'], repeat=job['repeat'],
                      nominal_offsets_ms=job['start_offsets_ms'], nominal_offset_ms=job['start_offsets_ms'][alias],
                      scheduled_time_ns=scheduled, actual_worker_start_ns=start, actual_worker_end_ns=end,
                      worker_start_lag_ms=(start-scheduled)/1e6, worker_latency_ms=row['worker_latency_ms'],
                      actual_peer_overlap_fraction=overlap/(end-start),
                      observed_peer_request_ids=[p[0] for p in peers],
                      generated_tokens=response.get('generated_tokens'), input_tokens=response.get('input_tokens'),
                      nominal_single_active_job=len(job['inputs']) == 1, **semantic_walls(row, start, end))
        rows.append(result); by_job[job['id']].append(result)
    require(len(rows) == 690 and len(by_job) == 264, 'Exact v7 actual 690-request/264-training-job cohort required')
    for jid, batch_rows in by_job.items():
        require(collections.Counter(r['alias'] for r in batch_rows) == collections.Counter(jobs[jid]['inputs'].keys())
                and {r['request_id'] for r in batch_rows} == set(batches[jid]['request_ids']), 'Job alias/request coverage mismatch')
    baselines = collections.defaultdict(list)
    for row in rows:
        if row['nominal_single_active_job'] and row['actual_peer_overlap_fraction'] == 0:
            baselines[(row['alias'], row['task'], row['pool_id'])].append(row)
    baseline_report = []
    for key, solo in sorted(baselines.items()):
        baseline_report.append(dict(alias=key[0], task=key[1], pool_id=key[2], request_ids=[r['request_id'] for r in solo],
                                    metrics={m: describe([r[m] for r in solo if r[m] is not None]) for m in METRICS},
                                    decode_forward_counts=[r['decode_forward_count'] for r in solo]))
    for row in rows:
        solo = baselines.get((row['alias'], row['task'], row['pool_id']), [])
        row['baseline_request_ids'] = [r['request_id'] for r in solo]; row['slowdown'] = {}
        for metric in METRICS:
            values = [s[metric] for s in solo if s[metric] is not None and s[metric] > 0]
            median = statistics.median(values) if values else None
            reason = ('NO_EXACT_ALIAS_TASK_INPUT_SOLO_BASELINE' if not solo else
                      'NO_POSITIVE_SOLO_PHASE_EVIDENCE' if median is None else
                      'NO_TARGET_PHASE_EVIDENCE' if row[metric] is None else None)
            row['slowdown'][metric] = dict(status='KNOWN' if reason is None else 'UNKNOWN',
                ratio=row[metric]/median if reason is None else None, baseline_median_ms=median,
                baseline_n=len(values), unknown_reason=reason)
        row['decode_count_matches_all_baselines'] = bool(solo) and all(row['decode_forward_count'] == s['decode_forward_count'] for s in solo)
    def aggregate(keys):
        groups = collections.defaultdict(list)
        for row in rows: groups[tuple(row[k] for k in keys)].append(row)
        return [{**dict(zip(keys, key)), 'requests': len(group),
                 'actual_peer_overlap_fraction': describe([r['actual_peer_overlap_fraction'] for r in group]),
                 'worker_start_lag_ms': describe([r['worker_start_lag_ms'] for r in group]),
                 'decode_count_mismatch_requests': sum(not r['decode_count_matches_all_baselines'] for r in group),
                 'slowdown': {m: {**describe([r['slowdown'][m]['ratio'] for r in group if r['slowdown'][m]['ratio'] is not None]),
                                  'unknown': sum(r['slowdown'][m]['ratio'] is None for r in group)} for m in METRICS}}
                for key, group in sorted(groups.items())]
    audit = dict(four_exact_residents=True, four_distinct_resident_PIDs=True,
                 telemetry_samples_with_all_four_owned_residents=len(raw['board_samples']),
                 all_native_requests_completed=len(all_native), priming_excluded=len(raw['priming_request_records']),
                 actual_training_jobs=len(by_job), actual_training_requests=len(rows),
                 unexecuted_validation_jobs_excluded=sum(j['split'] == 'validation' for j in plan['jobs']),
                 nominal_context_task_input_offset_and_observed_START_validated=True,
                 exact_solo_baseline_keys=len(baselines), solo_baseline_requests=sum(map(len, baselines.values())),
                 no_shared_or_other_input_fallback=True,
                 unknown_by_metric={m: sum(r['slowdown'][m]['ratio'] is None for r in rows) for m in METRICS})
    return (rows, baseline_report, aggregate(['nominal_context', 'alias', 'task']),
            aggregate(['nominal_context', 'alias', 'task', 'nominal_offset_ms']),
            aggregate(['nominal_context', 'alias', 'task', 'pool_id']), audit)


def plots(groups, output):
    with tempfile.TemporaryDirectory(prefix='microprofile-matplotlib-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm
        indexed = {(g['nominal_context'], g['alias'], g['task']): g for g in groups}
        for name, metrics in [('worker-slowdown', METRICS[:1]), ('phase-slowdown', METRICS[1:])]:
            fig, axes = plt.subplots(1, len(metrics), figsize=(8.4*len(metrics), 7.2), squeeze=False)
            for ax, metric in zip(axes[0], metrics):
                array = np.full((len(CONTEXTS), len(TARGETS)), np.nan)
                for i, context in enumerate(CONTEXTS):
                    for j, target in enumerate(TARGETS):
                        group = indexed.get((context, *target))
                        if group and group['slowdown'][metric]['median'] is not None:
                            array[i, j] = group['slowdown'][metric]['median']
                known = array[np.isfinite(array)]
                image = ax.imshow(np.ma.masked_invalid(array), cmap='YlOrRd', norm=LogNorm(vmin=min(.5, known.min()), vmax=max(2., known.max())), aspect='auto')
                for i, context in enumerate(CONTEXTS):
                    for j, target in enumerate(TARGETS):
                        group = indexed.get((context, *target))
                        value = array[i, j]
                        label = f'{value:.2f}x\nn={group["slowdown"][metric]["n"]}' if math.isfinite(value) else 'UNKNOWN' if group else '-'
                        ax.text(j, i, label, ha='center', va='center', fontsize=8, color='black')
                ax.set_xticks(range(len(TARGETS)), [a.replace('generalist_2#0', 'G')+'\n'+t for a, t in TARGETS])
                ax.set_yticks(range(len(CONTEXTS)), CONTEXTS)
                ax.set_title(metric.replace('_', ' ')+'\nMedian of exact-input matched ratios', fontsize=11)
                fig.colorbar(image, ax=ax, label='wall / median solo wall')
            fig.suptitle('v7 actual training microprofiles: descriptive wall slowdown', fontsize=14)
            fig.text(.5, .018, 'Nominal contexts; actual overlap varies. Python forward walls are not GPU busy time.\nDecode sums also vary with forward count; no causal attribution or model fit.', ha='center', fontsize=10)
            fig.tight_layout(rect=(0, .075, 1, .94))
            for suffix in ('png', 'pdf'): fig.savefig(output/(name+'.'+suffix), dpi=180)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); source = args.input.resolve(); output = args.output.resolve()
    require(not output.exists(), 'New output directory required')
    refs_before = [receipt(source/name) for name in ('plan.json', 'raw.json')]
    plan, raw = [json.loads(Path(ref['path']).read_text()) for ref in refs_before]
    rows, baselines, groups, offsets, inputs, audit = analyze(plan, raw, refs_before[0])
    require([receipt(r['path']) for r in refs_before] == refs_before, 'Direct input changed during analysis')
    output.mkdir(parents=True)
    plots(groups, output)
    refs_after = [receipt(r['path']) for r in refs_before]
    require(refs_after == refs_before, 'Direct plan/raw changed before publication')
    summary = dict(schema='gsdeploy.microprofile_slowdown_descriptive.v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
        input_receipts_before=refs_before, input_receipts_after=refs_after, direct_SHA256_unchanged=True, script=receipt(__file__), audit=audit,
        definition='Per actual training request, wall duration / median of nominal single-active jobs with no actual peer overlap, matching exact alias/task/pool_id.',
        semantic_definition='Only real paired semantic forward START/END ranges; prefill sum and decode forward sum. No residual or heuristic phase split.',
        limitations=['Actual worker overlap varies within each nominal context; overlap and start lag are recorded per request.',
                     'All four models remain resident even in single-active baselines.',
                     'Two solo repeats per exact input; descriptive ratios are not independent trial confidence intervals.',
                     'Decode sum includes generation-length and forward-count variation; mismatch flags are retained.',
                     'Worker and Python language-forward walls include host/sync/asynchronous effects; these are not GPU busy time or SM/DRAM/L2 counter attribution.',
                     'No causal physical complementarity, capacity qualification, model training or GPU replay is claimed.'],
        GPU_started=False, fitting_performed=False, frozen_module_imports=False, expensive_closure_verification_performed=False,
        groups=groups, groups_by_nominal_offset=offsets, groups_by_input=inputs)
    for name, value in [('summary', summary), ('requests', rows), ('baselines', baselines), ('audit', audit)]:
        with (output/(name+'.json')).open('x') as stream: json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    with (output/'group-summary.csv').open('x', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['context', 'alias', 'task', 'metric', 'n_known', 'n_unknown', 'median_ratio', 'p10_ratio', 'p90_ratio', 'median_actual_overlap_fraction'])
        for group in groups:
            for metric in METRICS:
                s = group['slowdown'][metric]
                writer.writerow([group['nominal_context'], group['alias'], group['task'], metric, s['n'], s['unknown'], s['median'], s['p10'], s['p90'], group['actual_peer_overlap_fraction']['median']])
    print(json.dumps(dict(output=str(output), audit=audit, direct_SHA256_unchanged=True), ensure_ascii=False))


if __name__ == '__main__': main()
