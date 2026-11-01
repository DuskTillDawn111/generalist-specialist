"""Direct CPU-only resource description paired with the existing v7 slowdown cohort."""
from __future__ import annotations

import argparse
import bisect
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


def require(value, message):
    if not value:
        raise ValueError(message)


def read_once(path):
    path = Path(path).resolve()
    before = path.stat()
    blob = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Source changed while reading')
    return json.loads(blob), dict(path=str(path), sha256=hashlib.sha256(blob).hexdigest()), after


def describe(values):
    values = sorted(values)
    def q(p):
        x = (len(values)-1)*p
        i = int(x)
        return values[i] + (values[min(i+1, len(values)-1)]-values[i])*(x-i)
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                p10=q(.1) if values else None, p90=q(.9) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def analyze(plan, raw, slowdown, matched, plan_ref):
    require(raw['status'] == 'completed' and raw['backend'] == 'native' and raw['kind'] == 'measured', 'Complete native source required')
    require(raw['plan'] == plan_ref and not any(raw.get(k) for k in ('errors', 'runtime_errors', 'telemetry_errors')), 'Source binding/error mismatch')
    jobs = {j['id']: j for j in plan['jobs']}
    batches = {b['job_id']: b for b in raw['batches']}
    training = {j['id'] for j in plan['jobs'] if j['split'] == 'training'}
    require(len(batches) == len(raw['batches']) == 264 and set(batches) == training, 'Exact 264 training jobs required')
    records = {r['request_id']: r for r in raw['request_records']}
    paired = {r['request_id']: r for r in matched}
    require(len(records) == len(paired) == len(matched) == 690 and records.keys() == paired.keys(), 'Exact slowdown request cohort required')
    by_job = collections.defaultdict(list)
    for rid, r in records.items():
        p = paired[rid]; response = r['worker_response']; job = jobs[r['profile_job_id']]
        require(r['status'] == 'completed' and r['split'] == job['split'] == 'training', 'Nontraining/incomplete record')
        require((r['profile_job_id'], r['model'], r['task'], r['pool_id'], job['context']) ==
                (p['job_id'], p['alias'], p['task'], p['pool_id'], p['nominal_context']), 'Slowdown identity mismatch')
        require((response['start_perf_ns'], response['end_perf_ns']) ==
                (p['actual_worker_start_ns'], p['actual_worker_end_ns']), 'Slowdown actual interval mismatch')
        require(job['start_offsets_ms'] == p['nominal_offsets_ms'] and job['inputs'][r['model']] == r['pool_id'], 'Input/offset mismatch')
        by_job[p['job_id']].append(p)
    samples = raw['board_samples']; times = [s['time_ns'] for s in samples]
    require(times == sorted(set(times)) and len(times) > 1, 'Ordered distinct telemetry required')
    aliases = set(plan['resident_aliases']); pids = {str(x['pid']) for x in raw['ready'].values()}
    require(len(aliases) == len(pids) == 4, 'Four distinct residents required')
    for s in samples:
        require(set(s['owned_aliases']) == aliases and set(s['processes_mib']) == pids and not s['foreign_pids'], 'Owned telemetry identity mismatch')
        require(all(isinstance(s[k], (int, float)) and math.isfinite(s[k]) and s[k] >= 0 for k in ('watts', 'owned_used_mib')), 'Invalid sample')
        require(abs(sum(s['processes_mib'].values())-s['owned_used_mib']) < 1e-6, 'Owned memory sum mismatch')
    def at(t):
        require(times[0] <= t <= times[-1], 'Unbracketed power interval')
        i = bisect.bisect_right(times, t)-1
        if times[i] == t:
            return samples[i]['watts']
        return samples[i]['watts'] + (samples[i+1]['watts']-samples[i]['watts'])*(t-times[i])/(times[i+1]-times[i])
    def window(a, b):
        require(a < b and times[0] <= a < b <= times[-1], 'Invalid/unbracketed window')
        lo, hi = bisect.bisect_left(times, a), bisect.bisect_left(times, b)
        local = samples[lo:hi]
        points = [(a, at(a)), *[(s['time_ns'], s['watts']) for s in local if a < s['time_ns'] < b], (b, at(b))]
        watt_ns = sum((d-c)*(x+y)/2 for (c, x), (d, y) in zip(points, points[1:]))
        return dict(start_ns=a, end_ns=b, duration_s=(b-a)/1e9, sample_count=len(local),
                    sample_power_watts=describe([s['watts'] for s in local]),
                    time_weighted_board_watts=watt_ns/(b-a), integrated_board_joules=watt_ns/1e9,
                    sampled_owned_peak_mib=max((s['owned_used_mib'] for s in local), default=None),
                    owned_peak_known=bool(local), power_bracketed=True)
    job_rows = []
    context_intervals = collections.defaultdict(list)
    previous_end = None
    for batch in sorted(batches.values(), key=lambda b: b['start_ns']):
        jid = batch['job_id']; job = jobs[jid]; rows = by_job[jid]; a, b = batch['start_ns'], batch['end_ns']
        require(previous_end is None or previous_end <= a, 'Overlapping job windows')
        previous_end = b
        require(set(batch['request_ids']) == {r['request_id'] for r in rows} and set(job['inputs']) == {r['alias'] for r in rows}, 'Job coverage mismatch')
        require(all(a <= r['actual_worker_start_ns'] < r['actual_worker_end_ns'] <= b for r in rows), 'Handler outside job window')
        cuts = sorted({a, b, *[r[k] for r in rows for k in ('actual_worker_start_ns', 'actual_worker_end_ns')]})
        for c, d in zip(cuts, cuts[1:]):
            active = tuple(sorted((r['alias'], r['task']) for r in rows if r['actual_worker_start_ns'] <= c < r['actual_worker_end_ns']))
            context_intervals[active].append(window(c, d))
        job_rows.append(dict(job_id=jid, nominal_context=job['context'], repeat=job['repeat'],
                             inputs=job['inputs'], nominal_offsets_ms=job['start_offsets_ms'], request_ids=batch['request_ids'],
                             **window(a, b), startup_through_drain_peak_mib=None,
                             startup_peak_status='UNAVAILABLE_MONITOR_STARTED_AFTER_ALL_WORKERS_READY_NO_PER_JOB_STARTUP'))
    def aggregate(rows):
        duration = sum(r['duration_s'] for r in rows)
        return dict(windows=len(rows), duration_s=duration, sample_count=sum(r['sample_count'] for r in rows),
                    missing_sample_windows=sum(not r['sample_count'] for r in rows),
                    integrated_board_joules=sum(r['integrated_board_joules'] for r in rows),
                    time_weighted_board_watts=sum(r['integrated_board_joules'] for r in rows)/duration,
                    sampled_owned_peak_mib=max((r['sampled_owned_peak_mib'] for r in rows if r['owned_peak_known']), default=None))
    groups = []
    for context in sorted({r['nominal_context'] for r in job_rows}):
        rows = [r for r in job_rows if r['nominal_context'] == context]
        groups.append(dict(nominal_context=context, **aggregate(rows),
                           job_mean_board_watts=describe([r['time_weighted_board_watts'] for r in rows]),
                           job_sampled_owned_peaks_mib=describe([r['sampled_owned_peak_mib'] for r in rows if r['owned_peak_known']]),
                           startup_through_drain_peak_mib=None))
    active = [dict(active_alias_tasks=[dict(alias=a, task=t) for a, t in key], **aggregate(rows))
              for key, rows in sorted(context_intervals.items())]
    left, right = raw['observation_start_ns'], raw['observation_end_ns']
    observed = window(left, right)
    require(all(left <= r['start_ns'] < r['end_ns'] <= right for r in job_rows), 'Jobs outside observation')
    priming_end = max(r['worker_response']['end_perf_ns'] for r in raw['priming_request_records'])
    require(priming_end <= left, 'Priming overlaps measurement')
    audit = dict(actual_training_jobs=len(job_rows), actual_training_requests=len(records),
                 strict_one_to_one_slowdown_request_interval_pairing=True, all_four_resident_PIDs_in_every_sample=True,
                 telemetry_samples_total=len(samples), priming_requests_excluded=len(raw['priming_request_records']),
                 unexecuted_validation_jobs_excluded=sum(j['split'] == 'validation' for j in plan['jobs']),
                 observation_sample_count=observed['sample_count'],
                 job_sample_count=sum(r['sample_count'] for r in job_rows),
                 samples_outside_job_measurement=sum(not any(r['start_ns'] <= t < r['end_ns'] for r in job_rows) for t in times),
                 all_jobs_power_bracketed=True, jobs_without_owned_peak_samples=sum(not r['sample_count'] for r in job_rows),
                 actual_active_context_groups=len(active),
                 actual_active_intervals_without_samples=sum(r['missing_sample_windows'] for r in active),
                 actual_active_context_groups_without_peak=sum(r['sampled_owned_peak_mib'] is None for r in active),
                 telemetry_spacing_ms=describe([(b-a)/1e6 for a, b in zip(times, times[1:])]),
                 query_duration_ms=describe([s['query_duration_ns']/1e6 for s in samples]),
                 startup_timestamps_present={k:k in raw for k in ('startup_begin_ns', 'startup_end_ns')},
                 startup_through_drain_owned_peak_status='UNAVAILABLE',
                 reason='One monitor begins after runtime.start/all READY; no loading samples and no per-job startup/reload. Batch windows cover scheduled batch start through drain return only.')
    return job_rows, groups, active, observed, audit


def plots(groups, output):
    with tempfile.TemporaryDirectory(prefix='resource-descriptive-mpl-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x = list(range(len(groups)))
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        means = [g['job_mean_board_watts']['median'] for g in groups]
        axes[0].bar(x, means, color='#377b9c')
        axes[0].errorbar(x, means, yerr=[[m-g['job_mean_board_watts']['p10'] for m, g in zip(means, groups)],
                                        [g['job_mean_board_watts']['p90']-m for m, g in zip(means, groups)]],
                         fmt='none', color='#173b50', capsize=3)
        axes[0].set_ylabel('Board watts')
        axes[0].set_title('Median job time-weighted power; whiskers = descriptive p10–p90')
        peaks = [g['sampled_owned_peak_mib']/1024 if g['sampled_owned_peak_mib'] is not None else 0 for g in groups]
        axes[1].bar(x, peaks, color='#a57541')
        axes[1].set_ylabel('Owned sampled peak (GiB)')
        axes[1].set_title('Maximum observed batch-window peak; four models resident in every group')
        axes[1].set_xticks(x, [g['nominal_context']+'\nn='+str(g['windows']) for g in groups], rotation=35, ha='right')
        for ax in axes:
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle('v7 actual training: board power and resident memory', fontsize=14)
        fig.text(.5, .015, '264 jobs / 690 paired requests; priming excluded. No startup samples.\nBoard power is not per-task attribution; sampled peaks are not capacity bounds. Whiskers are not confidence intervals.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .07, 1, .95))
        for suffix in ('png', 'pdf'):
            fig.savefig(output/('resource-summary.'+suffix), dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--slowdown', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Exclusive new output required')
    loaded = [read_once(p) for p in (args.input/'plan.json', args.input/'raw.json', args.slowdown/'summary.json', args.slowdown/'requests.json')]
    (plan, plan_ref, _), (raw, raw_ref, _), (slowdown, _, _), (matched, _, _) = loaded
    require(slowdown['input_receipts_before'] == [plan_ref, raw_ref] == slowdown['input_receipts_after'], 'Exact existing slowdown source hashes required')
    jobs, groups, active, observation, audit = analyze(plan, raw, slowdown, matched, plan_ref)
    args.output.mkdir(parents=True)
    plots(groups, args.output)
    for (_, ref, stat) in loaded:
        current = Path(ref['path']).stat()
        require((current.st_size, current.st_mtime_ns) == (stat.st_size, stat.st_mtime_ns), 'Source stat changed before publication')
    summary = dict(schema='gsdeploy.microprofile_resource_descriptive.v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
                   source_receipts=[r for _, r, _ in loaded], direct_sources_read_once=True,
                   unchanged_size_mtime_before_publication=True, closure_verified=False,
                   script=dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                   GPU_started=False, fitting_performed=False, predictions_run=False, tests_run=False, frozen_modules_imported=False,
                   audit=audit, measurement_observation=observation, nominal_context_groups=groups, actual_active_contexts=active,
                   definitions=dict(job_window='[batch.start_ns,batch.end_ns): scheduled batch start through drain return; no model startup.',
                                    power='Original NVML watts; trapezoidal integration with linear interpolation only between bracketing samples; no extrapolation.',
                                    power_statistics='Sample watts and per-job time-weighted means; nominal groups pool job durations for time-weighted watts.',
                                    memory='Maximum original owned_used_mib sample strictly inside stated half-open window; never interpolated.',
                                    active_context='Observed worker handler START<=t<END; idle means no observed handler, not GPU idle.',
                                    startup='Unavailable: monitor started only after runtime.start/all workers READY; no per-job reload.'),
                   limitations=['Board-wide watts includes all resident processes and device baseline; no task/SM/DRAM/L2 attribution or idle subtraction.',
                                'Actual active labels use host handler walls; NVML sampling/query latency and device power averaging can blur short transitions.',
                                'Nominal context includes offset-dependent partial overlap and drain; actual-active grouping is separately retained.',
                                'Memory includes all four residents and history-dependent allocations/cache, even for nominal solo activity.',
                                'No full startup-through-drain peak exists for any of the 264 jobs or for campaign loading; observed peak is neither physical bound nor confidence bound.',
                                'Power boundary interpolation is a declared sample model; short zero-sample intervals have interpolated power but memory peak remains unknown.',
                                'Descriptive p10/p90 and job count are not independent trial confidence intervals; no utility or capacity qualification.'])
    for name, document in [('summary.json', summary), ('jobs.json', jobs)]:
        (args.output/name).write_text(json.dumps(document, indent=2, ensure_ascii=False)+'\n')
    lines = ['# v7 training microprofile resources: descriptive supplement', '', 'UTC: '+summary['created_at_utc'], '',
             '**Existing telemetry supports board-power and observed batch-window memory summaries, but not startup-through-drain peaks.** The monitor starts after all workers are READY, and models are not reloaded per job. Every full startup peak is therefore UNKNOWN; none is synthesized.', '',
             'This supplement uses the exact 264 training jobs and 690 requests already described in the slowdown report, paired one-to-one by request/job/alias/task/pool/actual START/END and nominal offsets. The 18 priming requests and 264 unexecuted validation jobs are excluded from measurement statistics. All '+str(audit['telemetry_samples_total'])+' original telemetry samples retain four owned resident PIDs and no foreign PID.', '',
             'Power below uses each `[batch.start,batch.end)` window, from scheduled batch start through drain return. It includes partial overlap and offsets. Time-weighted means use the original piecewise-linear trapezoid convention. Memory is the maximum original sample within that window, without interpolation. No startup loading sample exists.', '',
             'Labels: S0/S1 are the two text specialists, G is generalist_2, O is OCR; S denotes one text specialist. Suffix image/asr is the G task, and no suffix means G text where present. All four models stay resident regardless of which aliases are active.', '',
             '| Nominal activity | Jobs | Samples | Time-weighted W | Job-mean W median [p10,p90] | Max observed owned MiB | Missing job peaks |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        d = g['job_mean_board_watts']
        lines.append(f"| {g['nominal_context']} | {g['windows']} | {g['sample_count']} | {g['time_weighted_board_watts']:.1f} | {d['median']:.1f} [{d['p10']:.1f},{d['p90']:.1f}] | {g['sampled_owned_peak_mib']:.1f} | {g['missing_sample_windows']} |")
    lines += ['', f"Across the post-priming observation span: {observation['sample_count']} samples, {observation['duration_s']:.3f}s, time-weighted {observation['time_weighted_board_watts']:.2f}W, sampled owned peak {observation['sampled_owned_peak_mib']:.1f}MiB. This span includes brief idle gaps around/between jobs. Job windows contain {audit['job_sample_count']} samples, with {audit['jobs_without_owned_peak_samples']} missing job peaks. The {audit['samples_outside_job_measurement']} other campaign samples include priming and outside-job gaps and do not enter the table.", '',
              f"Observed sample spacing: median {audit['telemetry_spacing_ms']['median']:.3f}ms, p90 {audit['telemetry_spacing_ms']['p90']:.3f}ms, max {audit['telemetry_spacing_ms']['maximum']:.3f}ms; nominal monitor sleep is 20ms. Query duration median/max {audit['query_duration_ms']['median']:.3f}/{audit['query_duration_ms']['maximum']:.3f}ms. The recorded timestamp is the query midpoint, not a kernel boundary.", '',
              '`summary.json` separately retains observed active alias/task combinations, using actual handler START/END intervals within jobs. Their duration, time-weighted whole-board watts, sample count, zero-sample interval count and observed owned peak distinguish actual overlap from nominal activity. Zero-sample short intervals may have bracket-interpolated power; their own memory peak remains null. This does not assert that the NVML power value arises solely from those active handlers.', '',
              f"There are {audit['actual_active_context_groups']} observed active-context groups, including no-handler intervals; {audit['actual_active_intervals_without_samples']} short intervals have no internal sample, and {audit['actual_active_context_groups_without_peak']} entire groups have no observed memory peak. Samples and durations partition the batch windows without duplication.", '',
              '![Resource summary]('+str((args.output/'resource-summary.png').resolve())+')', '',
              'All four models remain resident for solo and concurrent jobs. Peak changes can reflect retained allocations/cache and prior job history. Board watts include the device baseline and all residents; neither power nor memory is assigned to an individual task or to SM/DRAM demand. Short host intervals and NVML power averaging limit transition attribution. Peaks are sampled observations, not physical or capacity bounds. Whiskers are descriptive quantiles, not confidence intervals.', '',
              'Only these new descriptive outputs and their standalone CPU script were created. Inputs were read once, directly hashed and matched to the earlier slowdown receipts; source size/mtime remained unchanged before publication. No native raw closure verification, GPU work, fit, prediction, existing test or frozen-module import was performed.', '']
    (args.output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(args.output), audit=audit, observation=observation, groups=len(groups), actual_contexts=len(active)), indent=2))


if __name__ == '__main__':
    main()
