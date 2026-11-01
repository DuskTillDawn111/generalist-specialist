"""Directly sampled natural board power and owned memory training hypothesis.

This estimates resource terms only, never service slowdown or per-model causal
power. Entire simultaneous owner/task states are observed directly; unseen
states are not synthesized by adding solo or pair profiles.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import statistics

from .empirical_resources import integrate_samples
from .kernel_domain_bridge_review_v3 import load_report as load_v3
from .small_training_domain_bridge_review_v4 import load_report as load_v4
from .native_collect import receipt, save
from .native_execution_projection_v5 import native_execution_projection
from .profiling import RESIDENTS, digest

SCHEMA = 'gsdeploy.natural_direct_state_resource_hypothesis.v5'


def state_key(active, aliases):
    counts = Counter((aliases[alias]['family'], row['task']) for alias, row in active.items())
    members = [dict(family=f, task=t, count=n) for (f, t), n in sorted(counts.items())]
    return digest(members), members


def capture_intervals(raw, aliases):
    if (raw.get('kind') != 'measured' or raw.get('backend') != 'native'
            or raw.get('status') != 'completed'
            or any(raw.get(k) for k in ('errors', 'runtime_errors', 'telemetry_errors'))):
        raise ValueError('Completed error-free native training resource source required')
    rows = raw['request_records']
    if not rows or any(row.get('split') != 'training' or row.get('status') != 'completed' for row in rows):
        raise ValueError('Only actual completed training requests may label resource states')
    left, right = raw['observation_start_ns'], raw['observation_end_ns']
    if not left < right:
        raise ValueError('Positive actual active observation window required')
    intervals = []
    for row in rows:
        start, end = (row['worker_response'][k] for k in ('start_perf_ns', 'end_perf_ns'))
        if not left <= start < end <= right:
            raise ValueError('Actual handler intervals must lie inside observation window')
        intervals.append((start, end, row))
    for alias in aliases:
        owned = sorted((a, b) for a, b, row in intervals if row['model'] == alias)
        if any(a[1] > b[0] for a, b in zip(owned, owned[1:])):
            raise ValueError('One owner cannot execute overlapping request handlers')
    samples = raw['board_samples']
    stamps = [s['time_ns'] for s in samples]
    if len(stamps) < 2 or stamps != sorted(set(stamps)) or stamps[0] > left or stamps[-1] < right:
        raise ValueError('Ordered telemetry must bracket the complete observation')
    if max(b-a for a, b in zip(stamps, stamps[1:])) > 2_000_000_000:
        raise ValueError('Original two-second telemetry gap limit exceeded')
    # Include bracketing neighbors, but not startup/shutdown process transients.
    before = max(i for i, stamp in enumerate(stamps) if stamp <= left)
    after = min(i for i, stamp in enumerate(stamps) if stamp >= right)
    samples = samples[before:after+1]
    for sample in samples:
        if sample.get('foreign_pids') or set(sample['owned_aliases']) != set(aliases):
            raise ValueError('All four owned residents and no observed foreign process required')
        if any(type(sample.get(k)) not in (float, int) or not math.isfinite(sample[k]) or sample[k] < 0
               for k in ('watts', 'owned_used_mib')):
            raise ValueError('Actual finite board power and simultaneous owned memory required')
    cuts = sorted({left, right, *(t for a, b, _ in intervals for t in (a, b))})
    result = []
    for a, b in zip(cuts, cuts[1:]):
        active = {row['model']: row for start, end, row in intervals if start <= a and b <= end}
        key, members = state_key(active, aliases)
        local = [s for s in samples if a <= s['time_ns'] < b]
        result.append(dict(start_ns=a, end_ns=b, duration_ns=b-a, key=key, members=members,
            interior_sample_count=len(local),
            board_energy_joules=integrate_samples(samples, a, b)/1e9,
            sampled_owned_peak_mib=max(s['owned_used_mib'] for s in local) if local else None))
    return result


def fit(reviews, output):
    aliases = {a: dict(family=f) for a, f in RESIDENTS.items()}
    groups, captures, sources, index, seen = defaultdict(list), [], [], [], set()
    projection = None
    for path in reviews:
        doc = json.loads(Path(path).read_text())
        loader = load_v4 if doc.get('schema') == 'gsdeploy.startup_training_collection_strict_review.v4' else load_v3
        report, plan, raw = loader(path, 'natural')
        if (plan.get('split') != 'training' or any(j.get('split') != 'training' for j in plan['jobs'])
                or raw.get('execution_context_sha256') != plan['execution_context_sha256']):
            raise ValueError('Explicit same-context exclusively training plan required')
        source = report['raw']
        if source['sha256'] in seen:
            raise ValueError('Repeated raw is not an independent resource capture')
        seen.add(source['sha256'])
        context = json.loads(Path(report['context']['path']).read_text())
        identity = native_execution_projection(context)
        if projection is None:
            projection = identity['projection_sha256']
        if (projection != identity['projection_sha256'] or context['residents'] != RESIDENTS
                or context['execution'].get('admission') != 'default_shared_no_permit'):
            raise ValueError('Identical declared natural native projection and shared four owners required')
        intervals = capture_intervals(raw, aliases)
        run = 'raw-sha256:' + source['sha256']
        index.extend(dict(run_id=run, request_id=r['request_id'], pool_id=r['pool_id'], task=r['task'])
                     for r in raw['request_records'])
        for interval in intervals:
            if interval['interior_sample_count']:
                groups[interval['key']].append(dict(run_id=run, **interval))
        captures.append(dict(run_id=run, raw=source, review=receipt(path),
            original_execution_context_sha256=plan['execution_context_sha256'],
            projection_identity=identity, interval_count=len(intervals),
            intervals_without_direct_sample=sum(not i['interior_sample_count'] for i in intervals),
            unobserved_state_duration_ns=sum(i['duration_ns'] for i in intervals if not i['interior_sample_count'])))
        sources.extend([receipt(path), report['raw'], report['plan'], report['context']])
    if not captures:
        raise ValueError('Nonempty natural training captures required')
    states = {}
    for key, rows in groups.items():
        runs = {r['run_id'] for r in rows}
        per_run = [max(r['sampled_owned_peak_mib'] for r in rows if r['run_id'] == run) for run in runs]
        duration = sum(r['duration_ns'] for r in rows)
        mean_power = sum(r['board_energy_joules'] for r in rows) / (duration/1e9)
        states[key] = dict(members=rows[0]['members'], independent_capture_count=len(runs),
            known=len(runs) >= 2, intervals=rows, duration_ns=duration,
            scenarios=dict(mean=dict(watts=mean_power, peak_memory_mib=statistics.mean(per_run)),
                max=dict(watts=max(r['board_energy_joules']/(r['duration_ns']/1e9) for r in rows),
                         peak_memory_mib=max(per_run))))
    result = dict(schema=SCHEMA, native_projection_sha256=projection, states=states, captures=captures,
        provenance=dict(kind='measured', split='training'),
        minimum_independent_captures_per_state=2,
        scope='Direct full-state, handler-activity conditional board mean power and per-capture simultaneous owned sampled memory peaks; not causal model attribution or service prediction.',
        memory_safe_factor_applied=False, continuous_peak_known=False, capacity_bound=False,
        startup_energy_included=False, priming_included=False, drain_energy_included=False,
        collector_invariance_is_hypothesis=True, physical_accuracy_validated=False,
        sources=sources+[receipt(__file__), receipt(Path(__file__).with_name('empirical_resources.py')),
                         receipt(Path(__file__).with_name('native_execution_projection_v5.py'))],
        GPU_started=False, validation_started=False)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out/'training-index.json', dict(schema='gsdeploy.independent_prediction_gate_training_index.v2',
                                        split='training', records=index))
    result['training_index'] = receipt(out/'training-index.json')
    save(out/'model.json', result)
    return result


class NaturalResourceProvider:
    def __init__(self, model, target_context, *, allow_conditional_resource_hypothesis=False, resource_scenario='mean'):
        if not allow_conditional_resource_hypothesis:
            raise ValueError('Explicit conditional resource hypothesis opt-in required')
        projection = native_execution_projection(target_context)
        if (model.get('schema') != SCHEMA or model.get('provenance') != dict(kind='measured', split='training')
                or model['native_projection_sha256'] != projection['projection_sha256']
                or resource_scenario not in ('mean', 'max')):
            raise ValueError('Bound training natural resource projection/scenario required')
        if not model.get('sources') or any(receipt(ref['path']) != ref for ref in model['sources']):
            raise ValueError('Changed or missing natural resource training source receipt')
        self.document = copy.deepcopy(model)
        self.resource_scenario = resource_scenario
        self.execution_context_sha256 = projection['original_full_context_sha256']

    def _lookup(self, active, aliases):
        key, _ = state_key(active, aliases)
        state = self.document['states'].get(key)
        if not state or not state['known'] or state['independent_capture_count'] < 2:
            return None
        return state['scenarios'][self.resource_scenario]

    def power(self, active, aliases, scenario):
        value = self._lookup(active, aliases)
        return dict(known=value is not None, watts=value['watts'] if value else None,
                    evidence='Direct natural training state; missing full states remain UNKNOWN',
                    causal_per_model_attribution=False)

    def memory(self, active, aliases, scenario):
        value = self._lookup(active, aliases)
        return dict(known=value is not None, peak_memory_mib=value['peak_memory_mib'] if value else None,
                    evidence='Simultaneous owned sampled training state peak; no sum of per-process allocator maxima',
                    capacity_bound=False, safe_factor_applied=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = fit(args.reviews, args.output)
    print(json.dumps(dict(states=len(result['states']), supported=sum(s['known'] for s in result['states'].values()),
                          independent_captures=len(result['captures']))))


if __name__ == '__main__':
    main()
