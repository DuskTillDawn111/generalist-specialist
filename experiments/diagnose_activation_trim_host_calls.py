"""Retain matched outer host-call and CPU brackets in the heap comparison.

The primary runtime report keeps inner semantic duration alongside outer CPU
counters. This supplement labels both timing envelopes explicitly. It never
subtracts CPU from wall time to invent a waiting-time measurement, nor sums
possibly nested calls to infer request CPU time.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.diagnose_activation_host_brackets_v2 import describe_event
from experiments.analyze_activation_host_diagnostic_v2 import validate_trace, overlap_ms
from experiments.analyze_activation_semantic import summarize
from gsdeploy.forecasting.native_training import validate_capture


def analyze(captures):
    observations, sources, summaries = [], {}, defaultdict(list)
    for path in captures:
        bundle = load_bundle(path.resolve())
        validate_capture(bundle, allowed_splits=('development_host_trim_comparison',))
        raw, plan, context = (bundle[k] for k in ('raw', 'plan', 'context'))
        mode = plan['host_trim_mode']
        if (mode not in ('off', 'on') or plan.get('not_for_training') is not True
                or plan['gc_trace_mode'] != 'on' or plan['host_trace_mode'] != 'on'
                or context['postprime_host_trim']['mode'] != mode
                or any(b['env']['GS_POSTPRIME_HOST_TRIM'] != mode for b in context['worker_bindings'].values())):
            raise ValueError('Declared development heap arm with complete host/GC brackets required')
        if raw['run_id'] in sources:
            raise ValueError('Each separately launched capture may appear only once')
        sources[raw['run_id']] = bundle['source']
        for row in raw['request_records']:
            response = row['worker_response']
            trace = response['passive_gc_trace']
            family = raw['resident_aliases'][row['model']]
            validate_trace(trace, response['start_perf_ns'], response['end_perf_ns'],
                           raw['ready'][row['model']]['passive_gc_policy'])
            for event in response['semantic_phase_events']:
                value = dict(capture=raw['run_id'], mode=mode, family=family, task=row['task'],
                    request_id=row['request_id'], pool_id=row['pool_id'], **describe_event(event))
                value['semantic_interval_gc_overlap_ms'] = overlap_ms(trace['events'],
                    event['host_start_ns'], event['host_end_ns'])
                summaries[mode, family, row['task'], event['component']].append(len(observations))
                observations.append(value)
    groups = []
    metrics = ('semantic_host_ms', 'outer_wall_ms', 'thread_cpu_ms', 'usage_system_ms',
               'device_interval_ms', 'outer_minus_semantic_host_ms')
    for key, indices in sorted(summaries.items()):
        rows = [observations[i] for i in indices]
        groups.append(dict(mode=key[0], family=key[1], task=key[2], component=key[3],
            independent_capture_ids=sorted({r['capture'] for r in rows}),
            metrics={k: summarize([r[k] for r in rows]) for k in metrics},
            semantic_calls_with_system_cpu_over_10ms=sum(r['usage_system_ms'] > 10 for r in rows),
            longest_system_cpu_observation_indices=sorted(indices,
                key=lambda i: observations[i]['usage_system_ms'], reverse=True)[:5],
            observation_indices=indices))
    return dict(schema='gsdeploy.trim_host_call_diagnostic.v1', sources=sources,
        complete_ABBA_captures=set(sources) == {'control-round1', 'trim-round1', 'trim-round2', 'control-round2'},
        observations=observations, groups=groups, physical_predictor_qualified=False,
        formal_utility_qualified=False,
        limits=[
            'Outer wall brackets and thread CPU enclose the same call; their clocks can have different rates. rusage counters enclose a slightly wider bracket.',
            'Semantic duration is the inner interval, explicitly distinct from the outer host/CPU bracket.',
            'System CPU >10ms is a descriptive count of semantic calls, not an acceptance threshold or independent stall count; nested calls may overlap.',
            'All observations are retained, including CPU > wall. No latency subtraction or causal NUMA/CUDA attribution.',
            'An incomplete ABBA set is preliminary only. Runtime adoption and predictor qualification require separate gates.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.capture)
    paths = [Path(__file__).resolve(), ROOT/'experiments/diagnose_activation_host_brackets_v2.py',
             ROOT/'experiments/analyze_activation_host_diagnostic_v2.py']
    result['analysis_sources'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(captures=len(result['sources']), events=len(result['observations']),
        complete_ABBA_captures=result['complete_ABBA_captures'],
        groups=[{k: r[k] for k in ('mode', 'family', 'task', 'component', 'semantic_calls_with_system_cpu_over_10ms')}
                for r in result['groups'] if r['semantic_calls_with_system_cpu_over_10ms']])))
