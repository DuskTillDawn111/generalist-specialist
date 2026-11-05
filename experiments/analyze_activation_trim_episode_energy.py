"""Covered board energy through startup, all priming and formal full drain.

Diagnostic only. These serial captures do not qualify parallel priming or a
continuous-arrival energy predictor. No per-worker or sub-sample trim energy
is inferred from aggregate NVML polling.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fit_activation_native import load_bundle
from gsdeploy.energy import integrate_power
from gsdeploy.forecasting.native_training import validate_capture


def describe(raw):
    primes, formal = raw['priming_request_records'], raw['request_records']
    if not primes or not formal:
        raise ValueError('Complete priming and formal request records required')
    for row in primes + formal:
        ready = row['previous_handler_next_request_ready']
        response = row['worker_response']
        if not row['scheduled_time_ns'] <= response['start_perf_ns'] <= response['end_perf_ns'] <= ready['perf_ns']:
            raise ValueError('Invalid full request lifecycle ordering')
        if 'postprime_trim_receipt' in ready:
            receipt = ready['postprime_trim_receipt']
            if row in formal or not response['end_perf_ns'] <= receipt['procedure_start_ns'] <= receipt['procedure_end_ns'] <= ready['perf_ns']:
                raise ValueError('Trim is outside declared priming lifecycle')
    # Include both worker-ready and parent receipt, whichever is later. They
    # share perf_counter, but transmission means neither implies the other.
    drained = lambda rows: max(max(r['completion_time_ns'], r['previous_handler_next_request_ready']['perf_ns']) for r in rows)
    points = [raw['startup_begin_time_ns'], min(r['scheduled_time_ns'] for r in primes),
              drained(primes), raw['observation_start_ns'], drained(formal), raw['observation_end_ns']]
    if any(type(p) is not int for p in points) or any(a > b for a, b in zip(points, points[1:])):
        raise ValueError('Lifecycle windows do not cover ordered full drain')
    if min(r['scheduled_time_ns'] for r in formal) < points[3]:
        raise ValueError('Formal requests precede formal observation')
    samples = sorted(raw['board_samples'], key=lambda s: s['time_ns'])
    if any(s.get('foreign_pids') for s in samples) or raw.get('telemetry_errors'):
        raise ValueError('Clean whole-board telemetry required')
    if any(a['time_ns'] >= b['time_ns'] for a, b in zip(samples, samples[1:])):
        raise ValueError('Strictly ordered unique telemetry timestamps required')
    max_gap = .25  # Keep the existing native energy coverage requirement.
    def window(label, left, right):
        # A zero-duration transition is structurally zero, never missing data.
        joules = 0. if left == right else integrate_power(samples, left, right, max_gap_seconds=max_gap)
        return dict(window=label, start_ns=left, end_ns=right, seconds=(right-left)/1e9,
                    joules=joules, coverage_valid=joules is not None)
    labels = ['startup_to_first_priming_arrival', 'all_priming_through_full_ready',
              'post_priming_transition', 'formal_window_through_full_drain', 'post_drain_observation_tail']
    parts = [window(label, left, right) for label, left, right in zip(labels, points, points[1:])]
    totals = [window('startup_and_priming', points[0], points[3]),
              window('formal_observation', points[3], points[-1]),
              window('startup_through_observation_end', points[0], points[-1])]
    covered = all(w['coverage_valid'] for w in parts + totals)
    if covered and not math.isclose(sum(w['joules'] for w in parts), totals[-1]['joules'], rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError('Lifecycle integration does not conserve total board energy')
    in_window = [(a, b) for a, b in zip(samples, samples[1:])
                 if a['time_ns'] < points[-1] and b['time_ns'] > points[0]]
    gaps = [dict(left_ns=a['time_ns'], right_ns=b['time_ns'], seconds=(b['time_ns']-a['time_ns'])/1e9)
            for a, b in in_window if (b['time_ns']-a['time_ns'])/1e9 > max_gap]
    return dict(run_id=raw['run_id'], coverage_valid=covered, partitions=parts, totals=totals,
        maximum_allowed_sample_gap_seconds=max_gap, coverage_gaps=gaps,
        missing_energy_windows=[w['window'] for w in parts + totals if not w['coverage_valid']],
        telemetry_samples=len(samples), formal_requests=len(formal), priming_requests=len(primes),
        physical_predictor_qualified=False, formal_utility_qualified=False,
        scope='Measured GPU board energy, not host CPU/RAM/system energy. Serial startup and priming; includes trim through final worker READY. Shutdown energy is not assigned without explicit shutdown boundaries. No per-trim energy attribution or predictor admission.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for path in args.capture:
        bundle = load_bundle(path.resolve())
        validate_capture(bundle, allowed_splits=('development_host_trim_feasibility', 'development_host_trim_comparison'))
        report = describe(bundle['raw'])
        report['source'] = bundle['source']
        reports.append(report)
    result = dict(schema='gsdeploy.trim_episode_energy_diagnostic.v1', captures=reports,
        analysis_sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__).resolve(), ROOT/'gsdeploy/energy.py', ROOT/'experiments/fit_activation_native.py',
            ROOT/'gsdeploy/forecasting/native_training.py')}, physical_predictor_qualified=False)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps([dict(run_id=r['run_id'], coverage_valid=r['coverage_valid'], totals=r['totals']) for r in reports]))


if __name__ == '__main__':
    main()
