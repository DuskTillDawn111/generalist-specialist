"""Audit measured load/prime/warmup energy and the state entering formal FIFO."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsdeploy.energy import integrate_power
from gsdeploy.forecasting.native_training import stage_layout


def describe(path):
    raw = json.loads(path.read_text())
    energy = raw['energy_evidence']
    measure = raw['measurement_window']['start_time_ns']
    warmup = raw['warmup_window']
    prime = raw['priming_request_records']
    prefix = prime+raw['warmup_request_records']
    if (raw['status'] != 'completed' or raw['backend'] != 'native' or raw['errors']
            or not energy['episode_coverage_valid'] or energy.get('sampling_errors')
            or any(row['status'] != 'completed' for row in prefix)):
        raise ValueError('Complete covered native episode prefix required')
    spans = []
    for row in prefix:
        for span in stage_layout(row):
            spans.append(dict(span, request_id=row['request_id'], alias=row['model'], phase=row['phase']))
        if row['completion_time_ns'] > measure:
            raise ValueError('Prefix contains a request completed after formal measurement begins')
    if not spans or any(s['end_ns'] > measure for s in spans):
        raise ValueError('Prefix native service reaches into future measurement')
    native_ends = [s['end_ns'] for s in spans if s['kind'] == 'gpu']
    if not native_ends:
        raise ValueError('No observed native energy prefix')
    last_native = max(native_ends)
    after_prime_counts = dict.fromkeys(raw['replica_to_family'], 0)
    after_prime_counts.update(Counter(r['model'] for r in raw['warmup_request_records']))
    prime_end = max(r['completion_time_ns'] for r in prime)
    bounds = [energy['startup_begin_time_ns'], min(r['scheduled_time_ns'] for r in prime),
              prime_end, warmup['start_time_ns'], warmup['drain_end_time_ns'], measure]
    labels = ['load_to_first_prime_offer', 'concurrent_prime_through_last_receipt',
              'prime_to_warmup_controller_gap', 'warmup_and_drain', 'warmup_to_measurement_controller_gap']
    if bounds != sorted(bounds):
        raise ValueError('Measured startup/prime/warmup ordering changed')
    partition = []
    for label, left, right in zip(labels, bounds, bounds[1:]):
        joules = integrate_power(energy['samples'], left, right) if right > left else 0.
        if joules is None:
            raise ValueError('Missing prefix board energy coverage')
        partition.append(dict(part=label, duration_seconds=(right-left)/1e9, joules=joules))
    whole = integrate_power(energy['samples'], bounds[0], bounds[-1])
    if whole is None or abs(sum(r['joules'] for r in partition)-whole) > 1e-6:
        raise ValueError('Prefix energy partition does not reconstruct the whole')
    # Detect actual cross-owner native overlap during priming, without claiming
    # that host native spans are pure GPU kernels.
    primes = [s for s in spans if s['phase'] == 'priming' and s['kind'] == 'gpu']
    overlap = sum(max(0, min(a['end_ns'], b['end_ns'])-max(a['start_ns'], b['start_ns']))
                  for i, a in enumerate(primes) for b in primes[i+1:] if a['alias'] != b['alias'])
    return dict(source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        deployment=raw['plan']['deployment'], seed=raw['transport_config']['random_seed'],
        episode_protocol=energy['episode_protocol'], prefix_joules=whole, prefix_partition=partition,
        measured_post_prime_owner_ordinals_at_measurement=after_prime_counts,
        measured_energy_last_native_end_relative_ns=last_native-measure,
        idle_age_at_measurement_seconds=(measure-last_native)/1e9,
        priming_cross_owner_native_pair_overlap_seconds=overlap/1e9,
        training_priming_is_serial=True, measured_priming_is_concurrent=overlap > 0,
        paper_evaluator_zero_ordinal_matches_this_prefix=not any(after_prime_counts.values()),
        full_prefix_reusable_for_another_route_or_seed=False,
        prediction_prefix_qualified=False,
        limitations=['conditional observed-prefix diagnosis only; these timings are not autonomous predictions',
                    'priming native spans mix host/transfers/CUDA; overlap is not kernel contention',
                    'post-prime order reset cannot transfer across different priming histories without validation'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path, action='append')
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    report = dict(schema='gsdeploy.episode_prefix_audit.v1', runs=[describe(r) for r in args.run],
                  formal_utility_qualified=False)
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
