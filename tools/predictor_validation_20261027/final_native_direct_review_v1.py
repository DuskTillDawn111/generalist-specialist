"""Read-only cross-check of completed F8 cells using standard-library arithmetic.

This does not launch experiments, invoke the original verifier, rescore model
outputs, fit a predictor, or certify capacity. Q uses the native annotation;
the original verifier independently checks that annotation against references.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/predictor-gpu-20261027'
ORDER = ('shared_spill_high', 'admission2_spill_high', 'admission2_balanced',
         'shared_balanced', 'shared_skewed', 'admission2_skewed',
         'admission2_spill_low', 'shared_spill_low')


def load_ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return json.loads(data), {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


def close(a, b):
    assert type(a) in (int, float) and type(b) in (int, float)
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9), (a, b)


def p95(values):
    ordered = sorted(values)
    position = .95 * (len(ordered) - 1)
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def joules(samples, start, end):
    assert samples[0]['time_ns'] <= start < end <= samples[-1]['time_ns']
    areas = []
    covered = 0
    for a, b in zip(samples, samples[1:]):
        ta, tb = a['time_ns'], b['time_ns']
        assert ta < tb
        lo, hi = max(ta, start), min(tb, end)
        if lo >= hi:
            continue
        assert tb - ta <= 2_000_000_000
        wa, wb = a['watts'], b['watts']
        assert all(type(w) in (int, float) and math.isfinite(w) and w >= 0 for w in (wa, wb))
        # Integrate the linear segment analytically over the clipped interval.
        slope = (wb - wa) / (tb - ta)
        areas.append((wa * (hi - lo) + .5 * slope *
                      ((hi - ta) ** 2 - (lo - ta) ** 2)) / 1e9)
        covered += hi - lo
    assert covered == end - start
    return math.fsum(areas)


def review(candidate):
    assert candidate in ORDER
    execution = BASE / 'four-group-continuation-v2/execution'
    audit, audit_ref = load_ref(execution / (candidate + '-actual-audit.json'))
    plan, plan_ref = load_ref(BASE / 'four-group-diagnostic-v1/plan.json')
    seal, seal_ref = load_ref(BASE / 'four-group-continuation-v2/sealed/seal.json')
    raw, raw_ref = load_ref(execution / 'runs' / candidate / 'raw.json')
    scored, scored_ref = load_ref(execution / 'runs' / candidate / 'scored-raw.json')
    assert audit['native_raw'] == raw_ref and audit['scored_raw'] == scored_ref
    assert audit['plan'] == raw['plan'] == plan_ref and scored['native_raw'] == raw_ref
    assert {k: v for k, v in scored.items() if k not in ('native_raw', 'quality_evidence')} == raw
    assert raw['candidate_id'] == audit['candidate_id'] == candidate
    assert raw['startup_begin_ns'] > seal['created_ns'] >= plan['created_ns']
    assert plan['seed'] == 628301 and plan['window_seconds'] == 600.
    assert plan['drain_seconds'] == 300. and plan['startup_amortization_seconds'] == 3600.
    assert plan['memory_safe_factor'] == 1.110854 and plan['memory_budget_mib'] == 49140
    assert plan['energy_reference_watts'] == 450. and all(v == 1/6 for v in plan['weights'].values())
    assert not raw['runtime_errors'] and not raw['errors'] and not raw['telemetry_errors']
    assert not scored['quality_evidence']['errors'] and not raw['warmup_request_records']
    assert len(raw['ready']) == len(raw['shutdown']) == 4
    assert all(v['exit_code'] == 0 and not v['forced'] and v['bye_received'] for v in raw['shutdown'].values())
    start, end = raw['phase_start_ns'], raw['deadline_ns']
    assert end - start == 600_000_000_000
    assert end <= raw['observation_end_ns'] <= end + 300_000_000_000
    rows = raw['request_records']
    trace = {r['request_id']: r for r in plan['trace']}
    route = next(c for c in plan['candidates'] if c['id'] == candidate)['routes']
    assert len(rows) == len(trace) == len({r['request_id'] for r in rows}) == 1200
    quality = scored['quality_evidence']['totals_by_request']
    assert set(quality) == set(trace)
    latency, numerators, denominators = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        rid, task = row['request_id'], row['task']
        expected = trace[rid]
        assert row['status'] == 'completed' and row['phase'] == 'measurement'
        assert task == expected['task'] and row['pool_id'] == expected['pool_id']
        assert row['model'] == route[rid] and row['scheduled_time_ns'] - start == expected['offset_ns']
        assert row['scheduled_time_ns'] <= row['completion_time_ns'] <= raw['observation_end_ns']
        latency[task].append((row['completion_time_ns'] - row['scheduled_time_ns']) / 1e6)
        q = quality[rid]
        assert q['task'] == task and q['denominator'] > 0 and q['numerator'] >= 0
        assert all(type(q[k]) in (int, float) and math.isfinite(q[k])
                   for k in ('numerator', 'denominator'))
        numerators[task].append(q['numerator'])
        denominators[task].append(q['denominator'])
    rates = plan['workload_rps']
    assert set(latency) == set(numerators) == set(denominators) == set(rates)
    total = sum(rates.values())
    refs = plan['latency_reference_ms']
    samples = raw['board_samples']
    aliases = set(raw['resident_aliases'])
    assert all(not s['foreign_pids'] and s['owned_used_mib'] >= 0 for s in samples)
    assert all(set(s['owned_aliases']) == aliases for s in samples if s['time_ns'] >= raw['startup_end_ns'])
    peak = max(s['owned_used_mib'] for s in samples)
    startup_j = joules(samples, raw['startup_begin_ns'], start)
    run_j = joules(samples, start, end)
    within = sum(r['completion_time_ns'] <= end for r in rows)
    terms = {
        'Q': sum(rates[t] / total * math.fsum(numerators[t]) / math.fsum(denominators[t]) for t in rates),
        'T': 1 - within / len(rows),
        'L': sum(rates[t] / total * math.fsum(latency[t]) / len(latency[t]) / refs[t] for t in rates),
        'P': sum(rates[t] / total * p95(latency[t]) / refs[t] for t in rates),
        'M': 1.110854 * peak / 49140,
        'E': (run_j / 600 + startup_j / 3600) / 450,
    }
    score = audit['six_term_score']
    for key, value in terms.items():
        close(value, score['terms'][key])
    utility = -sum(terms.values()) / 6
    close(utility, score['diagnostic_U'])
    assert score['completed_within_window'] == within and score['completed'] == 1200
    for task, values in latency.items():
        close(p95(values), score['per_task'][task]['p95_e2e_ms'])
    close(startup_j, score['energy']['startup_joules'])
    close(run_j, score['energy']['run_joules'])
    assert score['memory']['safe_factor_application_count'] == 1
    direct_refs = [audit_ref, plan_ref, seal_ref, raw_ref, scored_ref]
    for ref in direct_refs:
        assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256']
    return {
        'schema': 'gsdeploy.F8_native_direct_arithmetic_review.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_id': candidate, 'checks_passed': True,
        'requested': 1200, 'completed': 1200, 'quality_count': len(quality),
        'completed_within_window': within, 'terms': terms, 'U': utility,
        'p95_e2e_ms': {t: p95(v) for t, v in latency.items()},
        'owned_sampled_peak_mib': peak, 'startup_joules': startup_j, 'measurement_joules': run_j,
        'telemetry_samples': len(samples),
        'deadline_to_observation_end_seconds': (raw['observation_end_ns'] - end) / 1e9,
        'original_stability_passed': audit['original_stability']['original_stability_passed'],
        'runtime_errors': [], 'quality_errors': [], 'telemetry_errors': [], 'shutdown_normal': True,
        'source_receipts': direct_refs,
        'source': load_ref_source(),
        'quality_independently_rescored_by_this_review': False,
        'source_closure_reverified_by_this_review': False,
        'capacity_qualification_performed': False, 'final_gain_eligible': False,
    }


def load_ref_source():
    path = Path(__file__).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, choices=ORDER)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve existing review; no overwrite')
    result = review(args.candidate)
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('candidate_id', 'checks_passed', 'completed', 'U')}))


if __name__ == '__main__':
    main()
