"""Check cross-run pairing and provenance without changing measurement artifacts."""

import argparse
from collections import Counter
import json
from pathlib import Path

from .routing_sweep import OUT, ROOT, SEEDS, SHARES, audit, protocol_hashes, sha256, write_json


def validate_dataset(output):
    plan = json.loads((output / 'plan.json').read_text())
    failures, runs, paired = [], [], {}
    if plan['protocol_hashes'] != protocol_hashes():
        failures.append('sealed protocol/data hashes changed')
    expected_pairs = {(seed, share) for seed in SEEDS for share in SHARES}
    actual_pairs = []
    reference_source = None
    for spec in plan['runs']:
        path = Path(spec['result'])
        if not path.exists():
            failures.append(f"missing result: {spec['run_id']}")
            continue
        raw = json.loads(path.read_text())
        cfg_path = Path(spec['config'])
        cfg = json.loads(cfg_path.read_text())
        checked = audit(raw, cfg)
        prefix = spec['run_id'] + ': '
        failures.extend(prefix + issue for issue in checked['issues'])
        if sha256(cfg_path) != spec['config_hash']:
            failures.append(prefix + 'config hash changed')
        source = raw['source']
        if reference_source is None:
            reference_source = source
        elif source != reference_source:
            failures.append(prefix + 'measurement source/environment differs')
        for field, hash_field in [('model_profile', 'model_profile_hash'), ('task_pool', 'task_pool_hash')]:
            rel = str(Path(source[field]).relative_to(ROOT))
            if source[hash_field] != plan['protocol_hashes'][rel]:
                failures.append(prefix + field + ' differs from planned source')
        rows = raw['request_records']
        if len({r['request_id'] for r in rows}) != len(rows):
            failures.append(prefix + 'duplicate request IDs')
        origin = raw['measurement_window']['start_time_ns']
        signature = [(r['task'], r['pool_id'], r['scheduled_time_ns'] - origin) for r in rows]
        seed = spec['seed']
        if seed in paired and signature != paired[seed]:
            failures.append(prefix + 'paired arrivals/sample sequence differs across routing')
        paired.setdefault(seed, signature)
        actual_pairs.append((seed, spec['specialist_share']))
        runs.append(dict(run_id=spec['run_id'], raw_sha256=sha256(path),
                         offered=len(rows), completed=checked['actual_completed_requests'],
                         per_task_offered=dict(Counter(r['task'] for r in rows)),
                         clean_shutdown=checked['clean_shutdown'],
                         overloaded_or_unstable=checked['overloaded_or_unstable']))
    if set(actual_pairs) != expected_pairs or len(actual_pairs) != len(expected_pairs):
        failures.append('dataset does not contain exactly five routings by three seeds')
    if len(paired) == len(SEEDS) and len({tuple(s) for s in paired.values()}) != len(SEEDS):
        failures.append('independent seeds produced identical arrivals/sample signatures')
    result = dict(passed=not failures, failures=failures, num_runs=len(runs),
                  offered_requests=sum(r['offered'] for r in runs),
                  completed_requests=sum(r['completed'] for r in runs),
                  all_clean_shutdown=all(r['clean_shutdown'] for r in runs),
                  overloaded_or_unstable=any(r['overloaded_or_unstable'] for r in runs),
                  paired_arrivals_and_samples_passed=not any('paired' in f for f in failures),
                  gpu_snapshot=reference_source['gpu_snapshot'] if reference_source else None,
                  runs=runs)
    write_json(output / 'dataset_validation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    result = validate_dataset(parser.parse_args().output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
