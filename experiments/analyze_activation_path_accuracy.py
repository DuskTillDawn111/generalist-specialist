"""Additional per-path timing gate; pooled accuracy cannot hide a weak model/task."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsdeploy.forecasting.native_validation import DEFAULT_CRITERIA, metrics


def path_accuracy(report):
    if report['criteria'] != DEFAULT_CRITERIA:
        raise ValueError('Unchanged original timing criteria required')
    groups = defaultdict(lambda: dict(expected=set(), predictions={}, runs=set()))
    for run in report['runs']:
        source, expected_sha = run['source'].rsplit(':sha256:', 1)
        content = Path(source).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_sha:
            raise ValueError('Prediction report raw receipt changed')
        raw = json.loads(content)
        if raw['run_id'] != run['run_id']:
            raise ValueError('Run identity mismatch')
        rows = {row['request_id']: row for row in raw['request_records']}
        if len(rows) != len(raw['request_records']):
            raise ValueError('Duplicate expected request identity')
        for row in rows.values():
            key = (raw['resident_aliases'][row['model']], row['task'])
            group = groups[key]
            group['expected'].add((run['run_id'], row['request_id']))
            group['runs'].add(run['run_id'])
        for batch in run['batches']:
            if batch['status'] != 'predicted':
                continue
            for prediction in batch['requests']:
                row = rows[prediction['request_id']]
                key = (raw['resident_aliases'][row['model']], row['task'])
                identity = (run['run_id'], row['request_id'])
                predictions = groups[key]['predictions']
                if identity in predictions:
                    raise ValueError('Duplicate predicted request identity')
                predictions[identity] = prediction['relative_error']
    paths = []
    for (family, task), group in sorted(groups.items()):
        observed = metrics(list(group['predictions'].values()))
        missing = sorted(group['expected']-set(group['predictions']))
        passed = (not missing and len(group['runs']) >= DEFAULT_CRITERIA['minimum_independent_runs']
            and observed['n'] > 0
            and observed['mean'] <= DEFAULT_CRITERIA['maximum_mean_e2e_relative_error']
            and observed['p95'] <= DEFAULT_CRITERIA['maximum_p95_e2e_relative_error'])
        paths.append(dict(family=family, task=task, expected_requests=len(group['expected']),
            independent_run_ids=sorted(group['runs']), latency_errors=observed,
            missing_request_ids=missing, per_path_timing_passed=passed))
    return dict(paths=paths, all_paths_timing_passed=bool(paths) and all(p['per_path_timing_passed'] for p in paths),
        original_overall_gate_passed=report.get('development_accuracy_passed', False),
        training_replay=report.get('training_replay', False), formal_utility_qualified=False,
        scope='additional timing check using unchanged mean/P95 limits on every family/task; original report unchanged')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    content = args.report.read_bytes()
    result = path_accuracy(json.loads(content))
    result.update(schema='gsdeploy.additional_per_path_timing_gate.v1', source_report=str(args.report.resolve()),
        source_report_sha256=hashlib.sha256(content).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
