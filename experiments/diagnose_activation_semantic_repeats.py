"""Separate unchanged-output subsets from natural repeat variation, without changing observer gates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.analyze_activation_semantic import output_identity, summarize
from gsdeploy.forecasting.semantic_profiles import split_semantic_native


def keyed_rows(bundle):
    rows = {}
    counts = defaultdict(int)
    for row in bundle['raw']['request_records']:
        key = (row['profile_job_id'], row['model'], row['task'], row['pool_id'])
        index = counts[key]
        counts[key] += 1
        rows[(*key, index)] = row
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    names = ['control-round1', 'semantic-round1', 'semantic-round2', 'control-round2']
    bundles = {name: load_bundle(args.base/name) for name in names}
    keyed = {name: keyed_rows(b) for name, b in bundles.items()}
    if any(set(rows) != set(keyed[names[0]]) for rows in keyed.values()):
        raise ValueError('Repeat diagnostic input/job cohorts differ')
    comparisons = []
    for left, right in [('control-round1', 'control-round2'), ('semantic-round1', 'semantic-round2'),
                         ('control-round1', 'semantic-round1'), ('control-round2', 'semantic-round2')]:
        groups = defaultdict(lambda: dict(all=[], same_output=[], changed_output=[]))
        differences = []
        for key, old in keyed[left].items():
            new = keyed[right][key]
            family = bundles[left]['raw']['resident_aliases'][old['model']]
            group = groups[family+'/'+old['task']]
            old_ms = (old['completion_time_ns']-old['scheduled_time_ns'])/1e6
            new_ms = (new['completion_time_ns']-new['scheduled_time_ns'])/1e6
            error = abs(new_ms-old_ms)/old_ms
            same = output_identity(old['worker_response']) == output_identity(new['worker_response'])
            group['all'].append(error)
            group['same_output' if same else 'changed_output'].append(error)
            if not same:
                differences.append(dict(job=key[0], alias=key[1], task=key[2], pool_id=key[3],
                    generated_tokens=[old['worker_response'].get('generated_tokens'),
                                      new['worker_response'].get('generated_tokens')]))
        comparisons.append(dict(left=left, right=right,
            path_metrics={k: {s: summarize(v) for s, v in group.items()} for k, group in groups.items()},
            output_differences=differences))
    layouts = []
    for name in ['semantic-round1', 'semantic-round2']:
        counts = defaultdict(list)
        for row in keyed[name].values():
            layout = split_semantic_native(row)
            family = bundles[name]['raw']['resident_aliases'][row['model']]
            roles = defaultdict(float)
            for span in layout:
                roles[span['semantic_role']] += (span['end_ns']-span['start_ns'])/1e6
            counts[family+'/'+row['task']].append(dict(pool_id=row['pool_id'], job=row['profile_job_id'],
                segments=len(layout), host_duration_ms_by_role=dict(roles)))
        layouts.append(dict(run=name, paths=dict(counts)))
    report = dict(schema='gsdeploy.semantic_repeat_diagnosis.v1',
        sources={name: b['source'] for name, b in bundles.items()}, comparisons=comparisons,
        semantic_layouts=layouts, original_acceptance_criteria_unchanged=True,
        physical_predictor_qualified=False,
        limitations=['same-output conditioning is post-completion diagnosis; forbidden as an online predictor feature',
                    'different request ordering and histories prevent causal attribution of deviations to instrumentation',
                    'full semantic sequences contain eventual decode stopping; not fixed future templates',
                    'this supplement never overrides the separately frozen observer pass/fail decision'])
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(output=str(args.output), comparisons=len(comparisons))))


if __name__ == '__main__':
    main()
