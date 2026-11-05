"""Describe a complete ABBA heap comparison without granting qualification.

Report aggregate path latency separately from per-request ratios, and retain
the two matched rounds separately: 320 requests are not 320 independent runs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_activation_semantic import summarize


def describe(report):
    if report['schema'] != 'gsdeploy.host_trim_runtime_diagnostic.v1':
        raise ValueError('Complete runtime comparison report required')
    names = {'control-round1', 'trim-round1', 'trim-round2', 'control-round2'}
    if set(report['sources']) != names:
        raise ValueError('All four ABBA captures required')
    paths = {}
    round_counts = defaultdict(int)
    for path, data in sorted(report['paired_latency'].items()):
        rows = data['observations']
        if {r['round'] for r in rows} != {1, 2}:
            raise ValueError('Both matched rounds required for every path')
        groups = {}
        for label, selected in [('pooled', rows)] + [
                (f'round{i}', [r for r in rows if r['round'] == i]) for i in (1, 2)]:
            left = summarize([r['control_ms'] for r in selected])
            right = summarize([r['trim_ms'] for r in selected])
            if left['n'] != right['n'] or left['mean'] <= 0 or left['p95'] <= 0:
                raise ValueError('Positive matched latency observations required')
            groups[label] = dict(control_ms=left, trim_ms=right,
                mean_change_fraction=right['mean'] / left['mean'] - 1,
                p95_change_fraction=right['p95'] / left['p95'] - 1,
                paired_ratio=summarize([r['trim_over_control'] for r in selected]))
        for i in (1, 2):
            keys = [json.dumps(r['key'], sort_keys=True) for r in rows if r['round'] == i]
            if len(keys) != len(set(keys)):
                raise ValueError('Duplicate matched input')
            round_counts[i] += len(keys)
        paths[path] = groups
    if len(paths) != 8 or dict(round_counts) != {1: 320, 2: 320}:
        raise ValueError('Exact eight-path 320-request cohort required in each round')
    bands = defaultdict(list)
    for band in report['bands']:
        bands[band['mode'], band['family'], band['task'], band['band']].append(band)
    phase_summary = []
    for key, rows in sorted(bands.items()):
        phase_summary.append(dict(mode=key[0], family=key[1], task=key[2], band=key[3],
            repeated_inputs=len(rows), raw_constant_minimax=summarize([
                r['raw_constant_minimax'] for r in rows]),
            raw_constant_minimax_over_point2=sum(r['raw_constant_minimax'] > .2 for r in rows),
            worst_inputs=sorted(rows, key=lambda r: r['raw_constant_minimax'], reverse=True)[:3]))
    receipts = defaultdict(list)
    for receipt in report['trim_receipts']:
        receipts[receipt['capture']].append(receipt)
    if set(receipts) != names or any(len(rows) != 5 or len({r['alias'] for r in rows}) != 5
                                    for rows in receipts.values()):
        raise ValueError('Five unique worker lifecycle receipts per capture required')
    memory = {name: dict(rss_reduction_mib=sum(r['rss_reduction_mib'] for r in rows),
                        procedure_ms=sum(r['procedure_ms'] for r in rows),
                        workers=[{k: r[k] for k in ('alias', 'rss_reduction_mib', 'procedure_ms')}
                                 for r in rows]) for name, rows in sorted(receipts.items())}
    return dict(schema='gsdeploy.host_trim_runtime_summary.v1', sources=report['sources'],
        issues=report['issues'], output_changes=report['output_changes'], paths=paths,
        phase_summary=phase_summary, reclamation=memory, episodes=report['episodes'],
        physical_predictor_qualified=False, formal_utility_qualified=False,
        interpretation=[
            'Negative latency changes indicate improvement; no observer-overhead rejection is applied to a runtime intervention.',
            'Two matched rounds are descriptive development evidence, not independent predictor qualification or a population confidence interval.',
            'Raw constant minimax is not root-order-adjusted; original work gates must be rerun on fresh training.',
            'Procedure time is included in priming. RSS reduction does not identify the kernel mechanism or establish an energy benefit.',
            'Keep all regressions, output changes and failures. No candidate is admitted by this summary.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    content = args.report.read_bytes()
    result = describe(json.loads(content))
    result['report_receipt'] = dict(path=str(args.report.resolve()), sha256=hashlib.sha256(content).hexdigest())
    result['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output.resolve()), issues=result['issues'],
        output_changes=len(result['output_changes']), paths={p: d['pooled'] for p, d in result['paths'].items()})))


if __name__ == '__main__':
    main()
