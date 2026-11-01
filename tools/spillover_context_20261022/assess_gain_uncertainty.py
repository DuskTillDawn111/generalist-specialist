"""Run-level uncertainty of already audited GPU pairs; never rescoring utility.

The independent unit is a whole seed/run pair. Request counts do not supply
degrees of freedom. A Student interval is descriptive under an explicit IID,
approximately normal paired-effect assumption; two serial pairs cannot verify it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from scipy.stats import t


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(report):
    groups = {}
    for row in report['observations']:
        if row['audit_status'] != 'VERIFIED_TERMINAL_EVIDENCE':
            raise ValueError('Only independently audited terminal evidence permitted')
        if not row['utility']['eligible']:
            raise ValueError('Ineligible observations require explicit separate accounting')
        plan = row['plan']
        if plan['deployment'] != dict(text=2, generalist_2=1, ocr=1):
            continue
        fraction = plan['routing']['text'].get('generalist_2', 0) / 1.85
        if min(abs(fraction), abs(fraction - 1/32)) > 1e-12:
            raise ValueError('Unexpected fine-route condition')
        condition = 'spillover_1of32' if fraction else 'no_spillover'
        pair = groups.setdefault(row['seed'], {})
        if condition in pair:
            raise ValueError('Duplicate run condition; do not average away replicate identity')
        pair[condition] = row
    pairs = []
    for seed, pair in sorted(groups.items()):
        if set(pair) != {'no_spillover', 'spillover_1of32'}:
            raise ValueError('Incomplete predeclared pair')
        a, b = pair['no_spillover'], pair['spillover_1of32']
        if a['trace_sha256'] != b['trace_sha256'] or a['cohort_sha256'] != b['cohort_sha256']:
            raise ValueError('Unmatched arrival/input cohort')
        contributions = {term: b['utility'][term+'_contribution'] - a['utility'][term+'_contribution']
                         for term in 'QTLPME'}
        delta = b['utility']['U'] - a['utility']['U']
        if abs(sum(contributions.values()) - delta) > 1e-10:
            raise ValueError('Original six-term contribution identity failed')
        pairs.append(dict(seed=seed, no_spillover_U=a['utility']['U'],
                          spillover_U=b['utility']['U'], delta_U=delta,
                          delta_original_contributions=contributions))
    values = [p['delta_U'] for p in pairs]
    if len(values) < 2:
        raise ValueError('At least two whole independent run pairs required')
    mean = statistics.mean(values)
    se = statistics.stdev(values) / len(values)**.5
    margin = float(t.ppf(.975, len(values)-1)) * se
    return dict(schema='gsdeploy.whole_pair_gain_uncertainty.v1', pairs=pairs,
        independent_units=len(values), request_level_pseudoreplication=False,
        descriptive_mean_delta_U=mean, descriptive_sample_sd=statistics.stdev(values),
        minimum_observed_delta_U=min(values), maximum_observed_delta_U=max(values),
        observed_direction_consistent=all(v > 0 for v in values) or all(v < 0 for v in values),
        paired_student_95_interval=dict(lower=mean-margin, upper=mean+margin,
            assumption='IID approximately normal whole-run paired effects; not established by two serial pairs',
            proves_stable_gain=False),
        leave_one_seed_out_means=[dict(omitted_seed=p['seed'], mean_delta_U=statistics.mean(
            q['delta_U'] for q in pairs if q['seed'] != p['seed'])) for p in pairs],
        stable_gain_proven=False, utility_contract_changed=False,
        selection_policy='Search on development inputs; freeze proposal and strong alternatives before new GPU outcomes. Evaluate all declared conditions. Report original U, feasibility, paired effects, and uncertainty.',
        limitations=['Seeds107/108 and their outcomes have already been used for development.',
            'Serial execution leaves time drift and GPU interference confounded with route effects.',
            'Positive mean of two pairs does not establish expected or uniform benefit.',
            'Changing a workload or route after seeing these outcomes creates a new development hypothesis.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = summarize(json.loads(args.audit.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    result['source'] = dict(path=str(args.audit.resolve()), sha256=sha(args.audit))
    (args.output/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (args.output/'README.zh.md').write_text(
        '# 当前分流收益的不确定性\n\n'
        f"完整运行配对数 {result['independent_units']}，平均 ΔU={result['descriptive_mean_delta_U']:.9f}；"
        f"逐配对范围 [{result['minimum_observed_delta_U']:.9f}, {result['maximum_observed_delta_U']:.9f}]。\n\n"
        '两个开发种子的方向不一致，留一后均值随保留种子反转。不能将千余请求当千余个独立实验。'
        'JSON列出原六项贡献及在明确IID/正态假设下的描述性Student区间；两次串行配对不能验证这些假设。\n\n'
        '继续自主搜索和同域DFS；先冻结提出的部署、路由及强替代，再做新运行配对，全部条件保留。'
        '原质量、效用权重、rho和稳定性规则均保持不变。\n')
    files = {f.name: sha(f) for f in args.output.iterdir() if f.is_file()}
    (args.output/'manifest.json').write_text(json.dumps(dict(script=dict(path=str(Path(__file__).resolve()),
        sha256=sha(__file__)), inputs={str(args.audit.resolve()):sha(args.audit)}, files=files), indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['independent_units', 'descriptive_mean_delta_U',
        'paired_student_95_interval', 'observed_direction_consistent']}))


if __name__ == '__main__':
    main()
