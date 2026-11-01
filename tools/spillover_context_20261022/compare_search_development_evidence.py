"""Compare searched numerical development scores with authorized old evidence.

This is a post-search development discrepancy check, not independent validation.
Reads the explicitly authorized terminal comparison report, never live raws.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.spillover_context_20261022.search_development import identity,write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=ROOT/'artifacts/static-spillover-study-20261022/staged-continuation-v1/execution/all-six-comparison/report.json'
    report=json.loads(source.read_text());inputs={str(source):hashlib.sha256(source.read_bytes()).hexdigest()};scores={}
    for seed in [107,108]:
        path=a.base/f'plain-dfs-seed{seed}'/'evaluations.jsonl';inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        scores[seed]={r['identity']:r for r in (json.loads(line) for line in path.read_text().splitlines())}
    comparisons=[]
    for o in report['observations']:
        if o['audit_status']!='VERIFIED_TERMINAL_EVIDENCE':raise ValueError('Only authorized audited terminal evidence allowed')
        d=scores[o['seed']][identity(o['plan'])];run=d['runs'][0];u=o['utility'];actual={t:u[t+'_norm'] for t in 'QTLPME'}
        comparisons.append(dict(label=o['label'],seed=o['seed'],plan=o['plan'],observed_U=u['U'],
            development_diagnostic_U=d['diagnostic_objective'],observed_terms=actual,development_terms=run['terms'],
            development_minus_observed_terms={t:run['terms'][t]-actual[t] for t in 'QTLPME'},
            observed_eligible=u['eligible'],development_planning_feasible=d['diagnostic_feasible'],
            reused_development_evidence=True,independent_validation=False))
    for seed in [107,108]:
        rows=[r for r in comparisons if r['seed']==seed]
        for field in ['observed_U','development_diagnostic_U']:
            for rank,row in enumerate(sorted(rows,key=lambda r:-r[field]),1):row[field+'_rank']=rank
    grouped={}
    for row in comparisons:grouped.setdefault(identity(row['plan']),[]).append(row)
    means=[]
    for key,rows in grouped.items():
        if {r['seed'] for r in rows}!={107,108}:raise ValueError('Paired observed evidence incomplete')
        means.append(dict(plan=rows[0]['plan'],observed_mean_U=sum(r['observed_U'] for r in rows)/2,
            development_mean_U=sum(r['development_diagnostic_U'] for r in rows)/2))
    for field in ['observed_mean_U','development_mean_U']:
        for rank,row in enumerate(sorted(means,key=lambda r:-r[field]),1):row[field+'_rank']=rank
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'report.json',dict(comparisons=comparisons,paired_means=means,inputs=inputs,
        conclusion='Development numerical ranking differs from already observed107/108 rankings; the mathematical optimum does not certify physical optimality.',
        formal_U=None,physical_optimality=False,independent_validation=False,source_results_previously_observed=True))
    write(a.output/'manifest.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=inputs,
        files={'report.json':hashlib.sha256((a.output/'report.json').read_bytes()).hexdigest()}))

if __name__=='__main__':main()
