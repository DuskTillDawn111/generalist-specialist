"""Independent Cartesian identity/count and utility audit of completed searches.

For this local domain each task has at most two supported families. Rational
rows are generated here without the solver route-domain implementation.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.spillover_context_20261022.development_oracle import DevelopmentOracle,sha
from tools.spillover_context_20261022.development_bounds import DevelopmentBounds

OPTIONS={'generalist_2':[0,1],'text':[0,1,2],'ocr':[0,1],'image':[0,1],'asr':[0,1,2]}


def key(plan,rates):
    return (tuple(sorted(plan['deployment'].items())),tuple((t,tuple(sorted(
        (m,round(q/rates[t],12)) for m,q in row.items() if q>0))) for t,row in sorted(plan['routing'].items())))


def identities(catalog,rates):
    out=set()
    for counts in itertools.product(*OPTIONS.values()):
        x={m:n for m,n in zip(OPTIONS,counts) if n};choices=[]
        for task,rate in sorted(rates.items()):
            supported=sorted(m for m in x if task in catalog['models'][m]['supported_tasks'])
            if len(supported)>2:raise ValueError('Independent audit only proves the declared pair domain')
            if len(supported)==0:rows=[]
            elif len(supported)==1:rows=[{supported[0]:rate}]
            else:
                d=32 if task=='text' else 2
                rows=[{m:rate*float(f) for m,f in zip(supported,[Fraction(i,d),Fraction(d-i,d)]) if f>0}
                      for i in range(d+1)]
            choices.append(rows)
        for rows in itertools.product(*choices):
            out.add(key(dict(deployment=x,routing=dict(zip(sorted(rates),rows))),rates))
    return out


def read_evaluations(path,rates,weights):
    rows={}
    for line in Path(path).read_text().splitlines():
        r=json.loads(line);k=key(r['plan'],rates)
        if k in rows:raise ValueError('Duplicate complete identity: '+str(path))
        if r['formal_U'] is not None or r['physical_optimality'] is not False:
            raise ValueError('Development evidence promoted to formal evidence')
        scores=[]
        for run in r['runs']:
            terms=run['terms']
            if run['diagnostic_U'] is not None:
                expected=-math.fsum(weights[t]*terms[t] for t in 'QTLPME')
                if not math.isclose(expected,run['diagnostic_U'],abs_tol=1e-11):raise ValueError('Six-term utility mismatch')
            scores.append(run['diagnostic_U'])
        u=sum(scores)/len(scores) if all(v is not None for v in scores) else None
        if (u is None)!=(r['diagnostic_objective'] is None) or (u is not None and abs(u-r['diagnostic_objective'])>1e-10):
            raise ValueError('Mean objective mismatch')
        rows[k]=r
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();oracle=DevelopmentOracle();rates=oracle.config['workload'];weights=oracle.config['weights']
    expected=identities(oracle.catalog,rates)
    if len(expected)!=7508:raise ValueError('Independent domain count mismatch')
    opened={};plain={};runs=[];bound_errors=[]
    for seed in [107,108]:
        d=args.base/f'plain-dfs-seed{seed}';path=d/'evaluations.jsonl'
        rows=read_evaluations(path,rates,weights);opened[str(path.resolve())]=sha(path)
        if set(rows)!=expected:raise ValueError('Missing or extra DFS identities')
        plain[seed]=rows
        bounds=DevelopmentBounds(oracle,[seed])
        maximum=-math.inf
        for row in rows.values():
            if row['diagnostic_objective'] is not None:
                floor=-bounds.upper(row['plan']['deployment'],row['plan']['routing'],0)
                violation=floor+row['diagnostic_objective'];maximum=max(maximum,violation)
                if violation>1e-9:bound_errors.append(dict(seed=seed,violation=violation,plan=row['plan']))
        runs.append(dict(seed=seed,identities=len(rows),max_derived_cost_floor_minus_actual_cost=maximum,
            unknown=sum(r['diagnostic_objective'] is None for r in rows.values()),
            planning_feasible=sum(r['diagnostic_feasible'] for r in rows.values())))
    if bound_errors:raise ValueError('Development bound exceeds complete objective cost: '+str(bound_errors[:1]))
    comparisons=[]
    for path in sorted(args.base.glob('*/report.json')):
        r=json.loads(path.read_text())
        if 'algorithm' not in r or 'result' not in r:continue
        opened[str(path.resolve())]=sha(path);seeds=r['seeds'];s=r['result'];winner=s['winner']
        if r['domain']['complete_plans']!=7508:raise ValueError('Comparison domain differs')
        if r['mathematical_sources_before']!=r['mathematical_sources_after']:raise ValueError('Source changed during search')
        eligible=[]
        for k in expected:
            rr=[plain[seed][k] for seed in seeds]
            if all(v['diagnostic_feasible'] and v['diagnostic_objective'] is not None for v in rr):
                eligible.append((sum(v['diagnostic_objective'] for v in rr)/len(rr),k))
        optimum=max(score for score,k in eligible)
        equal=winner is not None and abs(winner['score']-optimum)<=1e-8
        if s['optimality_certified'] and not equal:raise ValueError('Claimed optimum disagrees with complete independent DFS')
        if winner:
            k=key(winner,rates)
            actual=sum(plain[seed][k]['diagnostic_objective'] for seed in seeds)/len(seeds)
            if abs(actual-winner['score'])>1e-8:raise ValueError('Incumbent score disagrees with common evaluator')
        comparisons.append(dict(path=str(path.resolve()),algorithm=r['algorithm'],seeds=seeds,
            bounds=r['bound_variant'],wall_seconds=r['wall_seconds'],evaluations=r['unique_complete_evaluations'],
            status=s['status'],search_completed=s['search_completed'],
            diagnostic_math_optimality_certified=s['optimality_certified'],diagnostic_winner_score=winner['score'] if winner else None,
            full_DFS_diagnostic_optimum=optimum,objective_matches_full_DFS=equal,
            winner=winner and {k:winner[k] for k in ['deployment','routing','score']},
            formal_optimality_certified=False))
    oracle.verify_inputs();args.output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='gsdeploy.independent_development_search_audit.v1',
        independent_rational_cartesian_domain_count=len(expected),plain_DFS=runs,
        all_complete_leaf_bounds_checked=True,derived_bound_violations=0,comparisons=comparisons,
        source_inputs=opened,oracle_inputs=oracle.provenance(),formal_U=None,
        full_seven_pool_optimality=False,physical_stability_verified=False,GPU_started=False)
    dest=args.output/'report.json';dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(dict(script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        inputs=opened,files={'report.json':sha(dest)}),indent=2)+'\n')
    print(json.dumps(dict(domain=len(expected),all_leaf_bounds_checked=True,comparisons=len(comparisons))))


if __name__=='__main__':main()
