"""Aggregate uninstrumented optional-bound/session ablations without reruns."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main(args):
    source, out = Path(args.source), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    summary = json.loads((source/'summary.json').read_text())
    runs = [json.loads(p.read_text()) for p in sorted(source.glob('*.json'))
            if '-r' in p.stem]
    groups = {}
    for r in runs:
        groups.setdefault((r['case'],r['scope'],r.get('route_name'),r['algorithm']),[]).append(r)
    table, timing, rounds = [], [], []
    for (case,scope,route,algorithm), rows in groups.items():
        median = lambda f: statistics.median(f(r) for r in rows)
        table.append(dict(case=case,scope=scope,route=route,algorithm=algorithm,repeats=len(rows),
            seconds=median(lambda r:r['seconds']),setup_seconds=median(lambda r:r['setup_seconds']),
            predictions=median(lambda r:r['predictions']),utility=rows[0]['utility'],
            utilities_equal=all(r['utility']==rows[0]['utility'] for r in rows),
            all_certified=all(r['certified'] for r in rows),
            outer=median(lambda r:len(r['history'])),
            milp_calls=median(lambda r:r['metrics'].get('milp_solve_calls',r['metrics'].get('milp_calls',0))),
            bound_seconds=median(lambda r:sum(h.get('bound_seconds',0) for h in r['history'])),
            endpoints=median(lambda r:r['raw'].get('activation_coverage',{}).get('endpoint_evaluations',0)),
            seeded_faces=median(lambda r:r['raw'].get('activation_coverage',{}).get('seeded_faces',0))))
    for r in runs:
        if not r['algorithm'].startswith('old_'):
            parts = r['metrics']['seconds_by_component']
            bound = sum(h.get('bound_seconds',0) for h in r['history'])
            other = r['seconds']-r['setup_seconds']-parts['prediction']-parts['milp']-bound
            assert other >= -1e-6
            timing.append(dict(case=r['case'],scope=r['scope'],route=r.get('route_name'),algorithm=r['algorithm'],repeat=r['repeat'],
                total=r['seconds'],setup=r['setup_seconds'],bounds=bound,
                milp=parts['milp'],prediction=parts['prediction'],other_python=other))
            for i,h in enumerate(r['history']):
                for j,rr in enumerate(h.get('milp_rounds',[])):
                    rounds.append(dict(case=r['case'],scope=r['scope'],route=r.get('route_name'),algorithm=r['algorithm'],
                        repeat=r['repeat'],outer_index=i,round=j,**rr))
    # Same physical inputs within every joint case. Cached/uncached runs must
    # follow identical deterministic routes until a budget interrupts one.
    for case in {r['case'] for r in runs}:
        joint=[r for r in runs if r['case']==case and r['scope']=='joint']
        assert len({r['input_sha256'] for r in joint})==1
        for repeat in {r['repeat'] for r in joint}:
            rows=[r for r in joint if r['repeat']==repeat and r['history']]
            n=min(len(r['history']) for r in rows)
            for i in range(n):
                histories=[r['history'][i] for r in rows]
                assert all(h['routing_shares']==histories[0]['routing_shares'] for h in histories)
                exact=[h for h in histories if h['certified']]
                assert all(h['best_feasible_utility']==exact[0]['best_feasible_utility'] for h in exact)
    write_csv(out/'results.csv',table)
    write_csv(out/'timing.csv',timing)
    write_csv(out/'milp-rounds.csv',rounds)
    (out/'verification.json').write_text(json.dumps(dict(runs=len(runs),
        summary_runs=summary['runs'],same_inputs=True,same_route_prefix=True,
        same_certified_inner_utility=True,timing_closes=True),indent=2))
    assert len(runs)==summary['runs']
    for scope in ('joint','fixed_route'):
        print('\n'+scope)
        for row in table:
            if row['scope']==scope:
                print(row['case'],row['route'],row['algorithm'],
                      f"{row['seconds']:.3f}s setup={row['setup_seconds']:.3f}",
                      'pred=',row['predictions'],'U=',row['utility'],
                      'outer=',row['outer'],'end=',row['endpoints'],'faces=',row['seeded_faces'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
