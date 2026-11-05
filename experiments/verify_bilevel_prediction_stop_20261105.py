"""Six prespecified regression runs against sealed prediction trajectories."""
import copy
import difflib
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.compare_crossdomain_strategies_20261105 import run_one, audit, ARCHIVE, PROTOCOL
from experiments.bilevel_scalability_benchmark import read, receipt, save

OUT=ROOT/'artifacts/bilevel-prediction-budget-stop-20261105'


def main():
    plan=read(OUT/'plan.json');cfg=read(PROTOCOL);base=read(ARCHIVE/'plan.json')['protocol']
    assert cfg['budget']==plan['budget']
    inputs=[plan['snapshot'],*plan['archived_runs']]
    for r in inputs:assert receipt(r['path'])==r
    sources=[receipt(p) for p in [Path(__file__),
        ROOT/'gsdeploy/continuous/activation_bilevel.py', ROOT/'tests/test_bilevel_prediction_stop.py',
        ROOT/'experiments/compare_crossdomain_strategies_20261105.py',
        *sorted((ROOT/'gsdeploy/continuous').glob('*.py'))]]
    save(OUT/'regression-sources.json',dict(sources=sources,inputs=inputs,started_after_tests=True))
    keys=('index','phase','deployment','routing_qps','U','eligible')
    rows=[];errors=[];checked=0
    for n in plan['sizes']:
        for seed in plan['seeds']:
            raw,problem=run_one(n,'BO-MADS',seed,cfg,base)
            old=read(ROOT/f'artifacts/crossdomain-strategy-comparison-20261105/{n}-BO-MADS-{seed}.json')
            save(OUT/f'{n}-BO-MADS-{seed}.json',raw)
            equivalent=([{k:r[k] for k in keys} for r in raw['calls']]==
                        [{k:r[k] for k in keys} for r in old['calls']])
            old_at_cap=next(r['outer_evaluations'] for r in old['convergence'] if r['full_prediction_calls']==64)
            try:
                assert raw['status']=='PREDICTIONS_LIMIT'
                assert equivalent and raw['initial_signature']==old['initial_signature']
                assert raw['budget']['full_prediction_calls']==64
                assert raw['budget']['outer_evaluations']==old_at_cap
                assert raw['winner']['best_feasible_utility']==old['winner']['best_feasible_utility']
                assert raw['winner']['deployment']==old['winner']['deployment']
                assert raw['winner']['routing_shares']==old['winner']['routing_shares']
                checked+=audit(raw,problem,read(ARCHIVE/f'reference-{n}.json'),dict(base,budget=cfg['budget']))
            except AssertionError as exc:errors.append(dict(size=n,seed=seed,error=str(exc) or 'REGRESSION_FAILURE'))
            row=dict(size=n,seed=seed,status=raw['status'],trajectory_identical=equivalent,
                U_before=old['winner']['best_feasible_utility'],U_after=raw['winner']['best_feasible_utility'],
                predictions_before=old['budget']['full_prediction_calls'],predictions_after=raw['budget']['full_prediction_calls'],
                outer_before=old['budget']['outer_evaluations'],outer_after=raw['budget']['outer_evaluations'],
                seconds_before=old['budget']['elapsed_seconds'],seconds_after=raw['budget']['elapsed_seconds'],
                final_prediction_seconds=raw['calls'][-1]['elapsed_seconds'])
            rows.append(row);print(row,flush=True)
    assert len(rows)==plan['maximum_new_searches']
    for r in sources+inputs:assert receipt(r['path'])==r
    medians={str(n):{k:statistics.median(r[k] for r in rows if r['size']==n)
        for k in ('seconds_before','seconds_after','outer_before','outer_after')} for n in plan['sizes']}
    save(OUT/'comparison.json',dict(rows=rows,medians=medians,audit_errors=errors,
        tests_passed=67,new_searches=6,checked_inner_results=checked,paid_predictions=384,
        new_gpu_runs=0,predictor_refits=0,timing_scope='descriptive before/after batches, not a strict speedup experiment'))
    previous=Path(plan['snapshot']['path']).read_text().splitlines(keepends=True)
    current=(ROOT/'gsdeploy/continuous/activation_bilevel.py').read_text().splitlines(keepends=True)
    (OUT/'implementation.diff').write_text(''.join(difflib.unified_diff(previous,current,
        fromfile='activation_bilevel.before.py',tofile='gsdeploy/continuous/activation_bilevel.py')))
    save(OUT/'completion.json',dict(status='COMPLETE' if not errors else 'REGRESSION_FAILURE',
        audit_errors=errors,source_integrity=True,remaining_automatic_experiments=[]))
    if errors:raise SystemExit('Regression failures preserved')


if __name__=='__main__':main()
