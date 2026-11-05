"""One fixed six-run scheduler comparison against already sealed results."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.bilevel_scalability_benchmark import (ControlledOptimizer, QueueCostOracle,
    build_problem, read, receipt, save, summarize, validate_rows)
from gsdeploy.continuous.budget_scheduled_bilevel import BudgetScheduledBilevelOptimizer
from gsdeploy.continuous import ContinuousRoutingEvaluator
from gsdeploy.continuous.contracts import Budget

PROTOCOL = ROOT/'configs/bilevel_scheduler_improvement_20261105.json'


class ScheduledControlledOptimizer(ControlledOptimizer, BudgetScheduledBilevelOptimizer):
    """Archive's exact initializer + new scheduler, using cooperative super()."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    # Only the pre-change entrypoint snapshot may exist before this protocol.
    if out.exists() and {p.name for p in out.iterdir()}-{'prechange.json','solve_bilevel_v3.before.py'}:
        raise FileExistsError(out)
    out.mkdir(exist_ok=True)
    cfg = read(PROTOCOL)
    archive = ROOT/cfg['archive']
    previous = read(archive/'plan.json')
    baseline_protocol = previous['protocol']
    assert cfg['budget'] == baseline_protocol['budget']
    for r in previous['sources']:
        assert receipt(r['path']) == r, r['path']
    paths = [Path(__file__), PROTOCOL, ROOT/'gsdeploy/continuous/budget_scheduled_bilevel.py',
        ROOT/'tests/test_bilevel_scheduler.py', *[Path(r['path']) for r in previous['sources']]]
    sources = [receipt(p) for p in paths]
    old_results = [archive/f'{n}-BO-MADS-{s}.json' for n in cfg['sizes'] for s in cfg['seeds']]
    refs = [archive/f'reference-{n}.json' for n in cfg['sizes']]
    save(out/'plan.json',dict(protocol=cfg, sources=sources, archived_inputs=[receipt(p) for p in old_results+refs]))
    rows, checked = [], 0
    for n in cfg['sizes']:
        problem = build_problem(n, baseline_protocol)
        reference = read(archive/f'reference-{n}.json')
        for seed in cfg['seeds']:
            b = cfg['budget']
            e = ContinuousRoutingEvaluator(problem, QueueCostOracle(),
                budget=Budget(b['seconds'], b['predictions'], b['routes'], b['milp']))
            optimizer = ScheduledControlledOptimizer(e, 'BO-MADS', seed, baseline_protocol)
            assert all(getattr(optimizer,k) == v for k,v in cfg['scheduler'].items())
            result = optimizer.solve()
            result.update(method='BO-MADS-balanced',size=n, initial_signature=optimizer.initial_signature,
                inner_evaluations=optimizer.inner.history,search_domain_dimension=e.domain.dimension)
            save(out/f'{n}-balanced-{seed}.json', result)
            old = read(archive/f'{n}-BO-MADS-{seed}.json')
            assert result['initial_signature'] == old['initial_signature']
            assert result['problem_sha256'] == old['problem_sha256'] and result['oracle_version'] == old['oracle_version']
            checked += validate_rows(result,problem,reference,baseline_protocol)
            row = summarize(result,reference,problem)
            row.update(legacy_U=old['winner']['best_feasible_utility'],
                delta_U=row['U']-old['winner']['best_feasible_utility'], legacy_budget=old['budget'],
                scheduler=result['scheduler'])
            rows.append(row)
            print(json.dumps({k:row[k] for k in ('size','seed','U','delta_U','bo_ei_proposals','budget')},ensure_ascii=False),flush=True)
    assert len(rows) == cfg['maximum_new_runs']
    correct = all(r['bo_ei_proposals']>0 and r['scheduler']['charged_repeats']<=r['scheduler']['repeat_limit'] for r in rows)
    medians = {str(n):dict(legacy=statistics.median(r['legacy_U'] for r in rows if r['size']==n),
        balanced=statistics.median(r['U'] for r in rows if r['size']==n)) for n in cfg['sizes']}
    release = correct and all(v['balanced'] >= v['legacy']-1e-8 for v in medians.values())
    assert all(receipt(r['path'])==r for r in sources)
    save(out/'comparison.json',dict(rows=rows, medians=medians, release_as_default=release,
        scheduling_correct=correct, inner_results_checked=checked, new_searches=len(rows),
        legacy_searches_reused=len(rows), new_gpu_runs=0, predictor_refits=0,
        evidence='same manufactured convex queue-cost benchmark; no real seven-model qualification'))
    save(out/'completion.json',dict(status='FIXED_SCHEDULER_COMPARISON_COMPLETE',
        new_searches=6, inner_results_checked=checked, scheduling_correct=correct,
        release_as_default=release, remaining_automatic_experiments=[]))


if __name__ == '__main__':
    main()
