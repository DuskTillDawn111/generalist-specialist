"""One prespecified batch of balanced-scheduler component ablations.

Production optimizer and benchmark oracle are imported unchanged. Full balanced
runs and continuous references are reused, never rerun. Output is append-only.
"""
import argparse
from collections import Counter
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.bilevel_scalability_benchmark import (QueueCostOracle, build_problem,
    read, receipt, save, validate_rows)
from experiments.verify_bilevel_scheduler import ScheduledControlledOptimizer
from gsdeploy.continuous import ContinuousRoutingEvaluator
from gsdeploy.continuous.contracts import Budget, digest

PROTOCOL = ROOT/'configs/bilevel_balanced_ablation_20261105.json'


def make_optimizer(size, method, seed, protocol):
    problem = build_problem(size, protocol)
    b = protocol['budget']
    evaluator = ContinuousRoutingEvaluator(problem, QueueCostOracle(),
        budget=Budget(b['seconds'], b['predictions'], b['routes'], b['milp']))
    return ScheduledControlledOptimizer(evaluator, method, seed, protocol)


def summary_row(raw, reference, full, origin):
    winner = raw['winner']
    value = winner['best_feasible_utility'] if winner else None
    base = full['winner']['best_feasible_utility']
    history = raw['inner_evaluations']
    sources = Counter(r['source'] for r in raw['convergence'])
    return dict(size=raw['size'], method=raw['method'], seed=raw['seed'], origin=origin,
        status=raw['status'], U=value, full_U=base,
        delta_U_variant_minus_full=None if value is None else value-base,
        initialization_U=raw['initial_signature'][-1]['work']['best_feasible_utility'],
        initialization_sha256=digest(raw['initial_signature']),
        regret_upper=None if value is None else reference['upper_U']-value,
        budget=raw['budget'], full_budget=full['budget'],
        prediction_ratio=raw['budget']['full_prediction_calls']/full['budget']['full_prediction_calls'],
        milp_ratio=raw['budget']['milp_solve_calls']/full['budget']['milp_solve_calls'],
        certified_evaluations=sum(r['certified'] for r in history), evaluations=len(history),
        winner_route_inner_certified=bool(winner and any(r['certified']
            and digest(r['routing_shares'])==digest(winner['routing_shares'])
            and r['best_feasible_utility']==value for r in history)),
        non_K4=bool(winner and any(abs(s*4-round(s*4))>1e-8
            for shares in winner['routing_shares'].values() for s in shares.values())),
        winner_shares=winner['routing_shares'] if winner else None,
        sources=dict(sources), scheduler=raw['scheduler'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = read(PROTOCOL)
    archive, legacy = ROOT/cfg['balanced_archive'], ROOT/cfg['legacy_archive']
    oldplan = read(legacy/'plan.json'); protocol = oldplan['protocol']
    balancedplan = read(archive/'plan.json'); release = read(archive/'release.json')
    assert cfg['budget'] == protocol['budget'] == balancedplan['protocol']['budget']
    assert cfg['scheduler'] == balancedplan['protocol']['scheduler']
    # Strict identity checks include the original benchmark, controller and oracle.
    for r in oldplan['sources']+balancedplan['sources']+release['raw_runs']+[release['implementation']]:
        assert receipt(r['path']) == r, r['path']
    schedule = [(n, m, s) for n in cfg['sizes'] for m in cfg['variants'] for s in cfg['seeds']]
    assert len(schedule) == cfg['maximum_new_searches']
    out = args.output.resolve();out.mkdir(parents=True, exist_ok=False)
    source_paths = {Path(r['path']) for r in oldplan['sources']+balancedplan['sources']}
    source_paths.update([Path(__file__), PROTOCOL, ROOT/'tests/test_balanced_ablation.py'])
    sources = [receipt(p) for p in sorted(source_paths)]
    inputs = [archive/f'{n}-balanced-{s}.json' for n in cfg['sizes'] for s in cfg['seeds']]
    inputs += [legacy/f'reference-{n}.json' for n in cfg['sizes']]
    save(out/'plan.json',dict(protocol=cfg, schedule=schedule, sources=sources,
        archived_inputs=[receipt(p) for p in inputs], frozen_before_new_search=True))
    rows, errors, checked, reused_checked = [], [], 0, 0
    bases = {}
    for n in cfg['sizes']:
        ref = read(legacy/f'reference-{n}.json');problem = build_problem(n, protocol)
        for s in cfg['seeds']:
            full = read(archive/f'{n}-balanced-{s}.json');bases[n,s] = full
            reused_checked += validate_rows(full, problem, ref, protocol)
            row = summary_row(full, ref, full, 'reused_balanced');row['method'] = 'BO-MADS'
            rows.append(row)
    for n, method, seed in schedule:
        optimizer = make_optimizer(n, method, seed, protocol)
        assert all(getattr(optimizer,k) == v for k,v in cfg['scheduler'].items())
        result = optimizer.solve()
        result.update(method=method,size=n,initial_signature=optimizer.initial_signature,
            inner_evaluations=optimizer.inner.history,search_domain_dimension=optimizer.e.domain.dimension,
            component_switches_after_initialization=True)
        save(out/f'{n}-{method}-{seed}.json', result)
        full, ref = bases[n,seed], read(legacy/f'reference-{n}.json')
        # Preserve an audit failure as a result, without spending extra search budget.
        try:
            assert result['initial_signature'] == full['initial_signature']
            assert result['problem_sha256'] == full['problem_sha256']
            assert result['oracle_version'] == full['oracle_version']
            checked += validate_rows(result,optimizer.e.problem,ref,protocol)
            if method == 'BO-only':assert not any(r['source']=='mads_poll' for r in result['events'])
            if method == 'MADS-only':assert not any(r['source']=='bo_expected_improvement' for r in result['events'])
            assert result['scheduler']['charged_repeats'] <= result['scheduler']['repeat_limit']
        except AssertionError as exc:
            errors.append(dict(size=n,method=method,seed=seed,error=str(exc) or 'AUDIT_ASSERTION_FAILED'))
        row = summary_row(result,ref,full,'new_ablation');rows.append(row)
        print(json.dumps({k:row[k] for k in ('size','method','seed','U','delta_U_variant_minus_full','status','budget')},ensure_ascii=False),flush=True)
    assert all(receipt(r['path'])==r for r in sources)
    assert all(receipt(r['path'])==r for r in read(out/'plan.json')['archived_inputs'])
    save(out/'summary.json',dict(rows=rows,audit_errors=errors,new_searches=len(schedule),reused_full_searches=len(bases),
        new_inner_results_checked=checked,reused_inner_results_checked=reused_checked,
        evidence=cfg['evidence'],new_gpu_runs=0,real_predictor_calls=0,predictor_refits=0))
    save(out/'completion.json',dict(status='COMPLETE' if not errors else 'COMPLETE_WITH_AUDIT_FAILURES',
        new_searches=len(schedule),reused_full_searches=len(bases),audit_errors=errors,
        new_inner_results_checked=checked,source_integrity=True,remaining_automatic_experiments=[]))
    if errors:raise SystemExit('Audit failures retained; no performance retries allowed')


if __name__=='__main__':main()
