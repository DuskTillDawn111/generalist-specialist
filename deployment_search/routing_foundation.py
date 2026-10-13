"""Offline Module 4A validation and current-pool dry-run. Never starts GPU work."""
from pathlib import Path
import json
import math
import time

from deployment_core.models import ModelProfile, load_model_catalog
from deployment_core.workload import Workload

from .routing_search import GuidedRoutingSearch, RoutingSearchConfig, validate_routing_candidate
from .runtime import RuntimeEvaluation, SyntheticRuntimeEvaluator
from .runtime_cache import save_json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/routing_search_foundation'
MODULE3 = ROOT / 'results/deployment_search_multiworkload'


def profile(name, **service_ms):
    return ModelProfile(name, frozenset(service_ms), 1., service_ms)


def synthetic_cases():
    # The oracle is a separate response surface. None of these equations appears
    # in the pressure guidance or in the real measurement adapter.
    return {
        'case1_specialist': (
            Workload({'text': 1.}),
            {'generalist': profile('generalist', text=600.),
             'specialist': profile('specialist', text=300.)},
            lambda r: 80. + 100. * r['text', 'generalist']),
        'case2_runtime_correction': (
            Workload({'text': 1.}),
            {'generalist': profile('generalist', text=600.),
             'specialist': profile('specialist', text=300.)},
            lambda r: 80. + 200. * (r['text', 'generalist'] - .6) ** 2),
        'case3_four_capable_models': (
            Workload({'text': 1.}),
            {name: profile(name, text=service) for name, service in
             [('a', 300.), ('b', 250.), ('c', 200.), ('d', 150.)]},
            lambda r: 70. + 80. * (1. - r['text', 'd']) ** 2),
    }


def synthetic_runtime(objective):
    def evaluate(deployment, workload, routing):
        mean = objective(routing)
        return RuntimeEvaluation('steady', mean, mean * 1.5,
                                 sum(workload.arrival_rate.values()), 1.,
                                 {'source': 'synthetic'}, 'synthetic://routing')
    return SyntheticRuntimeEvaluator(evaluate)


def small_grid_oracle(workload, catalog, objective, rho_max=.8):
    """Coarse 0..100% baseline for exactly two capable models, never search entry."""
    assert len(workload.active_tasks) == 1 and len(catalog) == 2
    task = workload.active_tasks[0]
    a, b = sorted(catalog)
    rate = workload.arrival_rate[task]
    feasible = []
    for tenth in range(11):
        routing = {(task, a): rate * tenth / 10,
                   (task, b): rate * (10 - tenth) / 10}
        ok, _ = validate_routing_candidate(routing, {a: 1, b: 1}, workload,
                                           catalog, rho_max)
        if ok:
            feasible.append((objective(routing), routing))
    best, routing = min(feasible, key=lambda item: item[0])
    return dict(grid_step_fraction=.1, feasible_grid_points=len(feasible),
                objective=best, routing={f'{k}::{m}': v for (k, m), v in routing.items()})


def run_synthetic():
    validation, oracle_comparison = {}, {}
    for name, (workload, catalog, objective) in synthetic_cases().items():
        x = {m: 1 for m in catalog}
        evaluator = synthetic_runtime(objective)
        result = GuidedRoutingSearch(evaluator, predictor=None).search(
            x, workload, catalog,
            RoutingSearchConfig(max_iterations=8, no_improvement_patience=2,
                                runtime_evaluation_budget=17))
        before = result.initial_objective.weighted_mean_e2e
        after = result.final_objective.weighted_mean_e2e
        validation[name] = dict(num_models=len(catalog), initial_mean_ms=before,
                                final_mean_ms=after, improvement_ms=before-after,
                                initial_pressure=result.initial_pressure,
                                final_pressure=result.final_pressure,
                                initial_routing=result.initial_routing,
                                final_routing=result.final_routing,
                                accepted_mutations=len(result.accepted_mutations),
                                rejected_mutations=len(result.rejected_mutations),
                                mutations_generated=result.mutations_generated,
                                mutations_evaluated=result.mutations_evaluated,
                                runtime_evaluation_count=result.runtime_evaluation_count,
                                visited_routing_states=len(result.visited_routing_states),
                                termination_reason=result.termination_reason,
                                accepted_delta_z=[m.delta_z for m in result.accepted_mutations])
        if len(catalog) == 2:
            oracle = small_grid_oracle(workload, catalog, objective)
            oracle_comparison[name] = dict(**oracle, search_objective=after,
                                           absolute_gap_ms=after-oracle['objective'],
                                           relative_gap=(after-oracle['objective'])/oracle['objective'])
        if not after < before:
            raise AssertionError(f'{name} did not improve')
    return validation, oracle_comparison


def run_scaling():
    results = {}
    for size in (10, 20, 30):
        catalog = {f'm{i:02d}': profile(f'm{i:02d}', text=200. + i)
                   for i in range(size)}
        workload = Workload({'text': 1.})
        start = time.perf_counter()
        result = GuidedRoutingSearch(predictor=None).search(
            {m: 1 for m in catalog}, workload, catalog,
            RoutingSearchConfig(dry_run=True, max_iterations=1))
        elapsed = time.perf_counter()-start
        results[str(size)] = dict(num_models=size, capable_models=size,
                                  num_active_tasks=result.num_active_tasks,
                                  num_transfer_candidates=result.num_transfer_candidates,
                                  theoretical_ordered_pair_bound=size*(size-1),
                                  selected_top_b=len(result.iterations[0]['selected_top_b']),
                                  runtime_evaluation_count=result.runtime_evaluation_count,
                                  elapsed_seconds=elapsed,
                                  termination_reason=result.termination_reason)
        if result.num_transfer_candidates > size*(size-1):
            raise AssertionError('Candidate count exceeded pair bound')
    return results


def run_real_dry_runs():
    catalog = load_model_catalog()
    records = json.loads((MODULE3/'benchmark_workloads.json').read_text())['workloads']
    outputs = {}
    for i, entry in enumerate(records, 1):
        name = entry['name']
        run = json.loads((MODULE3/'algorithm_runs'/f'{name}__B2'/'run_result.json').read_text())
        x = run['final_deployment']
        workload = Workload(entry['arrival_rates_rps'])
        result = GuidedRoutingSearch(predictor=None).search(
            x, workload, catalog, RoutingSearchConfig(dry_run=True, max_iterations=1))
        trace = result.iterations[0] if result.iterations else None
        record = dict(workload=name, source_run_id=run['run_id'], fixed_deployment=x,
                      num_active_tasks=result.num_active_tasks,
                      routing_freedom=result.routing_freedom,
                      tasks_with_routing_freedom=[k for k, m in result.routing_freedom.items() if len(m)>1],
                      canonical_routing=result.initial_routing,
                      pressure_models=result.initial_pressure,
                      candidate_transfers=trace['candidates'] if trace else [],
                      num_transfer_candidates=result.num_transfer_candidates,
                      top_b_routing_mutations=[c for c in trace['candidates'] if c.selected_top_b]
                      if trace else [],
                      runtime_evaluation_count=result.runtime_evaluation_count,
                      termination_reason=result.termination_reason)
        save_json(OUT/f'w{i}_final_dry_run.json', record)
        outputs[name] = record
    return outputs


def summary(validation, scaling, comparison, dry):
    lines = ['# Module 4A: Non-enumerative Routing Search Foundation', '',
             'Scope: fixed deployment, canonical LP start, one-step workload transfers, pressure/delta_z ranking, Top-2 pruning, shared Module 3 RuntimeEvaluator and RuntimeObjective. No GPU measurement was run.', '',
             '## Synthetic validation', '',
             '| Case | Initial mean (ms) | Final mean (ms) | Accepted | Evaluated | Accepted Δz |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for name, v in validation.items():
        deltas = ', '.join(f'{z:.5f}' for z in v['accepted_delta_z'])
        lines.append(f"| {name} | {v['initial_mean_ms']:.3f} | {v['final_mean_ms']:.3f} | {v['accepted_mutations']} | {v['mutations_evaluated']} | {deltas} |")
    lines += ['', 'Case 2 deliberately has a runtime optimum away from standalone pressure balance. At the initial state, the analytically ranked first move has Δz = -0.03 and worsens synthetic mean latency; the ranked second move has Δz = -0.06 and improves latency, so runtime feedback rejects the first and accepts the second. This is an offline correction example, not a real serving claim.', '',
              '## Small grid oracle (baseline only)', '',
              '| Case | Grid oracle (ms) | Search (ms) | Relative gap |',
              '| --- | ---: | ---: | ---: |']
    for name, v in comparison.items():
        lines.append(f"| {name} | {v['objective']:.3f} | {v['search_objective']:.3f} | {v['relative_gap']:.2%} |")
    lines += ['', 'The 10% grid exists only in `small_grid_oracle`; `GuidedRoutingSearch.search` never calls it.', '',
              '## Scaling', '',
              '| Models | One-step transfers | Ordered-pair bound | Top-B | Runtime calls |',
              '| ---: | ---: | ---: | ---: | ---: |']
    for size, v in scaling.items():
        lines.append(f"| {size} | {v['num_transfer_candidates']} | {v['theoretical_ordered_pair_bound']} | {v['selected_top_b']} | {v['runtime_evaluation_count']} |")
    lines += ['', '## Current pool dry-run', '',
              '| Workload | Tasks with >=2 deployed capable models | Transfers | Top-B |',
              '| --- | --- | ---: | --- |']
    for name, v in dry.items():
        freedom = ', '.join(v['tasks_with_routing_freedom']) or 'none'
        top = ', '.join(f"{c.task}:{c.source}→{c.destination} (Δz={c.delta_z:.5f})"
                        for c in v['top_b_routing_mutations']) or 'none'
        lines.append(f"| {name} | {freedom} | {v['num_transfer_candidates']} | {top} |")
    lines += ['', 'Each workload has 6 active tasks: 4 have two deployed capable models and 2 have only one. The canonical routing assigns all 4 flexible tasks to one model initially, yielding 4 one-step transfers. Every initial real-pool Δz is negative. Because canonical routing globally minimizes max pressure under the Stage-1 constraints, a strictly positive Δz is impossible at the canonical start apart from numerical noise. Pressure ranking therefore limits surrogate damage; only runtime feedback can justify accepting a pressure-worsening transfer. No runtime objective was measured for any real workload, so no real routing improvement claim is made.', '',
              '## Interfaces and limits', '',
              '`GuidedRoutingSearch` accepts any existing `RuntimeEvaluator` instance, including an adapter with a shared measurement cache and Data Quality Audit status semantics. The search adds no serving adapter or cache of its own. Runtime accept/rollback uses the Module 3 `RuntimeObjective` and fixed deployment memory. The optional predictor can be `None`.', '',
              'Termination supports max iterations, no improvement patience, runtime evaluation budget, and no unvisited candidate. The routing state hash uses normalized 9-digit shares to avoid floating-point cycling. No real routing search or joint search was run.', '']
    (OUT/'summary.md').write_text('\n'.join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    validation, comparison = run_synthetic()
    scaling = run_scaling()
    dry = run_real_dry_runs()
    save_json(OUT/'synthetic_validation.json', validation)
    save_json(OUT/'routing_scalability.json', scaling)
    save_json(OUT/'small_grid_oracle_comparison.json', comparison)
    workload, catalog, objective = synthetic_cases()['case2_runtime_correction']
    example = GuidedRoutingSearch(synthetic_runtime(objective)).search(
        {m: 1 for m in catalog}, workload, catalog,
        RoutingSearchConfig(max_iterations=5, runtime_evaluation_budget=11))
    save_json(OUT/'routing_search_trace_example.json', example.to_dict())
    summary(validation, scaling, comparison, dry)


if __name__ == '__main__':
    main()
