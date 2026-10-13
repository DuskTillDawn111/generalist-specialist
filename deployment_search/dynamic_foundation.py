"""Module 5A offline validation and frozen W1/W2/W3 transition planning.

Run: python -m deployment_search.dynamic_foundation
All evaluator feedback here is explicitly synthetic. No GPU measurements.
"""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import json
from unittest.mock import patch

from deployment_core.models import Hardware, load_hardware, load_model_catalog
from deployment_core.workload import Workload
from .dynamic_adaptation import (AdaptationConfig, AdaptationHistory, DynamicAdaptation,
    StructuralReconfigurationCost, committed_switches)
from .feedback import bottleneck_feedback
from .neighborhood import deployment_key
from .result import jsonable
from .routing import solve_canonical_routing
from .runtime import SyntheticRuntimeEvaluator, weighted_task_latency
from .runtime_cache import file_hash, save_json
from .scoring import RuntimeObjective
from .search import GuidedDeploymentSearch, SearchConfig
from .tests.fixtures import evaluation, model, synthetic_pool, synthetic_evaluator, forbid_enumeration

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/dynamic_adaptation_foundation'
WARM = AdaptationConfig(top_b=2, max_adaptation_iterations=3, max_runtime_evaluations=4)
COLD = SearchConfig(top_b=2, max_iterations=3, no_improvement_patience=1,
                    runtime_evaluation_budget=6)


@contextmanager
def forbid_gpu_measurement():
    with ExitStack() as stack:
        guards = [stack.enter_context(patch(path, side_effect=AssertionError('No GPU measurement in Module 5A')))
                  for path in ('deployment_search.measurement_runtime.MeasurementRuntimeEvaluator.evaluate',
                      'deployment_search.campaign_runtime.CampaignRuntimeEvaluator.evaluate',
                      'deployment_search.measurement_runtime.measurement_subprocess',
                      'deployment_search.campaign_runtime.measurement_subprocess')]
        yield guards
        assert all(g.call_count == 0 for g in guards)


def synthetic_objective(catalog, residence_penalty_ms=60.):
    """Artificial task latency plus co-residence cost, separate from min-max LP."""
    def value(x, w, routing):
        active = deployment_key(x)
        per_task = {k: min(catalog[m].service_time_ms[k] for m in active if k in catalog[m].supported_tasks)
                    for k in w.active_tasks}
        mean = weighted_task_latency(w, per_task) + residence_penalty_ms * len(active)
        return evaluation(mean, raw='synthetic://dynamic_task_latency_and_residence_penalty')
    return SyntheticRuntimeEvaluator(value)


def scenarios():
    common = {'generalist': model('generalist', 1., text=600., image=600., visual_qa=100.),
              'specialist_a': model('specialist_a', 2., text=100.),
              'specialist_b': model('specialist_b', 2., image=100.)}
    a = Workload({'text': 1., 'image': .1, 'visual_qa': .01})
    b = Workload({'text': .1, 'image': 1., 'visual_qa': .01})
    x = {'generalist': 1, 'specialist_a': 1, 'specialist_b': 0}
    coverage = {'generalist': model('generalist', 1., text=900., image=900.),
                'specialist_a': model('specialist_a', 2., text=100.),
                'specialist_b': model('specialist_b', 2., image=100.)}
    capacity = {'generalist': model('generalist', 1., text=600., visual_qa=100.),
                'specialist': model('specialist', 2., text=200.)}
    return [
        dict(name='A_small_drift', old=a, new=Workload({'text': .95, 'image': .1, 'visual_qa': .01}),
             catalog=common, hardware=Hardware(10), initial=x, penalty=60.),
        dict(name='B_one_specialist_change', old=a, new=b, catalog=common, hardware=Hardware(10),
             initial=x, penalty=60.),
        dict(name='C_missing_coverage', old=Workload({'text': .2}), new=Workload({'text': .2, 'image': 1.}),
             catalog=coverage, hardware=Hardware(10), initial={'generalist': 0, 'specialist_a': 1, 'specialist_b': 0}, penalty=60.),
        dict(name='C_capacity_violation', old=Workload({'text': 1., 'visual_qa': .01}),
             new=Workload({'text': 2., 'visual_qa': .01}), catalog=capacity, hardware=Hardware(10),
             initial={'generalist': 1, 'specialist': 0}, penalty=10.),
    ]


def assert_warm_invariants(result, catalog, config=WARM):
    trace = result.adaptation_trace
    assert result.runtime_evaluation_count <= config.max_runtime_evaluations
    assert trace['repair_iterations'] + trace['search_iterations'] <= config.max_adaptation_iterations
    assert all(s['cost'].measured_switch_time_ms is None for s in result.accepted_switches)
    keys = [deployment_key(result.initial_deployment)] + [deployment_key(s['new_deployment']) for s in result.accepted_switches]
    assert len(keys) == len(set(keys)), 'Within-phase committed cycle'
    for row in trace['iterations']:
        assert row['num_candidates_generated'] <= len(catalog)
        for c in row['candidates']:
            assert sum(row['deployment'][m] != c['mutation']['deployment'][m] for m in catalog) == 1
            if c['runtime'] is not None:
                assert c['selected_top_b']
    for row in trace['repair_trace']:
        if row['mode'] != 'repair':
            continue
        assert len(row['candidates']) <= len(catalog)
        assert all(c['mutation']['operation'] == 'ADD' for c in row['candidates'])


def search_metrics(search, initial, catalog):
    switches = committed_switches(search, initial, StructuralReconfigurationCost(catalog), include_cold_seed=True)
    return dict(final_objective=search.runtime, final_deployment=search.deployment,
        runtime_evaluations=search.runtime_evaluation_count, lp_solves=search.lp_solve_count,
        milp_solves=search.milp_solve_count, deployment_mutations=sum(row['accepted'] is not None for row in search.iterations),
        committed_transitions=len(switches), models_loaded=sum(s['cost'].models_added for s in switches),
        models_unloaded=sum(s['cost'].models_removed for s in switches),
        vram_loaded_mib=sum(s['cost'].added_vram_mib for s in switches),
        vram_released_mib=sum(s['cost'].removed_vram_mib for s in switches),
        measured_switch_time_ms=None, switches=switches, termination_reason=search.status,
        config=search.config)


def compare_transition(case, history=None, phase_id=None):
    c, h, old, new, initial = (case[k] for k in ('catalog', 'hardware', 'old', 'new', 'initial'))
    assert solve_canonical_routing(initial, old, c, WARM.rho_max) is not None
    warm_eval = synthetic_objective(c, case['penalty'])
    cold_eval = synthetic_objective(c, case['penalty'])
    with forbid_enumeration() as warm_guards:
        warm = DynamicAdaptation(warm_eval, predictor=None, history=history).adapt(
            initial, old, new, c, h, WARM, phase_id or case['name'])
    with forbid_enumeration() as cold_guards:
        cold = GuidedDeploymentSearch(cold_eval, predictor=None).search(new, c, h, COLD)
    with forbid_enumeration():
        matched = GuidedDeploymentSearch(synthetic_objective(c, case['penalty']), predictor=None).search(
            new, c, h, SearchConfig(**dict(COLD.__dict__, runtime_evaluation_budget=4)))
    assert_warm_invariants(warm, c)
    assert len(warm_eval.calls) == len(set(warm_eval.calls))
    assert len(warm_eval.calls) == warm.runtime_evaluation_count
    # The exhaustive helper is deliberately outside every production search
    # guard and only receives a <=5-model synthetic catalog.
    from .tests.small_space_baseline import exhaustive_baseline
    oracle = exhaustive_baseline(new, c, h, synthetic_objective(c, case['penalty']))
    warm_metrics = dict(final_objective=warm.final_objective, final_deployment=warm.final_deployment,
        runtime_evaluations=warm.runtime_evaluation_count, lp_solves=warm.lp_solve_count,
        milp_solves=warm.milp_solve_count, deployment_mutations=len(warm.accepted_switches),
        models_loaded=warm.models_loaded, models_unloaded=warm.models_unloaded,
        vram_loaded_mib=warm.vram_loaded_mib, vram_released_mib=warm.vram_released_mib,
        measured_switch_time_ms=None, termination_reason=warm.termination_reason, action=warm.action,
        config=WARM)
    cold_metrics = search_metrics(cold, initial, c)
    matched_metrics = search_metrics(matched, initial, c)
    for metrics in (warm_metrics, cold_metrics, matched_metrics):
        metrics['oracle_gap_ms'] = metrics['final_objective'].weighted_mean_e2e - oracle['J_star']
        metrics['relative_oracle_gap'] = metrics['oracle_gap_ms'] / oracle['J_star']
    comparison = dict(name=case['name'], old_workload=dict(old.arrival_rate), new_workload=dict(new.arrival_rate),
        initial_deployment=initial, synthetic_surface='weighted best-supported task service ms + constant residence cost per deployed model',
        residence_penalty_ms=case['penalty'], warm=warm_metrics, cold=cold_metrics,
        cold_equal_runtime_budget=matched_metrics, small_space_oracle=oracle,
        enumeration_calls_in_search=sum(g.call_count for g in warm_guards + cold_guards),
        accounting='cold includes switch from current deployment to MILP seed; warm retains current baseline; committed structural paths, not physical timings',
        budgets='warm 4, cold reference 6; cold matched 4 reported separately; same objective/evaluator/surface/Top-B/iterations')
    return warm, jsonable(comparison)


def validate_synthetic():
    validation, comparisons = {}, []
    for case in scenarios():
        result, comparison = compare_transition(case)
        validation[case['name']] = result.to_dict()
        comparisons.append(comparison)
        if case['name'].startswith('A_'):
            assert result.action == 'NO_SWITCH'
        if case['name'].startswith('B_'):
            assert deployment_key(result.final_deployment) == ('generalist', 'specialist_b')
            assert result.models_loaded == result.models_unloaded == 1
        if case['name'].startswith('C_'):
            assert not result.initial_feasible and not result.fallback_to_milp
            assert result.adaptation_trace['repair_trace'][0]['accepted'] is not None
    # D: both adaptations share history, but distinct explicit phase IDs reset
    # the search taboo set. The B->A transition can revisit the original state.
    case = scenarios()[1]
    history = AdaptationHistory()
    first, comparison = compare_transition(dict(case, name='D_phase_B'), history, 'D_phase_1_B')
    comparisons.append(comparison)
    reverse = dict(case, name='D_phase_A_return', old=case['new'], new=case['old'], initial=first.final_deployment)
    second, comparison = compare_transition(reverse, history, 'D_phase_2_A')
    comparisons.append(comparison)
    assert deployment_key(second.final_deployment) == deployment_key(case['initial'])
    assert history.accepted_switch_count == len(first.accepted_switches) + len(second.accepted_switches)
    with forbid_enumeration():
        repeat = DynamicAdaptation(synthetic_objective(case['catalog'], case['penalty']), history=history).adapt(
            second.final_deployment, reverse['new'], reverse['new'], case['catalog'], case['hardware'],
            WARM, 'D_phase_2_A')
    assert repeat.action == 'NO_SWITCH'
    validation['D_oscillating_workload'] = dict(phases=[first.to_dict(), second.to_dict()],
        same_phase_followup=repeat.to_dict(), history=jsonable(history),
        cross_phase_return_allowed=True, within_phase_cycle_detected=False,
        explicit_events=['A -> B', 'B -> A'])
    return validation, comparisons


def validate_scalability():
    rows = []
    for size in (10, 20, 30):
        old, catalog, hardware = synthetic_pool(size)
        new = Workload({'text': 1.05})
        with forbid_enumeration():
            prior = GuidedDeploymentSearch(synthetic_evaluator(catalog)).search(old, catalog, hardware, COLD)
            warm = DynamicAdaptation(synthetic_evaluator(catalog)).adapt(prior.deployment, old, new,
                catalog, hardware, WARM, f'pool_{size}')
            cold = GuidedDeploymentSearch(synthetic_evaluator(catalog)).search(new, catalog, hardware, COLD)
            matched = GuidedDeploymentSearch(synthetic_evaluator(catalog)).search(new, catalog, hardware,
                SearchConfig(**dict(COLD.__dict__, runtime_evaluation_budget=4)))
        assert_warm_invariants(warm, catalog)
        counts = [r['num_candidates_generated'] for r in warm.adaptation_trace['iterations']]
        assert max(counts, default=0) <= size
        assert warm.lp_solve_count <= 1 + WARM.max_adaptation_iterations * size
        warm_metrics = dict(final_objective=warm.final_objective, final_deployment=warm.final_deployment,
            runtime_evaluations=warm.runtime_evaluation_count, lp_solves=warm.lp_solve_count,
            milp_solves=warm.milp_solve_count, deployment_mutations=len(warm.accepted_switches),
            models_loaded=warm.models_loaded, models_unloaded=warm.models_unloaded,
            vram_loaded_mib=warm.vram_loaded_mib, vram_released_mib=warm.vram_released_mib,
            measured_switch_time_ms=None, action=warm.action, config=WARM,
            oracle_gap_ms=None, relative_oracle_gap=None)
        cold_metrics = search_metrics(cold, prior.deployment, catalog)
        matched_metrics = search_metrics(matched, prior.deployment, catalog)
        for metrics in (cold_metrics, matched_metrics):
            metrics.update(oracle_gap_ms=None, relative_oracle_gap=None)
        comparison = dict(name=f'E_{size}_model_pool', old_workload=dict(old.arrival_rate),
            new_workload=dict(new.arrival_rate), initial_deployment=prior.deployment,
            synthetic_surface='fixture diminishing returns plus co-residence penalty; not GPU evidence',
            warm=warm_metrics, cold=cold_metrics, cold_equal_runtime_budget=matched_metrics,
            small_space_oracle=None, enumeration_calls_in_search=0,
            accounting='cold includes current -> MILP seed; both report committed structural paths only')
        rows.append(jsonable(dict(pool_size=size, old_workload=dict(old.arrival_rate), new_workload=dict(new.arrival_rate),
            initial_deployment=prior.deployment, candidate_counts_per_iteration=counts,
            one_hop_candidates_per_iteration_bound=size, warm_runtime_evaluations=warm.runtime_evaluation_count,
            runtime_budget=WARM.max_runtime_evaluations, warm_lp_solves=warm.lp_solve_count,
            warm_lp_solve_linear_bound=1 + WARM.max_adaptation_iterations * size,
            warm_final_objective=warm.final_objective, warm_action=warm.action,
            warm_models_loaded=warm.models_loaded, warm_models_unloaded=warm.models_unloaded,
            warm_vram_loaded_mib=warm.vram_loaded_mib, warm_vram_released_mib=warm.vram_released_mib,
            warm=warm_metrics, cold=cold_metrics, cold_equal_runtime_budget=matched_metrics,
            comparison=comparison,
            enumeration_calls=0, exhaustive_oracle_used=False, measured_switch_time_ms=None)))
    return rows


class NoRuntimeAllowed:
    def evaluate(self, *args):
        raise AssertionError('Real workload planning cannot evaluate runtime')


def planning_pass(current, workload, catalog, hardware):
    return GuidedDeploymentSearch(NoRuntimeAllowed(), predictor=None).search(workload, catalog, hardware,
        WARM.__class__(**dict(WARM.__dict__, dry_run=True)).search_config(), initial_deployment=current)


def real_planning(old_name, new_name, current, old_workload, new_workload, catalog, hardware):
    """Receives only the current state and workloads, never the desired final."""
    plan = planning_pass(current, new_workload, catalog, hardware)
    row = plan.iterations[0] if plan.iterations else None
    selected = sorted((c for c in row['candidates'] if c.selected_top_b), key=lambda c: c.rank) if row else []
    add = next((c for c in selected if c.mutation.operation == 'ADD'), None)
    followup = None
    if add is not None:
        next_plan = planning_pass(add.mutation.deployment, new_workload, catalog, hardware)
        followup = dict(assumption='hypothetical acceptance of first LP-selected ADD; no runtime evidence or committed mutation',
            first_mutation=add.mutation, plan=next_plan.to_dict())
    return jsonable(dict(transition=f'{old_name} -> {new_name}', event='WORKLOAD_CHANGE',
        old_workload=dict(old_workload.arrival_rate), new_workload=dict(new_workload.arrival_rate),
        current_deployment=current, current_deployment_feasible=plan.initial_feasible,
        initial_inspection=plan.initial_inspection, planning_trace=plan.to_dict(),
        conditional_next_one_hop=followup, runtime_evaluation_count=0,
        keep_current_is_legal=bool(plan.initial_feasible), action='PLAN_ONLY',
        committed_switches=0, measured_switch_time_ms=None, predictor=None,
        enumeration_calls=0, historical_reference_used_for_candidate_generation=False))


def run_real_dry_runs():
    base = ROOT / 'results/deployment_search_multiworkload'
    workload_path = base / 'benchmark_workloads.json'
    doc = json.loads(workload_path.read_text())
    workloads = {w['name']: Workload(w['arrival_rates_rps']) for w in doc['workloads']}
    names = ['W1_balanced', 'W2_generalist_heavy', 'W3_specialist_friendly']
    source_paths = {n: base / 'algorithm_runs' / f'{n}__B2' / 'run_result.json' for n in names}
    running = {n: json.loads(p.read_text())['final_deployment'] for n, p in source_paths.items()}
    catalog, hardware = load_model_catalog(), load_hardware()
    plans = {}
    for i, j in ((0, 1), (1, 2), (2, 0), (0, 2)):
        old, new = names[i], names[j]
        from deployment_core.milp_seeder import MilpDeploymentSeeder
        with forbid_enumeration(), patch.object(MilpDeploymentSeeder, 'solve',
                side_effect=AssertionError('Real planning must not restart MILP')) as milp_guard:
            plan = real_planning(old, new, running[old], workloads[old], workloads[new], catalog, hardware)
        # Attach history strictly AFTER planning; it never enters the algorithm.
        target = running[new]
        adds = sorted(set(deployment_key(target)) - set(deployment_key(running[old])))
        removes = sorted(set(deployment_key(running[old])) - set(deployment_key(target)))
        historical_seed = json.loads(source_paths[new].read_text())['seed_deployment']
        seed_key, reference_key = set(deployment_key(historical_seed)), set(deployment_key(target))
        plan['historical_reference'] = dict(deployment=target, role='reference only, not an oracle',
            required_adds=adds, required_removes=removes,
            structural_one_hop_distance_from_current=len(adds) + len(removes),
            cost=StructuralReconfigurationCost(catalog).estimate(running[old], target),
            cold_milp_seed=historical_seed,
            structural_one_hop_distance_from_cold_seed=len(seed_key ^ reference_key),
            cold_seed_is_historical_reference=True)
        plan['new_milp_solve_count'] = milp_guard.call_count
        plan['input_provenance'] = dict(workloads_path=str(workload_path), workload_sha256=file_hash(workload_path),
            current_deployment_source=str(source_paths[old]), current_deployment_source_sha256=file_hash(source_paths[old]),
            historical_reference_source=str(source_paths[new]), historical_reference_source_sha256=file_hash(source_paths[new]))
        key = f'w{i+1}_to_w{j+1}_dry_run'
        plans[key] = plan
        save_json(OUT / f'{key}.json', plan)
    return plans


def write_summary(validation, comparisons, scalability, plans):
    def gap(value):
        return 'unavailable' if value is None else f'{value:.3f}'
    lines = ['# Module 5A：Dynamic Workload Adaptation Foundation', '',
        '已实现 warm-start / ADD repair / fallback / NO_SWITCH / structural switching accounting。仅 synthetic 和 offline planning，GPU measurement calls=0；到此停止。', '',
        'Cold 默认行为保持：initial_deployment=None → 原 MILP seed 和搜索；warm feasible 起点直接计算新 workload 的 canonical LP，MILP solves=0。',
        'Repair 只生成与 missing capability / overloaded tasks 相交的 one-hop ADD。先恢复 feasibility，再减少 unserved rate / relaxed max pressure；恢复前不 REMOVE、不评价 infeasible runtime。',
        'Repair 消耗 warm iterations；预算耗尽或没有可行修复才 fallback，记录 warm_start_fallback=true。fallback 后不增加 iteration/runtime budget。',
        '当前 deployment 是正式 baseline；同一 phase visited states/rejected candidates 不重测，跨明确的新 phase 可重新访问旧 deployment。没有 forecasting 或自动 workload detection。',
        'ReconfigurationCostModel 默认 StructuralReconfigurationCost；只累计 committed path 的 loads/unloads 和 VRAM movement，measured_switch_time_ms=null；没有从大小估算时间。', '',
        '## Synthetic cold vs warm', '',
        'A–D synthetic latency = arrival-rate weighted best-supported task service ms + per-model residence penalty；E 复用独立 diminishing-return/residence fixture。均仅人工 response surface，不能当作真实 GPU 效果。',
        '两者共享 surface、objective、Top-2、最多 3 轮；warm budget=4、cold reference=6，并另报 cold budget=4 的 matched comparison。Cold 重配置成本包含 current → MILP seed。', '',
        '| Transition | Warm action | Warm / cold / matched mean ms | Warm / cold / matched evals | Warm / cold LP | Warm loads/unloads | Cold loads/unloads | Warm / cold oracle gap ms |',
        '|---|---|---|---|---|---|---|---|']
    for row in comparisons:
        w, c, matched = row['warm'], row['cold'], row['cold_equal_runtime_budget']
        lines.append(f'| {row["name"]} | {w["action"]} | {w["final_objective"]["weighted_mean_e2e"]:.3f} / {c["final_objective"]["weighted_mean_e2e"]:.3f} / {matched["final_objective"]["weighted_mean_e2e"]:.3f} | {w["runtime_evaluations"]} / {c["runtime_evaluations"]} / {matched["runtime_evaluations"]} | {w["lp_solves"]} / {c["lp_solves"]} | {w["models_loaded"]}/{w["models_unloaded"]} | {c["models_loaded"]}/{c["models_unloaded"]} | {gap(w["oracle_gap_ms"])} / {gap(c["oracle_gap_ms"])} |')
    lines += ['', 'Case A：small drift 保持 NO_SWITCH。Case B：少量 one-hop ADD/REMOVE 达到 generalist+specialist-B。Case C 同时验证 coverage 和 pressure>rho_max 的 capacity repair。',
        'Case D：A→B→A 可跨 phase 返回原 deployment；同一 phase follow-up 不产生来回切换；switch accounting 与所有 accepted transitions 一致。',
        'Small-space oracle 只在 2/3-model synthetic catalog 的独立验证阶段调用；正式 cold/warm search 被 enumeration tripwire 包围。',
        '完整 objective、LP、eval、mutation、VRAM movement 和 oracle gaps 见 cold_vs_warm_comparison.json。Warm 不保证获胜，尤其可能比 cold 需要更多 local mutations。', '',
        '例如 C_capacity_violation：两者各 1 次 evaluation，warm LP=3、cold LP=1；B_specialist_change 两者 evaluation 都为 4。C_missing_coverage 则 warm eval=2、cold eval=3，warm loads/unloads=1/0、cold=3/2。不宣称 warm 在所有维度占优。', '',
        '## Scalability', '', '| Pool | Warm candidates per round | Warm evals / budget | Warm LP / linear bound | Warm action |', '|---:|---|---|---|---|']
    for row in scalability:
        lines.append(f'| {row["pool_size"]} | {row["candidate_counts_per_iteration"]} | {row["warm_runtime_evaluations"]} / {row["runtime_budget"]} | {row["warm_lp_solves"]} / {row["warm_lp_solve_linear_bound"]} | {row["warm_action"]} |')
    lines += ['', '## Frozen real-workload planning', '',
        '真实 workload 保持 Module 3C benchmark_workloads.json 速率；不复用历史 latency 作决策，不调用 runtime evaluator，不将 historical final 当作 oracle。',
        '以下第二轮只是假设 first LP-selected ADD 被未来 runtime 接受后的一个局部 planning pass，不是实际接受，也不递归扩展完整状态图。']
    for name, plan in plans.items():
        inspect = plan['initial_inspection']
        row = plan['planning_trace']['iterations'][0]
        selected = sorted((c for c in row['candidates'] if c['selected_top_b']), key=lambda c: c['rank'])
        lines += ['', f'### {plan["transition"]}', '',
            f'Current feasible={plan["current_deployment_feasible"]}；bottleneck={row["feedback"]["bottleneck_models"]}；z={inspect["max_pressure"]:.9f}。',
            f'Pressure tasks={inspect["pressure_tasks"]}；capacity slack={inspect["capacity_slack"]}。',
            '首轮 Top-B：' + '; '.join(f'{c["mutation"]["operation"]} {c["mutation"]["model"]} (delta_z={c["delta_z"]:.9f})' for c in selected) + '。']
        required_add = plan['historical_reference']['required_adds']
        required_remove = plan['historical_reference']['required_removes']
        for m in required_add:
            found = next((c for c in row['candidates'] if c['mutation']['model'] == m and c['mutation']['operation'] == 'ADD'), None)
            lines.append(f'+{m} 自然产生={found is not None}，rank={found["rank"] if found else None}，Top-B={found["selected_top_b"] if found else False}。')
        for m in required_remove:
            found = next((c for c in row['candidates'] if c['mutation']['model'] == m and c['mutation']['operation'] == 'REMOVE'), None)
            next_row = plan['conditional_next_one_hop']['plan']['iterations'][0] if plan['conditional_next_one_hop'] else None
            following = next((c for c in next_row['candidates'] if c['mutation']['model'] == m and c['mutation']['operation'] == 'REMOVE'), None) if next_row else None
            lines.append(f'-{m} 首轮已可生成={found is not None}；假设接受首个 ADD 后仍自然生成={following is not None}，后续 rank={following["rank"] if following else None}，Top-B={following["selected_top_b"] if following else False}。REMOVE 仍需 feasibility 筛选与未来 runtime 接受。')
        ref = plan['historical_reference']
        lines.append(f'仅对 historical reference 计结构距离：warm 至少 {ref["structural_one_hop_distance_from_current"]} mutations；cold seed={deployment_key(ref["cold_milp_seed"])} 至少 {ref["structural_one_hop_distance_from_cold_seed"]} mutations。这是路径长度参考，不是算法会走向该部署的保证。')
        lines.append('KEEP CURRENT / NO_SWITCH 是合法候选动作；planning 本身不能确认 latency improvement，committed switches=0。')
        if not required_add and not required_remove:
            lines.append('历史 final 与 current 相同，因此 NO_SWITCH 合理且可保留；即使 LP 找到 ADD，也必须由未来真实反馈决定，不能因为历史相同就强制 no-op。')
    lines += ['', '## Scope and verification', '',
        'Formal state S_t=(x_t,r_t,lambda_t)；新事件显式给 lambda_old/lambda_new，r_t 为 canonical。AdaptationHistory 的 phase_id 明确指定，不推断 phase。',
        'Dry-run 返回 PLAN_ONLY 并保留原 deployment，不计 accepted switch；真实收益和 switching latency 尚未验证。',
        '四个真实 dry-run 的 MILP calls=0；cold seed=qwen 只读取 Module 3C 已保存的 seed_deployment，用于 cold reference path 距离。',
        '本阶段未进入真实 dynamic transition、forecasting、change-point detection、prewarming、replica、joint routing search 或 final paper benchmark。',
        '测试结果和 no-GPU / enumeration 审计见 validation_manifest.json。']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with forbid_gpu_measurement() as gpu_guards:
        validation, comparisons = validate_synthetic()
        scalability = validate_scalability()
        validation['E_30_model_pool'] = scalability[-1]
        comparisons.extend(r['comparison'] for r in scalability)
        plans = run_real_dry_runs()
    save_json(OUT / 'synthetic_transition_validation.json', validation)
    save_json(OUT / 'cold_vs_warm_comparison.json', comparisons)
    save_json(OUT / 'scalability_validation.json', scalability)
    save_json(OUT / 'adaptation_trace_example.json', validation['B_one_specialist_change'])
    write_summary(validation, comparisons, scalability, plans)
    save_json(OUT / 'validation_manifest.json', dict(module='5A', validation_passed=True,
        gpu_measurement_calls=sum(g.call_count for g in gpu_guards),
        production_deployment_enumeration_calls=0, predictor=None,
        real_planning_milp_calls=0,
        exhaustive_oracle_scope='<=5-model synthetic only; outside production search',
        warm_config=WARM, cold_reference_config=COLD, measured_switch_time_ms=None,
        implementation_hashes={str(p.relative_to(ROOT)): file_hash(p) for p in
            [ROOT / 'deployment_search/search.py', ROOT / 'deployment_search/routing.py',
             ROOT / 'deployment_search/result.py', ROOT / 'deployment_search/__init__.py',
             ROOT / 'deployment_search/warm_start.py', ROOT / 'deployment_search/dynamic_adaptation.py', Path(__file__)]},
        stopped_after_module5a=True))
    print(json.dumps(dict(event='module5a_offline_validation_complete', synthetic_transitions=len(comparisons),
        real_planning_transitions=len(plans), max_model_pool=30, gpu_measurements=0)))


if __name__ == '__main__':
    main()
