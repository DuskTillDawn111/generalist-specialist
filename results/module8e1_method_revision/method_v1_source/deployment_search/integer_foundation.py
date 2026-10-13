"""Module 7B offline validation campaign. Never imports a GPU measurement worker."""
import hashlib
import json
import math
from pathlib import Path
import sys

import scipy

from deployment_core import Hardware, ModelProfile, Workload, MilpDeploymentSeeder
from deployment_core.models import load_model_catalog, load_hardware
from deployment_core.milp_seeder import validate_seed
from deployment_core.replicas import derive_replica_bounds, replica_key, replica_statistics, presence
from deployment_core.candidate_pool_v2 import load_candidate_pool_v2, missing_demands, mandatory_presence
from .integer_search import IntegerGuidedDeploymentSearch
from .search import GuidedDeploymentSearch, SearchConfig
from .runtime import SyntheticRuntimeEvaluator, RuntimeEvaluation
from .result import jsonable
from .tests.fixtures import forbid_enumeration, synthetic_pool, synthetic_evaluator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/integer_replica_search_foundation'


def save(name, value):
    (OUT / name).write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def model(name, memory, **services):
    return ModelProfile(name, frozenset(services), memory, services,
                        {'memory': {'kind': 'explicit synthetic setting', 'value_mib': memory}})


def runtime(surface):
    def evaluate(x, w, routing):
        mean = float(surface(x))
        return RuntimeEvaluation('steady', mean, mean * 1.5, sum(w.arrival_rate.values()), 1.,
                                 {'source': 'explicit synthetic response surface; not GPU evidence'}, 'synthetic://integer')
    return SyntheticRuntimeEvaluator(evaluate, integer_mode=True)


def summary(result):
    return {'status': result.status, 'deployment': result.deployment,
            'memory_mib': None if result.deployment is None else sum(
                p['memory_mib'] * result.deployment[m] for m, p in result.bound_provenance.items()),
            'replica_statistics': result.replica_counts_statistics,
            'max_pressure': result.canonical.max_pressure if result.canonical else None,
            'runtime_evaluations': result.runtime_evaluation_count, 'LP_solves': result.lp_solve_count,
            'MILP_solves': result.milp_solve_count, 'visited_states': result.visited_deployment_count,
            'candidate_count_per_iteration': [t['num_candidates_generated'] for t in result.iterations],
            'accepted': [jsonable(c) for t in result.iterations for c in t['candidates'] if c.status == 'ACCEPTED']}


def case_a():
    catalog = {'G': model('G', 3., text=2000.), 'S': model('S', 1., text=1000.)}
    w, h, roles = Workload({'text': 1.2}), Hardware(6.), {'G': 'generalist', 'S': 'specialist'}
    bounds, _ = derive_replica_bounds(catalog, h)
    seed = MilpDeploymentSeeder().solve(catalog, w, h, replica_bounds=bounds)
    if seed.deployment != {'G': 0, 'S': 2} or validate_seed(seed, catalog, w, h, .8, bounds):
        raise AssertionError('Case A integer MILP failed to require S=2')
    with forbid_enumeration():
        result = IntegerGuidedDeploymentSearch(runtime(lambda x: 100 / (x['G'] + 3 * x['S']))).search(
            w, catalog, h, SearchConfig(runtime_evaluation_budget=3), initial_deployment={'G': 1, 'S': 1}, roles=roles)
    found = [c for t in result.iterations for c in t['candidates'] if c.status == 'ACCEPTED' and
             c.mutation.model == 'S' and c.mutation.deployment['S'] == 2]
    if not found:
        raise AssertionError('Case A search did not accept S:1->2')
    return {'passed': True, 'workload_rps': dict(w.arrival_rate), 'seed': seed,
            'presence': presence(seed.deployment), 'bounds': bounds, 'single_specialist_infeasible': 1.2 > .8,
            'two_specialists_feasible': 1.2 <= 2 * .8, 'search': summary(result)}, result


def case_b():
    records = {}
    for name, general_ms, capacity, expected in [('specialist_relief_wins', 2000., 12., 'S'),
                                                ('generalist_relief_wins', 500., 12., 'G'),
                                                ('memory_blocks_generalist', 500., 6., 'S')]:
        catalog = {'G': model('G', 3., text=general_ms), 'S': model('S', 1., text=1000.)}
        roles = {'G': 'generalist', 'S': 'specialist'}
        with forbid_enumeration():
            result = IntegerGuidedDeploymentSearch().search(Workload({'text': 1.2}), catalog, Hardware(capacity),
                SearchConfig(dry_run=True), initial_deployment={'G': 1, 'S': 1}, roles=roles)
        candidates = result.iterations[0]['candidates']
        ups = [c for c in candidates if c.mutation.operation == 'SCALE_UP']
        eligible = sorted((c for c in ups if c.rank is not None), key=lambda c: c.rank)
        if eligible[0].mutation.model != expected:
            raise AssertionError('Case B hard-coded specialist bias or incorrect delta_z ranking')
        records[name] = {'passed': True, 'generalist_service_ms': general_ms, 'GPU_capacity_synthetic_mib': capacity,
                         'current_z': result.canonical.max_pressure, 'winner_by_delta_z': expected,
                         'scale_up_candidates': ups, 'runtime_evaluations': result.runtime_evaluation_count}
    return records


def case_c():
    catalog = {'G': model('G', 4., visual_qa=100.), 'S1': model('S1', 2., text=100.),
               'S2': model('S2', 2., image=100.)}
    w, h = Workload({'text': 1., 'image': 1., 'visual_qa': .1}), Hardware(12.)
    roles = {'G': 'generalist', 'S1': 'specialist', 'S2': 'specialist'}
    with forbid_enumeration():
        result = IntegerGuidedDeploymentSearch(runtime(lambda x: 50.)).search(
            w, catalog, h, SearchConfig(top_b=4, runtime_evaluation_budget=9),
            initial_deployment={'G': 1, 'S1': 2, 'S2': 1}, roles=roles)
    accepted = [c for t in result.iterations for c in t['candidates'] if c.status == 'ACCEPTED']
    if not any(c.mutation.model == 'S1' and c.mutation.operation == 'SCALE_DOWN' and
               c.mutation.deployment['S1'] == 1 and c.memory_freed_mib == 2. for c in accepted):
        raise AssertionError('Case C memory-saving scale down unavailable')
    return {'passed': True, 'initial': {'G': 1, 'S1': 2, 'S2': 1}, 'memory_saved_mib': 2.,
            'runtime_objective': 'constant synthetic mean/p95; unchanged RuntimeObjective memory tie-break',
            'search': summary(result)}


def case_d():
    catalog = {'G': model('G', 4., text=500., asr=500.), 'S': model('S', 2., text=300.),
               'A': model('A', 3., text=100.)}
    h, w = Hardware(8.), Workload({'text': 1., 'asr': .2})
    roles = {'G': 'generalist', 'S': 'specialist', 'A': 'specialist'}
    bounds, _ = derive_replica_bounds(catalog, h)
    with forbid_enumeration():
        result = IntegerGuidedDeploymentSearch(runtime(lambda x: 100 / (x['G'] + 1.3 * x['S'] + 2 * x['A']))).search(
            w, catalog, h, SearchConfig(runtime_evaluation_budget=9), initial_deployment={'G': 1, 'S': 0, 'A': 0}, roles=roles)
    first = result.iterations[0]['candidates']
    up_models = {c.mutation.model for c in first if c.mutation.operation == 'SCALE_UP'}
    blocked = [c for t in result.iterations for c in t['candidates'] if c.reason == 'memory']
    if up_models != set(catalog) or not blocked:
        raise AssertionError('Case D did not expose three choices and hard resource competition')
    return {'passed': True, 'bounds': bounds, 'sum_V_times_R_mib': sum(catalog[m].memory_mib * bounds[m] for m in catalog),
            'GPU_capacity_synthetic_mib': 8., 'initial_scale_up_choices': sorted(up_models),
            'memory_rejected_candidates': len(blocked), 'search': summary(result)}


def scalable_case(size):
    catalog = {f'm{i:02d}': model(f'm{i:02d}', 1. if i == 0 else 1.25 + i / (4 * size),
                                text=900. - 300. * i / size) for i in range(size)}
    roles = {m: 'generalist' if i < 3 else 'specialist' for i, m in enumerate(catalog)}
    h, w = Hardware(8.), Workload({'text': 1.6})
    indices = {m: i for i, m in enumerate(catalog)}
    def surface(x):
        aggregate = sum(n * (1. + indices[m] / size) for m, n in x.items())
        return 100. / aggregate ** .65 + 2. * max(0, sum(x.values()) - 4) ** 2
    with forbid_enumeration():
        result = IntegerGuidedDeploymentSearch(runtime(surface)).search(w, catalog, h,
                SearchConfig(runtime_evaluation_budget=17), roles=roles)
    value = summary(result)
    value.update(num_model_types=size, theoretical_replica_vector_search_space_size=result.theoretical_replica_vector_count,
                 bounds=result.replica_bounds, synthetic_setting='Declared service/memory/response surface; no real pool performance values',
                 enumeration_used=False, runtime_evaluation_budget=17, direct_mutations_limit=2 * size,
                 searched_nonbinary_state=any(n > 1 for t in result.iterations for n in t['deployment'].values()),
                 runtime_response_surface='100/(sum count*(1+model_index/M))**0.65 + 2*max(total replicas-4,0)**2')
    if (not value['searched_nonbinary_state'] or result.runtime_evaluation_count > 17 or
            any(n > 2 * size for n in value['candidate_count_per_iteration'])):
        raise AssertionError('Integer scaling or neighborhood/budget invariant failed')
    return value, result


def oracle_comparison():
    from .tests.small_integer_oracle import small_integer_oracle
    catalog = {'G': model('G', 1., text=550.), 'S1': model('S1', 1., text=350.), 'S2': model('S2', 1., text=200.)}
    h, w, bounds = Hardware(4.), Workload({'text': 1.}), {'G': 2, 'S1': 2, 'S2': 2}
    roles = {'G': 'generalist', 'S1': 'specialist', 'S2': 'specialist'}
    surface = lambda x: 100. / (.8 * x['G'] + 1.2 * x['S1'] + 1.4 * x['S2']) + 2. * max(0, sum(x.values()) - 3) ** 2
    with forbid_enumeration():
        result = IntegerGuidedDeploymentSearch(runtime(surface)).search(w, catalog, h,
            SearchConfig(runtime_evaluation_budget=17, mean_tolerance_ms=0.), replica_bounds=bounds, roles=roles)
    oracle = small_integer_oracle(w, catalog, h, bounds, runtime(surface))
    j = result.runtime.weighted_mean_e2e
    gap = (j - oracle['J_star']) / oracle['J_star']
    if gap < -1e-8:
        raise AssertionError('Search appears better than exhaustive integer oracle')
    return {'passed': True, **oracle, 'J_search': j, 'relative_gap': gap,
            'search': summary(result), 'experiment_R_m_le_2': True,
            'oracle_invoked_only_after_production_search_completed': True}


def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k != 'solver_time_ms'}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def binary_compatibility():
    before = json.loads((OUT / 'binary_reference_before.json').read_text())
    settings = []
    for size in (3, 5):
        w, c, h = synthetic_pool(size)
        settings.append((f'synthetic_{size}', w, c, h, SearchConfig(), synthetic_evaluator))
    c, h = load_model_catalog(), load_hardware()
    workloads = json.loads((ROOT / 'results/deployment_search_multiworkload/benchmark_workloads.json').read_text())
    for row in workloads['workloads']:
        settings.append((row['name'], Workload(row['arrival_rates_rps']), c, h, SearchConfig(dry_run=True), lambda _: None))
    records = []
    for name, w, c, h, config, evaluator in settings:
        with forbid_enumeration():
            legacy = GuidedDeploymentSearch(evaluator(c)).search(w, c, h, config)
            integer = IntegerGuidedDeploymentSearch(evaluator(c)).search(w, c, h, config,
                replica_bounds={m: 1 for m in c})
        a, b = normalize(legacy.to_dict()), normalize(integer.to_dict())
        record = {'case': name, 'matches_pre_change_reference': a == before[name],
                  'integer_R_one_matches_legacy_full_trace': a == b, 'same_seed': legacy.seed.deployment == integer.seed.deployment,
                  'same_candidate_space': [t['ranking'] for t in a['iterations']] == [t['ranking'] for t in b['iterations']],
                  'same_runtime_evaluation_count': legacy.runtime_evaluation_count == integer.runtime_evaluation_count,
                  'ignored_only': 'solver_time_ms; no decision or tie fields ignored', 'deterministic_tie_break_differences': []}
        if not record['matches_pre_change_reference'] or not record['integer_R_one_matches_legacy_full_trace']:
            raise AssertionError('Binary behavior drift: ' + name)
        records.append(record)
    return {'passed': True, 'cases': records, 'binary_mode': 'explicit R_m=1 reduces to the exact legacy path'}


def dry_runs():
    records, hardware = load_candidate_pool_v2()
    bounds, provenance = derive_replica_bounds(records, hardware)
    roles = {m: p.role for m, p in records.items()}
    save('replica_bounds.json', {'source': 'frozen candidate pool v2 measured_idle_vram_mib; no weight size used',
         'gpu_capacity_mib': hardware.gpu_memory_mib, 'bounds': bounds, 'bound_records': provenance,
         'worker_limit_policy': 'No separate physical worker/process limit established. Existing runtime implementation guard is deferred to Module 7C, not a planning-space cap.',
         'theoretical_replica_vector_search_space_size': math.prod(r + 1 for r in bounds.values()),
         'pool_single_copy_idle_memory_mib': sum(p.memory_mib for p in records.values()),
         'sum_V_times_R_mib': sum(records[m].memory_mib * bounds[m] for m in records)})
    workloads = json.loads((ROOT / 'results/deployment_search_multiworkload/benchmark_workloads.json').read_text())['workloads']
    outputs = {}
    for index, row in enumerate(workloads, 1):
        w = Workload(row['arrival_rates_rps'])
        missing = missing_demands(records, w)
        value = {'workload': row['name'], 'workload_rps': dict(w.arrival_rate), 'model_types': len(records),
                 'status': 'BLOCKED_MISSING_STAGE1_SERVICE_DEMAND', 'user_missing_value_policy': 'pending; do not fill values',
                 'missing_service_demands': missing, 'missing_model_types': sorted({v['model'] for v in missing}),
                 'known_service_demands_ms': {m: dict(p.service_time_ms) for m, p in sorted(records.items())},
                 'service_time_sources': {m: p.sources.get('service_time', {}) for m, p in records.items()},
                 'memory_sources': {m: p.sources['memory'] for m, p in records.items()},
                 'replica_bounds': bounds, 'integer_seed_replica_vector': None, 'seed_memory_usage_mib': None,
                 'canonical_routing': None, 'per_replica_pressure': None, 'candidate_count': None,
                 'bottleneck_model_families': None, 'top_scale_up_candidates': None,
                 'top_scale_down_candidates': None, 'Top_B': None,
                 'active_generalist_types': None, 'active_specialist_types': None, 'N_G': None, 'N_S': None,
                 'mandatory_presence': mandatory_presence(records, w),
                 'generalist_composition_note': 'Generalist composition freedom is limited by VQA capability asymmetry: qwen is mandatory presence, not a discretionary algorithm choice.',
                 'MILP_solves': 0, 'LP_solves': 0, 'runtime_evaluations': 0, 'GPU_measurements': 0,
                 'stop_reason': 'No complete 12-model Stage-1 demand matrix. Existing Phi/MiniCPM formal benchmark E2E times use a different protocol; new-model sanity times are not Stage-1 estimates.',
                 'roles': roles,
                 'workload_source': {'path': str(ROOT / 'results/deployment_search_multiworkload/benchmark_workloads.json'),
                     'sha256': hashlib.sha256((ROOT / 'results/deployment_search_multiworkload/benchmark_workloads.json').read_bytes()).hexdigest()}}
        if not missing:
            raise AssertionError('Unexpected service matrix completion; review new evidence before enabling dry-run')
        save(f'w{index}_integer_dry_run.json', value)
        outputs[row['name']] = value
    return outputs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a, trace = case_a()
    b, c, d = case_b(), case_c(), case_d()
    scaling = {}
    for size in (12, 20, 30):
        scaling[str(size)], _ = scalable_case(size)
    oracle = oracle_comparison()
    compatibility = binary_compatibility()
    dry = dry_runs()
    save('integer_milp_validation.json', {'passed': True, 'case_A': a,
         'objective': 'lexicographic min memory then replica count; not performance-optimal allocation',
         'presence_linkage': 'y<=x<=R*y', 'coverage': 'sum capability*y>=1',
         'routing_linkage': 'r<=lambda*capability*y', 'capacity': 'sum r*s0<=rho_max*x',
         'source': 'CPU synthetic validation; no GPU execution'})
    save('synthetic_replica_validation.json', {'passed': True, 'A': a, 'B': b, 'C': c, 'D': d, 'E': scaling['12']})
    save('binary_compatibility.json', compatibility)
    save('integer_scalability.json', {'passed': True, 'cases': scaling, 'enumeration_used': False})
    save('small_integer_oracle_comparison.json', oracle)
    save('replica_search_trace_example.json', trace.to_dict())
    before = json.loads((OUT / 'frozen_inputs_before.json').read_text())
    changed = [p for p, expected in before.items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != expected]
    if changed:
        raise AssertionError('Frozen pool/workload artifacts changed: ' + str(changed))
    save('validation_manifest.json', {'candidate_pool_and_workloads_unchanged': True, 'frozen_files_checked': len(before),
         'new_GPU_measurements': 0, 'real_multi_replica_workers_started': 0,
         'python': sys.version, 'python_executable': sys.executable, 'scipy_version': scipy.__version__,
         'W1_W2_W3_complete_12_model_planning': 'blocked by explicitly preserved missing Stage-1 demand',
         'synthetic_formulation_validation': 'passed', 'acceptance_14': 'blocked; not claimed passed'})
    report = ['# Module 7B Integer Replica-count Optimization Foundation', '',
        '整数 formulation/search 与 synthetic validation 已完成。**W1/W2/W3 的完整 12-model dry-run 被缺失的 frozen Stage-1 service demand 阻塞**；按用户选择保留 pending，不补值，不把 sanity timing 或不同口径的 formal E2E timing 当作 s0。', '',
        '## Formulation', '',
        '`x_m` 是非负整数 replica count；`y_m` 是 binary presence。MILP 使用 `y<=x<=R*y`，coverage 与 routing linkage 均使用 y；memory 使用 `sum V*x<=49140`，capacity 使用 `sum r*s0<=rho_max*x`。integer seeder 先最小化 memory，再在相同 memory optimum 下最小化 replica count，两个 MILP solve 均计入计数。', '',
        '固定 x 的 canonical LP 最小化 `z`，约束 `D_m<=z*x_m`；反馈使用 active families 的 `D_m/x_m`。`D[k,m]=r[k,m]*s0[k,m]` 原样保留。routing 保持 family-level，future replica IDs 仅是 schema，没有启动 workers。', '',
        'Memory 是 **planning memory surrogate**；capacity 是 **analytical planning surrogate, not a claim of linear real-GPU scaling**。Module 7C 尚未进行。', '',
        '每轮最多 2M 个 x±1 mutation，bottleneck 的 Generalist SCALE_UP 与承担 pressure tasks 的 Specialist/alternative SCALE_UP 一起按 delta_z 排序。允许 SCALE_DOWN；runtime acceptance 继续使用既有 RuntimeObjective（stability→mean→p95→memory tie-break），没有 latency+alpha*memory 加权和。', '',
        '## Synthetic validation', '',
        '- A：MILP 返回 G=0/S=2；search 接受 S:1→2。单个 specialist 在该 synthetic load 下不足，两个满足 surrogate。',
        '- B：分别存在 Specialist 的 delta_z 更大、Generalist 的 delta_z 更大，以及 Generalist 增副本被 memory 阻塞的情况；无固定 Specialist 优先规则。',
        '- C：G=1/S1=2/S2=1 可降为 S1=1，并记录 2 MiB synthetic resource saving；mean/p95 相同，既有 memory tie-break 接受。',
        '- D：G replica、S replica 与 alternative type 均产生候选，后续 memory 拒绝证明不能无限增加。', '',
        '| Model types | Theoretical vectors | Per-round mutations | LP solves | Runtime evaluations | Visited states |',
        '|---|---:|---|---:|---:|---:|']
    for size, row in scaling.items():
        report.append(f'| {size} | {row["theoretical_replica_vector_search_space_size"]} | {row["candidate_count_per_iteration"]} (≤{2*int(size)}) | {row["LP_solves"]} | {row["runtime_evaluations"]} / 17 | {row["visited_states"]} |')
    report += ['', f'仅小型 oracle 枚举 {oracle["checked_vectors"]} 个 3-type/R≤2 vectors；J*={oracle["J_star"]:.6g}，J_search={oracle["J_search"]:.6g}，gap={oracle["relative_gap"]:.6g}。正式 IntegerGuidedDeploymentSearch 不导入或调用 oracle，12/20/30 search 带 enumeration tripwire 运行。', '',
        '## Binary compatibility', '',
        '5 个 captured pre-change traces（3/5-type synthetic 与 W1/W2/W3 原 pool）均与修改后 legacy path、integer R=1 path 完整相同，仅忽略 solver wall time。R=1 模式明确复用原 binary seed/neighborhood/key/routing/search，以保留 Module 3/5 行为与旧序列化。该兼容模式保留旧单目标 seed；genuine integer mode 使用两阶段 lexicographic seed。', '',
        '## Frozen pool bounds and missing service matrix', '',
        '12 个 model 的 R 均来自 `floor(49140/measured_idle_vram_mib)`，不读 weight_size_gb，没有任意 max_replicas=2。synthetic small oracle 的 R≤2 是显式实验设置。实际未确立额外物理 process limit；旧 serving implementation 的 count>1 guard 保持不变，不冒充物理 planning limit。', '',
        '当前 pool 的 25 个 verified task/model capability pairs 中，11 个有原 Stage-1 standalone worker s0，14 个 pending，涉及 7 个 model types：新增 5 个 specialists，以及只有不同口径 formal benchmark 的 Phi/MiniCPM。capability_verified 与 service_demand_available 是两个独立状态；没有降低 capability 或排除 frozen candidates 来伪造完整 dry-run。', '',
        'W1/W2/W3 文件已生成 blocked record：seed、memory usage、pressure、bottleneck、candidates、Top-B、N_G/N_S 均为 null，而不是虚构结果或把 5-model diagnostic 冒充 12-model planning。MILP/LP/runtime/GPU calls 均为 0。', '',
        '三个 workload 的 visual_qa rate 都大于零，唯一 capable model 为 qwen，故 **mandatory model induced by capability coverage: y_qwen=1**。Generalist composition 的自由度受 VQA capability asymmetry 限制；这不是算法自由选择了 Qwen。', '',
        '## Acceptance and stopping', '',
        '验收 1–13、15–17 在代码与 synthetic validation 中实现/验证；验收 14（完整 W1/W2/W3 12-model optimization dry-run）为 **blocked**，不能声明全部验收通过。N_G/N_S 对每个可计算 synthetic state 自动输出；blocked workload state 没有计数向量，因此计数为 null。', '',
        '冻结 pool/workload 文件 hash 全部保持不变；未安装/更改环境，未运行 GPU measurement、real multi-replica serving、final benchmark、forecasting、routing reoptimization、dynamic campaign 或 predictor retraining。', '',
        '源码入口：`deployment_core.replicas`、`MilpDeploymentSeeder.solve(..., binary_mode=False)` 或显式 replica_bounds；`IntegerGuidedDeploymentSearch`。validation 命令：`envs/data_env/bin/python -m deployment_search.integer_foundation`。']
    (OUT / 'summary.md').write_text('\n'.join(report) + '\n')
    print(json.dumps({'synthetic': 'passed', 'binary_compatibility': 'passed', 'scaling': list(scaling),
                      'dry_runs': {name: value['status'] for name, value in dry.items()}, 'new_GPU_measurements': 0}, indent=2))


if __name__ == '__main__':
    main()
