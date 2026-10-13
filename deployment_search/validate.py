"""Reproducible CPU-only Module 3A validation and current-pool planning traces.

The small exhaustive oracle is imported only by this offline validation entrypoint.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import unittest

import scipy

from deployment_core.models import load_hardware, load_model_catalog
from deployment_core.workload import Workload
from . import GuidedDeploymentSearch, SearchConfig, SyntheticRuntimeEvaluator
from .result import jsonable
from .tests.fixtures import synthetic_pool, synthetic_evaluator, forbid_enumeration, evaluation
from .tests.small_space_baseline import exhaustive_baseline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'results/deployment_search_foundation'


def save(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')


def budget_row(result):
    names = ('num_models', 'lp_solve_count', 'milp_solve_count', 'runtime_evaluation_count',
             'visited_deployment_count', 'search_iterations', 'num_candidates_generated',
             'num_candidates_lp_evaluated', 'num_runtime_candidates', 'status')
    return {name: getattr(result, name) for name in names}


def validate(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName('deployment_search.tests.test_search')
    tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (output / 'offline_tests.txt').write_text(stream.getvalue())
    if not tests.wasSuccessful():
        raise RuntimeError('Offline test suite failed; see offline_tests.txt')
    config = SearchConfig(mean_tolerance_ms=0.)
    runs = []
    results = {}
    for size in (5, 10, 20, 30):
        w, c, h = synthetic_pool(size)
        evaluator = synthetic_evaluator(c)
        with forbid_enumeration() as guards:
            result = GuidedDeploymentSearch(evaluator, predictor=None).search(w, c, h, config)
        results[size] = result
        per_iteration = [{k: t[k] for k in ('iteration', 'num_models', 'num_candidates_generated',
                                           'num_candidates_lp_evaluated', 'num_runtime_candidates')}
                         for t in result.iterations]
        checks = dict(
            improved=result.runtime.weighted_mean_e2e < result.seed_runtime.weighted_mean_e2e,
            candidate_bound=all(t['num_candidates_generated'] <= size for t in result.iterations),
            top_b_bound=all(t['num_runtime_candidates'] <= config.top_b for t in result.iterations),
            runtime_budget=result.runtime_evaluation_count <= config.runtime_evaluation_budget,
            no_repeat=len(evaluator.calls) == len(set(evaluator.calls)),
            enumeration_never_called=all(g.call_count == 0 for g in guards),
            no_predictor=True,
        )
        runs.append(dict(**budget_row(result), full_binary_space_size=2 ** size,
                         runtime_fraction_of_full_space=result.runtime_evaluation_count / 2 ** size,
                         initial_mean_ms=result.seed_runtime.weighted_mean_e2e,
                         final_mean_ms=result.runtime.weighted_mean_e2e,
                         max_candidates_in_iteration=max(t['num_candidates_generated'] for t in result.iterations),
                         enumeration_call_count=sum(g.call_count for g in guards), checks=checks,
                         passed=all(checks.values()), iterations=per_iteration))
    results[10].save(output / 'search_trace_example.json')
    save(output / 'scalability_validation.json', dict(
        purpose='Synthetic algorithm budget validation, not GPU or paper performance evidence',
        config=config, passed=all(r['passed'] for r in runs), runs=runs,
        complexity='At most M one-hop candidates per iteration; up to top_b expensive evaluations. '
                   'No claim about MILP worst-case solver complexity.'))
    w, c, h = synthetic_pool(5)
    oracle = exhaustive_baseline(w, c, h, synthetic_evaluator(c))
    j = results[5].runtime.weighted_mean_e2e
    baseline = dict(oracle, J_search=j, gap=(j - oracle['J_star']) / oracle['J_star'],
                    called_by_guided_search=False, model_count=5,
                    limitation='One deterministic synthetic landscape; no global optimality guarantee.')
    w, c, h = synthetic_pool(10)
    rejecting = SyntheticRuntimeEvaluator(lambda x, *_: evaluation(
        10 if sum(x.values()) == 1 else 100, raw='synthetic://rollback-result-retained'))
    with forbid_enumeration():
        rollback = GuidedDeploymentSearch(rejecting).search(w, c, h, SearchConfig())
    save(output / 'synthetic_validation.json', dict(
        passed=all(r['passed'] for r in runs) and tests.wasSuccessful(),
        tests_run=tests.testsRun, test_log='offline_tests.txt',
        python_version=platform.python_version(), scipy_version=scipy.__version__,
        synthetic_definition='One text task at 1 req/s; minimum-memory base plus heterogeneous-speed '
                             'text models. Synthetic mean=100/sum(speed)+20*max(0,model_count-3)^2.',
        scalability=runs, small_exhaustive_baseline=baseline, rollback_trace=rollback.to_dict()))

    # Read frozen workload definitions and existing standalone profiles only.
    # No imports from run_measurements, worker implementations, or model frameworks.
    source = ROOT / 'results/performance_model_comparable_groups/workload_groups.json'
    workloads = json.loads(source.read_text())['groups']
    catalog, hardware = load_model_catalog(), load_hardware()
    dry = []
    for name, rates in workloads.items():
        with forbid_enumeration():
            result = GuidedDeploymentSearch(predictor=None).search(
                Workload(rates), catalog, hardware, SearchConfig(dry_run=True))
        assert result.runtime_evaluation_count == 0
        dry.append(dict(workload_group=name, workload_rps=rates, search=result.to_dict()))
    save(output / 'current_pool_dry_run.json', dict(
        stage='Module 3A', real_runtime_evaluations=0, new_gpu_profiling=False,
        workload_source=dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        catalog={m: dict(memory_mib=p.memory_mib, supported_tasks=sorted(p.supported_tasks),
                         service_time_ms=dict(p.service_time_ms), sources=p.sources) for m, p in catalog.items()},
        hardware=dict(gpu_memory_mib=hardware.gpu_memory_mib, source=hardware.source), groups=dry))
    lines = ['# 当前 5-model pool：Module 3A dry run', '',
             '使用 Module 2G 冻结的三个 workload；只读已有 standalone profiles。',
             'Seed 由最小内存 MILP 实际求解；没有预设 seed。Canonical min-max LP 不代表真实稳定性。',
             'Runtime evaluation = 0；未运行 GPU profiling。每组只检查 seed 的一跳邻域。', '']
    for group in dry:
        r = group['search']
        lines += [f"## {group['workload_group']}", '', f"Status: `{r['status']}`", '',
                  '```text', 'Iteration 0', 'seed:']
        lines += [f'  {m}' for m, n in r['deployment'].items() if n]
        if r['iterations']:
            t = r['iterations'][0]
            f = t['feedback']
            lines += [f"canonical z={f['max_pressure']:.9f}", 'bottleneck:']
            lines += [f"  {m} rho={f['model_pressure'][m]:.9f}" for m in f['bottleneck_models']]
            lines += ['pressure tasks: ' + ', '.join(f['pressure_tasks']), 'candidates:']
            for v in t['candidates']:
                sign = '+' if v['mutation']['operation'] == 'ADD' else '-'
                lines += [f"  {sign}{v['mutation']['model']}: {v['status']}"]
                if v['canonical']:
                    lines += [f"    z={v['canonical']['max_pressure']:.9f} delta_z={v['delta_z']:+.9f} rank={v['rank']}",
                              f"    relief_per_memory={v['relief_per_memory']} memory_freed_mib={v['memory_freed_mib']}"]
                if v['reason']:
                    lines += ['    reason=' + v['reason']]
            lines += ['Top-B:'] + ['  ' + ', '.join(x) for x in t['selected_top_b']]
        lines += ['```', '', f"LP solves={r['lp_solve_count']}; runtime evaluations={r['runtime_evaluation_count']}", '']
    (output / 'current_pool_dry_run.md').write_text('\n'.join(lines))
    summary = ['# Module 3A：LP-Guided Non-enumerative Deployment Search', '',
               f'离线测试：{tests.testsRun} 项通过。当前真实 5-model pool 的三个 workload 均仅做 dry run；GPU profiling=0。', '',
               '| 模型数 | LP solves | Runtime evaluations（含 seed） | Visited | Iterations | 每轮最大候选数 | 全空间 |',
               '|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        summary.append(f"| {r['num_models']} | {r['lp_solve_count']} | {r['runtime_evaluation_count']} | "
                       f"{r['visited_deployment_count']} | {r['search_iterations']} | {r['max_candidates_in_iteration']} | {r['full_binary_space_size']} |")
    summary += ['', f"5-model 独立 exhaustive baseline：J*={oracle['J_star']:.6f} ms；"
                f"J_search={j:.6f} ms；gap={baseline['gap']:.6%}。只在 tests/small_space_baseline.py 中枚举 32 个状态，正式 search 未调用。", '',
                '## 实现与边界', '',
                '- 首步为 MilpDeploymentSeeder.solve；随后重解 canonical min-max LP，不使用 seed routing witness 指导 mutation。',
                '- Bottleneck 按 LP pressure，pressure tasks 按瓶颈上的 demand contribution；ADD 仅考虑这些任务，REMOVE 用绝对 pressure 阈值及 coverage 检查。',
                '- 一轮最多 M 个 one-hop candidates；coverage/memory/LP screening → delta_z 主排序 → Top-B。relief_per_memory 仅记录。',
                '- Predictor=None 完整通过；不可用/异常 prediction 不阻止搜索；有值时只对相同 delta_z 作次排序。',
                '- Runtime 稳定性三态优先；同为 steady 时比较 mean（绝对 ms tolerance）、weighted task P95、memory；同为非 steady 不宣称改善。',
                '- 每轮最多接受一个 Top-B 候选；必须优于轮初 incumbent。其余测量保留为 ROLLBACK；visited/rejected 阻止重试。',
                '- 测试覆盖无可行 seed、LP infeasible、solver error、canonical 与 2G 一致、rollback、budget、patience、iteration 和无候选停止。',
                '- LP solve count 是实际 canonical LP 调用（含 seed 重解，不含 MILP 内部 relaxation）；重复候选缓存 LP。Runtime budget 含 seed；visited 含 seed 和已测量状态，不含仅做 LP 的候选。',
                '- Synthetic latency 是人工离线 objective，扩展性表采用单任务异速模型池，不构成真实性能、全局最优或论文最终验证。O(M) 指邻域候选数，不是 MILP 时间复杂度。',
                '- Canonical LP 在多最优解时不保证 routing 唯一；不扩展到其他 routing optimization。',
                '- RuntimeEvaluator 接口完成；真实 GPU adapter 尚未启用，未来必须复用 joint_deploy/measurement。', '',
                'Module 2 结论保留：cross-deployment regression PARTIAL；local bias calibration FAIL；ranking INSUFFICIENT COMPARABLE GROUPS。预测器不作为搜索正确性的必要依赖。', '',
                '本阶段完成后停止；未进入 full real search、forecasting、multi-replica 或 final paper evaluation。', '',
                '复现：`OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m deployment_search.validate`。', '']
    (output / 'summary.md').write_text('\n'.join(summary))
    return dict(output=str(output), tests_passed=tests.testsRun,
                scalability_passed=all(r['passed'] for r in runs), small_space_gap=baseline['gap'],
                real_runtime_evaluations=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(validate(args.output), indent=2))
