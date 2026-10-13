"""Correctness-only validation: exhaustive oracle for 19 scenarios, guarded synthetic MILPs."""
from contextlib import ExitStack, contextmanager
import json
import math
from pathlib import Path
import argparse
from unittest.mock import patch

import scipy

from deployment_core import MilpDeploymentSeeder, Workload
from deployment_core.candidate_generator import generate_feasible_deployments
from deployment_core.deployment import Deployment, FeasibleDeployment, RoutingResult
from deployment_core.models import ModelProfile, Hardware, load_model_catalog, load_hardware
from deployment_core.milp_seeder import validate_seed
from deployment_core.seed import seed_as_dict
from deployment_core.experiments.stage1_5_sweep import BASE, SCALES, RHOS, COMPOSITIONS, validate_routing
from deployment_core.workload import TASKS

OUTPUT = Path(__file__).resolve().parents[2]/'results/deployment_seed'


@contextmanager
def forbid_enumeration():
    targets = ('deployment_core.candidate_generator.enumerate_deployments',
               'deployment_core.candidate_generator.generate_feasible_deployments',
               'deployment_core.candidate_generator.screen_deployments',
               'deployment_core.candidate_generator.combinations',
               'deployment_core.candidate_generator.Deployment',
               'deployment_core.deployment.Deployment')
    with ExitStack() as stack:
        guards = [stack.enter_context(patch(name,side_effect=AssertionError('Enumeration/Deployment object creation forbidden: '+name))) for name in targets]
        yield guards
        for guard in guards: guard.assert_not_called()


def scenarios():
    return ([(f'load_{a:.2f}',{k:a*v for k,v in BASE.items()},.8) for a in SCALES]
        + [(name,dict(zip(TASKS,rates)),.8) for name,rates in COMPOSITIONS.items()]
        + [(f'rho_{rho:.2f}',BASE,rho) for rho in RHOS])


def real_validation(catalog, hardware):
    rows=[]
    for name,rates,rho in scenarios():
        workload=Workload(rates)
        # Explicit oracle call lives only in this validation module.
        baseline=generate_feasible_deployments(catalog,workload,hardware,rho)
        with forbid_enumeration():
            seed=MilpDeploymentSeeder().solve(catalog,workload,hardware,rho)
        failures=[]
        if seed.feasible != bool(baseline): failures.append('Feasibility mismatch')
        baseline_min=min((r.memory_mib for r in baseline),default=None)
        membership=None
        if seed.feasible:
            deployed=tuple(sorted(m for m,n in seed.deployment.items() if n))
            membership=deployed in {r.deployment.models for r in baseline}
            if not membership: failures.append('Seed is outside exhaustive feasible set')
            if baseline_min is None or not math.isclose(seed.objective_value,baseline_min,rel_tol=0,abs_tol=1e-9):
                failures.append('Minimum-memory objective mismatch')
            failures.extend(validate_seed(seed,catalog,workload,hardware,rho))
            # Independent, already-validated Stage 1.5 witness checker.
            witness=FeasibleDeployment(Deployment(deployed),seed.objective_value,RoutingResult(
                {(k,m):v for (k,m),v in seed.routing_rps.items() if m in deployed and k in catalog[m].supported_tasks},
                {m:seed.utilization[m] for m in deployed}))
            failures.extend(validate_routing(witness,catalog,workload,rho,name))
        rows.append(dict(name=name,workload_rps=dict(workload.arrival_rate),rho_max=rho,
            baseline_feasible_count=len(baseline),baseline_feasible_set=[list(r.deployment.models) for r in baseline],
            baseline_minimum_memory_mib=baseline_min,seed_belongs_to_baseline=membership,
            enumeration_called_by_seeder=False,seed=seed_as_dict(seed,catalog,workload,hardware,rho),
            failures=failures,passed=not failures))
    return dict(purpose='Real Stage 1 standalone profiles; feasibility/objective correctness only, not measured performance',
                scipy_version=scipy.__version__,num_scenarios=len(rows),passed=all(r['passed'] for r in rows),scenarios=rows)


def synthetic_catalog(size):
    catalog={}
    for i in range(size):
        name=f'synthetic_{i:02d}'
        tasks=TASKS if i%7==0 else (TASKS[i%len(TASKS)],)
        catalog[name]=ModelProfile(name,frozenset(tasks),1200.+31*i if len(tasks)>1 else 120.+7*i,
                                  {k:180.+3*i if len(tasks)>1 else 40.+i for k in tasks})
    return catalog


def synthetic_validation():
    rows=[]
    for n in (20,30):
        catalog=synthetic_catalog(n)
        workload=Workload(dict.fromkeys(TASKS,3.))
        hardware=Hardware(8000.,'synthetic single-GPU budget')
        with forbid_enumeration():
            seed=MilpDeploymentSeeder().solve(catalog,workload,hardware)
        failures=validate_seed(seed,catalog,workload,hardware,.8)
        rows.append(dict(num_models=n,enumeration_called=False,deployment_objects_created=0,
                         num_variables=seed.num_variables,num_constraints=seed.num_constraints,
                         solver_time_ms=seed.solver_time_ms,passed=seed.feasible and not failures,failures=failures,
                         seed=seed_as_dict(seed,catalog,workload,hardware,.8)))
    return dict(purpose='Synthetic solver scalability/call-path test only; NOT real model-serving or paper performance data',
                passed=all(r['passed'] for r in rows),tests=rows)


def write(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def run(output=OUTPUT):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    catalog,hardware=load_model_catalog(),load_hardware()
    real=real_validation(catalog,hardware)
    synthetic=synthetic_validation()
    example=MilpDeploymentSeeder().solve(catalog,Workload(BASE),hardware)
    write(output/'stage1_5_milp_validation.json',real)
    write(output/'synthetic_scalability.json',synthetic)
    write(output/'example_seed.json',seed_as_dict(example,catalog,Workload(BASE),hardware,.8))
    lines=['# MILP deployment seed validation','',
        '正式入口：`MilpDeploymentSeeder.solve(...)` / `python -m deployment_core.seed`。',
        '一次稀疏 MILP 联合求整数实例计数 x 和 req/s routing r，仅返回一个 minimum-memory feasible seed。',
        '约束复用 Stage 1：coverage、真实显存、workload conservation、capability/deployment linkage、standalone capacity。',
        '目标仅为 Σ V_m x_m，mip_rel_gap=0；不加 secondary objective，不声明跨 solver 版本的同值解相同。',
        'utilization=Σ r*s0、slack=rho_max*x−utilization、逐任务 demand=r*s0 均保留完整矩阵；不是实测 GPU utilization。',
        'solver_time_ms 仅计 solver 调用，不包括数据读取/建模/审计；一次计时不代表普遍 scalability。','',
        '| Scenario | Feasible deployments (oracle) | Seed memory MiB | Oracle min MiB | Solver ms | Check |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for row in real['scenarios']:
        seed=row['seed']
        lines.append(f"| {row['name']} | {row['baseline_feasible_count']} | {seed['objective_value']} | {row['baseline_minimum_memory_mib']} | {seed['solver_time_ms']:.3f} | {'PASS' if row['passed'] else 'FAIL'} |")
    lines+=['',f"19 scenarios: {'PASS' if real['passed'] else 'FAIL'}。有解 {sum(r['seed']['feasible'] for r in real['scenarios'])} 个，无解 {sum(not r['seed']['feasible'] for r in real['scenarios'])} 个。",
        '所有 seeder 调用期间均将 enumeration/combinations/旧 Deployment object 构造替换为会抛错的 guard；baseline 仅在 guard 之外执行。','',
        '| Synthetic models | MILP variables | constraints | Solver ms | Enumeration | Check |','| ---: | ---: | ---: | ---: | --- | --- |']
    for row in synthetic['tests']:
        lines.append(f"| {row['num_models']} | {row['num_variables']} | {row['num_constraints']} | {row['solver_time_ms']:.3f} | not invoked | {'PASS' if row['passed'] else 'FAIL'} |")
    lines+=['','Synthetic 数据仅验证直接求解与调用路径，不是真实 serving 性能数据。','',
        f"示例 seed：{', '.join(m for m,n in example.deployment.items() if n)}；显存 {example.objective_value:g} MiB。",
        '该 seed 仅用于可行初始化，不是最终性能最优 deployment。',
        '正式接口允许零负载时全零计数；旧 exhaustive baseline 为历史兼容只枚举非空组合。19 个验证场景均为正负载。',
        '数值校验使用原 Stage 1 守恒相对容差 1e-7、capacity 绝对容差 1e-9；约 1e-16 的负 slack 可由浮点误差产生，保留原计算值。',
        'solver 只有明确 infeasible 状态返回 feasible=false；超时、数值错误、无效 witness 抛 SeedSolverError。',
        '未修改 measurement、CUDA/PyTorch/model 环境；未运行性能 sweep、未实现性能优化或 guided search。','']
    (output/'summary.md').write_text('\n'.join(lines))
    return real['passed'] and synthetic['passed']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    args=parser.parse_args()
    passed=run(args.output_dir)
    print('MILP SEED VALIDATION: '+('PASS' if passed else 'FAIL'))
    return 0 if passed else 1

if __name__=='__main__':
    raise SystemExit(main())
