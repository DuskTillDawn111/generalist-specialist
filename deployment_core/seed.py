"""Production/research initialization CLI: one joint MILP deployment/routing seed."""
import argparse
import json
from pathlib import Path

import scipy

from .milp_seeder import MilpDeploymentSeeder, SeedResult, SeedSolverError
from .models import DEFAULT_SOURCES, Hardware, load_hardware, load_model_catalog
from .workload import parse_workload

NOTE = 'This is a feasibility-oriented initialization, not a performance-optimal deployment.'


def seed_as_dict(seed: SeedResult, catalog, workload, hardware, rho_max):
    return dict(schema_version=1,method='MilpDeploymentSeeder',feasible=seed.feasible,
        deployment=seed.deployment,workload_rps=dict(workload.arrival_rate),rho_max=rho_max,
        routing_rps=None if seed.routing_rps is None else [dict(task=k,model=m,rate_rps=v) for (k,m),v in seed.routing_rps.items()],
        utilization=seed.utilization,capacity_slack=seed.capacity_slack,
        service_demand_by_task_model=[dict(task=k,model=m,demand=v) for (k,m),v in seed.service_demand_by_task_model.items()],
        objective='min sum_m V_m*x_m (MiB); no secondary objective', objective_value=seed.objective_value,
        solver_status=seed.solver_status,solver_time_ms=seed.solver_time_ms,solver_message=seed.solver_message,
        solver='scipy.optimize.milp / HiGHS',scipy_version=scipy.__version__,
        solver_options={'mip_rel_gap':0.0},
        numerical_tolerances={'capacity_absolute':1e-9,'routing_conservation_relative':1e-7},
        num_variables=seed.num_variables,num_constraints=seed.num_constraints,
        hardware=dict(name=hardware.name,gpu_memory_mib=hardware.gpu_memory_mib,source=hardware.source),
        model_catalog={m:dict(supported_tasks=sorted(p.supported_tasks),memory_mib=p.memory_mib,
                             service_time_ms=dict(p.service_time_ms),sources=p.sources) for m,p in sorted(catalog.items())},
        scope=NOTE+' Utilization is standalone service demand, not measured GPU utilization.',
        deployment_domain='integer instance counts; prototype bounds 0 <= x_m <= 1',
        zero_workload='Empty deployment is allowed; legacy exhaustive baseline only lists nonempty deployments.')


def print_seed(seed, hardware):
    print('===== MILP DEPLOYMENT SEED =====')
    print('Feasible: '+('yes' if seed.feasible else 'no'))
    if seed.feasible:
        print('\nDeployment:')
        for m,x in seed.deployment.items(): print(f'  {m:12s} {x}')
        print('\nRouting (nonzero req/s):')
        for (k,m),v in seed.routing_rps.items():
            if v: print(f'  {k:12s} -> {m:12s} {v:.9g}')
        print(f'\nMemory: {seed.objective_value:g} / {hardware.gpu_memory_mib:g} MiB')
        print('\nStandalone capacity estimate:')
        for m,rho in seed.utilization.items(): print(f'  {m}: rho={rho:.9g}, slack={seed.capacity_slack[m]:.9g}')
        print('\nDemand contributions (nonzero standalone service demand):')
        for (k,m),v in seed.service_demand_by_task_model.items():
            if v: print(f'  {m}/{k}: {v:.9g}')
    print(f'\nSolver:\n  status: {seed.solver_status}\n  time_ms: {seed.solver_time_ms:.3f}')
    print('\nNOTE:\n'+NOTE)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload',required=True)
    parser.add_argument('--rho-max',type=float,default=.8)
    parser.add_argument('--gpu-memory-mib',type=float)
    parser.add_argument('--catalog-sources',type=Path,default=DEFAULT_SOURCES)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    try:
        catalog=load_model_catalog(args.catalog_sources)
        workload=parse_workload(args.workload)
        hardware=load_hardware(args.catalog_sources)
        if args.gpu_memory_mib is not None:
            hardware=Hardware(args.gpu_memory_mib,hardware.name,{'override':'--gpu-memory-mib','measured_snapshot':hardware.source})
        result=MilpDeploymentSeeder().solve(catalog,workload,hardware,args.rho_max)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(seed_as_dict(result,catalog,workload,hardware,args.rho_max),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    except (ValueError,KeyError,TypeError,OSError,SeedSolverError) as exc:
        parser.error(str(exc))
    print_seed(result,hardware)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
