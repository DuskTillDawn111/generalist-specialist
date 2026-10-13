"""Planning-only input and prediction schema; no runtime state or deployment IDs."""
from dataclasses import dataclass, field
import math
from collections.abc import Mapping

from deployment_core.workload import TASKS, validate_number


@dataclass
class TaskPrediction:
    mean_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    throughput_rps: float | None = None
    status: str = 'insufficient_profile_data'


@dataclass
class PerformancePrediction:
    per_task: dict[str, TaskPrediction]
    system_throughput_rps: float | None = None
    status: str = 'insufficient_profile_data'
    diagnostics: dict = field(default_factory=dict)
    model_role: str = 'planning_predictor'


def normalize_inputs(deployment, workload, routing, catalog):
    """Accept count dicts and nested or (task,model)-keyed req/s routing, including MILP output."""
    workload = getattr(workload, 'arrival_rate', workload)
    if not isinstance(deployment,Mapping) or not isinstance(workload,Mapping) or not isinstance(routing,Mapping):
        raise ValueError('Expected deployment/workload/routing mappings')
    if set(deployment)-set(catalog) or set(workload)-set(TASKS):
        raise ValueError('Unknown model or task')
    x = {m:deployment.get(m,0) for m in sorted(catalog)}
    if any(type(v) is not int or v<0 for v in x.values()):
        raise ValueError('Deployment must contain nonnegative integer instance counts')
    rates = {k:workload.get(k,0.) for k in TASKS}
    for k,v in rates.items(): validate_number(v,'workload.'+k)
    r = {k:{m:0. for m in sorted(catalog)} for k in TASKS}
    for key,value in routing.items():
        if isinstance(key,tuple) and len(key)==2:
            entries=[(key[0],key[1],value)]
        elif isinstance(key,str) and isinstance(value,Mapping):
            if key not in r: raise ValueError('Unknown routing task')
            entries=[(key,m,v) for m,v in value.items()]
        else: raise ValueError('Routing must be nested or pair-keyed req/s')
        for k,m,v in entries:
            if k not in r or m not in catalog: raise ValueError('Unknown routing path')
            validate_number(v,f'routing.{k}.{m}')
            if v>0 and (x[m]==0 or k not in catalog[m].supported_tasks):
                raise ValueError('Routing to undeployed or unsupported service path')
            r[k][m]=float(v)
    for k,rate in rates.items():
        if not math.isclose(math.fsum(r[k].values()),rate,rel_tol=1e-7,abs_tol=0):
            raise ValueError('Routing conservation failed: '+k)
    return x,rates,r
