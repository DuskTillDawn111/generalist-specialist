"""Targeted capability replacement; no vector enumeration or runtime labels."""
from dataclasses import dataclass, field
from deployment_core.replicas import normalize_counts, replica_key
from deployment_search.neighborhood import (generate_neighborhood, generate_replica_neighborhood, missing_coverage)

@dataclass
class ReplacementMutation:
    operation: str
    model: str
    deployment: dict
    remove_model: str
    add_model: str
    origins: list = field(default_factory=list)
    exact_replacement_closure: bool = False
    service_dominated_removal: bool = False


def service_redundant(model, before, after, catalog, workload):
    """Last family replica can be replaced on every active capability by no-slower paths.

    A remaining replica is not declared capacity redundant: LP must screen it.
    This uses only admitted capabilities and frozen standalone service demand.
    """
    if before[model] != 1 or after[model] != 0:
        return False
    tasks = catalog[model].supported_tasks & set(workload.active_tasks)
    return all(any(after[m] > 0 and task in p.supported_tasks and
                   p.service_time_ms[task] <= catalog[model].service_time_ms[task]
                   for m,p in catalog.items()) for task in tasks)


def generate_targeted_neighborhood(deployment, catalog, workload, feedback, bounds,
                                   memory, memory_budget, roles=None, binary=False,
                                   remove_pressure_threshold=.2):
    x=normalize_counts(deployment,catalog,bounds)
    base=(generate_neighborhood(x,catalog,workload,feedback,remove_pressure_threshold) if binary else
          generate_replica_neighborhood(x,catalog,workload,feedback,bounds,remove_pressure_threshold))
    active=set(workload.active_tasks)
    providers={task:{m for m,p in catalog.items() if task in p.supported_tasks} for task in active}
    add_targets={c.model for c in base if c.operation in ('ADD','SCALE_UP') and x[c.model]==0}
    swaps={};proposals=0;rejections={'coverage':0,'memory':0};role_map=roles or {}
    for i in sorted(m for m in catalog if x[m]>0):
        served={task for (task,m),demand in feedback.task_contributions.items() if m==i and demand>0 and task in active}
        key_tasks=served or (catalog[i].supported_tasks & active)
        source_active=catalog[i].supported_tasks & active
        # Task index restricts pairs before state construction. No cartesian
        # state space, and no swaps between unrelated active capabilities.
        targets=set().union(*(providers[t] for t in source_active)) if source_active else set()
        for j in sorted(targets):
            if i==j or x[j]>=bounds[j]:continue
            overlap=bool(key_tasks) and key_tasks<=catalog[j].supported_tasks
            specialist=(role_map.get(i)==role_map.get(j)=='specialist' and bool(source_active & catalog[j].supported_tasks))
            prospective_closure=j in add_targets and overlap
            if not (overlap or specialist or prospective_closure):continue
            proposals+=1;y=dict(x);y[i]-=1;y[j]+=1
            if missing_coverage(y,catalog,workload):rejections['coverage']+=1;continue
            if memory(y)>memory_budget:rejections['memory']+=1;continue
            redundant=service_redundant(i,x,y,catalog,workload)
            origins=[]
            if overlap:origins.append('CAPABILITY_OVERLAP_REPLACEMENT')
            without_add=dict(x);without_add[i]-=1
            becomes_redundant=redundant and not service_redundant(i,x,without_add,catalog,workload)
            exact_closure=prospective_closure and becomes_redundant
            if exact_closure:origins.append('ACCEPTED_ADD_REPLACEMENT_CLOSURE')
            if specialist:origins.append('SPECIALIST_REPLACEMENT')
            mu=ReplacementMutation('SWAP',i,y,i,j,origins,
                                   exact_closure,redundant)
            swaps[replica_key(y)]=mu
    return base+list(swaps.values()),{'base_add_remove_candidates':len(base),'targeted_swap_candidates':len(swaps),
        'total_candidates':len(base)+len(swaps),'targeted_pairs_constructed':proposals,'swap_filter_rejections':rejections,
        'formal_ordered_pair_upper_bound':len(catalog)*(len(catalog)-1),'deployment_enumeration_count':0}
