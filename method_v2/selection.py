"""Two runtime slots diversified by class, not predicted real outcomes."""
from .neighborhood import service_redundant

def candidate_class(candidate):
    return 'R' if candidate.mutation.operation in ('REMOVE','SCALE_DOWN','SWAP') else 'P'

def replacement_key(candidate,current,catalog,workload,current_memory):
    mu=candidate.mutation
    closure=bool(getattr(mu,'exact_replacement_closure',False))
    redundant=bool(getattr(mu,'service_dominated_removal',False))
    if mu.operation in ('REMOVE','SCALE_DOWN'):
        redundant=service_redundant(mu.model,current,mu.deployment,catalog,workload)
    # No sign cutoff on delta_z. Feasibility was checked by the original LP.
    return (not closure,not redundant,candidate.memory_mib>current_memory,
            -candidate.delta_z,candidate.memory_mib,tuple(sorted(mu.deployment.items())))

def diversified_order(candidates,current,catalog,workload,current_memory):
    global_order=sorted(candidates,key=lambda c:(-c.delta_z,c.mutation.model,tuple(sorted(c.mutation.deployment.items()))))
    pressure=[c for c in global_order if candidate_class(c)=='P']
    replacements=sorted((c for c in global_order if candidate_class(c)=='R'),
                        key=lambda c:replacement_key(c,current,catalog,workload,current_memory))
    first=pressure[0] if pressure else (replacements[0] if replacements else None)
    selected=[first] if first is not None else []
    second=next((c for c in replacements if c is not first),None)
    if second is None:second=next((c for c in global_order if c is not first),None)
    if second is not None:selected.append(second)
    return selected+[c for c in global_order if all(c is not s for s in selected)]
