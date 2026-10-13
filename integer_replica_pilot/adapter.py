"""Family routing followed by deterministic round-robin to owned worker processes.

Frozen model handlers and DeploymentRuntime transport remain untouched.
"""
from dataclasses import replace
import math
from deployment_core.replicas import normalize_counts, replica_identities
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from joint_deploy.measurement.router import Router


class PlanningMemoryInfeasible(ValueError):pass
class RuntimeResourceInfeasible(RuntimeError):pass


def validate_memory(deployment, capabilities, bounds, table, capacity):
    x=normalize_counts(deployment,capabilities,bounds)
    try:
        values=[table[m][str(n)] for m,n in x.items()]
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 or (x[m]>0 and v==0) for (m,_),v in zip(x.items(),values)):
            raise ValueError('Invalid measured memory value')
        memory=math.fsum(values)
    except KeyError as exc:raise ValueError('Missing evidence-bound memory value; no extrapolation') from exc
    if memory>capacity:raise PlanningMemoryInfeasible(f'Planning memory {memory} MiB exceeds {capacity} MiB')
    return x,memory


def expand(config, capabilities, bounds, table, capacity, registry):
    config.validate(capabilities)
    counts,memory=validate_memory(config.deployment,capabilities,bounds,table,capacity)
    ids=replica_identities(counts)
    mapping={i:m for m,replicas in ids.items() for i in replicas}
    transport=replace(config,deployment={i:1 for i in mapping},routing_rps={
        k:{i:rate/counts[m] for m,rate in route.items() if counts[m] for i in ids[m]}
        for k,route in config.routing_rps.items()})
    transport.validate({i:capabilities[m] for i,m in mapping.items()})
    return transport,{i:registry[m] for i,m in mapping.items()},mapping,memory


class FamilyReplicaRouter:
    """Preserve family sequence exactly; use one round-robin counter per family."""
    def __init__(self,routing,counts):
        self.family=Router(routing)
        self.ids=replica_identities(counts)
        self.index=dict.fromkeys(self.ids,0)

    def choose(self,task):
        m=self.family.choose(task)
        replicas=self.ids[m];i=replicas[self.index[m]%len(replicas)]
        self.index[m]+=1
        return i,m


class ReplicaAwareDeploymentRuntime(DeploymentRuntime):
    def __init__(self,config,registry,capabilities,bounds,table,capacity):
        transport,expanded,mapping,memory=expand(config,capabilities,bounds,table,capacity,registry)
        self.family_config=config
        self.replica_to_family=mapping
        self.planning_memory_mib=memory
        super().__init__(transport,expanded)
