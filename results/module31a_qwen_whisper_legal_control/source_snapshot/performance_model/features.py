"""Planning features from x/r/lambda and offline profiles only."""
import math
from deployment_core.workload import TASKS
from .schema import normalize_inputs


def build_colocation_pressure(target_model, deployment, routing, resource_profiles):
    """sum_{j != m,k} r[k,j]*p[k,j], independently for each fingerprint key.

    A missing positive-rate contributor makes the total null, retaining a known
    partial sum and missing-path list. Inactive/unsupported paths contribute zero.
    Units are the fingerprint's units times req/s, not measured utilization.
    """
    if deployment.get(target_model,0)<=0: raise ValueError('Target model is not deployed')
    keys=sorted({f for p in resource_profiles.values() for f in p['values']})
    paths=[]
    for k,row in routing.items():
        for m,rate in row.items():
            if rate<0 or not math.isfinite(rate): raise ValueError('Invalid routing rate')
            if rate>0 and deployment.get(m,0)<=0: raise ValueError('Undeployed contributor')
            if m!=target_model and rate>0: paths.append((k,m,rate))
    out={}
    for feature in keys:
        terms=[]; missing=[]
        for k,m,rate in paths:
            value=resource_profiles.get((k,m),{}).get('values',{}).get(feature)
            if value is None: missing.append(dict(task=k,model=m,rate_rps=rate))
            else: terms.append(rate*value)
        total=math.fsum(terms)
        out[feature]=dict(value=None if missing else total,known_partial_sum=total,missing_paths=missing,
                          complete=not missing,definition='sum routing_rps * offline fingerprint; units fingerprint * req/s')
    return out


class FeatureExtractor:
    def __init__(self,model_catalog,resource_profiles=None):
        self.catalog=model_catalog
        self.resource_profiles=resource_profiles or {}

    def extract(self,deployment,workload,routing):
        x,rates,r=normalize_inputs(deployment,workload,routing,self.catalog)
        total=math.fsum(rates.values())
        deployment_features=dict(total_model_instances=sum(x.values()),
            total_resident_memory_mib=math.fsum(self.catalog[m].memory_mib*n for m,n in x.items()))
        per_model={}
        for m,n in x.items():
            if not n: continue
            demand={k:r[k][m]*self.catalog[m].service_time_ms.get(k,0)/1000. for k in TASKS}
            u=math.fsum(demand.values())
            mix={k:demand[k]/u if u>0 else 0. for k in TASKS}
            per_model[m]=dict(instances=n,model_memory_mib=self.catalog[m].memory_mib,
                resident_memory_mib=n*self.catalog[m].memory_mib,standalone_service_demand=u,
                task_demand_mix=mix,task_demand_mix_defined=u>0,service_demand_by_task=demand,
                standalone_service_ms={k:self.catalog[m].service_time_ms.get(k) for k in TASKS},
                colocation_pressure=build_colocation_pressure(m,x,r,self.resource_profiles))
        return dict(task_order=list(TASKS),deployment=x,workload=rates,routing=r,
                    deployment_features=deployment_features,per_model=per_model,
                    workload_composition={k:rates[k]/total if total else 0. for k in TASKS},
                    load_definition='sum r_km*s0_km; standalone service demand, not measured GPU utilization')


def curve_context(features,model):
    """Feature-based controlled-curve context, not a deployment-combination ID.

    Co-load scales with u: compare pressure/u, task mix and memory/count features.
    Unseen deployment compositions may share these features; no table of IDs is used.
    Matching features does not establish generalization accuracy.
    """
    target=features['per_model'][model]; u=target['standalone_service_demand']
    return dict(deployment_features=features['deployment_features'],instances=target['instances'],
        model_memory_mib=target['model_memory_mib'],task_demand_mix=target['task_demand_mix'],
        workload_composition=features['workload_composition'],
        pressure_per_demand={k:v['value']/u if u>0 and v['value'] is not None else None for k,v in target['colocation_pressure'].items()})
