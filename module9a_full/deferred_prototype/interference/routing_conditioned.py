"""Pure (deployment, routing rates, workload rates) -> service time evaluator.

No CUDA, GPU profiling, search, routing mutation, or internal fixed point.
"""
import csv,json,math
from pathlib import Path

class InterferenceEstimator:
 def __init__(self,standalone_ms,capacity_factors,pair_delays_ms,profile_confidence=None,scheduler_policy='FCFS_CONTEXT_SERIAL'):
  self.s0=dict(standalone_ms)
  self.g={m:{int(n):float(v) for n,v in table.items()} for m,table in capacity_factors.items()}
  self.pairs=dict(pair_delays_ms);self.profile_confidence=profile_confidence or {};self.policy=scheduler_policy
  if any(not math.isfinite(v) or v<=0 for v in self.s0.values()):raise ValueError('positive finite s0 required')
  if any(not math.isfinite(v) or v<0 for v in self.pairs.values()):raise ValueError('nonnegative finite delays required')
  if any(p[0]==q[0] for p,q in self.pairs):raise ValueError('self-interference handled by g')
 @classmethod
 def from_directory(cls,directory):
  d=Path(directory);catalog=json.loads((d/'catalog.json').read_text())
  s0={(r['model'],r['task']):r['standalone_s0_ms'] for r in catalog['paths']}
  with (d/'pair_predictions.csv').open() as f:rows=list(csv.DictReader(f))
  pairs={((r['target_model'],r['target_task']),(r['co_model'],r['co_task'])):float(r['additional_delay_ms']) for r in rows}
  confidence={(r['model'],r['task']):{'complete_kernel_attributes':r['kernel_profile_status']=='COMPLETE'} for r in catalog['paths']}
  return cls(s0,catalog['capacity_factors'],pairs,confidence)
 def evaluate(self,*,deployment,routing,workload):
  x=dict(deployment);lam=dict(workload);route={}
  for m,n in x.items():
   if m not in self.g or type(n) is not int or n not in self.g[m]:raise ValueError(f'outside frozen replica bound: {m}={n}')
  for k,v in lam.items():
   if not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:raise ValueError('finite nonnegative workload required')
  for key,value in routing.items():
   if isinstance(key,tuple) and len(key)==2:
    k,m=key;route[(m,k)]=value
   elif isinstance(value,dict):
    for m,v in value.items():route[(m,key)]=v
   elif isinstance(key,str) and '::' in key:
    k,m=key.split('::');route[(m,k)]=value
   else:raise ValueError('routing must map (task,model), task::model, or task->{model:rate}')
  for (m,k),rate in route.items():
   if (m,k) not in self.s0:raise ValueError(f'unadmitted task path {(m,k)}')
   if type(rate) not in (int,float) or not math.isfinite(rate) or rate<0:raise ValueError('finite nonnegative route required')
   if rate>0 and not x.get(m,0):raise ValueError('positive routing to absent model')
   if k not in lam:raise ValueError('routed task absent from workload')
  for k,rate in lam.items():
   total=math.fsum(v for (m,t),v in route.items() if t==k)
   if not math.isclose(total,rate,rel_tol=1e-9,abs_tol=1e-10):raise ValueError(f'workload/routing conservation failure: {k}')
  resident=[m for m,n in x.items() if n>0];paths={m:sorted(p for p in self.s0 if p[0]==m) for m in resident}
  demand={m:math.fsum(route.get(p,0)*self.s0[p]/1000 for p in paths[m]) for m in resident}
  rho={m:demand[m]/self.g[m][x[m]] for m in resident}
  mix={m:{k:route.get((m,k),0)*self.s0[(m,k)]/1000/demand[m] if demand[m] else 0. for _,k in paths[m]} for m in resident}
  effective={};slowdown={};contributions={};confidence={}
  for m in resident:
   for p in paths[m]:
    terms=[]
    for j in resident:
     if j==m or demand[j]==0:continue
     for q in paths[j]:
      weight=min(rho[j],1.)*mix[j][q[1]]
      if weight==0:continue
      if (p,q) not in self.pairs:raise ValueError(f'missing offline pair prediction: {p} <- {q}')
      terms.append({'co_model':j,'co_task':q[1],'busy_probability':min(rho[j],1.),'co_task_mix':mix[j][q[1]],'delay_ms':weight*self.pairs[(p,q)]})
    t=self.s0[p]+math.fsum(v['delay_ms'] for v in terms)
    k=p[1];effective.setdefault(k,{})[m]=t;slowdown.setdefault(k,{})[m]=t/self.s0[p];contributions.setdefault(k,{})[m]=terms
    complete=self.profile_confidence.get(p,{}).get('complete_kernel_attributes',True) and all(self.profile_confidence.get((v['co_model'],v['co_task']),{}).get('complete_kernel_attributes',True) for v in terms)
    confidence.setdefault(k,{})[m]={'complete_kernel_attributes':complete,'duration_reconstruction_confidence':'LOW_DEVICE_TIMING_CPU_GAPS_PROPORTIONALLY_ALLOCATED','pair_inside_profiled_task_domain':True,'multi_model_aggregation_count':len({v['co_model'] for v in terms}),'overloaded_rho_ge_1':rho[m]>=1,'scheduler_policy':self.policy,'scheduler_policy_uncertainty':True,'same_model_replica_scaling_from_frozen_g':True,'replica_task_mix_transfer':x[m]>1,'low_confidence':bool(terms),'hard_prune':False}
  return {'effective_service_time':effective,'predicted_slowdown':slowdown,'task_mix':mix,'base_utilization':rho,'pairwise_contributions':contributions,'confidence':confidence,'overload_flags':{m:rho[m]>=1 for m in resident},'units':{'effective_service_time':'ms','routing':'requests/s','workload':'requests/s'},'SELF_INTERFERENCE_HANDLED_BY_G':True,'routing_modified':False}
