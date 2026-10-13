"""Directional, parameter-free offline path-pair simulation.

The bounded alignment sampler is an engineering adaptation; no learned coefficients.
"""
import hashlib, math
import numpy as np
from .kernel_overlap import kernel_array,simulate_arrays
from .schemas import PairPrediction

def feasible_alignments(na,nb,identity,budget=64):
 if na<=0 or nb<=0:raise ValueError('empty sequence')
 size=na*nb
 if size<=budget:indices=np.arange(size)
 else:
  seed=int(hashlib.sha256(('9AF|'+identity).encode()).hexdigest()[:16],16)
  # Prefix-stable samples make the 32/64/128 sensitivity interpretable.
  rng=np.random.default_rng(seed);selected=set();values=[]
  while len(values)<min(budget,size):
   v=int(rng.integers(size))
   if v not in selected:selected.add(v);values.append(v)
  indices=np.array(values)
 return [(int(v//nb),int(v%nb)) for v in indices]

def predict_pair(p,q,limits,policy='FCFS_CONTEXT_SERIAL',budget=64,raw=False):
 if p.model==q.model:raise ValueError('Self-interference handled exclusively by frozen g_m(x)')
 ident=f'{p.model}::{p.task}<-{q.model}::{q.task}';delays=[];gammas=[];counts=[]
 for ip,rp in enumerate(p.requests):
  a=kernel_array(rp['kernels'],limits,raw)
  for iq,rq in enumerate(q.requests):
   b=kernel_array(rq['kernels'],limits,raw);align=feasible_alignments(len(a),len(b),ident+f'|{ip}|{iq}',budget)
   # Both possible simultaneous arrival phases; no artificial stream priority.
   dd=[simulate_arrays(a,b,s,limits,policy,first) for s in align for first in (0,1)]
   gamma=max(len(a)/len(b),1.);delays.append(float(np.median(dd))*gamma);gammas.append(gamma);counts.append(len(align))
 target_time=float(np.median([sum(k['duration_profile_ns']/1e6 for k in r['kernels']) for r in p.requests])) if raw else p.standalone_ms
 delay=float(np.median(delays))
 confidence={'complete_kernel_attributes':p.metadata_complete and q.metadata_complete,'duration_reconstruction':'PROJECT_ENGINEERING_ADAPTATION' if not raw else 'RAW_DIAGNOSTIC_ONLY','duration_reconstruction_confidence':'LOW_DEVICE_TIMING_CPU_GAPS_PROPORTIONALLY_ALLOCATED','inside_profiled_task_domain':True,'scheduler_policy_uncertainty':True,'runtime_context_constraint':policy=='FCFS_CONTEXT_SERIAL','alignment_sampling_budget':budget,'alignment_sampling_exact':all(len(rp['kernels'])*len(rq['kernels'])<=budget for rp in p.requests for rq in q.requests),'request_pair_delay_range_ms':[min(delays),max(delays)],'low_confidence':True}
 return PairPrediction(p.model,p.task,q.model,q.task,target_time,delay,target_time+delay,delay/target_time,sum(counts)*2,float(np.median(gammas)),policy,confidence)
