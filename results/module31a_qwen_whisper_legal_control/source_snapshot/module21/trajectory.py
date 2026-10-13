"""Native Poisson trajectories; paired inputs generated once, never favorable seed search."""
from .common import *
import random

def arrivals(rates,duration,seed,label):
 result=[]
 for task,q in sorted(rates.items()):
  rng=random.Random(f'{seed}:{label}:arrivals:{task}');route=random.Random(f'{seed}:{label}:route:{task}');rep=random.Random(f'{seed}:{label}:replica:{task}');t=0.
  while True:
   t+=rng.expovariate(q)
   if t>=duration:break
   result.append({'offset_seconds':t,'task':task,'routing_uniform':route.random(),'replica_uniform':rep.random()})
 return sorted(result,key=lambda r:(r['offset_seconds'],r['task']))

def choose(event,routing,deployment):
 task=event['task'];row=routing[task];total=sum(row.values());u=event['routing_uniform'];cum=0.;selected=None
 for m,q in sorted(row.items()):
  if q<=0:continue
  cum+=q/total
  if u<cum:selected=m;break
 if selected is None:selected=sorted(m for m,q in row.items() if q>0)[-1]
 return f'{selected}#{min(deployment[selected]-1,int(event["replica_uniform"]*deployment[selected]))}'
