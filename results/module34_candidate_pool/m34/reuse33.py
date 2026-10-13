from .common import *
from module26.common import arithmetic
from .cpu_core import evaluate,catalog

def main():
 fm=read(OUT/'frozen_problem_manifest.json');sp=read(fm['space']['path']);d=read(sp['pools']['C']['path']);refs=read(fm['utility_reference']['path'])['latency_reference_ms'];weights=read(fm['utility_definition']['path'])['weights'];old=ROOT/'results/module33_mixed_oracle';obs=read(old/'development_observations.json');rows=[];cp=catalog(d)
 for o in obs:
  raw=read(o['raw']['path']);slot=raw['slot'];x=slot['deployment'];r=slot['routing'];reasons=[]
  if not all(m in d['models'] for m in x):reasons.append('MODEL_NOT_IN_C')
  if any(n!=1 for n in x.values()):reasons.append('REPLICA_DOMAIN')
  if any(len(v)!=1 for v in r.values()):reasons.append('ENDPOINT_ROUTE_DOMAIN')
  if not o['eligible']:reasons.append('ORIGINAL_GPU_INELIGIBLE')
  new=copy.deepcopy(raw);new['slot']['safe_memory_budget_mib']=sp['budget_mib'];parts=arithmetic(new,d,refs,weights)
  if parts.get('safe_memory_mib',1e99)>sp['budget_mib']:reasons.append('NEW_MEMORY_BUDGET')
  p,j,e=evaluate(d,x,r,slot['lambda_rps'],sp['budget_mib'],refs,weights,cp) if not reasons else ({},None,[])
  rows.append(dict(original_run_id=o['run_id'],candidate_id=o['candidate_id'],admitted_C_endpoint_binary=not reasons,reasons=reasons,original_U=o['U'],rebased_arithmetic_U=parts.get('raw_arithmetic_U'),rebased_legal_U=parts.get('raw_arithmetic_U') if not reasons else None,predicted_U=p.get('U') if not reasons else None,
    raw_source=ref(o['raw']['path']),deployment=x,routing=r,original_budget=slot['safe_memory_budget_mib'],new_budget=sp['budget_mib'],scope='Historical development only; new34 independent trajectories not covered. No new all-pool GPU Oracle.',environment=o['environment']))
 csvsave('module33_GPU_reuse.csv',rows);save('module33_GPU_reuse.json',rows);print('REUSE33',len(rows),'admitted',sum(r['admitted_C_endpoint_binary'] for r in rows))
if __name__=='__main__':main()
