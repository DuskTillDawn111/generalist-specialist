from .common import *
from .cpu_core import catalog
from module20.optimizer import predict_scenarios
from deployment_search.runtime import RuntimeEvaluation
from deployment_search.real_objective import NoiseAwareRuntimeObjective

def main():
 fm=read(OUT/'frozen_problem_manifest.json');sp=read(fm['space']['path']);refs=read(fm['utility_reference']['path'])['latency_reference_ms'];weights=read(fm['utility_definition']['path'])['weights'];dec=read(OUT/'optimizer_decisions.json');out=[];improvers=[];rule=NoiseAwareRuntimeObjective(1.,.05)
 for pool,pr in sp['pools'].items():
  d=read(pr['path'])
  for scenario,L in sp['scenarios'].items():
   rows=list(csv.DictReader((OUT/f'cpu/{scenario}_{pool}_candidates.csv').open()));groups={}
   for r in rows:
    if r['eligible']!='True':continue
    key=r['deployment'];rr=json.loads(r['routing']);order=(float(r['predicted_J_seconds']),identity(json.loads(key),rr));cur=groups.get(key)
    if cur is None or order<cur[0]:groups[key]=(order,r)
   def forecast(r):
    x=json.loads(r['deployment']);rr=json.loads(r['routing']);p=predict_scenarios(d,x,rr,L['rates'],sp['budget_mib'],refs,weights);total=sum(L['rates'].values());mean=sum(q*p['per_task'][t]['mean_ms'] for t,q in L['rates'].items())/total;tail=sum(q*p['per_task'][t]['p95_ms'] for t,q in L['rates'].items())/total
    return RuntimeEvaluation('steady',mean,tail,p['predicted_within_window_QPS'],1.),p
   controlled=next(r for r in dec if r['pool']==pool and r['scenario']==scenario and r['algorithm']=='V3_J_GRID_CONTROL');final=dict(deployment=json.dumps(controlled['deployment'],sort_keys=True),routing=json.dumps(controlled['routing']));current,cp=forecast(final);wins=[];reason_counts={}
   for order,r in groups.values():
    candidate,p=forecast(r);better,reason=rule.compare(candidate,current,p['safe_memory_mib'],cp['safe_memory_mib']);reason_counts[reason]=reason_counts.get(reason,0)+1
    if better:
     row=dict(pool=pool,scenario=scenario,candidate_id=r['candidate_id'],reason=reason,weighted_mean_ms=candidate.weighted_mean_e2e,weighted_p95_ms=candidate.weighted_p95_e2e,current_mean_ms=current.weighted_mean_e2e,current_p95_ms=current.weighted_p95_e2e,predicted_U=p['U'],current_U=cp['U'],deployment=json.loads(r['deployment']),routing=json.loads(r['routing']));wins.append(row);improvers.append(row)
   out.append(dict(pool=pool,scenario=scenario,J_grid_routed_deployment_states=len(groups),original_acceptance_improvers=len(wins),no_strict_acceptance_improver=not wins,comparison_reason_counts=reason_counts,scope='All finite lawful resident states, each uses exactJ-grid router. Original frozen forecastRuntimeEvaluation and unchangedNoiseAwareRuntimeObjective. This is a full-domain no-improver audit for a nonscalar possibly path-dependent comparator, not a unique scalar global-optimum theorem.'))
 save('V3_original_acceptance_audit.json',out);csvsave('V3_original_acceptance_improvers.csv',improvers,fields=['pool','scenario','candidate_id']);print('ORIGINAL_ACCEPTANCE_AUDIT',[(r['pool'],r['scenario'],r['original_acceptance_improvers']) for r in out])
if __name__=='__main__':main()
