from .common import *
import statistics
from .cpu_core import evaluate,catalog

def main():
 fm=read(OUT/'frozen_problem_manifest.json');sp=read(fm['space']['path']);oracles=read(OUT/'cpu_oracles.json');dec=read(OUT/'optimizer_decisions.json');out=[];runtime=list(csv.DictReader((OUT/'optimizer_runtime_benchmark.csv').open()));margins=[]
 for o in oracles:
  pool=o['pool'];scenario=o['scenario'];uu=o['U_oracle'];jj=o['J_oracle']
  rows=list(csv.DictReader((OUT/f'cpu/{scenario}_{pool}_candidates.csv').open()));eligible=[r for r in rows if r['eligible']=='True'];uniqU=sorted(set(float(r['predicted_U']) for r in eligible),reverse=True)
  margins.append(dict(pool=pool,scenario=scenario,distinct_U_best_margin=uniqU[0]-uniqU[1] if len(uniqU)>1 else None,U_J_oracle_same=identity(uu['deployment'],uu['routing'])==identity(jj['deployment'],jj['routing']),U_at_J_best=jj['predicted_U'],U_objective_opportunity_loss=uu['predicted_U']-jj['predicted_U']))
  for r in [r for r in dec if r['pool']==pool and r['scenario']==scenario]:
   a=dict(r,oracle_U=uu['predicted_U'],J_oracle_seconds=jj['predicted_J_seconds'],J_excess_seconds=r.get('predicted_J_seconds',0)-jj['predicted_J_seconds'] if r.get('status')=='FEASIBLE' else None,objective_caution='V3 engine acceptance is original multimetric noise comparator; globalJ is routing/ranking reference, not its sole formal objective.')
   times=[float(t['seconds']) for t in runtime if t['pool']==pool and t['scenario']==scenario and t['algorithm']==r['algorithm']];a['seconds_median']=statistics.median(times) if times else r.get('seconds');a['oracle_over_solver_speedup']=o['seconds']/a['seconds_median'] if a['seconds_median'] else None
   ts=[t for t in runtime if t['pool']==pool and t['scenario']==scenario and t['algorithm']==r['algorithm']];a['calls_fraction_of_oracle_routes']=int(ts[0]['visited'])/o['evaluated_routes'] if ts and ts[0]['visited'] else None;a['calls_fraction_caution']='Runtime/candidate call counts may repeat states; not a unique-domain visit proof.'
   if r['algorithm']=='20A_U_GRID':
    fit=read(OUT/f'cpu/{scenario}_{pool}_20A_U_GRID_1.json');ok=[c for c in fit['candidates'] if c['prediction']['eligible']];a['generated_ranker_exact']=abs(max(c['prediction']['U'] for c in ok)-r['predicted_U'])<1e-10;a['error_class']='CANDIDATE_GENERATION_SEARCH_GAP' if r['regret']>1e-8 else 'FINITE_U_OPTIMUM_FOUND'
    a['generated_unique']=len(fit['candidates']);a['generated_legal']=len(ok);a['generated_legal_fraction']=len(ok)/o['feasible_routes']
   out.append(a)
 csvsave('objective_and_search_error.csv',out);save('oracle_margins.json',margins)
 monotone=[]
 for scenario in sp['scenarios']:
  for algorithm in fm['methods']:
   rr=[next(r for r in dec if r['scenario']==scenario and r['pool']==pool and r['algorithm']==algorithm) for pool in ['A','B','C']]
   good=all(r['status']=='FEASIBLE' for r in rr);monotone.append(dict(scenario=scenario,algorithm=algorithm,complete_legal=good,U_by_pool=[r.get('predicted_U') if r['status']=='FEASIBLE' else None for r in rr],monotone=all(b['predicted_U']>=a['predicted_U']-1e-8 for a,b in zip(rr,rr[1:])) if good else None))
 save('method_pool_expansion.json',monotone)
 # Explicit no-benefit text candidate ablation: original complete census subset,
 # same evaluation and previously frozen route/count/budget, no new GPU selection.
 bart=next(m for m in sp['members']['C'] if m.startswith('textattack/'));abl=[]
 for scenario in sp['scenarios']:
  rows=list(csv.DictReader((OUT/f'cpu/{scenario}_C_candidates.csv').open()));rows=[r for r in rows if r['eligible']=='True' and bart not in json.loads(r['deployment'])];best=max(rows,key=lambda r:float(r['predicted_U']));o=next(o for o in oracles if o['scenario']==scenario and o['pool']=='C')
  abl.append(dict(scenario=scenario,removed=bart,census_subset_best_U=float(best['predicted_U']),original_C_best_U=o['U_oracle']['predicted_U'],difference=o['U_oracle']['predicted_U']-float(best['predicted_U']),interpretation='Finite oracle no-benefit audit, not an optimizer rerun after GPU.'))
 save('unhelpful_candidate_control.json',abl)
 print('CPU_ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
