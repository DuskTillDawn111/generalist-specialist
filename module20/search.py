"""All predeclared cells, including infeasible results; no favorable-instance filter."""
from .common import *
from .optimizer import Generator,predict_scenarios,fingerprint
import itertools,statistics

def utility_from_norms(p,w):return -sum(w[k]*p[k+'_norm'] for k in ['T','L','P','M']) if p.get('eligible') else None

def main():
 definition=read(OUT/'utility_definition.json');manifest=read(OUT/'utility_reference_manifest.json');refs=manifest['latency_reference_ms'];weights=definition['weights'];candidate_rows=[];sensitivity=[];cells=[];details=[];start=time.perf_counter();solve_total=0.;predict_total=0.;attempts=0
 for pool,poolref in manifest['model_pools'].items():
  d=read(poolref['path']);gen=Generator(d,refs)
  direct_states=math.prod(1+sum(int(n)>0 for n in d['capacity_factors'][m]) for m in d['models'])
  for budget,target in itertools.product(manifest['budgets_mib'],manifest['target_QPS_grid']):
   for qps in manifest['target_QPS_grid'][target]:
    cid=f'{pool}_B{budget:g}_{target}_{qps:g}';rates={**manifest['background_QPS'],target:qps};candidates={};generation=[]
    for fraction,cap,wm in itertools.product(manifest['generator']['target_G_fractions'],manifest['generator']['capacity_caps'],manifest['generator']['generation_memory_weights']):
     w={**weights,'M':wm};r=gen.solve(rates,budget,target,fraction,cap,w);attempts+=1;solve_total+=r['solve_seconds'];generation.append({k:v for k,v in r.items() if k not in ['deployment','routing']})
     if 'deployment' in r:
      candidates.setdefault(fingerprint(r['deployment'],r['routing']),{'deployment':r['deployment'],'routing':r['routing'],'source':'MILP_LINEAR_SEED','generator_status':r['status']})
    # Retain historical G-only/S-only/mixed routes when cell exactly matches.
    historical=read(ROOT/'results/module19_strategy_benchmark/cpu_results.json')['results']
    for rec in historical:
     case=rec['case']
     if case['rates']!=rates or abs(case['budget_mib']-budget)>1e-6:continue
     if read(case['catalog']['path'])['models'].keys()!=d['models'].keys():continue
     fits=[z['fit'] for z in rec['strategies'].values()]
     if rec.get('routing_ablation_S_only'):fits.append(rec['routing_ablation_S_only'])
     for fit in fits:
      h=fit.get('winner')
      if not h:continue
      rr={t:{} for t in rates}
      for key,v in h['routing'].items():
       t,m=key.split('::');rr[t][m]=v
      candidates.setdefault(fingerprint(h['deployment'],rr),{'deployment':h['deployment'],'routing':rr,'source':'FROZEN_HISTORICAL_CANDIDATE','generator_status':0})
    for i,c in enumerate(candidates.values()):
     pstart=time.perf_counter();p=predict_scenarios(d,c['deployment'],c['routing'],rates,budget,refs,weights);predict_total+=time.perf_counter()-pstart;c.update(id=cid+f'_C{i:03d}',prediction=p)
     gf=sum(v for m,v in c['routing'][target].items() if d['models'][m]['role']=='generalist')/qps;c['target_G_fraction']=gf
     c['target_S_resident']=any(d['models'][m]['role']=='specialist' and target in d['models'][m]['supported_tasks'] for m in c['deployment'])
     rho=p.get('rho',{});demand={m:rho[m]*d['capacity_factors'][m][str(n)] for m,n in c['deployment'].items()};c['J_seconds']=sum(v/(1-rho[m]) for m,v in demand.items())/sum(rates.values()) if p.get('eligible') else None
     candidate_rows.append({'cell_id':cid,'candidate_id':c['id'],'pool':pool,'budget_mib':budget,'target':target,'target_QPS':qps,'deployment':c['deployment'],'routing':c['routing'],'source':c['source'],'target_G_fraction':gf,'target_S_resident':c['target_S_resident'],'J_seconds':c['J_seconds'],**{k:v for k,v in p.items() if k!='per_task'},'predicted_task_E2E':p.get('per_task')})
    valid=[c for c in candidates.values() if c['prediction'].get('eligible')]
    def best(cs):return max(cs,key=lambda c:(c['prediction']['U'],-c['prediction']['safe_memory_mib'],c['id'])) if cs else None
    winner=best(valid);G=best([c for c in valid if c['target_G_fraction']>=1-1e-7]);S=best([c for c in valid if c['target_G_fraction']<=1e-7]);pure=best([c for c in [G,S] if c]);old=min(valid,key=lambda c:(c['prediction']['safe_memory_mib'],c['J_seconds'],c['id'])) if valid else None
    delta=winner['prediction']['U']-old['prediction']['U'] if winner else None
    cell={'cell_id':cid,'pool':pool,'budget_mib':budget,'target':target,'target_QPS':qps,'rates':rates,'generation_attempts':len(generation),'unique_candidates':len(candidates),'nominal_eligible_candidates':len(valid),'certified_runtime_stable_candidates':0,'generation_seconds':sum(r['solve_seconds'] for r in generation),'MILP_infeasible_attempts':sum(r['status']==2 for r in generation),'MILP_unresolved_attempts':sum(r['status'] not in [0,2] for r in generation),'status':'NOMINAL_CANDIDATE_RUNTIME_UNVERIFIED' if winner else 'NO_NOMINAL_FEASIBLE_CANDIDATE_NOT_EXHAUSTIVE_PROOF','selected_candidate_id':winner['id'] if winner else None,'selected_U':winner['prediction']['U'] if winner else None,'selected_G_fraction':winner['target_G_fraction'] if winner else None,'selected_target_S_resident':winner['target_S_resident'] if winner else None,'selected_operational_08':winner['prediction']['operational_rho_08'] if winner else None,'selected_stress_all_eligible':winner['prediction']['stress_all_eligible'] if winner else None,'G_only_id':G['id'] if G else None,'G_only_U':G['prediction']['U'] if G else None,'S_only_id':S['id'] if S else None,'S_only_U':S['prediction']['U'] if S else None,'best_pure_U':pure['prediction']['U'] if pure else None,'old_resource_then_J_same_candidate_set_id':old['id'] if old else None,'delta_U_vs_resource_J_same_candidate_set':delta,'direct_deployment_state_count':direct_states,'state_visit_reduction_fraction_vs_naive':1-len({json.dumps(c['deployment'],sort_keys=True) for c in candidates.values()})/direct_states}
    cells.append(cell)
    for w in definition['sensitivity_grid']:
     selected=max(valid,key=lambda c:(utility_from_norms(c['prediction'],w),-c['prediction']['safe_memory_mib'],c['id'])) if valid else None
     sensitivity.append({'cell_id':cid,'weight_id':w['id'],'weights':w,'selected_candidate_id':selected['id'] if selected else None,'U':utility_from_norms(selected['prediction'],w) if selected else None,'selected_G_fraction':selected['target_G_fraction'] if selected else None,'same_as_main':selected['id']==winner['id'] if selected else None,'main_selection_regret':utility_from_norms(selected['prediction'],w)-utility_from_norms(winner['prediction'],w) if selected else None,'grid_scope':'Frozen candidate set; heuristic seed coverage, not exhaustive reoptimization'})
    details.append({'cell':cell,'candidates':list(candidates.values()),'generation_outcomes':generation})
    save('cells/'+cid+'.json',details[-1]);print('SEARCH',len(cells),cid,'candidates',len(candidates),'winner_G',cell['selected_G_fraction'],flush=True)
 # Natural three-phase scan, all cells considered. Positive gain needs feasible same-load control.
 witnesses=[];phase_groups=[]
 for pool,budget,target in itertools.product(manifest['model_pools'],manifest['budgets_mib'],manifest['target_QPS_grid']):
  cc=[c for c in cells if c['pool']==pool and c['budget_mib']==budget and c['target']==target]
  low=[c for c in cc if c['selected_G_fraction'] is not None and c['selected_G_fraction']>=1-1e-7 and not c['selected_target_S_resident']]
  mid=[c for c in cc if c['selected_G_fraction'] is not None and c['selected_G_fraction']<=1e-7 and c['selected_target_S_resident']]
  high=[c for c in cc if c['selected_G_fraction'] is not None and 1e-7<c['selected_G_fraction']<1-1e-7]
  structural=[(l,m,h) for l,m,h in itertools.product(low,mid,high) if l['target_QPS']<m['target_QPS']<h['target_QPS']]
  positive=[]
  for l,m,h in structural:
   gains=[l['selected_U']-l['S_only_U'] if l['S_only_U'] is not None else None,m['selected_U']-m['G_only_U'] if m['G_only_U'] is not None else None,h['selected_U']-h['best_pure_U'] if h['best_pure_U'] is not None else None]
   if all(v is not None and v>.01 for v in gains):
    positive.append((l,m,h,gains));witnesses.append({'pool':pool,'budget_mib':budget,'target':target,'stages':{name:next(d for d in details if d['cell']['cell_id']==c['cell_id']) for name,c in zip(['LOW','MID','HIGH'],[l,m,h])},'delta_U_against_feasible_S_G_pure':gains,'runtime_validation_required':True})
  phase_groups.append({'pool':pool,'budget_mib':budget,'target':target,'LOW_cells':[c['cell_id'] for c in low],'MID_cells':[c['cell_id'] for c in mid],'HIGH_cells':[c['cell_id'] for c in high],'natural_structural_triples':len(structural),'positive_triples':len(positive),'not_found_is_grid_and_heuristic_limited':True})
 csvsave('optimizer_candidate_comparison.csv',candidate_rows);csvsave('optimizer_search_grid.csv',cells);csvsave('weight_sensitivity.csv',sensitivity);save('three_phase_all_groups.json',phase_groups)
 save('frozen_module20b_candidates.json',{'at_utc':now(),'status':'UTILITY_POSITIVE_THREE_PHASE_CPU_CANDIDATE' if witnesses else 'NO_POSITIVE_THREE_PHASE_CANDIDATE','definition':ref(OUT/'utility_definition.json'),'manifest':ref(OUT/'utility_reference_manifest.json'),'positive_three_phase_candidates':witnesses,'unselected_all_cell_inputs':'cells/*.json: full inputs, deployment/routing, candidate choices and generation failures preserved','GPU_launch_authorized':False,'weights_never_changed':True,'independent_validation_supported':False})
 save('search_performance.json',{'at_utc':now(),'cells':len(cells),'unique_candidates':len(candidate_rows),'generation_attempts':attempts,'generation_seconds':solve_total,'prediction_seconds':predict_total,'wall_seconds':time.perf_counter()-start,'nominal_feasible_cells':sum(c['selected_U'] is not None for c in cells),'nominal_infeasible_or_unresolved_cells':sum(c['selected_U'] is None for c in cells),'naive_state_counts_by_pool':{p:next(c['direct_deployment_state_count'] for c in cells if c['pool']==p) for p in manifest['model_pools']},'measured_exhaustive_walltime_speedup':None,'comparison_limit':'State reduction compares unique visited deployment states with full zero-or-measured-count Cartesian enumeration. MILP branch work is not counted as visited deployments; no measured exhaustive time speedup claim. No complete optimality claim.','main_selection_stability_fraction':sum(r['same_as_main'] is True for r in sensitivity)/sum(r['same_as_main'] is not None for r in sensitivity)})
 print('SEARCH_DONE',len(cells),len(candidate_rows),len(witnesses),flush=True)
if __name__=='__main__':main()
