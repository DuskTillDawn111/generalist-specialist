"""Revalidate structure, distinguish capacity evidence artifacts and memory rejection."""
from .common import *
from module14c.common import E,H
from module14c.analyze import pairs,minimum_real_memory_cover
from module14c.replay import PRIMARY

def main():
 rows=read(OUT/'structural_replay_rows.json');cmp=pairs(rows,'EMPIRICAL_14D_COUNTS');old=pairs(read(OLD/'screen_rows.json'),PRIMARY)
 oldmap={(p['workload'],p['budget']):p for p in old};rowmap={(r['workload'],r['budget'],r['Specialist_regime']):r for r in rows}
 h=read(OUT/'heavy_empirical_replica_table.json');f=read(OLD/'screen_protocol_freeze.json');path_records=h['task_path_capacity_records'];e=read(OLD/'regime_e.json')
 for p in cmp:
  previous=oldmap[p['workload'],p['budget']];p['old_14C']=previous
  a=rowmap[p['workload'],p['budget'],E];z=rowmap[p['workload'],p['budget'],H]
  oldE=next(r for r in read(OLD/'screen_rows.json') if r['workload']==p['workload'] and r['budget']==p['budget'] and r['Specialist_regime']==E and r['count_scope']==PRIMARY)
  assert a['feasible']==oldE['feasible'] and abs(a['J_star_seconds']-oldE['J_star_seconds'])<=2e-8
  if p['comparable_feasible']:
   p['old_NS_reduction_persists']=previous['T1'] and p['T1'];p['old_NS_reduction_artifact_removed']=previous['T1'] and not p['T1']
   p['old_effective_NS_reduction_persists']=previous.get('effective_positive_flow_T1',False) and p.get('effective_positive_flow_T1',False)
   if previous['T1'] and not p['T1']:p['attribution']='MISSING_REPLICA_EVIDENCE_ARTIFACT: reduction disappears with measured count2; E/pool/context/quality/memory budgets unchanged'
   elif p['T1']:p['attribution']='PERSISTENT_EMPIRICAL_COUNT_SPACE_NS_REDUCTION; Generalist substitution requires positive routing/split evidence'
   else:p['attribution']='REAL_HEAVY_RESOURCE_OR_COMPOSITION_DIFFERENCE_NO_NS_REDUCTION'
  else:
   work=next(w for w in f['workloads'] if w['id']==p['workload']);p['memory_cover']=minimum_real_memory_cover(h,[k for k,v in work['rates'].items() if v>0]);p['attribution']='MEMORY_INFEASIBILITY' if p['memory_cover']['minimum_safe_memory_mib']>a['memory_budget_mib'] else 'UNRESOLVED_CAPACITY_OR_MEMORY'
 # Independent reconstruction of objective, conservation, capacity, memory.
 validation=[]
 for r in rows:
  if not r['feasible']:continue
  data=e if r['Specialist_regime']==E else h;work=next(w for w in f['workloads'] if w['id']==r['workload']);x=r['replicas'];route=r['routing'];D={m:math.fsum(v*data['models'][m]['service_time_ms'][p.split('::')[0]]/1000 for p,v in route.items() if p.split('::')[1]==m) for m in x};rho={m:D[m]/data['capacity_factors'][m][str(x[m])] for m in x};mem=data['safe_factor']*math.fsum(data['memory_table_mib'][m][str(n)] for m,n in x.items());J=math.fsum(D[m]/(1-rho[m]) for m in D)/math.fsum(work['rates'].values());con=max(abs(math.fsum(v for p,v in route.items() if p.split('::')[0]==k)-rate) for k,rate in work['rates'].items())
  assert con<1e-7 and max(rho.values())<=.8+1e-9 and abs(mem-r['safe_memory_mib'])<1e-6 and abs(J-r['J_star_seconds'])<1e-9 and r['global_gap_seconds']<=1e-8
  validation.append({'context':r['workload']+'/'+r['budget'],'regime':r['Specialist_regime'],'conservation_residual':con,'J_residual_seconds':J-r['J_star_seconds'],'certificate_gap_seconds':r['global_gap_seconds']})
 common=[p for p in cmp if p['comparable_feasible']];event_counts={name:sum(p[key] for p in common) for name,key in [('NS_reduction','T1'),('G_share_increase','T2'),('new_GS_split','T3'),('multi_task_G_consolidation','T4'),('NG_increase','T5'),('new_second_G','T6')]}
 event_counts['effective_positive_flow_NS_reduction']=sum(p.get('effective_positive_flow_T1',False) for p in common)
 persistent=[p['workload']+'/'+p['budget'] for p in cmp if p.get('old_NS_reduction_persists')];artifacts=[p['workload']+'/'+p['budget'] for p in cmp if p.get('old_NS_reduction_artifact_removed')];infeas=[p['workload']+'/'+p['budget'] for p in cmp if not p['H_feasible']]
 orig_infeas=[p['workload']+'/'+p['budget'] for p in old if not p['H_feasible']]
 # Minimal x3 evidence decision: do not invent g3/M3 or run that count.
 decisions=[]
 for m in sorted({p['planning_id'] for p in path_records}):
  pr=[p for p in path_records if p['planning_id']==m];active=[]
  for r in rows:
   if r['Specialist_regime']!=H or not r['feasible'] or r['replicas'].get(m)!=2:continue
   tasks=set(h['models'][m]['supported_tasks']);co=[n for n in r['Specialist_identities'] if n!=m and tasks.intersection(h['models'][n]['supported_tasks']) and math.fsum(v for p,v in r['routing'].items() if p.split('::')[1]==n)>1e-9]
   if co and all(p['g2'] is not None and p['g2']>1 for p in pr):active.append({'context':r['workload']+'/'+r['budget'],'rho_x2':r['rho'][m],'other_active_same_task_queues':co,'memory_slack_mib':r['memory_slack_mib'],'evidence':'Measured positive x1->x2 increment; optimum uses x2 at evidence bound with nonzero flow and competing same-task queues. Unknown x3 could change active queue consolidation/J boundary; no numeric g3/M3 assumed.'})
  decisions.append({'planning_id':m,'decision':'X3_PROFILING_REQUIRED' if active else 'X3_NOT_REQUIRED','witnesses':active,'x3_measured':False})
 x3={'at_utc':now(),'decision':'X3_PROFILING_REQUIRED' if any(d['witnesses'] for d in decisions) else 'X3_NOT_REQUIRED','decisions':decisions,'no_x3_measurements_performed':True,'no_linear_extrapolation':True,'scope':'Decision only after completed empirical x2 replay; not a claim of unmeasured capacity'};save('x3_profiling_decision.json',x3)
 active_unprofiled=[]
 for r in rows:
  if r['Specialist_regime']!=H or not r['feasible']:continue
  for m in r['Specialist_identities']:
   if h['evidence_bounds'][m]==1 and math.fsum(v for p,v in r['routing'].items() if p.split('::')[1]==m)>1e-9:active_unprofiled.append({'context':r['workload']+'/'+r['budget'],'planning_id':m})
 events=[p for p in common if p.get('effective_positive_flow_T1') or p['T2'] or p['T3'] or p['T5']]
 # 'Multiple' is at least two independently frozen contexts; effective counts
 # remove numerical near-zero residency artifacts under original event epsilon.
 gate='HEAVY_TRANSITION_EMPIRICALLY_SUPPORTED' if len(events)>=2 else 'HEAVY_TRANSITION_WEAK' if events else 'HEAVY_TRANSITION_NOT_SUPPORTED'
 if active_unprofiled or h['failed_replica_states']:gate='HEAVY_TRANSITION_WEAK'
 result={'at_utc':now(),'original_contexts':27,'E_feasible':sum(p['E_feasible'] for p in cmp),'H_feasible':sum(p['H_feasible'] for p in cmp),'comparable_feasible':len(common),'original_NS_reductions':14,'persistent_original_NS_reduction_count':len(persistent),'persistent_original_NS_reduction_contexts':persistent,'missing_replica_artifact_count':len(artifacts),'missing_replica_artifact_contexts':artifacts,'event_counts':event_counts,'original_H_infeasible':orig_infeas,'current_H_infeasible':infeas,'resolved_H_infeasible':sorted(set(orig_infeas)-set(infeas)),'new_H_infeasible':sorted(set(infeas)-set(orig_infeas)),'comparisons':cmp,'active_unprofiled_specialists':active_unprofiled,'family_capacity_scope':h['family_transfer_limitation'],'baseline_E_all_J_reproduced':True,'independent_metric_checks':validation,'generalist_sharing_gain_claim':bool(event_counts['G_share_increase'] or event_counts['new_GS_split']),'counts_are_not_GS_substitution_without_routing_evidence':True}
 save('structural_transition_comparison.json',result)
 save('module14d_gate.json',{'at_utc':now(),'gate':gate,'rule':'Multiple (>=2) feasible contexts with effective persistent NS reduction, G share increase, new GS split or NG increase; near-zero resident ties excluded.','event_counts':event_counts,'persistent_original_NS_reduction_count':len(persistent),'x3_decision':x3['decision'],'automatic_GPU_validation_required':gate=='HEAVY_TRANSITION_EMPIRICALLY_SUPPORTED','automatic_continuation_status':'REQUIRED_PENDING_BOUNDED_VALIDATION' if gate=='HEAVY_TRANSITION_EMPIRICALLY_SUPPORTED' else 'STOPPED_BY_REVALIDATION_GATE','active_unprofiled_specialists':active_unprofiled,'Generalist_sharing_gain_supported':result['generalist_sharing_gain_claim'],'empirical_family_transfer_limitation':h['family_transfer_limitation']})
 verify(read(OUT/'protected_before.json')['files']);save('completion_integrity.json',{'at_utc':now(),'historical_files_unchanged':True,'protected_file_count':len(read(OUT/'protected_before.json')['files']),'metric_checks_passed':len(validation),'E_J_reproduced_all_27':True})
 summary=f"# Module 14D\n\nGate: **{gate}**.\n\n7 requested task paths across 6 checkpoints: {sum(p['valid_all_three'] for p in path_records)} measured with three independent x2 repetitions each. Historical g1/M1 and existing Turbo/RoBERTa/ConvNeXt evidence preserved.\n\n14C NS reductions: **{len(persistent)}/14** persist; {len(artifacts)} disappear as missing-replica evidence artifacts. E feasible 27/27; H feasible {27-len(infeas)}/27.\n\nG share increases: {event_counts['G_share_increase']}; new GS splits: {event_counts['new_GS_split']}; NG increases: {event_counts['NG_increase']}; new second G: {event_counts['new_second_G']}. Persistent NS reduction alone is not Generalist-sharing gain.\n\nOriginal 8 Heavy infeasible contexts: {len(set(infeas)&set(orig_infeas))} remain; resolved {result['resolved_H_infeasible']}; new {result['new_H_infeasible']}.\n\nx3: {x3['decision']}. Decision only; no x>=3 measured or extrapolated.\n\nShared Whisper-medium stores ASR and LID empirical gains separately. The unchanged family scalar search uses the minimum measured path gain, and maximum measured path residency. Mixed-task capacity remains a conservative transfer surrogate requiring real GPU validation.\n\nAutomatic continuation: {gate=='HEAVY_TRANSITION_EMPIRICALLY_SUPPORTED'}. Results in separate subsequent module directories.\n"
 (OUT/'module14d_summary.md').write_text(summary)
 print('GATE',gate,'PERSISTENT',len(persistent),'EVENTS',event_counts,'X3',x3['decision'],flush=True)
if __name__=='__main__':main()
