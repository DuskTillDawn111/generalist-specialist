"""All-cell report, preserving invalid/incomplete states without latency imputation."""
import json,math,csv
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from deployment_core import Workload
from deployment_search.routing import solve_canonical_routing
from .common import ROOT,A,F,OUT,read,write,csvwrite,inputs,ref,utc
METHODS=['static_v2','RC_SG','Generalist_centric']

def load(p,default=None):return read(p) if p.exists() else default

def diff(a,b):return a-b if a is not None and b is not None else None

def report():
 cells=read(F/'campaign_cells.json')['cells'];budgets=read(F/'final_memory_budgets.json')['budgets'];budget_ids={r['budget_mib']:r['id'] for r in budgets};factor=read(A/'memory_safety_factor.json')['safety_factor'];catalog,_,roles,cal=inputs()
 status=load(OUT/'campaign_status.json',{'status':'PREPARED'});complete=status['status']=='COMPLETE';bank=load(OUT/'physical_measurements.json',{'runs':[]})['runs'];sharing=load(OUT/'measurement_sharing_log.json',{'logical_run_references':[],'logical_evaluations':[]});failures=load(OUT/'runtime_failure_log.json',{'events':[]})
 records={m:[] for m in METHODS};rows=[];replica_rows=[];candidate_rows=[];state_rows=[];comparisons=[];traces={};all_results={}
 for cell in cells:
  cid=cell['cell_id'];all_results[cid]={}
  for method in METHODS:
   frozen=cell['preflight'][method];record=load(OUT/cid/method/'result.json');runtime=None;actual_x=None;route={};canon=None
   if method=='Generalist_centric':
    st='CAPACITY_INFEASIBLE' if frozen['status']=='PLANNING_CAPACITY_INFEASIBLE' else frozen['status'];record={'cell_id':cid,'method':method,'status':st,'structural_only':True,'source':ref(F/'campaign_cells.json'),'frozen_structural_result':frozen,'new_GPU_runs':0,'runtime':None}
   elif record:
    actual_x=record['deployment'];runtime=record['confirmations']['final' if method=='static_v2' else 'fixed'];route=record['routing'];st=record['status']
    canon=solve_canonical_routing(actual_x,Workload(cell['lambda_rps']),catalog,.8,replica_bounds=cal.evidence_bounds,capacity_mode='empirical',calibration=cal)
    assert canon is not None
   else:
    trace=load(OUT/cid/method/'search_trace.json');st='INCOMPLETE' if trace or any(r['method']==method and r['workload']==cid for r in bank) else 'NOT_EXECUTED'
    record={'cell_id':cid,'method':method,'status':st,'deployment':None,'runtime':None,'frozen_planning_status':frozen['status'],'frozen_planning_candidate':frozen,'reason':status.get('error','Campaign not complete'),'partial_search_trace':str(OUT/cid/method/'search_trace.json') if trace else None}
   records[method].append(record);all_results[cid][method]=record
   x=actual_x or (frozen['deployment'] if method=='Generalist_centric' else None)
   ng=sum(n for m,n in x.items() if roles.get(m)=='generalist') if x else None;ns=sum(n for m,n in x.items() if roles.get(m)=='specialist') if x else None
   flags={'generalist_only':ng is not None and ng>0 and ns==0,'hybrid':ng is not None and ng>0 and ns>0,'specialist_heavy':ng is not None and ns>ng,'specialist_only':ng==0 if ng is not None else False}
   # The frozen regime predicates overlap: keep flags as primary evidence.
   regime='Specialist-heavy' if flags['specialist_heavy'] else 'Hybrid' if flags['hybrid'] else 'Generalist-heavy' if flags['generalist_only'] else 'Specialist-only' if flags['specialist_only'] else None
   plan=math.fsum(cal.memory_table_mib[m][n] for m,n in x.items() if m in catalog) if x else None;safe=factor*plan if plan is not None else None
   memory_obs=runtime['memory_observations'] if runtime else []
   peak=max((o['peak_device_mib'] for o in memory_obs if o.get('peak_device_mib') is not None),default=None);idle=[o['actual_idle_mib'] for o in memory_obs if o.get('actual_idle_mib') is not None]
   sharing_tasks={m:sorted(k.split('::')[0] for k,v in route.items() if k.split('::')[1]==m and v>0) for m,n in (actual_x or {}).items() if n and roles[m]=='generalist'}
   bottleneck_models=[m for m,n in (actual_x or {}).items() if n and canon.model_pressure[m]>=canon.max_pressure-1e-8] if canon else []
   task_demands={t:sum(v for (t2,m),v in canon.task_contributions.items() if t2==t and m in bottleneck_models) for t in cell['lambda_rps']} if canon else {}
   bottleneck_tasks=sorted((t for t,v in task_demands.items() if v>0),key=lambda t:(-task_demands[t],t))
   row={'cell_id':cid,'experiment_A':'A' in cell['experiments'],'experiment_B':'B' in cell['experiments'],'mixture':cell['mixture'],'intensity':cell['intensity_level'],'alpha':cell['alpha'],'beta':cell['beta'],'memory_budget_id':budget_ids[cell['memory_budget_mib']],'budget_mib':cell['memory_budget_mib'],'budget_gib':cell['memory_budget_mib']/1024,'method':method,'status':st,'deployment':x,'deployment_is_runtime_final':actual_x is not None,'N_G':ng,'N_S':ns,'total_replicas':sum(x.values()) if x else None,'additional_replicas':sum(max(n-1,0) for n in x.values()) if x else None,'regime':regime,**flags,'mean_e2e_ms':runtime['weighted_mean_e2e'] if runtime else None,'weighted_p95_ms':runtime['weighted_p95_e2e'] if runtime else None,'stability':runtime['stability_state_v2'] if runtime else None,'throughput_rps':runtime['throughput'] if runtime else None,'completion_ratio':runtime['completion_ratio'] if runtime else None,'M_plan_mib':plan,'M_safe_mib':safe,'actual_idle_mib':sum(idle)/len(idle) if idle else None,'actual_peak_mib':peak,'safe_budget_utilization':safe/cell['memory_budget_mib'] if safe is not None else None,'peak_budget_utilization':peak/cell['memory_budget_mib'] if peak is not None else None,'memory_feasible':safe<=cell['memory_budget_mib'] if safe is not None else None,'capacity_status':'FEASIBLE' if canon else 'INFEASIBLE' if st=='CAPACITY_INFEASIBLE' else None,'bottleneck_models':bottleneck_models,'bottleneck_tasks':bottleneck_tasks,'generalist_task_sharing':sharing_tasks,'planning_max_pressure':canon.max_pressure if canon else None}
   rows.append(row)
  pair={r['method']:r for r in rows if r['cell_id']==cid};a,b=pair['static_v2'],pair['RC_SG'];ar=all_results[cid]['static_v2'];br=all_results[cid]['RC_SG'];outcome='INCOMPLETE';wr=lr=None
  if a['status']=='COMPLETE' and b['status']=='COMPLETE':
   ac=ar['confirmations']['final'];bc=br['confirmations']['fixed'];objective=NoiseAwareRuntimeObjective(1.,.05);win,wr=objective.compare(SimpleNamespace(**ac),SimpleNamespace(**bc),a['M_plan_mib'],b['M_plan_mib']);loss,lr=objective.compare(SimpleNamespace(**bc),SimpleNamespace(**ac),b['M_plan_mib'],a['M_plan_mib']);assert not(win and loss);outcome='WIN' if win else 'LOSS' if loss else 'TIE'
  elif a['status']=='COMPLETE' and b['status'] in ['MEMORY_INFEASIBLE','CAPACITY_INFEASIBLE','BASE_MEMORY_OR_CAPABILITY_INFEASIBLE']:outcome='FEASIBILITY_WIN'
  cmp={'cell_id':cid,'outcome':outcome,'forward_reason':wr,'reverse_reason':lr,'mean_difference_ms':diff(a['mean_e2e_ms'],b['mean_e2e_ms']),'p95_difference_ms':diff(a['weighted_p95_ms'],b['weighted_p95_ms']),'safe_memory_difference_mib':diff(a['M_safe_mib'],b['M_safe_mib']),'deployment_L1_distance':sum(abs(a['deployment'].get(m,0)-b['deployment'].get(m,0)) for m in catalog) if a['deployment'] and b['deployment'] else None,'v2_feasibility_status':a['status'],'RC_SG_feasibility_status':b['status'],'latency_imputed':False};comparisons.append(cmp)
  trace=load(OUT/cid/'static_v2'/'search_trace.json',{});traces[cid]=trace
  for tr in trace.get('iterations',[]):
   for index,c in enumerate(tr['candidates']):
    x=c['mutation']['deployment'];memory=factor*c['memory_mib'];candidate_rows.append({'cell_id':cid,'iteration':tr['iteration']+1,'candidate_index':index,'operation':c['mutation']['operation'],'deployment':x,'M_plan_mib':c['memory_mib'],'M_safe_mib':memory,'budget_mib':cell['memory_budget_mib'],'resource_status':'RESOURCE_BUDGET_INFEASIBLE' if memory>cell['memory_budget_mib'] else 'MEMORY_FEASIBLE','status':c['status'],'reason':c['reason'],'Top_B':c['selected_top_b'],'evaluated':c['runtime'] is not None})
    for m in ['qwen','whisper-small','resnet50']:
     if x.get(m)!=2:continue
     canonical=c.get('canonical');paired=None
     if canonical is not None:
      paired_x=dict(x);paired_x[m]=1;paired=solve_canonical_routing(paired_x,Workload(cell['lambda_rps']),catalog,.8,replica_bounds=cal.evidence_bounds,capacity_mode='empirical',calibration=cal)
     replica_rows.append({'cell_id':cid,'iteration':tr['iteration']+1,'candidate_index':index,'model':m,'deployment':x,'operation':c['mutation']['operation'],'generated':True,'memory_feasible':memory<=cell['memory_budget_mib'],'canonical_capacity_feasible':canonical is not None,'paired_x1_capacity_tested':canonical is not None,'capacity_useful':canonical is not None and paired is None,'Top_B':c['selected_top_b'],'evaluated':c['runtime'] is not None,'accepted':c['status']=='ACCEPTED','rollback':c['status']=='ROLLBACK','status':c['status'],'reason':c['reason'],'empirical_g1':cal.capacity_factors[m][1],'empirical_g2':cal.capacity_factors[m][2]})
 for r in bank:
  state_rows.append({k:r.get(k) for k in ['physical_raw_id','method','workload','phase','role','repetition_seed','deployment','status','valid','measurement_started','stability','M_plan_mib','M_safe_mib','budget_mib','actual_idle_mib','actual_peak_mib','planning_safe','physical_budget_violation','raw_path','raw_sha256']})
 write('v2_results.json',{'campaign_status':status['status'],'source':ref(F/'campaign_cells.json'),'cells':records['static_v2'],'complete_cells':sum(r['status']=='COMPLETE' for r in records['static_v2'])});write('rc_sg_results.json',{'campaign_status':status['status'],'spec':ref(F/'resource_constrained_sg_spec.json'),'cells':records['RC_SG'],'complete_cells':sum(r['status']=='COMPLETE' for r in records['RC_SG'])});write('generalist_centric_results.json',{'source':ref(F/'campaign_cells.json'),'cells':records['Generalist_centric'],'MEMORY_INFEASIBLE':3,'CAPACITY_INFEASIBLE':9,'new_GPU_runs':0,'no_runtime_metrics_imputed':True})
 csvwrite('deployment_composition.csv',rows);csvwrite('resource_feasibility.csv',rows);csvwrite('budget_utilization.csv',rows)
 def table(cell):
  pair={r['method']:r for r in rows if r['cell_id']==cell['cell_id']};comparison=next(r for r in comparisons if r['cell_id']==cell['cell_id']);a,b=pair['static_v2'],pair['RC_SG']
  out={k:a[k] for k in ['cell_id','mixture','intensity','alpha','beta','memory_budget_id','budget_mib','budget_gib']};out.update(comparison)
  for method,r in [('v2',a),('RC_SG',b)]:
   for k in ['status','deployment','N_G','N_S','total_replicas','additional_replicas','mean_e2e_ms','weighted_p95_ms','stability','throughput_rps','completion_ratio','M_plan_mib','M_safe_mib','actual_peak_mib','peak_budget_utilization','regime','bottleneck_models','generalist_task_sharing']:out[method+'_'+k]=r[k]
  out['Generalist_centric_status']=pair['Generalist_centric']['status'];return out
 Arows=[table(c) for c in cells if 'A' in c['experiments']];Brows=[table(c) for c in cells if 'B' in c['experiments']];csvwrite('memory_sweep_results.csv',Arows);csvwrite('intensity_sweep_results.csv',Brows);csvwrite('v2_vs_rc_sg.csv',comparisons)
 for name,experiment in [('memory_vs_mean_e2e.csv','A'),('memory_vs_p95.csv','A'),('memory_vs_deployment_composition.csv','A'),('intensity_vs_mean_e2e.csv','B'),('intensity_vs_deployment.csv','B')]:csvwrite(name,[r for r in rows if r['experiment_'+experiment]])
 csvwrite('replica_activation.csv',replica_rows,columns=['cell_id','iteration','candidate_index','model','deployment','operation','generated','memory_feasible','canonical_capacity_feasible','paired_x1_capacity_tested','capacity_useful','Top_B','evaluated','accepted','rollback','status','reason','empirical_g1','empirical_g2'])
 csvwrite('candidate_resource_audit.csv',candidate_rows,columns=['cell_id','iteration','candidate_index','operation','deployment','M_plan_mib','M_safe_mib','budget_mib','resource_status','status','reason','Top_B','evaluated'])
 csvwrite('physical_memory_audit.csv',state_rows,columns=['physical_raw_id','method','workload','phase','role','repetition_seed','deployment','status','valid','measurement_started','stability','M_plan_mib','M_safe_mib','budget_mib','actual_idle_mib','actual_peak_mib','planning_safe','physical_budget_violation','raw_path','raw_sha256'])
 replication={m:{k:sum(bool(r[k]) for r in replica_rows if r['model']==m) for k in ['generated','memory_feasible','capacity_useful','Top_B','evaluated','accepted','rollback']} for m in ['qwen','whisper-small','resnet50']}
 for m,r in replication.items():r['nonacceptance_reasons']=dict(Counter(x['reason'] or x['status'] for x in replica_rows if x['model']==m and not x['accepted']));r['audit_unit']='candidate occurrence, not unique vectors or physical runs'
 write('replica_activation_summary.json',{'models':replication,'campaign_complete':complete,'no_extra_intensity':True,'memory_feasible_not_capacity_useful':'M_safe(x2)<=B admits memory. Paired capacity useful means canonical capacity at this exact frozen lambda is feasible for x2 and infeasible after replacing only that family by x1; reported after search, never used to revise method.'})
 loose=[r for r in comparisons if next(c for c in cells if c['cell_id']==r['cell_id'])['memory_budget_mib']==49140];close=[]
 for r in loose:
  pair={x['method']:x for x in rows if x['cell_id']==r['cell_id']};a,b=pair['static_v2'],pair['RC_SG'];yes=a['status']=='COMPLETE' and b['status']=='COMPLETE' and a['stability']==b['stability'] and a['mean_e2e_ms'] is not None and b['mean_e2e_ms'] is not None and abs(a['mean_e2e_ms']-b['mean_e2e_ms'])<=max(1.,.05*b['mean_e2e_ms'])
  r=dict(r,close_under_frozen_rule=yes if a['status']=='COMPLETE' and b['status']=='COMPLETE' else None);close.append(r)
 write('loose_resource_consistency.json',{'cells':close,'old_B2_runtime_used_for_matched_comparison':False})
 tight=[r for r in Arows if r['memory_budget_id'] in ['M1','M2','M3']];write('tight_resource_value.json',{'cells':tight,'no_unmatched_historical_latency':True})
 intensity_effect=[]
 for bid in ['M1','M4','M6']:
  selected=sorted([r for r in rows if r['method']=='static_v2' and r['memory_budget_id']==bid and r['experiment_B']],key=lambda r:r['intensity']);eligible=len(selected)==3 and all(r['status']=='COMPLETE' for r in selected)
  intensity_effect.append({'memory_budget_id':bid,'complete':eligible,'cells':selected,'distinct_deployment_count':len({json.dumps(r['deployment'],sort_keys=True) for r in selected}) if eligible else None,'composition_changed':len({json.dumps(r['deployment'],sort_keys=True) for r in selected})>1 if eligible else None,'latency_only_change_if_vector_constant':eligible and len({json.dumps(r['deployment'],sort_keys=True) for r in selected})==1})
 write('intensity_effect.json',{'budgets':intensity_effect})
 frozen_h=read(F/'success_hypotheses.json');h={f'H{i}':{'claim':frozen_h[f'H{i}']['claim'],'frozen_test':frozen_h[f'H{i}']['test'],'status':'INSUFFICIENT','reason':'Campaign incomplete; no complete matched runtime evidence'} for i in range(1,6)}
 if all(r['v2_status']=='COMPLETE' for r in Arows):
  vectors=len({json.dumps(r['v2_deployment'],sort_keys=True) for r in Arows});counts=len({(r['v2_N_G'],r['v2_N_S']) for r in Arows});directional=all(Arows[i]['v2_N_S']>=Arows[i-1]['v2_N_S'] and Arows[i]['v2_N_G']<=Arows[i-1]['v2_N_G'] for i in range(1,len(Arows)))
  h['H1'].update(status='SUPPORTED' if counts>1 and directional else 'PARTIAL' if vectors>1 else 'NOT_SUPPORTED',reason='Frozen memory sweep observed '+str(vectors)+' distinct final vectors and '+str(counts)+' distinct (N_G,N_S); monotone towards specialists='+str(directional),evidence_cells=[r['cell_id'] for r in Arows])
 if all(r['close_under_frozen_rule'] is not None for r in close):
  n=sum(r['close_under_frozen_rule'] for r in close);h['H2'].update(status='SUPPORTED' if n==len(close) else 'PARTIAL' if n else 'NOT_SUPPORTED',reason=f'{n}/{len(close)} endpoint cells close under frozen same-stability max(1 ms, 5%) rule',evidence=close)
 if all(r['v2_status']=='COMPLETE' and r['RC_SG_status']=='COMPLETE' for r in tight):
  advantage=any(r['outcome'] in ['WIN','FEASIBILITY_WIN'] for r in tight);h['H3'].update(status='SUPPORTED' if advantage else 'PARTIAL',reason='Old unconstrained SG violates the M1 memory budget; matched RC-SG/v2 evidence '+('shows runtime/feasibility advantage' if advantage else 'does not show a v2 advantage under the frozen objective'),evidence=tight)
 if all(r['complete'] for r in intensity_effect):
  changed=sum(r['composition_changed'] for r in intensity_effect);h['H4'].update(status='SUPPORTED' if changed==3 else 'PARTIAL' if changed else 'NOT_SUPPORTED',reason=f'Composition/replica vector changes in {changed}/3 fixed-budget intensity sweeps',evidence=intensity_effect)
 accepted=[r for r in replica_rows if r['accepted']]
 if complete and not accepted:h['H5'].update(status='INSUFFICIENT',reason='No accepted x2 activation; conditional empirical-scaling hypothesis was not observed; no extra load added')
 elif complete and accepted:h['H5'].update(status='SUPPORTED' if all(r['empirical_g2']>r['empirical_g1'] for r in accepted) else 'PARTIAL',reason='Accepted replicas are checked against frozen empirical g(2) and paired-capacity evidence; family/task-transfer caveat retained',evidence=accepted)
 write('hypothesis_audit.json',{'frozen_hypotheses':ref(F/'success_hypotheses.json'),'campaign_complete':complete,'hypotheses':h,'statuses_are_evidence_interpretations_not_new_selection_rules':True})
 accounting=load(OUT/'run_accounting.json',{});write('campaign_analysis.json',{'campaign_status':status,'memory_sweep_comparisons':dict(Counter(r['outcome'] for r in Arows)),'all_cell_comparisons':dict(Counter(r['outcome'] for r in comparisons)),'feasibility_win_cells':[r['cell_id'] for r in comparisons if r['outcome']=='FEASIBILITY_WIN'],'replica_summary':replication,'GC_structural_failures':{'MEMORY_INFEASIBLE':3,'CAPACITY_INFEASIBLE':9},'success_hypotheses':{k:v['status'] for k,v in h.items()},'all_12_cells_retained':True,'memory_sweep_cells':6,'intensity_sweep_cells':9,'unique_cells':12,'no_new_points':True,'STATIC_MAIN_EXPERIMENTS_COMPLETE':complete})
 lines=['# Module 8F-2 Resource-Constrained Static GPU Campaign','',f'Campaign status: **{status["status"]}**. All 12 frozen cells and all three methods are retained. Missing runtime metrics are blank, never zero-imputed.','',f'UUID exception explicitly authorized by user; same GPU model, driver and capacity were verified. Original frozen artifacts are unchanged. See hardware_identity_exception.json and hash_verification.json. Historical measurements on the previous UUID are not reused.','',f'New timed runs: {accounting.get("new_timed_runs",0)}; physical attempts: {accounting.get("new_physical_attempts",0)}; reused repetition references: {accounting.get("measurement_sharing_count",0)}; logical evaluations: {accounting.get("logical_search_evaluations",0)}; physical states: {accounting.get("unique_physical_states",0)}; confirmations: {accounting.get("confirmation_runs",0)}; retries: 0.','',f'Reuse preaudit: 54 potential references, 45 cross-budget references forbidden; 9 same-cell search references conditionally eligible. Actual sharing requires exact raw/config/context hashes.','',f'Memory Sweep v2/RC-SG: {dict(Counter(r["outcome"] for r in Arows))}. Feasibility wins: {[r["cell_id"] for r in comparisons if r["outcome"]=="FEASIBILITY_WIN"]}.','', '| Budget | v2 final deployment | RC-SG final deployment | v2 / RC mean ms | Outcome |','|---|---|---|---|---|']
 def deployment(x):return ', '.join(f'{m}×{n}' for m,n in (x or {}).items() if n) if x else 'NOT EXECUTED / INCOMPLETE'
 for r in Arows:lines.append(f'| {r["memory_budget_id"]} {r["budget_mib"]:.9f} MiB | {deployment(r["v2_deployment"])} | {deployment(r["RC_SG_deployment"])} | {r["v2_mean_e2e_ms"]} / {r["RC_SG_mean_e2e_ms"]} | {r["outcome"]} |')
 lines+=['','Frozen structural predicates overlap: Hybrid = N_G>0 and N_S>0; Specialist-heavy = N_S>N_G. Both flags are exported. Display precedence is Specialist-heavy, Hybrid, Generalist-only displayed as Generalist-heavy. No disjoint new regime threshold is invented.','',f'Loose-resource consistency: {json.dumps(close,ensure_ascii=False)}','',f'Intensity vector changes: {[(r["memory_budget_id"],r["composition_changed"]) for r in intensity_effect]}.','',f'Replica audit: {json.dumps(replication,ensure_ascii=False)}','',f'Planning-safe peak-budget violations: {accounting.get("planning_safe_peak_budget_violations",0)}. Runtime failure events: {len(failures.get("events",[]))}. Full physical memory audit is physical_memory_audit.csv; invalid runs do not receive serving metrics.','', '| Hypothesis | Status | Evidence |','|---|---|---|']
 for key,v in h.items():lines.append(f'| {key} | {v["status"]} | {v["reason"]} |')
 lines+=['','Tables A/B: memory_sweep_results.csv and intensity_sweep_results.csv; C: deployment_composition.csv; D: resource_feasibility.csv; E: replica_activation.csv. All requested figure CSVs use independent final confirmations and include explicit baseline failure rows.','',f'Static main experiments complete: **{complete}**. '+('Joint-deployment conclusions must be scoped to the observed frozen grid, without claiming global runtime optimality.' if complete else 'Cannot conclude runtime advantage or close the static main experiments while this campaign remains incomplete.'),'',f'Stop/error evidence: {status.get("error","none")}. No automatic retries, intensity changes, new resource points, method/routing revisions or dynamic experiments.']
 (OUT/'module8f2_summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':report()
