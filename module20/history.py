"""Read all six historical modules; no imported GPU execution code."""
from .common import *
from .optimizer import components,predict_scenarios
import numpy as np,statistics
from types import SimpleNamespace
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS
DIRECTORIES={14:'module14f_memory_ocr_x3',15:'module15_three_phase',16:'module16_fullpool_gpu',17:'module17_runtime_capacity',18:'module18_cross_task',19:'module19_strategy_benchmark'}

def catalog_for(module,raw,manifest):
 if module==19:
  cs=read(ROOT/'results/module19_strategy_benchmark/input_protocol_freeze.json')['cases'];cid=raw['slot'].get('owner_case',raw['slot']['block_id']);return read(next(c['catalog']['path'] for c in cs if c['case_id']==cid))
 if module==15:return read(ROOT/'results/module15_three_phase/existing_empirical_catalog.json')
 return read(manifest['model_pools']['FULL_HEAVY_19']['path'])

def analyze(raw,module,path,definition,manifest):
 d=catalog_for(module,raw,manifest);slot=raw['slot'];phasekey='low_load' if module==14 else 'measurement';ph=raw['phases'][phasekey];rs=ph['request_records'];w=ph['window'];rates=slot['lambda_rps'];x=slot['deployment'];routing=slot['routing'];refs=manifest['latency_reference_ms'];weights=definition['weights'];horizon=w.get('planned_duration_seconds',slot.get('measurement_seconds',w['duration_seconds']));budget=slot.get('safe_memory_budget_mib',49140.)
 factor=d['safe_factor'];idle=[statistics.mean(s['owned_used_mib'] for s in raw[k]) for k in ['postwarm_idle','postmeasurement_idle'] if raw.get(k)];memory=factor*max(idle) if len(idle)==2 else None
 done=[r for r in rs if r['status']=='completed'];taskstats={};paths=[]
 for t,q in rates.items():
  rr=[r for r in rs if r['task']==t];dd=[r for r in rr if r['status']=='completed'];v=[r['e2e_latency_ms'] for r in dd]
  if v:taskstats[t]={'mean_ms':statistics.mean(v),'p95_ms':float(np.percentile(v,95)),'offered':len(rr),'completed':len(dd),'within_window':sum(r['completion_time_ns']<=w['end_time_ns'] for r in dd)}
  for m in x:
   md=[r for r in dd if raw['replica_to_family'][r['model']]==m]
   if md:paths.append({'run_id':slot['run_id'],'module':module,'task':t,'model':m,'s0_ms':d['models'][m]['service_time_ms'][t],'mean_service_ms':statistics.mean(r['worker_latency_ms'] for r in md),'service_ratio':statistics.mean(r['worker_latency_ms'] for r in md)/d['models'][m]['service_time_ms'][t],'mean_queue_ms':statistics.mean(r['worker_queue_wait_ms'] for r in md),'mean_E2E_ms':statistics.mean(r['e2e_latency_ms'] for r in md)})
 workers=[]
 for worker in raw['worker_identities']:
  wr=[r for r in rs if r['model']==worker];m=raw['replica_to_family'][worker]
  if not wr:continue
  tmp={'request_records':wr,'measurement_window':w,'status':'completed','errors':raw.get('errors',[]),'drain_timeout_seconds':60,'worker_identities':{worker:raw['worker_identities'][worker]}}
  ev=reconstruct(tmp,{worker:SimpleNamespace(service_time_ms=d['models'][m]['service_time_ms'])},DEFAULT_THRESHOLDS);cl=classify(ev,DEFAULT_THRESHOLDS)
  workers.append({'worker':worker,'state':cl['stability_state_v2'],'reasons':cl.get('hard_reasons',[])+cl.get('congestion_warning_reasons',[]),'tail_slope_rps':ev.get('tail_backlog_slope'),'end_backlog':ev.get('end_backlog_requests'),'final_backlog':ev.get('final_backlog_after_drain'),'mean_service_ms':statistics.mean(r['worker_latency_ms'] for r in wr if r['status']=='completed') if any(r['status']=='completed' for r in wr) else None})
 st=ph['stability']['stability_state_v2'];ev=ph['stability_evidence'];within=sum(r['completion_time_ns']<=w['end_time_ns'] for r in done);T=1-min(1.,len(done)/len(rs),within/(sum(rates.values())*horizon));quality=all(r['task'] in d['models'][raw['replica_to_family'][r['model']]]['supported_tasks'] for r in rs)
 full=len(done)==len(rs);growth=any(v['state']=='unstable' for v in workers) or st=='unstable';coverage=set(taskstats)==set(rates)
 eligible=bool(raw.get('valid_session',False) and full and not growth and quality and coverage and memory is not None and memory<=budget+1e-6 and not raw.get('errors'))
 norms=components(taskstats,rates,refs,T,memory/budget,weights) if coverage and memory is not None else {'U':None}
 pred=predict_scenarios(d,x,routing,rates,budget,refs,weights)
 # Recompute J and nominal family rho, including runs lacking old CPU_J.
 rho={m:sum(routing.get(t,{}).get(m,0)*d['models'][m]['service_time_ms'][t]/1000 for t in rates if t in d['models'][m]['supported_tasks'])/d['capacity_factors'][m][str(n)] for m,n in x.items()}
 demand={m:rho[m]*d['capacity_factors'][m][str(n)] for m,n in x.items()}
 J=sum(demand[m]/(1-rho[m]) for m in x)/sum(rates.values()) if max(rho.values())<1 else None
 row={'module':module,'run_id':slot['run_id'],'config_id':slot.get('config_id',slot['run_id'].rsplit('_rep',1)[0]),'case_id':slot.get('owner_case',slot.get('block_id',slot.get('target_task','')+'_'+slot.get('stage','').lower())),'repetition':slot['repetition'],'phase':phasekey,'split':'RETROSPECTIVE_HOLDOUT' if module==16 else 'DEVELOPMENT','deployment':x,'routing':routing,'offered_QPS':rates,'budget_mib':budget,'observed_safe_memory_mib':memory,'CPU_predicted_J_ms':J*1000 if J is not None else None,'CPU_predicted_demand_seconds_per_second':demand,'CPU_predicted_rho':rho,'CPU_logged_rho':slot.get('CPU_rho'),'actual_system_mean_ms':statistics.mean(r['e2e_latency_ms'] for r in done) if done else None,'actual_pooled_request_P95_ms':float(np.percentile([r['e2e_latency_ms'] for r in done],95)) if done else None,'task_E2E':taskstats,'actual_within_window_QPS':within/horizon,'offered':len(rs),'completed':len(done),'planned_measurement_seconds':horizon,'observed_window_seconds':w['duration_seconds'],'system_stability':st,'worker_states':workers,'end_backlog':ev.get('end_backlog_requests'),'tail_backlog_slope_rps':ev.get('tail_backlog_slope'),'final_backlog':ev.get('final_backlog_after_drain'),'censored':not full,'eligible':eligible,'quality_admission_passed':quality,**norms,'U':norms.get('U') if eligible else None,'diagnostic_completed_subset_U_NOT_sortable':norms.get('U') if not eligible else None,'predicted_U':pred['U'],'predicted_components':{k:pred.get(k) for k in ['L_norm','P_norm','T_norm','M_norm','L_contribution','P_contribution','T_contribution','M_contribution']},'predicted_max_rho':pred.get('max_rho'),'predicted_task_E2E':pred.get('per_task'),'scenario_U_low':pred['scenario_U_low'],'scenario_U_high':pred['scenario_U_high'],'stress_all_eligible':pred['stress_all_eligible'],'uncertainty':pred['uncertainty'],'raw_path':str(path),'raw_sha256':ref(path)['sha256']}
 if module==14:
  row['capacity_profile_not_independent_validation']=True
  # Isolated profile has no deployment utility budget. Do not invent one.
  row.update(budget_mib=None,M_norm=None,M_contribution=None,U=None,predicted_U=None,scenario_U_low=None,scenario_U_high=None,eligible=False,diagnostic_completed_subset_U_NOT_sortable=None)
  row['utility_unavailable_reason']='Isolated profiling slot has no frozen deployment budget; g/M are predictor inputs, not system utility validation.'
 return row,paths

def main():
 defi=read(OUT/'utility_definition.json');manifest=read(OUT/'utility_reference_manifest.json');rows=[];paths=[]
 for module,dirname in DIRECTORIES.items():
  for p in sorted((ROOT/'results'/dirname/'raw_runs').glob('*.json')):
   row,pr=analyze(read(p),module,p,defi,manifest);rows.append(row);paths.extend(pr)
  print('HISTORY',module,len([r for r in rows if r['module']==module]),flush=True)
 csvsave('historical_utility_analysis.csv',rows);csvsave('historical_path_diagnosis.csv',paths);save('historical_analysis.json',rows)
 # Only same offered workload contrasts in Module19, physical configs once.
 validations=[];diagnoses=[]
 f=read(ROOT/'results/module19_strategy_benchmark/frozen_benchmark_manifest.json')
 configs={c['config_id']:c for c in f['configurations']}
 comparisons=[c for c in f['comparisons'] if c['kind']=='FIXED_RESIDENCY_ROUTING_ABLATION']
 for c in f['comparisons']:
  if c['kind']!='DEPLOYMENT_STRATEGY' or c['control_label']!='BASELINE_B_S_ORIENTED' or not c.get('control_id'):continue
  a,b=configs[c['proposed_id']],configs[c['control_id']];target=a['target_task'];data=read(next(z['catalog']['path'] for z in f['cases'] if z['case_id']==c['case_id']))
  ar={data['models'][m]['role'] for m,q in a['routing'][target].items() if q>0};br={data['models'][m]['role'] for m,q in b['routing'][target].items() if q>0}
  if a['deployment']==b['deployment'] and ar=={'generalist','specialist'} and br=={'specialist'} and all(a['routing'][t]==b['routing'][t] for t in a['rates'] if t!=target) and not any(z['case_id']==c['case_id'] and z['control_id']==c['control_id'] for z in comparisons):comparisons.append(c)
 for comparison in comparisons:
  cid=comparison['case_id'];deltas=[]
  for rep in [1,2,3]:
   a=next((r for r in rows if r['module']==19 and r['config_id']==comparison['proposed_id'] and r['repetition']==rep),None)
   b=next((r for r in rows if r['module']==19 and r['config_id']==comparison['control_id'] and r['repetition']==rep),None)
   if not a or not b:continue
   actual=a['U']-b['U'] if a['U'] is not None and b['U'] is not None else None;new=a['predicted_U']-b['predicted_U'] if a['predicted_U'] is not None and b['predicted_U'] is not None else None;old=b['CPU_predicted_J_ms']-a['CPU_predicted_J_ms'] if a['CPU_predicted_J_ms'] is not None and b['CPU_predicted_J_ms'] is not None else None
   validations.append({'case_id':cid,'repetition':rep,'split':'MODULE19_DEVELOPMENT_NOT_INDEPENDENT','GS_run':a['run_id'],'S_run':b['run_id'],'control_operationally_legal':comparison['control_operationally_legal'],'actual_delta_U_GS_minus_S':actual,'predicted_delta_U_GS_minus_S':new,'old_J_gain_ms_GS':old,'old_J_U_reversal':old*actual<0 if actual is not None and old is not None else None,'new_ranking_correct':new*actual>0 if actual is not None and new is not None else None,'predicted_delta_error':new-actual if actual is not None and new is not None else None,'uncertainty':'HIGH; scenario range is not a statistical bound','high_confidence_prediction':False,'eligible':actual is not None})
   if actual is not None:deltas.append(actual)
  diagnoses.append({'case_id':cid,'valid_pairs':len(deltas),'delta_U_GS_minus_S_values':deltas,'mean_delta_U':statistics.mean(deltas) if deltas else None,'J_U_reversal_count':sum(r['old_J_U_reversal'] is True for r in validations if r['case_id']==cid),'new_correct_count':sum(r['new_ranking_correct'] is True for r in validations if r['case_id']==cid),'control_operationally_legal':comparison['control_operationally_legal']})
 for r in rows:
  if r['module']!=16:continue
  validations.append({'case_id':r['case_id'],'run_id':r['run_id'],'split':'RETROSPECTIVE_HOLDOUT_MODULE16','actual_U':r['U'],'predicted_U':r['predicted_U'],'absolute_U_error':r['predicted_U']-r['U'] if r['U'] is not None and r['predicted_U'] is not None else None,'actual_eligible':r['eligible'],'predicted_nominal_eligible':r['predicted_U'] is not None,'stress_all_eligible':r['stress_all_eligible'],'scenario_contains_actual_U':r['scenario_U_low']<=r['U']<=r['scenario_U_high'] if r['U'] is not None and r['scenario_U_low'] is not None else None,'independent_matched_ranking_pair':False,'ranking_accuracy':None,'reason':'One deployed configuration per frozen offered workload; repeats are not alternative deployment-ranking samples.'})
 csvsave('prediction_ranking_validation.csv',validations);save('diagnosis_and_validation.json',{'routing_diagnoses':diagnoses,'independent_matched_ranking_groups':0,'utility_model_holdout_supported':False,'existing_real_benefit_evidence':'Module15/18 show steady selected deployments and structural behavior, not a matched system-utility advantage; Module14F replica profiles measure capacity/resource tradeoffs but are predictor inputs, not independent ranking tests. No independently demonstrated GS utility gain can be honestly counted as B success.','unexplained_errors':'Standalone service change, sample variability, cross-model/CPU/GPU interference not identified causally; FIFO mean service misses these. No global fitted correction.','module17_scope':'Local mean envelopes have original independent support only at the exact original residency/count/task/routing conditions; neither applied to new combinations nor counted as utility-ranking validation.'})
if __name__=='__main__':main()
