"""Audit frozen real inputs and memory; never choose a latency SLO from outcomes."""
import copy
from module14f.common import ROOT, ref, read, verify, now
from module14c.common import catalog
from pathlib import Path
import json,csv
OUT=ROOT/'results/module15_three_phase'
F=ROOT/'results/module14f_memory_ocr_x3'

def save(name,value):
 (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
 assert not (OUT/'input_audit.json').exists(),'Existing preparation is immutable; reuse it.'
 previous=read(F/'artifact_manifest.json');verify(previous['artifacts']+previous['implementation'])
 protected=read(F/'protected_before.json')['files']+previous['artifacts']+previous['implementation']+[ref(F/'artifact_manifest.json')]
 protected=sorted({r['path']:r for r in protected}.values(),key=lambda r:r['path']);verify(protected)
 save('protected_before.json',{'at_utc':now(),'files':protected,'verified_before':True})
 e=read(F/'regime_e_reconciled.json');h=read(F/'heavy_empirical_x3_reconciled.json')
 catalog(e);catalog(h)
 union=copy.deepcopy(e)
 for m in h['models']:
  if m in union['models']:
   for key in ['memory_table_mib','capacity_factors','evidence_bounds']:assert union[key][m]==h[key][m],(m,key)
   for key in ['role','supported_tasks','service_time_ms','idle_memory_mib']:assert union['models'][m][key]==h['models'][m][key],(m,key)
  else:
   for key in ['models','memory_table_mib','capacity_factors','evidence_bounds','model_evidence']:union[key][m]=copy.deepcopy(h[key][m])
 union['source_hashes']=e['source_hashes']+h['source_hashes'] if isinstance(e['source_hashes'],list) else {**e['source_hashes'],**h['source_hashes']}
 union['module']='15';union['regime']='UNION_OF_EXISTING_FROZEN_E_H_IDENTITIES_FOR_MODULE15_EXISTENCE_TEST'
 union['pool_policy']='Only previously quality-admitted measured task paths; no new identity/capability/count; 14F memory values unchanged.'
 catalog(union);save('existing_empirical_catalog.json',union)
 admission=read(ROOT/'results/module8b_quality_pool_finalization/quality_admission_matrix.json')['rows']
 quality={(r['model'],r['task']):r for r in admission}
 heavy=read(ROOT/'results/module14_heavy_structural/heavy_candidate_pool.json')
 for task,ps in heavy['tasks'].items():
  for p in ps:
   quality[p['planning_id'],task]={'planning_admissible':p['original_admission_pass'],'measured_quality':p['quality'],'quality_metric':p['quality_metric'],'source':p['measurement_source']}
 paths=[]
 for m,p in union['models'].items():
  for task in p['supported_tasks']:
   q=quality[m,task];assert q['planning_admissible']
   paths.append({'model':m,'role':p['role'],'task':task,'measured_s0_ms':p['service_time_ms'][task],'measured_quality':q['measured_quality'],'quality_metric':q['quality_metric'],'original_admission_pass':True,'quality_source':q['source'],'service_time_source':p['sources'],'empirical_counts':sorted(int(n) for n in union['capacity_factors'][m] if int(n)>0),'memory_mib':union['memory_table_mib'][m],'gain':union['capacity_factors'][m],'multi_task_g_transfer_surrogate_not_joint_certificate':p['role']=='generalist'})
 save('task_path_measurement_audit.json',{'at_utc':now(),'paths':paths,'no_new_measurements':True,'original_quality_SLO_source':ref(ROOT/'results/module8b_quality_pool_finalization/quality_policy.json')})
 budgets=read(ROOT/'results/module14_heavy_structural/screen_protocol_freeze.json')['budgets'];memory=[]
 # Pure memory eligibility only, not latency/QPS selection or a three-phase witness.
 for task in ['asr','lid','text','ocr','image']:
  for g,gp in union['models'].items():
   if gp['role']!='generalist' or task not in gp['supported_tasks']:continue
   for s,sp in union['models'].items():
    if sp['role']!='specialist' or task not in sp['supported_tasks']:continue
    for b in budgets:
     factor=union['safe_factor'];g1=factor*union['memory_table_mib'][g]['1'];gs=g1+factor*union['memory_table_mib'][s]['1'];extra=g1+factor*union['memory_table_mib'][s]['2'] if '2' in union['memory_table_mib'][s] else None
     memory.append({'task':task,'Generalist':g,'Specialist':s,'background_capabilities':sorted(set(gp['supported_tasks'])-{task}),'budget_id':b['id'],'budget_mib':b['safe_memory_budget_mib'],'G_only_safe_mib':g1,'G_S1_safe_mib':gs,'G_S2_safe_mib':extra,'G_S1_fits':gs<=b['safe_memory_budget_mib'],'measured_S2_excluded_by_budget':extra is not None and gs<=b['safe_memory_budget_mib']<extra,'S2_unknown_not_extrapolated':extra is None})
 with (OUT/'memory_eligibility_audit.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(memory[0]));writer.writeheader();writer.writerows(memory)
 missing={'at_utc':now(),'project_fixed_E2E_latency_SLO_found':False,'historical_quality_policy_found':True,'historical_rho_max':.8,'historical_J_surrogate':'mean congestion surrogate, not an E2E latency upper bound or P95 contract','scope_checked':['project source/configuration/docs','placement_scenario.json historical quality presets','Module8B quality_policy.json','Module14C screen_protocol_freeze.json','Module14F reconciled E/H catalogs','Module14E validation_freeze.json'],'required_input':None,'user_amended_capacity_only':True,'latency_SLO_not_required_or_claimed':True,'reported_before_CPU_QPS_search':True,'latency_contract_may_not_be_inferred_from_CPU_or_GPU_outcome':True,'CPU_QPS_search_started':False,'GPU_sessions':0}
 save('latency_SLO_audit.json',missing)
 save('input_audit.json',{'at_utc':now(),'module':'15','status':'READY_FOR_CAPACITY_CONSTRAINED_CPU_SEARCH','sources':[ref(F/'regime_e_reconciled.json'),ref(F/'heavy_empirical_x3_reconciled.json'),ref(ROOT/'results/module8b_quality_pool_finalization/quality_policy.json'),ref(ROOT/'results/module14_heavy_structural/screen_protocol_freeze.json')],'budgets':budgets,'safe_factor':union['safe_factor'],'original_V3_baseline_preserved':True,'original_V3_implementation':[ref(ROOT/'module12/solver.py'),ref(ROOT/'deployment_core/routing/e2e_surrogate.py')],'resource_priority_design':'New Module15 solver only: primary minimum measured safe memory; among exactly minimum-memory states use frozen J-routing. No replica-count objective ahead of J. Full existing model pool competes; original V3 retained separately, never edited; rho<=0.8, no latency SLO.','task_priority':['asr','lid','text','ocr','image'],'background_preference':'An independently required quality-admitted task served by the same G; prefer visual_qa where available. Its load stays fixed across all stages.','no_unmeasured_count_extrapolation':True,'no_model_quality_budget_history_changes':True,'CPU_QPS_searches':0,'GPU_sessions':0,'formal_gate':None,'user_amendment':'Capacity-Constrained Three-Phase Existence Test; no invented E2E SLO; primary measured safe memory, secondary frozen J-routing; E2E mean/P95 reported only','not_found_gate_not_claimed_for_missing_SLO':True,'next_step':'Freeze CPU selection rules, search full existing measured pool under rho<=0.8 and corrected memory; freeze complete theoretical witness before exactly LOW/MID/HIGH x3 GPU validations.'})
 (OUT/'module15_preparation.md').write_text('# Module 15 preparation\n\n状态：READY_FOR_CAPACITY_CONSTRAINED_CPU_SEARCH。项目没有预定义E2E latency SLO；已在搜索前报告。用户明确修改为容量约束测试，不设置人为延迟阈值。原quality/capability、rho≤0.8、14F实测safe memory均保留。E2E mean/P95只作为GPU指标。\n\n完整现有E/H identity并集参加部署竞争；原V3对照不修改。主目标是最小实测驻留safe memory，同等最小资源成本状态中使用冻结J-routing。已核验原质量admission和真实Q/s0/g(x)/M(x)，未选择三阶段QPS、未运行GPU。\n')
 print('MODULE15_PREPARED',len(protected),'protected',len(paths),'paths',len(memory),'memory combinations','READY_FOR_CAPACITY_CONSTRAINED_CPU_SEARCH')
if __name__=='__main__':main()
