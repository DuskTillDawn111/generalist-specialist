"""Measured g3 and common E/H postwarm memory; no extrapolation."""
import copy
from collections import defaultdict
from .common import *
from module11b.analyze_replica import full_capacity
from module11b.common import stat

def new_rows():
 rows=[]
 for slot in read(OUT/'profiling_freeze.json')['slots']:
  p=OUT/'raw_runs'/(slot['run_id']+'.json')
  if not p.exists():continue
  r=read(p);phase=r.get('phases',{}).get('closed_loop_capacity',{});idle=r.get('postwarm_idle',[]);low=r.get('phases',{}).get('low_load',{});factor=read(D/'heavy_empirical_replica_table.json')['safe_factor'];valid=full_capacity(r);n=slot['deployment'][slot['model']]
  rows.append({'run_id':slot['run_id'],'family':slot['model'],'task_path':'ocr','count':n,'worker_count':len(r['worker_identities']),'repetition':slot['repetition'],'seed':slot['seed'],'throughput_rps':phase.get('throughput_rps'),'service_time_ms':stat(v.get('worker_latency_ms') for v in phase.get('request_records',[]) if v['status']=='completed'),'postwarm_idle_mib':statistics.mean(s['owned_used_mib'] for s in idle) if idle else None,'safe_idle_mib':factor*statistics.mean(s['owned_used_mib'] for s in idle) if idle else None,'peak_GPU_owned_mib':max((s['owned_used_mib'] for s in r['telemetry']['samples']),default=None),'peak_GPU_device_mib':max((s['used_mib'] for s in r['telemetry']['samples']),default=None),'stability':low.get('stability',{}).get('stability_state_v2'),'valid_session':r['valid_session'],'full_capacity':valid,'errors':r['errors'],'worker_PIDs':{k:v['pid'] for k,v in r['worker_identities'].items()},'gpu_release_verified':r['gpu_release']['verified'],'raw':ref(p)})
 return rows

def main():
 freeze=read(OUT/'profiling_freeze.json');rows=new_rows();assert len(rows)==24;csvsave('ocr_x3_measurements.csv',rows);save('ocr_x3_measurement_rows.json',rows)
 h=copy.deepcopy(read(D/'heavy_empirical_replica_table.json'));e=copy.deepcopy(read(C/'regime_e.json'));points=[];newobs=[]
 for m in freeze['requested_checkpoints']:
  a=sorted([r for r in rows if r['family']==m and r['count']==2],key=lambda r:r['repetition']);b=sorted([r for r in rows if r['family']==m and r['count']==3],key=lambda r:r['repetition']);assert len(a)==len(b)==3
  valid=all(r['full_capacity'] and r['worker_count']==r['count'] and r['gpu_release_verified'] for r in a+b);ratios=[y['throughput_rps']/x['throughput_rps'] for x,y in zip(a,b)] if valid else None
  g2=h['capacity_factors'][m]['2'];g3=g2*statistics.median(ratios) if valid else None;mem=statistics.mean(r['postwarm_idle_mib'] for r in b) if valid else None
  point={'family':m,'task_path':'ocr','previous_count':2,'new_count':3,'g2_frozen':g2,'paired_throughput_ratios':ratios,'g3':g3,'M3_postwarm_idle_mib':mem,'M3_safe_mib':h['safe_factor']*mem if mem is not None else None,'x3_stability':[r['stability'] for r in b],'valid_all_three':valid,'raw_repetitions':[r['raw'] for r in a+b],'no_extrapolation':True,'state_status':'MEASURED' if valid else 'REAL_STATE_EXECUTION_FAILURE_NO_IMPUTATION'}
  points.append(point)
  if valid:
   h['capacity_factors'][m]['3']=g3;h['memory_table_mib'][m]['3']=mem;h['evidence_bounds'][m]=3;h['model_evidence'][m].update(measured_replica_counts=[1,2,3],Module14F_x3=point,unknown_ge4=True)
  for r in a+b:
   if r['valid_session'] and r['postwarm_idle_mib'] is not None:newobs.append({**{k:r[k] for k in ['family','count','repetition','seed','postwarm_idle_mib','raw']},'group':'MODULE14F_OCR_X3_PAIRED_REFERENCES','observation_scope':'unchanged closed-loop postwarm-idle window'})
 observations=read(OUT/'existing_postwarm_memory_observations.json')+newobs;groups=defaultdict(list)
 for r in observations:groups[r['family'],r['count'],r['group']].append(r)
 complete=[]
 for (m,n,label),rs in groups.items():
  if len(rs)!=3 or {r['repetition'] for r in rs}!={1,2,3} or len({r['seed'] for r in rs})!=3:continue
  complete.append({'family':m,'count':n,'group':label,'group_mean_postwarm_idle_mib':statistics.mean(r['postwarm_idle_mib'] for r in rs),'independent_repetition_values':[r['postwarm_idle_mib'] for r in sorted(rs,key=lambda r:r['repetition'])],'raw_repetitions':[r['raw'] for r in rs]})
 corrections=[]
 for regime,data in [(E,e),(H,h)]:
  before=read(C/'regime_e.json') if regime==E else read(D/'heavy_empirical_replica_table.json')
  for m in data['models']:
   for key in list(data['memory_table_mib'][m]):
    n=int(key)
    if not n:continue
    eligible=[g for g in complete if g['family']==m and g['count']==n];old=data['memory_table_mib'][m][key];new=max([old]+[g['group_mean_postwarm_idle_mib'] for g in eligible]);data['memory_table_mib'][m][key]=new
    corrections.append({'regime':regime,'family':m,'count':n,'old_M_x_mib':before['memory_table_mib'][m].get(key),'new_M_x_mib':new,'changed':new!=before['memory_table_mib'][m].get(key),'candidate_group_means':eligible,'memory_semantics':'maximum of observed compatible THREE-repetition postwarm-idle group means, retaining old measured floor; no peak/linear next-count imputation'})
   data['models'][m]['idle_memory_mib']=data['memory_table_mib'][m]['1']
   for k in ['supported_tasks','service_time_ms','role','sources']:assert data['models'][m][k]==before['models'][m][k]
   for k,v in before['capacity_factors'][m].items():assert data['capacity_factors'][m][k]==v
  data.update(module='14F',at_utc=now(),memory_semantics=ref(OUT/'memory_semantics_freeze.json'),original_model_identities_and_latency_preserved=True)
  from module14c.common import catalog
  catalog(data)
  save('regime_e_reconciled.json' if regime==E else 'heavy_empirical_x3_reconciled.json',data)
 # Shared states use identical memory under the same observation rule.
 for m in set(e['models'])&set(h['models']):
  for n in set(e['memory_table_mib'][m])&set(h['memory_table_mib'][m]):assert e['memory_table_mib'][m][n]==h['memory_table_mib'][m][n],(m,n)
 save('ocr_x3_empirical_points.json',{'at_utc':now(),'points':points,'all_requested_states_measured':all(p['valid_all_three'] for p in points),'x_ge4_runs':0});save('memory_semantics_corrections.json',{'at_utc':now(),'policy':read(OUT/'memory_semantics_freeze.json'),'complete_independent_observation_groups':complete,'corrections':corrections,'E_H_shared_memory_consistency_passed':True});csvsave('memory_semantics_corrections.csv',corrections)
 print('X3_ANALYZED',[(p['family'],p['g3'],p['M3_postwarm_idle_mib'],p['x3_stability']) for p in points],flush=True)
 print('MEMORY_CORRECTIONS',[(r['regime'],r['family'],r['count'],r['old_M_x_mib'],r['new_M_x_mib']) for r in corrections if r['changed']],flush=True)
if __name__=='__main__':main()
