"""Build empirical count2 points from exactly three valid paired repetitions."""
import copy
from collections import defaultdict
from .common import *
from module11b.common import factor,stat
from module11b.analyze_replica import full_capacity

def rows():
 out=[]
 for slot in read(OUT/'replica_measurement_manifest.json')['slots']:
  p=OUT/'raw_runs'/(slot['run_id']+'.json')
  if not p.exists():continue
  raw=read(p);phase=raw.get('phases',{}).get('closed_loop_capacity',{});low=raw.get('phases',{}).get('low_load',{});idle=raw.get('postwarm_idle',[]);samples=raw['telemetry']['samples']
  completed=[r for r in phase.get('request_records',[]) if r['status']=='completed']
  idle_mean=statistics.mean(s['owned_used_mib'] for s in idle) if idle else None
  out.append({'run_id':slot['run_id'],'planning_id':slot['model'],'task_path':slot['task_path'],'replica_count':slot['deployment'][slot['model']],'worker_count':len(raw['worker_identities']),'repetition':slot['repetition'],'seed':slot['seed'],'evidence_role':'NEW_X2_STATE' if slot['deployment'][slot['model']]==2 else 'PAIRED_X1_CAPACITY_REFERENCE_DOES_NOT_REPLACE_G1_M1','throughput_rps':phase.get('throughput_rps'),'service_time_ms':stat(r.get('worker_latency_ms') for r in completed),'idle_GPU_memory_mib':idle_mean,'safe_memory_mib':factor()*idle_mean if idle_mean else None,'peak_GPU_memory_mib':max((s['owned_used_mib'] for s in samples),default=None),'peak_device_memory_mib':max((s['used_mib'] for s in samples),default=None),'stability':low.get('stability',{}).get('stability_state_v2'),'errors':raw['errors'],'status':raw['status'],'full_capacity':full_capacity(raw),'valid_session':raw['valid_session'],'max_outstanding_per_replica':phase.get('max_outstanding_per_replica'),'worker_PIDs':{i:v['pid'] for i,v in raw['worker_identities'].items()},'gpu_release_verified':raw['gpu_release']['verified'],'raw':ref(p)})
 return out

def main():
 manifest=read(OUT/'replica_measurement_manifest.json');r=rows();assert len(r)==len(manifest['slots'])
 csvsave('replica_measurements.csv',r)
 h=copy.deepcopy(read(OLD/'regime_h.json'));path_records=[];failures=[]
 for path in manifest['requested_paths']:
  m=path['planning_id'];task=path['task_path'];new=[s for s in r if s['planning_id']==m and s['task_path']==task and s['replica_count']==2];prev=[s for s in r if s['planning_id']==m and s['task_path']==task and s['replica_count']==1]
  assert len(new)==len(prev)==3
  valid=all(s['full_capacity'] and s['worker_count']==s['replica_count'] and s['gpu_release_verified'] for s in new+prev)
  paired=[next(s['throughput_rps'] for s in new if s['repetition']==rep)/next(s['throughput_rps'] for s in prev if s['repetition']==rep) for rep in [1,2,3]] if valid else None
  record={**path,'paired_throughput_gains':paired,'valid_all_three':valid,'g2':statistics.median(paired) if valid else None,'M2_idle_mib':statistics.mean(s['idle_GPU_memory_mib'] for s in new) if valid else None,'M2_safe_mib':factor()*statistics.mean(s['idle_GPU_memory_mib'] for s in new) if valid else None,'x2_throughput_rps':[s['throughput_rps'] for s in new],'x1_reference_throughput_rps':[s['throughput_rps'] for s in prev],'x2_stability':[s['stability'] for s in new],'errors':[e for s in new+prev for e in s['errors']],'raw_repetitions':[s['raw'] for s in prev+new],'g1_M1_policy':'Reuse frozen values, never overwrite with auxiliary x1 references','no_extrapolation':True}
  pool=read(OLD/'heavy_candidate_pool.json')
  source=next(p for p in pool['tasks'][task] if p['planning_id']==m)
  record['task_path_M1_idle_mib']=source['idle_memory_mib'];record['task_path_M1_safe_mib']=source['safe_memory_mib'];record['M1_source']=source['measurement_source']
  record['state_status']='MEASURED' if valid else 'REAL_EXECUTION_FAILED_NOT_IMPUTED'
  if not valid:failures.append(record)
  path_records.append(record)
 for m in sorted({p['planning_id'] for p in path_records}):
  paths=[p for p in path_records if p['planning_id']==m]
  if not all(p['valid_all_three'] for p in paths):continue
  g=min(p['g2'] for p in paths);mem=max(p['M2_idle_mib'] for p in paths)
  h['capacity_factors'][m]['2']=g;h['memory_table_mib'][m]['2']=mem;h['evidence_bounds'][m]=2
  h['model_evidence'][m].update(measured_replica_counts=[1,2],unknown_gt1=False,unknown_ge3=True,Module14D_paths=paths,capacity_scope='Each path measured independently; shared family uses minimum empirical normalized gain. Other mixed paths remain a conservative family transfer surrogate.',memory_scope='Maximum of measured path postwarm idle means; shared residency counted once per independent checkpoint worker')
 h.update(at_utc=now(),module='14D',task_path_capacity_records=path_records,failed_replica_states=failures,measurement_manifest=ref(OUT/'replica_measurement_manifest.json'),empirical_counts_only=True,original_single_replica_values_preserved=True,family_transfer_limitation=manifest['family_capacity_rule'])
 before=read(OLD/'regime_h.json')
 for m in before['models']:
  for n,g in before['capacity_factors'][m].items():assert h['capacity_factors'][m][n]==g
  for n,mem in before['memory_table_mib'][m].items():assert h['memory_table_mib'][m][n]==mem
 assert h['models']==before['models']
 from module14c.common import catalog
 catalog(h)
 save('heavy_empirical_replica_table.json',h)
 manifest.update(status='COMPLETE',completed_at_utc=now(),successful_path_count=sum(p['valid_all_three'] for p in path_records),failed_replica_states=failures,measurements=ref(OUT/'replica_measurements.csv'),historical_g1_M1_preserved=True)
 # Table points reference original frozen manifest; store amended completion separately.
 save('replica_measurement_completion.json',manifest)
 print('ANALYZED',[(p['task_path'],p['model_id'],p['g2'],p['M2_idle_mib']) for p in path_records],flush=True)
if __name__=='__main__':main()
