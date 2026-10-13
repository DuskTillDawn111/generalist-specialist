"""Bounded counterbalanced frozen comparison campaign; stop each failed config."""
import os,time
from .common import *
from .metrics import summarize
from module11b.measure import transport,prime,phase
from module16.gpu import ProcessMonitor,memory_values
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from replica_scaling.telemetry import snapshot,release
def run(slot,plan):
 target=OUT/'raw_runs'/(slot['run_id']+'.json');assert not target.exists(),'No outcome retry or overwritten observation.'
 for k,v in plan['environment_overrides'].items():os.environ[k]=v
 baseline=snapshot();assert not baseline['processes'] and baseline['used_mib']<=128
 assert baseline['uuid']==plan['GPU']['uuid']
 mapping,config,registry=transport(slot,plan);runtime=DeploymentRuntime(config,registry);monitor=ProcessMonitor(runtime,plan['telemetry_interval_seconds'],plan['GPU']['total_mib']);begin=time.perf_counter_ns();monitor.start()
 result={'slot':slot,'status':'STARTING','errors':[],'phases':{},'baseline_GPU':baseline,'worker_identities':{},'replica_to_family':mapping,'transport_config':config.as_dict(),'frozen_validation_plan':ref(OUT/'frozen_benchmark_manifest.json')}
 try:
  runtime.start();result['worker_identities']={i:{'family':m,'pid':runtime.workers[i].process.pid,'READY':runtime.ready[i]} for i,m in mapping.items()}
  assert len({v['pid'] for v in result['worker_identities'].values()})==sum(slot['deployment'].values())
  result['path_probe_records']=prime(runtime,config,plan,mapping,slot)
  result['warmup']=phase(runtime,config,plan,mapping,'warmup',config.warmup_seconds,None,monitor)
  result['postwarm_idle']=monitor.idle_window('postwarm_idle',2.)
  result['phases']['measurement']=phase(runtime,config,plan,mapping,'measurement',slot['measurement_seconds'],None,monitor)
  result['postmeasurement_idle']=monitor.idle_window('postmeasurement_idle',2.);monitor.check();result['status']='COMPLETED'
 except (Exception,KeyboardInterrupt) as exc:result['status']='FAILED';result['errors'].append(repr(exc))
 finally:
  try:result['shutdown']=runtime.close()
  except Exception as exc:result['errors'].append('shutdown:'+repr(exc))
  monitor.close();result['errors']+=runtime.errors+monitor.errors;result['telemetry']={'samples':monitor.samples,'errors':monitor.errors};result['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process]);result['valid_session']=bool(result['status']=='COMPLETED' and result['gpu_release']['verified'] and not result['errors'] and all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in result.get('shutdown',{}).values()));result['elapsed_seconds']=(time.perf_counter_ns()-begin)/1e9;save('raw_runs/'+target.name,result)
 return result

def main():
 f=read(OUT/'frozen_benchmark_manifest.json');assert f['status']=='FROZEN_BEFORE_GPU';verify(f['implementation']+f['reused_implementation']+[f['input_protocol'],f['CPU']]);cases={c['case_id']:c for c in f['cases']};configs={c['config_id']:c for c in f['configurations']};completed=[];stopped={};global_abort=None;skipped=[]
 for slot in f['slots']:
  ident=slot['config_id'];cfg=configs[ident];case=cases[cfg['owner_case']]
  if ident in stopped or global_abort:skipped.append({'run_id':slot['run_id'],'reason':stopped.get(ident,global_abort)});continue
  path=OUT/'raw_runs'/(slot['run_id']+'.json')
  if path.exists():raw=read(path)
  else:
   print('START',slot['run_id'],slot['configuration_role'],flush=True);raw=run(slot,case['historical_plan'])
  row,_,workers,_,_=summarize(raw,cfg,case);completed.append(slot['run_id']);hard_worker=[w['worker'] for w in workers if w['worker_stability']=='unstable'];print('FINISH',slot['run_id'],'valid',row['valid_session'],'system',row['system_stability'],'workers',[(w['model'],w['worker_stability']) for w in workers],'memory',row['memory_passed'],'target_E2E',row['target_mean_E2E_ms'],row['target_P95_E2E_ms'],'completed/within',row['target_completed'],row['target_within_window_completed'],'errors',row['errors'],flush=True)
  if row['system_stability']=='unstable' or hard_worker or not row['valid_session'] or not row['memory_passed'] or not row['release_verified']:
   stopped[ident]={'run_id':slot['run_id'],'system_stability':row['system_stability'],'worker_unstable':hard_worker,'valid_session':row['valid_session'],'memory_passed':row['memory_passed'],'errors':row['errors'],'reason':'First unstable/sustainedbacklog/runtime/memory failure; no retry, laterrep skipped.'}
  if not row['release_verified']:global_abort={'run_id':slot['run_id'],'reason':'GPU release failure; abort remaining experiment.'}
  save('campaign_status.json',{'at_utc':now(),'completed_runs':completed,'stopped_configurations':stopped,'global_abort':global_abort,'skipped_runs':skipped,'frozen_config_changed':False,'no_outcome_retry':True})
 save('campaign_status.json',{'at_utc':now(),'completed_runs':completed,'stopped_configurations':stopped,'global_abort':global_abort,'skipped_runs':skipped,'frozen_config_changed':False,'no_outcome_retry':True});print('BENCHMARK_GPU_COMPLETE','runs',len(completed),'stopped',stopped,'skipped',len(skipped),flush=True)
if __name__=='__main__':main()
