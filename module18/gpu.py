"""Independent cross-task GPU campaign; immutable inputs and bounded stop rules."""
import os,time
from .common import *
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
 result={'slot':slot,'status':'STARTING','errors':[],'phases':{},'baseline_GPU':baseline,'worker_identities':{},'replica_to_family':mapping,'transport_config':config.as_dict(),'frozen_validation_plan':ref(OUT/'frozen_task_manifests.json')}
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
 from .finalize import summarize
 f=read(OUT/'frozen_task_manifests.json');assert f['status']=='FROZEN_BEFORE_GPU';verify(f['implementation']+f['reused_implementation']+[f['CPU'],f['CPU_protocol'],f['catalog'],f['controlled_pool'],f['quality_audit'],f['sample_pool']]);completed=[];task_status={};global_abort=None
 for task in f['GPU_protocol']['task_order']:
  ts=[s for s in f['slots'] if s['target_task']==task];failure=None
  if not ts:task_status[task]={'CPU_status':'NOT_SUPPORTED','completed':[],'failure':None};continue
  if global_abort:task_status[task]={'completed':[],'failure':global_abort};continue
  for stage in f['GPU_protocol']['phase_order']:
   phase_rows=[]
   for slot in [s for s in ts if s['stage']==stage]:
    path=OUT/'raw_runs'/(slot['run_id']+'.json')
    if path.exists():raw=read(path)
    else:
     print('START',slot['run_id'],flush=True);raw=run(slot,f['plan'])
    row,*_=summarize(raw,f);phase_rows.append(row);completed.append(slot['run_id']);print('FINISH',slot['run_id'],'valid',row['valid_session'],'stability',row['stability'],'passed',row['passed'],'target G/S',row['target_G_completed'],row['target_S_completed'],'memory',row['memory_passed'],'errors',row['errors'],flush=True)
    if row['stability']=='unstable' or not row['valid_session'] or not row['memory_passed'] or not row['release_verified'] or not row['stage_condition'] or not row['six_tasks_completed']:
     failure={'stage':stage,'run_id':slot['run_id'],'reason':'unstable_or_runtime_memory_phase_failure','observation':row}
     if not row['release_verified']:global_abort={'reason':'GPU_RELEASE_FAILURE','run_id':slot['run_id']}
     break
   if not failure and any(not r['passed'] for r in phase_rows):failure={'stage':stage,'run_id':next(r['run_id'] for r in phase_rows if not r['passed']),'reason':'completed_phase_not_3_of_3_steady','observations':phase_rows}
   if failure:break
  task_status[task]={'completed':[r for r in completed if r.startswith(task+'_')],'failure':failure,'frozen_config_changed':False}
  save('campaign_status.json',{'at_utc':now(),'completed_runs':completed,'task_status':task_status,'global_abort':global_abort,'skipped_runs':[s['run_id'] for s in f['slots'] if s['run_id'] not in completed],'no_retry':True,'frozen_config_changed':False})
 save('campaign_status.json',{'at_utc':now(),'completed_runs':completed,'task_status':task_status,'global_abort':global_abort,'skipped_runs':[s['run_id'] for s in f['slots'] if s['run_id'] not in completed],'no_retry':True,'frozen_config_changed':False});print('CAMPAIGN_COMPLETE',[(t,len(s['completed']),s['failure']['reason'] if s['failure'] else None) for t,s in task_status.items()],flush=True)
if __name__=='__main__':main()
