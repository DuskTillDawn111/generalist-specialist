"""Two predeclared controls, four sessions max; no model/inference patch."""
import os,time,statistics
from pathlib import Path
import psutil
from .common import *
from module11b.measure import transport,prime,phase
from module16.gpu import ProcessMonitor,memory_values
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from replica_scaling.telemetry import snapshot,release
class DiagnosticMonitor(ProcessMonitor):
 def sample(self):
  v=super().sample();rows=[]
  for pid in v['owned_pids']:
   try:
    p=psutil.Process(pid);ctx=p.num_ctx_switches();tasks=[]
    for entry in Path('/proc',str(pid),'task').iterdir():
     try:
      a=list(map(int,(entry/'schedstat').read_text().split()));tasks.append({'tid':int(entry.name),'cpu_run_ns':a[0],'cpu_runqueue_wait_ns':a[1],'timeslices':a[2]})
     except (FileNotFoundError,PermissionError,ValueError):pass
    rows.append({'pid':pid,'CPU_affinity':p.cpu_affinity(),'context_switches_voluntary':ctx.voluntary,'context_switches_involuntary':ctx.involuntary,'thread_schedstat':tasks})
   except (psutil.Error,FileNotFoundError,PermissionError):pass
  v['scheduler_observation']=rows;return v
def run(slot,plan):
 target=OUT/'raw_runs'/(slot['run_id']+'.json');assert not target.exists(),'No outcome retry or overwritten observation.'
 for k,v in plan['environment_overrides'].items():os.environ[k]=v
 baseline=snapshot();assert not baseline['processes'] and baseline['used_mib']<=128
 assert baseline['uuid']==plan['GPU']['uuid']
 mapping,config,registry=transport(slot,plan);runtime=DeploymentRuntime(config,registry);monitor=DiagnosticMonitor(runtime,plan['telemetry_interval_seconds'],plan['GPU']['total_mib']);begin=time.perf_counter_ns();monitor.start()
 result={'slot':slot,'status':'STARTING','errors':[],'phases':{},'baseline_GPU':baseline,'worker_identities':{},'replica_to_family':mapping,'transport_config':config.as_dict(),'frozen_validation_plan':ref(OUT/'diagnostic_manifest.json')}
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
 f=read(OUT/'diagnostic_manifest.json');verify(f['implementation']+f['reused_implementation']+[f['catalog'],f['stage_A'],f['training_proposal'],f['plan']['sample_pool']]);completed=[];failure=None
 save('runtime_host_environment.json',{'at_utc':now(),'CPU_affinity':os.sched_getaffinity(0) and sorted(os.sched_getaffinity(0)),'cgroup_CPU_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'logical_CPU_count':os.cpu_count(),'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'MKL_NUM_THREADS':os.environ.get('MKL_NUM_THREADS'),'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS')})
 for slot in f['slots']:
  p=OUT/'raw_diagnostics'/(slot['run_id']+'.json')
  # run uses raw_runs to match the unchanged transport's storage convention.
  p=OUT/'raw_runs'/(slot['run_id']+'.json')
  if p.exists():raw=read(p)
  else:
   print('START',slot['run_id'],flush=True);raw=run(slot,f['plan'])
  completed.append(slot['run_id']);ph=raw['phases'].get('measurement',{});st=ph.get('stability',{}).get('stability_state_v2');mem=memory_values(raw,read(f['catalog']['path'])['safe_factor']);print('FINISH',slot['run_id'],raw['valid_session'],st,mem,raw['errors'],flush=True)
  if not raw['valid_session'] or not mem['passed'] or not raw['gpu_release']['verified']:
   failure={'run_id':slot['run_id'],'errors':raw['errors'],'memory':mem,'release':raw['gpu_release']['verified']};break
 save('diagnostic_campaign_status.json',{'at_utc':now(),'completed_runs':completed,'skipped_runs':[s['run_id'] for s in f['slots'] if s['run_id'] not in completed],'failure':failure,'no_outcome_retry':True,'diagnosis_not_three_phase_validation':True});print('DIAGNOSTICS_STOPPED',flush=True)
if __name__=='__main__':main()
