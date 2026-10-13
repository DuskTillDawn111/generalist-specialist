"""Sequential frozen six-task validation; stop first unstable or infeasible run."""
import os,time,statistics
from .common import *
from module11b.measure import transport,prime,phase
from module11b.telemetry import CostMonitor
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from replica_scaling.telemetry import snapshot,release
class ProcessMonitor(CostMonitor):
 def __init__(self,*args):
  super().__init__(*args);self.nvml=None;self.util_error=None;self.last_util_timestamp=0
  try:
   import pynvml
   pynvml.nvmlInit();self.nvml=pynvml;self.handle=pynvml.nvmlDeviceGetHandleByIndex(0)
  except Exception as exc:self.util_error=repr(exc)
 def sample(self):
  v=super().sample();v['per_process_gpu_utilization']=[]
  if self.nvml and not self.util_error:
   try:
    samples=self.nvml.nvmlDeviceGetProcessUtilization(self.handle,self.last_util_timestamp)
    v['per_process_gpu_utilization']=[{'pid':a.pid,'timestamp_us':a.timeStamp,'sm_percent':a.smUtil,'memory_percent':a.memUtil} for a in samples if a.pid in v['owned_pids']]
    if samples:self.last_util_timestamp=max(a.timeStamp for a in samples)
   except Exception as exc:
    # NoData is expected before CUDA workers launch; unsupported is permanent.
    if 'NotFound' not in type(exc).__name__:self.util_error=repr(exc)
  v['per_process_gpu_utilization_status']='UNAVAILABLE: '+self.util_error if self.util_error else 'SUPPORTED'
  return v
def run(slot,plan):
 target=OUT/'raw_runs'/(slot['run_id']+'.json');assert not target.exists(),'No outcome retry or overwritten observation.'
 for k,v in plan['environment_overrides'].items():os.environ[k]=v
 baseline=snapshot();assert not baseline['processes'] and baseline['used_mib']<=128
 assert baseline['uuid']==plan['GPU']['uuid']
 mapping,config,registry=transport(slot,plan);runtime=DeploymentRuntime(config,registry);monitor=ProcessMonitor(runtime,plan['telemetry_interval_seconds'],plan['GPU']['total_mib']);begin=time.perf_counter_ns();monitor.start()
 result={'slot':slot,'status':'STARTING','errors':[],'phases':{},'baseline_GPU':baseline,'worker_identities':{},'replica_to_family':mapping,'transport_config':config.as_dict(),'frozen_validation_plan':ref(OUT/'frozen_gpu_manifest.json')}
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

def memory_values(raw,factor):
 a=statistics.mean(v['owned_used_mib'] for v in raw['postwarm_idle']) if raw.get('postwarm_idle') else None
 b=statistics.mean(v['owned_used_mib'] for v in raw['postmeasurement_idle']) if raw.get('postmeasurement_idle') else None
 peak=max((v['owned_used_mib'] for v in raw['telemetry']['samples']),default=None);B=raw['slot']['safe_memory_budget_mib']
 return {'postwarm_idle_mib':a,'postmeasurement_idle_mib':b,'postwarm_safe_mib':factor*a if a is not None else None,'postmeasurement_safe_mib':factor*b if b is not None else None,'sampled_owned_peak_mib':peak,'budget_mib':B,'passed':all(v is not None and factor*v<=B+1e-6 for v in [a,b]) and peak is not None and peak<=B+1e-6}
def main():
 f=read(OUT/'frozen_gpu_manifest.json');assert f['status']=='FROZEN_BEFORE_GPU';verify(f['implementation']+f['reused_implementation']+[f['CPU'],f['catalog']]);factor=read(f['catalog']['path'])['safe_factor'];completed=[];failure=None
 for slot in f['slots']:
  path=OUT/'raw_runs'/(slot['run_id']+'.json')
  if path.exists():raw=read(path)
  else:
   print('START',slot['run_id'],flush=True);raw=run(slot,f['plan'])
  completed.append(slot['run_id']);ph=raw['phases'].get('measurement',{});st=ph.get('stability',{}).get('stability_state_v2');mem=memory_values(raw,factor)
  print('FINISH',slot['run_id'],'valid',raw['valid_session'],'stability',st,'memory',mem,'errors',raw['errors'],flush=True)
  if st=='unstable' or not raw['valid_session'] or not mem['passed'] or not raw['gpu_release']['verified']:
   failure={'run_id':slot['run_id'],'stage':slot['stage'],'stability':st,'valid_session':raw['valid_session'],'memory':mem,'errors':raw['errors'],'suspected_mechanism':'See immutable request queues, stability reasons, errors and memory observations; no causal mechanism inferred from CPU alone.'};break
 save('campaign_status.json',{'at_utc':now(),'completed_runs':completed,'stopped_on_first_failure':failure,'skipped_runs':[s['run_id'] for s in f['slots'] if s['run_id'] not in completed],'frozen_config_changed':False,'no_retry':True});print('CAMPAIGN_STOPPED' if failure else 'NINE_RUNS_COMPLETED',flush=True)
if __name__=='__main__':main()
