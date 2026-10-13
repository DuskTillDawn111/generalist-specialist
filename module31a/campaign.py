import time,threading,copy,fcntl,traceback,subprocess,sys
import numpy as np
from .common import *
from module30.telemetry_v2 import SafeMonitorV2
from module30 import gpu as native
from module26.common import arithmetic,cohort
from module25.common import utility_inputs

class TimedMonitor(SafeMonitorV2):
 instances=[]
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.check_seconds=[];self.publish_seconds=[];self.instances.append(self)
 def check(self):
  start=time.perf_counter()
  try:return super().check()
  finally:self.check_seconds.append(time.perf_counter()-start)
 def set_active(self,value):
  start=time.perf_counter()
  try:return super().set_active(value)
  finally:self.publish_seconds.append(time.perf_counter()-start)

def verify(f):
 for r in f['sources']+f['frozen_inputs']:
  assert ref(r['path'])['sha256']==r['sha256'],r['path']
 for p,v in f['checkpoint_metadata'].items():
  assert [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==v,p
 for s in f['slots']:
  for r in s['trajectories'].values():assert ref(r['path'])['sha256']==r['sha256']

def slow_io_check():
 from types import SimpleNamespace
 results=[]
 for i in range(12):
  started=threading.Event()
  def sensor():
   started.set();time.sleep(.25);return dict(time_ns=time.perf_counter_ns(),used_mib=0,owned_used_mib=0,foreign_pids=[])
  m=TimedMonitor(SimpleNamespace(lock=threading.RLock(),workers={},ready={}),{'run_id':'CPU_SLOW_'+str(i)},1,sensor=sensor)
  t=threading.Thread(target=m.sample);t.start();assert started.wait(2)
  m.check();m.set_active(None);t.join(3);assert not t.is_alive()
  assert max(m.check_seconds+m.publish_seconds)<.01
  results.append(dict(check_seconds=m.check_seconds,publish_seconds=m.publish_seconds))
 save('nonblocking_control_acceptance.json',dict(passed=True,sensor_block_seconds=.25,repetitions=results))

def execute(slot,f):
 from .harness import run
 before=len(TimedMonitor.instances)
 raw=run(slot,f)
 m=TimedMonitor.instances[-1];assert len(TimedMonitor.instances)==before+1
 raw['telemetry_version']='SafeMonitorV2 with timing-only wrapper'
 raw['experiment_manifest']=ref(OUT/'experiment_manifest.json')
 raw['control_latency_seconds']={'check':m.check_seconds,'phase_publish':m.publish_seconds,'max_check':max(m.check_seconds,default=0),'max_publish':max(m.publish_seconds,default=0),'threshold_seconds':.01}
 # Append-only provenance/integrity fields; original phase records are never rewritten.
 raw['control_nonblocking']=max(m.check_seconds+m.publish_seconds,default=0)<.01
 save('raw_runs/'+slot['run_id']+'.json',raw)
 return raw

def smoke_accept(raw):
 _,report=configure();a=report.journal_audit(raw)
 ph=raw.get('phases',{}).get('measurement');checks=dict(session=raw['valid_session'],journal=a['passed'],nonblocking=raw['control_nonblocking'],measurement=bool(ph))
 if ph:
  rs=ph['request_records'];trace=read(raw['slot']['trajectories']['measurement']['path'])['events']
  checks['all_completed']=len(rs)==len(trace) and all(r['status']=='completed' for r in rs)
  checks['trace_matched']=all(r['task']==e['task'] and r['sample_id']==e['item']['sample'].get('id') and r['frozen_offset_seconds']==e['offset_seconds'] for r,e in zip(rs,trace))
  checks['six_tasks']=len(set(r['task'] for r in rs))==6
  checks['native_timestamps']=all(all(r.get('worker_response',{}).get(k) is not None for k in ['submit_ns','start_ns','end_ns']) and abs(r['worker_latency_ms']-(r['worker_response']['end_ns']-r['worker_response']['start_ns'])/1e6)<1e-6 and abs(r['worker_queue_wait_ms']-(r['worker_response']['start_ns']-r['worker_response']['submit_ns'])/1e6)<1e-6 for r in rs)
  d,refs=utility_inputs();v=arithmetic(raw,read(OUT/'selected_pool_catalog.json'),refs,d['weights'])
  from module20.optimizer import components
  norm=components(v['per_task'],raw['slot']['lambda_rps'],refs,v['T_norm'],v['M_norm'],d['weights'])
  checks['utility_recomputed']=abs(norm['U']-v['raw_arithmetic_U'])<1e-12
  checks['valid_sensors']=all(all(s.get(k) is not None and np.isfinite(s[k]) for k in ['time_ns','unix_ns','owned_used_mib','gpu_util_percent','worker_CPU_core_percent','host_RSS_bytes','temperature_gpu_c','sm_clock_mhz']) for s in raw['telemetry']['samples'])
  return dict(passed=all(checks.values()),checks=checks,U=v['raw_arithmetic_U'],raw_arithmetic=v,journal=a,max_check_seconds=raw['control_latency_seconds']['max_check'])
 return dict(passed=False,checks=checks,journal=a,errors=raw['errors'])

def main():
 lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 f=read(OUT/'experiment_manifest.json');configure();native.SafeMonitor=TimedMonitor
 slow_io_check();smokes=[];formal=[];error=None
 save('campaign_status.json',dict(status='SMOKE',pid=__import__('os').getpid()))
 try:
  for slot in f['slots']:
   verify(f);s=copy.deepcopy(slot);s['run_id']='SMOKE_'+slot['run_id'];s['measurement_seconds']=30.
   s['trajectories']['measurement']=ref(OUT/'trajectories'/('smoke_'+slot['run_id']+'.json'))
   print('SMOKE_START',s['run_id'],flush=True);r=execute(s,f);v=smoke_accept(r);smokes.append(v);save('smoke_acceptance_partial.json',smokes)
   print('SMOKE_FINISH',s['run_id'],v['passed'],v['checks'],flush=True)
   assert v['passed'],('SMOKE_INVALID',s['run_id'],v)
  assert all(cohort(read(OUT/'raw_runs'/('SMOKE_'+s['run_id']+'.json')))==cohort(read(OUT/'raw_runs/SMOKE_A0.json')) for s in f['slots'])
  save('smoke_acceptance.json',dict(passed=True,arms=smokes,delta_U=[v['U']-smokes[0]['U'] for v in smokes],formal_measurement_included=False))
  for slot in f['slots']:
   verify(f);save('campaign_status.json',dict(status='FORMAL',arm=slot['run_id'],pid=__import__('os').getpid()))
   print('FORMAL_START',slot['run_id'],flush=True);r=execute(slot,f)
   from .report import assess,generate
   o=assess(r);formal.append(o);save('formal_acceptance_partial.json',formal);generate()
   print('FORMAL_FINISH',slot['run_id'],o['eligible'],o['U'],r['errors'],flush=True)
   if not o['technical_valid']:raise RuntimeError(('INVALID_WINDOW_DIAGNOSE_BEFORE_RERUN',slot['run_id'],o['validation']))
 except BaseException as exc:
  error=dict(error=repr(exc),traceback=traceback.format_exc());save('campaign_failure.json',error);print('CAMPAIGN_BLOCKED',repr(exc),flush=True)
 finally:
  from .report import generate
  save('campaign_status.json',dict(status='BLOCKED' if error else 'COMPLETED',formal_arms_attempted=[o['run_id'] for o in formal],failure=error))
  generate();print('MODULE31A_FINISHED',read(OUT/'gate_decision.json')['gate'],flush=True)
 if error:raise SystemExit(1)
if __name__=='__main__':main()
