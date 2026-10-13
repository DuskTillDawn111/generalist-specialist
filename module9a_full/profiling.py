"""Only standalone metadata profiling; native handlers untouched."""
import os, queue, subprocess, threading, time, sys
from .common import *
sys.path.insert(0,str(ROOT/'joint_deploy'))
from joint_deploy.measurement.deployment_runtime import ObservedWorker
from replica_scaling.telemetry import snapshot,release
class ProfileWorker(ObservedWorker):
 def __init__(self,m,spec,events,output):super().__init__(m,spec,events);self.output=output
 def start(self):
  env=dict(os.environ,PILOT_NATIVE_WORKER=self.config['script'])
  cmd=['nsys','profile','--trace=cuda,nvtx','--capture-range=cudaProfilerApi','--capture-range-end=stop','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--force-overwrite=false','-o',str(self.output),self.config['python'],str(ROOT/'module9a_full/profile_worker.py')]
  self.process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1);threading.Thread(target=self._reader,daemon=True).start()
def run(m,k):
 p=load(OUT/'profile_protocol.json');os.environ.update(p['environment']);base=snapshot();assert not base['processes'] and base['used_mib']<=128
 tag=m+'__'+k;d=OUT/'raw_profiles';d.mkdir(exist_ok=True);assert not (d/(tag+'.json')).exists()
 events=queue.Queue();w=ProfileWorker(m,p['registry'][m],events,d/tag);raw={'model':m,'task':k,'status':'STARTED','warmup_records':[],'target_records':[],'profile_timing_ground_truth':False};ready={}
 def receive(rid=None):
  deadline=time.monotonic()+300
  while time.monotonic()<deadline:
   try:n,kind,r=events.get(timeout=.1)
   except queue.Empty:continue
   if kind in ['ERROR','EXIT']:raise RuntimeError((kind,r))
   if kind=='READY' and rid is None:return r
   if kind=='RESULT' and r['request_id']==rid:return r
  raise TimeoutError(rid)
 def one(item,rid,i):
  w.send({'request_id':rid,'pool_id':item['pool_id'],'task_type':item['task_type'],'sample':item['sample'],'arrival_index':i,'submit_ns':time.time_ns()});return receive(rid)
 try:
  w.start();ready=receive();raw['ready']=ready
  for i in range(6):raw['warmup_records'].append(one(p['items'][k][i%3],f'warm:{i}',i))
  one({'pool_id':'control','task_type':'__profile_start__','sample':{}},'control:start',-1)
  for i,item in enumerate(p['items'][k][:3]):raw['target_records'].append(one(item,f'target:{i}',i))
  one({'pool_id':'control','task_type':'__pilot_profile_stop__','sample':{}},'control:stop',-1);raw['status']='COMPLETE'
 except BaseException as e:raw['status']='FAILED';raw['error']=repr(e)
 finally:
  w.shutdown();w.process.wait(timeout=120);raw['exit_code']=w.process.returncode;raw['gpu_release']=release(base,[ready.get('pid',w.process.pid)]);save('raw_profiles/'+tag+'.json',raw)
 assert raw['status']=='COMPLETE' and raw['exit_code']==0 and raw['gpu_release']['verified'],raw.get('error')
 subprocess.run(['nsys','export','--type=sqlite','--output',str(d/(tag+'.sqlite')),str(d/(tag+'.nsys-rep'))],check=True)
def main():
 p=load(OUT/'profile_protocol.json')
 for i,(m,k) in enumerate(p['paths']):
  previous=OUT/'raw_profiles'/(m+'__'+k+'.json')
  if previous.exists():
   assert load(previous)['status']=='COMPLETE' and (OUT/'raw_profiles'/(m+'__'+k+'.sqlite')).exists(),'failed measurement cannot be silently reused'
   continue
  print(json.dumps({'event':'PROFILE_START','index':i,'model':m,'task':k}),flush=True)
  run(m,k);save('campaign_state.json',{'phase':'PROFILING','completed_profiles':i+1,'total_profiles':len(p['paths']),'new_validation_runs':0})
  print(json.dumps({'event':'PROFILE_COMPLETE','index':i}),flush=True)
 save('campaign_state.json',{'phase':'PROFILES_COMPLETE','completed_profiles':len(p['paths']),'new_validation_runs':0})
if __name__=='__main__':main()
