"""CPU-only, retained outliers; no acceptance retry, no GPU sensors."""
import sys, threading, queue, traceback, resource, argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32diag.isolation import install, VIOLATIONS
install()
from m32diag.common import *
from m32diag.monitor import IncrementalMonitor
from router_optimized import DeploymentRuntime
from vqa_aware_dynamic_router import VQAAwareRouter
from types import SimpleNamespace
P=read(OUT/'frozen_routing_parameters.json'); META=read(OUT/'audio_metadata_cache.json'); DEFAULT=max(x['S_p75_ms'] for x in P['cost_bins'])
def quant(a):
 a=list(a)
 return dict(n=len(a),mean_ms=float(np.mean(a)) if a else None,**{f'p{p}_ms':float(np.percentile(a,p)) if a else None for p in (50,90,95,99)},max_ms=max(a) if a else None)
def legacy(rt):
 return [dict(role='G' if x['model']==G else 'S',task=x['task'],submit_perf_ns=x.get('actual_submit_time_ns'),enqueue_perf_ns=x['enqueued_time_ns'],S_service_estimate_ms=x.get('router_S_service_estimate_ms',DEFAULT)) for x in rt.pending.values() if x['model'] in (G,S) and x['status'] in ('queued','submitted')]
class ObservedLock:
 def __init__(self,lock):self.lock=lock;self.owner=None
 def __enter__(self):self.lock.acquire();self.owner=threading.get_native_id();return self
 def __exit__(self,*args):self.owner=None;self.lock.release()
class SegmentedMonitor(IncrementalMonitor):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.state_lock=ObservedLock(self.state_lock);self.detailed=[];self.spans=[];self.context={}
 def check(self):
  # Frozen predicates and locking unchanged; full wall timer includes instrumentation.
  usage0=resource.getrusage(resource.RUSAGE_THREAD);cpu0=time.thread_time_ns();tick=time.perf_counter();begin=time.perf_counter_ns()
  stamps=[(begin,time.thread_time_ns())];owner=self.state_lock.owner;error=None
  try:
   with self.state_lock:
    stamps.append((time.perf_counter_ns(),time.thread_time_ns()));errors=tuple(self.errors);stamps.append((time.perf_counter_ns(),time.thread_time_ns()))
   stamps.append((time.perf_counter_ns(),time.thread_time_ns()))
   if errors or self.error_event.is_set():raise RuntimeError('TELEMETRY_ERROR: '+repr(errors))
   if self.resource_event.is_set() or self.resource_failure:raise RuntimeError('RUNTIME_RESOURCE_INFEASIBLE: actual GPU residency exceeds frozen physical budget')
   if self.foreign_event.is_set():raise RuntimeError('external_gpu_process_interference')
  except BaseException as exc:error=repr(exc);raise
  finally:
   elapsed=time.perf_counter()-tick;end=time.perf_counter_ns();cpu=time.thread_time_ns()-cpu0;stamps.append((end,time.thread_time_ns()));usage1=resource.getrusage(resource.RUSAGE_THREAD)
   row=dict(call_id=len(self.detailed)+1,**self.context,phase=self.phase,perf_ns=begin,end_perf_ns=end,wall_ms=elapsed*1000,thread_cpu_ms=cpu/1e6,off_cpu_estimate_ms=max(0,elapsed*1000-cpu/1e6),state_owner_before_acquire=owner,thread=threading.current_thread().name,native_tid=threading.get_native_id(),voluntary_switches=usage1.ru_nvcsw-usage0.ru_nvcsw,involuntary_switches=usage1.ru_nivcsw-usage0.ru_nivcsw,error=error,threshold_violation=elapsed>=.01,synchronous_telemetry_write_ms=0)
   for name,a,b in zip(['acquire','state_read','release','predicates'],stamps,stamps[1:]):
    row[name+'_wall_ms']=(b[0]-a[0])/1e6;row[name+'_cpu_ms']=(b[1]-a[1])/1e6
   self.check_seconds.append(elapsed);self.control_probes.append(row);self.detailed.append(row)
   if self.enforce_control and elapsed>=.01:raise RuntimeError('CONTROL_CHECK_OVER_10MS: '+repr(elapsed))
 def coherent_records(self):
  tick=time.perf_counter_ns();cpu=time.thread_time_ns()
  try:return super().coherent_records()
  finally:self.spans.append(dict(kind='coherent_records_lock_and_deepcopy',begin_ns=tick,end_ns=time.perf_counter_ns(),cpu_ms=(time.thread_time_ns()-cpu)/1e6,native_tid=threading.get_native_id()))
 def journal_snapshot(self,*a,**kw):
  tick=time.perf_counter_ns();cpu=time.thread_time_ns()
  try:return super().journal_snapshot(*a,**kw)
  finally:self.spans.append(dict(kind='journal_snapshot_including_json_write_fsync',begin_ns=tick,end_ns=time.perf_counter_ns(),cpu_ms=(time.thread_time_ns()-cpu)/1e6,native_tid=threading.get_native_id()))
def cgroup():
 p=Path('/sys/fs/cgroup/cpu.stat')
 return dict(zip((s.split()[0] for s in p.read_text().splitlines()),(int(s.split()[1]) for s in p.read_text().splitlines()))) if p.exists() else {}
def run(case):
 raw=read(ROOT/'results/module32c_dynamic_replication/raw_runs/P1_D.json');ph=raw['phases']['measurement'];start=ph['window']['start_time_ns'];rr=[x for x in ph['request_records'] if x['enqueued_time_ns']<start+case['logical_seconds']*10**9]
 events=sorted([(x[k],i,x) for x in rr for i,k in enumerate(['enqueued_time_ns','actual_submit_time_ns','completion_time_ns'])],key=lambda e:(e[0],e[1]))
 trace=read(ph['trajectory']['path'])['events'];inputs={x['request_id']:e['item']['sample']['input'] for x,e in zip(ph['request_records'],trace) if x['task']=='asr'}
 rt=DeploymentRuntime(SimpleNamespace());router=VQAAwareRouter(P,META,2400);records=[{k:v for k,v in x.items() if k in ['request_id','model','task','pool_id','sample_id','router_S_service_estimate_ms']} for x in rr]
 for r in records:r['status']='planned'
 by={r['request_id']:r for r in records};sent={k:threading.Event() for k in by};sq=queue.Queue();cq=queue.Queue();errors=[];timings=[];route=[];oracle=[];waits=[];mutations=[];cpu_samples=[];sampler_stop=threading.Event()
 for i in range(len(ph['request_records'])*case['history_multiplier']):rt.pending[f'old:{i}']=dict(request_id=f'old:{i}',model=S,task='asr',status='completed')
 def mutate(q,status):
  try:
   while True:
    z=q.get()
    if z is None:break
    r,x=z
    if status=='completed':assert sent[r['request_id']].wait(5)
    tick=time.perf_counter_ns()
    with rt.lock:
     entered=time.perf_counter_ns();r['status']=status
     if status=='submitted':r['actual_submit_time_ns']=x['actual_submit_time_ns']
     else:r.update(completion_time_ns=x['completion_time_ns'],worker_response=x['worker_response'],worker_latency_ms=x['worker_latency_ms'],worker_queue_wait_ms=x['worker_queue_wait_ms'])
     rt.refresh(r);mutations.append(dict(kind=status,request_id=r['request_id'],perf_ns=time.perf_counter_ns(),wait_ms=(entered-tick)/1e6))
    if status=='submitted':sent[r['request_id']].set()
  except BaseException:errors.append(traceback.format_exc())
 def sample_cpu():
  prevwall=time.perf_counter_ns();prevcpu=time.process_time_ns()
  while not sampler_stop.wait(.5):
   wall=time.perf_counter_ns();cpu=time.process_time_ns();cpu_samples.append(dict(perf_ns=wall,cpu_percent_one_core=100*(cpu-prevcpu)/(wall-prevwall),cgroup=cgroup()));prevwall=wall;prevcpu=cpu
 threads=[threading.Thread(target=mutate,args=(sq,'submitted'),name='sender',daemon=True),threading.Thread(target=mutate,args=(cq,'completed'),name='collector',daemon=True)]
 monitor=SegmentedMonitor(rt,dict(run_id=case['id']),1,output_dir=OUT/'runs'/case['id'],sensor=lambda:dict(time_ns=time.perf_counter_ns(),unix_ns=time.time_ns(),used_mib=0,owned_used_mib=0,foreign_pids=[]))
 # Original journal JSON/fsync stay real; segment their telemetry-thread CPU/wall.
 original_dumps=json.dumps;original_fsync=os.fsync
 def wrapped(fun,kind,*a,**kw):
  if threading.current_thread() is not monitor.thread:return fun(*a,**kw)
  tick=time.perf_counter_ns();cpu=time.thread_time_ns()
  try:return fun(*a,**kw)
  finally:monitor.spans.append(dict(kind=kind,begin_ns=tick,end_ns=time.perf_counter_ns(),cpu_ms=(time.thread_time_ns()-cpu)/1e6,native_tid=threading.get_native_id()))
 json.dumps=lambda *a,**kw:wrapped(original_dumps,'json_serialization',*a,**kw)
 os.fsync=lambda *a,**kw:wrapped(original_fsync,'fsync',*a,**kw)
 sampler=threading.Thread(target=sample_cpu,name='cpu_sampler',daemon=True)
 epoch=time.perf_counter_ns();cpu_start=time.process_time_ns();cg0=cgroup();fatal=None;active_peak=0;overlimit=[];publish=[]
 try:
  for t in threads:t.start()
  monitor.set_active(dict(label='measurement',planned_start_time_ns=start,planned_end_time_ns=start+case['logical_seconds']*10**9,planned_duration_seconds=case['logical_seconds'],request_records=records));monitor.start();sampler.start()
  for ts,kind,x in events:
   deadline=epoch+round((ts-start)/case['arrival_scale']);remaining=(deadline-time.perf_counter_ns())/1e9
   if remaining>0:sampler_stop.wait(remaining)
   r=by[x['request_id']]
   if kind==1:sq.put((r,x));continue
   if kind==2:cq.put((r,x));continue
   monitor.context=dict(request_id=r['request_id'],logical_seconds=(ts-start)/1e9,history_records=len(rt.pending),arrival_scale=case['arrival_scale'])
   try:monitor.check()
   except RuntimeError as e:
    if not str(e).startswith('CONTROL_CHECK_OVER_10MS'):raise
    overlimit.append(repr(e)) # Diagnosis continues; this case remains failed, never discard outlier.
   tick=time.perf_counter_ns()
   with rt.lock:
    entered=time.perf_counter_ns();waits.append((entered-tick)/1e6);count,oldest=rt.pressure(ts);active=list(rt.active.values());assert count==len(active);assert oldest==max(((ts-z['enqueued_time_ns'])/1e9 for z in active),default=0)
    if r['task']=='asr':
     projection=rt.projection(G,S,DEFAULT);decision=router.decide(ts,inputs[r['request_id']],projection)
    rt.queues.setdefault(r['model'],queue.Queue(maxsize=10000));rt.enqueue(r,{});r['enqueued_time_ns']=ts
   cost=(time.perf_counter_ns()-tick)/1e6
   if r['task']=='asr':route.append(dict(request_id=r['request_id'],logical_seconds=(ts-start)/1e9,wall_ms=cost,lock_wait_ms=waits[-1],decision=decision))
   # Legacy ON reproduces original test-only full scan under shared runtime lock.
   if case['legacy_scan']:
    tick2=time.perf_counter_ns()
    with rt.lock:assert legacy(rt)==rt.projection(G,S,DEFAULT)
    oracle.append(dict(request_id=r['request_id'],wall_ms=(time.perf_counter_ns()-tick2)/1e6))
   else:
    # Exact independent oracle can be reconstructed after timing using current active records.
    with rt.lock:
     observed=[dict(z) for z in rt.active.values()];projected=rt.projection(G,S,DEFAULT)
    mock=SimpleNamespace(pending={z['request_id']:z for z in observed});timings.append((mock,projected))
   active_peak=max(active_peak,len(rt.active))
 except BaseException:fatal=traceback.format_exc()
 finally:
  sq.put(None);cq.put(None)
  for t in threads:
   if t.ident:t.join(5)
  try:monitor.close()
  except BaseException:errors.append(traceback.format_exc())
  sampler_stop.set()
  if sampler.ident:sampler.join(2)
  wall=(time.perf_counter_ns()-epoch)/1e9;cpu=(time.process_time_ns()-cpu_start)/1e9;cg1=cgroup()
  try:monitor.flush_active()
  except BaseException:errors.append(traceback.format_exc())
  json.dumps=original_dumps;os.fsync=original_fsync
  for mock,projected in timings:assert legacy(mock)==projected
  prefix='runs/'+case['id']+'/'
  csvsave(prefix+'control_probes.csv',monitor.detailed);csvsave(prefix+'routing_times.csv',route);csvsave(prefix+'telemetry_spans.csv',monitor.spans);csvsave(prefix+'mutation_locks.csv',mutations);csvsave(prefix+'cpu_samples.csv',cpu_samples);csvsave(prefix+'legacy_scan_times.csv',oracle)
  result=dict(**case,offered=len(rr),completed=sum(r['status']=='completed' for r in records),wall_seconds=wall,cpu_seconds=cpu,cpu_percent_one_core=100*cpu/wall,cpu_max_sample_percent=max([z['cpu_percent_one_core'] for z in cpu_samples],default=0),cgroup_delta={k:cg1[k]-cg0[k] for k in cg0},router=quant(z['wall_ms'] for z in route),control=quant(z['wall_ms'] for z in monitor.detailed),control_acquire=quant(z['acquire_wall_ms'] for z in monitor.detailed),router_lock_wait=quant(waits),legacy_scan_cost=quant(z['wall_ms'] for z in oracle),publish_ms=[z*1000 for z in monitor.publish_seconds],max_active=active_peak,control_violations=overlimit,errors=errors,fatal=fatal,guard_violations=VIOLATIONS,offline_projection_checks=len(timings),threads_exited=all(not t.is_alive() for t in threads),cpu_only=True,history_shallow_bytes=sys.getsizeof(rt.pending)+sum(sys.getsizeof(x) for x in rt.pending.values()))
  result['passed']=not(fatal or errors or overlimit or VIOLATIONS) and result['router']['p95_ms']<=1 and max(result['publish_ms'],default=0)<10 and result['completed']==len(rr)
  save(prefix+'summary.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('case');args=parser.parse_args();manifest=read(OUT/'diagnostic_protocol.json');run(next(c for c in manifest['cases'] if c['id']==args.case))
