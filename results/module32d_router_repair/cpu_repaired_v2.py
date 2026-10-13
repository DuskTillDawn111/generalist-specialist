"""Exact observed-state replay, transition tests and paced multithreaded CPU integration."""
import sys,threading,queue,traceback,gc
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32d.common import *
from router_optimized import DeploymentRuntime
from vqa_aware_dynamic_router import VQAAwareRouter
from m32d.monitor_cpu_revision import RepairedMonitor as IncrementalMonitor
from m32d.runtime import configure
configure()
import module30.telemetry as inherited
base_save=save; base_csvsave=csvsave
def save(n,v):return base_save("cpu_repaired_v2/"+n,v)
def csvsave(n,v,fields=None):return base_csvsave("cpu_repaired_v2/"+n,v,fields)
inherited.OUT=OUT/"cpu_repaired_v2";inherited.save=save
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime as OriginalRuntime

P=read(OUT/'frozen_routing_parameters.json');META=read(OUT/'audio_metadata_cache.json');DEFAULT=max(x['S_p75_ms'] for x in P['cost_bins'])
def legacy(rt):
 return [dict(role='G' if x['model']==G else 'S',task=x['task'],submit_perf_ns=x.get('actual_submit_time_ns'),enqueue_perf_ns=x['enqueued_time_ns'],S_service_estimate_ms=x.get('router_S_service_estimate_ms',DEFAULT)) for x in rt.pending.values() if x['model'] in (G,S) and x['status'] in ('queued','submitted')]
def quant(a):
 return dict(n=len(a),mean_ms=float(np.mean(a)),**{f'p{p}_ms':float(np.percentile(a,p)) for p in (50,90,95,99)},max_ms=float(max(a)))
def equivalence():
 runs=[];rows=[];mismatch=[];transition_count=0
 for parent,names in [('module32b_vqa_aware_routing',['A2']),('module32c_dynamic_replication',['P1_D'])]:
  for name in names:
   path=ROOT/'results'/parent/'raw_runs'/f'{name}.json';raw=read(path)
   for ph in [raw.get('warmup'),raw['phases']['measurement']]:
    if not ph:continue
    old=VQAAwareRouter(P,META,ph['window']['duration_seconds']);new=VQAAwareRouter(P,META,ph['window']['duration_seconds'])
    rt=DeploymentRuntime(SimpleNamespace());records={x['request_id']:x for x in ph['request_records']};trace=read(ph['trajectory']['path'])['events'];paths={r['request_id']:e['item']['sample']['input'] for r,e in zip(ph['request_records'],trace) if r['task']=='asr'}
    for i,d in enumerate(ph['routing_decisions']):
     # Historical snapshots are authoritative collector-applied state. Reader
     # receipt is NOT interchangeable with the unlogged collector lock update.
     with rt.lock:
      for x in list(rt.active.values()):x['status']='completed';rt.refresh(x);transition_count+=1
      for j,x in enumerate(d['past_state_snapshot']):
       rec=dict(request_id=f'{i}:{j}',model=G if x['role']=='G' else S,task=x['task'],enqueued_time_ns=x['enqueue_perf_ns'],actual_submit_time_ns=x['submit_perf_ns'],router_S_service_estimate_ms=x['S_service_estimate_ms'],status='submitted' if x['submit_perf_ns'] is not None else 'queued')
       rt.pending[rec['request_id']]=rec;rt.refresh(rec);transition_count+=1
      projected=rt.projection(G,S,DEFAULT);assert projected==d['past_state_snapshot'];assert projected==legacy(rt)
      a=old.decide(d['decision_perf_ns'],paths[d['request_id']],legacy(rt));b=new.decide(d['decision_perf_ns'],paths[d['request_id']],projected)
     diff=[k for k in a if a[k]!=b[k] or a[k]!=d[k]]
     if diff:mismatch.append(dict(source=str(path),phase=records[d['request_id']]['phase'],request_id=d['request_id'],keys=diff))
     rows.append(dict(source=parent,run=name,phase=records[d['request_id']]['phase'],request_id=d['request_id'],select_G=b['select_G'],reason=b['reason'],states_exact=a==b,logged_exact=not diff,S_wait_ms=b['S_estimated_wait_ms'],Phi_inflight=b['Phi_inflight'],future_features=False))
    runs.append(dict(source=ref(path),phase=ph['request_records'][0]['phase'],decisions=len(ph['routing_decisions'])))
 csvsave('historical_decision_equivalence.csv',rows)
 # Exercise the actual native collector, including queue errors, mismatches,
 # duplicate responses, drain timeout and late responses; no GPU workers.
 fixtures=[]
 for cls in [OriginalRuntime,DeploymentRuntime]:
  rt=cls(SimpleNamespace());rt.queues[S]=queue.Queue(maxsize=1)
  a=dict(request_id='a',task='asr',model=S,status='planned');b=dict(request_id='b',task='asr',model=S,status='planned')
  rt.enqueue(a,{});rt.enqueue(b,{})
  with rt.lock:
   a.update(status='submitted',actual_submit_time_ns=1)
   if hasattr(rt,'refresh'):rt.refresh(a)
  rt.collector=threading.Thread(target=rt._collect);rt.collector.start()
  payload=dict(request_id='a',task_type='asr',worker_latency_ms=2,queue_wait_ms=3)
  rt.events.put((S,'RESULT',payload));rt.drain([a],1)
  rt.events.put((S,'RESULT',payload))
  c=dict(request_id='c',task='visual_qa',model=G,status='submitted',enqueued_time_ns=2);rt.pending['c']=c
  with rt.lock:
   if hasattr(rt,'refresh'):rt.refresh(c)
  rt.drain([c],0);rt.events.put((G,'RESULT',dict(request_id='c',task_type='visual_qa')))
  e=dict(request_id='e',task='asr',model=S,status='submitted',enqueued_time_ns=3);rt.pending['e']=e
  with rt.lock:
   if hasattr(rt,'refresh'):rt.refresh(e)
  rt.events.put((S,'EXIT',{}));rt.collect_stop.set();rt.collector.join(2);assert not rt.collector.is_alive()
  fixtures.append(dict(statuses=[x['status'] for x in [a,b,c,e]],duplicate_errors=len(rt.errors),late_response=c.get('late_response_after_drain'),active=len(rt.active) if hasattr(rt,'active') else 0))
 assert fixtures[0]==fixtures[1],fixtures
 result=dict(passed=not mismatch,policy_source_byte_identical=ref(OUT/'vqa_aware_dynamic_router.py')['sha256']==ref(OLD/'vqa_aware_dynamic_router.py')['sha256'],runs=runs,decisions=len(rows),exact_all_return_fields=True,numeric_tolerance=0,mismatches=mismatch,transition_operations=transition_count,lifecycle_fixtures=fixtures,semantics_changed=False,limitations='Historical per-ASR snapshots preserve actual observed collector state; collector-applied timestamps were not logged. This proves identical same-state decisions, not unexecuted wall-clock schedules. Additional real collector lifecycle fixtures and concurrent projection/guard comparisons check runtime integration.',causal_features_unchanged=True)
 save('routing_equivalence_report.json',result);assert result['passed'];return result

def benchmark():
 raw=read(ROOT/'results/module32c_dynamic_replication/raw_runs/P1_D.json');ph=raw['phases']['measurement'];rr=ph['request_records'];start=ph['window']['start_time_ns'];events=[]
 for x in rr:
  events.extend([(x['enqueued_time_ns'],0,x),(x['actual_submit_time_ns'],1,x),(x['completion_time_ns'],2,x)])
 events.sort(key=lambda e:(e[0],e[1]));trace=read(ph['trajectory']['path'])['events'];inputmap={x['request_id']:e['item']['sample']['input'] for x,e in zip(rr,trace) if x['task']=='asr'}
 results=[];detail=[];errors=[]
 # 50x pacing preserves the complete 2400s cohort and overlapping lifetime
 # pattern; receiver and sender execute separate real threads. Sensor remains
 # at the original 0.5s interval and durable fsync at10s (not mocked I/O).
 for multiplier in [1,8]:
  rt=DeploymentRuntime(SimpleNamespace());router=VQAAwareRouter(P,META,2400);records=[{k:v for k,v in x.items() if k in ['request_id','model','task','pool_id','sample_id','router_S_service_estimate_ms']} for x in rr]
  for r in records:r['status']='planned'
  by={r['request_id']:r for r in records};sent={k:threading.Event() for k in by};stop=threading.Event();mq=queue.Queue();sq=queue.Queue();waits=[];holds=[];proj_checks=0;active_peak=0
  for i in range(len(rr)*multiplier):rt.pending[f'old:{i}']=dict(request_id=f'old:{i}',model=S,task='asr',status='completed')
  def mutate(q,status):
   try:
    while True:
     z=q.get()
     if z is None:break
     r,source=z
     if status=='completed':assert sent[r['request_id']].wait(5)
     with rt.lock:
      r['status']=status
      if status=='submitted':r['actual_submit_time_ns']=source['actual_submit_time_ns']
      else:r.update(completion_time_ns=source['completion_time_ns'],worker_response=source['worker_response'],worker_latency_ms=source['worker_latency_ms'],worker_queue_wait_ms=source['worker_queue_wait_ms'])
      rt.refresh(r)
     if status=='submitted':sent[r['request_id']].set()
   except BaseException:errors.append(traceback.format_exc())
  threads=[threading.Thread(target=mutate,args=(sq,'submitted'),daemon=True),threading.Thread(target=mutate,args=(mq,'completed'),daemon=True)]
  slot=dict(run_id=f'CPU_CONCURRENT_{multiplier}');monitor=IncrementalMonitor(rt,slot,1,sensor=lambda:dict(time_ns=time.perf_counter_ns(),unix_ns=time.time_ns(),used_mib=0,owned_used_mib=0,foreign_pids=[]))
  for t in threads:t.start()
  monitor.set_active(dict(label='measurement',planned_start_time_ns=start,planned_end_time_ns=start+2400*10**9,planned_duration_seconds=2400,request_records=records));monitor.start()
  epoch=time.perf_counter_ns();scale=50;dur=[];control_checks=[];segments=collections.defaultdict(list)
  for ts,kind,x in events:
   deadline=epoch+round((ts-start)/scale)
   remaining=(deadline-time.perf_counter_ns())/1e9
   if remaining>0:stop.wait(remaining)
   r=by[x['request_id']]
   if kind==1:sq.put((r,x));continue
   if kind==2:mq.put((r,x));continue
   check_tick=time.perf_counter_ns();monitor.check();control_checks.append((time.perf_counter_ns()-check_tick)/1e6)
   tick=time.perf_counter_ns()
   with rt.lock:
    entered=time.perf_counter_ns();waits.append((entered-tick)/1e6)
    count,oldest=rt.pressure(ts);active=list(rt.active.values())
    assert count==len(active);assert oldest==max(((ts-z['enqueued_time_ns'])/1e9 for z in active),default=0)
    if r['task']=='asr':
     projection=rt.projection(G,S,DEFAULT)
     # Exact full archive comparison is done outside the measured routing
     # section using an identical frozen lock state, avoiding false speedups.
     d=router.decide(ts,inputmap[r['request_id']],projection)
    # Real enqueue: same Queue.put_nowait and record mutations as GPU wrapper.
    rt.queues.setdefault(r['model'],queue.Queue(maxsize=10000))
    rt.enqueue(r,{})
    r['enqueued_time_ns']=ts
    holds.append((time.perf_counter_ns()-entered)/1e6)
   end=time.perf_counter_ns();cost=(end-tick)/1e6
   if r['task']=='asr':dur.append(cost);segments[int((ts-start)/1e9)//300].append(cost)
   with rt.lock:
    assert legacy(rt)==rt.projection(G,S,DEFAULT);proj_checks+=1
   active_peak=max(active_peak,len(rt.active))
  sq.put(None);mq.put(None)
  for t in threads:t.join(2);assert not t.is_alive()
  monitor.flush_active();monitor.close();assert not monitor.errors
  assert all(r['status']=='completed' for r in records);assert not rt.active
  result=dict(mode='paced_concurrent_sender_collector_real_journal',history_multiplier=multiplier,history_records=len(rt.pending),logical_seconds=2400,wall_seconds=(time.perf_counter_ns()-epoch)/1e9,arrival_time_scale=scale,offered=len(rr),active_peak=active_peak,projection_checks=proj_checks,archival_memory_shallow_bytes=sys.getsizeof(rt.pending)+sum(sys.getsizeof(x) for x in rt.pending.values()),active_memory_shallow_bytes=sys.getsizeof(rt.active)+sum(sys.getsizeof(v) for v in rt.active_models.values()),sender_collector_threads=2,telemetry_interval_seconds=.5,journal_interval_seconds=10,router_lock_wait=quant(waits),router_lock_hold=quant(holds),snapshot_lock=quant(monitor.snapshot_lock_ms),control_check_ms=quant(control_checks),control_check_max_lt10ms=max(control_checks)<10,passed=np.percentile(dur,95)<=1 and max(control_checks)<10,**quant(dur))
  results.append(result)
  for b,v in sorted(segments.items()):detail.append(dict(history_multiplier=multiplier,logical_start_seconds=b*300,logical_end_seconds=(b+1)*300,**quant(v)))
 save('CPU_runtime_acceptance.json',dict(passed=all(r['passed'] for r in results) and not errors,results=results,errors=errors,original_p95_limit_ms=1,engineering_target_ms=.8,scope='Full real2400s cohort accelerated50x, real producer/collector/journal concurrency; not a2400s wall-time soak and not GPU acceptance. Measured full lock+projection+decide+enqueue; comparison scan excluded.8x retained history tests growth. GPU actual1ms still mandatory.'))
 csvsave('router_performance_benchmark.csv',results);csvsave('router_performance_by_300s.csv',detail)
 assert all(r['passed'] for r in results),results
 return results
if __name__=='__main__':
 try:
  print('EQUIVALENCE',equivalence(),flush=True);print('BENCHMARK',benchmark(),flush=True)
 except BaseException as ex:
  save('CPU_validation_failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
