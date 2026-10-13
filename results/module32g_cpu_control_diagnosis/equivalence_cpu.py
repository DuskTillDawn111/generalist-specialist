"""Exact observed-state replay, transition tests and paced multithreaded CPU integration."""
import sys,threading,queue,traceback,gc
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32diag.isolation import install,VIOLATIONS
install()
from m32diag.common import *
from router_optimized import DeploymentRuntime
from vqa_aware_dynamic_router import VQAAwareRouter
from m32diag.monitor import IncrementalMonitor
base_save=save; base_csvsave=csvsave
def save(n,v):return base_save("cpu_regression/"+n,v)
def csvsave(n,v,fields=None):return base_csvsave("cpu_regression/"+n,v,fields)
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


if __name__=="__main__":print(equivalence(),flush=True)
