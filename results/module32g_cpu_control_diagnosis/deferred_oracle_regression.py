"""Unmeasured full archive oracle on complete historical lifecycle, CPU only."""
from control_diagnosis import *
raw=read(ROOT/'results/module32c_dynamic_replication/raw_runs/P1_D.json');ph=raw['phases']['measurement'];rr=ph['request_records'];trace=read(ph['trajectory']['path'])['events']
inputs={r['request_id']:e['item']['sample']['input'] for r,e in zip(rr,trace) if r['task']=='asr'}
rt=DeploymentRuntime(SimpleNamespace());records={r['request_id']:dict(request_id=r['request_id'],model=r['model'],task=r['task'],status='planned',router_S_service_estimate_ms=r.get('router_S_service_estimate_ms',DEFAULT)) for r in rr}
for i in range(len(rr)*8):rt.pending[f'old:{i}']=dict(request_id=f'old:{i}',model=S,task='asr',status='completed')
a=VQAAwareRouter(P,META,2400);b=VQAAwareRouter(P,META,2400);rows=[];decisions=0
for ts,kind,x in sorted([(x[k],i,x) for x in rr for i,k in enumerate(['enqueued_time_ns','actual_submit_time_ns','completion_time_ns'])],key=lambda e:(e[0],e[1])):
 r=records[x['request_id']]
 with rt.lock:
  if kind==0:
   if r['task']=='asr':assert a.decide(ts,inputs[r['request_id']],legacy(rt))==b.decide(ts,inputs[r['request_id']],rt.projection(G,S,DEFAULT));decisions+=1
   rt.queues.setdefault(r['model'],queue.Queue(maxsize=10000));rt.enqueue(r,{});r['enqueued_time_ns']=ts
  elif kind==1:r.update(status='submitted',actual_submit_time_ns=ts);rt.refresh(r)
  else:r.update(status='completed',completion_time_ns=ts);rt.refresh(r)
  assert legacy(rt)==rt.projection(G,S,DEFAULT)
  rows.append(dict(request_id=r['request_id'],kind=kind,perf_ns=ts,active=len(rt.active),archive=len(rt.pending),exact_full_archive_projection=True))
csvsave('cpu_regression/full_lifecycle_oracle.csv',rows)
save('cpu_regression/deferred_oracle_regression.json',dict(passed=True,transitions=len(rows),ASR_same_state_decisions=decisions,history_multiplier=8,complete_archive_records=len(rt.pending),all_completed=all(r['status']=='completed' for r in records.values()),final_active=len(rt.active),future_features=False,scope='Deterministic historical timestamps simulate completed lifecycles; full oracle runs outside performance tests. Logged historical same-state1381 replay remains authoritative; not a counterfactual utility replay.'))
print('FULL_ARCHIVE_ORACLE_PASS',len(rows),decisions,flush=True)
