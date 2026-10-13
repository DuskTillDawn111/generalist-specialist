"""Full frozen engineering/stability audit without calculating utility."""
from .common import *
from .engineering import assess as engineering
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS
from types import SimpleNamespace

def assess(raw,f):
 a=engineering(raw,f);checks=dict(a['checks']);ph=raw.get('phases',{}).get('measurement');workers=[]
 if ph:
  w=ph['window'];rs=ph['request_records'];ss=[s for s in raw['telemetry']['samples'] if w['start_time_ns']<=s['time_ns']<=w['end_time_ns']];ts=[s['time_ns'] for s in ss];us=[s['unix_ns'] for s in ss]
  checks.update(exact_2400s=w['duration_seconds']==2400. and w['end_time_ns']-w['start_time_ns']==2400000000000,
   telemetry_coverage=bool(ts) and (ts[0]-w['start_time_ns'])/1e9<2 and (w['end_time_ns']-ts[-1])/1e9<2 and max(np.diff(ts)/1e9,default=0)<5,
   timestamps_monotonic=all(b>a for a,b in zip(ts,ts[1:])) and all(b>a for a,b in zip(us,us[1:])),
   no_foreign_GPU=all(not s['foreign_pids'] for s in ss),
   bounded_drain=w['drain_seconds']<=60.1,
   unix_perf_alignment=bool(ss) and max(abs((s['unix_ns']-w['start_unix_ns'])-(s['time_ns']-w['start_time_ns']))/1e9 for s in ss)<1)
  ev=read(ph['trajectory']['path'])['events'];checks['route_uniforms_match']=len(rs)==len(ev) and all(r['routing_uniform']==e['routing_uniform'] and r['replica_uniform']==e['replica_uniform'] for r,e in zip(rs,ev))
  for worker,ident in raw['worker_identities'].items():
   wr=[r for r in rs if r['model']==worker]
   if not wr:continue
   tmp=dict(request_records=wr,measurement_window=w,status='completed',errors=raw['errors'],drain_timeout_seconds=60,worker_identities={worker:ident})
   family=raw['replica_to_family'][worker];e=reconstruct(tmp,{worker:SimpleNamespace(service_time_ms=f['plan']['models'][family]['service_time_ms'])},DEFAULT_THRESHOLDS);c=classify(e,DEFAULT_THRESHOLDS)
   workers.append(dict(worker=worker,classification=c,evidence=e,busy_fraction=sum(r['worker_latency_ms'] for r in wr if r['status']=='completed')/(2400*1000)))
  checks['workers_stable_original']=all(x['classification']['stability_state_v2']!='unstable' and (x['classification']['stability_state_v2']!='recoverable_congested' or x['classification'].get('final_recovery_confirmed',False)) for x in workers)
 # Zero cooperation/unstable scientific result is retained; transport/resource
 # and all hard timing/trace/inference/telemetry gates still stop the campaign.
 nontechnical={'G_and_S_ASR_executed','stability_original','workers_stable_original'}
 a.update(checks=checks,technical_valid=all(v for k,v in checks.items() if k not in nontechnical),eligible=all(checks.values()),workers=workers,passed=all(checks.values()),U=None,utility_computed=False,engineering_only=False)
 return a
