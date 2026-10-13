import sys,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32b.common import *
from vqa_aware_dynamic_router import VQAAwareRouter
p=read(OUT/'draft_routing_parameters.json');metadata=read(OUT/'audio_metadata_cache.json');short=next(k for k,v in metadata.items() if v<=p['duration_cuts_seconds'][0]);other=next(k for k,v in metadata.items() if v>p['duration_cuts_seconds'][1]);now=10**12
ss=[dict(role='S',task='asr',submit_perf_ns=now,enqueue_perf_ns=now,S_service_estimate_ms=2000.) for i in range(3)]
def router():
 r=VQAAwareRouter(copy.deepcopy(p),metadata,600);r.asr_seen=100;return r
r=router();assert not r.decide(now,short,[])['select_G']
for task in ['asr','visual_qa']:
 r=router();z=r.decide(now,short,ss+[dict(role='G',task=task,submit_perf_ns=now,enqueue_perf_ns=now,S_service_estimate_ms=0.)]);assert not z['select_G']
r=router();assert not r.decide(now,other,ss)['select_G'];r=router();z=r.decide(now,short,ss);assert z['select_G'];assert not r.decide(now+1,short,ss)['select_G']
r=router();r.risk_used=3.6;assert not r.decide(now,short,ss)['select_G'];r=router();r.asr_seen=0;assert not r.decide(now,short,ss)['select_G']
# Poison data outside the allowed state schema cannot alter a decision: no futures are passed at all.
a=router().decide(now,short,ss);b=router().decide(now,short,copy.deepcopy(ss));assert a==b
calls=[]
for i in range(2000):
 r=router();t=time.perf_counter_ns();r.decide(now,short,ss);calls.append((time.perf_counter_ns()-t)/1e6)
assert np.percentile(calls,95)<1.
save('CPU_router_acceptance.json',dict(passed=True,checks=['low_S_pressure','VQA_pending_ban','ASR_inflight_ban','long_execution_tail_ban','causal_share_limit','future_VQA_exposure_limit','cooldown','determinism_without_future_inputs'],timing_ms=stats(calls),required_route_compute_p95_ms=1.,inference_executed=False))
print('CPU_ROUTER_PASSED',stats(calls))
