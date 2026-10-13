"""Causal admission only. No future arrivals, labels, native START/END or IDs as policy features.
The shared Phi worker remains nonpreemptive. Pending VQA always blocks new ASR admission;
ASR already executing may still block a future VQA. No worker/inference changes.
"""
import math
class VQAAwareRouter:
 def __init__(self,parameters,audio_metadata,duration_seconds):
  self.p=parameters;self.metadata=audio_metadata;self.duration=duration_seconds
  self.armed=False;self.asr_seen=0;self.G_admitted=0;self.risk_used=0.;self.last_G_ns=None
 def bin(self,d):return int(d>self.p['duration_cuts_seconds'][0])+int(d>self.p['duration_cuts_seconds'][1])
 def cost(self,path):
  d=self.metadata.get(path)
  return (d,self.p['cost_bins'][self.bin(d)]) if d is not None else (None,None)
 def decide(self,now_ns,input_path,pending):
  self.asr_seen+=1;d,b=self.cost(input_path);p=self.p
  phi=[x for x in pending if x['role']=='G'];s=sorted([x for x in pending if x['role']=='S'],key=lambda x:x['submit_perf_ns'] or x['enqueue_perf_ns'])
  w=0.
  for i,x in enumerate(s):
   v=x['S_service_estimate_ms']
   if i==0 and x['submit_perf_ns'] is not None:v=max(0.,v-(now_ns-x['submit_perf_ns'])/1e6)
   w+=v
  if w>=p['S_pressure_on_ms']:self.armed=True
  elif w<=p['S_pressure_off_ms']:self.armed=False
  riskcost=None;score=None;exposure=None;why='eligible'
  if b is not None:
   riskcost=p['future_VQA_mean_risk_multiplier']*p['VQA_arrival_rate_qps']*b['G_second_moment_seconds2']/2*p['ASR_ref_ms']/p['VQA_ref_ms']*1000
   score=w+b['S_p75_ms']-b['G_p75_ms']-riskcost
   exposure=p['VQA_arrival_rate_qps']*b['G_p95_ms']/1000
  if b is None:why='unknown_input_cost'
  elif not self.armed:why='S_low_pressure'
  elif any(x['task']=='visual_qa' for x in phi):why='VQA_background_protection'
  elif phi:why='Phi_busy_backpressure'
  elif b['G_p95_ms']>p['G_tail_max_ms']:why='G_execution_tail_risk'
  elif score<=p['positive_margin_ms']:why='nonpositive_risk_adjusted_margin'
  elif self.G_admitted+1>math.floor(p['max_G_ASR_share']*self.asr_seen):why='causal_G_share_budget'
  elif self.risk_used+exposure>p['expected_VQA_blocked_fraction_budget']*p['VQA_arrival_rate_qps']*self.duration:why='VQA_risk_budget'
  elif self.last_G_ns is not None and (now_ns-self.last_G_ns)/1e9<p['min_G_cooldown_seconds']:why='cooldown'
  allow=why=='eligible'
  if allow:self.G_admitted+=1;self.risk_used+=exposure;self.last_G_ns=now_ns
  return dict(select_G=allow,reason=why,input_duration_seconds=d,S_estimated_wait_ms=w,S_inflight=len(s),Phi_inflight=len(phi),Phi_VQA_inflight=sum(x['task']=='visual_qa' for x in phi),Phi_ASR_inflight=sum(x['task']=='asr' for x in phi),pressure_armed=self.armed,G_p75_ms=b['G_p75_ms'] if b else None,G_p95_ms=b['G_p95_ms'] if b else None,S_p75_ms=b['S_p75_ms'] if b else None,future_VQA_expected_cost_ASR_equivalent_ms=riskcost,predicted_margin_ms=score,expected_future_VQA_exposure_used=self.risk_used,G_admitted=self.G_admitted,ASR_seen=self.asr_seen,uses_future=False)
