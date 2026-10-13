"""Module28 experimental router, independent of the deployment optimizer.
Inputs are host-visible outstanding work and already received service reports.
No model answers, future arrivals, handler START/END, or unreceived completions.
Costs are conservative development proxies, not causal counterfactual outcomes.
"""
from collections import defaultdict

class DynamicRouter:
 def __init__(self,parameters):
  self.p=parameters;self.mode=False;self.tokens=1.;self.last_time=None
  self.next_g_time=0.;self.fallback_until=0.;self.service={};self.predictions={}
  self.prediction_errors=[]
 def observe_completion(self,record,now):
  key=(record['model'],record['task']);value=record.get('worker_latency_ms')
  if value is None:return
  old=self.service.get(key,value);self.service[key]=(1-self.p['ewma_alpha'])*old+self.p['ewma_alpha']*value
  estimate=self.predictions.pop(record['request_id'],None)
  if estimate:
   ratio=value/estimate;self.prediction_errors.append(dict(request_id=record['request_id'],observed_service_ms=value,predicted_service_ms=estimate,ratio=ratio,observed_at_seconds=now))
   if ratio>self.p['prediction_error_ratio']:
    self.fallback_until=max(self.fallback_until,now+self.p['fallback_seconds']);self.mode=False
  if record['model']==self.p['G_worker'] and record['task']==self.p['target']:
   self.next_g_time=max(self.next_g_time,now+self.p['cooldown_seconds'])
  if record['model']==self.p['S_worker'] and record['task']==self.p['target'] and self.service[key]>self.p['S_service_alarm_ms']:
   self.fallback_until=max(self.fallback_until,now+self.p['fallback_seconds']);self.mode=False
 def estimate(self,pending):
  p=self.p;g=pending.get(p['G_worker'],[]);s=pending.get(p['S_worker'],[])
  # The oldest S job may still be in service; exclude it instead of guessing
  # its start/completion. Remaining backlog gets conservative P10 path costs.
  swait=sum(p['S_lower_by_task_ms'][r['task']] for r in s[1:])
  sexec=p['S_lower_by_task_ms'][p['target']]
  gexec=max(p['G_service_ms'],self.service.get((p['G_worker'],p['target']),0.))
  gwait=sum(p['G_upper_by_task_ms'][r['task']] for r in g)
  inflation=p['S_inflation_per_pending_ms']*len(s)
  # Arrival-rate normalized aggregate-latency externality from development.
  # Fixed estimates, not a look-ahead or a forecast of specific future jobs.
  background=p['background_cost_ms']
  return dict(S_pending=len(s),G_pending=len(g),G_target_inflight=sum(r['task']==p['target'] for r in g),S_wait_est_ms=swait,S_exec_est_ms=sexec,G_wait_est_ms=gwait,G_exec_est_ms=gexec,S_interference_est_ms=inflation,background_externality_est_ms=background,S_system_cost_proxy_ms=swait+sexec,G_system_cost_proxy_ms=gwait+gexec+inflation+background)
 def decide(self,request_id,now,pending,admitted=True,capacity_ok=True):
  p=self.p
  if self.last_time is not None:self.tokens=min(1.,self.tokens+max(0.,now-self.last_time)*p['G_token_rate_rps'])
  self.last_time=now;e=self.estimate(pending)
  advantage=e['S_system_cost_proxy_ms']-e['G_system_cost_proxy_ms']
  if self.mode and advantage<p['exit_margin_ms']:self.mode=False
  elif not self.mode and advantage>p['entry_margin_ms']:self.mode=True
  reason='S_LOWER_MARGINAL_COST'
  if not admitted or not capacity_ok:reason='G_NOT_ADMITTED_OR_CAPACITY'
  elif now<self.fallback_until:reason='PREDICTION_ERROR_FALLBACK'
  elif e['G_target_inflight']>=p['G_max_target_inflight']:reason='G_INFLIGHT_BACKPRESSURE'
  elif e['G_pending']>0:reason='G_BACKGROUND_OR_OTHER_WORK_PENDING'
  elif now<self.next_g_time:reason='G_COMPLETION_COOLDOWN'
  elif self.tokens<1.:reason='G_TOKEN_BACKPRESSURE'
  elif self.mode and advantage>p['exit_margin_ms']:
   reason='G_LOWER_SYSTEM_MARGINAL_COST';self.tokens-=1.;self.predictions[request_id]=e['G_exec_est_ms']
  route=p['G_worker'] if reason=='G_LOWER_SYSTEM_MARGINAL_COST' else p['S_worker']
  return route,dict(**e,reason=reason,route=route,margin_ms=advantage,hysteresis_mode=self.mode,tokens=self.tokens,fallback_until_seconds=self.fallback_until,capacity_ok=capacity_ok,admitted=admitted)
