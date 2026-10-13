import fcntl, traceback
from .common import *
from .runtime import configure, TimedMonitor, r31
from .harness import run

def assess(raw):
 configure();o=r31.assess(raw);ph=raw.get('phases',{}).get('measurement')
 if ph:
  w=ph['window'];ss=[s for s in raw['telemetry']['samples'] if w['start_time_ns']<=s['time_ns']<=w['end_time_ns']]
  f=read(OUT/'frozen_validation_manifest.json')
  ec=dict(power_limit_unchanged=bool(ss) and all(s['power_limit_w']==f['GPU']['power_limit_w'] for s in ss),no_observed_thermal_slowdown=bool(ss) and all(s['hw_thermal_slowdown']=='Not Active' and s['sw_thermal_slowdown']=='Not Active' for s in ss))
  o['environment_checks']=ec;o['technical_valid']=bool(o['technical_valid'] and all(ec.values()));o['eligible']=bool(o['eligible'] and o['technical_valid'])
 if ph and raw['slot']['arm']=='dynamic':
  ds=ph['routing_decisions'];rr=[r for r in ph['request_records'] if r['task']=='asr']
  checks=dict(one_decision_per_ASR=len(ds)==len(rr),actual_routes_match_decisions=all((r['model']==G)==d['select_G'] for r,d in zip(rr,ds)),VQA_protected_at_admission=all(d['Phi_inflight']==0 and d['Phi_ASR_inflight']==0 and d['Phi_VQA_inflight']==0 for d in ds if d['select_G']),causal_state_projection=all(not d['uses_future'] and all(x['enqueue_perf_ns']<=d['decision_perf_ns'] and (x['submit_perf_ns'] is None or x['submit_perf_ns']<=d['decision_perf_ns']) for x in d['past_state_snapshot']) for d in ds),router_overhead_p95_below1ms=bool(ds) and np.percentile([d['routing_compute_ms'] for d in ds],95)<1.)
  o['dynamic_validation']=checks;o['technical_valid']=bool(o['technical_valid'] and all(checks.values()));o['eligible']=bool(o['eligible'] and o['technical_valid']);o['actual_dynamic_G_executed']=o.get('G_completed',0)>0
 if not o['eligible']:o['U']=None;o['components']['U']=None
 return o

def paired_environment(a,b):
 f=read(OUT/'frozen_validation_manifest.json')['environment_control']
 td=abs(a['temperature']['mean']-b['temperature']['mean']);cd=abs(a['sm_clock']['mean']/b['sm_clock']['mean']-1);pd=abs(a['postwarm_temperature_mean']-b['postwarm_temperature_mean'])
 return dict(passed=td<=f['measurement_mean_temperature_difference_max_c'] and cd<=f['measurement_mean_sm_clock_relative_difference_max'] and pd<=f['postwarm_pair_temperature_difference_max_c'],mean_temperature_difference_c=td,mean_sm_clock_relative_difference=cd,postwarm_temperature_difference_c=pd)

def execute(slot,f):
 configure()
 for r in f['sources']+f['frozen_inputs']:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 for p,v in f['checkpoint_metadata'].items():assert [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==v,p
 print('RUN_START',slot['run_id'],slot['arm'],flush=True);raw=run(slot,f);m=TimedMonitor.instances[-1]
 raw['control_latency_seconds']=dict(max_check=max(m.check_seconds,default=0),max_publish=max(m.publish_seconds,default=0));raw['control_nonblocking']=max(m.check_seconds+m.publish_seconds,default=0)<.01;raw['router_source']=ref(OUT/'vqa_aware_dynamic_router.py');save('raw_runs/'+slot['run_id']+'.json',raw)
 print('RUN_FINISH',slot['run_id'],raw['valid_session'],raw['errors'],flush=True);return raw

def main():
 lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 f=read(OUT/'experiment_manifest.json');attempted=[];failed=None;engineering=[];stop_gate=None
 assert not list((OUT/'run_attempts').glob('*.json')), 'Campaign is single-use; no retries'
 try:
  for rid in f['engineering_order']:
   slot=next(s for s in f['engineering_slots'] if s['run_id']==rid);save('campaign_status.json',dict(status='ENGINEERING_PREFLIGHT',run_id=rid,pid=os.getpid(),attempted=attempted))
   raw=execute(slot,f);o=assess(raw)
   o['U']=None;o['components']['U']=None;o['engineering_only']=True;o['not_independent_utility_evidence']=True
   save('engineering_acceptance/'+rid+'.json',o);engineering.append(o)
   print('ENGINEERING_ACCEPT',rid,o['eligible'],o['validation'],o.get('dynamic_validation'),o.get('environment_checks'),flush=True)
   if not o['technical_valid'] or not o['eligible']:
    stop_gate='ROUTER_PERFORMANCE_NOT_FIXED' if o.get('dynamic_validation',{}).get('router_overhead_p95_below1ms') is False else 'GPU_ENVIRONMENT_UNCONTROLLED' if not all(o.get('environment_checks',{}).values()) else 'DYNAMIC_VALIDATION_INVALID'
    raise RuntimeError(('ENGINEERING_PREFLIGHT_FAILED_NO_RETRY',rid))
  env=paired_environment(*engineering);save('engineering_environment_comparison.json',env)
  if not env['passed']:stop_gate='GPU_ENVIRONMENT_UNCONTROLLED';raise RuntimeError('ENGINEERING_ENVIRONMENT_PAIR_FAILED')
  save('stage_D_gate.json',dict(passed=True,engineering_only=True,formal_U_used=False,environment=env,acceptance=[ref(OUT/'engineering_acceptance'/f'{x["run_id"]}.json') for x in engineering]))
  for rid in f['formal_order']:
   slot=next(s for s in f['slots'] if s['run_id']==rid);attempted.append(rid);save('campaign_status.json',dict(status='INDEPENDENT_REPLICATION',run_id=rid,pid=os.getpid(),attempted=attempted,formal_max_runs=6))
   raw=execute(slot,f);o=assess(raw);save('formal_acceptance/'+rid+'.json',o)
   print('FORMAL_ACCEPT',rid,o['eligible'],o['U'],o['stability'],o['validation'],o.get('dynamic_validation'),flush=True)
   from .report import generate
   generate()
   # Scientific negative/unstable results are retained and do not replace a seed.
   # A transport/telemetry/resource failure blocks the campaign, with no retry.
   if not o['technical_valid']:
    stop_gate='ROUTER_PERFORMANCE_NOT_FIXED' if o.get('dynamic_validation',{}).get('router_overhead_p95_below1ms') is False else 'GPU_ENVIRONMENT_UNCONTROLLED' if not all(o.get('environment_checks',{}).values()) else 'DYNAMIC_VALIDATION_INVALID'
    raise RuntimeError(('TECHNICAL_INVALID_NO_RETRY',rid))
   ids=[x for x in f['formal_order'] if x.startswith(f'P{slot["pair"]}_')]
   if all((OUT/'formal_acceptance'/f'{x}.json').exists() for x in ids):
    env=paired_environment(*[read(OUT/'formal_acceptance'/f'{x}.json') for x in ids]);save(f'paired_environment/P{slot["pair"]}.json',env)
    if not env['passed']:stop_gate='GPU_ENVIRONMENT_UNCONTROLLED';raise RuntimeError(('PAIRED_ENVIRONMENT_FAILED_NO_RETRY',slot['pair']))
 except BaseException as ex:
  failed=dict(error=repr(ex),traceback=traceback.format_exc());save('campaign_failure.json',failed);print('STOPPED',repr(ex),flush=True)
 finally:
  save('campaign_status.json',dict(status='STOPPED_INCONCLUSIVE' if failed else 'COMPLETED',attempted=attempted,engineering_attempted=[o['run_id'] for o in engineering],failed=failed,stop_gate=stop_gate,formal_max_runs=6,no_retries=True,independent_replication=True))
  from .report import generate
  generate()
if __name__=='__main__':main()
