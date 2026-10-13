import fcntl, traceback
from .common import *
from .runtime import configure, TimedMonitor, r31
from .harness import run

def assess(raw):
 configure();o=r31.assess(raw);ph=raw.get('phases',{}).get('measurement')
 if ph and raw['slot']['arm']=='dynamic':
  ds=ph['routing_decisions'];rr=[r for r in ph['request_records'] if r['task']=='asr']
  checks=dict(one_decision_per_ASR=len(ds)==len(rr),actual_routes_match_decisions=all((r['model']==G)==d['select_G'] for r,d in zip(rr,ds)),VQA_protected_at_admission=all(d['Phi_inflight']==0 and d['Phi_ASR_inflight']==0 and d['Phi_VQA_inflight']==0 for d in ds if d['select_G']),causal_state_projection=all(not d['uses_future'] and all(x['enqueue_perf_ns']<=d['decision_perf_ns'] and (x['submit_perf_ns'] is None or x['submit_perf_ns']<=d['decision_perf_ns']) for x in d['past_state_snapshot']) for d in ds),router_overhead_p95_below1ms=bool(ds) and np.percentile([d['routing_compute_ms'] for d in ds],95)<1.)
  o['dynamic_validation']=checks;o['technical_valid']=bool(o['technical_valid'] and all(checks.values()));o['eligible']=bool(o['eligible'] and o['technical_valid']);o['actual_dynamic_G_executed']=o.get('G_completed',0)>0
  if not o['eligible']:o['U']=None;o['components']['U']=None
 return o

def execute(slot,f):
 configure()
 for r in f['sources']+f['frozen_inputs']:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 for p,v in f['checkpoint_metadata'].items():assert [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==v,p
 print('RUN_START',slot['run_id'],slot['arm'],flush=True);raw=run(slot,f);m=TimedMonitor.instances[-1]
 raw['control_latency_seconds']=dict(max_check=max(m.check_seconds,default=0),max_publish=max(m.publish_seconds,default=0));raw['control_nonblocking']=max(m.check_seconds+m.publish_seconds,default=0)<.01;raw['router_source']=ref(OUT/'vqa_aware_dynamic_router.py');save('raw_runs/'+slot['run_id']+'.json',raw)
 print('RUN_FINISH',slot['run_id'],raw['valid_session'],raw['errors'],flush=True);return raw

def main():
 lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 f=read(OUT/'experiment_manifest.json');attempted=[];failed=None
 assert not list((OUT/'run_attempts').glob('*.json')), 'Campaign is single-use; no retries'
 try:
  for rid in f['formal_order']:
   slot=next(s for s in f['slots'] if s['run_id']==rid);attempted.append(rid);save('campaign_status.json',dict(status='INDEPENDENT_REPLICATION',run_id=rid,pid=os.getpid(),attempted=attempted,formal_max_runs=6))
   raw=execute(slot,f);o=assess(raw);save('formal_acceptance/'+rid+'.json',o)
   print('FORMAL_ACCEPT',rid,o['eligible'],o['U'],o['stability'],o['validation'],o.get('dynamic_validation'),flush=True)
   from .report import generate
   generate()
   # Scientific negative/unstable results are retained and do not replace a seed.
   # A transport/telemetry/resource failure blocks the campaign, with no retry.
   assert o['technical_valid'],('TECHNICAL_INVALID_NO_RETRY',rid)
 except BaseException as ex:
  failed=dict(error=repr(ex),traceback=traceback.format_exc());save('campaign_failure.json',failed);print('STOPPED',repr(ex),flush=True)
 finally:
  save('campaign_status.json',dict(status='STOPPED_INCONCLUSIVE' if failed else 'COMPLETED',attempted=attempted,failed=failed,formal_max_runs=6,no_retries=True,independent_replication=True))
  from .report import generate
  generate()
if __name__=='__main__':main()
