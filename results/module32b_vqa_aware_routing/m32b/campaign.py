import os,fcntl,traceback
from .common import *
from .runtime import configure,r31,TimedMonitor
from .harness import run
from module26.common import cohort

def assess(raw):
 configure();o=r31.assess(raw)
 ph=raw.get('phases',{}).get('measurement')
 if ph and raw['slot']['arm']=='dynamic':
  ds=ph['routing_decisions'];rr=[x for x in ph['request_records'] if x['task']=='asr'];checks=dict(one_decision_per_ASR=len(ds)==len(rr),actual_routes_match_decisions=all((r['model']==G)==d['select_G'] for r,d in zip(rr,ds)),VQA_protected_at_admission=all(d['Phi_inflight']==0 and d['Phi_ASR_inflight']==0 and d['Phi_VQA_inflight']==0 for d in ds if d['select_G']),causal_state_projection=all(not d['uses_future'] and all(x['submit_perf_ns'] is None or x['submit_perf_ns']<=d['decision_perf_ns'] for x in d['past_state_snapshot']) for d in ds),router_overhead_p95_below1ms=bool(ds) and np.percentile([d['routing_compute_ms'] for d in ds],95)<1.)
  o['dynamic_validation']=checks;o['technical_valid']=bool(o['technical_valid'] and all(checks.values()));o['eligible']=bool(o['eligible'] and o['technical_valid']);o['actual_dynamic_G_executed']=o.get('G_completed',0)>0
  if not o['eligible']:o['U']=None;o['components']['U']=None
 return o

def execute(s,f):
 configure()
 for r in f['sources']+f['frozen_inputs']:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 for p,v in f['checkpoint_metadata'].items():assert [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==v,p
 print('RUN_START',s['run_id'],s['arm'],flush=True);raw=run(s,f);m=TimedMonitor.instances[-1]
 raw['control_latency_seconds']=dict(max_check=max(m.check_seconds,default=0),max_publish=max(m.publish_seconds,default=0));raw['control_nonblocking']=max(m.check_seconds+m.publish_seconds,default=0)<.01;raw['router_source']=ref(OUT/'vqa_aware_dynamic_router.py');save('raw_runs/'+s['run_id']+'.json',raw)
 print('RUN_FINISH',s['run_id'],raw['valid_session'],raw['errors'],flush=True);return raw

def main():
 lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);f=read(OUT/'experiment_manifest.json');attempted=[];failed=None
 try:
  for rid in f['formal_order']:
   s=next(x for x in f['slots'] if x['run_id']==rid);save('campaign_status.json',dict(status='FORMAL_EXPLORATION',run_id=rid,pid=os.getpid(),attempted=attempted));attempted.append(rid);raw=execute(s,f);o=assess(raw);save('formal_acceptance/'+rid+'.json',o)
   print('FORMAL_ACCEPT',rid,o['eligible'],o['U'],o['stability'],o['validation'],o.get('dynamic_validation'),flush=True)
   from .report import generate
   generate()
   assert o['technical_valid'],('TECHNICAL_INVALID_NO_RETRY',rid)
 except BaseException as ex:
  failed=dict(error=repr(ex),traceback=traceback.format_exc());save('campaign_failure.json',failed);print('STOPPED',repr(ex),flush=True)
 finally:
  save('campaign_status.json',dict(status='STOPPED_INCONCLUSIVE' if failed else 'COMPLETED',attempted=attempted,failed=failed,formal_max_runs=3,no_retries=True,independent_replication=False))
  from .report import generate
  generate()
if __name__=='__main__':main()
