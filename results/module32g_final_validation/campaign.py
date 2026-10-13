"""Exactly six frozen independent GPU runs; no utility-driven control flow."""
from m32g.isolation import install
install('supervisor')
from m32g.common import *
from m32g.harness import run
from m32g.acceptance import assess
import fcntl,subprocess,traceback

def main():
 lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not list((OUT/'run_attempts').glob('*.json')),'Single-use campaign; no retries'
 f=read(OUT/'frozen_validation_manifest.json');attempted=[];failure=None;accepted={}
 try:
  for rid in f['formal_order']:
   for x in f['sources']+f['frozen_inputs']:assert ref(x['path'])['sha256']==x['sha256'],x['path']
   for p,v in f['checkpoint_metadata'].items():assert [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==v,p
   env={}
   for py,prior in f['environment'].items():
    actual=json.loads(subprocess.check_output([py,'-c','import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'],text=True));assert actual==prior,py;env[py]=actual
   save(f'environment_audits/{rid}_before.json',dict(packages=env,checkpoint_metadata_unchanged=True,frozen_sources_inputs_unchanged=True))
   slot=next(s for s in f['slots'] if s['run_id']==rid);attempted.append(rid)
   save('campaign_status.json',dict(status='RUNNING',run_id=rid,attempted=attempted,pid=os.getpid(),U_not_yet_computed=True))
   print('RUN_START',rid,slot['seed'],2400,flush=True)
   raw=run(slot,f);a=assess(raw,f);save(f'formal_acceptance/{rid}.json',a);accepted[rid]=a
   print('RUN_FINISH',rid,a['technical_valid'],a['passed'],a['details'],raw['errors'],flush=True)
   save('completed_run_status.json',dict(completed=list(accepted),summary={k:dict(technical_valid=v['technical_valid'],eligible=v['eligible'],details=v['details']) for k,v in accepted.items()},U_not_yet_computed=True))
   assert a['technical_valid'],('TECHNICAL_FAILURE_STOP_NO_RETRY',rid,[k for k,v in a['checks'].items() if not v])
   sid=f"P{slot['pair']}_S";did=f"P{slot['pair']}_D"
   if sid in accepted and did in accepted:
    raws=[read(OUT/'raw_runs'/f'{r}.json') for r in [sid,did]];temps=[];clocks=[];post=[]
    for r in raws:
     w=r['phases']['measurement']['window'];ss=[s for s in r['telemetry']['samples'] if w['start_time_ns']<=s['time_ns']<=w['end_time_ns']]
     temps.append(float(np.mean([s['temperature_gpu_c'] for s in ss])));clocks.append(float(np.mean([s['sm_clock_mhz'] for s in ss])));post.append(float(np.mean([s['temperature_gpu_c'] for s in r['postwarm_idle']])))
    c=f['environment_control'];dif=dict(mean_temperature_difference_c=abs(temps[1]-temps[0]),mean_sm_clock_relative_difference=abs(clocks[1]/clocks[0]-1),postwarm_temperature_difference_c=abs(post[1]-post[0]))
    ok=dif['mean_temperature_difference_c']<=c['measurement_mean_temperature_difference_max_c'] and dif['mean_sm_clock_relative_difference']<=c['measurement_mean_sm_clock_relative_difference_max'] and dif['postwarm_temperature_difference_c']<=c['postwarm_pair_temperature_difference_max_c']
    save(f'environment_audits/P{slot["pair"]}_paired.json',dict(passed=ok,**dif,original_limits=c))
    assert ok,('PAIRED_ENVIRONMENT_INVALID_NO_RETRY',slot['pair'],dif)
 except BaseException as exc:
  failure=dict(error=repr(exc),traceback=traceback.format_exc());save('campaign_failure.json',failure);print('STOPPED',repr(exc),flush=True)
 finally:
  save('campaign_completed.json',dict(status='STOPPED_INVALID' if failure else 'COMPLETED',attempted=attempted,failure=failure,skipped=[x for x in f['formal_order'] if x not in attempted],no_retries=True,maximum_formal_runs=6,U_not_yet_computed=True))
  save('campaign_status.json',dict(status='STOPPED_INVALID' if failure else 'COMPLETED',attempted=attempted,failure=failure,pid=os.getpid(),U_not_yet_computed=True))
if __name__=='__main__':main()
