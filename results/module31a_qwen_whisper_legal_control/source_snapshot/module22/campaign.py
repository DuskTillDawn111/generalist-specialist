"""Independent stages, fixed maximum of twelve new observations, no retries."""
from .common import *
from .gpu import run
from .report import write_outputs

def main():
 f=read(OUT/'frozen_experiment_manifest.json');assert f['status']=='FROZEN_BEFORE_GPU'
 for r in f['implementation']+f['reused_implementation']+[f['cpu_recheck'],f['protected_before']]:
  assert ref(r['path'])['sha256']==r['sha256'],r['path']
 assert read(f['cpu_recheck']['path'])['Stage_A_passed']
 assert len(f['slots'])==12
 raws=[];skipped=[];global_failure=None;stage_stops={}
 def status():
  save('campaign_status.json',{'at_utc':now(),'completed_runs':[r['slot']['run_id'] for r in raws],
       'stage_stops':stage_stops,'global_safety_failure':global_failure,'skipped_runs':skipped,
       'pending_runs':[s['run_id'] for s in f['slots'] if s['run_id'] not in {r['slot']['run_id'] for r in raws} and s['run_id'] not in {v['run_id'] for v in skipped}],
       'no_retries_or_extension':True})
 for stage in RATES:
  for slot in [s for s in f['slots'] if s['stage']==stage]:
   if global_failure or stage in stage_stops:
    skipped.append({'run_id':slot['run_id'],'reason':global_failure or stage_stops[stage]});status();continue
   path=OUT/'raw_runs'/(slot['run_id']+'.json')
   if path.exists():
    raw=read(path);assert raw['slot']==slot,'Existing observation must match exact frozen slot'
    print('RESUME_EXISTING',slot['run_id'],flush=True)
   else:
    print('START',slot['run_id'],flush=True)
    try:raw=run(slot,f['plan'])
    except Exception as exc:
     global_failure={'run_id':slot['run_id'],'reason':'Failure before retained run or during release','error':repr(exc)}
     save('prelaunch_failure.json',global_failure);skipped.append({'run_id':slot['run_id'],'reason':global_failure});status();continue
   raws.append(raw);rows=write_outputs(raws,f);row=rows[-1]
   print('FINISH',slot['run_id'],'U',row.get('U'),'system',row['system_state'],'workers',row['worker_states'],'safe_mib',row.get('observed_safe_memory_mib'),'errors',raw['errors'],flush=True)
   if raw['errors'] or not raw['valid_session'] or not row['memory_passed'] or not raw['gpu_release']['verified']:
    global_failure={'run_id':slot['run_id'],'reason':'Runtime/generator/isolation/memory/release safety failure','errors':raw['errors'],'memory_passed':row['memory_passed']}
   elif row['hard_unstable'] or row.get('censored'):
    stage_stops[stage]={'run_id':slot['run_id'],'reason':'First sustained worker/system instability or incomplete cohort; affected stage remaining pairs stopped','worker_states':row['worker_states'],'system_state':row['system_state']}
   status()
 write_outputs(raws,f);status()
 print('MODULE22_CAMPAIGN_STOP',len(raws),'new runs',stage_stops,global_failure,flush=True)

if __name__=='__main__':main()
