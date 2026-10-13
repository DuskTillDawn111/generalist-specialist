"""Complete one frozen slot that failed before workers or a measurement existed."""
import os,time,shutil,fcntl
from .common import *
from . import gpu as native,report_v2
from .telemetry_v2 import SafeMonitorV2
from replica_scaling.telemetry import snapshot

def main():
 lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 runid='C2_split05';f=read(OUT/'frozen_gpu_manifest_v2.json');slot=next(s for s in f['slots'] if s['run_id']==runid)
 prior=read(OUT/'campaign_completed.json');failure=read(OUT/'v2_prelaunch_failure.json')
 assert failure==dict(run_id=runid,error="RuntimeError('THERMAL_LAUNCH_FAILURE')")
 assert prior['stop']==failure and not (OUT/'raw_runs'/(runid+'.json')).exists()
 assert not (OUT/'request_journals'/(runid+'.jsonl')).exists() and not (OUT/'thermal_prelaunch'/(runid+'.json')).exists()
 assert 'MEASUREMENT_START '+runid not in (OUT/'gpu_campaign_v2_resumed.log').read_text()
 previous=[read(p) for p in (OUT/'raw_runs').glob('*.json')]
 assert len(previous)==5 and all(r['status']=='COMPLETED' and r['phases']['measurement']['window']['duration_seconds']==600 for r in previous)
 for r in f['implementation']+f['reused_implementation']+f['source_refs']+f['implementation_v2']:
  assert ref(r['path'])['sha256']==r['sha256'],r['path']
 archive=OUT/'before_unmeasured_slot_completion';archive.mkdir(exist_ok=False)
 for name in ['campaign_completed.json','module30_gate.json','module30_summary.md','positive_split_mechanism.md','optimizer_integration_plan.md','v2_supervisor_process.json','run_observations.json','v2_prelaunch_failure.json','run_attempts/C2_split05.json']:
  p=OUT/name;dest=archive/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 save('v2_supervisor_process.json',dict(pid=os.getpid(),status='WAITING_ORIGINAL_TEMPERATURE_FOR_UNMEASURED_SLOT'))
 # Extra idle preparation only; original <=38 C threshold, 300s prelaunch check,
 # <=40 C resident check, priming/warmup/drain and all measured conditions stay fixed.
 deadline=time.monotonic()+900;samples=[];ready=0
 while time.monotonic()<deadline:
  s=snapshot()
  from module26.gpu import thermal
  s.update(thermal());samples.append(s)
  assert not s['processes'] and s['used_mib']<=128
  assert all(s[k]==f['GPU'][k] for k in ['uuid','name','driver','total_mib'])
  ready=ready+1 if s['temperature_gpu_c']<=38 else 0
  if ready>=3:break
  time.sleep(2)
 cold=ready>=3
 recovery=dict(prelaunch_only_failure=ref(archive/'v2_prelaunch_failure.json'),original_manifest_v2=ref(OUT/'frozen_gpu_manifest_v2.json'),
  recovery_code=ref(ROOT/'module30/complete_unmeasured_slot.py'),frozen_slot=slot,prior_completed_formal_windows=5,
  original_formal_maximum=6,measured_window_retries=0,model_workers_existed_in_failed_setup=False,
  no_input_seed_route_QPS_memory_quality_utility_change=True,temperature_thresholds_and_original_checks_unchanged=True,
  additional_idle_preparation_max_seconds=900,cold_ready=cold,idle_samples=samples,
  methodological_deviation='Original controller stopped on prelaunch timeout. Complete the previously unmeasured frozen sixth window only after the unchanged cold criterion passes; never repeat a measured window.')
 save('frozen_gpu_manifest_unmeasured_slot_completion.json',recovery)
 if not cold:
  save('unmeasured_slot_completion_result.json',dict(started_measurement=False,reason='Original cold condition never reached; no threshold relaxation'))
  save('v2_supervisor_process.json',dict(pid=os.getpid(),status='COMPLETE_NO_NEXT_MODULE'))
  return
 save('v2_supervisor_process.json',dict(pid=os.getpid(),status='RUNNING_PREVIOUSLY_UNUSED_GPU_SLOTS'))
 native.SafeMonitor=SafeMonitorV2;print('UNMEASURED_FROZEN_SLOT_START',runid,flush=True)
 try:raw=native.run(slot,f)
 except Exception as exc:
  save('unmeasured_slot_completion_result.json',dict(started_measurement=False,error=repr(exc),no_more_recovery=True))
  save('v2_supervisor_process.json',dict(pid=os.getpid(),status='COMPLETE_NO_NEXT_MODULE'));return
 source=OUT/'raw_runs'/(runid+'.json');dest=OUT/'v2_native_raw_before_provenance'/source.name;shutil.copy2(source,dest)
 raw.update(telemetry_version='v2_nonblocking_control',frozen_manifest_v2=ref(OUT/'frozen_gpu_manifest_v2.json'),native_raw_before_provenance=ref(dest),
  unmeasured_slot_completion=ref(OUT/'frozen_gpu_manifest_unmeasured_slot_completion.json'))
 save('raw_runs/'+source.name,raw);o=report_v2.observation(raw)
 print('UNMEASURED_FROZEN_SLOT_FINISH',runid,'eligible',o['eligible'],'U',o['U'],'errors',raw['errors'],flush=True)
 save('unmeasured_slot_completion_result.json',dict(started_measurement=bool(raw.get('phases',{}).get('measurement')),raw=ref(source),
  full_measurement=bool(raw.get('phases',{}).get('measurement')),eligible=o['eligible'],U=o['U'],errors=raw['errors'],no_more_recovery=True))
 completed=dict(prior,stop=None if raw['valid_session'] else dict(run_id=runid,reason='UNMEASURED_SLOT_SESSION_FAILED',errors=raw['errors']),
  previous_prelaunch_stop=ref(archive/'campaign_completed.json'),same_frozen_slot_setup_attempts=2,actual_maximum_formal_windows=6,
  actual_formal_windows=6 if raw.get('phases',{}).get('measurement') else 5,measured_window_retries=0,
  methodological_deviation=ref(OUT/'frozen_gpu_manifest_unmeasured_slot_completion.json'))
 save('campaign_completed.json',completed);report_v2.final()
 gate=read(OUT/'module30_gate.json');gate.update(unmeasured_slot_completion=ref(OUT/'frozen_gpu_manifest_unmeasured_slot_completion.json'),
  formal_window_retries=0,failed_prelaunch_attempts=1,measured_window_cap_respected=True);save('module30_gate.json',gate)
 save('v2_supervisor_process.json',dict(pid=os.getpid(),status='COMPLETE_NO_NEXT_MODULE'))
if __name__=='__main__':main()
