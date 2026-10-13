"""Stop v1 supervisor only after current raw verifies worker exit and GPU release."""
import os,signal,time
from .common import *
def main():
 process=read(OUT/'campaign_process.json');pid=process['pid'];target=OUT/'raw_runs/C1_s_only.json';deadline=time.monotonic()+900
 while not target.exists():
  if time.monotonic()>deadline:raise RuntimeError('Current-window stop deadline exceeded')
  time.sleep(.02)
 raw=read(target);assert raw['gpu_release']['verified']
 assert all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in raw['shutdown'].values())
 try:os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:pass
 save('campaign_v1_stopped.json',dict(reason='NEW_TELEMETRY_CHECK_LOCK_BLOCKS_REQUEST_DISPATCH',pid=pid,
  completed_raw=[ref(p) for p in sorted((OUT/'raw_runs').glob('*.json'))],
  GPU_release_verified_before_supervisor_stop=True,no_measurement_retry=True,
  used_formal_slots=['C1_split05','C1_s_only'],remaining_frozen_slots=['C1_split10','C2_split10','C2_s_only','C2_split05'],
  detection=ref(OUT/'telemetry_observer_lock_reproduction.json'),maximum_total_formal_slots=6))
 print('V1_STOPPED_AFTER_CLEAN_GPU_RELEASE',flush=True)
if __name__=='__main__':main()
