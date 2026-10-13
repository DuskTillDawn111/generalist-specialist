"""Wait clean v1 release, repeat CPU infrastructure soak, then four unused GPU slots."""
import os,time,subprocess
from .common import *
def main():
 assert not (OUT/'telemetry_stress_result_v2.json').exists()
 deadline=time.monotonic()+1000
 while not (OUT/'campaign_v1_stopped.json').exists():
  if time.monotonic()>deadline:raise RuntimeError('V1_STOP_NOT_CONFIRMED_NO_NEW_GPU')
  time.sleep(.5)
 save('v2_supervisor_process.json',dict(pid=os.getpid(),status='CPU_ONLY_SLOW_SENSOR_LONG_STRESS'))
 with (OUT/'telemetry_stress_v2.log').open('x') as stream:
  result=subprocess.run([os.sys.executable,'-m','module30.stress_v2','--seconds','2100'],stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 assert result.returncode==0,'Infrastructure v2 failed; no new GPU authorized by gate'
 save('v2_supervisor_process.json',dict(pid=os.getpid(),status='RUNNING_PREVIOUSLY_UNUSED_GPU_SLOTS'))
 from .gpu_v2 import main as campaign
 campaign()
 save('v2_supervisor_process.json',dict(pid=os.getpid(),status='COMPLETE_NO_NEXT_MODULE'))
if __name__=='__main__':main()
