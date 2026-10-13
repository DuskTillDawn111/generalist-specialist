"""Resume only an interrupted CPU audit. No additional formal GPU slots or retries."""
import os,subprocess,time,shutil
from .common import *
from replica_scaling.telemetry import snapshot

def main():
 assert not (OUT/'telemetry_stress_result_v2.json').exists()
 assert not (OUT/'frozen_gpu_manifest_v2.json').exists()
 assert not (OUT/'campaign_completed.json').exists()
 prior=read(OUT/'v2_supervisor_process.json')
 try:os.kill(prior['pid'],0)
 except ProcessLookupError:pass
 else:raise RuntimeError('Previous supervisor still exists; do not duplicate')
 s=snapshot();f=read(OUT/'frozen_gpu_manifest.json')
 assert not s['processes'] and all(s[k]==f['GPU'][k] for k in ['uuid','name','driver','total_mib'])
 evidence=OUT/'interrupted_cpu_attempt';evidence.mkdir(exist_ok=False)
 shutil.copy2(OUT/'v2_supervisor_process.json',evidence/'v2_supervisor_process.json')
 shutil.copy2(OUT/'telemetry_stress_v2.log',evidence/'telemetry_stress_v2.log')
 for p in (OUT/'request_journals').glob('stress_v2*.jsonl'):shutil.move(p,evidence/p.name)
 save('cpu_pressure_interruption.json',dict(previous_process=prior,previous_log=ref(evidence/'telemetry_stress_v2.log'),
  previous_journals=[ref(p) for p in evidence.glob('*.jsonl')],termination_cause='UNKNOWN_PROCESS_ABSENT_NO_PASS_RESULT',
  last_reported_elapsed_seconds=1920,not_accepted_as_pass=True,GPU_slots_used=2,GPU_slots_remaining=4,
  no_formal_retry=True,unchanged_GPU=s,CPU_resume_code=ref(ROOT/'module30/resume_after_cpu_interruption.py')))
 save('v2_supervisor_process.json',dict(pid=os.getpid(),status='CPU_ONLY_RESTART_AFTER_INTERRUPTION'))
 with (OUT/'telemetry_stress_v2_resumed.log').open('x') as stream:
  result=subprocess.run([os.sys.executable,'-m','module30.stress_v2','--seconds','2100'],stdout=stream,stderr=subprocess.STDOUT,
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 assert result.returncode==0,'Incomplete CPU infrastructure audit cannot authorize GPU'
 save('v2_supervisor_process.json',dict(pid=os.getpid(),status='RUNNING_PREVIOUSLY_UNUSED_GPU_SLOTS'))
 from .gpu_v2 import main as campaign
 campaign()
 save('v2_supervisor_process.json',dict(pid=os.getpid(),status='COMPLETE_NO_NEXT_MODULE'))
if __name__=='__main__':main()
