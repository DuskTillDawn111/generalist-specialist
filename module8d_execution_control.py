"""Continue the same frozen campaign after a GPU-idle coordinator boundary."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from module8d_real_baselines.common import ROOT,OUT,F,read,ref,save_json,verify_inputs


def main():
    freeze={'controller_source':ref(Path(__file__).resolve()),
        'confirmation_compatibility':ref(OUT/'confirmation_donor_compatibility_freeze.json'),
        'original_frozen_campaign':ref(OUT/'campaign_manifest.json'),
        'same_campaign_job_order_budgets_and_inputs':True,'only_resume_completed_conditions':True,
        'no_active_measurement_may_be_interrupted':True}
    path=OUT/'execution_control_freeze.json'
    if path.exists():assert read(path)==freeze
    else:save_json(path,freeze)
    paused=OUT/'coordinator_pause_record.json'
    while not paused.exists():time.sleep(10)
    row=read(paused);pid=row['pid'];verify_inputs()
    assert row['timed_measurements_interrupted']==0 and row['gpu_idle']['used_mib']<=128 and not row['gpu_idle']['processes']
    latest=read(OUT/'new_measurement_manifest.json')['runs'][-1]
    assert latest['status']!='started' and latest['valid_timed_execution']
    assert Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[1:4]==[b'-u',b'-m',b'module8d_real_baselines.campaign']
    assert not list(OUT.glob('b4_seed_*/ */search_complete.json'.replace(' ','')))
    # Terminate only the stopped coordinator, after all its workers and windows
    # completed. No measurement child is killed or repeated.
    os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (OUT/'campaign.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,'-u','-m','module8d_confirmation_compat.campaign'],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        record={'status':'running','old_coordinator_pid':pid,'new_coordinator_pid':proc.pid,
            'same_campaign_manifest':ref(OUT/'campaign_manifest.json'),'compatibility_freeze':ref(OUT/'confirmation_donor_compatibility_freeze.json'),
            'completed_condition_boundary':row['condition'],'physical_attempts_at_restart':len(read(OUT/'new_measurement_manifest.json')['runs']),
            'timed_runs_interrupted_repeated_or_discarded':0,'start_unix':time.time()}
        save_json(OUT/'coordinator_restart_record.json',record)
        print(json.dumps({'event':'same_campaign_coordinator_resumed','pid':proc.pid,'no_timed_interruption':True}),flush=True)
        code=proc.wait()
        record.update(status='complete' if code==0 else 'blocked',returncode=code,end_unix=time.time())
        save_json(OUT/'coordinator_restart_record.json',record)
        print(json.dumps({'event':'continued_campaign_exit','returncode':code}),flush=True)
        if code:raise SystemExit(code)


if __name__=='__main__':main()
