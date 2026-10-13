from common_replication import *
import gpu_replication as gpu
from safety_replication import SafeMonitor
from report_replication import describe_run,write_report
from replica_scaling.telemetry import snapshot

def main():
    f=json.loads((OUT/'frozen_replication_manifest.json').read_text())
    assert f['module']=='29' and f['status']=='FROZEN_BEFORE_GPU'
    assert json.loads((OUT/'cpu_preflight.json').read_text())['passed']
    for r in f['implementation']+f['reused_implementation']+[f[k] for k in ['catalog','utility_definition','utility_reference','quality_policy','model_files','environment','development_inventory']]:
        assert ref(r['path'])['sha256']==r['sha256'],r['path']
    for slot in f['independent_slots']:
        for r in slot['trajectories'].values():assert ref(r['path'])['sha256']==r['sha256']
    assert not (OUT/'attempt_ledger.json').exists(),'No campaign rerun/retry'
    gpu.DurableMonitor=SafeMonitor
    ledger=[];fatal=None
    for slot in f['independent_slots']:
        rid=slot['run_id'];assert len(ledger)<9
        ledger.append(dict(run_id=rid,group=slot['group'],arm=slot['arm'],order=slot['run_order'],seed=slot['seed'],status='STARTING',start_unix_ns=time.time_ns(),startup_GPU=snapshot()))
        save('attempt_ledger.json',ledger);save('live_progress.json',dict(run_id=rid,phase='STARTING',elapsed_seconds=0,planned_seconds=2400))
        print('START',rid,flush=True)
        try:
            raw=gpu.run(slot,f)
            v=describe_run(raw,f)
            ledger[-1].update(status=raw['status'],end_unix_ns=time.time_ns(),eligible=v['eligible'],U=v['U'],worker_states=v['worker_states'])
            save('attempt_ledger.json',ledger)
            print('FINISH',rid,'eligible',v['eligible'],'U',v['U'],'workers',v['worker_states'],'errors',raw['errors'],flush=True)
            if not v['eligible']:
                fatal=dict(run_id=rid,reason='FROZEN_SAFETY_TECHNICAL_OR_ORIGINAL_HARD_CONSTRAINT_STOP',errors=raw['errors'],evaluation=v)
            del raw
        except (Exception,KeyboardInterrupt) as exc:
            fatal=dict(run_id=rid,reason=repr(exc),traceback=traceback.format_exc(),GPU=snapshot())
            save('technical_failure_'+rid+'.json',fatal)
            ledger[-1].update(status='EXCEPTION',end_unix_ns=time.time_ns(),error=fatal);save('attempt_ledger.json',ledger)
            print('EXCEPTION',repr(exc),flush=True)
        # CPU report exceptions preserve the GPU observation and stop; never rerun it.
        try:
            report=write_report(f,fatal,final=False)
            if any(not p['thermal_comparable'] for p in report['pairs']):
                fatal=dict(run_id=rid,reason='FROZEN_THERMAL_PAIRING_FAILURE')
        except Exception as exc:
            fatal=dict(run_id=rid,reason='CPU_REPORT_FAILURE',exception=repr(exc),traceback=traceback.format_exc())
            save('technical_failure_report.json',fatal)
        if fatal:break
    gate=write_report(f,fatal,final=True)
    save('campaign_completed.json',dict(at_unix_ns=time.time_ns(),completed_and_stopped=True,attempts=len(ledger),fatal=fatal,
        gate=gate['gate'],no_retries=True,no_next_module=True,no_optimizer_change=True))
    save('live_progress.json',dict(phase='STOPPED',gate=gate['gate'],attempts=len(ledger)))
    print('FINAL_GATE',gate['gate'],flush=True)

if __name__=='__main__':main()
