"""Six-attempt bounded 28R campaign, reusing unchanged Module28 GPU/router code."""
from audit_followup import *
import time, traceback
sys.path.insert(0,str(OLD))
import audit as original_audit
import gpu as original_gpu
import report as original_report
original_audit.OUT=OUT
original_gpu.OUT=OUT
original_report.OUT=OUT
from report_followup import write_reports

BaseMonitor=original_gpu.DurableMonitor

class SafeMonitor(BaseMonitor):
    def __init__(self,runtime,slot,budget):
        super().__init__(runtime,slot,budget)
        self.safety=json.loads((OUT/'frozen_candidate_manifest.json').read_text())['safety']
        self.safety_failure=None

    def sample(self):
        value=super().sample()
        limits=self.safety
        reason=None
        with self.runtime.lock:
            pending=[r for r in self.runtime.pending.values() if r['status'] in ['queued','submitted']]
            counts=collections.Counter(r['model'] for r in pending)
            oldest=max(((time.perf_counter_ns()-r['enqueued_time_ns'])/1e9 for r in pending),default=0)
        if len(pending)>limits['max_host_outstanding_total']:reason='SAFETY_TOTAL_BACKLOG'
        elif max(counts.values(),default=0)>limits['max_host_outstanding_per_worker']:reason='SAFETY_WORKER_BACKLOG'
        elif value['owned_used_mib']>limits['max_owned_gpu_mib'] or value['used_mib']>limits['max_total_gpu_mib']:reason='SAFETY_GPU_MEMORY'
        elif value['host_available_RAM_bytes']<limits['min_host_available_ram_gib']*1024**3:reason='SAFETY_HOST_MEMORY'
        elif value['temperature_gpu_c']>limits['max_gpu_temperature_c']:reason='SAFETY_GPU_TEMPERATURE'
        # Maximum oldest pending age is deliberately conservative relative to
        # no-completion watchdog; no live worker START timestamps are required.
        elif oldest>limits['max_worker_no_completion_seconds']:reason='SAFETY_OLDEST_PENDING_AGE'
        active=self.active_phase
        if active and self.phase=='measurement' and (time.perf_counter_ns()-active['planned_end_time_ns'])/1e9>limits['max_formal_overrun_seconds']:
            reason='SAFETY_FORMAL_OVERRUN'
        value['safety_host_outstanding']=len(pending);value['safety_oldest_pending_seconds']=oldest
        if reason and not self.safety_failure:
            self.safety_failure=dict(reason=reason,at_unix_ns=time.time_ns(),outstanding=len(pending),per_worker=dict(counts),oldest_seconds=oldest)
            save('safety_stop_'+self.slot['run_id']+'.json',self.safety_failure)
        return value

    def check(self):
        super().check()
        if self.errors:raise RuntimeError('TELEMETRY_ERROR: '+repr(self.errors))
        if self.safety_failure:raise RuntimeError(self.safety_failure['reason'])
        active=self.active_phase
        if active and self.phase in ['measurement','warmup']:
            now=time.perf_counter_ns()
            rows=active['request_records']
            next_planned=next((r for r in rows if r['status']=='planned'),None)
            if next_planned and (now-next_planned['scheduled_time_ns'])/1e9>self.safety['max_scheduled_dispatch_lag_seconds']:
                raise RuntimeError('SAFETY_DISPATCH_LAG')
        for name,w in self.runtime.workers.items():
            if name in self.runtime.ready and w.process.poll() is not None:
                raise RuntimeError('WORKER_EXITED_DURING_RUN: '+name)

def main():
    f=json.loads((OUT/'frozen_candidate_manifest.json').read_text())
    assert f['module']=='28R' and f['status']=='FROZEN_BEFORE_GPU'
    anchor=json.loads((OUT/'gpu_run_manifest.json').read_text())
    for r in f['implementation']+f['reused_implementation']+[f['catalog'],f['utility_definition'],f['utility_reference'],f['quality_policy'],anchor['workloads'],anchor['protocol']]:
        assert ref(r['path'])['sha256']==r['sha256'],r['path']
    original_gpu.DurableMonitor=SafeMonitor
    ledger=[];raws=[];fatal=None
    assert not (OUT/'attempt_ledger.json').exists(), 'No campaign retry'
    for slot in f['exploration_slots']:
        assert len(ledger)<6 and not (OUT/'raw_runs'/(slot['run_id']+'.json')).exists()
        ledger.append(dict(run_id=slot['run_id'],scenario=slot['scenario'],arm=slot['arm'],order=slot['run_order'],status='STARTING',start_unix_ns=time.time_ns()))
        save('attempt_ledger.json',ledger)
        print('START',slot['run_id'],flush=True)
        try:
            raw=original_gpu.run(slot,f);raws.append(raw)
            v=original_report.evaluate(raw,f)
            ledger[-1].update(status=raw['status'],end_unix_ns=time.time_ns(),eligible=v['eligible'],U=v['U'],worker_states=v['worker_states'])
            save('attempt_ledger.json',ledger)
            print('FINISH',slot['run_id'],'eligible',v['eligible'],'U',v['U'],'worker_states',v['worker_states'],'errors',raw['errors'],flush=True)
            # A failed stability gate alone does not prevent safe capacity controls.
            safe=(raw['valid_session'] and v['full_cohort'] and v['capability'] and v['measured_counts'] and
                  v['semantic_valid'] and v['memory_ok'] and v['recovery_confirmed'] and not raw['errors'])
            if not safe:
                fatal=dict(run_id=slot['run_id'],reason='INFRASTRUCTURE_SAFETY_OR_ORIGINAL_HARD_CONSTRAINT',errors=raw['errors'],evaluation=v)
        except (Exception,KeyboardInterrupt) as exc:
            fatal=dict(run_id=slot['run_id'],reason=repr(exc),traceback=traceback.format_exc())
            ledger[-1].update(status='EXCEPTION',error=fatal,end_unix_ns=time.time_ns());save('attempt_ledger.json',ledger)
            print('EXCEPTION',fatal,flush=True)
        write_reports(raws,f,fatal,final=False)
        if fatal:break
    gate=write_reports(raws,f,fatal,final=True)
    save('campaign_completed.json',dict(at_unix_ns=time.time_ns(),completed_and_stopped=True,
        executed=[r['slot']['run_id'] for r in raws],attempts=len(ledger),formal_seconds=sum(600 for r in raws if r.get('phases',{}).get('measurement')),
        fatal=fatal,gates=gate,no_automatic_next_module=True,no_independent_long_window=True))
    print('FINAL_GATES',gate,flush=True)

if __name__=='__main__':main()
