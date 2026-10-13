from common_replication import *
import gpu_replication as gpu
BaseMonitor=gpu.DurableMonitor

class SafeMonitor(BaseMonitor):
    def __init__(self,runtime,slot,budget):
        super().__init__(runtime,slot,budget)
        self.safety=json.loads((OUT/'frozen_replication_manifest.json').read_text())['safety']
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

