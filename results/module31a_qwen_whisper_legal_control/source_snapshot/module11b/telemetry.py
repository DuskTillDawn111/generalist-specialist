"""Read-only host/NVML sampling; no CUDA initialization or worker changes."""
import math,time
import psutil
from replica_scaling.telemetry import Monitor,snapshot,owned_sample

class CostMonitor(Monitor):
    def __init__(self,runtime,interval,capacity):
        super().__init__(runtime,interval);self.capacity=capacity;self.previous_cpu={};self.resource_failure=False
        self.first_ready_seen={};self.first_pid_seen={};self.startup_origin=time.perf_counter_ns()

    def host(self,pids):
        now=time.perf_counter_ns();processes={}
        for pid in sorted(pids):
            try:
                root=psutil.Process(pid)
                for p in [root]+root.children(recursive=True):
                    if p.pid in processes:continue
                    with p.oneshot():
                        times=p.cpu_times();cpu=times.user+times.system;rss=p.memory_info().rss
                    prev=self.previous_cpu.get(p.pid);util=100*(cpu-prev[1])/((now-prev[0])/1e9) if prev and now>prev[0] else None
                    self.previous_cpu[p.pid]=(now,cpu)
                    processes[p.pid]={'pid':p.pid,'root_worker_pid':pid,'rss_bytes':rss,'cpu_core_percent':util,'cpu_seconds':cpu,
                        'create_time_unix_seconds':p.create_time()}
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        for name in list(self.runtime.ready):
            if name not in self.first_ready_seen:self.first_ready_seen[name]=now
        for name,w in list(self.runtime.workers.items()):
            if w.process and name not in self.first_pid_seen:self.first_pid_seen[name]=now
        return {'host_sample_time_ns':now,'host_processes':list(processes.values()),
            'owned_process_tree_count':len(processes),'host_RSS_bytes':sum(p['rss_bytes'] for p in processes.values()),
            'worker_CPU_core_percent':math.fsum(p['cpu_core_percent'] for p in processes.values() if p['cpu_core_percent'] is not None),
            'controller_RSS_bytes':psutil.Process().memory_info().rss,
            'host_available_RAM_bytes':psutil.virtual_memory().available,
            'cpu_unit':'100 percent equals one fully utilized CPU core; worker trees only, controller kept separate'}

    def sample(self):
        pids={w.process.pid for w in list(self.runtime.workers.values()) if w.process is not None}
        value=owned_sample(snapshot(),pids);value.update(self.host(pids));value['phase']=self.phase
        if value['used_mib']>self.capacity or value['owned_used_mib']>self.capacity:self.resource_failure=True
        self.samples.append(value);return value

    def collect(self):
        while not self.stop.is_set():
            try:self.sample()
            except Exception as exc:self.errors.append(repr(exc))
            self.stop.wait(self.interval)

    def idle_window(self,phase,duration):
        self.phase=phase;begin=time.perf_counter_ns();time.sleep(duration)
        samples=[s for s in self.samples if s['phase']==phase and s['time_ns']>=begin]
        if not samples or any(s['foreign_pids'] or not s['each_owned_pid_has_cuda_context'] for s in samples):
            raise RuntimeError('Idle residency/process isolation failed')
        return samples

    def check(self):
        if self.resource_failure:raise RuntimeError('RUNTIME_RESOURCE_INFEASIBLE: actual GPU residency exceeds frozen physical budget')
        if any(s['foreign_pids'] for s in self.samples):raise RuntimeError('external_gpu_process_interference')
