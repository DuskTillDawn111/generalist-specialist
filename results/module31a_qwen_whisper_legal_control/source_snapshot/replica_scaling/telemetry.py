"""NVML CLI sampling; parent never creates a CUDA context."""
import subprocess
import threading
import time

def numeric(value):
    try:
        return float(value.strip())
    except ValueError:
        return None

def snapshot():
    command = ['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw', '--format=csv,noheader,nounits']
    output = subprocess.check_output(command, text=True, timeout=10).strip().splitlines()
    if len(output) != 1:
        raise RuntimeError('Characterization requires exactly one visible GPU')
    fields = [x.strip() for x in output[0].split(',')]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True, timeout=10)
    return dict(time_ns=time.perf_counter_ns(), unix_ns=time.time_ns(), uuid=fields[0], name=fields[1], driver=fields[2],
                total_mib=numeric(fields[3]), used_mib=numeric(fields[4]), gpu_util_percent=numeric(fields[5]),
                memory_util_percent=numeric(fields[6]), power_w=numeric(fields[7]),
                processes=[dict(pid=int(parts[0]), used_mib=numeric(parts[1])) for line in processes.splitlines()
                           if (parts := line.split(',')) and len(parts) == 2])

def owned_sample(value, pids):
    value = dict(value)
    value['owned_pids'] = sorted(pids)
    value['owned_used_mib'] = sum(p['used_mib'] for p in value['processes'] if p['pid'] in pids and p['used_mib'] is not None)
    value['foreign_pids'] = [p['pid'] for p in value['processes'] if p['pid'] not in pids]
    value['each_owned_pid_has_cuda_context'] = set(pids) <= {p['pid'] for p in value['processes']}
    return value

class Monitor:
    def __init__(self, runtime, interval):
        self.runtime, self.interval = runtime, interval
        self.samples, self.errors = [], []
        self.stop = threading.Event()
        self.phase = 'load'
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        while not self.stop.is_set():
            try:
                current = snapshot()
                pids = {w.process.pid for w in list(self.runtime.workers.values()) if w.process is not None}
                value = owned_sample(current, pids)
                value['phase'] = self.phase
                self.samples.append(value)
            except Exception as exc:
                self.errors.append(str(exc))
            self.stop.wait(self.interval)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=12)

    def idle(self, phase):
        self.phase = phase
        result = []
        for _ in range(3):
            pids = {w.process.pid for w in self.runtime.workers.values()}
            value = owned_sample(snapshot(), pids)
            value['phase'] = phase
            self.samples.append(value)
            result.append(value)
            time.sleep(self.interval)
        if any(v['foreign_pids'] or not v['each_owned_pid_has_cuda_context'] for v in result):
            raise RuntimeError('Idle residency/process-isolation gate failed')
        return result

def release(baseline, owned_pids, timeout=15):
    deadline = time.monotonic() + timeout
    values = []
    while True:
        value = snapshot()
        values.append(value)
        no_owned = not ({p['pid'] for p in value['processes']} & set(owned_pids))
        no_foreign = not value['processes']
        memory_released = value['used_mib'] <= baseline['used_mib'] + 64
        if no_owned and no_foreign and memory_released:
            return dict(verified=True, samples=values, tolerance_mib=64)
        if time.monotonic() >= deadline:
            return dict(verified=False, samples=values, tolerance_mib=64)
        time.sleep(.5)
