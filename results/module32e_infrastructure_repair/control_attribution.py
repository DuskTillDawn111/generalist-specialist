"""CPU-only replay of actual journal copy path; no GPU timing manipulation."""
import sys,threading,traceback,resource
from types import SimpleNamespace
from m32e.isolation import install,VIOLATIONS
install()
from m32e.common import *
from m32e.monitor import IncrementalMonitor
from router_optimized import DeploymentRuntime

raw=read(ROOT/'results/module32d_router_repair/raw_runs/E0_S.json')
hist=raw.get('control_latency_seconds',{})
records=read(ROOT/'results/module32c_dynamic_replication/raw_runs/P1_D.json')['phases']['measurement']['request_records']
rows=[];errors=[]
for mode in ['old_state_lock_copy','repaired_reference_capture']:
 for rep in range(8):
  rt=DeploymentRuntime(SimpleNamespace());m=IncrementalMonitor(rt,dict(run_id=f'CONTROL_{mode}_{rep}'),1,enforce_control=False)
  # Original actual full cohort and worker payloads, not dummy compute/sleep.
  m.set_active(dict(label='copy_replay',request_records=records))
  entered=threading.Event();finish=[]
  def copy_path():
   try:
    tick=time.perf_counter_ns()
    if mode=='old_state_lock_copy':
     with m.state_lock:
      entered.set()
      # Exact location of v1 copy: return expression evaluated while owning lock.
      snapshot=copy.deepcopy([m.cached[k] for k in m.cache_order])
    else:
     with m.state_lock:cached=m.cached;order=m.cache_order
     entered.set();snapshot=copy.deepcopy([cached[k] for k in order])
    finish.append((time.perf_counter_ns()-tick)/1e6)
    assert len(snapshot)==len(records)
   except BaseException:errors.append(traceback.format_exc())
  t=threading.Thread(target=copy_path);t.start();assert entered.wait(2);m.check();t.join(5);assert not t.is_alive()
  rows.append(dict(mode=mode,repetition=rep,records=len(records),copy_wall_ms=finish[0],**m.control_probes[-1]))
# Real stalled sensor and stalled durable publisher, outside the control lock.
# Event gates emulate blocking I/O on CPU; they never touch GPU inference.
for subsystem in ['sensor','durable_io']:
 for rep in range(8):
  rt=DeploymentRuntime(SimpleNamespace());entered=threading.Event();release=threading.Event()
  def sensor():
   entered.set();assert release.wait(2)
   return dict(time_ns=time.perf_counter_ns(),used_mib=0,owned_used_mib=0,foreign_pids=[])
  m=IncrementalMonitor(rt,dict(run_id=f'CONTROL_{subsystem}_{rep}'),1,sensor=sensor,enforce_control=False)
  def io_gate():
   with m.sample_lock:entered.set();assert release.wait(2)
  t=threading.Thread(target=m.sample if subsystem=='sensor' else io_gate);t.start();assert entered.wait(2);m.check();release.set();t.join(3)
  rows.append(dict(mode=subsystem,repetition=rep,**m.control_probes[-1]))
csvsave('control_probe_reproduction.csv',rows)
repaired=[r['wall_ms'] for r in rows if r['mode']!='old_state_lock_copy'];old=[r['wall_ms'] for r in rows if r['mode']=='old_state_lock_copy']
save('control_probe_attribution.json',dict(passed=not errors and max(repaired)<10,threshold_ms=10,threshold_unchanged=True,
 historical_control_summary=hist,historical_run=ref(ROOT/'results/module32d_router_repair/raw_runs/E0_S.json'),
 historical_unique_outlier_cause_identifiable=False,historical_limitation='Raw 32D stores max_check/max_publish only, no per-check wall stamps, lock-owner/scheduler samples. Source+reproduction identifies a credible direct blocking path, cannot uniquely attribute historical 19.36ms to a particular copy event.',
 timer='perf_counter host monotonic wall, error-state lock wait + predicates; no GPU sync; not worker service time',
 reproduced_old_max_ms=max(old),repaired_max_ms=max(repaired),reproduced_old_over_10ms=sum(x>=10 for x in old),
 wall_vs_thread_cpu_logged=True,IO_not_on_control_lock=True,unexpected_writes=VIOLATIONS,errors=errors,
 load_average=os.getloadavg(),cpu_count=os.cpu_count(),max_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
assert not errors and max(repaired)<10 and not VIOLATIONS
print('CONTROL_CPU_PASS',max(old),max(repaired),flush=True)
