"""Post-hoc engineering accounting of a startup failure; no rerun/source edits.

Frozen executable assess() assumed a journal existed; retain that secondary
FileNotFoundError separately and report the original CUDA startup failure.
"""
from m32e.isolation import install,VIOLATIONS
install()
from m32e.common import *
raw=read(OUT/'raw_runs/E0_DYNAMIC.json');f=read(OUT/'frozen_validation_manifest.json')
assert raw['status']=='FAILED' and not raw.get('phases') and not raw['all_runtime_records']
assert len(list((OUT/'run_attempts').glob('*.json')))==1
probes=raw['control_probes'];samples=raw['telemetry']['samples']
csvsave('gpu_control_probes.csv',probes,fields=['phase','perf_ns','wall_ms','thread_cpu_ms','state_wait_ms','state_hold_ms','error','thread'])
csvsave('gpu_sensor_samples.csv',samples)
save('gpu_preflight_acceptance.json',dict(passed=False,checks=dict(startup=False,measurement_complete=False,router_online_verified=False,control_online_verified=False,
 requests_measured=False,quality_runtime_verified=False,residency_verified=False,six_workers_ready=False,
 no_unexpected_python_write_attempts=not raw['unexpected_write_violations'],collector_exited=raw['telemetry']['collector_exited'],GPU_clean_release=raw['gpu_release']['verified']),
 primary_failure=raw['errors'],secondary_report_failure=read(OUT/'gpu_preflight_failure.json'),
 details=dict(offered=None,completed=None,G_ASR=None,S_ASR=None,router=None,stability=None),
 control=dict(check_samples=len(probes),publish_samples=len(raw['control_latency_seconds']['phase_publish']),
  max_check=None,max_publish=None,threshold_seconds=.01,status='NOT_MEASURED_STARTUP_ABORT'),
 planned_measurement_offered=f['slot']['frozen_offered_counts'],planned_trace_preserved=f['slot']['trajectories'],
 runtime_records_retained=0,actual_worker_started=[dict(model=k,**v) for k,v in raw.get('shutdown',{}).items()],
 U=None,utility_computed=False,engineering_only=True,no_GPU_retry=True,no_threshold_changes=True))
save('GPU_startup_failure_audit.json',dict(primary='CUDA runtime cudaGetDeviceCount Error304 / OS call failed while first PARSeq worker model.to(cuda:0)',
 frozen_source_changed=False,new_isolation_GPU_compatibility_verified=False,unique_root_cause_identified=False,
 uncertainty='Failure occurred under new Landlock/seccomp/audit confinement. Native CUDA syscalls and driver initialization details are not traced in this run. No isolation-on/off GPU counterfactual was executed; cannot attribute uniquely to a specific denied syscall, scheduler, model, driver or device access.',
 no_inference_requests=True,no_control_threshold_breach_observed=True,control_probes_executed=0,online_router_acceptance_undetermined=True,
 stopped_after_one_frozen_attempt=True,scope='Engineering startup failure, not a routing/utility negative result',
 shutdown=raw['shutdown'],release=raw['gpu_release'],source=ref(OUT/'raw_runs/E0_DYNAMIC.json')))
assert not VIOLATIONS
print('GPU_FAILURE_ACCOUNTED_NO_RETRY',flush=True)
