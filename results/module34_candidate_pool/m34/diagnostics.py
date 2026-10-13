from .common import *
import numpy as np
from module20.optimizer import predict_scenarios

def main():
 fm=read(OUT/'frozen_problem_manifest.json');sp=read(fm['space']['path']);refs=read(fm['utility_reference']['path'])['latency_reference_ms'];weights=read(fm['utility_definition']['path'])['weights'];obs=read(OUT/'gpu_observations.json') if (OUT/'gpu_observations.json').exists() else [];rows=[];env=[]
 for o in obs:
  raw=read(o['raw']['path']);s=raw['slot'];d=read(sp['pools'][s['pool']]['path']);p=predict_scenarios(d,s['deployment'],s['routing'],s['lambda_rps'],sp['budget_mib'],refs,weights);ph=raw.get('phases',{}).get('measurement')
  if not ph:continue
  for task in s['lambda_rps']:
   records=[r for r in ph['request_records'] if r['task']==task and r['status']=='completed'];model=next(iter(s['routing'][task]));means={}
   for key in ['e2e_latency_ms','worker_latency_ms','worker_queue_wait_ms','runtime_latency_ms','schedule_delay_ms']:
    values=[r[key] for r in records if r.get(key) is not None];means[key]=float(np.mean(values)) if values else None
   means['runtime_or_transport_residual_ms']=means['runtime_latency_ms']-means['worker_latency_ms']-means['worker_queue_wait_ms'] if all(means[k] is not None for k in ['runtime_latency_ms','worker_latency_ms','worker_queue_wait_ms']) else None
   service=d['models'][model]['service_time_ms'][task];rows.append(dict(run_id=s['run_id'],group=s['group'],stage=s['stage'],arm=s['arm'],task=task,model=model,offered=sum(r['task']==task for r in ph['request_records']),completed=len(records),nominal_model_service_ms=service,forecast_mean_ms=p['per_task'][task]['mean_ms'],forecast_p95_ms=p['per_task'][task]['p95_ms'],observed_p95_ms=float(np.percentile([r['e2e_latency_ms'] for r in records],95)) if records else None,
    mean_components_ms=means,service_ratio=means['worker_latency_ms']/service if means['worker_latency_ms'] is not None else None,observed_U=o['U'],forecast_U=p['U'],eligible=o['eligible']))
  env.append(dict(run_id=s['run_id'],stage=s['stage'],baseline_GPU=raw['baseline_GPU'],environment=o['environment'],sampled_peak_mib=o['components'].get('sampled_peak_mib'),safe_memory_mib=o['components'].get('safe_memory_mib'),elapsed_seconds=raw['elapsed_seconds'],native_shutdown=raw['shutdown'],gpu_release_verified=raw['gpu_release']['verified']))
 csvsave('per_task_prediction_diagnostics.csv',rows);save('environment_and_session_diagnostics.json',dict(runs=env,profile_hardware=read(ROOT/'results/module14_heavy_replica/measurement_plan.json')['GPU'],current_hardware=read(OUT/'frozen_gpu_manifest_v2.json')['GPU'],unchanged_worker_registry=read(OUT/'frozen_gpu_manifest_v2.json')['registry_original'],differences='14F profile source physical UUID differs from33/34; name, driver, totalMiB match. Worker code/environment identities unchanged; workload co-residence/service interference can differ. Per-run temperatures/clocks and safe-memory included.'))
 print('DIAGNOSTICS',len(rows))
if __name__=='__main__':main()
