"""Single short GPU engineering acceptance; never imports/calls utility analysis."""
import sys,fcntl,traceback
from m32e.isolation import install,VIOLATIONS
install()
from m32e.common import *
from m32e.harness import run
from vqa_aware_dynamic_router import VQAAwareRouter
def assess(raw,f):
 ph=raw.get('phases',{}).get('measurement');checks=dict(session=raw['valid_session'],control_lt10ms=raw['control_nonblocking'],no_unexpected_writes=not raw['unexpected_write_violations'])
 journal=OUT/'request_journals'/f"{raw['slot']['run_id']}.jsonl";term={};dups=[]
 for line in journal.read_text().splitlines():
  row=json.loads(line)
  if row['kind']=='REQUEST_TERMINAL':
   k=(row['phase'],row['record']['request_id'])
   if k in term:dups.append(k)
   term[k]=row['record']
 cohorts=[raw.get('conditioning'),raw.get('warmup')]+list(raw['phases'].values())
 if raw.get('aborted_phase'):cohorts.append(raw['aborted_phase'])
 rr=[r for phx in cohorts if phx for r in phx['request_records']];missing=[];mismatch=[]
 for r in rr:
  k=(r['phase'],r['request_id']);j=term.get(k)
  if j is None:missing.append(k)
  elif any(j.get(t)!=r.get(t) for t in ['status','task','model','actual_submit_time_ns','completion_time_ns','worker_latency_ms','worker_queue_wait_ms','worker_response']):mismatch.append(k)
 checks['journal_exact']=not missing and not mismatch and not dups
 telemetry=raw['telemetry']['samples'];checks['sensors_complete']=bool(telemetry) and all(all(s.get(k) is not None and np.isfinite(s[k]) for k in ['time_ns','unix_ns','owned_used_mib','gpu_util_percent','worker_CPU_core_percent','host_RSS_bytes','temperature_gpu_c','sm_clock_mhz','memory_clock_mhz','power_limit_w','power_draw_w']) for s in telemetry)
 checks['single_replica_pool']=len(raw.get('worker_identities',{}))==6
 guards=OUT/'work/control_guards';checks['workers_confined']=all((guards/f"{x['pid']}.json").exists() for x in raw.get('worker_identities',{}).values()) and checks['single_replica_pool']
 checks['workers_no_write_attempts']=all((guards/f"{x['pid']}_exit.json").exists() and not read(guards/f"{x['pid']}_exit.json")['unexpected_writes'] for x in raw.get('worker_identities',{}).values()) and checks['single_replica_pool']
 peak=max((x['owned_used_mib'] for x in telemetry),default=float('inf'));idle=[np.mean([x['owned_used_mib'] for x in raw.get(k,[])]) for k in ['postwarm_idle','postmeasurement_idle']]
 checks['memory_budget']=bool(all(np.isfinite(idle)) and f['safe_factor']*max(idle)<=f['slot']['safe_memory_budget_mib'] and peak<=f['slot']['safe_memory_budget_mib'])
 if ph:
  rs=ph['request_records'];ds=ph['routing_decisions'];ev=read(ph['trajectory']['path'])['events'];asr=[r for r in rs if r['task']=='asr']
  checks['trace_matched']=len(rs)==len(ev) and all(r['task']==e['task'] and r['sample_id']==e['item']['sample'].get('id') and r['pool_id']==e['item']['pool_id'] and r['frozen_offset_seconds']==e['offset_seconds'] for r,e in zip(rs,ev))
  checks['full_complete_six_tasks']=all(r['status']=='completed' for r in rs) and set(r['task'] for r in rs)==set(f['slot']['lambda_rps'])
  checks['timing_valid']=all(all(r.get('worker_response',{}).get(k) is not None for k in ['submit_ns','start_ns','end_ns']) and r['worker_latency_ms']>=0 and r['worker_queue_wait_ms']>=0 and abs(r['worker_latency_ms']-(r['worker_response']['end_ns']-r['worker_response']['start_ns'])/1e6)<1e-6 and abs(r['worker_queue_wait_ms']-(r['worker_response']['start_ns']-r['worker_response']['submit_ns'])/1e6)<1e-6 for r in rs if r['status']=='completed')
  checks['quality_admission_original']=all(r['task'] in f['plan']['models'][raw['replica_to_family'][r['model']]]['supported_tasks'] for r in rs) and not any(r.get('worker_response',{}).get('valid_closed_set_output') is False for r in rs)
  checks['route_decisions_complete']=len(ds)==len(asr) and all((r['model']==G)==d['select_G'] for r,d in zip(asr,ds))
  checks['router_P95_le1ms']=bool(ds) and np.percentile([d['routing_compute_ms'] for d in ds],95)<=1.
  checks['background_protected']=all(d['Phi_inflight']==0 and d['Phi_VQA_inflight']==0 and d['Phi_ASR_inflight']==0 for d in ds if d['select_G'])
  checks['causal']=all(not d['uses_future'] and all(x['enqueue_perf_ns']<=d['decision_perf_ns'] and (x['submit_perf_ns'] is None or x['submit_perf_ns']<=d['decision_perf_ns']) for x in d['past_state_snapshot']) for d in ds)
  router=VQAAwareRouter(read(OUT/'frozen_routing_parameters.json'),read(OUT/'audio_metadata_cache.json'),ph['window']['duration_seconds']);paths={r['request_id']:e['item']['sample']['input'] for r,e in zip(rs,ev) if r['task']=='asr'}
  failures=[]
  for d in ds:
   expected=router.decide(d['decision_perf_ns'],paths[d['request_id']],d['past_state_snapshot'])
   if any(expected[k]!=d[k] for k in expected):failures.append(d['request_id'])
  checks['same_state_policy_exact']=not failures
  checks['G_and_S_ASR_executed']=any(r['model']==G and r['status']=='completed' for r in asr) and any(r['model']==S and r['status']=='completed' for r in asr)
  checks['stability_original']=ph['stability']['stability_state_v2']!='unstable' and (ph['stability']['stability_state_v2']!='recoverable_congested' or ph['stability'].get('final_recovery_confirmed',False))
  times=[d['routing_compute_ms'] for d in ds];statsroute={f'p{p}_ms':float(np.percentile(times,p)) for p in [50,95,99]};statsroute['max_ms']=max(times)
  w=ph['window'];sensors=[s for s in telemetry if w['start_time_ns']<=s['time_ns']<=w['end_time_ns']]
  checks['GPU_operating_state']=bool(sensors) and all(s['power_limit_w']==f['GPU']['power_limit_w'] and s['hw_thermal_slowdown']=='Not Active' and s['sw_thermal_slowdown']=='Not Active' for s in sensors)
  details=dict(offered=len(rs),completed=sum(r['status']=='completed' for r in rs),G_ASR=sum(r['model']==G for r in asr),S_ASR=sum(r['model']==S for r in asr),router=statsroute,stability=ph['stability'],policy_replay_mismatches=failures)
  csvsave('gpu_router_decisions.csv',ds);csvsave('gpu_all_requests.csv',rs)
  csvsave('gpu_per_task_engineering_metrics.csv',[dict(task=t,**x) for t,x in ph['per_task_metrics'].items()])
 else:checks['full_measurement_completed']=False;details={}
 csvsave('gpu_control_probes.csv',raw['control_probes']);csvsave('gpu_sensor_samples.csv',telemetry)
 return dict(passed=all(checks.values()),checks=checks,details=details,control=raw['control_latency_seconds'],sampled_peak_mib=peak,
  journal=dict(missing=missing,mismatch=mismatch,duplicates=dups,terminal=len(term)),U=None,utility_computed=False,engineering_only=True)
def main():
 lock=(OUT/'preflight.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 f=read(OUT/'frozen_validation_manifest.json');assert not list((OUT/'run_attempts').glob('*.json')),'One GPU preflight, no retries'
 for r in f['sources']+f['frozen_inputs']:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 assert read(OUT/'history_hashes/before_GPU.json')['files']
 try:
  print('GPU_PREFLIGHT_START',flush=True);raw=run(f['slot'],f);a=assess(raw,f);save('gpu_preflight_acceptance.json',a)
  print('GPU_PREFLIGHT_FINISH',a['passed'],a['checks'],flush=True)
 except BaseException as exc:
  save('gpu_preflight_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),unexpected_writes=VIOLATIONS));raise
if __name__=='__main__':main()
