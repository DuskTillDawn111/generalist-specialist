"""Stage A CPU-only reconstruction, unchanged timestamps and stability rules."""
from collections import Counter,defaultdict
from types import SimpleNamespace
import numpy as np
from performance_model.local_calibration.stability_v2 import timeline,DEFAULT_THRESHOLDS as T
from .common import *
def stats(values):
 xs=list(values)
 return {'n':len(xs),'mean':statistics.mean(xs) if xs else None,'P95':float(np.percentile(xs,95)) if xs else None,'max':max(xs) if xs else None}
def slopes(bins):
 x=np.array([b['offset_seconds'] for b in bins]);y=np.array([b['mean_backlog_requests'] for b in bins])
 return {'slope_rps':float(np.dot(x-x.mean(),y-y.mean())/np.sum((x-x.mean())**2)),'growth_requests':float(y[-1]-y[0]),'increasing_fraction':float(np.mean(np.diff(y)>0))}
def first_growth(bins):
 n=math.ceil(len(bins)*T['tail_fraction'])
 for i in range(n,len(bins)+1):
  seg=bins[i-n:i];s=slopes(seg)
  if s['slope_rps']>=T['tail_slope_min_requests_per_second'] and s['growth_requests']>=T['tail_growth_min_requests'] and s['increasing_fraction']>=T['tail_increasing_fraction'] and seg[-1]['end_backlog_requests']>=T['tail_end_min_requests']:
   return {'first_certifying_window_end_seconds':seg[-1]['offset_seconds'],'window_start_seconds':seg[0]['offset_seconds']-(seg[1]['offset_seconds']-seg[0]['offset_seconds']),**s}
 return None

def analyze(raw,data):
 ph=raw['phases']['measurement'];win=ph['window'];records=ph['request_records'];start,end,stop=[win[k] for k in ['start_time_ns','end_time_ns','drain_end_time_ns']];dur=win['duration_seconds'];mapping=raw['replica_to_family'];slot=raw['slot'];results=[];routes=[];bins=[]
 aliases={i:SimpleNamespace(service_time_ms=data['models'][m]['service_time_ms']) for i,m in mapping.items()};samples=[v for v in raw['telemetry']['samples'] if start<=v['time_ns']<end];unix_start=win['start_unix_ns'];unix_end=unix_start+round(dur*1e9)
 for worker,meta in raw['worker_identities'].items():
  fam=mapping[worker];rs=[r for r in records if r['model']==worker];done=[r for r in rs if r['status']=='completed'];pid=meta['pid'];observed_paths={k:[r for r in rs if r['task']==k] for k in data['models'][fam]['supported_tasks'] if any(r['task']==k for r in rs)}
  predictions={k:q.get(fam,0) for k,q in slot['routing'].items() if q.get(fam,0)>0};g=data['capacity_factors'][fam][str(slot['deployment'][fam])];D=sum(v*data['models'][fam]['service_time_ms'][k]/1000 for k,v in predictions.items());dem_actual=0
  for task,rr in observed_paths.items():
   dd=[r for r in rr if r['status']=='completed'];s0=data['models'][fam]['service_time_ms'][task];svc=stats(r['worker_latency_ms'] for r in dd);arr=len(rr)/dur;dem_actual+=arr*svc['mean']/1000
   routes.append({'run_id':slot['run_id'],'worker':worker,'task':task,'predicted_QPS':predictions.get(task,0),'s0_ms':s0,'g_x':g,'predicted_rho_task':predictions.get(task,0)*s0/1000/g,'offered_count':len(rr),'host_enqueued_count':sum('enqueued_time_ns' in r for r in rr),'host_send_attempt_count':sum('actual_submit_time_ns' in r for r in rr),'worker_accepted_proven_by_handler_count':sum(bool(r.get('worker_response',{}).get('start_ns')) for r in rr),'completed_count':len(dd),'completed_within_window':sum(r['completion_time_ns']<=end for r in dd),'arrival_QPS':arr,'window_completion_QPS':sum(r['completion_time_ns']<=end for r in dd)/dur,'cohort_completion_QPS':len(dd)/dur,'worker_service_mean_ms':svc['mean'],'worker_service_P95_ms':svc['P95'],'service_inflation_vs_s0':svc['mean']/s0,'worker_queue_ms':stats(r['worker_queue_wait_ms'] for r in dd),'E2E_ms':stats(r['e2e_latency_ms'] for r in dd)})
  submitted=timeline(rs,start,end,stop,aliases,20,'actual_submit_time_ns');offered=timeline(rs,start,end,stop,aliases,20,'scheduled_time_ns');half=slopes(submitted['bins'][10:]);tail=slopes(submitted['bins'][-5:]);first=first_growth(submitted['bins'])
  for a,b in zip(submitted['bins'],offered['bins']):bins.append({'run_id':slot['run_id'],'worker':worker,**a,'offered_mean_backlog_requests':b['mean_backlog_requests'],'offered_end_backlog_requests':b['end_backlog_requests']})
  serviced=sorted((r['worker_response']['start_ns'],r['worker_response']['end_ns']) for r in done);overlap=sum(serviced[i][0]<serviced[i-1][1] for i in range(1,len(serviced)));handler_busy=sum(max(0,min(b,unix_end)-max(a,unix_start)) for a,b in serviced)/1e9/dur
  lag_schedule=stats((r['enqueued_time_ns']-r['scheduled_time_ns'])/1e6 for r in rs);host_queue=stats((r['actual_submit_time_ns']-r['enqueued_time_ns'])/1e6 for r in done);write=stats((r['submit_write_completed_time_ns']-r['actual_submit_time_ns'])/1e6 for r in done);receive=stats((r['completion_unix_ns']-r['worker_response']['end_ns'])/1e6 for r in done)
  cpu=[p['cpu_core_percent'] for v in samples for p in v['host_processes'] if p['root_worker_pid']==pid and p['cpu_core_percent'] is not None];gpu=[u['sm_percent'] for v in samples for u in v.get('per_process_gpu_utilization',[]) if u['pid']==pid];mem=[p['used_mib'] for v in samples for p in v['processes'] if p['pid']==pid]
  tags=[]
  if dem_actual/D>1.15:tags.append('SERVICE_TIME_INFLATION')
  else:tags.append('CAPACITY_MATCH')
  if first or stats(r['worker_queue_wait_ms'] for r in done)['P95']>=1000 or submitted['end_backlog_requests']>=5:tags.insert(0,'QUEUE_BOTTLENECK')
  runtime_problem=bool(overlap or any(r['status']!='completed' for r in rs) or raw['errors'] or lag_schedule['P95']>50 or receive['P95']>50)
  if runtime_problem:tags.append('RUNTIME_ANOMALY')
  # CUDA kernel/context scheduling and CPU wait stack are not recorded; honest
  # mechanism insufficiency is separate from identifying a busy serial worker.
  results.append({'run_id':slot['run_id'],'repetition':slot['repetition'],'worker':worker,'family':fam,'checkpoint':meta['READY'].get('model_id',meta['READY'].get('model',meta['READY'].get('worker'))),'revision':meta['READY'].get('revision',fam.split('@')[-1] if '@' in fam else 'Pinned unchanged historical launcher/catalog'),'replica_count':slot['deployment'][fam],'worker_PID':pid,'task_mapping':predictions,'predicted_D':D,'g_x':g,'predicted_rho':D/g,'observed_cohort_service_demand_seconds_per_second':dem_actual,'service_demand_ratio_observed_vs_predicted':dem_actual/D,'worker_handler_busy_wall_fraction':handler_busy,'offered_count':len(rs),'host_enqueued_count':sum('enqueued_time_ns' in r for r in rs),'host_send_attempt_count':sum('actual_submit_time_ns' in r for r in rs),'worker_accepted_total_proven_by_handler':len(serviced),'accepted_receipt_timestamp_available':False,'completed_count':len(done),'within_window_completion_count':sum(r['completion_time_ns']<=end for r in done),'offered_QPS':len(rs)/dur,'within_window_throughput_QPS':sum(r['completion_time_ns']<=end for r in done)/dur,'worker_queue_ms':stats(r['worker_queue_wait_ms'] for r in done),'host_schedule_lag_ms':lag_schedule,'host_sender_queue_ms':host_queue,'pipe_write_block_ms':write,'worker_end_to_host_result_ms':receive,'worker_service_ms':stats(r['worker_latency_ms'] for r in done),'E2E_ms':stats(r['e2e_latency_ms'] for r in done),'half_backlog_slope_rps':half['slope_rps'],'tail_backlog_slope_rps':tail['slope_rps'],'tail_growth':tail['growth_requests'],'first_sustained_growth_witness':first,'end_backlog':submitted['end_backlog_requests'],'peak_backlog':submitted['peak_backlog_requests'],'offered_end_backlog':offered['end_backlog_requests'],'end_host_not_yet_send_backlog':offered['end_backlog_requests']-submitted['end_backlog_requests'],'mean_host_CPU_core_percent':statistics.mean(cpu) if cpu else None,'mean_reported_NVML_process_SM_percent':statistics.mean(gpu) if gpu else None,'mean_device_GPU_percent':statistics.mean(v['gpu_util_percent'] for v in samples),'owned_process_GPU_mib_range':[min(mem),max(mem)] if mem else None,'host_available_RAM_bytes_min':min(v['host_available_RAM_bytes'] for v in samples),'serial_handler_overlap_count':overlap,'classification':tags[0],'diagnosis_flags':tags,'mechanism_telemetry':'INSUFFICIENT_TELEMETRY: no per-kernel GPU scheduling or CPU off-CPU/wait stack trace; cannot causally partition cross-model GPU competition vs frontend/CPU vs shared-task state using logs alone.','count_conservation_passed':len(rs)==len(done)==len(serviced) and not raw['errors']})
 return results,routes,bins

def main():
 assert not (OUT/'runtime_bottleneck_evidence.json').exists()
 old=ROOT/'results/module16_fullpool_gpu';a=read(old/'artifact_manifest.json');refs=read(old/'protected_before.json')['files']+a['artifacts']+a['implementation']+[ref(old/'artifact_manifest.json')];refs=list({r['path']:r for r in refs}.values());verify(refs);save('protected_before.json',{'files':refs,'at_utc':now()})
 f=read(old/'frozen_gpu_manifest.json');data=read(f['catalog']['path']);workers=[];tasks=[];bins=[];sources=[]
 for rep in range(1,4):
  p=old/f'raw_runs/mid_rep{rep}.json';raw=read(p);sources.append(ref(p));w,t,b=analyze(raw,data);workers+=w;tasks+=t;bins+=b
 csvsave('mid_worker_diagnosis.csv',workers);csvsave('module17_mid_worker_diagnosis.csv',workers);csvsave('mid_task_path_diagnosis.csv',tasks);csvsave('mid_worker_queue_timeline.csv',bins)
 repeat=[]
 for rep in range(1,4):
  w=[r for r in workers if r['repetition']==rep];candidates=[r for r in w if r['first_sustained_growth_witness']];first=min(candidates,key=lambda r:r['first_sustained_growth_witness']['first_certifying_window_end_seconds']) if candidates else None
  repeat.append({'repetition':rep,'first_sustained_growth_worker':first['worker'] if first else None,'first_certifying_window':first['first_sustained_growth_witness'] if first else None,'workers_with_growth_witness':[r['worker'] for r in candidates],'ending_backlog_by_worker':{r['worker']:r['end_backlog'] for r in w},'workers_flagged_runtime_anomaly':[r['worker'] for r in w if 'RUNTIME_ANOMALY' in r['diagnosis_flags']]})
 standalone=[]
 for pattern in ['results/module11_replica_and_cost/raw_runs/whisper_turbo_frontier2_closed_loop_x1_rep*.json','results/module14_heavy_replica/raw_runs/path6_lid_x1_rep*.json']:
  for p in sorted(ROOT.glob(pattern)):
   raw=read(p);ph=raw['phases']['closed_loop_capacity'];rs=ph['request_records'];standalone.append({'source':ref(p),'task':rs[0]['task'],'model':rs[0]['model'],'closed_loop_capacity_QPS':ph['throughput_rps'],'worker_service_ms':stats(r['worker_latency_ms'] for r in rs if r['status']=='completed'),'protocol':'existing actual closed-loop absolute service capacity; distinct from g(x) replica throughput ratios'})
 save('runtime_bottleneck_evidence.json',{'at_utc':now(),'stage':'A_CPU_ONLY_COMPLETE_BEFORE_ANY_NEW_GPU','source_MID':sources,'catalog':f['catalog'],'repetition_comparison':repeat,'worker_rows':workers,'standalone_capacity_comparison':standalone,'first_sustained_definition':'First 30-second rolling witness over original six-second submitted backlog bins satisfying unmodified stability_v2 slope/growth/increasing/end thresholds. It is a local diagnostic, not a new stability gate.','main_worker_identified':all(r['first_sustained_growth_worker']=='whisper_turbo#0' for r in repeat),'serial_worker_semantics':'one independent process per replica, one handler at a time; turbo serially shares ASR/LID. Medium supports both but measured MID carries only LID. No simultaneous ASR/LID handlers inside a single PID.','service_time_semantics':'worker end_ns-start_ns includes native frontend/model/synchronize; excludes worker queue_wait. Not E2E. No CPU/GPU kernel split in existing log.','accepted_semantics':'Host enqueued/send-attempt and final handler starts/results separately counted. Worker receipt ACK/timestamp absent; a host send-start is not a worker acceptance ACK.','runtime_bug_found':False,'runtime_bug_causally_ruled_out':False,'need_minimal_GPU_diagnostics':True,'why_GPU_needed':'Single-model historical capacities approximately match s0, concurrent speech service is inflated, but existing full MID logs cannot partition cross-model active computation vs ASR/LID sharing vs CPU/frontend waits. Controlled unchanged-residency task isolation is needed.','mechanisms_unresolved':['cross-model GPU competition','ASR/LID shared serial checkpoint cost and mixed-task state','CPU/frontend scheduling or wait overhead'],'new_GPU_runs_before_Stage_A':0,'Module9_V3_modified':False})
 print('STAGE_A_DONE',repeat,flush=True)
 print('WORKERS',[(r['run_id'],r['family'],r['classification'],r['service_demand_ratio_observed_vs_predicted'],r['end_backlog'],r['half_backlog_slope_rps']) for r in workers],flush=True)
if __name__=='__main__':main()
