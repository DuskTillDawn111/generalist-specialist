from .common import *
from .campaign import assess
from .runtime import configure
from module26.common import cohort

def generate():
 f=read(OUT/'experiment_manifest.json');configure();raws=[read(OUT/'raw_runs'/(s['run_id']+'.json')) for s in f['slots'] if (OUT/'raw_runs'/(s['run_id']+'.json')).exists()];obs=[assess(r) for r in raws];by={o['run_id']:o for o in obs};save('run_observations.json',obs);rows=[];tasks=[];paths=[];back=[];resources=[];sensors=[];dec=[];vqa=[];decomp=[];rawby={r['slot']['run_id']:r for r in raws};b=by.get('A0',{});control={r['request_id']:r for r in rawby.get('A0',{}).get('phases',{}).get('measurement',{}).get('request_records',[])}
 _,rep=configure()
 for r,o in zip(raws,obs):
  aid=o['run_id'];delta=o['U']-b['U'] if o['eligible'] and b.get('eligible') else None
  rows.append(dict(arm=aid,route=r['slot']['arm'],eligible=o['eligible'],U=o['U'],delta_U_vs_S_only=delta,actual_G_fraction=o.get('G_fraction'),G_ASR=o.get('G_completed'),S_ASR=o.get('S_completed'),completed=o.get('completed'),offered=o.get('offered'),within_qps=o.get('within_qps'),cohort_qps=o.get('completed_qps'),stability=o.get('stability'),safe_memory_mib=o.get('safe_memory_mib'),sampled_peak_mib=o.get('sampled_peak_mib'),drain_seconds=o.get('drain_seconds'),**{k:v for k,v in o.get('components',{}).items() if k!='U'}))
  tasks.extend(dict(arm=aid,task=t,**v) for t,v in o.get('per_task',{}).items());paths.extend(dict(arm=aid,**v) for v in o.get('paths',[]));back.extend(rep.backlog(r));sensors.extend(dict(arm=aid,**v) for v in r['telemetry']['samples']);ph=r.get('phases',{}).get('measurement')
  if not ph:continue
  rr=ph['request_records'];ds=ph.get('routing_decisions',[]);dec.extend(dict(arm=aid,**x) for x in ds);w=ph['window'];ss=[x for x in r['telemetry']['samples'] if w['start_time_ns']<=x['time_ns']<=w['end_time_ns']];resources.append(dict(arm=aid,errors=r['errors'],**{k:stats(x[k] for x in ss) for k in ['temperature_gpu_c','sm_clock_mhz','memory_clock_mhz','power_w','owned_used_mib','gpu_util_percent','worker_CPU_core_percent']},router_compute_ms=stats(x['routing_compute_ms'] for x in ds)))
  ga=[x for x in rr if x['model']==G and x['task']=='asr' and x['status']=='completed'];vv=[x for x in rr if x['task']=='visual_qa' and x['status']=='completed']
  for x in vv:
   wait=(x['worker_response']['submit_ns'],x['worker_response']['start_ns']);ov=sum(overlap(wait,interval(y)) for y in ga);vqa.append(dict(arm=aid,request_id=x['request_id'],sample_id=x['sample_id'],service_ms=x['worker_latency_ms'],queue_ms=x['worker_queue_wait_ms'],E2E_ms=x['e2e_latency_ms'],ASR_handler_overlap_in_wait_ms=ov,simultaneous_Phi_native_execution_ms=sum(overlap(interval(x),interval(y)) for y in ga)))
  if aid!='A0' and control:
   ar=[x for x in rr if x['task']=='asr' and x['status']=='completed' and x['request_id'] in control];same=[x for x in ar if x['model']==S];gg=[x for x in ar if x['model']==G]
   def diff(z,k):return sum(x[k]-control[x['request_id']][k] for x in z)
   decomp.append(dict(arm=aid,retained_S_n=len(same),diverted_G_n=len(gg),S_wait_reduction_mean_ms=-diff(same,'worker_queue_wait_ms')/len(same) if same else None,S_wait_total_saved_seconds=-diff(same,'worker_queue_wait_ms')/1000,S_service_delta_mean_ms=diff(same,'worker_latency_ms')/len(same) if same else None,G_execution_absolute_seconds=sum(x['worker_latency_ms'] for x in gg)/1000,G_wait_absolute_seconds=sum(x['worker_queue_wait_ms'] for x in gg)/1000,G_extra_execution_vs_same_input_S_seconds=diff(gg,'worker_latency_ms')/1000,legal_pair=o['eligible'] and b.get('eligible',False)))
 csvsave('gpu_exploration_comparison.csv',rows);csvsave('gpu_per_task_latency.csv',tasks);csvsave('gpu_worker_service_wait.csv',paths);csvsave('gpu_backlog_timeseries.csv',back);csvsave('gpu_resource_comparison.csv',resources);csvsave('gpu_sensor_samples.csv',sensors);csvsave('dynamic_decision_log.csv',dec,['arm','request_id','select_G','reason']);csvsave('gpu_Phi_VQA_overlap.csv',vqa);csvsave('gpu_native_ASR_decomposition.csv',decomp,['arm','retained_S_n','diverted_G_n'])
 matched=len(raws)==3 and all(cohort(x)==cohort(raws[0]) for x in raws);legal=len(obs)==3 and all(x['eligible'] for x in obs);dyn=by.get('A2',{});fixed=by.get('A1',{});duS=dyn['U']-b['U'] if dyn.get('eligible') and b.get('eligible') else None;duF=dyn['U']-fixed['U'] if dyn.get('eligible') and fixed.get('eligible') else None;real=bool(dyn.get('G_completed',0)>0 and dyn.get('S_completed',0)>0)
 gate='DYNAMIC_EVIDENCE_INCONCLUSIVE' if not (matched and legal and real) else 'DYNAMIC_EXPLORATORY_POSITIVE' if duS>.01 and duF>.01 else 'DYNAMIC_NO_POSITIVE_GAIN'
 save('module32b_gate.json',dict(gate=gate,formal_arms=len(obs),all_three_legal_stable=legal,matched_trajectory=matched,dynamic_G_ASR=dyn.get('G_completed'),dynamic_actual_fraction=dyn.get('G_fraction'),dynamic_cooperation_really_occurred=real,dynamic_delta_U_vs_S_only=duS,dynamic_delta_U_vs_fixed=duF,threshold=.01,independent_replication=False,optimizer_modified=False,preliminary_integrity_pending=True))
 print('GATE',gate,'delta_S',duS,'delta_fixed',duF,'G',dyn.get('G_completed'),flush=True)
