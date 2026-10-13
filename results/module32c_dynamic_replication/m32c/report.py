"""Full cohort reporting, pair-level inference; never tunes routing."""
from bisect import bisect_right
from .common import *
from .campaign import assess
from .runtime import configure
from module26.common import cohort
from module25.common import utility_inputs

def generate():
 f=read(OUT/'experiment_manifest.json');D,refs=utility_inputs();raws=[read(OUT/'raw_runs'/(s['run_id']+'.json')) for s in f['slots'] if (OUT/'raw_runs'/(s['run_id']+'.json')).exists()]
 obs=[assess(r) for r in raws];by={o['run_id']:o for o in obs};rb={r['slot']['run_id']:r for r in raws};save('run_observations.json',obs)
 rows=[];tasks=[];paths=[];sensors=[];resources=[];decisions=[];vqa=[];back=[];req=[];quant=[];tail=[];physics=[]
 _,rep=configure()
 for raw,o in zip(raws,obs):
  rid=o['run_id'];pair=raw['slot']['pair'];sec=raw['slot']['measurement_seconds']
  rows.append(dict(pair=pair,run_id=rid,arm=raw['slot']['arm'],eligible=o['eligible'],technical_valid=o['technical_valid'],U=o['U'],G_ASR=o.get('G_completed'),S_ASR=o.get('S_completed'),actual_G_fraction=o.get('G_fraction'),offered=o.get('offered'),completed=o.get('completed'),within_qps=o.get('within_qps'),cohort_qps=o.get('completed_qps'),stability=o.get('stability'),drain_seconds=o.get('drain_seconds'),safe_memory_mib=o.get('safe_memory_mib'),sampled_peak_mib=o.get('sampled_peak_mib'),**{k:v for k,v in o.get('components',{}).items() if k!='U'}))
  tasks.extend(dict(pair=pair,run_id=rid,task=t,**v) for t,v in o.get('per_task',{}).items());paths.extend(dict(pair=pair,run_id=rid,**p) for p in o.get('paths',[]));sensors.extend(dict(run_id=rid,**s) for s in raw['telemetry']['samples']);back.extend(rep.backlog(raw))
  ph=raw.get('phases',{}).get('measurement')
  if not ph:continue
  rr=ph['request_records'];ds=ph['routing_decisions'];w=ph['window'];ss=[s for s in raw['telemetry']['samples'] if w['start_time_ns']<=s['time_ns']<=w['end_time_ns']]
  resources.append(dict(pair=pair,run_id=rid,errors=raw['errors'],GPU=raw['baseline_GPU'],**{k:stats(s[k] for s in ss) for k in ['temperature_gpu_c','sm_clock_mhz','memory_clock_mhz','power_w','gpu_util_percent','owned_used_mib','worker_CPU_core_percent']},router_compute_ms=stats(d['routing_compute_ms'] for d in ds)))
  decisions.extend(dict(pair=pair,run_id=rid,**d) for d in ds);req.extend(dict(pair=pair,run_id=rid,**x) for x in rr)
  ga=[x for x in rr if x['task']=='asr' and x['model']==G and x['status']=='completed'];vv=[x for x in rr if x['task']=='visual_qa' and x['status']=='completed']
  for x in vv:vqa.append(dict(pair=pair,run_id=rid,request_id=x['request_id'],sample_id=x['sample_id'],service_ms=x['worker_latency_ms'],queue_ms=x['worker_queue_wait_ms'],E2E_ms=x['e2e_latency_ms'],Phi_ASR_handler_overlap_in_wait_ms=sum(overlap((x['worker_response']['submit_ns'],x['worker_response']['start_ns']),interval(g)) for g in ga),simultaneous_Phi_execution_ms=sum(overlap(interval(x),interval(g)) for g in ga)))
  # One-second task and system cohorts; arrivals and completions in perf clock.
  for t in raw['slot']['lambda_rps']:
   z=[x for x in rr if x['task']==t];scheduled=sorted(x['scheduled_time_ns'] for x in z);done=sorted(x['completion_time_ns'] for x in z if x['status']=='completed')
   for offset in range(int(sec)+1):at=w['start_time_ns']+offset*10**9;back.append(dict(run_id=rid,worker='task:'+t,offset_seconds=offset,offered_backlog=bisect_right(scheduled,at)-bisect_right(done,at)))
  for t in ['text','visual_qa']:
   z=sorted([x for x in rr if x['task']==t and x['status']=='completed'],key=lambda x:x['e2e_latency_ms'])
   if not z:continue
   rank=.95*(len(z)-1);lo=math.floor(rank);hi=math.ceil(rank)
   quant.append(dict(pair=pair,run_id=rid,task=t,n=len(z),zero_based_rank=rank,lower_request=z[lo]['request_id'],upper_request=z[hi]['request_id'],lower_ms=z[lo]['e2e_latency_ms'],upper_ms=z[hi]['e2e_latency_ms'],lower_weight=1-(rank-lo),upper_weight=rank-lo,p95_ms=np.percentile([x['e2e_latency_ms'] for x in z],95)))
   for rank1,x in enumerate(z,1):
    if rank1>len(z)-math.ceil(len(z)*.1):tail.append(dict(pair=pair,run_id=rid,task=t,ascending_rank=rank1,request_id=x['request_id'],sample_id=x['sample_id'],offset_seconds=x['frozen_offset_seconds'],service_ms=x['worker_latency_ms'],queue_ms=x['worker_queue_wait_ms'],E2E_ms=x['e2e_latency_ms'],host_residual_ms=x['e2e_latency_ms']-x['worker_latency_ms']-x['worker_queue_wait_ms'],Phi_ASR_handler_overlap_in_service_ms=sum(overlap(interval(x),interval(g)) for g in ga)))
  clocks=[x['actual_submit_unix_ns']-x['actual_submit_time_ns'] for x in rr if x.get('actual_submit_unix_ns')]+[x['completion_unix_ns']-x['completion_time_ns'] for x in rr if x.get('completion_unix_ns')]
  physics.append(dict(run_id=rid,Unix_perf_offset_spread_ms=(max(clocks)-min(clocks))/1e6 if clocks else None,native_formula_passed=o.get('native_timings_consistent'),timing_definition='service=end_unix-start_unix; wait=start_unix-submit_unix; E2E=completion_perf-scheduled_perf; host residual=E2E-service-wait; handler wall service includes synchronized CUDA and CPU/I/O'))
 pairs=[];decomp=[];paired=[];asrs=[];triggers=[];sensitivity=[]
 for pair in range(1,4):
  sid=f'P{pair}_S';did=f'P{pair}_D';a=by.get(did);b=by.get(sid)
  if a is None or b is None:continue
  ra=rb[did];rbase=rb[sid];matched=cohort(ra)==cohort(rbase);legal=a['eligible'] and b['eligible'] and matched;real=a.get('G_completed',0)>0 and a.get('S_completed',0)>0
  delta=a['U']-b['U'] if legal else None;temp=abs(a['temperature']['mean']-b['temperature']['mean']) if a.get('temperature',{}).get('mean') is not None and b.get('temperature',{}).get('mean') is not None else None;clock=abs(a['sm_clock']['mean']/b['sm_clock']['mean']-1) if a.get('sm_clock',{}).get('mean') and b.get('sm_clock',{}).get('mean') else None
  pairs.append(dict(pair=pair,seed=ra['slot']['seed'],order=[x for x in f['formal_order'] if x.startswith(f'P{pair}_')],matched_inputs_arrivals=matched,both_legal_stable=legal,real_G_and_S=real,S_only_U=b['U'],dynamic_U=a['U'],delta_U=delta,delta_above_001=legal and delta>.01,G_ASR=a.get('G_completed'),S_ASR=a.get('S_completed'),actual_G_fraction=a.get('G_fraction'),mean_temperature_difference_c=temp,mean_sm_clock_relative_difference=clock,original_thermal_diagnostic_passed=temp is not None and temp<=5 and clock is not None and clock<=.15,**{k+'_delta':a.get('components',{}).get(k+'_contribution',0)-b.get('components',{}).get(k+'_contribution',0) if legal else None for k in ['Q','T','L','P','M']}))
  if not ra.get('phases',{}).get('measurement') or not rbase.get('phases',{}).get('measurement'):continue
  rr=ra['phases']['measurement']['request_records'];baseline={x['request_id']:x for x in rbase['phases']['measurement']['request_records']};rates=ra['slot']['lambda_rps'];total=sum(rates.values())
  for t,q in rates.items():
   pa=a.get('per_task',{}).get(t,{});pb=b.get('per_task',{}).get(t,{})
   if pa.get('mean_ms') is None or pb.get('mean_ms') is None:continue
   lm=-D['weights']['L']*q/total*(pa['mean_ms']-pb['mean_ms'])/refs[t] if legal else None;lp=-D['weights']['P']*q/total*(pa['p95_ms']-pb['p95_ms'])/refs[t] if legal else None
   decomp.append(dict(pair=pair,task=t,n_S=pb['n_completed'],n_D=pa['n_completed'],S_mean_ms=pb['mean_ms'],D_mean_ms=pa['mean_ms'],S_p95_ms=pb['p95_ms'],D_p95_ms=pa['p95_ms'],L_delta_U=lm,P_delta_U=lp,L_P_delta_U=lm+lp if legal else None,legal_pair=legal))
  ga=[x for x in rr if x['task']=='asr' and x['model']==G and x['status']=='completed'];vqa_arrival=[x for x in rr if x['task']=='visual_qa' and x.get('actual_submit_unix_ns')];decisionmap={x['request_id']:x for x in ra['phases']['measurement']['routing_decisions']}
  for x in rr:
   if x['status']!='completed' or x['request_id'] not in baseline or baseline[x['request_id']]['status']!='completed':continue
   c=baseline[x['request_id']]
   paired.append(dict(pair=pair,request_id=x['request_id'],task=x['task'],sample_id=x['sample_id'],offset_seconds=x['frozen_offset_seconds'],S_model=c['model'],D_model=x['model'],same_worker=c['model']==x['model'],S_E2E_ms=c['e2e_latency_ms'],D_E2E_ms=x['e2e_latency_ms'],E2E_delta_ms=x['e2e_latency_ms']-c['e2e_latency_ms'],S_service_ms=c['worker_latency_ms'],D_service_ms=x['worker_latency_ms'],service_delta_ms=x['worker_latency_ms']-c['worker_latency_ms'],S_wait_ms=c['worker_queue_wait_ms'],D_wait_ms=x['worker_queue_wait_ms'],wait_delta_ms=x['worker_queue_wait_ms']-c['worker_queue_wait_ms'],host_residual_delta_ms=(x['e2e_latency_ms']-x['worker_latency_ms']-x['worker_queue_wait_ms'])-(c['e2e_latency_ms']-c['worker_latency_ms']-c['worker_queue_wait_ms']),Phi_ASR_handler_overlap_in_D_service_ms=sum(overlap(interval(x),interval(g)) for g in ga),legal_pair=legal))
  ar=[x for x in rr if x['task']=='asr' and x['status']=='completed' and x['request_id'] in baseline];ss=[x for x in ar if x['model']==S]
  def diff(z,key):return sum(x[key]-baseline[x['request_id']][key] for x in z)
  asrs.append(dict(pair=pair,legal_pair=legal,retained_S_n=len(ss),diverted_G_n=len(ga),S_wait_reduction_mean_ms=-diff(ss,'worker_queue_wait_ms')/len(ss) if ss else None,S_wait_total_saved_seconds=-diff(ss,'worker_queue_wait_ms')/1000,S_service_delta_mean_ms=diff(ss,'worker_latency_ms')/len(ss) if ss else None,G_absolute_service_seconds=sum(x['worker_latency_ms'] for x in ga)/1000,G_absolute_wait_seconds=sum(x['worker_queue_wait_ms'] for x in ga)/1000,G_extra_service_vs_same_input_S_seconds=diff(ga,'worker_latency_ms')/1000,all_ASR_E2E_delta_seconds=diff(ar,'e2e_latency_ms')/1000))
  for x in ga:
   d=decisionmap[x['request_id']];triggers.append(dict(**d,pair=pair,sample_id=x['sample_id'],offset_seconds=x['frozen_offset_seconds'],actual_service_ms=x['worker_latency_ms'],actual_queue_ms=x['worker_queue_wait_ms'],actual_E2E_ms=x['e2e_latency_ms'],S_baseline_service_ms=baseline[x['request_id']]['worker_latency_ms'],S_baseline_wait_ms=baseline[x['request_id']]['worker_queue_wait_ms'],future_VQA_arrivals_during_handler=sum(x['worker_response']['start_ns']<=v['actual_submit_unix_ns']<x['worker_response']['end_ns'] for v in vqa_arrival)))
  td=next((z for z in decomp if z['pair']==pair and z['task']=='text'),None)
  if legal and td:sensitivity.append(dict(pair=pair,official_delta_U=delta,Text_mean_contribution=td['L_delta_U'],Text_p95_contribution=td['P_delta_U'],Text_total_contribution=td['L_P_delta_U'],diagnostic_without_Text_p95_contribution=delta-td['P_delta_U'],diagnostic_without_Text_total_contribution=delta-td['L_P_delta_U'],NOT_formal_U=True,NOT_counterfactual_execution=True))
 csvsave('gpu_independent_runs.csv',rows);csvsave('gpu_independent_pairs.csv',pairs);csvsave('gpu_per_task_latency.csv',tasks);csvsave('gpu_worker_service_wait.csv',paths);csvsave('gpu_sensor_samples.csv',sensors);csvsave('gpu_resource_comparison.csv',resources);csvsave('gpu_backlog_timeseries.csv',back);csvsave('gpu_all_requests.csv',req);csvsave('dynamic_decision_log.csv',decisions);csvsave('gpu_VQA_execution_overlap.csv',vqa);csvsave('gpu_text_VQA_quantile_support.csv',quant);csvsave('gpu_text_VQA_tail_requests.csv',tail);csvsave('gpu_clock_audit.csv',physics);csvsave('per_task_utility_decomposition.csv',decomp);csvsave('paired_request_latency_decomposition.csv',paired);csvsave('Whisper_paired_queue_service.csv',asrs);csvsave('dynamic_Phi_ASR_trigger_execution.csv',triggers);csvsave('utility_sensitivity_diagnostic.csv',sensitivity)
 complete=len(obs)==6 and len(pairs)==3;legal=complete and all(p['both_legal_stable'] for p in pairs);real=complete and all(p['real_G_and_S'] for p in pairs);ds=[p['delta_U'] for p in pairs if p['delta_U'] is not None];stat=None
 if len(ds)==3:
  mean=float(np.mean(ds));sd=float(np.std(ds,ddof=1));se=sd/math.sqrt(3);crit=f['replication_statistical_policy']['t_critical'];stat=dict(n_independent_pairs=3,deltas=ds,mean_delta_U=mean,sample_SD=sd,SE=se,CI95_low=mean-crit*se,CI95_high=mean+crit*se,df=2,t_critical=crit,assumptions='Pair-level independent arrival/input-order streams; approximate t normality, unknown serial drift, only n=3; not independent requests or held-out dataset/model quality',Text_p95_mean_contribution=float(np.mean([x['Text_p95_contribution'] for x in sensitivity])),all_three_Text_p95_positive=all(x['Text_p95_contribution']>0 for x in sensitivity))
 allgt=legal and real and all(p['delta_U']>.01 for p in pairs);ci_positive=bool(stat and stat['CI95_low']>0)
 gate='DYNAMIC_REPLICATION_INCONCLUSIVE' if not (complete and legal and real) else 'DYNAMIC_REPLICATION_CONFIRMED' if allgt and ci_positive else 'POSITIVE_TREND_EVIDENCE_INSUFFICIENT' if allgt else 'DYNAMIC_REPLICATION_FAILED'
 final=(OUT/'final_data_acceptance.json').exists();audit=read(OUT/'final_data_acceptance.json') if final else None
 if final and not audit['passed']:gate='DYNAMIC_REPLICATION_INCONCLUSIVE'
 save('pair_level_statistics.json',stat or dict(n_complete_legal_pairs=len(ds),insufficient_for_planned_inference=True))
 save('module32c_gate.json',dict(gate=gate,formal_runs=len(obs),complete_three_pairs=complete,all_legal_stable=legal,real_G_and_S_each_dynamic=real,all_three_delta_above_001=allgt,mean_CI95_above_zero=ci_positive,statistics=stat,original_six_task_utility_unchanged=True,parameters_adjusted=False,seeds_replaced=False,optimizer_modified=False,next_stage_started=False,integrity_audit_pending=not final,final_audit_passed=audit['passed'] if final else None,history_preserved=audit.get('history_preserved') if final else None,observed_Text_mechanism_NOT_causally_identified=True))
 print('REPLICATION_GATE',gate,'pairs',[(x['pair'],x['delta_U'],x['G_ASR']) for x in pairs],'stats',stat,flush=True)
