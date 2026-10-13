"""Complete system U and conditional selection; no predictor fitting."""
from .common import *
from module26.report import p95_bounds,block_difference
from scipy.stats import t as student_t

def pair(a,b,f):
 av,ao=arithmetic(a);bv,bo=arithmetic(b);valid=ao['eligible'] and bo['eligible'];same=bool(a.get('phases',{}).get('measurement') and b.get('phases',{}).get('measurement') and cohort(a)==cohort(b))
 temperatures=[float(np.mean([v['temperature_gpu_c'] for v in r.get('postwarm_idle',[])])) if r.get('postwarm_idle') else None for r in [a,b]]
 tempdiff=abs(temperatures[0]-temperatures[1]) if all(v is not None for v in temperatures) else None
 pmin=min((v['count'] for o in [ao,bo] for v in o.get('per_task',{}).values()),default=0)
 actual=[r for r in a.get('phases',{}).get('measurement',{}).get('request_records',[]) if r['task']==TARGET and r['status']=='completed']
 ng=sum(a['replica_to_family'][r['model']]==G for r in actual);ns=sum(a['replica_to_family'][r['model']]==MEDIUM for r in actual)
 result=dict(point_id=a['slot']['point_id'],repetition=a['slot']['repetition'],stage=a['slot']['stage'],
  proposed_run=a['slot']['run_id'],control_run=b['slot']['run_id'],G_fraction=a['slot']['G_fraction'],
  proposed_U=ao['U'],control_U=bo['U'],delta_U=ao['U']-bo['U'] if valid else None,both_legal_stable=valid,
  same_arrivals_inputs=same,same_deployment=a['slot']['deployment']==b['slot']['deployment'],
  only_target_routing_differs=all(a['slot']['routing'][t]==b['slot']['routing'][t] for t in a['slot']['routing'] if t!=TARGET),
  minimum_task_count=pmin,postwarm_temperatures_c=temperatures,temperature_difference_c=tempdiff,
  thermal_comparable=tempdiff is not None and tempdiff<=f['thermal_protocol']['postwarm_pair_temperature_difference_max_c'],
  actual_target_G=ng,actual_target_S=ns,actual_target_G_fraction=ng/len(actual) if actual else None,
  subset_statistics_NOT_normal_U=not valid)
 if valid:
  result.update({k+'_delta':ao['actual_components'][k+'_contribution']-bo['actual_components'][k+'_contribution'] for k in ['T','L','P','M']})
  assert abs(result['delta_U']-sum(result[k+'_delta'] for k in ['T','L','P','M']))<1e-12
  if same:
   d,refs,weights=inputs();result['conditional_block_sensitivity']=block_difference(a,b,d,refs,weights,27300+a['slot']['repetition'])
  effects=[]
  for model in a['slot']['deployment']:
   for task in a['slot']['lambda_rps']:
    common=[(ra,rb) for ra,rb in zip(a['phases']['measurement']['request_records'],b['phases']['measurement']['request_records'])
     if ra['task']==task and a['replica_to_family'][ra['model']]==model and b['replica_to_family'][rb['model']]==model]
    if not common:continue
    effects.append(dict(model=model,task=task,matched_requests=len(common),
     proposed_service_mean_ms=float(np.mean([ra['worker_latency_ms'] for ra,rb in common])),
     control_service_mean_ms=float(np.mean([rb['worker_latency_ms'] for ra,rb in common])),
     proposed_queue_mean_ms=float(np.mean([ra['worker_queue_wait_ms'] for ra,rb in common])),
     control_queue_mean_ms=float(np.mean([rb['worker_queue_wait_ms'] for ra,rb in common])),
     common_input_interference_scope='Matched requests on same path; workload/route changes bundled, not unique physical GPU causal attribution'))
  result['matched_path_service_effects']=effects
 return result

def stage_b(executed,f,fatal=None):
 raws=[read(OUT/'raw_runs'/(n+'.json')) for n in executed];pairs=[];eligible=[]
 for point in f['points']:
  rr=[r for r in raws if r['slot']['point_id']==point['point_id']];control=next((r for r in rr if r['slot']['G_fraction']==0),None)
  if control is None:continue
  for a in rr:
   if a['slot']['G_fraction']==0:continue
   p=pair(a,control,f);pairs.append(p);rule=f['exploratory_selection']
   support=p['minimum_task_count']>=rule['min_task_requests']
   qualified=bool(not fatal and p['both_legal_stable'] and p['same_arrivals_inputs'] and p['same_deployment'] and p['only_target_routing_differs']
    and p['thermal_comparable'] and support and p['actual_target_G']>0 and p['actual_target_S']>0 and p['delta_U']>.01
    and p.get('conditional_block_sensitivity',{}).get('low',-math.inf)>0)
   p.update(adequate_samples_for_conditional_P95=support,qualifies_for_independent_replication=qualified)
   if qualified:eligible.append(p)
 winner=min(eligible,key=lambda p:(-p['delta_U'],p['proposed_run'])) if eligible else None
 result=dict(at_utc=now(),status='POSITIVE_EXPLORATORY_ONLY' if winner else 'NO_POSITIVE_EXPLORATORY_CANDIDATE',
  completed_runs=executed,pairs=pairs,selected=winner,Stage_C_permitted=winner is not None,development_only=True,
  no_development_data_counted_independently=True,conditional_block_intervals_not_independent_CI=True,fatal=fatal)
 csvsave('exploratory_utility_comparison.csv',pairs);save('exploratory_result.json',result)
 if winner:
  source=next(r for r in raws if r['slot']['run_id']==winner['proposed_run']);point=next(p for p in f['points'] if p['point_id']==winner['point_id']);slots=[]
  from .prepare import slot
  for group,order in zip(point['independent_pre_generated'],f['independent_protocol']['orders']):
   for arm in order:
    fraction=winner['G_fraction'] if arm=='proposed' else 0.
    slots.append(slot(f'C_{point["point_id"]}_{arm}_rep{group["rep"]}','C',point['point_id'],point['rates'],fraction,group['seed'],group['rep'],2400.,group['trajectories']))
  frozen=dict(at_utc=now(),status='FROZEN_BEFORE_INDEPENDENT_GPU',selection_rule_source=ref(OUT/'discovery_protocol.json'),
   development_only_selection=winner,models=f['models'],deployment=source['slot']['deployment'],rates=source['slot']['lambda_rps'],
   routing=source['slot']['routing'],control_routing=routing(point['rates'],0.),budget_mib=BUDGET,slots=slots,
   independent_seeds_and_durations_predeclared=True,development_U_not_independent=True)
  save('frozen_positive_candidate.json',frozen)
 else:save('frozen_positive_candidate.json',dict(status='NO_QUALIFIED_CANDIDATE',candidates=[],Stage_C=False))
 print('STAGE_B_RESULT',result['status'],'winner',winner['proposed_run'] if winner else None,flush=True)
 return result

def stage_c(executed,f):
 raws=[read(OUT/'raw_runs'/(n+'.json')) for n in executed];pairs=[]
 for rep in [1,2,3]:
  rr=[r for r in raws if r['slot']['repetition']==rep];arms={r['slot']['arm']:r for r in rr}
  if set(arms)=={'control','proposed'}:pairs.append(pair(arms['proposed'],arms['control'],f))
 deltas=[p['delta_U'] for p in pairs if p['delta_U'] is not None];ci=None
 if len(deltas)>=2:
  mean=float(np.mean(deltas));half=float(student_t.ppf(.975,len(deltas)-1)*np.std(deltas,ddof=1)/math.sqrt(len(deltas)))
  ci=dict(mean_delta_U=mean,low=mean-half,high=mean+half,n=len(deltas),method='Approximate paired t95; independent session effects assumed normal; n3 not broad generalization')
 comparable=len(pairs)==3 and all(p['both_legal_stable'] and p['same_arrivals_inputs'] and p['same_deployment'] and p['only_target_routing_differs']
  and p['thermal_comparable'] and p['minimum_task_count']>=f['independent_protocol']['min_task_requests'] and p['actual_target_G']>0 and p['actual_target_S']>0 for p in pairs)
 all_positive=len(deltas)==3 and all(v>.01 for v in deltas);replicated=bool(comparable and all_positive and ci and ci['low']>0)
 result=dict(at_utc=now(),status='POSITIVE_SPLIT_REPLICATED' if replicated else 'POSITIVE_NOT_REPRODUCED',pairs=pairs,
  independent_delta_U=deltas,paired_effect_interval=ci,all_three_over001=all_positive,all_comparable=comparable,
  independent_positive=replicated,Stage_D_permitted=replicated,development_data_excluded=True,
  small_sample_limit='Three successes have exact one-sided sign p minimum0.125; paired t interval assumes normal effects; not distribution-free5 percent proof')
 csvsave('independent_paired_utility.csv',pairs);save('independent_result.json',result)
 print('STAGE_C_RESULT',result['status'],deltas,ci,flush=True);return result

def write_run_tables(executed):
 runrows=[];taskrows=[];workers=[];decomposition=[];thermalrows=[]
 for name in executed:
  raw=read(OUT/'raw_runs'/(name+'.json'));v,o=arithmetic(raw);s=raw['slot'];rs=raw.get('phases',{}).get('measurement',{}).get('request_records',[])
  done=[r for r in rs if r['status']=='completed'];target=[r for r in done if r['task']==TARGET];ng=sum(raw['replica_to_family'][r['model']]==G for r in target)
  runrows.append(dict(run_id=name,stage=s['stage'],point_id=s['point_id'],arm=s['arm'],repetition=s['repetition'],seed=s['seed'],G_fraction=s['G_fraction'],
   actual_target_G_fraction=ng/len(target) if target else None,actual_target_G=ng,actual_target_S=len(target)-ng,
   actual_U=o['U'],eligible=o['eligible'],hard_unstable=o['hard_unstable'],valid_session=raw['valid_session'],errors=raw['errors'],raw=ref(OUT/'raw_runs'/(name+'.json')),
   **{k:val for k,val in v.items() if k not in ['U','eligible','per_task']}))
  for k in ['T','L','P','M']:decomposition.append(dict(run_id=name,stage=s['stage'],component=k,contribution=v.get(k+'_contribution'),eligible=o['eligible']))
  for task,q in s['lambda_rps'].items():
   rows=[r for r in done if r['task']==task];lat=[r['e2e_latency_ms'] for r in rows];responses=[r['worker_response'] for r in rows]
   correct=[r['correct'] for r in responses if isinstance(r.get('correct'),bool)];words=[r for r in responses if r.get('word_errors') is not None and r.get('reference_words') is not None]
   refs_words=sum(r['reference_words'] for r in words)
   taskrows.append(dict(run_id=name,stage=s['stage'],task=task,offered=sum(r['task']==task for r in rs),completed=len(rows),offered_qps=q,
    E2E=stats(lat),P95_interval=p95_bounds(lat),service=stats([r['worker_latency_ms'] for r in rows]),queue=stats([r['worker_queue_wait_ms'] for r in rows]),
    quality_admission='Original frozen capability/quality policy unchanged; no utility reward',accuracy=sum(correct)/len(correct) if correct else None,
    accuracy_scored_requests=len(correct),weighted_WER=sum(r['word_errors'] for r in words)/refs_words if refs_words else None,WER_scored_requests=len(words),
    invalid_closed_set=sum(r.get('valid_closed_set_output') is False for r in responses),missing_quality_not_imputed=True,
    subset_latency_only_diagnostic=not o['eligible']))
  window=raw.get('phases',{}).get('measurement',{}).get('window')
  for state in o.get('workers',[]):
   rr=[r for r in done if r['model']==state['worker']];m=raw['replica_to_family'][state['worker']]
   busy=sum(max(0,min(r['worker_response']['end_ns'],window['start_unix_ns']+int(window['planned_duration_seconds']*1e9))-max(r['worker_response']['start_ns'],window['start_unix_ns'])) for r in rr)/(window['planned_duration_seconds']*1e9) if window else None
   for task in sorted({r['task'] for r in rr}):
    tt=[r for r in rr if r['task']==task];d,_,_=inputs()
    workers.append(dict(run_id=name,stage=s['stage'],model=m,task=task,**state,worker_busy_wall_fraction=busy,
     count=len(tt),service=stats([r['worker_latency_ms'] for r in tt]),queue=stats([r['worker_queue_wait_ms'] for r in tt]),
     service_vs_standalone=float(np.mean([r['worker_latency_ms'] for r in tt]))/d['models'][m]['service_time_ms'][task]))
  for phase_name,samples in [('postwarm_idle',raw.get('postwarm_idle',[])),('measurement',[v for v in raw['telemetry']['samples'] if v['phase']=='measurement'])]:
   samples=[v for v in samples if 'temperature_gpu_c' in v]
   if samples:thermalrows.append(dict(run_id=name,phase=phase_name,mean_temperature_c=float(np.mean([v['temperature_gpu_c'] for v in samples])),
    max_temperature_c=max(v['temperature_gpu_c'] for v in samples),mean_SM_clock_MHz=float(np.mean([v['sm_clock_mhz'] for v in samples])),
    mean_power_w=float(np.mean([v['power_draw_w'] for v in samples])),foreign_GPU_samples=sum(bool(v['foreign_pids']) for v in samples)))
 csvsave('gpu_discovery_runs.csv',runrows);csvsave('per_task_latency.csv',taskrows);csvsave('worker_queue_stability.csv',workers)
 csvsave('utility_decomposition.csv',decomposition);csvsave('GPU_thermal_state.csv',thermalrows)
 return runrows
