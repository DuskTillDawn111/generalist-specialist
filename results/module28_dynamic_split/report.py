"""Frozen six-task utility and uncertainty; never replace missing/unstable U."""
from audit import *
from scipy.stats import t as student_t
from module26.report import p95_bounds,block_difference
from module26.common import cohort
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS
from types import SimpleNamespace

def records(raw):return raw.get('phases',{}).get('measurement',{}).get('request_records',[])
def evaluate(raw,f):
 d=json.loads(Path(f['catalog']['path']).read_text());a=arithmetic(raw,d,f['refs'],f['weights']);rs=records(raw);done=[r for r in rs if r['status']=='completed'];ph=raw.get('phases',{}).get('measurement');workers=[];states={}
 if ph:
  for w,m in raw['replica_to_family'].items():
   wr=[r for r in rs if r['model']==w]
   if not wr:continue
   tmp=dict(request_records=wr,measurement_window=ph['window'],status='completed',errors=raw['errors'],drain_timeout_seconds=60,worker_identities={w:raw['worker_identities'][w]})
   ev=reconstruct(tmp,{w:SimpleNamespace(service_time_ms=f['plan']['models'][m]['service_time_ms'])},DEFAULT_THRESHOLDS);cl=classify(ev,DEFAULT_THRESHOLDS);states[w]=cl['stability_state_v2'];workers.append(dict(worker=w,family=m,state=cl['stability_state_v2'],hard_reasons=cl['hard_reasons'],peak_backlog=ev['peak_backlog_requests'],end_backlog=ev['end_backlog_requests'],tail_slope_rps=ev['tail_backlog_slope'],final_backlog=ev['final_backlog_after_drain']))
 hard=not ph or ph['stability']['stability_state_v2']=='unstable' or any(v=='unstable' for v in states.values())
 recovery=bool(ph and (ph['stability']['stability_state_v2']!='recoverable_congested' or ph['stability'].get('final_recovery_confirmed',False)))
 capability=all(r['task'] in f['plan']['models'][raw['replica_to_family'][r['model']]]['supported_tasks'] for r in rs)
 measured_counts=all(n==1 and '1' in d['capacity_factors'][m] and '1' in d['memory_table_mib'][m] for m,n in raw['slot']['deployment'].items())
 semantic_valid=not any(r.get('worker_response',{}).get('valid_closed_set_output') is False for r in done)
 memok=a.get('safe_memory_mib') is not None and a['safe_memory_mib']<=f['budget_mib'] and a.get('sampled_peak_mib') is not None and a['sampled_peak_mib']<=f['budget_mib']
 full=bool(rs) and len(rs)==len(done);coverage=set(r['task'] for r in done)==set(f['rates'])
 eligible=bool(raw.get('valid_session') and full and coverage and capability and measured_counts and semantic_valid and memok and not hard and recovery and not raw['errors'])
 a.update(U=a.get('raw_arithmetic_U') if eligible else None,eligible=eligible,hard_unstable=hard,workers=workers,worker_states=states,full_cohort=full,admission_scope='Unchanged original measured-quality admitted task paths; subset output metrics diagnostic, no missing quality filled',capability=capability,measured_counts=measured_counts,semantic_valid=semantic_valid,memory_ok=memok,recovery_confirmed=recovery)
 return a

def match(a,b,f):
 av=evaluate(a,f);bv=evaluate(b,f);valid=av['eligible'] and bv['eligible'];same=bool(records(a) and records(b) and cohort(a)==cohort(b))
 ts=[float(np.mean([r['temperature_gpu_c'] for r in raw.get('postwarm_idle',[])])) if raw.get('postwarm_idle') else None for raw in [a,b]];tdiff=abs(ts[0]-ts[1]) if all(x is not None for x in ts) else None
 target=[r for r in records(a) if r['task']==f['target'] and r['status']=='completed'];ng=sum(r['model']==f['parameters']['G_worker'] for r in target)
 result=dict(stage=a['slot']['stage'],profile=a['slot']['arrival_profile'],repetition=a['slot']['repetition'],dynamic_run=a['slot']['run_id'],control_run=b['slot']['run_id'],control_arm=b['slot']['arm'],dynamic_U=av['U'],control_U=bv['U'],delta_U=av['U']-bv['U'] if valid else None,both_legal_stable=valid,same_arrivals_inputs=same,same_deployment=a['slot']['deployment']==b['slot']['deployment'],only_target_routing_differs=all(a['slot']['routing'][t]==b['slot']['routing'][t] for t in f['rates'] if t!=f['target']),minimum_task_count=min((p['count'] for v in [av,bv] for p in v.get('per_task',{}).values()),default=0),postwarm_temperatures_c=ts,temperature_difference_c=tdiff,thermal_comparable=tdiff is not None and tdiff<=f['selection']['thermal_match_c'],actual_G=ng,actual_S=len(target)-ng,actual_G_fraction=ng/len(target) if target else None)
 if valid:
  for key in ['T','L','P','M']:result[key+'_delta']=av[key+'_contribution']-bv[key+'_contribution']
  assert abs(result['delta_U']-sum(result[k+'_delta'] for k in ['T','L','P','M']))<1e-12
  result['task_contributions']={t:dict(mean_delta=-f['weights']['L']*q/sum(f['rates'].values())*(av['per_task'][t]['mean_ms']-bv['per_task'][t]['mean_ms'])/f['refs'][t],p95_delta=-f['weights']['P']*q/sum(f['rates'].values())*(av['per_task'][t]['p95_ms']-bv['per_task'][t]['p95_ms'])/f['refs'][t]) for t,q in f['rates'].items()}
  if same:result['conditional_block_sensitivity']=block_difference(a,b,json.loads(Path(f['catalog']['path']).read_text()),f['refs'],f['weights'],28300+a['slot']['repetition'])
 return result

def summarize_groups(raws,f,stage):
 groups=collections.defaultdict(dict)
 for raw in raws:
  s=raw['slot']
  if s['stage']==stage:groups[s['arrival_profile'],s['repetition']][s['arm']]=raw
 summaries=[];pairs=[]
 for (profile,rep),arms in sorted(groups.items()):
  if set(arms)!=set(['s_only','fixed10','dynamic']):
   summaries.append(dict(profile=profile,repetition=rep,qualifies=False,reason='INCOMPLETE_GROUP',arms=list(arms)));continue
  contrasts=[match(arms['dynamic'],arms[k],f) for k in ['s_only','fixed10']];pairs.extend(contrasts)
  strongest=max(contrasts,key=lambda x:x['control_U'] if x['control_U'] is not None else -np.inf)
  valid=all(p['both_legal_stable'] and p['same_arrivals_inputs'] and p['same_deployment'] and p['only_target_routing_differs'] and p['thermal_comparable'] and p['minimum_task_count']>=f['selection']['min_task_count'] and p['actual_G']>0 and p['actual_S']>0 for p in contrasts)
  qualified=bool(valid and strongest['delta_U'] is not None and strongest['delta_U']>.01)
  summaries.append(dict(profile=profile,repetition=rep,all_comparable=valid,qualifies=qualified,strongest_control=strongest['control_arm'],delta_to_strongest=strongest['delta_U'],contrasts=contrasts))
 return summaries,pairs

def write_tables(raws,f):
 runrows=[];taskrows=[];workerrows=[];decisions=[];thermals=[];decomp=[];interference=[]
 for order,raw in enumerate(raws,1):
  s=raw['slot'];v=evaluate(raw,f);rs=records(raw);done=[r for r in rs if r['status']=='completed'];target=[r for r in done if r['task']==f['target']];ng=sum(r['model']==f['parameters']['G_worker'] for r in target);lat=[r.get('routing_decision',{}).get('overhead_us') for r in target];lat=[x for x in lat if x is not None]
  runrows.append(dict(run_id=s['run_id'],stage=s['stage'],profile=s['arrival_profile'],arm=s['arm'],repetition=s['repetition'],seed=s['seed'],run_order=order,status=raw['status'],valid_session=raw['valid_session'],eligible=v['eligible'],U=v['U'],actual_G=ng,actual_S=len(target)-ng,actual_G_fraction=ng/len(target) if target else None,router_overhead_mean_us=float(np.mean(lat)) if lat else None,router_overhead_p95_us=float(np.percentile(lat,95)) if lat else None,formal_seconds=s['measurement_seconds'],warmup_seconds=raw.get('warmup',{}).get('window',{}).get('planned_duration_seconds'),drain_seconds=raw.get('phases',{}).get('measurement',{}).get('window',{}).get('drain_seconds'),errors=raw['errors'],raw=ref(OUT/'raw_runs'/(s['run_id']+'.json')),**{k:x for k,x in v.items() if k not in ['U','eligible','workers','per_task']}))
  for t in f['rates']:
   rr=[r for r in done if r['task']==t];responses=[r['worker_response'] for r in rr];correct=[r['correct'] for r in responses if 'correct' in r];wers=[r['wer'] for r in responses if 'wer' in r];cers=[r['cer'] for r in responses if 'cer' in r];matches=[r['exact_match'] for r in responses if 'exact_match' in r]
   taskrows.append(dict(run_id=s['run_id'],stage=s['stage'],profile=s['arrival_profile'],arm=s['arm'],task=t,offered=sum(r['task']==t for r in rs),completed=len(rr),E2E=stats([r['e2e_latency_ms'] for r in rr]),p95_interval=p95_bounds([r['e2e_latency_ms'] for r in rr]),service=stats([r['worker_latency_ms'] for r in rr]),queue=stats([r['worker_queue_wait_ms'] for r in rr]),accuracy=float(np.mean(correct)) if correct else None,accuracy_n=len(correct),WER=float(np.mean(wers)) if wers else None,WER_n=len(wers),CER=float(np.mean(cers)) if cers else None,exact_match=float(np.mean(matches)) if matches else None,semantic_invalid=sum(r.get('valid_closed_set_output') is False for r in responses),quality_scope='Subset diagnostic; admitted catalog and original semantics remain the hard path requirement'))
   if t in v.get('per_task',{}):
    p=v['per_task'][t];q=f['rates'][t]/sum(f['rates'].values());decomp.append(dict(run_id=s['run_id'],stage=s['stage'],component=t,mean_contribution=-f['weights']['L']*q*p['mean_ms']/f['refs'][t],p95_contribution=-f['weights']['P']*q*p['p95_ms']/f['refs'][t],eligible=v['eligible']))
  window=raw.get('phases',{}).get('measurement',{}).get('window',{})
  for state in v['workers']:
   w=state['worker'];wr=[r for r in done if r['model']==w];begin=window['start_unix_ns'];end=begin+round(window['planned_duration_seconds']*1e9);busy=sum(max(0,min(r['worker_response']['end_ns'],end)-max(r['worker_response']['start_ns'],begin)) for r in wr)/(end-begin)
   for t in set(r['task'] for r in wr):
    tr=[r for r in wr if r['task']==t];workerrows.append(dict(run_id=s['run_id'],stage=s['stage'],profile=s['arrival_profile'],arm=s['arm'],task=t,n=len(tr),worker_native_busy_fraction=busy,service=stats([r['worker_latency_ms'] for r in tr]),queue=stats([r['worker_queue_wait_ms'] for r in tr]),**state))
  for r in rs:
   if 'routing_decision' in r:decisions.append(dict(run_id=s['run_id'],stage=s['stage'],profile=s['arrival_profile'],arm=s['arm'],request_id=r['request_id'],task=r['task'],sample_id=r['sample_id'],final_status=r['status'],actual_route=r['model'],actual_service_ms=r['worker_latency_ms'],actual_wait_ms=r['worker_queue_wait_ms'],actual_E2E_ms=r.get('e2e_latency_ms'),actual_submit_time_ns=r.get('actual_submit_time_ns'),completion_time_ns=r.get('completion_time_ns'),**r['routing_decision']))
  for row in raw.get('telemetry',{}).get('samples',[]):thermals.append(dict(run_id=s['run_id'],run_order=order,**{k:row.get(k) for k in ['time_ns','unix_ns','phase','temperature_gpu_c','power_w','sm_clock_mhz','memory_clock_mhz','performance_state','used_mib','owned_used_mib','gpu_util_percent','foreign_pids']}))
 # Matched-input service/wait contrasts on the actually shared route. Includes
 # target S subset and every background task; it is bundled coexistence evidence.
 for stage in ['C','D']:
  groups=collections.defaultdict(dict)
  for raw in raws:
   if raw['slot']['stage']==stage:groups[raw['slot']['arrival_profile'],raw['slot']['repetition']][raw['slot']['arm']]=raw
  for (profile,rep),arms in groups.items():
   if 's_only' not in arms:continue
   control=arms['s_only'];cr={r['request_id']:r for r in records(control) if r['status']=='completed'}
   for arm in ['fixed10','dynamic']:
    if arm not in arms:continue
    a=arms[arm]
    for worker in a['replica_to_family']:
     for task in f['rates']:
      common=[(r,cr[r['request_id']]) for r in records(a) if r['status']=='completed' and r['request_id'] in cr and r['task']==task and r['model']==worker and cr[r['request_id']]['model']==worker]
      if not common:continue
      assert all(r['sample_id']==c['sample_id'] and r['frozen_offset_seconds']==c['frozen_offset_seconds'] for r,c in common)
      interference.append(dict(stage=stage,profile=profile,repetition=rep,arm=arm,run_id=a['slot']['run_id'],control_run=control['slot']['run_id'],worker=worker,task=task,matched_inputs=len(common),control_service_mean_ms=float(np.mean([c['worker_latency_ms'] for r,c in common])),arm_service_mean_ms=float(np.mean([r['worker_latency_ms'] for r,c in common])),service_delta_ms=float(np.mean([r['worker_latency_ms']-c['worker_latency_ms'] for r,c in common])),control_queue_mean_ms=float(np.mean([c['worker_queue_wait_ms'] for r,c in common])),arm_queue_mean_ms=float(np.mean([r['worker_queue_wait_ms'] for r,c in common])),queue_delta_ms=float(np.mean([r['worker_queue_wait_ms']-c['worker_queue_wait_ms'] for r,c in common])),scope='Matched actual routes/inputs; bundled routing/coexistence/thermal observations, not unique physical interference attribution'))
 csvsave('gpu_exploration_runs.csv',[r for r in runrows if r['stage']=='C'],fields=['run_id','stage','arm','U','eligible']);csvsave('all_gpu_runs.csv',runrows,fields=['run_id','stage','arm','U','eligible']);csvsave('per_task_latency.csv',taskrows);csvsave('worker_metrics.csv',workerrows);csvsave('routing_decision_trace.csv',decisions,fields=['run_id','request_id','actual_route','reason']);csvsave('worker_interference_analysis.csv',interference,fields=['run_id','worker','task','service_delta_ms','queue_delta_ms']);csvsave('GPU_thermal_state.csv',thermals);csvsave('utility_decomposition.csv',decomp)
 return runrows

def finalize(raws,f,fatal=None):
 rows=write_tables(raws,f);cs,cp=summarize_groups(raws,f,'C');ds,dp=summarize_groups(raws,f,'D');csvsave('utility_comparison.csv',cp+dp,fields=['stage','profile','repetition','control_arm','delta_U'])
 if dp:csvsave('independent_validation.csv',dp)
 qualified=[r for r in cs if r['qualifies']];winner=max(qualified,key=lambda r:(r['delta_to_strongest'],r['profile']=='steady')) if qualified else None
 ci=None;deltas=[r.get('delta_to_strongest') for r in ds if r.get('delta_to_strongest') is not None]
 if len(deltas)>=2:
  mean=float(np.mean(deltas));half=float(student_t.ppf(.975,len(deltas)-1)*np.std(deltas,ddof=1)/np.sqrt(len(deltas)));ci=dict(mean_delta_U=mean,low=mean-half,high=mean+half,n=len(deltas),method='Paired independent-session effects Student t95; approximate normal-effects assumption; n3 highly assumption-sensitive')
 replicated=bool(winner and len(ds)==3 and all(r['qualifies'] for r in ds) and ci and ci['low']>0 and not fatal)
 gate='DYNAMIC_POSITIVE_REPLICATED' if replicated else 'DYNAMIC_NOT_REPRODUCED' if ds else 'DYNAMIC_EXPLORATORY_POSITIVE_ONLY' if winner else 'DYNAMIC_NO_UTILITY_GAIN' if raws else 'NO_DYNAMIC_OPPORTUNITY'
 historical_before=json.loads((OUT/'historical_metadata_before.json').read_text());after=metadata();changes=[p for p,stat in historical_before.items() if after.get(p)!=stat];hash_changes=[r['path'] for r in json.loads((OUT/'historical_sources.json').read_text()) if ref(r['path'])['sha256']!=r['sha256']];assert not changes and not hash_changes,(changes,hash_changes)
 doc=dict(gate=gate,at_unix_ns=time.time_ns(),completed_and_stopped=True,stage_A='SUPPORTED_REAL_BACKLOG_AND_G_EMPTY_WINDOWS; unexecuted marginal benefit UNSUPPORTED until GPU',stage_B='INDEPENDENT_FROZEN_PROTOTYPE',exploration_groups=cs,selected_candidate=winner,independent_groups=ds,independent_delta_U=deltas,independent_t95_interval=ci,independently_replicated=replicated,fatal=fatal,exploration_attempts=sum(r['slot']['stage']=='C' for r in raws),validation_attempts=sum(r['slot']['stage']=='D' for r in raws),formal_seconds=sum(r.get('phases',{}).get('measurement',{}).get('window',{}).get('planned_duration_seconds',0) for r in raws),maximum_exploration_attempts=f['maximum_exploration_attempts'],maximum_total_attempts=f['maximum_total_attempts'],optimizer_modified=False,optimizer_integration_worth_studying=replicated,next_module_automatic=False,historical_files_checked=len(historical_before),historical_metadata_changes=changes,historical_source_hash_changes=hash_changes,small_sample_limit='n=3: paired-t assumes normal effects; one-sided exact sign p minimum0.125; conditional single-pair block intervals are not independent CI; no cross-pool generalization',definitions_and_parameters_unchanged=True)
 save('historical_integrity.json',dict(metadata_files=len(historical_before),metadata_changes=changes,read_source_hash_changes=hash_changes));save('module28_gate.json',doc)
 text=['# Module28：Queue-Aware Dynamic G/S Routing Feasibility Study\n\n',f'Gate：**{gate}**。完成后停止，不自动进入下一Module。\n\n','固定Qwen2.5-Omni、Whisper-medium、RoBERTa、ConvNeXt、PARSeq各1副本，六任务QPS：Text/Image/OCR/ASR/VQA各0.2，LID11.5。B4=34567.5534MiB；20A效用、归一化和质量准入未改。S-only、开发预选最佳已测非零固定10%、Dynamic对照；0%同样是固定路由，且是历史更强对照。非零比例并非全局连续最优。G-only因服务需求明显超过单副本容量而预先排除。\n\n',f'探索{doc["exploration_attempts"]}/6次，每臂600s，独立验证{doc["validation_attempts"]}/9次，预算15次/9000s正式窗口。warmup30s、prime、冷却与排空单报；失败不重试，全部记录保留。\n\n','|阶段/轨迹/组|对照|ΔU|合法稳定|匹配热状态|\n|---|---|---:|---|---|\n']
 for p in cp+dp:text.append(f'|{p["stage"]}/{p["profile"]}/{p["repetition"]}|{p["control_arm"]}|{p["delta_U"]}|{p["both_legal_stable"]}|{p["thermal_comparable"]}|\n')
 text.append('\n逐任务效用贡献、六任务mean/P95及IID条件P95区间见utility_decomposition.csv、per_task_latency.csv；全系统pooled mean/P95、offered/completed QPS、safe/peak显存、实际路由比例和开销见all_gpu_runs.csv；逐请求状态/原因、worker queue/service与匹配输入干扰见routing_decision_trace.csv、worker_metrics.csv和worker_interference_analysis.csv。pooled P95不代替效用中的逐任务P95。\n\n')
 text.append(f'独立效应：{deltas}，t95区间：{ci}。n=3小样本强依赖近似正态，符号检验最小单侧p=0.125；单配对60s block仅条件敏感性，队列相关影响IID区间；不推广到其他模型池。\n\n')
 text.append('质量保持原冻结准入路径，实测子集accuracy/WER/CER/closed-set有效性同时保留，不填缺失指标，不把小子集得分当新准入认证。GPU显存为0.5s NVML采样高水位，瞬时allocator峰值UNSUPPORTED。无worker receipt/START ACK，在线只用已入队未完成状态和已收到RESULT；原始handler时间仅用于事后分析。混合运行服务变化不能唯一归因到GPU/CPU/热状态中的某一种。\n\n')
 if winner:
  text.append(f'探索候选：{winner["profile"]}，相对最强合法稳定对照ΔU={winner["delta_to_strongest"]}。动态G/S均实际非零。\n\n')
 else:text.append('未得到满足预注册门槛的动态正收益候选；独立复现不执行，停止扩展参数/负载网格。\n\n')
 if fatal:text.append(f'停止原因：{fatal}。缺失比较或不完整运行不可计算正常U。\n\n')
 text.append('五个问题：\n\n')
 text.append(f'1. 真实动态正收益：探索={bool(winner)}；独立复现={replicated}。\n')
 text.append('2. 路由到G时的状态：S保守队列代价高，G没有host可见未完成请求、目标在途为0，token/冷却/准入/容量通过；没有把平均GPU利用率当空闲。具体收益和状态必须结合实测表，不能由路由条件直接证明因果。\n')
 text.append(f'3. 优于S-only和最佳已测固定对照：探索={bool(winner)}，独立={replicated}；逐组差值全部报告。\n')
 text.append(f'4. 独立GPU收益复现：{replicated}。\n')
 text.append('5. '+('值得进一步研究正式优化器如何自主选择启用，但本Module不修改优化器。' if replicated else '当前证据不支持集成进正式部署优化器；不宣称优化器创新成功。')+'\n\n')
 text.append(f'历史{len(historical_before)}文件metadata和已读科学来源SHA256保持不变，全部负结果保留。\n')
 (OUT/'module28_summary.md').write_text(''.join(text))
 return doc
