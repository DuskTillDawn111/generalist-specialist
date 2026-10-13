"""CPU raw accounting and original Module20A utility; no optimizer change."""
import collections,math,statistics,time
from bisect import bisect_right
from types import SimpleNamespace
import numpy as np
from .common import *
from module20.optimizer import components
from module25.common import utility_inputs
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS

def stats(rows,key):
 a=[r[key] for r in rows if r.get(key) is not None]
 return dict(n=len(a),mean=float(np.mean(a)),p95=float(np.percentile(a,95))) if a else dict(n=0,mean=None,p95=None)
def journal_audit(raw):
 slot=raw['slot'];path=OUT/'request_journals'/(slot['run_id']+'.jsonl');terminal={};duplicates=[];headers=[]
 try:
  with path.open() as f:
   for line in f:
    row=json.loads(line)
    if row['kind']=='PHASE_WINDOW':headers.append(row['label'])
    if row['kind']=='REQUEST_TERMINAL':
     key=(row['phase'],row['record']['request_id'])
     if key in terminal:duplicates.append(key)
     terminal[key]=row['record']
 except Exception as exc:return dict(passed=False,error=repr(exc))
 cohorts=[raw.get('warmup')]+list(raw.get('phases',{}).values())+[raw.get('aborted_phase')]
 keys=[];missing=[];mismatch=[];fields=['status','task','model','actual_submit_time_ns','completion_time_ns','worker_latency_ms','worker_queue_wait_ms','worker_response']
 for ph in cohorts:
  if not ph:continue
  for r in ph['request_records']:
   key=(r['phase'],r['request_id']);keys.append(key);s=terminal.get(key)
   if s is None:missing.append(key);continue
   if any(s.get(k)!=r.get(k) for k in fields):mismatch.append(key)
 return dict(passed=not missing and not duplicates and not mismatch and len(keys)==len(set(keys)),expected=len(keys),terminal=len(terminal),missing=missing,duplicates=duplicates,mismatch=mismatch,headers=headers)
def observation(raw):
 f=read(OUT/'frozen_gpu_manifest.json');slot=raw['slot'];ph=raw.get('phases',{}).get('measurement');audit=journal_audit(raw)
 out=dict(run_id=slot['run_id'],candidate=slot['candidate'],arm=slot['arm'],Generalist=slot['Generalist'],Specialist=slot['Specialist'],
  status=raw['status'],valid_session=raw.get('valid_session',False),eligible=False,U=None,journal_audit=audit,errors=raw['errors'])
 if not ph:return {**out,'stability':'UNASSESSABLE_ABORT','reason':'No full measurement phase, entire planned cohort retained'}
 rates=slot['lambda_rps'];rs=ph['request_records'];done=[r for r in rs if r['status']=='completed'];mapping=raw['replica_to_family'];window=ph['window']
 taskstats={};per={};paths=[];workers=[]
 for t,q in rates.items():
  rr=[r for r in rs if r['task']==t];dd=[r for r in rr if r['status']=='completed'];e=stats(dd,'e2e_latency_ms')
  per[t]=dict(n_offered=len(rr),n_completed=len(dd),mean_ms=e['mean'],p95_ms=e['p95'],offered_qps=len(rr)/600,completed_qps=len(dd)/600,
   within_qps=sum(r['completion_time_ns']<=window['end_time_ns'] for r in dd)/600)
  if e['n']:taskstats[t]=dict(mean_ms=e['mean'],p95_ms=e['p95'])
  requested_workers={r['model'] for r in rr}
  if t==slot['target_task']:requested_workers.update([slot['Generalist']+'#0',slot['Specialist']+'#0'])
  for worker in sorted(requested_workers):
   pp=[r for r in dd if r['model']==worker];paths.append(dict(worker=worker,model=mapping[worker],task=t,completed=len(pp),
    service=stats(pp,'worker_latency_ms'),wait=stats(pp,'worker_queue_wait_ms'),e2e=stats(pp,'e2e_latency_ms'),host_schedule_delay=stats(pp,'schedule_delay_ms')))
 for worker,ident in raw['worker_identities'].items():
  wr=[r for r in rs if r['model']==worker]
  if not wr:continue
  tmp=dict(request_records=wr,measurement_window=window,status='completed',errors=raw['errors'],drain_timeout_seconds=60,worker_identities={worker:ident})
  ev=reconstruct(tmp,{worker:SimpleNamespace(service_time_ms=f['plan']['models'][mapping[worker]]['service_time_ms'])},DEFAULT_THRESHOLDS)
  cl=classify(ev,DEFAULT_THRESHOLDS);workers.append(dict(worker=worker,classification=cl,evidence=ev,busy_fraction=sum(r['worker_latency_ms'] for r in wr if r['status']=='completed')/(600*1000)))
 idle=[float(np.mean([v['owned_used_mib'] for v in raw[k]])) for k in ['postwarm_idle','postmeasurement_idle'] if raw.get(k)]
 safe=f['safe_factor']*max(idle) if len(idle)==2 else None;peak=max((v['owned_used_mib'] for v in raw['telemetry']['samples']),default=None)
 memok=safe is not None and safe<=slot['safe_memory_budget_mib'] and peak is not None and peak<=slot['safe_memory_budget_mib']
 hard=ph['stability']['stability_state_v2']=='unstable' or any(w['classification']['stability_state_v2']=='unstable' for w in workers)
 recovery=all(c.get('final_recovery_confirmed',False) if c['stability_state_v2']=='recoverable_congested' else True for c in [ph['stability']]+[w['classification'] for w in workers])
 quality=all(r['task'] in f['plan']['models'][mapping[r['model']]]['supported_tasks'] for r in rs) and not any(r.get('worker_response',{}).get('valid_closed_set_output') is False for r in done)
 trace=read(slot['trajectories']['measurement']['path'])['events']
 native_timings_ok=all(all(r.get('worker_response',{}).get(k) is not None for k in ['start_ns','end_ns','submit_ns']) and r['worker_latency_ms']>=0 and r['worker_queue_wait_ms']>=0 and abs(r['worker_latency_ms']-(r['worker_response']['end_ns']-r['worker_response']['start_ns'])/1e6)<1e-6 and abs(r['worker_queue_wait_ms']-(r['worker_response']['start_ns']-r['worker_response']['submit_ns'])/1e6)<1e-6 for r in done)
 traceok=len(rs)==len(trace) and all(r['task']==e['task'] and r['sample_id']==e['item']['sample'].get('id') and r['pool_id']==e['item']['pool_id'] and r['frozen_offset_seconds']==e['offset_seconds'] and r['routing_uniform']==e['routing_uniform'] and r['replica_uniform']==e['replica_uniform'] for r,e in zip(rs,trace))
 eligible=raw['valid_session'] and audit['passed'] and traceok and native_timings_ok and quality and memok and not hard and recovery and len(done)==len(rs) and set(taskstats)==set(rates)
 within=sum(r['completion_time_ns']<=window['end_time_ns'] for r in done);T=1-min(1.,len(done)/len(rs),within/(sum(rates.values())*600))
 definition,refs=utility_inputs();norms=components(taskstats,rates,refs,T,safe/slot['safe_memory_budget_mib'],definition['weights']) if len(taskstats)==len(rates) and safe is not None else {'U':None}
 # Independent original raw arithmetic, without observation/predictor fitting.
 from module26.common import arithmetic
 check=arithmetic(raw,read(OUT/'selected_pool_catalog.json'),refs,definition['weights'])
 if norms['U'] is not None:assert abs(norms['U']-check['raw_arithmetic_U'])<1e-12
 times=[v for v in raw['telemetry']['samples'] if v['phase']=='measurement' and window['start_time_ns']<=v['time_ns']<=window['end_time_ns']]
 def sensorstat(key):return stats(times,key)
 target=[r for r in done if r['task']==slot['target_task']];g=sum(mapping[r['model']]==slot['Generalist'] for r in target);s=len(target)-g
 out.update(eligible=bool(eligible),U=norms['U'] if eligible else None,components={**norms,'U':norms['U'] if eligible else None},
  diagnostic_completed_subset_U_NOT_sortable=norms['U'] if not eligible else None,per_task=per,paths=paths,workers=workers,
  stability=ph['stability']['stability_state_v2'],hard_unstable=hard,recovery=recovery,quality_admitted=quality,matched_trace=traceok,native_timings_consistent=native_timings_ok,
  offered=len(rs),completed=len(done),within=within,offered_qps=len(rs)/600,completed_qps=len(done)/600,within_qps=within/600,
  G_completed=g,S_completed=s,G_fraction=g/len(target) if target else None,
  G_offered=sum(r['task']==slot['target_task'] and mapping[r['model']]==slot['Generalist'] for r in rs),
  S_offered=sum(r['task']==slot['target_task'] and mapping[r['model']]==slot['Specialist'] for r in rs),
  G_offered_fraction=sum(r['task']==slot['target_task'] and mapping[r['model']]==slot['Generalist'] for r in rs)/sum(r['task']==slot['target_task'] for r in rs),safe_memory_mib=safe,sampled_peak_mib=peak,memory_passed=memok,
  pooled_mean_ms=stats(done,'e2e_latency_ms')['mean'],pooled_p95_ms=stats(done,'e2e_latency_ms')['p95'],drain_seconds=window['drain_seconds'],
  temperature=sensorstat('temperature_gpu_c'),sm_clock=sensorstat('sm_clock_mhz'),memory_clock=sensorstat('memory_clock_mhz'),
  postwarm_temperature_mean=float(np.mean([v['temperature_gpu_c'] for v in raw['postwarm_idle']])) if raw.get('postwarm_idle') else None,
  telemetry_samples=len(times),GPU_uuid=raw['baseline_GPU']['uuid'])
 return out

def backlog(raw):
 ph=raw.get('phases',{}).get('measurement') or raw.get('aborted_phase')
 if not ph:return []
 rs=ph['request_records'];window=ph.get('window',ph);start=window.get('start_time_ns',window.get('planned_start_time_ns'));end=window.get('end_time_ns',window.get('planned_end_time_ns'));abort=ph.get('abort_time_ns');out=[]
 for worker in ['system']+sorted(raw['worker_identities']):
  rr=rs if worker=='system' else [r for r in rs if r['model']==worker]
  schedules=sorted(r['scheduled_time_ns'] for r in rr);enqueues=sorted(r['enqueued_time_ns'] for r in rr if r.get('enqueued_time_ns'));submits=sorted(r['actual_submit_time_ns'] for r in rr if r.get('actual_submit_time_ns'));completions=sorted(r['completion_time_ns'] for r in rr if r['status']=='completed')
  for offset in range(601):
   at=start+offset*1000000000
   if abort and at>abort:continue
   done=bisect_right(completions,at)
   out.append(dict(run_id=raw['slot']['run_id'],worker=worker,offset_seconds=offset,offered_backlog=bisect_right(schedules,at)-done,host_backlog=bisect_right(enqueues,at)-done,submitted_backlog=bisect_right(submits,at)-done))
 return out

def compare(raws,obs):
 by={o['run_id']:o for o in obs};results=[];interference=[];taskchanges=[]
 f=read(OUT/'frozen_gpu_manifest.json');_,refs=utility_inputs();weights=read(ROOT/'results/module20_utility_optimizer/utility_definition.json')['weights']
 for cid in ['C1','C2']:
  group=[r for r in raws if r['slot']['candidate']==cid];control=next((r for r in group if r['slot']['arm']=='s_only'),None);oo=[by[r['slot']['run_id']] for r in group]
  alllegal=len(oo)==3 and all(o['eligible'] for o in oo)
  for arm in ['split05','split10']:
   raw=next((r for r in group if r['slot']['arm']==arm),None)
   result=dict(candidate=cid,arm=arm,status='NOT_EXECUTED' if raw is None else 'NO_LEGAL_COMPARATOR',delta_U=None,all_three_legal_stable=alllegal,positive=False)
   if not raw or not control:results.append(result);continue
   a,b=by[raw['slot']['run_id']],by[control['slot']['run_id']];legal=a['eligible'] and b['eligible']
   if 'temperature' not in a or 'temperature' not in b:results.append(result);continue
   dt=abs(a['temperature']['mean']-b['temperature']['mean']);dw=abs(a['postwarm_temperature_mean']-b['postwarm_temperature_mean']);dc=abs(a['sm_clock']['mean']/b['sm_clock']['mean']-1)
   thermal=dt<=5 and dw<=5 and dc<=.15
   sampleok=a['G_completed']>=20 and a['S_completed']>0 and all(a['per_task'][t]['n_completed']>=72 and b['per_task'][t]['n_completed']>=72 for t in a['per_task'] if t!='asr')
   result.update(status='LEGAL_MATCHED' if legal else 'INELIGIBLE_NULL_U',Generalist=a['Generalist'],Specialist=a['Specialist'],
    G=a['G_completed'],S=a['S_completed'],G_fraction=a['G_fraction'],split_U=a['U'],S_only_U=b['U'],
    delta_U=a['U']-b['U'] if legal else None,temperature_difference_c=dt,postwarm_temperature_difference_c=dw,sm_clock_relative_difference=dc,environment_comparable=thermal,sample_gate=sampleok,
    positive=bool(legal and alllegal and thermal and sampleok and a['U']-b['U']>.01))
   if legal:
    for k in ['T','L','P','M']:result[k+'_delta']=a['components'][k+'_contribution']-b['components'][k+'_contribution']
   if not raw.get('phases',{}).get('measurement') or not control.get('phases',{}).get('measurement'):results.append(result);continue
   aa=raw['phases']['measurement']['request_records'];bb=control['phases']['measurement']['request_records'];lookup={r['request_id']:r for r in bb}
   for task in a['per_task']:
    pa,pb=a['per_task'][task],b['per_task'][task];q=raw['slot']['lambda_rps'][task];total=sum(raw['slot']['lambda_rps'].values())
    taskchanges.append(dict(candidate=cid,arm=arm,task=task,n_split=pa['n_completed'],n_control=pb['n_completed'],
     mean_delta_ms=pa['mean_ms']-pb['mean_ms'] if pa['mean_ms'] is not None and pb['mean_ms'] is not None else None,
     p95_delta_ms=pa['p95_ms']-pb['p95_ms'] if pa['p95_ms'] is not None and pb['p95_ms'] is not None else None,
     mean_U_delta=-weights['L']*q/total*(pa['mean_ms']-pb['mean_ms'])/refs[task] if legal else None,
     p95_U_delta=-weights['P']*q/total*(pa['p95_ms']-pb['p95_ms'])/refs[task] if legal else None,legal=legal))
    for worker in sorted({r['model'] for r in aa if r['task']==task}):
     pairs=[(r,lookup[r['request_id']]) for r in aa if r['task']==task and r['model']==worker and r['status']=='completed' and lookup[r['request_id']]['status']=='completed']
     same=[(r,c) for r,c in pairs if r['model']==c['model']]
     def difference(pp,key):return float(np.mean([r[key]-c[key] for r,c in pp])) if pp else None
     interference.append(dict(candidate=cid,arm=arm,task=task,worker=worker,n=len(pairs),n_same_worker=len(same),
      paired_control_service_mean_ms=float(np.mean([c['worker_latency_ms'] for r,c in same])) if same else None,
      paired_split_service_mean_ms=float(np.mean([r['worker_latency_ms'] for r,c in same])) if same else None,
      paired_service_inflation_ratio=float(np.mean([r['worker_latency_ms'] for r,c in same])/np.mean([c['worker_latency_ms'] for r,c in same])) if same else None,
      paired_service_delta_ms=difference(same,'worker_latency_ms'),paired_wait_delta_ms=difference(same,'worker_queue_wait_ms'),paired_e2e_delta_ms=difference(same,'e2e_latency_ms'),
      diverted_vs_S_control_service_delta_ms=difference([p for p in pairs if p not in same],'worker_latency_ms'),diverted_vs_S_control_wait_delta_ms=difference([p for p in pairs if p not in same],'worker_queue_wait_ms'),diverted_vs_S_control_e2e_delta_ms=difference([p for p in pairs if p not in same],'e2e_latency_ms'),
      legal=legal,scope='Same request inputs; shared-worker overlap measures observed service expansion, not uniquely GPU-causal attribution'))
   results.append(result)
 return results,interference,taskchanges

def interim():
 raws=[read(p) for p in sorted((OUT/'raw_runs').glob('*.json'))];obs=[observation(r) for r in raws]
 save('run_observations.json',obs)
 runs=[];tasks=[];pathrows=[];workerrows=[];environment=[];backlogs=[]
 for raw,o in zip(raws,obs):
  runs.append({k:v for k,v in o.items() if k not in ['per_task','paths','workers','journal_audit','components']})
  for t,v in o.get('per_task',{}).items():tasks.append(dict(run_id=o['run_id'],candidate=o['candidate'],arm=o['arm'],task=t,**v,legal=o['eligible']))
  pathrows.extend(dict(run_id=o['run_id'],**p) for p in o.get('paths',[]))
  workerrows.extend(dict(run_id=o['run_id'],**p) for p in o.get('workers',[]))
  environment.extend(dict(run_id=o['run_id'],**s) for s in raw['telemetry']['samples'])
  backlogs.extend(backlog(raw))
 csvsave('gpu_feasibility_runs.csv',runs,['run_id','candidate','arm','eligible','U']);csvsave('per_task_latency.csv',tasks,['run_id','task']);csvsave('worker_service_wait.csv',pathrows,['run_id','worker','task']);csvsave('worker_stability.csv',workerrows,['run_id','worker']);csvsave('GPU_environment.csv',environment,['run_id']);csvsave('backlog_timeseries.csv',backlogs,['run_id','worker','offset_seconds'])
 comparisons,interference,changes=compare(raws,obs)
 csvsave('system_utility_comparison.csv',comparisons,['candidate','arm','delta_U']);csvsave('worker_interference_analysis.csv',interference,['candidate','arm','task','worker']);csvsave('task_utility_changes.csv',changes,['candidate','arm','task'])
 return raws,obs,comparisons,interference,changes

def integrity():
 from .prepare import inventory
 before=read(OUT/'protected_metadata_before.json');after=inventory();changed=[p for p,meta in before.items() if after.get(p)!=meta];added=[p for p in after if p not in before]
 f=read(OUT/'frozen_gpu_manifest.json');hashchanged=[r['path'] for r in f['source_refs']+f['implementation']+f['reused_implementation'] if ref(r['path'])['sha256']!=r['sha256']]
 save('historical_integrity_after.json',dict(passed=not changed and not added and not hashchanged,files=len(before),changed=changed,added=added,hashchanged=hashchanged,model_weights_preserved=not any('/models/' in p for p in changed)))
 return not changed and not added and not hashchanged

def finalize():
 raws,obs,comparisons,interference,changes=interim();f=read(OUT/'frozen_gpu_manifest.json');campaign=read(OUT/'campaign_completed.json')
 passed=integrity();assert passed,'Historical source changed'
 positives=[p for p in comparisons if p['positive']]
 signals=[p for p in comparisons if p.get('delta_U') is not None and p['delta_U']>.01 and not p['positive']]
 # An unstable result has no legal deltaU; report a qualitative possibility only, never censored gain.
 if positives:gate='EXPLORATORY_POSITIVE_SPLIT'
 elif signals or campaign['stop']:gate='INCONCLUSIVE_POSITIVE_SIGNAL' if signals else 'NO_POSITIVE_FEASIBILITY_CANDIDATE'
 else:gate='NO_POSITIVE_FEASIBILITY_CANDIDATE'
 winner=max(positives,key=lambda p:p['delta_U']) if positives else None
 save('module30_gate.json',dict(module='Module30',gate=gate,infrastructure_passed=True,formal_runs_attempted=len(campaign['attempted']),formal_runs_completed=sum(o['status']=='COMPLETED' for o in obs),
  legal_stable_runs=sum(o['eligible'] for o in obs),candidate_count=2,positive_candidates=positives,limited_legal_positive_signals=signals,winner=winner,
  independent_validation=False,optimizer_modified=False,Stage_D='CPU_ANALYSIS_ONLY' if positives else 'NOT_EXECUTED_NO_EXPLORATORY_POSITIVE',
  stopped=True,campaign=campaign,historical_integrity=passed,negative_results_retained=True,no_seed_retries=True,next_module_automatic=False,
  legal_U_policy='NULL for unstable/incomplete/technical failure; no positive comparison from censored latency'))
 if winner:
  frozen_slot=next(s for s in f['slots'] if s['candidate']==winner['candidate'] and s['arm']==winner['arm'])
  save('exploratory_positive_freeze.json',dict(slot=frozen_slot,manifest=ref(OUT/'frozen_gpu_manifest.json'),winner=winner,service_profiles=[o for o in obs if o['candidate']==winner['candidate']],code=f['implementation'],independent_reproduction_authorized=False))
 lines=['# Module30 正分流物理机制','',f'Gate：{gate}。本轮为开发探索，独立验证未启动。','']
 for p in comparisons:
  lines += [f"{p['candidate']} / {p['arm']}：合法ΔU={p.get('delta_U')}，三臂合法稳定={p['all_three_legal_stable']}，环境可比={p.get('environment_comparable')}。"]
  for r in interference:
   if r['candidate']==p['candidate'] and r['arm']==p['arm'] and r['task'] in ['asr','lid','visual_qa']:
    lines.append(f"- {r['task']} / {r['worker']}：共享输入n={r['n_same_worker']}，执行变化{r['paired_service_delta_ms']}ms，真实等待变化{r['paired_wait_delta_ms']}ms；转交G的输入相对S控制执行变化{r['diverted_vs_S_control_service_delta_ms']}ms，等待变化{r['diverted_vs_S_control_wait_delta_ms']}ms。")
  cc=[r for r in changes if r['candidate']==p['candidate'] and r['arm']==p['arm']]
  lines += ['', '逐任务全系统效用变化：']+[f"- {r['task']}：mean/P95变化{r['mean_delta_ms']}/{r['p95_delta_ms']}ms；效用贡献变化{r['mean_U_delta']}/{r['p95_U_delta']}。" for r in cc]+['']
 lines += ['同输入共享S的执行变化是共置路由改变后观测到的服务膨胀；温度、频率和CPU/NVML同时记录，但没有额外因果隔离实验，不能唯一归因纯GPU竞争。G的native等待为handler-start减host-submit，原协议没有精确worker receipt ACK；host unsent与发送延迟分列，未冒充worker内部队列。', '', '所有完整窗口U按原20A六任务offered-QPS加权mean/P95、固定600秒窗口吞吐损失、safe memory计算；诊断pooled P95不是系统效用。S等待下降本身不构成成功。']
 (OUT/'positive_split_mechanism.md').write_text('\n'.join(lines)+'\n')
 if winner:
  plan='''# 后续优化器改造建议（仅CPU分析）

探索候选、冻结负载和比例见exploratory_positive_freeze.json；独立长窗口复现另行执行，本模块未改正式优化器。

1. 相对此前Qwen+Whisper-medium LID，当前ASR组合改变了G/S执行差距、串行S共享任务需求与排队压力；实际改善证据必须以positive_split_mechanism.md的S等待/服务和G代价及背景分项为准。旧LID的S约69ms而G约数百ms，历史分流增加S服务且未降低等待；模型名不进入判据。
2. 正收益来自各task_utility_changes分项的实测和。S排队下降、S服务膨胀、G新增执行与排队、VQA及其它背景mean/P95分开归因；不能从单个S均值推定系统收益。
3. 可在线观察：已到达未完成/host待发送、最新已收到RESULT服务分布、worker繁忙区间、S后半队列斜率、G残余需求、背景tail、显存/温度/时钟与最近路由变更后的S服务膨胀。当前协议无START/receipt ACK，应使用保守区间而不是准确内部队列假设。
4. 模型无关边际判据：对候选route质量及内存硬准入后，估计ΔU=-wTΔT-Σ_t(q_t/Σq)[wLΔmean_t/ref_t+wPΔP95_t/ref_t]-wMΔM。服务/等待从共置上下文实时条件分布估计，含新任务对其它worker的外部代价和不确定性；用保守收益下界大于+0.01再考虑切换，不用固定模型名或固定QPS规则。ρ接近1/队列增长/缺画像时拒绝认证。
5. 候选生成同时保留纯S、纯G、混合和已实测副本配置；在每个驻留状态内自主搜索分流比例，约束G残余工作与S服务膨胀，保留0%作为基线；通过滞回、最小驻留时间与有界探索更新比例。正式accept/rollback依据真实全系统U及稳定性，不能靠旧J的等待代理强制分流。需要独立复现、跨模型/负载留出与在线特征因果识别后再实施。
'''
 else:
  plan='''# 后续优化器集成状态

Stage D未执行：没有符合全对照合法稳定且ΔU>+0.01的探索性正组合。正式部署优化器未修改。当前负结果不足以证实通用新边际判据；不进入算法集成、不自动启动独立复现或下一Module。保存冻结模型池、六任务负载、比例、代码与全部服务/队列/背景损失以供后续独立研究。
'''
 (OUT/'optimizer_integration_plan.md').write_text(plan)
 table=['|组合|方案|合法U|ΔU vs S-only|稳定/合法|','|---|---|---:|---:|---|']
 for o in obs:
  p=next((p for p in comparisons if p['candidate']==o['candidate'] and p['arm']==o['arm']),None)
  table.append(f"|{o['Generalist']} + whisper-small|{o['arm']}|{o['U']}|{p.get('delta_U') if p else '基线'}|{o['stability']} / {o['eligible']}|")
 best=max((p for p in comparisons if p.get('delta_U') is not None),key=lambda p:p['delta_U'],default=None)
 bg=[r for r in changes if r['legal'] and r['task']!='asr'];worst=min(bg,key=lambda r:r['mean_U_delta']+r['p95_U_delta'],default=None)
 summary=['# Module30 GPU-Measured G/S Positive-Split Feasibility Screening','',f'Gate：**{gate}**。已停止；不自动进入下一Module或独立GPU复现。','',
 'Stage0：2100秒CPU并发压力、传感器/写盘/退出故障注入及200请求原生运输fixture通过；Module29错误、原代码及Gate保持不变。新遥测通过runtime.lock快照与串行采样修复并发冲突，日志与raw可逐请求核对。','',
 'StageA：39个已准入G/S共同任务配对排名；最多两个组合，Qwen+Whisper-small与Phi+Whisper-small ASR。两组五模型各1副本；Text/Image/OCR/LID/VQA各0.2QPS，ASR1QPS；显存34567.5534MiB；0/5/10%比例、样本、轨迹和顺序GPU前冻结。历史负收益和服务漂移保留，无共置预测倍率。','',
 f"StageB：尝试{len(campaign['attempted'])}/6，完整{sum(o['status']=='COMPLETED' for o in obs)}，合法稳定{sum(o['eligible'] for o in obs)}。每次600秒，固定30秒预热、60秒有界排空、热状态控制。未运行槽和技术停止见campaign_completed.json。",'',*table,'',
 '最终六问：',f"1. 测前优先级第一是Qwen+Whisper-small ASR；本轮实测最佳合法差值：{best}。优先级不是预言正收益。",
 f"2. 实际观测到合格G/S优于S-only：{bool(positives)}。有限制的合法正信号：{signals}。不稳定U为NULL，未用截尾延迟计算收益。",
 '3. 物理变化逐组合见positive_split_mechanism.md与worker_interference_analysis.csv：S同输入服务/等待、G额外执行与等待、背景mean/P95及最终U均分列；不把服务变化唯一归因纯GPU。',
 f"4. 背景任务最大负效用贡献：{worst}；全部六任务和有效n见per_task_latency.csv/task_utility_changes.csv。",
 f"5. 值得另行独立长窗口验证的合格候选：{winner if winner else '无'}；本轮只提供开发证据，独立验证未自动启动。",
 '6. 若无正收益，应结合实际S/G性能差距、共享S服务膨胀、VQA等待和S-only队列压力判别，不能把一个负载点推为所有模型/负载不存在。对本轮这些因素的数值分解已完整保存，StageD无正候选时不执行。','',
 '局限：每组合只有一条配对轨迹、三个600秒开发窗口；未证明长时域稳定或论文独立复现。G样本仅21–53左右，尾部估计仍有限；只测两个固定比例/一个负载点，不宣称全局最优。所有已准入质量门槛、20A定义/参考与原stability_v2阈值未改。', '',
 f"完整性：{read(OUT/'historical_integrity_after.json')['files']}个历史文件/模型权重元数据和冻结科学源码hash保持不变。正式优化器未修改。"]
 (OUT/'module30_summary.md').write_text('\n'.join(summary)+'\n')
if __name__=='__main__':finalize()
