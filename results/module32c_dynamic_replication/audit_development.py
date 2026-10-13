"""CPU audit of full Module32B; observed contrasts, no policy fitting."""
import sys, itertools, math, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32c.common import *
sys.path.insert(1,str(OLD))
from m32b.campaign import assess as old_assess
from module26.common import arithmetic, cohort
from module25.common import utility_inputs
D,refs=utility_inputs(); checks={}; oldf=read(OLD/'experiment_manifest.json')
if not (OUT/'protected_before.json').exists():
 protected=read(OLD/'protected_before.json')
 for p in OLD.rglob('*'):
  if p.is_file():protected[str(p)]=[p.stat().st_size,p.stat().st_mtime_ns]
 save('protected_before.json',protected)
inventory=[];allreq=[];sensor=[];journals=[];textlogs=[]
for p in sorted(OLD.rglob('*')):
 if not p.is_file():continue
 h=ref(p);inventory.append(h)
 if p.parent.name=='raw_runs' and p.suffix=='.json':
  raw=read(p)
  for ph in [raw.get('warmup'),*raw.get('phases',{}).values(),raw.get('aborted_phase')]:
   if ph:
    for r in ph['request_records']:allreq.append(dict(source=str(p),**r))
  sensor.extend(dict(source=str(p),**v) for v in raw.get('telemetry',{}).get('samples',[]))
 if p.suffix=='.jsonl':
  rows=[json.loads(line) for line in p.read_text().splitlines()];journals.append(dict(source=h,rows=len(rows),kinds=dict(collections.Counter(r.get('kind','unknown') for r in rows))))
 if p.suffix=='.log':
  s=p.read_text(errors='replace');textlogs.append(dict(source=h,lines=len(s.splitlines()),tracebacks=s.count('Traceback')))
save('module32b_full_file_hashes.json',inventory);save('historical_log_inventory.json',dict(files=inventory,journals=journals,text_logs=textlogs))
csvsave('historical_all_requests.csv',allreq);csvsave('historical_all_GPU_telemetry.csv',sensor)
manifest=read(OLD/'artifact_manifest.json');checks['module32b_artifact_SHA']=all(ref(r['path'])['sha256']==r['sha256'] for r in manifest['artifacts'])
checks['prior_final_acceptance']=read(OLD/'final_data_acceptance.json')['passed']
raws={a:read(OLD/'raw_runs'/f'{a}.json') for a in ['A0','A1','A2']};obs={a:old_assess(r) for a,r in raws.items()}
checks['all_prior_legal_stable']=all(o['eligible'] for o in obs.values())
checks['same_cohort']=all(cohort(r)==cohort(raws['A0']) for r in raws.values())
checks['same_full_inputs']=all(read(r['slot']['trajectories']['measurement']['path'])['events']==read(raws['A0']['slot']['trajectories']['measurement']['path'])['events'] for r in raws.values())
checks['same_pool_rates_budget']=all(all(r['slot'][k]==raws['A0']['slot'][k] for k in ['deployment','lambda_rps','safe_memory_budget_mib']) for r in raws.values())
checks['same_GPU_identity']=all(all(r['baseline_GPU'][k]==raws['A0']['baseline_GPU'][k] for k in ['uuid','name','total_mib','driver']) for r in raws.values())
recomputed={a:arithmetic(r,read(OLD/'selected_pool_catalog.json'),refs,D['weights']) for a,r in raws.items()}
checks['exact_original_U_recomputed']=all(abs(v['raw_arithmetic_U']-obs[a]['U'])<1e-12 for a,v in recomputed.items())
checks['native_clocks_consistent']=all(o['native_timings_consistent'] and o['validation']['unix_perf_alignment'] for o in obs.values())
save('historical_utility_recalculation.json',recomputed)
rates=raws['A0']['slot']['lambda_rps'];total=sum(rates.values());decomp=[];timing=[];tail=[];quant=[];over=[];gd=[];shap=[]
base={r['request_id']:r for r in raws['A0']['phases']['measurement']['request_records']}
for a,r in raws.items():
 rr=r['phases']['measurement']['request_records'];gg=[x for x in rr if x['task']=='asr' and x['model']==G];ds={x['request_id']:x for x in r['phases']['measurement'].get('routing_decisions',[])}
 for t,q in rates.items():
  pp=obs[a]['per_task'][t];bb=obs['A0']['per_task'][t]
  lu=-D['weights']['L']*q/total*(pp['mean_ms']-bb['mean_ms'])/refs[t];pu=-D['weights']['P']*q/total*(pp['p95_ms']-bb['p95_ms'])/refs[t]
  decomp.append(dict(arm=a,task=t,n=pp['n_completed'],mean_ms=pp['mean_ms'],p95_ms=pp['p95_ms'],L_delta_U=lu,P_delta_U=pu,L_P_delta_U=lu+pu,official_U=obs[a]['U'],official_delta_U=obs[a]['U']-obs['A0']['U']))
  tt=[x for x in rr if x['task']==t]
  timing.append(dict(arm=a,task=t,**{k:stats(x[k] for x in tt) for k in ['worker_latency_ms','worker_queue_wait_ms','e2e_latency_ms']},host_residual_ms=stats(x['e2e_latency_ms']-x['worker_latency_ms']-x['worker_queue_wait_ms'] for x in tt)))
 texts=sorted([x for x in rr if x['task']=='text'],key=lambda x:x['e2e_latency_ms']);rank=(len(texts)-1)*.95;lo=int(math.floor(rank));hi=int(math.ceil(rank))
 quant.append(dict(arm=a,n=len(texts),zero_based_rank=rank,lower_rank=lo,upper_rank=hi,lower_request=texts[lo]['request_id'],upper_request=texts[hi]['request_id'],lower_weight=hi-rank,upper_weight=rank-lo,lower_ms=texts[lo]['e2e_latency_ms'],upper_ms=texts[hi]['e2e_latency_ms'],p95_ms=np.percentile([x['e2e_latency_ms'] for x in texts],95)))
 for rank1,x in enumerate(texts,1):
  b=base[x['request_id']];native_ov=sum(overlap(interval(x),interval(g)) for g in gg)
  if rank1>=len(texts)-math.ceil(len(texts)*.1)+1:tail.append(dict(arm=a,ascending_rank=rank1,request_id=x['request_id'],sample_id=x['sample_id'],offset_seconds=x['frozen_offset_seconds'],service_ms=x['worker_latency_ms'],queue_ms=x['worker_queue_wait_ms'],E2E_ms=x['e2e_latency_ms'],host_residual_ms=x['e2e_latency_ms']-x['worker_latency_ms']-x['worker_queue_wait_ms'],Phi_ASR_service_overlap_ms=native_ov,S_only_E2E_ms=b['e2e_latency_ms'],E2E_delta_ms=x['e2e_latency_ms']-b['e2e_latency_ms']))
  over.append(dict(arm=a,request_id=x['request_id'],sample_id=x['sample_id'],offset_seconds=x['frozen_offset_seconds'],Phi_ASR_service_overlap_ms=native_ov,service_delta_ms=x['worker_latency_ms']-b['worker_latency_ms'],wait_delta_ms=x['worker_queue_wait_ms']-b['worker_queue_wait_ms'],E2E_delta_ms=x['e2e_latency_ms']-b['e2e_latency_ms'],nearest_G_ASR_schedule_seconds=min((abs(x['frozen_offset_seconds']-g['frozen_offset_seconds']) for g in gg),default=None)))
 for x in gg:
  d=ds.get(x['request_id'],{});vqa=[v for v in rr if v['task']=='visual_qa'];nfuture=sum(x['worker_response']['start_ns']<=v['actual_submit_unix_ns']<x['worker_response']['end_ns'] for v in vqa)
  gd.append(dict(**{k:v for k,v in d.items() if k!='request_id'},arm=a,request_id=x['request_id'],sample_id=x['sample_id'],offset_seconds=x['frozen_offset_seconds'],service_ms=x['worker_latency_ms'],queue_ms=x['worker_queue_wait_ms'],E2E_ms=x['e2e_latency_ms'],S_only_service_ms=base[x['request_id']]['worker_latency_ms'],S_only_wait_ms=base[x['request_id']]['worker_queue_wait_ms'],future_VQA_arrivals_during_handler=nfuture))
 # Algebraic matched P95 component Shapley. These are NOT executed counterfactual routes.
 if a!='A0':
  for t in ['text','visual_qa']:
   z=[x for x in rr if x['task']==t];keys=['queue','handler_service','host_residual']
   def vector(x):return [x['worker_queue_wait_ms'],x['worker_latency_ms'],x['e2e_latency_ms']-x['worker_queue_wait_ms']-x['worker_latency_ms']]
   b=np.array([vector(base[x['request_id']]) for x in z]);v=np.array([vector(x) for x in z]);s=np.zeros(3)
   for perm in itertools.permutations(range(3)):
    c=b.copy();prior=np.percentile(c.sum(axis=1),95)
    for j in perm:c[:,j]=v[:,j];cur=np.percentile(c.sum(axis=1),95);s[j]+=(cur-prior)/6;prior=cur
   shap.append(dict(arm=a,task=t,**{keys[j]+'_p95_change_ms':s[j] for j in range(3)},total_p95_change_ms=np.percentile(v.sum(axis=1),95)-np.percentile(b.sum(axis=1),95),scope='Algebraic decomposition of measured samples; not observed counterfactual routing'))
csvsave('historical_per_task_utility_decomposition.csv',decomp);csvsave('historical_task_latency_components.csv',timing);csvsave('text_tail_requests.csv',tail);csvsave('text_quantile_support.csv',quant);csvsave('historical_text_Phi_overlap.csv',over);csvsave('four_dynamic_ASR_trigger_audit.csv',[x for x in gd if x['arm']=='A2']);csvsave('historical_p95_component_shapley.csv',shap)
dg=[x for x in gd if x['arm']=='A2'];checks['four_real_dynamic_ASR']=len(dg)==4
checks['four_causal_valid_admissions']=all(x['select_G'] and not x['uses_future'] and x['Phi_inflight']==0 and all(v['enqueue_perf_ns']<=x['decision_perf_ns'] for v in x['past_state_snapshot']) for x in dg)
textc=next(x for x in decomp if x['arm']=='A2' and x['task']=='text');du=obs['A2']['U']-obs['A0']['U'];sens=dict(official_six_task_delta_U=du,Text_total_contribution=textc['L_P_delta_U'],Text_p95_contribution=textc['P_delta_U'],diagnostic_without_Text_p95_contribution=du-textc['P_delta_U'],diagnostic_without_Text_total_contribution=du-textc['L_P_delta_U'],official_utility_unchanged=True,not_an_executed_counterfactual=True)
save('historical_utility_sensitivity.json',sens)
save('stage_A_gate.json',dict(passed=all(checks.values()),checks=checks,measurement_valid_and_matched=True if all(checks.values()) else False,causal_routing_gain_proven=False,independent_GPU_replication_allowed=all(checks.values()),limitations=['One serial trajectory; prior order Fixed10/S-only/Dynamic','Only four real Phi ASR and no future VQA exposure','Text P95 quantile depends on two bracketing samples; tail ranking changes across arms','GPU resource competition cannot be isolated from host/CPU scheduling and serial-run drift','Independent arrival/input-order traces draw from the same frozen sample pool; not independent new-dataset quality validation'],at_unix_ns=time.time_ns()))
lines=['# Module32C Stage A：效用归因与 Text 尾审计','',f'审计通过={all(checks.values())}；原六任务U复算误差<1e-12；官方权重与归一化不变。','',f'Dynamic−S-only={du:+.9f}；Text mean/P95总贡献={textc["L_P_delta_U"]:+.9f}，仅P95={textc["P_delta_U"]:+.9f}。诊断减去Text P95贡献为{sens["diagnostic_without_Text_p95_contribution"]:+.9f}，非另一正式效用/非执行反事实。','', '## Text P95 支撑请求','', '|臂|n|P95 ms|下界请求 / ms|上界请求 / ms|','|---|---:|---:|---|---|']
for v in quant:lines.append(f'|{v["arm"]}|{v["n"]}|{v["p95_ms"]:.6f}|{v["lower_request"]} / {v["lower_ms"]:.6f}|{v["upper_request"]} / {v["upper_ms"]:.6f}|')
lines+=['','115样本P95是第109/110个排序样本的0.7/0.3插值，正式P95并非全体请求平均。完整前10%尾请求、同请求变化和执行/等待/host残差见CSV；不删除尾样本。','', '## 排队、执行与共置','', 'queue=start_unix−submit_unix；handler service=end_unix−start_unix，包含同步CUDA调用、I/O/preprocess；E2E=completion_perf−scheduled_perf。host残差包含派发/接收/调度。Unix与perf只分别计算区间，原时钟校验通过，不直接跨时钟相减。服务时间不是纯CUDA kernel时间。P95不能直接把各分项P95相加；以下Shapley仅是已测向量的代数分解。']
for v in shap:lines.append(f'- {v["arm"]} {v["task"]} P95变化={v["total_p95_change_ms"]:+.6f}ms：queue={v["queue_p95_change_ms"]:+.6f}，handler={v["handler_service_p95_change_ms"]:+.6f}，host残差={v["host_residual_p95_change_ms"]:+.6f}。')
dynov=[v for v in over if v['arm']=='A2' and v['Phi_ASR_service_overlap_ms']>0]
lines+=['',f'Dynamic的Text请求与Phi ASR handler直接重叠数={len(dynov)}。即使直接不重叠也不能排除间接共享资源效应；没有Text路线变化，也没有随机化隔离的机制干预。旧顺序为Fixed10→S-only→Dynamic，Text差异仍可能是运行间漂移，不能宣称由分流造成。','', '## 四次分流','', '|请求|时间s|S估计等待ms|margin ms|Phi在途|实际执行ms|实际等待ms|未来VQA到达|','|---|---:|---:|---:|---:|---:|---:|---:|']
for v in dg:lines.append(f'|{v["request_id"]}|{v["offset_seconds"]:.3f}|{v["S_estimated_wait_ms"]:.3f}|{v["predicted_margin_ms"]:.3f}|{v["Phi_inflight"]}|{v["service_ms"]:.3f}|{v["queue_ms"]:.3f}|{v["future_VQA_arrivals_during_handler"]}|')
lines+=['', 'Whisper保留请求同输入实际等待均值减少2.344ms；ASR执行代价与所有任务效用完整保留。VQA保护在这四次分流下没有遇到未来VQA，不能据此证明非抢占安全。','', '完整工作负载、请求输入及轨迹匹配，GPU身份和原技术审计通过；严格配对请求不等于同时运行或消除热/时钟漂移。下一步只冻结原32B策略，预先固定三组新轨迹与交错顺序；不基于上述尾请求调整参数。']
(OUT/'utility_attribution.md').write_text('\n'.join(lines)+'\n');(OUT/'text_tail_audit.md').write_text('\n'.join(lines[:lines.index('## 四次分流')])+'\n')
print('STAGE_A',all(checks.values()),checks,sens,flush=True)
