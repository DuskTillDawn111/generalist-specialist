"""Final independent evidence, original six-task utility, no tuning or optimizer edit."""
from m32g.isolation import install
install('supervisor')
from m32g.common import *
from m32g.acceptance import assess
from module20.optimizer import components
from module26.common import arithmetic,cohort
from bisect import bisect_right
import subprocess

def main():
 f=read(OUT/'frozen_validation_manifest.json');history=read(OUT/'history_hashes/before_regression_vs_after_campaign.json')
 source_bad=[x['path'] for x in f['sources']+f['frozen_inputs'] if ref(x['path'])['sha256']!=x['sha256']]
 env={}
 for py,prior in f['environment'].items():
  actual=json.loads(subprocess.check_output([py,'-c','import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'],text=True));env[py]=dict(unchanged=actual==prior,current=actual)
 audit=dict(passed=history['passed'] and not source_bad and all(x['unchanged'] for x in env.values()),history=history,source_mismatches=source_bad,dependencies=env)
 save('final_data_acceptance.json',audit)
 ctx=read(OUT/'frozen_utility_context.json');D=ctx['definition'];refs=ctx['refs'];rates=ctx['rates'];total=sum(rates.values())
 assert D==read(ROOT/'results/module20_utility_optimizer/utility_definition.json')
 assert refs==read(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json')['latency_reference_ms']
 raws={};obs={};per=[];paths=[];sensors=[];controls=[];decisions=[];quant=[];tails=[];vqas=[];back=[];requests=[];resources=[]
 for slot in f['slots']:
  rid=slot['run_id'];path=OUT/'raw_runs'/f'{rid}.json'
  if not path.exists():continue
  r=read(path);raws[rid]=r;o=assess(r,f);obs[rid]=o
  sensors.extend(dict(run_id=rid,**s) for s in r['telemetry']['samples']);controls.extend(dict(run_id=rid,**s) for s in r['control_probes'])
  ph=r.get('phases',{}).get('measurement')
  if not ph:continue
  rr=ph['request_records'];w=ph['window'];requests.extend(dict(run_id=rid,**x) for x in rr);ds=ph['routing_decisions'];decisions.extend(dict(run_id=rid,**d) for d in ds)
  ss=[s for s in r['telemetry']['samples'] if w['start_time_ns']<=s['time_ns']<=w['end_time_ns']]
  resources.append(dict(run_id=rid,**{k:stats(s[k] for s in ss) for k in ['temperature_gpu_c','sm_clock_mhz','memory_clock_mhz','owned_used_mib','power_draw_w','gpu_util_percent','worker_CPU_core_percent']},router_ms=stats(d['routing_compute_ms'] for d in ds),control_check_max_ms=r['control_latency_seconds']['max_check']*1000,control_publish_max_ms=r['control_latency_seconds']['max_publish']*1000,hw_thermal_active=sum(s['hw_thermal_slowdown']=='Active' for s in ss),sw_thermal_active=sum(s['sw_thermal_slowdown']=='Active' for s in ss)))
  done=[x for x in rr if x['status']=='completed'];ga=[x for x in done if x['task']=='asr' and x['model']==G]
  for t,q in rates.items():
   offered=[x for x in rr if x['task']==t];z=[x for x in done if x['task']==t]
   per.append(dict(pair=slot['pair'],run_id=rid,task=t,n_offered=len(offered),n_completed=len(z),offered_qps=len(offered)/2400,completed_qps=len(z)/2400,within_qps=sum(x['completion_time_ns']<=w['end_time_ns'] for x in z)/2400,mean_ms=stats(x['e2e_latency_ms'] for x in z)['mean'],p95_ms=stats(x['e2e_latency_ms'] for x in z)['p95'],service=stats(x['worker_latency_ms'] for x in z),wait=stats(x['worker_queue_wait_ms'] for x in z)))
   for model in sorted({x['model'] for x in offered}):
    zz=[x for x in z if x['model']==model];paths.append(dict(run_id=rid,task=t,worker=model,n_completed=len(zz),service=stats(x['worker_latency_ms'] for x in zz),wait=stats(x['worker_queue_wait_ms'] for x in zz),e2e=stats(x['e2e_latency_ms'] for x in zz)))
   scheduled=sorted(x['scheduled_time_ns'] for x in offered);completed=sorted(x['completion_time_ns'] for x in z)
   for off in range(2401):at=w['start_time_ns']+off*10**9;back.append(dict(run_id=rid,group='task:'+t,offset_seconds=off,offered_backlog=bisect_right(scheduled,at)-bisect_right(completed,at)))
   if t in ['text','visual_qa'] and z:
    ordered=sorted(z,key=lambda x:x['e2e_latency_ms']);rank=.95*(len(z)-1);lo=math.floor(rank);hi=math.ceil(rank)
    quant.append(dict(run_id=rid,task=t,n=len(z),rank=rank,lower_request=ordered[lo]['request_id'],upper_request=ordered[hi]['request_id'],lower_ms=ordered[lo]['e2e_latency_ms'],upper_ms=ordered[hi]['e2e_latency_ms'],upper_weight=rank-lo,P95_ms=float(np.percentile([x['e2e_latency_ms'] for x in z],95))))
    for i,x in enumerate(ordered):
     if i>=len(z)-math.ceil(.1*len(z)):tails.append(dict(run_id=rid,task=t,request_id=x['request_id'],sample_id=x['sample_id'],offset_seconds=x['frozen_offset_seconds'],e2e_ms=x['e2e_latency_ms'],service_ms=x['worker_latency_ms'],queue_ms=x['worker_queue_wait_ms'],host_residual_ms=x['e2e_latency_ms']-x['worker_latency_ms']-x['worker_queue_wait_ms'],Phi_ASR_overlap_service_ms=sum(overlap(interval(x),interval(g)) for g in ga)))
  for x in done:
   if x['task']=='visual_qa':vqas.append(dict(run_id=rid,request_id=x['request_id'],sample_id=x['sample_id'],e2e_ms=x['e2e_latency_ms'],service_ms=x['worker_latency_ms'],queue_ms=x['worker_queue_wait_ms'],Phi_ASR_overlap_wait_ms=sum(overlap((x['worker_response']['submit_ns'],x['worker_response']['start_ns']),interval(g)) for g in ga),simultaneous_native_Phi_execution_ms=sum(overlap(interval(x),interval(g)) for g in ga)))
  for worker in ['system']+sorted(r['worker_identities']):
   wr=rr if worker=='system' else [x for x in rr if x['model']==worker];arrival=sorted(x['enqueued_time_ns'] for x in wr if x.get('enqueued_time_ns'));sub=sorted(x['actual_submit_time_ns'] for x in wr if x.get('actual_submit_time_ns'));end=sorted(x['completion_time_ns'] for x in wr if x['status']=='completed')
   for off in range(2401):at=w['start_time_ns']+off*10**9;n=bisect_right(end,at);back.append(dict(run_id=rid,group=worker,offset_seconds=off,host_backlog=bisect_right(arrival,at)-n,submitted_backlog=bisect_right(sub,at)-n))
 pairs=[];contrib=[];paired=[];asr=[];triggers=[];sensitivity=[];utilities={};unit=[]
 for pair in [1,2,3]:
  sid=f'P{pair}_S';did=f'P{pair}_D';a=obs.get(sid);b=obs.get(did);e=OUT/'environment_audits'/f'P{pair}_paired.json';envok=e.exists() and read(e)['passed']
  full=bool(a and b and raws[sid].get('phases',{}).get('measurement') and raws[did].get('phases',{}).get('measurement'))
  matched=bool(full and cohort(raws[sid])==cohort(raws[did]));legal=bool(a and b and a['eligible'] and b['eligible'] and envok and matched and audit['passed'])
  real=bool(b and b['details'].get('G_ASR',0)>0 and b['details'].get('S_ASR',0)>0)
  for rid in [sid,did]:
   utilities[rid]=None
   if not legal:continue
   r=raws[rid];ph=r['phases']['measurement'];rr=ph['request_records'];done=[x for x in rr if x['status']=='completed'];taskstats={t:dict(mean_ms=stats(x['e2e_latency_ms'] for x in done if x['task']==t)['mean'],p95_ms=stats(x['e2e_latency_ms'] for x in done if x['task']==t)['p95']) for t in rates}
   within=sum(x['completion_time_ns']<=ph['window']['end_time_ns'] for x in done);T=1-min(1.,len(done)/len(rr),within/(total*2400))
   safe=f['safe_factor']*max(float(np.mean([x['owned_used_mib'] for x in r[k]])) for k in ['postwarm_idle','postmeasurement_idle']);c=components(taskstats,rates,refs,T,safe/r['slot']['safe_memory_budget_mib'],D['weights']);ind=arithmetic(r,read(OUT/'selected_pool_catalog.json'),refs,D['weights']);assert abs(c['U']-ind['raw_arithmetic_U'])<1e-12
   utilities[rid]=c['U'];save(f'utility_per_run/{rid}.json',dict(legal=True,components=c,independent_arithmetic=ind,original_definition=True))
   unit.append(dict(pair=pair,run_id=rid,U=c['U'],**{k:v for k,v in c.items() if k!='U'}))
  delta=utilities[did]-utilities[sid] if legal else None
  pairs.append(dict(pair=pair,seed=f['trace_independence']['seeds_chosen_once'][pair-1],matched=matched,environment_passed=envok,both_legal_stable=legal,real_G_and_S=real,S_U=utilities[sid],Dynamic_U=utilities[did],delta_U=delta,G_ASR=b['details'].get('G_ASR') if b else None,S_ASR=b['details'].get('S_ASR') if b else None,G_fraction=(b['details']['G_ASR']/(b['details']['G_ASR']+b['details']['S_ASR'])) if b and b['details'].get('S_ASR') else None,status='VALID' if legal else 'INVALID_OR_NOT_EXECUTED'))
  if not(a and b and raws[sid].get('phases',{}).get('measurement') and raws[did].get('phases',{}).get('measurement')):continue
  ar=raws[sid]['phases']['measurement']['request_records'];br=raws[did]['phases']['measurement']['request_records'];base={x['request_id']:x for x in ar};ga=[x for x in br if x['task']=='asr' and x['model']==G and x['status']=='completed'];ss=[x for x in br if x['task']=='asr' and x['model']==S and x['status']=='completed'];ds={x['request_id']:x for x in raws[did]['phases']['measurement']['routing_decisions']}
  for t in rates:
   aa=next(x for x in per if x['run_id']==sid and x['task']==t);bb=next(x for x in per if x['run_id']==did and x['task']==t)
   L=-D['weights']['L']*rates[t]/total*(bb['mean_ms']-aa['mean_ms'])/refs[t] if legal else None;P=-D['weights']['P']*rates[t]/total*(bb['p95_ms']-aa['p95_ms'])/refs[t] if legal else None
   contrib.append(dict(pair=pair,task=t,legal_pair=legal,n_S=aa['n_completed'],n_D=bb['n_completed'],S_mean_ms=aa['mean_ms'],D_mean_ms=bb['mean_ms'],S_P95_ms=aa['p95_ms'],D_P95_ms=bb['p95_ms'],mean_contribution=L,P95_contribution=P,total_latency_contribution=L+P if legal else None))
  for x in br:
   c=base.get(x['request_id'])
   if x['status']!='completed' or not c or c['status']!='completed':continue
   paired.append(dict(pair=pair,request_id=x['request_id'],task=x['task'],sample_id=x['sample_id'],offset_seconds=x['frozen_offset_seconds'],S_model=c['model'],D_model=x['model'],E2E_delta_ms=x['e2e_latency_ms']-c['e2e_latency_ms'],service_delta_ms=x['worker_latency_ms']-c['worker_latency_ms'],wait_delta_ms=x['worker_queue_wait_ms']-c['worker_queue_wait_ms'],Phi_ASR_overlap_service_ms=sum(overlap(interval(x),interval(g)) for g in ga),legal_pair=legal))
  def diff(z,key):return sum(x[key]-base[x['request_id']][key] for x in z if x['request_id'] in base and base[x['request_id']]['status']=='completed')
  asr.append(dict(pair=pair,legal_pair=legal,retained_S_n=len(ss),diverted_G_n=len(ga),Whisper_wait_reduction_mean_ms=-diff(ss,'worker_queue_wait_ms')/len(ss) if ss else None,Whisper_wait_saved_seconds=-diff(ss,'worker_queue_wait_ms')/1000,Whisper_service_delta_mean_ms=diff(ss,'worker_latency_ms')/len(ss) if ss else None,Phi_absolute_service_seconds=sum(x['worker_latency_ms'] for x in ga)/1000,Phi_wait_seconds=sum(x['worker_queue_wait_ms'] for x in ga)/1000,Phi_extra_service_vs_matching_S_seconds=diff(ga,'worker_latency_ms')/1000))
  for x in ga:
   future=[v for v in br if v['task']=='visual_qa' and v.get('actual_submit_unix_ns') and x['worker_response']['start_ns']<=v['actual_submit_unix_ns']<x['worker_response']['end_ns']]
   triggers.append(dict(pair=pair,**ds[x['request_id']],sample_id=x['sample_id'],offset_seconds=x['frozen_offset_seconds'],actual_service_ms=x['worker_latency_ms'],actual_wait_ms=x['worker_queue_wait_ms'],actual_E2E_ms=x['e2e_latency_ms'],VQA_arrivals_during_handler=len(future),posthoc_only_future_count=True))
  if legal:
   t=next(x for x in contrib if x['pair']==pair and x['task']=='text');sensitivity.append(dict(pair=pair,official_delta_U=delta,Text_mean_contribution=t['mean_contribution'],Text_P95_contribution=t['P95_contribution'],diagnostic_delta_without_Text_P95=delta-t['P95_contribution'],NOT_formal_U=True,NOT_executed_counterfactual=True))
 runrows=[]
 for slot in f['slots']:
  rid=slot['run_id'];o=obs.get(rid);d=o['details'] if o else {};runrows.append(dict(pair=slot['pair'],run_id=rid,arm=slot['arm'],seed=slot['seed'],U=utilities.get(rid),status='COMPLETED' if rid in raws and raws[rid]['valid_session'] else 'FAILED' if rid in raws else 'NOT_EXECUTED',technical_valid=o['technical_valid'] if o else False,eligible=o['eligible'] if o else False,offered=d.get('offered'),completed=d.get('completed'),G_ASR=d.get('G_ASR'),S_ASR=d.get('S_ASR'),router=d.get('router'),control=o['control'] if o else None,stability=d.get('stability'),failure=raws[rid]['errors'] if rid in raws else None))
 for name,data in [('gpu_paired_runs.csv',runrows),('utility_comparison.csv',pairs),('utility_components.csv',unit),('per_task_metrics.csv',per),('worker_service_wait.csv',paths),('gpu_sensor_samples.csv',sensors),('control_probes.csv',controls),('dynamic_decision_log.csv',decisions),('gpu_environment_comparison.csv',resources),('backlog_timeseries.csv',back),('all_measurement_requests.csv',requests),('Text_VQA_P95_support.csv',quant),('Text_VQA_tail_requests.csv',tails),('Phi_VQA_execution_overlap.csv',vqas),('per_task_utility_decomposition.csv',contrib),('paired_request_decomposition.csv',paired),('Whisper_paired_queue_service.csv',asr),('Phi_ASR_trigger_execution.csv',triggers),('Text_P95_sensitivity_diagnostic.csv',sensitivity)]:csvsave(name,data)
 ds=[p['delta_U'] for p in pairs if p['delta_U'] is not None];statsout=dict(n_legal_pairs=len(ds),planned_n=3,primary_unit='independent matched trajectory pair',inference_valid_for_planned_three_pairs=len(ds)==3)
 if len(ds)==3:
  mean=float(np.mean(ds));sd=float(np.std(ds,ddof=1));se=sd/math.sqrt(3);crit=f['positive_gate']['t_critical'];statsout.update(deltas=ds,mean_delta_U=mean,SD=sd,SE=se,df=2,t_critical=crit,CI95_low=mean-crit*se,CI95_high=mean+crit*se,mean_S_U=float(np.mean([p['S_U'] for p in pairs])),mean_Dynamic_U=float(np.mean([p['Dynamic_U'] for p in pairs])),small_sample_limits='n=3, approximate pair-normality, residual serial/order drift unknown; reused input corpus, one GPU/model/load configuration. Request samples are not independent replications.')
 complete=len(raws)==6 and all(p['both_legal_stable'] and p['real_G_and_S'] for p in pairs)
 allgt=complete and all(p['delta_U']>.01 for p in pairs);ci=bool(len(ds)==3 and statsout['CI95_low']>0)
 gate='VALIDATION_INVALID' if not complete else 'DYNAMIC_POSITIVE_REPLICATED' if allgt and ci else 'DYNAMIC_POSITIVE_INCONCLUSIVE' if allgt else 'DYNAMIC_NO_POSITIVE_GAIN'
 final=dict(module='32G',gate=gate,engineering_integrity=audit['passed'],formal_runs=len(raws),planned_runs=6,complete_legal_three_pairs=complete,all_three_delta_gt_001=allgt,CI95_supports_positive_mean=ci,pairs=pairs,statistics=statsout,original_six_task_utility=True,history_preserved=history['passed'],router_semantics_changed=False,thresholds_adjusted=False,seeds_replaced=False,optimizer_modified=False,optimizer_autonomous_enablement_not_tested=True,route_mechanism_causal_claim=False,next_stage_started=False)
 save('pair_level_statistics.json',statsout);save('module32g_gate.json',final);save('run_observations.json',obs)
 lines=['# Module32G 最终独立GPU效用验证','',f'Gate：`{gate}`。已产生原始GPU运行{len(raws)}/6，三组完整合法且真实分流={complete}。','', '|组|S-only U|Dynamic U|ΔU|Phi/Whisper ASR|合法|','|---|---|---|---|---|---|']
 for p in pairs:lines.append(f"|{p['pair']}|{p['S_U']}|{p['Dynamic_U']}|{p['delta_U']}|{p['G_ASR']}/{p['S_ASR']}|{p['both_legal_stable']}|")
 lines+=['',f'三组配对统计：{json.dumps(statsout,ensure_ascii=False)}','',f'历史SHA/metadata完整核验={history["passed"]}，当前冻结源码与依赖核验={audit["passed"]}。32D旧progress无法逐字节恢复的已知例外只保留说明，不伪称恢复；本轮历史文件/Gate不改。所有新数据隔离至Module32G。','', '执行使用原六任务模型/权重/质量准入/预算/QPS/单副本，Module32F修复runtime、CUDA隔离与native cache provider，原路由源码和阈值byte-identical。新种子320701/320702/320703，预定S/D、D/S、S/D交错。每臂2400秒，300秒实际S-only六任务调理、30秒warmup及60秒有界排空。技术失败STOP_NO_RETRY，所有负结果及未执行槽保存。没有32B开发U或32C无效U进入独立统计。','', '服务=end_unix-start_unix；worker等待=start_unix-submit_unix；E2E=completion_perf-scheduled_perf。完整原生返回、控制器Unix/perf、每次路由过去状态、GPU遥测、controller/worker退出证据在raw_runs与request_journals。CPU/网络/驱动与GPU资源竞争不能仅凭重叠相关性唯一归因。非抢占ASR可阻塞随后到达VQA，路由器不读取未来；Phi_ASR_trigger_execution中的未来VQA计数仅事后诊断。','', '六任务mean/P95/worker服务等待、吞吐、逐任务和worker backlog、稳定性、Text/VQA P95插值支持样本与尾请求、Whisper同输入保留请求等待、Phi额外服务和VQA串行阻塞均完整保存。原正式效用不删任务、不改权重；去掉Text P95贡献仅见Text_P95_sensitivity_diagnostic.csv，不能当反事实策略或正式U。']
 if gate=='DYNAMIC_POSITIVE_REPLICATED':lines+=['','此单配置验证支持后续研究优化器自主启用：以实测共置服务需求/稳定容量、实时Whisper等待和Phi背景保护预测系统边际效用，联合比较部署S与G及关闭/启用动态路由的方案，用不确定性下界及独立数据校验后选择；模型名称、ASR QPS和比例不能硬编码。此处仅建议，未实现或验证优化器自主决策。即使三组系统U改善，也未把罕见分流对Text尾延迟的因果效应单独识别。']
 else:lines+=['','不进入正式优化器集成；当前配置没有获得全部预定正收益和可信完整证据。保留结果，不通过事后调参、换种子、改负载或延长窗口补救。']
 lines+=['','本Module完成后停止，不自动启动下一阶段。'];(OUT/'module32g_summary.md').write_text('\n'.join(lines)+'\n')
 print('FINAL_GATE',gate,'PAIRS',pairs,'STATS',statsout,flush=True)
if __name__=='__main__':main()
