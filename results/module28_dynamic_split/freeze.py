import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from audit import *
import copy,random
from module21.trajectory import arrivals
from joint_deploy.measurement.load_generator import SamplePool
from replica_scaling.telemetry import snapshot

def main():
 assert not (OUT/'frozen_candidate_manifest.json').exists()
 old=json.loads((ROOT/'results/module27_positive_split_discovery/discovery_protocol.json').read_text());plan=copy.deepcopy(old['plan']);d=json.loads((ROOT/'results/module27_positive_split_discovery/selected_pool_catalog.json').read_text());gpu=snapshot();assert not gpu['processes'] and gpu['used_mib']<=128
 plan['GPU']=gpu
 for m,v in plan['registry'].items():
  if 'results' in Path(v['script']).parts:
   dest=OUT/'launchers'/Path(v['script']).name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(v['script']).read_bytes());v['script']=str(dest)
 save('selected_pool_catalog.json',d)
 G='qwen';S=next(m for m in plan['models'] if 'medium@' in m);gw=G+'#0';sw=S+'#0';rates=dict(text=.2,image=.2,ocr=.2,asr=.2,lid=11.5,visual_qa=.2)
 development=[];samples=collections.defaultdict(list);means={}
 for p in sorted((ROOT/'results/module27_positive_split_discovery/raw_runs').glob('*.json')):
  raw=json.loads(p.read_text());assert raw['baseline_GPU']['uuid']==gpu['uuid'],'Current-GPU development coverage required';development.append(ref(p))
  rs=raw['phases']['measurement']['request_records']
  for r in rs:samples[(r['model'],r['task'])].append(r['worker_latency_ms'])
  means[p.name]=dict(S=float(np.mean([r['worker_latency_ms'] for r in rs if r['model']==sw and r['task']=='lid'])),G_fraction=raw['slot']['G_fraction'])
 def quant(w,t,q):return float(np.quantile(samples[w,t],q))
 s_lower={t:quant(sw,t,.1) for t in ['lid','asr']};g_upper={t:quant(gw,t,.95) for t in ['lid','visual_qa']}
 pure=[v['S'] for v in means.values() if v['G_fraction']==0];mixed=[v['S'] for v in means.values() if v['G_fraction']>0]
 inflation=max(mixed)-min(pure);bg=rates['visual_qa']*g_upper['lid']**2/(2*1000) # ms expected aggregate FIFO blocking
 # S ASR and Text/Image/OCR externality envelope adds observed service inflation
 # for approximately one G handler duration of S processing.
 bg+=inflation*g_upper['lid']/s_lower['lid']
 params=dict(G_worker=gw,S_worker=sw,target='lid',S_lower_by_task_ms=s_lower,G_upper_by_task_ms=g_upper,G_service_ms=g_upper['lid'],S_inflation_per_pending_ms=inflation,background_cost_ms=bg,G_max_target_inflight=1,G_token_rate_rps=rates['lid']*.05,cooldown_seconds=g_upper['lid']/1000,entry_margin_ms=s_lower['lid'],exit_margin_ms=s_lower['lid']/2,ewma_alpha=.1,prediction_error_ratio=1.5,fallback_seconds=30.,S_service_alarm_ms=max(mixed)*1.5)
 save('router_parameters.json',params)
 mapping={f'{m}#0':m for m in plan['models']};routes={t:dict(row) for t,row in old['exploration_slots'][0]['routing'].items()};routes['lid']={S:rates['lid']};routes['asr']={S:rates['asr']}
 def trace(name,seed,burst,duration=600.):
  result={}
  for label,seconds in [('warmup',30.),('measurement',duration)]:
   pool=SamplePool(plan['sample_pool']['path'],list(rates),seed);events=arrivals(rates,seconds,seed,label)
   for e in events:
    if burst and e['task']=='lid':
     cycle=int(e['offset_seconds']//10);u=e['offset_seconds']-10*cycle
     e['offset_seconds']=10*cycle+(u/2 if u<4 else 2+(u-4)/.75)
    e['item']=pool.next(e['task'])
   events.sort(key=lambda e:(e['offset_seconds'],e['task']));fn=f'trajectories/{name}_{label}.json';save(fn,dict(rates=rates,duration=seconds,seed=seed,label=label,arrival_profile='burst' if burst else 'steady',events=events));result[label]=ref(OUT/fn)
  return result
 def slot(name,arm,seed,profile,traces,stage,rep=1):
  rr=copy.deepcopy(routes);fraction=.1 if arm=='fixed10' else 0.
  # Advertise both target paths for READY/capacity structural checks. Actual
  # dynamic dispatch is determined exclusively by the independent router.
  if arm=='dynamic':fraction=.05
  rr['lid']={S:rates['lid']*(1-fraction)}
  if fraction:rr['lid'][G]=rates['lid']*fraction
  return dict(run_id=name,stage=stage,arm=arm,arrival_profile=profile,point_id=profile,seed=seed,repetition=rep,lambda_rps=rates,deployment={m:1 for m in plan['models']},routing=rr,target_task='lid',measurement_seconds=600.,safe_memory_budget_mib=old['budget_mib'],trajectories=traces,G_fraction=fraction)
 slots=[];independent=[]
 for profile,seed,arms in [('steady',28101,['s_only','fixed10','dynamic']),('burst',28102,['dynamic','s_only','fixed10'])]:
  traces=trace('C_'+profile,seed,profile=='burst')
  slots.extend(slot('C_'+profile+'_'+arm,arm,seed,profile,traces,'C') for arm in arms)
 for profile in ['steady','burst']:
  for rep,seed in enumerate([28201,28202,28203],1):
   tr=trace('D_'+profile+'_rep'+str(rep),seed,profile=='burst')
   arms=[['s_only','fixed10','dynamic'],['dynamic','fixed10','s_only'],['fixed10','s_only','dynamic']][rep-1]
   independent.extend(slot(f'D_{profile}_rep{rep}_{arm}',arm,seed,profile,tr,'D',rep) for arm in arms)
 f=dict(status='FROZEN_BEFORE_GPU',at_unix_ns=time.time_ns(),plan=plan,catalog=ref(OUT/'selected_pool_catalog.json'),GPU=gpu,models=list(plan['models']),target='lid',extra_task_pair=None,rates=rates,budget_mib=old['budget_mib'],safe_factor=old['safe_factor'],weights=old['weights'],refs=old['refs'],parameters=params,parameter_source='Same current GPU Module27 development measurements only; no Stage C/D tuning',parameter_derivation=dict(S_lower='pooled P10 of same-worker per-task native handler times',G_upper='pooled P95 same-worker per-task native handler times',S_inflation='max mixed-session S target mean minus min pure-session S mean',background='VQA FIFO blocking lambda*G_service^2/2 plus S overlap inflation envelope',margins='one and half conservative S target service time',token_limit='half of best tested nonzero fixed10; engineering bound within measured stable envelope',error_ratio='1.5 engineering safety factor',fallback='30s, same as warmup, engineering safety duration; not statistically calibrated'),development_sources=development,utility_definition=ref(ROOT/'results/module20_utility_optimizer/utility_definition.json'),utility_reference=ref(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json'),quality_policy=ref(ROOT/'results/module8b_quality_pool_finalization/quality_policy.json'),exploration_slots=slots,independent_slots=independent,maximum_exploration_attempts=6,user_exploration_upper_limit=12,maximum_validation_attempts=9,maximum_total_attempts=15,maximum_formal_gpu_seconds=9000,exploration_seconds=600.,thermal_protocol=old['thermal_protocol'],fixed_comparator=dict(fraction=.1,reason='Module27 10% beats20% at10.5; only tested nonzero fraction11.5; best measured nonzero prior. S-only0% is stronger prior and included. Not exhaustive optimum.'),G_only=dict(executed=False,reason='UNSUPPORTED stable capacity: 11.5*measured G LID service exceeds one handler-second per second by >3x; reject before GPU'),profiles=dict(steady='Stationary per-task Poisson',burst='LID inverse integrated-intensity warp: 2s at2x then8s at0.75x in10s cycles. Other5 tasks stationary. Same expected mean; exact offered cohort count unchanged by warp.'),selection=dict(delta_over=.01,all_three_legal_stable=True,both_dynamic_paths_nonzero=True,min_task_count=72,thermal_match_c=5.,rank='largest dynamic minus strongest legal stable measured control; tie steady first; select at most one profile'),independent_protocol=dict(pairs=3,seconds=600.,new_seeds=[28201,28202,28203],all_deltas_over=.01,paired_t95_lower_over_zero=True,min_task_count=72,all_three_controls_legal_stable=True,both_paths_nonzero=True,negative_seeds_never_replaced=True,code_and_thresholds_never_tuned=True),failure_policy='Any infrastructure/quality/memory/unstable failure stops; no outcome retry. C no positive: stop. D any failing group retained, finish registered groups if safe; no integration.',optimizer_modified=False,next_module_automatic=False,implementation=[])
 f['implementation']=[ref(p) for p in sorted(OUT.glob('*.py'))]+[ref(v['script']) for v in plan['registry'].values()]+[ref(ROOT/p) for p in ['joint_deploy/measurement/deployment_runtime.py','module22/gpu.py','module16/gpu.py','module13b/worker.py','module14b/worker.py','module24/calibration.py','module20/optimizer.py']]
 save('frozen_candidate_manifest.json',f)
 (OUT/'opportunity_audit.md').write_text('''# Module28 Stage A：CPU-only 机会审计\n\n读取 Module19–27 原始结果，逐请求重建见 historical_target_request_audit.csv；六任务及全系统 mean/P95 重算见 historical_system_audit.json，worker 服务/等待及质量可用性见 historical_worker_paths.csv。无独立 raw 的 Module20/23/25 保留原摘要与链接关系，不制造新观测。\n\nModule27 11.5QPS S-only：6806个LID中1010个(14.84%)到达时S≥8个已入队未完成请求且G没有未完成请求；737个(10.83%)满足S≥10。真实native handler区间确认这些G空队列窗口也没有G执行。平均GPU利用率不作为模型空闲证据。S服务mean69.13ms，等待294.02ms；固定10%后S服务74.60ms、等待318.23ms，G服务383.52ms、等待190.90ms，VQA等待20.49→161.16ms。\n\nModule26 ASR S等待约1s、G空队列短时窗口也存在，但三次18%固定分流负收益且G ASR约1.3s、S服务约19%膨胀；本轮不增加ASR候选。保留同一Module27 LID池的一个保守队列策略，不再调整固定比例网格。选择11.5因为同机实测尾部排队/G空队列窗口更充分，S-only和10%曾合法稳定，所有六任务正QPS。\n\n可支持：真实S排队、G host-visible空队列和retrospective native闲态、固定分流服务膨胀和背景等待损失。机会满足前两项，第三项仅可形成开发估计，需要GPU补测；第四项仅有同池历史硬约束先验，新动态尚未认证。\n\nUNSUPPORTED：未执行路由的逐请求反事实完成时间或ΔU；没有worker receipt ACK，host send不得冒充接收；native START/END只能回顾，不能作为当时已知信息；共置物理GPU/CPU/热态因果无法唯一分离；预测proxy不构成系统效用证明。新GPU轨迹是检验这些证据缺口的合理有界补测。\n\n参数、模型、负载、效用、质量、预算、运行顺序、稳态/突发及独立验证门槛全部见 frozen_candidate_manifest.json，冻结后不得依据结果调参。G-only容量不合理，预先排除。正式部署优化器不修改。\n''')
 print('FROZEN',params,flush=True)
if __name__=='__main__':main()
