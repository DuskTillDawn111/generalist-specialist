"""CPU historical screening. No model imports, inference or GPU queries."""
import collections,statistics,copy,time
from .common import *
from joint_deploy.measurement.load_generator import SamplePool
from module21.trajectory import arrivals

TEXT='roberta_sst2';IMAGE='convnext_tiny';OCR='baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt'
def inventory():
 roots=[ROOT/'results',ROOT/'models',ROOT/'deployment_core',ROOT/'joint_deploy',ROOT/'performance_model',ROOT/'replica_scaling',*[p for p in ROOT.glob('module*') if p.is_dir() and p.name!='module30']]
 return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for root in roots for p in root.rglob('*') if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts}
def main():
 assert not (OUT/'cpu_screening.json').exists(),'No rescreening after outcomes'
 save('protected_metadata_before.json',inventory())
 catalog=read(ROOT/'results/module15_three_phase/existing_empirical_catalog.json')
 base=read(ROOT/'results/module15_three_phase/gpu_validation_freeze.json')['plan']
 current=read(ROOT/'results/module29_dynamic_replication/frozen_replication_manifest.json')
 evidence=list(csv.DictReader(open(ROOT/'results/module27_positive_split_discovery/historical_service_evidence.csv')))
 source_refs=[ref(ROOT/'results/module27_positive_split_discovery/historical_service_evidence.csv')]
 summaries=[]
 for p in sorted((ROOT/'results').glob('module*/*summary.md')):
  if any(p.parent.name.startswith('module'+str(n)+'_') for n in range(15,30)) or p.parent.name=='module14f_memory_ocr_x3':
   summaries.append(dict(source=ref(p),text=p.read_text()));source_refs.append(ref(p))
 save('historical_summary_evidence.json',summaries)
 paths=collections.defaultdict(list)
 for r in evidence:
  paths[(r['model'],r['task'])].append(dict(module=r['module'],run_id=r['run_id'],raw=json.loads(r['raw']),
   service=json.loads(r['actual_service']),queue=json.loads(r['actual_queue']),rates=json.loads(r['rates']),deployment=json.loads(r['deployment']),eligible=r['eligible']))
 save('historical_measured_paths.json',[dict(model=m,task=t,sessions=rs) for (m,t),rs in sorted(paths.items())])
 gs=[m for m,v in catalog['models'].items() if v['role']=='generalist'];ss=[m for m,v in catalog['models'].items() if v['role']=='specialist']
 rows=[]
 priority={('qwen','whisper-small','asr'):1,('phi4mm','whisper-small','asr'):2}
 for g in gs:
  for s in ss:
   for t in sorted(set(catalog['models'][g]['supported_tasks'])&set(catalog['models'][s]['supported_tasks'])):
    gp=paths[(g,t)];sp=paths[(s,t)];key=(g,s,t);selected=key in priority
    reason='Sparse or absent matched coexistence evidence; outside the maximum two candidate budget'
    if g=='minicpm':reason='No matched target-worker coexistence U control; x2 empirical gain <1 and high residency; less path evidence than two selected candidates'
    if t in ('text','image') and g=='qwen':reason='Module19 matched fixed residency negative U evidence; useful historical fractions only about 3% Text or 0.36% Image; slow G and background externalities'
    if t=='ocr':reason='G OCR much slower than admitted S; no measured queue relief and matched system-U evidence'
    if s.startswith('openai/whisper-large') and g=='qwen':reason='Module26 ASR18% three long pairs negative deltaU -0.315/-0.422/-0.366; background waits and G slow path'
    if t=='lid' and s.startswith('openai/whisper-medium') and g=='qwen':reason='Modules19/22/27/29 direct negative system-U; Module29 dynamic vs S-only -0.207053, Module27 fixed10/20 negative'
    if s=='whisper_turbo':reason='Module16/17 shared ASR/LID queue failure and measured service expansion; selected Whisper-small has direct split evidence and smaller memory'
    if s=='distil_whisper':reason='Standalone admitted but no measured relevant full six-task coexistence; faster S leaves narrower queue opportunity; extra LID worker required'
    if selected:
     reason=('G ASR measured substantially faster than Phi and six-task background path evidence; Whisper-small direct target split/queue observations, new pair remains unprofiled' if g=='qwen' else 'Module15 direct same pair completed nonzero ASR split and real queues; Module19 slowdown/unstable high point respected by conservative fixed load')
    rows.append(dict(rank=priority.get(key,100),Generalist=g,Specialist=s,target=t,decision='SELECT' if selected else 'ELIMINATE_THIS_SCREEN',reason=reason,
     G_standalone_ms=catalog['models'][g]['service_time_ms'][t],S_standalone_ms=catalog['models'][s]['service_time_ms'][t],
     G_measured_sessions=len(gp),S_measured_sessions=len(sp),
     G_observed_service_min_ms=min((r['service']['mean_ms'] for r in gp),default=None),G_observed_service_max_ms=max((r['service']['mean_ms'] for r in gp),default=None),
     S_observed_service_min_ms=min((r['service']['mean_ms'] for r in sp),default=None),S_observed_service_max_ms=max((r['service']['mean_ms'] for r in sp),default=None),
     S_observed_queue_max_ms=max((r['queue']['mean_ms'] for r in sp),default=None),
     pair_safe_memory_mib=catalog['safe_factor']*(catalog['memory_table_mib'][g]['1']+catalog['memory_table_mib'][s]['1']),
     missing_profile='Current GPU UUID six-task same-residency route-conditioned service/tail/queue; no positive counterfactual established',predicted_delta_U=None))
 rows.sort(key=lambda r:(r['rank'],r['Generalist'],r['target'],r['Specialist']))
 for i,r in enumerate(rows,1):r['rank']=i
 csvsave('candidate_pool_ranking.csv',rows)
 # Use exact empirical service, never an invented coexistence multiplier.
 small_asr_max=max(r['service']['mean_ms'] for r in paths[('whisper-small','asr')] if r['eligible']=='True')
 small_lid=catalog['models']['whisper-small']['service_time_ms']['lid']
 demand=1.*small_asr_max/1000+.2*small_lid/1000
 assert demand<.8,(small_asr_max,demand)
 rates=dict(text=.2,image=.2,ocr=.2,asr=1.,lid=.2,visual_qa=.2)
 # .05/.10 gives ~30/~60 G requests and historical Phi upper demand below .55.
 candidates=[];slots=[]
 for index,g in enumerate(['qwen','phi4mm'],1):
  cid='C'+str(index);models=[g,'whisper-small',TEXT,IMAGE,OCR];deployment={m:1 for m in models}
  safe=catalog['safe_factor']*sum(catalog['memory_table_mib'][m]['1'] for m in models)
  seed=301000+index;trajectories={}
  for label,duration in [('warmup',30.),('measurement',600.)]:
   pool=SamplePool(base['sample_pool']['path'],list(rates),seed);events=arrivals(rates,duration,seed,label)
   for e in events:e['item']=pool.next(e['task'])
   name='trajectories/'+cid+'_'+label+'.json';save(name,dict(events=events,duration=duration,seed=seed,rates=rates));trajectories[label]=ref(OUT/name)
  target_events=[e for e in events if e['task']=='asr'];counts={str(p):sum(e['routing_uniform']<p for e in target_events) for p in [.05,.10]}
  # Routing uses sorted model order; both qwen and phi4mm sort before whisper-small.
  order=['split05','s_only','split10'] if index==1 else ['split10','s_only','split05']
  for arm in order:
   frac=dict(s_only=0.,split05=.05,split10=.10)[arm]
   route=dict(text={TEXT:.2},image={IMAGE:.2},ocr={OCR:.2},visual_qa={g:.2},lid={'whisper-small':.2},asr={'whisper-small':1-frac})
   if frac:route['asr'][g]=frac
   slots.append(dict(run_id=cid+'_'+arm,candidate=cid,arm=arm,stage='B',seed=seed,repetition=1,target_task='asr',Generalist=g,Specialist='whisper-small',
    deployment=deployment,lambda_rps=rates,routing=route,measurement_seconds=600.,safe_memory_budget_mib=34567.5534,trajectories=trajectories,frozen_G_fraction=frac))
  candidates.append(dict(candidate=cid,Generalist=g,Specialist='whisper-small',target='asr',rates=rates,deployment=deployment,
   budget_mib=34567.5534,CPU_measured_table_safe_memory_mib=safe,seed=seed,split_fractions=[.05,.10],frozen_actual_G_requests=counts,
   S_only_empirical_upper_demand=demand,G_split10_historical_upper_demand=.1*max(r['service']['mean_ms'] for r in paths[(g,'asr')])/1000+.2*max(r['service']['mean_ms'] for r in paths[(g,'visual_qa')])/1000,
   missing_profile='Exact six-task co-resident route-conditioned service on current GPU; stable feasibility and positive U unknown'))
  assert safe<34567.5534
 models={m:catalog['models'][m] for c in candidates for m in c['deployment']}
 plan=copy.deepcopy(base);plan['models']=models;plan['registry']={m:base['registry'][m] for m in models}
 # Reuse the fully admitted current PARSeq native launcher; same frozen checkpoint.
 plan['registry'][OCR]=current['plan']['registry'][OCR]
 plan['environment_overrides']=current['plan']['environment_overrides'];plan['warmup_seconds']=30.;plan['measurement_seconds']=600.
 save('selected_pool_catalog.json',{**catalog,'models':models})
 source_refs += [ref(ROOT/'results/module15_three_phase/existing_empirical_catalog.json'),ref(ROOT/'results/module15_three_phase/gpu_validation_freeze.json'),ref(ROOT/'results/module29_dynamic_replication/frozen_replication_manifest.json'),ref(ROOT/'results/module20_utility_optimizer/utility_definition.json'),ref(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json'),ref(current['quality_policy']['path']),ref(base['sample_pool']['path'])]
 save('cpu_screening.json',dict(CPU_only=True,completed_unix_ns=time.time_ns(),candidates=candidates,slots=slots,plan=plan,safe_factor=catalog['safe_factor'],
  quality_policy=current['quality_policy'],thermal_protocol=current['thermal_protocol'],source_refs=source_refs,
  limitations=['Observed session extrema are historical evidence, not confidence bounds or transferable capacity certificates','Module15/19 GPU UUID may differ; no equivalence assumed','Standalone S LID contribution is unprofiled on this exact deployment; explicitly missing','No predicted positive U or generic coexistence multiplier'],maximum_formal_runs=6,formal_seconds=600,search_expansion=False))
 text='''# Module30 候选冻结依据（CPU-only）

排名第一：Qwen2.5-Omni + Whisper-small，ASR；排名第二：Phi-4-multimodal + Whisper-small，ASR。这只是优先测量顺序，尚无正收益认证。

完整已准入池的共同任务配对逐项见 candidate_pool_ranking.csv。Qwen 的真实 ASR 比 Phi 较快且已有六任务背景路径；Phi 配对在 Module15 有真实 G/S 请求和 S 等待，但不具备匹配 S-only 效用反事实。Module19 Phi/Small 高点两臂不稳定以及慢服务被保留；淘汰旧 4.3768QPS/8.02% 点，未淘汰整个模型配对。Qwen/Large ASR、Qwen/Medium LID 长窗口负结果直接降低优先级；其它配对缺乏完整共置画像，在最多两个组合上限内淘汰，未宣称数学上不可能。

两个组合均冻结 ASR=1 QPS；其余五任务各0.2QPS。Text→RoBERTa-base，Image→ConvNeXt-tiny，OCR→PARSeq，LID→Whisper-small，VQA→该 Generalist。五模型各一副本；预算34567.5534MiB，safe-factor与20A效用/refs及六任务质量准入原样使用。S额外承担LID时真实共享等待由新测量报告。两个组分别固定种子301001/301002，一条轨迹匹配三臂，输入池与顺序在GPU前冻结。

ASR点用合法历史 Small ASR 最大 session mean '''+str(small_asr_max)+'''ms 核对；ASR1加LID0.2的历史需求约 '''+str(demand)+'''，低于0.8，仅是筛选依据而非新组合稳定性保证。这个点同时规避Module19失败高点，并允许Poisson到达形成真实S排队。若S实际服务更快、队列压力不足，将保留负结果，不移动点。

分流比例预冻结5%/10%，约30/60个G请求；Phi在历史最慢ASR均值下10%加VQA需求仍有显著余量。实际冻结请求数见cpu_screening.json；没有根据收益换种子或提高样本数。历史极值不是逐请求上界。10%没有被包装成最优比例。不同模型副本的g/M表全部保留；Small x2 gain约1.238、x3约1.320，Phi x2约1.636，Qwen x2约1.216，MiniCPM x2<1；新增副本可能降低S压力，不能当同驻留路由收益，此轮固定x1。

最重要的缺失画像：当前GPU UUID、上述六任务共置条件下S-only和两种split的服务/等待/tail、S膨胀、G余量、VQA外部代价、低QPS背景mean/P95。历史数据已显示服务漂移，故不使用统一共置倍率，不预测正ΔU。两组各三次600秒探索，独立长窗口复现不自动执行。
'''
 (OUT/'candidate_selection_rationale.md').write_text(text)
 print('CPU_SCREENING_COMPLETE',len(rows),candidates,flush=True)
if __name__=='__main__':main()
