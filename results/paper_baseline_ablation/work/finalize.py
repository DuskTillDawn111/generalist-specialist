import os,sys,json,csv,time,math,hashlib,statistics,shutil,platform,itertools
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('/root/autodl-tmp/generalist-specialist');OUT=Path(os.environ.get('PAPER_EVAL_OUT',str(ROOT/'results/paper_baseline_ablation')));M34=ROOT/'results/module34_candidate_pool';M35=ROOT/'results/module35_utility_routing';EX=ROOT/'results/paper_extra_g_only_retention'
sys.stdout=(OUT/'work/finalize.log').open('a',buffering=1);sys.stderr=sys.stdout
sys.dont_write_bytecode=True;sys.path[:0]=[str(M35),str(M34),str(ROOT)]
from m35 import isolation
isolation.ROOT=OUT;isolation.install('supervisor')
import numpy as np
from scipy.stats import t,ttest_1samp
from module20.optimizer import predict_scenarios
from m34.gpu_assess import assess
from module26.common import arithmetic,cohort
from module21.trajectory import choose

def read(p):return json.loads(Path(p).read_text())
def save(name,x):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
def ref(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
def table(p):
 rows=list(csv.DictReader(Path(p).open()))
 for r in rows:
  for k,v in list(r.items()):
   if v=='':r[k]=None
   elif v in ['True','False']:r[k]=v=='True'
   elif v.startswith(('{','[')):
    try:r[k]=json.loads(v)
    except ValueError:pass
   else:
    try:r[k]=float(v)
    except ValueError:pass
 return rows

def csvsave(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r));p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True,default=lambda x:x.item() if hasattr(x,'item') else str(x)) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def identity(x,r):return json.dumps([x,{t:{m:round(q,10) for m,q in rr.items() if q>1e-9} for t,rr in r.items()}],sort_keys=True)
def copyraw(p,relative=None):
 p=Path(p);dest=OUT/(relative or ('raw_data/'+p.name));dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());return str(dest)
assert read(OUT/'CPU_completed.json')['cells']==12
protocol=read(OUT/'experiment_protocol.json');sp=protocol['candidate_space'];refs=protocol['latency_reference_ms'];weights=protocol['weights'];budget=protocol['budget_mib']
base=read(OUT/'baseline_cpu_results.json');abl=read(OUT/'ablation_cpu_results.json');timings=table(OUT/'solver_timing_repetitions.csv');coverage=table(OUT/'route_candidate_coverage.csv');lookup={(r['scenario'],r['pool'],r['algorithm']):r for r in base}
# Recover actual historical counts from preserved candidate logs, avoiding duplicated V3 caches.
fitnames={'20A_U_GRID':'20A_GRID','20A_NATIVE_CONTINUOUS_PRESERVED':'20A_NATIVE','V3_J_GRID_CONTROL':'V3_GRID','V3_NATIVE_CONTINUOUS_PRESERVED':'V3_NATIVE'}
for r in base:
 alg=r['algorithm'];s=r['scenario'];pool=r['pool']
 if alg in fitnames:
  source=M35/f'cpu/{s}_{pool}_{fitnames[alg]}.json';f=read(source);copyraw(source);r['raw_fit']=str(source)
  if alg.startswith('20A'):
   r.update(generator_MILP_calls=len(f['generation_calls']),full_utility_evaluations=len(f['candidates']),scenario_prediction_calls_search=4*len(f['candidates']),scenario_prediction_calls_final_validation=4 if f.get('winner') else 0,search_states=len(f['candidates']),search_state_definition='unique_generated_completeplans; MILP internal tree nodes not recorded')
  else:
   visited=f.get('result',{}).get('visited_deployments',[])
   states={tuple(sorted((m,n) for m,n in x.items() if n)) for x in visited} if isinstance(visited,list) and all(isinstance(x,dict) for x in visited) else set()
   # Actual grid J cache keys from two roles share the same cache. Reconstruct the declared calls from logged candidates.
   audit=next((z for z in read(M34/'optimizer_evaluation_count_audit.json') if z['pool']==pool and z['scenario']==s and z['algorithm']==alg),None)
   r.update(full_utility_evaluations=len(f['backend_calls']),scenario_prediction_calls_search=4*len(f['backend_calls']),scenario_prediction_calls_final_validation=4 if f.get('decision') else 0,
    search_states=f.get('result',{}).get('visited_deployment_count'),backend_complete_plan_calls=len(f['backend_calls']),canonical_LP_calls=f.get('result',{}).get('lp_solve_count'),seed_MILP_calls=f.get('result',{}).get('milp_solve_count'),
    J_surrogate_evaluations=audit.get('controlled_exact_J_surrogate_calls') if audit else None,J_count_status='retrospectively_reconstructed_from_frozen_development_logs' if audit and alg=='V3_J_GRID_CONTROL' else 'NOT_INSTRUMENTED_PRESERVED',search_state_definition='visited deployment states; J-route internal states tracked separately')
 elif alg=='U_ALIGNED_DP':r.update(scenario_prediction_calls_search=0,scenario_prediction_calls_final_validation=4)
 elif alg=='CPU_U_ORACLE':r.update(scenario_prediction_calls_search=r['full_prediction_calls'],scenario_prediction_calls_final_validation=0)
 elif alg in ['FASTEST_PATH_STATIC','MIN_MEMORY_COVER_FASTEST']:r.update(full_utility_evaluations=0,scenario_prediction_calls_search=0,scenario_prediction_calls_final_validation=4,search_states=1,search_state_definition='one constructed completeplan; cover enumeration counted separately')
 r['model_pool_count']=len(sp['members'][pool])
 if alg=='MIN_MEMORY_COVER_FASTEST':r['deployment_vectors_considered']=2**r['model_pool_count'];r['search_state_definition']='one constructed completeplan plus all binarycovervectors; see deployment_vectors_considered'
 if alg=='FASTEST_PATH_STATIC':r['deployment_vectors_considered']=0
csvsave('baseline_cpu_results.csv',base);save('baseline_cpu_results.json',base)
# Freeze GPU reuse choice after CPU, before new raw evaluation. No physical new measurement.
GPUcases=[('A_TEXT_HALF_MID18','A','TEXT_HALF_MID18'),('B_TEXT_HALF_MID18','B','TEXT_HALF_MID18'),('C_ASR32B','C','ASR32B')]
save('GPU_reuse_decision.json',dict(decided_after_CPU=True,CPU_completion=ref(OUT/'CPU_completed.json'),new_GPU_scenarios=0,
 reason='Existing34 has three frozen within-pool same-trace fullconfiguration comparisons covering35 versusfastest (A,B,C) and20A (A,B). No result-directed extra trials.',
 groups=[g for g,p,s in GPUcases],independent_seeds=[340201,340202,340203],requirements=['exact fullconfigurationmatch','same pool admissibility','same budget','same trajectoryhash/content','same U references/weights','same duration/warmup','original ordering/stabilitygates retained'],
 expansion35='Context only: B->C candidate domains differ; cannot attribute to optimizer improvement'))
protocol['GPU'].update(status='HISTORICAL_REUSE_ONLY',decision_file='GPU_reuse_decision.json',new_scenarios=0,groups=[g for g,p,s in GPUcases],traces=[340201,340202,340203],measurement_seconds=240,warmup_seconds=30,drain_seconds=60)
save('experiment_protocol.json',protocol)
obs=read(M34/'gpu_observations.json');byrun={r['run_id']:r for r in obs};gpu=[];pairs=[];audits=[];copyindex=[]
for group,pool,s in GPUcases:
 d=read(sp['pools'][pool]['path']);nr=lookup[s,pool,'U_ALIGNED_DP'];fast=lookup[s,pool,'FASTEST_PATH_STATIC']
 for seed in [340201,340202,340203]:
  arms={}
  for arm in ['ORACLE','COMPETITOR']:
   o=next(r for r in obs if r['stage']=='INDEPENDENT' and r['run_id'].endswith(group+'_'+arm) and read(r['raw']['path'])['slot']['seed']==seed)
   source=Path(o['raw']['path']);raw=read(source);a=assess(raw);comp=arithmetic(raw,d,refs,weights);sl=raw['slot'];target=nr if arm=='ORACLE' else fast
   configeq=identity(sl['deployment'],sl['routing'])==identity(target['deployment'],target['routing'])
   assert configeq,(group,seed,arm,'CONFIG_MISMATCH')
   assert a['eligible'] and a['stable'] and a['technical_valid'];assert abs(comp['raw_arithmetic_U']-o['U'])<1e-9
   assert sl['safe_memory_budget_mib']==budget and sl['measurement_seconds']==240
   tr=read(sl['trajectories']['measurement']['path']);actual=ref(sl['trajectories']['measurement']['path']);assert actual['sha256']==sl['trajectories']['measurement']['sha256']
   # Full native requests, trace content and frozen config checked by original assess.
   copied=copyraw(source,'raw_data/gpu/'+source.name);copyindex.append(ref(copied))
   for kind in ['measurement','warmup']:
    pp=Path(sl['trajectories'][kind]['path']);cp=OUT/'raw_data/gpu'/pp.name
    if not cp.exists():copyraw(pp,'raw_data/gpu/'+pp.name);copyindex.append(ref(cp))
   alias=[z['algorithm'] for z in base if z['scenario']==s and z['pool']==pool and z.get('deployment') and identity(z['deployment'],z['routing'])==identity(sl['deployment'],sl['routing'])]
   row=dict(group=group,pool=pool,scenario=s,seed=seed,arm=arm,method='MODULE35_FROZEN' if arm=='ORACLE' else 'FASTEST_PATH_STATIC',same_configuration_methods=alias,
    deployment=sl['deployment'],routing=sl['routing'],request_trace_sha256=actual['sha256'],U=comp['raw_arithmetic_U'],throughput_loss=comp['T_norm'],mean_E2E_ms=comp['pooled_mean_ms'],P95_E2E_ms=comp['pooled_p95_ms'],
    per_task=comp['per_task'],safe_memory_mib=comp['safe_memory_mib'],sampled_peak_mib=comp['sampled_peak_mib'],offered=comp['offered'],completed=comp['completed'],within=comp['within'],measurement_seconds=240,
    eligible=a['eligible'],technical_valid=a['technical_valid'],stable=a['stable'],checks=a['checks'],source_raw=str(source),raw_sha256=ref(source)['sha256'],copied_raw=copied,original_protocol=str(M34/'frozen_gpu_manifest_v2.json'),
    pooled_latency_is_not_U_normalized_latency=True,measurement_equivalence='shared configuration measurement; not an independent run for each algorithmalias')
   gpu.append(row);arms[arm]=(row,raw,a)
  aa,bb=arms['ORACLE'],arms['COMPETITOR'];same_trace=cohort(aa[1])==cohort(bb[1]);samehash=aa[0]['request_trace_sha256']==bb[0]['request_trace_sha256'];samerate=aa[1]['slot']['lambda_rps']==bb[1]['slot']['lambda_rps']
  ca=read(M34/'gpu_contrasts.json');historic=next(z for z in ca if z['group']==group)
  assert same_trace and samehash and samerate
  pair=dict(group=group,pool=pool,scenario=s,seed=seed,delta_U=aa[0]['U']-bb[0]['U'],both_eligible=True,exact_trace_match=same_trace,trace_hash_match=samehash,same_rates=samerate,
   same_budget=True,same_window=True,complete_configuration_match=True,environment_original_gate=historic['environment_comparable'],same_pool_candidate_domain=True,all_checks=True)
  pairs.append(pair);audits.append(pair)
 vals=[r['delta_U'] for r in pairs if r['group']==group];mean=float(np.mean(vals));sd=float(np.std(vals,ddof=1));half=float(t.ppf(.975,2)*sd/math.sqrt(3));pval=float(ttest_1samp(vals,0).pvalue)
 for row in gpu:
  if row['group']==group:row.update(paired_delta_U_mean=mean,paired_CI95_low=mean-half,paired_CI95_high=mean+half,paired_n=3,paired_raw_p=pval)
print('GPU_REUSE_RAW_AUDIT_DONE',len(gpu),flush=True)
csvsave('gpu_baseline_results.csv',gpu);save('gpu_baseline_results.json',gpu);csvsave('gpu_paired_differences.csv',pairs);save('gpu_reuse_raw_audit.json',audits)
summary=[]
for group,pool,s in GPUcases:
 rr=[r for r in gpu if r['group']==group];first=rr[0];historical=next(z for z in table(M34/'gpu_empirical_regret.csv') if z['group']==group)
 assert abs(first['paired_delta_U_mean']-historical['observed_delta'])<1e-9
 assert abs(first['paired_CI95_low']-historical['ci95'][0])<1e-9
 summary.append(dict(group=group,pool=pool,scenario=s,delta_U=first['paired_delta_U_mean'],CI95_low=first['paired_CI95_low'],CI95_high=first['paired_CI95_high'],n=3,
  holm_adjusted_p=historical['holm_adjusted_p'],verdict='SUPPORTED_WITHIN_FROZEN_PAIR' if historical['holm_adjusted_p']<.05 and first['paired_CI95_low']>0 else 'INCONCLUSIVE',
  comparison='Module35 selected config vs fastest selected config',aliases_new=next(r['same_configuration_methods'] for r in rr if r['arm']=='ORACLE'),aliases_old=next(r['same_configuration_methods'] for r in rr if r['arm']=='COMPETITOR'),scope='frozen withinpool pairs; reattribution by exactconfiguration; no independent35algorithm runs or allpoolGPUoracle'))
csvsave('gpu_paired_summary.csv',summary);save('gpu_paired_summary.json',summary)
for p in [M34/'frozen_gpu_manifest_v2.json',M34/'statistical_analysis_plan.json',M34/'gpu_contrasts.json',M34/'gpu_empirical_regret.csv',M35/'gpu_paired_contrasts.json',M35/'frozen_gpu_protocol.json',ROOT/'results/module33_mixed_oracle/candidate_space_definition.json',ROOT/'results/module20_utility_optimizer/module20a_gate.json',EX/'final_gate.json',M34/'quality_and_profile_preservation.json',M34/'quality_admission_provenance.csv']:
 copyraw(p);copyindex.append(ref(p))
# Solver efficiency. Fresh U-only comparison; historicalbaseline measurements retained with explicit scope.
eff=[]
for scenario in dict.fromkeys(r['scenario'] for r in timings):
 for pool in sp['pools']:
  groups={method:[r for r in timings if r['scenario']==scenario and r['pool']==pool and r['method']==method] for method in ['FINITE_U_ONLY_ORACLE','MODULE35_FROZEN']}
  osec=statistics.median(r['seconds'] for r in groups['FINITE_U_ONLY_ORACLE']);msec=statistics.median(r['seconds'] for r in groups['MODULE35_FROZEN']);count=groups['FINITE_U_ONLY_ORACLE'][0]['full_utility_evaluations']
  for method,rr in groups.items():
   r=rr[0];sec=statistics.median(z['seconds'] for z in rr);new=lookup[scenario,pool,'U_ALIGNED_DP']
   eff.append(dict(scenario=scenario,pool=pool,model_pool_count=len(sp['members'][pool]),split=r['split'],method=method,strict_same_domain=True,K=1,seconds_median=sec,seconds_min=min(z['seconds'] for z in rr),seconds_max=max(z['seconds'] for z in rr),repetitions=3,
    oracle_candidates=count,search_states=r['search_states'],state_definition=r['state_definition'],deployment_vectors=r.get('deployment_vectors'),full_utility_evaluations=r['full_utility_evaluations'],scenario_prediction_calls=r['scenario_prediction_calls'],worker_prediction_calls=r['worker_prediction_calls'],
    predicted_U=new['predicted_U'],oracle_regret=0.,speedup_vs_finite_U_only_oracle=osec/sec,time_scope=r['time_scope'],source='solver_timing_repetitions.csv',MILP_internal_nodes=None,DP_transitions=r.get('transitions'),DP_dominance_pruned=r.get('dominance_pruned'),counter_scope='DPprefixlabels, workerbundle attempts, utilityplans and 4scenario predictioncalls are different counts'))
  for b in [r for r in base if r['scenario']==scenario and r['pool']==pool and r['algorithm'] not in ['U_ALIGNED_DP','CPU_U_ORACLE']]:
   eff.append(dict(scenario=scenario,pool=pool,model_pool_count=len(sp['members'][pool]),split=b['split'],method=b['algorithm'],strict_same_domain=b['strict_same_search_domain'],K=1 if b['strict_same_search_domain'] else None,seconds_median=b['seconds'],repetitions=1,
    oracle_candidates=count,search_states=b.get('search_states'),state_definition=b.get('search_state_definition'),full_utility_evaluations=b.get('full_utility_evaluations'),scenario_prediction_calls=b.get('scenario_prediction_calls_search'),scenario_prediction_calls_final_validation=b.get('scenario_prediction_calls_final_validation'),worker_prediction_calls=0,
    predicted_U=b.get('predicted_U'),oracle_regret=b.get('regret'),speedup_vs_finite_U_only_oracle=None,time_scope=b['time_scope'],source='historical35baseline; different measurementbatch; no fresh speedup claimed',MILP_generation_calls=b.get('generator_MILP_calls'),deployment_vectors_considered=b.get('deployment_vectors_considered'),MILP_internal_nodes=None,J_surrogate_evaluations=b.get('J_surrogate_evaluations'),J_count_status=b.get('J_count_status')))
for a in abl:
 if a.get('seconds_median') is not None:
  met=a['metrics'];eff.append(dict(scenario=a['scenario'],pool=a['pool'],split=a['split'],method=a['ablation'],strict_same_domain=False,seconds_median=a['seconds_median'],repetitions=3,search_states=met.get('search_states'),worker_prediction_calls=met.get('worker_prediction_calls'),full_utility_evaluations=met.get('full_utility_evaluations',met.get('full_prediction_calls',0)/4),scenario_prediction_calls=met.get('scenario_prediction_calls',met.get('full_prediction_calls')),oracle_regret=a['oracle_regret'],predicted_U=a['predicted_U'],speedup_vs_finite_U_only_oracle=None,time_scope='fixed_deployment_route_subprocess; oldroute_replay_excludes_generation' if a['ablation'].startswith('A_OLD') else 'fixed_deployment_route_subprocess',source=a['raw_result'],scope=a['scope']))
csvsave('solver_efficiency.csv',eff);save('solver_efficiency.json',eff)
# A matrix for the exact eight comparability axes. Methodacceptance differs from common evaluationgate.
mat=[]
for method in dict.fromkeys(r['algorithm'] for r in base):
 native='NATIVE_CONTINUOUS' in method
 mat.append(dict(method=method,candidate_model_pool='same per A/B/Ccell',GPU_memory_budget='same49140MiB safe memory',six_task_load_quality='same percell rates and admitted endpoints',replica_domain='same binary0/1',routing_domain='continuous; differs from K1Oracle' if native else 'same complete K1endpointdomain',performance_prediction='same frozen20A FIFO predict_scenarios in CPU backend',U_weights_normalization='same frozen20A weights and reference',stability_legality='common outputchecks rho<=.8 plus complete coverage/memory/drain; original internalacceptance retained',strict_search_domain=not native,
  internal_objective='V3 J routing/ranking plus native noiseaware meanP95memory 5%acceptance' if method.startswith('V3') else 'U' if method in ['CPU_U_ORACLE','U_ALIGNED_DP','20A_U_GRID','20A_NATIVE_CONTINUOUS_PRESERVED'] else 'fastestservice/static memory rule; no Usearch',historical_semantics_preserved=True))
csvsave('baseline_comparability_matrix.csv',mat)
# Figuredata. Native continuous failures/mismatches remain in csv, not forced into finiteRegretbars.
fig1=[r for r in base if r['strict_same_search_domain']];csvsave('paper_figures/figure1_regret.csv',fig1)
fig2=[r for r in abl if r.get('predicted_U') is not None];csvsave('paper_figures/figure2_ablation_utility.csv',fig2)
fig3=[r for r in eff if r['method'] in ['FINITE_U_ONLY_ORACLE','MODULE35_FROZEN']];csvsave('paper_figures/figure3_solver_time.csv',fig3)
fig4=[{**r,'kind':'arm'} for r in gpu]+[{**r,'kind':'paired_contrast'} for r in summary];csvsave('paper_figures/figure4_gpu_utility.csv',fig4)
# Complete raw evidence copied locally and indexed by figure; shared copies avoid unnecessary duplication.
for name,raws in [('figure1', [str(p) for p in (OUT/'raw_data').glob('*.json')]+[str(p) for p in (OUT/'raw_data').glob('*oracle_candidates.csv')]+[str(OUT/'baseline_cpu_results.json')]),
 ('figure2',[str(OUT/a['raw_result']) for a in abl if a.get('raw_result')]+[str(OUT/'ablation_cpu_results.json'),str(OUT/'route_candidate_coverage.csv')]),
 ('figure3',[str(p) for p in (OUT/'raw_data').glob('timing_*.json')]+[str(OUT/'solver_timing_repetitions.csv'),str(OUT/'raw_data/solver_runtime_comparison.csv')]),
 ('figure4',[str(p) for p in (OUT/'raw_data/gpu').glob('*')]+[str(OUT/'gpu_baseline_results.json'),str(OUT/'gpu_paired_summary.json'),str(OUT/'gpu_reuse_raw_audit.json'),str(OUT/'raw_data/frozen_gpu_manifest_v2.json'),str(OUT/'raw_data/statistical_analysis_plan.json')])]:
 save(f'paper_figures/{name}_raw_data.json',dict(complete_data_in_shared_raw_directory=True,files=[ref(p) for p in dict.fromkeys(raws)],native_domain_results_excluded_from_strict_regret_only=True if name=='figure1' else None))
# Quantitative findings without statisticalclaims for deterministic CPUgaps.
methods=['U_ALIGNED_DP','20A_U_GRID','V3_J_GRID_CONTROL','FASTEST_PATH_STATIC','MIN_MEMORY_COVER_FASTEST']
agg=[]
for split in ['development','historical_holdout']:
 for m in methods:
  rr=[r for r in base if r['split']==split and r['algorithm']==m and r.get('regret') is not None];agg.append(dict(split=split,method=m,n=len(rr),zero_regret=sum(abs(r['regret'])<1e-8 for r in rr),positive_regret=sum(r['regret']>1e-8 for r in rr),mean_regret=statistics.mean(r['regret'] for r in rr),max_regret=max(r['regret'] for r in rr)))
csvsave('all_scenario_aggregate.csv',agg)
cs=[r for r in eff if r['method']=='MODULE35_FROZEN' and r['pool']=='C'];arows=[r for r in abl if r['ablation']=='B_FIXED_M35_DEPLOYMENT_FASTEST'];newrows=[r for r in base if r['algorithm']=='U_ALIGNED_DP'];fastrows=[r for r in base if r['algorithm']=='FASTEST_PATH_STATIC']
findings=dict(cells=12,Module35_all_zero=all(abs(r['regret'])<1e-8 for r in newrows),fastest_zero=sum(abs(r['regret'])<1e-8 for r in fastrows),fastest_positive=sum(r['regret']>1e-8 for r in fastrows),
 candidate_recovery=coverage,route_only_positive=sum(r['full_minus_variant_U']>1e-8 for r in arows),route_only_zero=sum(abs(r['full_minus_variant_U'])<1e-8 for r in arows),route_only_negative=sum(r['full_minus_variant_U']<-1e-8 for r in arows),
 C_seconds_range=[min(r['seconds_median'] for r in cs),max(r['seconds_median'] for r in cs)],C_speedup_range=[min(r['speedup_vs_finite_U_only_oracle'] for r in cs),max(r['speedup_vs_finite_U_only_oracle'] for r in cs)],GPU_pairs=summary)
save('quantitative_findings.json',findings)
# Documents in the requested direct research style.
def md(name,text):(OUT/name).write_text(text.strip()+'\n')
matrows='\n'.join('| '+ ' | '.join(str(r[k]) for k in ['method','replica_domain','routing_domain','strict_search_domain','internal_objective'])+' |' for r in mat)
md('baseline_comparability_audit.md',f'''# Paper Baseline and Ablation Evaluation：可比性审计

本轮复用 Module35 冻结的九个开发单元与 TEXT3 三池原始基线；新增固定部署诊断和同域 U-only Oracle 计时。完整八维矩阵见 `baseline_comparability_matrix.csv`。每条结果同时给出模型、副本、绝对QPS路由、比例、硬约束和来源。

所有主CPU有限域单元使用 A/B/C 同一池清单、49140 MiB、原质量准入、0/1副本、K=1完整端点路由、原20A FIFO120秒 phase=.37性能预测、原U权重与归一化、共同rho≤.8及完整drain约束。开发场景为LOW18、ASR32B、TEXT_HALF_MID18；后者实际Text=10QPS，保留原场景名。TEXT3是此前独立冻结的一种工作负载的三池验证；本轮诊断已知道其结果，不能将新增消融重新称为盲测。GPU持续稳定性不是CPU Gate。

| 方法 | 副本域 | 路由域 | 严格同搜索域 | 内部选择规则 |
|---|---|---|---|---|
{matrows}

原生20A：原MILP连续路由及42种种子生成；最终由原U评分。历史20A本身还包含特定历史候选复用，本轮使用既有Module34/35域适配驱动，不能称为重新执行原20A全部历史研究流程。20A网格控制只改变MILP变量单位/整数性，不补候选。原生终点即使偶然落入K1，也不能据此把原生搜索时间或Regret作为严格同域比较；保留 `selected_plan_reference_gap`，正式 `regret` 置空。

V3：J用于路由和候选排序，正式接受逻辑仍为原多指标噪声门槛；不能描述为全局单标量J优化器。CPU使用冻结预测后端。原生连续V3与独立J网格控制分开记录。原生的域不同或数值失败保留，不能替换为控制版成功。V3内部门槛不同于统一输出合法性Gate。

静态基线：最快服务路径采用既有 `m34.cpu.static`；最小显存覆盖加最快路由采用既有实现，完整合法验证后报告U。其历史微秒时间只包括配置构造，最终四情形预测在时间之外；不能把该数字称为带验证的端到端优化时间。

Module33：预算34567.5534MiB、K=2、限定Mixed、Whisper-small仅LID、rho<1名义可行性，池SMALL6/LARGE7；与本轮49140MiB、K=1、A/B/C完整结构域不同。历史速度或Regret仅作背景，不混入同域均值。保留33 Gate。G-only Retention：P0/P1/P2池不同、LOW18与比例缩放开发负载；有效6单元最快启发式同最优，最低1/8负载缺LID预测样本、U=null，不是成功保持或0Regret。仅作为负结果背景，不并入主12单元统计。

严格组件消融的边界：冻结35的K1任务束DP没有候选扩展开关；A不能在保持全局部署搜索机制不变的情况下直接恢复20A MILP候选机制。A采用固定旧部署的候选覆盖诊断；只替换完整路由候选集合，旧生成器的完整候选和时间均保留，不能叫作全局单变量消融。B固定35完整部署，以同一DP核心限制到合法最快路由，保持驻留闲置模型显存计费，避免删模型产生多变量差异。它检验路由组件，不检验全局部署搜索；全局对照由既有最快路径基线提供。C无可关闭的共置修正，跳过。未开发新预测器或优化器。

GPU：直接复用34的A_TEXT10、B_TEXT10、C_ASR32B三个冻结同域配对，逐条核验35/最快/20A/V3完整配置等价、请求轨迹hash和内容、240秒窗口、30秒预热、49140MiB、原U和稳定性Gate。每个方法别名共享同一配置测量，不计作新的独立重复。复用34原配对顺序与多重校正。35 B→C跨会话收益保留为扩池背景，候选域与顺序控制不相同，不写为同域算法优势。GPU未新增测量。

求解效率：`solver_efficiency.csv`同时保存历史后端时间、新U-only Oracle三次冷缓存计时、固定部署路由子过程时间。U-only枚举去掉J诊断，预测公式与硬约束不变。状态单位不同：DP前缀label、Oracle完整路线leaf、MILP生成候选和内部节点不能互换。未记录的MILP内部节点、原生J调用数为null，不伪造。质量是历史能力准入，不等于重新测量质量分数。原20A `UTILITY_MODEL_INSUFFICIENT` Gate保持有效。
''')
aggrows='\n'.join(f"| {r['split']} | {r['method']} | {r['zero_regret']}/{r['n']} | {r['mean_regret']:.6f} | {r['max_regret']:.6f} |" for r in agg)
coverrows='\n'.join(f"| {r['scenario']}/{r['pool']} | {r['old_U']:.6f} | {r['expanded_U']:.6f} | {r['recovered_U']:.6f} | {r['split']} |" for r in coverage)
gpurows='\n'.join(f"| {r['group']} | {r['delta_U']:+.6f} | [{r['CI95_low']:+.6f}, {r['CI95_high']:+.6f}] | {r['holm_adjusted_p']:.6g} | {r['verdict']} |" for r in summary)
effrows='\n'.join(f"| {r['scenario']}/{r['pool']} | {r['seconds_median']:.6f} | {next(x['seconds_median'] for x in eff if x['scenario']==r['scenario'] and x['pool']==r['pool'] and x['method']=='FINITE_U_ONLY_ORACLE'):.6f} | {r['speedup_vs_finite_U_only_oracle']:.2f} | {r['search_states']} | {r['worker_prediction_calls']} | {r['full_utility_evaluations']} |" for r in eff if r['method']=='MODULE35_FROZEN')
md('paper_ready_findings.md',f'''# 论文 Evaluation：可直接使用的结果与证据边界

本轮输出12个CPU单元×8方法=96条基线，60条消融/跳过记录、三次重复的同域U-only效率测量和18个复用GPU独立窗口（9对）。未修改Module35、历史Gate、权重、负载、模型池或性能预测。九个开发单元与一种历史独立TEXT3工作负载的三池分开报告；不从中挑选容易获胜的单元计算平均优势。

## 1. Module35改进什么

冻结K1算法以任务束FIFO成本分解和Pareto动态规划联合选模型、测量副本与完整任务路径。相对20A的线性MILP种子加有限U评分，它覆盖原先未生成的队列避让路径；四个开发缺口全部补齐，同域预测最优9/9，TEXT3三池3/3。该最优性限于冻结预测器与有限0/1、K1域。V3网格控制的J路由/排名与FIFO U不一致，开发最大U Regret=0.355421；还包含性能代理差别，不能全归为目标权重。相对最快路径，35有时选择服务时间稍慢但队列效用更高的任务路由，有时选择显存成本更低的完整部署。

| 集合 | 方法 | 0Regret | 平均Regret（全部单元） | 最大Regret |
|---|---|---|---|---|
{aggrows}

原生20A/V3没有严格同域Regret；终点参考差与失败见基线CSV。不能把网格控制写为未经修改的原生版本，也不能删除失败。CPU数值确定性差异不是统计显著性。

## 2. 优势、无差异与负结果

相对20A网格：开发4/9有预测优势、5/9同最优；TEXT3 A/B有优势、C相同。相对最快路径：开发7/9有优势、2/9相同；TEXT3三池都有预测优势。相对最小显存覆盖加最快路由：35在C池和部分低负载单元无独占效用优势。G-only Retention有效6单元的20A与最快路径也同最优，负结果保持；不加入主12单元的平均数。

GPU只有A_TEXT10达到原多重校正后的显著优势；B_TEXT10与C_ASR32B的区间跨0，标记inconclusive，不解释为统计等价或成功。Module35并不在所有场景优于简单基线，且若忽略验证成本，简单启发式配置构造更快。

## 3. 路由覆盖对Regret的贡献

A是匹配固定部署的候选覆盖诊断；没有真实独立的候选扩展开关，不标为保持全局搜索机制不变的消融。下面包括全部12单元，避免只报四个正例。两侧保持模型、副本、资源、质量、预测U和Gate，只更换旧生成的完整路由集合与35完整路由搜索。全部旧生成集内U排名损失为0。四个开发缺口的1.281830、0.041308、10.844309、1.444254 U均由候选遗漏解释，覆盖修复回收100%；其他五个开发单元为0。TEXT3 A/B另回收12.940414和0.740633，C为0。这是CPU搜索归因，不是共置误差或GPU因果份额。

| 场景/池 | 旧路由候选U | 完整路由U | 回收U | 数据属性 |
|---|---|---|---|---|
{coverrows}

## 4. U路由是否优于静态路由

B在35冻结选出的完整部署上，调用相同DP核心，合法最快路径通过任务束mask限制；硬覆盖、驻留显存、任务质量、rho和drain均保留。6/12正收益、6/12无差异、0/12负收益。四个开发正例与A一致，另两个为TEXT3 A/B。C四个单元的路由控制全相同；全局最快路径基线在C的四个小Regret（0.004391、0.013090、0.019999、0.017764）来自完整部署差异，不能归给固定部署路由优化。此B检验的是条件于35部署的路由贡献；没有证明保留全局部署DP的开关式消融，更没有证明对任意静态配置都占优。

## 5. 共置干扰消融

C跳过。冻结predict_scenarios使用单模型服务画像、测量副本容量与周期FIFO队列；共置/尾延迟未测量的不确定性明确为HIGH_UNMEASURED_COLOCATION_AND_SERVICE_TAIL。没有独立共置修正可关闭。1.25/2服务倍率是诊断压力包络，不是正式可开关的共置机制。关闭队列、改服务时间或改rho Gate会混入其他变量，因此未实施。不能量化共置机制收益；这是论文局限。

## 6. 同域效率与效用速度权衡

下面为新同域U-only枚举和冻结35的三次冷缓存中位数；相同CPU环境、不加载GPU、输入/输出序列化不计时。原历史Oracle还计算J诊断，其计时保存在原始副本，未覆盖。35的全效用评价是一次最终完整配置验证（内部4情形预测），此前评估49/53/60个任务束；不能称为仅一次计算或把bundle隐藏。A/B/C有限Oracle分别8/32/8154个经过覆盖显存筛选的完整路线候选，每个做4情形预测；部署向量分别8/32/1024。DP状态数与路线数单位不同，不用状态计数直接宣称复杂度倍率。

| 场景/池 | 35秒 | U-only Oracle秒 | 加速比 | DP状态 | bundle评价 | 完整U评价 |
|---|---:|---:|---:|---:|---:|---:|
{effrows}

C池35新中位数{findings['C_seconds_range'][0]:.6f}–{findings['C_seconds_range'][1]:.6f}秒，U-only同域加速{findings['C_speedup_range'][0]:.1f}–{findings['C_speedup_range'][1]:.1f}倍。历史1184–1508倍保留原含义，新数据单独标记。三次样本是描述性计时，没有提供总体运行时间CI或渐近定理。A小池枚举仅几十毫秒，绝对优势小；启发式构造更快但可能Regret大。20A/V3历史后端时间在效率CSV中，测量批次不同不与本轮新Oracle计算严格加速比。固定部署路由子过程单列，不能代替联合端到端优化时间。K>1只支持固定部署离散路由搜索，本次未重跑分流研究；不宣称连续分流联合全局最优。

## 7. GPU是否支持相对基线的提升

CPU完成后决定直接复用34；原始18窗口、完整同轨迹9对重新做原Gate与U算术审计。模型/副本/任务QPS与35及最快路径逐条一致；两方法别名不产生新的独立测量。30秒预热、240秒正式窗口、60秒drain、独立种子340201/202/203；原配对顺序、环境阈值、稳定性与Holm三项检验均保留。完整请求必须完成，否则U=null。吞吐损失、pooled Mean/P95 E2E、每任务延迟、安全显存和采样峰值均在GPU CSV；U使用各任务归一化，不能用pooled P95直接代入。原协议没有引入新的任意数值SLO阈值，保持原rho/drain/质量/稳定性Gate。

| 同域比较 | 平均ΔU（35−最快） | 配对95%t CI | Holm p | 判定 |
|---|---:|---|---:|---|
{gpurows}

A_TEXT10：35完整方案等于原CPU Oracle/V3网格控制；最快路径方案等于20A，因此该效用优势不是35对V3的优势，也不是独立运行35的新的因果试验。B同20A对照只有微小实测差，跨0。C_ASR32B中35与20A、V3网格和最小显存启发式同配置，优于全局最快路径的预测微差未得到明确GPU支持。

35 B→C配对ΔU=+7.415061，CI[5.333009,9.497114]，来自扩池且跨会话；C方案也被20A及最小显存启发式选中。不能用它声称35相对基线的因果优势。没有新增GPU测量或结果导向追加；CPU有限最优不等于真实GPU全局最优。

## 8. 可以写入论文与必须保留的局限

可以写：冻结K1预测有限域12/12最优；完整全部场景的基线表；旧候选覆盖损失及固定部署路由诊断；U-only同域绝对时间、实际调用与加速比；A_TEXT10冻结配置对的实测效用提升；B/C的不确定结果；多个合理启发式同最优及G-only负结果。

不能写：普遍优于启发式、原生V3的严格有限Regret/速度优势、共置机制消融有效、全局单变量A/B开关消融、未见消融盲测、真实GPU全池最优、B→C收益归因于新算法、统计跨0等于无差异、K>1/连续域端到端联合最优。原20A性能模型不足Gate未消除。历史画像/周期到达与真实Poisson/共置环境差别、小n=3、有限模型池与仅一种独立工作负载、历史配置等价复用、固定部署诊断均必须保留。

图1只画严格同域Regret，原生结果保留CSV；图2明确固定部署诊断；图3展示U-only同域时间；图4展示三项配对效用与跨0区间。每图有PNG/PDF/CSV及完整原始数据索引。实验收尾后停止，不自动开启下一实验。
''')
md('final_summary.md',f'''# Paper Baseline and Ablation Evaluation：完成与停止

核心CPU12单元、96条基线、候选覆盖/固定部署路由诊断、三次U-only同域计时、18个历史GPU窗口审计已完成。产物位于 `results/paper_baseline_ablation/`。Module35及旧Gate保持冻结；无新增优化器、预测器、模型池、画像、工作负载或GPU测量。方法和负结果全部保留。

1. 35补齐20A合法U候选覆盖，K1预测Regret12/12为0；相对V3网格控制避免部分J/FIFO-U不一致，相对最快路径10/12有预测优势、2/12相同。
2. GPU A_TEXT10有显著配对优势：+21.658291 U，95%CI[15.152162,28.164419]；B_TEXT10和C_ASR32B跨0。CPU正差不是统计显著性。
3. 原四开发路由遗漏的Regret全部回收：1.281830、0.041308、10.844309、1.444254 U；TEXT3 A/B再回收12.940414、0.740633。A仅是固定部署候选覆盖诊断，严格全局开关消融不可实施。
4. B固定35完整部署：U路由6/12优于合法最快路径，6/12相同，0负；未保持全局部署搜索的开关式消融，限制明确标注。
5. C无可独立关闭共置组件，跳过；没有数值收益结论。
6. 同域U-only C池35耗时{findings['C_seconds_range'][0]:.6f}–{findings['C_seconds_range'][1]:.6f}秒，加速{findings['C_speedup_range'][0]:.1f}–{findings['C_speedup_range'][1]:.1f}倍；限K1，路由子过程与端到端求解分开。全部绝对时间和实际计数见效率CSV。
7. 真实GPU只支持A冻结配置对的相对最快/20A优势；该35方案与V3控制相同。B/C不确定；B→C扩池收益不作算法优势。
8. 可以写有限域正确性、全场景基线、路由覆盖归因、固定部署诊断和同域效率；必须保留预测器不足、单工作负载holdout、小样本GPU、配置等价复用、A/B不可全局独立关闭、C缺失和非真实全局最优的局限。

详见 `paper_ready_findings.md`、`baseline_comparability_audit.md`、`experiment_protocol.json`、`final_gate.json`。所有正、负、无差异单元均纳入；本轮结束，不开启下一实验。
''')
# Independent baseline membership/legality and diagnostics audit; allskip/null rows explicit.
checks=[]
for r in base:
 if not r.get('deployment'):continue
 d=read(sp['pools'][r['pool']]['path']);rates=next(z['rates'] for z in protocol['scenarios'] if z['scenario']==r['scenario']);x=r['deployment'];rt=r['routing'];p=predict_scenarios(d,x,rt,rates,budget,refs,weights)
 check=dict(scenario=r['scenario'],pool=r['pool'],method=r['algorithm'],capability=all(m in x and t in d['models'][m]['supported_tasks'] for t,row in rt.items() for m,q in row.items() if q>1e-9),replicas=all(n==1 for n in x.values()),conservation=all(abs(sum(rt[t].values())-rates[t])<1e-6 for t in rates),
  memory=p['safe_memory_mib']<=budget+1e-6,stability=p['max_rho']<=.8+1e-8,nominal_drain=p['incomplete_after_60s_drain']==0,finite_grid=all(abs(q/rates[t]-round(q/rates[t]))<1e-7 for t,row in rt.items() for q in row.values()))
 if r['strict_same_search_domain']:assert all(v for k,v in check.items() if isinstance(v,bool))
 checks.append(check)
save('independent_cpu_legality_audit.json',checks)
# All protected scientific artifacts unchanged, SHA256 and mtime. Run after compute so no overlapping CPUtimings.
changed=[];before=read(OUT/'history_sha256_before.json')
for prev in before:
 now=ref(prev['path'])
 if now['sha256']!=prev['sha256'] or now['mtime_ns']!=prev['mtime_ns']:changed.append(prev['path'])
assert not changed,changed
save('history_integrity_after.json',dict(passed=True,files=len(before),changed=[],scope='all scientificfiles in original20code, core, methodV2, v3, results20,33,34,35,retention excluding transientwork/__pycache__; fullSHA256 and mtime; no modelweight rewrite or profiling executed'))
gate=dict(status='COMPLETE_WITH_EXPLICIT_COMPONENT_LIMITATIONS',completed_local=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),stages=dict(A='COMPLETE',B='COMPLETE_REUSED_12_CELLS',C='FIXED_DEPLOYMENT_DIAGNOSTICS_COMPLETE; GLOBAL_A_B_SWITCHES_UNAVAILABLE; C_SKIPPED',D='COMPLETE_RAW_AUDITED_REUSE; NO_NEW_GPU',E='COMPLETE_U_ONLY_SAME_DOMAIN',F='TABLES_AND_REPORTS_COMPLETE; FIGURES_PENDING'),
 CPU_ORACLE_MATCHED_12_OF_12=findings['Module35_all_zero'],FASTEST_HEURISTIC_ALSO_OPTIMAL_COUNT=findings['fastest_zero'],GPU_RELATIVE_FASTEST_SUPPORTED_SCENARIOS=['A_TEXT_HALF_MID18'],GPU_COMPARISON_INCONCLUSIVE_SCENARIOS=['B_TEXT_HALF_MID18','C_ASR32B'],
 STRICT_GLOBAL_SINGLE_VARIABLE_ABLATION_A_EXECUTED=False,STRICT_GLOBAL_SINGLE_VARIABLE_ABLATION_B_EXECUTED=False,FIXED_DEPLOYMENT_ROUTE_DIAGNOSTICS_LEGAL=True,COLOCATION_ABLATION_AVAILABLE=False,COLOCATION_ABLATION_SKIPPED=True,
 NO_NEW_OPTIMIZER=True,FROZEN_35_CODE_UNCHANGED=True,HISTORICAL_RESULTS_GATES_UNCHANGED=True,UTILITY_WEIGHTS_FORMULA_UNCHANGED=True,CANDIDATE_POOL_UNCHANGED=True,NO_WORKLOAD_RETUNING=True,
 NEW_GPU_SCENARIOS=0,HISTORY_INTEGRITY=ref(OUT/'history_integrity_after.json'),ORIGINAL_20A_UTILITY_MODEL_INSUFFICIENT=True,REAL_GPU_GLOBAL_OPTIMALITY_PROVEN=False,
 ALL_SCENARIOS_INCLUDED=True,CORE_EVIDENCE_READY=True,PAPER_ASSETS_READY=False,stop=True,no_next_experiment=True,deadline_local='2026-10-16 Asia/Shanghai')
save('final_gate.json',gate)
save('environment_record.json',dict(experiment_python=sys.executable,python=sys.version,numpy=np.__version__,scipy=__import__('scipy').__version__,platform=platform.platform(),packages_installed=False,write_confinement='existing35 Landlock+auditguard with processlocal outputroot; frozen source unchanged',GPU_new_measurements=0))
print('FINALIZE_DATA_PASS',findings,flush=True)
