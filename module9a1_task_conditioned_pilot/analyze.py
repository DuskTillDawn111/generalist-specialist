"""Parameter-free signature distances and precommitted pilot statistics."""
import csv,itertools,json,math,re,sqlite3,subprocess
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance,spearmanr
from .prepare import ROOT,OUT,DEV,load,save,csvsave,sha,now

def kernel_class(name):
 s=name.lower()
 for label,pattern in [('attention','flash|attention|fmha'),('gemm','gemm|matmul|cutlass|cublas'),('convolution','cudnn|convolution|conv'),('fft','fft'),('softmax','softmax'),('normalization','layernorm|layer_norm|rmsnorm|rms_norm|norm'),('reduction','reduce|reduction'),('elementwise','elementwise|pointwise|unrolled|vectorized'),('copy_index_cast','copy|cast|gather|scatter|index|slice')]:
  if re.search(pattern,s):return label
 return 'other:'+name.split('<')[0].split('(')[0].strip()
def js(p,q):
 p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float);p=p/p.sum(axis=-1,keepdims=True);q=q/q.sum(axis=-1,keepdims=True);m=(p+q)/2
 with np.errstate(divide='ignore',invalid='ignore'):
  a=np.where(p>0,p*np.log2(p/m),0);b=np.where(q>0,q*np.log2(q/m),0)
 return .5*(a.sum(axis=-1)+b.sum(axis=-1))
def counter_distance(a,b):
 keys=sorted(set(a)|set(b));return float(js([a[k] for k in keys],[b[k] for k in keys]))
def occupancy(rows):
 props=load(OUT/'profiler_capability_audit.json')['device_attributes'];names=['major','minor','max_threads_per_block','max_threads_per_sm','regs_per_block','regs_per_sm','warp_size','shared_per_block','shared_per_sm','sm_count','shared_optin_per_block','reserved_shared_per_block'];base=' '.join(str(props[k]) for k in names);configs=sorted(set((r['threads_per_block'],r['registers_per_thread'],r['static_shared_bytes'],r['dynamic_shared_bytes']) for r in rows));lines=[base+' '+' '.join(map(str,c)) for c in configs]
 out=subprocess.check_output([str(ROOT/'module9a1_task_conditioned_pilot/occupancy')],input='\n'.join(lines)+'\n',text=True).splitlines();lookup={}
 for c,line in zip(configs,out):
  rc,blocks,value,flags=line.split();lookup[c]={'status':'OFFICIAL_CALCULATOR_DEFAULT_FUNCTION_ATTRIBUTES' if int(rc)==0 else 'UNAVAILABLE_CALCULATOR_ERROR','blocks_per_sm':int(blocks) if int(rc)==0 else None,'theoretical_occupancy':float(value) if int(rc)==0 else None,'limiting_factors':int(flags),'rc':int(rc)}
 for r in rows:r.update(lookup[(r['threads_per_block'],r['registers_per_thread'],r['static_shared_bytes'],r['dynamic_shared_bytes'])])

def extract(path):
 c=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;strings={r['id']:r['value'] for r in c.execute('select * from StringIds')};ranges=[]
 for row in c.execute('select * from NVTX_EVENTS'):
  r=dict(row);text=r.get('text') or strings.get(r.get('textId'),'')
  if text.startswith('PILOT|') and '|target:' in text and r.get('end') is not None:ranges.append({'start':r['start'],'end':r['end'],'label':text,'sample_index':int(text.split('target:')[1])})
 assert len(ranges)==3,('Incomplete NVTX request ranges',path,len(ranges));rows=[]
 for row in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start,streamId,correlationId'):
  r=dict(row);request=next((q for q in ranges if q['start']<=r['start'] and r['end']<=q['end']),None)
  if request is None:continue
  name=strings[r['demangledName']];grid=[r['gridX'],r['gridY'],r['gridZ']];block=[r['blockX'],r['blockY'],r['blockZ']];rows.append({'sample_index':request['sample_index'],'kernel_sequence_index':len(rows),'start_ns_profiled':r['start'],'duration_ns_profiled_diagnostic_only':r['end']-r['start'],'kernel_name':name,'kernel_class':kernel_class(name),'stream_id':r['streamId'],'registers_per_thread':r['registersPerThread'],'static_shared_bytes':r['staticSharedMemory'],'dynamic_shared_bytes':r['dynamicSharedMemory'],'grid':grid,'block':block,'threads_per_block':math.prod(block),'grid_blocks':math.prod(grid),'shared_limit_config':r.get('sharedMemoryLimitConfig'),'cluster_dimensions':[r.get('clusterX'),r.get('clusterY'),r.get('clusterZ')]})
 assert rows,('No kernels',path);occupancy(rows);c.close();return rows

def profiles():
 p=load(OUT/'measurement_protocol_freeze.json');records=[];summary=[]
 for model,tasks in p['tasks'].items():
  for task in tasks:
   combined=[];samples=[]
   for seed in p['repetition_seeds']:
    tag=f'{model}__{task}__none__{seed}';directory=OUT/'task_path_kernel_profiles/kernel';rows=extract(directory/(tag+'.sqlite'));combined.extend(rows);(directory/(tag+'.kernels.json')).write_text(json.dumps(rows,ensure_ascii=False)+'\n')
    for i in range(3):
     rr=[r for r in rows if r['sample_index']==i];classes=[r['kernel_class'] for r in rr];a={'model':model,'task':task,'seed':seed,'sample_index':i,'kernel_count':len(rr),'class_counts':dict(Counter(classes)),'resource_counts':dict(Counter(json.dumps([r['registers_per_thread'],r['static_shared_bytes'],r['dynamic_shared_bytes'],r['block'],r['grid']],sort_keys=True) for r in rr)),'bigram_counts':dict(Counter(json.dumps(x) for x in zip(classes,classes[1:]))),'occupancy_values':[r['theoretical_occupancy'] for r in rr if r['theoretical_occupancy'] is not None]};samples.append(a);records.append(a)
   classes=Counter(r['kernel_class'] for r in combined);occ=[r['theoretical_occupancy'] for r in combined if r['theoretical_occupancy'] is not None];summary.append({'model':model,'task':task,'profile_repetitions':3,'profile_requests':9,'kernel_occurrences':len(combined),'unique_kernel_names':len(set(r['kernel_name'] for r in combined)),'kernel_class_frequency':{k:v/len(combined) for k,v in classes.items()},'kernel_name_occurrence_counts':dict(Counter(r['kernel_name'] for r in combined)),'theoretical_occupancy_mean':float(np.mean(occ)) if occ else None,'occupancy_metric_coverage':len(occ)/len(combined),'resource_configuration_count':len(set(json.dumps([r['registers_per_thread'],r['static_shared_bytes'],r['dynamic_shared_bytes'],r['block'],r['grid']]) for r in combined)),'kernel_count_per_request':[r['kernel_count'] for r in samples],'kernel_timing_used_as_ground_truth':False,'occupancy_caveat':'NVIDIA CUDA 11.8 standalone occupancy calculator supports Ada 8.9. Device limits from driver; traced registers/shared/block dimensions. Unknown function barrier count uses official conversion default 1; function max-thread limit uses device maximum; this is theoretical metadata reconstruction, not achieved occupancy or validated interference prediction.'})
 csvsave('task_path_kernel_summary.csv',summary);save('kernel_request_signatures.json',{'records':records,'kernel_class_rules_source_sha256':sha(__file__),'frequency_definition':'per-launch counts; bootstrap uses request-normalized class frequencies, every profiled request retained'})
 rng=np.random.default_rng(902003);B=p['bootstrap_resamples'];distances=[]
 for model,tasks in p['tasks'].items():
  for ta,tb in itertools.combinations(tasks,2):
   a=[r for r in records if r['model']==model and r['task']==ta];b=[r for r in records if r['model']==model and r['task']==tb];keys=sorted(set().union(*(r['class_counts'] for r in a+b)));vector=lambda rr:np.array([[r['class_counts'].get(k,0)/r['kernel_count'] for k in keys] for r in rr]);va,vb=vector(a),vector(b)
   def boot(v):
    # Independent run blocks, then request sample within each selected block.
    z=v.reshape(3,3,-1);runs=rng.integers(0,3,size=(B,3));items=rng.integers(0,3,size=(B,3,3));return z[runs[:,:,None],items].mean(axis=(1,2))
   ba,bb=boot(va),boot(vb);between=js(ba,bb);floor=max(float(np.quantile(js(boot(va),boot(va)),.95)),float(np.quantile(js(boot(vb),boot(vb)),.95)));ci=np.quantile(between,[.025,.975]);resource=lambda rr:sum((Counter(r['resource_counts']) for r in rr),Counter());bigrams=lambda rr:sum((Counter(r['bigram_counts']) for r in rr),Counter());oa=sum((r['occupancy_values'] for r in a),[]);ob=sum((r['occupancy_values'] for r in b),[]);counts_a=np.array([r['kernel_count'] for r in a]);counts_b=np.array([r['kernel_count'] for r in b]);stable=bool(ci[0]>floor)
   distances.append({'model':model,'task_a':ta,'task_b':tb,'kernel_class_jsd_bits':float(js(va.mean(axis=0),vb.mean(axis=0))),'kernel_class_jsd_95CI_low':float(ci[0]),'kernel_class_jsd_95CI_high':float(ci[1]),'within_task_bootstrap_jsd_p95_floor':floor,'stable_structural_difference':stable,'class_total_variation':float(np.abs(va.mean(axis=0)-vb.mean(axis=0)).sum()/2),'resource_config_jsd_bits':counter_distance(resource(a),resource(b)),'occupancy_wasserstein':float(wasserstein_distance(oa,ob)) if oa and ob else None,'sequence_bigram_jsd_bits':counter_distance(bigrams(a),bigrams(b)),'normalized_kernel_count_difference':float(abs(counts_a.mean()-counts_b.mean())/max(counts_a.mean(),counts_b.mean())),'profile_requests_per_task':9,'profile_independent_runs_per_task':3,'duration_weighted':False})
 csvsave('task_path_kernel_distance.csv',distances);return distances

def slowdown():
 p=load(OUT/'measurement_protocol_freeze.json');rows=[];arrays={};rng=np.random.default_rng(p['bootstrap_seed']);B=p['bootstrap_resamples']
 for model,tasks in p['tasks'].items():
  for task in tasks:
   base=[]
   for seed in p['repetition_seeds']:
    raw=load(OUT/f'pair_measurements/standalone/{model}__{task}__none__{seed}.json');assert raw['status']=='COMPLETE';base.append([r['worker_latency_ms'] for r in raw['target_records']])
   for co in p['co_models']:
    pair=[];overlaps=[]
    for seed in p['repetition_seeds']:
     raw=load(OUT/f'pair_measurements/pair/{model}__{task}__{co}__{seed}.json');assert raw['status']=='COMPLETE';pair.append([r['worker_latency_ms'] for r in raw['target_records']]);overlaps.extend(r['active_overlap_ratio'] for r in raw['target_records'])
    a=np.array(base);b=np.array(pair);assert a.shape==b.shape==(3,30);s=float(b.mean()/a.mean());arrays[(model,task,co)]=(a,b);rows.append({'model':model,'task':task,'co_model':co,'n_repetitions':3,'samples_per_repetition':30,'standalone_latency_mean_ms':float(a.mean()),'pair_latency_mean_ms':float(b.mean()),'S_real':s,'gamma_real':s-1,'minimum_active_overlap_ratio':min(overlaps),'valid_overlap_protocol':min(overlaps)>=p['minimum_overlap_ratio'],'profile_timing_used':False,'frozen_catalog_s0_refit':False})
 csvsave('task_conditioned_slowdown.csv',rows);tests=[]
 for model,tasks in p['tasks'].items():
  for co in p['co_models']:
   for ta,tb in itertools.combinations(tasks,2):
    a1,b1=arrays[(model,ta,co)];a2,b2=arrays[(model,tb,co)];runs=rng.integers(0,3,size=(B,3));ix=rng.integers(0,30,size=(B,3,30))
    def bs(a,b):return b[runs[:,:,None],ix].mean(axis=(1,2))/a[runs[:,:,None],ix].mean(axis=(1,2))
    delta=bs(a1,b1)-bs(a2,b2);point=b1.mean()/a1.mean()-b2.mean()/a2.mean();ci=np.quantile(delta,[.025,.975]);valid=all(r['valid_overlap_protocol'] for r in rows if r['model']==model and r['co_model']==co and r['task'] in [ta,tb]);exclude=bool(ci[0]>0 or ci[1]<0);exceed=bool(abs(point)>p['historical_noise_floor']);tests.append({'model':model,'co_model':co,'task_a':ta,'task_b':tb,'Delta_S':float(point),'abs_Delta_S':float(abs(point)),'bootstrap_95CI':ci.tolist(),'CI_excludes_zero':exclude,'historical_noise_floor':p['historical_noise_floor'],'exceeds_historical_noise_floor':exceed,'valid_controlled_pair_overlap':valid,'TASK_CONDITIONED_INTERFERENCE_DETECTED':bool(exclude and exceed and valid),'bootstrap_samples':B,'bootstrap_unit':'independent run block and matched sample index; common repetition block across task paths; standalone and pair indices jointly resampled'})
 save('statistical_tests.json',{'frozen_rule':p['statistical_rule'],'tests':tests,'maximum_abs_task_conditioned_deltaS':max(t['abs_Delta_S'] for t in tests),'new_thresholds_fitted':False});return tests,rows

def final():
 distances=profiles();tests,rows=slowdown();p=load(OUT/'measurement_protocol_freeze.json');lookup={(d['model'],d['task_a'],d['task_b']):d for d in distances};pairs=[{'model':t['model'],'task_a':t['task_a'],'task_b':t['task_b'],'co_model':t['co_model'],'kernel_jsd':lookup[(t['model'],t['task_a'],t['task_b'])]['kernel_class_jsd_bits'],'absolute_interference_difference':t['abs_Delta_S']} for t in tests];rho,pval=spearmanr([r['kernel_jsd'] for r in pairs],[r['absolute_interference_difference'] for r in pairs]);save('mechanism_correlation.json',{'status':'INSUFFICIENT_FOR_CORRELATION','spearman_rho_descriptive_only':float(rho) if math.isfinite(rho) else None,'nominal_p_value_not_valid_for_clustered_inference':float(pval) if math.isfinite(pval) else None,'observations':pairs,'unique_task_pair_contrasts':len(distances),'independent_generalist_clusters':len(p['generalists']),'reason':'Only two Generalist clusters and shared-task contrasts; duplicating each kernel distance for two co-models does not create independent samples. Descriptive Spearman is reported, no statistical mechanism-correlation claim.','causality_claim':False})
 structural=any(d['stable_structural_difference'] for d in distances);interference=any(t['TASK_CONDITIONED_INTERFERENCE_DETECTED'] for t in tests);matches=[t for t in tests if t['TASK_CONDITIONED_INTERFERENCE_DETECTED'] and lookup[(t['model'],t['task_a'],t['task_b'])]['stable_structural_difference']];gate='GATE_STRONG_SUPPORT' if matches else ('GATE_PARTIAL' if structural or interference else 'GATE_NOT_SUPPORTED');distortion=[]
 for model,tasks in p['tasks'].items():
  for task in tasks:
   pro=[];base=[]
   for seed in p['repetition_seeds']:
    a=load(OUT/f'task_path_kernel_profiles/kernel/{model}__{task}__none__{seed}.json');b=load(OUT/f'pair_measurements/standalone/{model}__{task}__none__{seed}.json');pro.extend(r['worker_latency_ms'] for r in a['target_records']);base.extend(r['worker_latency_ms'] for r in b['target_records'][:3])
   distortion.append({'model':model,'task':task,'profiled_to_unprofiled_matched_mean_ratio':float(np.mean(pro)/np.mean(base)),'profiled_mean_ms':float(np.mean(pro)),'unprofiled_matched_mean_ms':float(np.mean(base)),'diagnostic_only':True,'includes_instrumentation_and_run_variability':True})
 save('profiler_timing_distortion_audit.json',{'paths':distortion,'profiled_latency_ground_truth_used':False,'frozen_s0_modified':False,'no_causal_overhead_precision_claim':True});save('pilot_gate.json',{'status':gate,'stable_kernel_structural_difference':structural,'task_conditioned_interference_detected':interference,'same_contrast_strong_evidence':matches,'recommend_full_Module9A':gate=='GATE_STRONG_SUPPORT','full_v3_started':False,'routing_revised':False,'candidate_ranking_revised':False,'8F2_heldout_validation_allowed':False,'next_action':'Stop this pilot; STRONG_SUPPORT recommends a separately authorized full Module 9A, PARTIAL audits causes, NOT_SUPPORTED stops this direction.'});account=load(OUT/'pilot_run_accounting.json');selection=load(OUT/'pilot_selection.json');noise=load(OUT/'noise_audit.json');maxdelta=max(t['abs_Delta_S'] for t in tests);lines=['# Module 9A-0 / 9A-1 — Task-Conditioned Interference Pilot','',f'状态：COMPLETE；gate：**{gate}**。8F-2 仅用于 development/failure evidence，永久禁止作为 v3 held-out final validation。','', '## 12 项最终回答','',f'1. Generalists：{selection["generalists"]}；12 个 v2 final occurrence：{selection["final_generalist_counts"]}；替换 accepted：{selection["accepted_replacement_counts"]}，评估计数：{selection["evaluated_replacement_counts"]}。',f'2. 全部任务：{selection["tasks"]}；来自所有冻结实测 states 的 active/admitted/nonzero routes，包括候选 LID，没有删任务。',f'3. Co-models：{selection["co_models"]}；12 个 v2 final routes 累计 r*s0 排名：{selection["co_ranking"]}。',f'4. Historical noise floor：**{noise["historical_noise_floor"]}** absolute Delta-S。最大同 state 重复 run-means null-ratio bootstrap Q95，阈值未根据新结果修改。',f'5. Stable kernel signature differences：{structural}；详见 task_path_kernel_distance.csv。',f'6. Fixed co-model task-conditioned slowdown detected：{interference}；8 个 task-pair/co-model comparisons 的独立重复 bootstrap 95% CI 和噪声门槛全部保存。',f'7. 最大 absolute Delta-S：**{maxdelta}**；不是人工追加模型/负载后选出的结果。',f'8. Kernel distance vs interference：INSUFFICIENT_FOR_CORRELATION；descriptive Spearman rho={float(rho) if math.isfinite(rho) else None}，仅 2 个 Generalist clusters，不能把共享任务和重复 co-model 当独立数据。',f'9. Profiler timing distortion diagnostic ratios：{ {r["model"]+"/"+r["task"]:r["profiled_to_unprofiled_matched_mean_ratio"] for r in distortion} }。profiler timing 从未作为 service ground truth 或 slowdown 分母；catalog s0 未修改。','10. Roomie primitive 可用性见 profiler_capability_audit.json：只审计 kernel resource metadata 和官方 occupancy calculator；不宣称完整 Roomie predictor 已验证。计数器权限受限不通过更换 CUDA/driver 绕过。',f'11. Gate：**{gate}**。',f'12. 是否建议完整 Module9A：{gate=="GATE_STRONG_SUPPORT"}。完成 pilot 后停止；完整 v3、路由/排序修订均未开始。','',f'运行统计：{account}。','', '## Task-conditioned slowdown','', '| Generalist | Task | Co-model | Standalone ms | Pair ms | S_real | gamma |','|---|---|---|---|---|---|---|']
 for r in rows:lines.append(f'| {r["model"]} | {r["task"]} | {r["co_model"]} | {r["standalone_latency_mean_ms"]:.6f} | {r["pair_latency_mean_ms"]:.6f} | {r["S_real"]:.6f} | {r["gamma_real"]:.6f} |')
 lines+=['','## Contrast tests','','| Generalist | Co | Tasks | Delta S | 95% CI | > historical floor | Detected |','|---|---|---|---|---|---|---|']
 for t in tests:lines.append(f'| {t["model"]} | {t["co_model"]} | {t["task_a"]} / {t["task_b"]} | {t["Delta_S"]:.6f} | {t["bootstrap_95CI"]} | {t["exceeds_historical_noise_floor"]} | {t["TASK_CONDITIONED_INTERFERENCE_DETECTED"]} |')
 lines+=['','## Interpretation limits','','This pilot is task-path isolation under a fixed active-pair characterization protocol, not a new benchmark mixture or memory/intensity resource point. Saturated co-backlog is identical across target tasks. Handler-time interval overlap is not proof of simultaneous SM execution; process time-slicing, CPU preprocessing and context effects may contribute. Kernel-signature/slowdown coexistence supports a hypothesis, not a causal attribution or validated analytical predictor.','', 'Roomie kernel/occupancy/pairwise ideas, USHER/iGniter interference-aware resources, and Cascadia/RouterWise deployment-routing coupling remain prior art. The proposed chain remains HYPOTHESIZED GAP; no first/novel claim.','', 'Historical noise is transferred conservatively from multi-model steady runs. Limited repetitions/input coverage and profiler overhead constrain conclusions. All tasks and failed/negative contrasts are retained.']
 (OUT/'module9a1_summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':final()
