import numpy as np,collections,math
from .common import *
from module26.common import arithmetic,cohort
from module25.common import utility_inputs

def assess(raw):
 _,report=configure();o=report.observation(raw)
 ph=raw.get('phases',{}).get('measurement');checks={'valid_session':raw.get('valid_session',False),'durable_journal':o['journal_audit']['passed'],'nonblocking_control':raw.get('control_nonblocking',False)}
 if ph:
  w=ph['window'];ss=[s for s in raw['telemetry']['samples'] if w['start_time_ns']<=s['time_ns']<=w['end_time_ns']]
  ts=[s['time_ns'] for s in ss];un=[s['unix_ns'] for s in ss]
  checks.update(exact_600s=w['duration_seconds']==600 and w['end_time_ns']-w['start_time_ns']==600000000000,
   trace_matched=o['matched_trace'],native_timestamps=o['native_timings_consistent'],full_cohort=o['offered']==o['completed'],
   sensor_values=bool(ss) and all(all(s.get(k) is not None and np.isfinite(s[k]) for k in ['time_ns','unix_ns','used_mib','owned_used_mib','gpu_util_percent','memory_util_percent','worker_CPU_core_percent','host_RSS_bytes','temperature_gpu_c','sm_clock_mhz']) for s in ss),
   monotonic_timestamps=all(b>a for a,b in zip(ts,ts[1:])) and all(b>a for a,b in zip(un,un[1:])),
   no_external_GPU=all(not s['foreign_pids'] for s in ss),
   telemetry_coverage=bool(ts) and (ts[0]-w['start_time_ns'])/1e9<2 and (w['end_time_ns']-ts[-1])/1e9<2 and max(np.diff(ts)/1e9,default=0)<5,
   bounded_drain=w['drain_seconds']<=60.1,
   unix_perf_alignment=bool(ss) and max(abs((s['unix_ns']-w['start_unix_ns'])-(s['time_ns']-w['start_time_ns']))/1e9 for s in ss)<1)
  d,refs=utility_inputs();v=arithmetic(raw,read(OUT/'selected_pool_catalog.json'),refs,d['weights']);o['raw_arithmetic']=v
  checks['utility_recomputed']=o.get('components',{}).get('U') is None or abs(o['components']['U']-v['raw_arithmetic_U'])<1e-12
  o['telemetry_max_gap_seconds']=max(np.diff(ts)/1e9,default=None)
 o['validation']=checks;o['technical_valid']=all(checks.values());o['eligible']=bool(o['eligible'] and o['technical_valid'])
 if not o['eligible']:
  o['U']=None
  if 'components' in o:o['components']['U']=None
 o['window_status']='VALID' if o['technical_valid'] else 'INVALID'
 return o

def bootstrap(raws,obs):
 by={r['slot']['run_id']:r for r in raws};oo={o['run_id']:o for o in obs};base=by['A0']['phases']['measurement']['request_records'];d,refs=utility_inputs();rates=by['A0']['slot']['lambda_rps'];total=sum(rates.values())
 bins=[np.array([i for i,r in enumerate(base) if int(r['frozen_offset_seconds']//30)==b],dtype=int) for b in range(20)]
 rng=np.random.default_rng(310031);selections=[np.concatenate([bins[j] for j in rng.integers(0,20,size=20)]) for _ in range(2000)]
 taskmask={t:np.array([r['task']==t for r in base]) for t in rates};end={a:r['phases']['measurement']['window']['end_time_ns'] for a,r in by.items()}
 def us(a,ix):
  rr=by[a]['phases']['measurement']['request_records'];lat=np.array([r['e2e_latency_ms'] for r in rr]);inside=np.array([r['completion_time_ns']<=end[a] for r in rr])
  L=0.;P=0.
  for t in rates:
   vals=lat[ix[taskmask[t][ix]]]
   if not len(vals):return None
   L+=rates[t]/total*float(np.mean(vals))/refs[t];P+=rates[t]/total*float(np.percentile(vals,95))/refs[t]
  T=1-min(1.,sum(inside[ix])/(total*600));M=oo[a]['components']['M_norm']
  return -d['weights']['T']*T-d['weights']['L']*L-d['weights']['P']*P-d['weights']['M']*M
 result=[]
 for aid in ['A1','A2']:
  vals=[]
  for ix in selections:
   u0=us('A0',ix);u1=us(aid,ix)
   if u0 is not None and u1 is not None:vals.append(u1-u0)
  result.append(dict(arm=aid,n_bootstrap=len(vals),paired_block_seconds=30,diagnostic_percentile_95_low=float(np.percentile(vals,2.5)),diagnostic_percentile_95_high=float(np.percentile(vals,97.5)),fraction_delta_positive=float(np.mean(np.array(vals)>0)),independent_replication=False,scope='Within one paired trajectory; serial arm order, thermal drift and between-run uncertainty not identified'))
 return result

def generate():
 f=read(OUT/'experiment_manifest.json');_,report=configure();raws=[read(OUT/'raw_runs'/(s['run_id']+'.json')) for s in f['slots'] if (OUT/'raw_runs'/(s['run_id']+'.json')).exists()];obs=[assess(r) for r in raws]
 save('run_observations.json',obs);by={o['run_id']:o for o in obs};comparisons,interference,changes=report.compare(raws,obs)
 comparisons=[r for r in comparisons if r['candidate']=='C1'];save('pair_comparisons.json',comparisons)
 complete=len(obs)==3;alllegal=complete and all(o['eligible'] for o in obs);matched=complete and all(cohort(r)==cohort(raws[0]) for r in raws)
 rows=[];tasks=[];paths=[];sensors=[]
 for o,r in zip(obs,raws):
  delta=o['U']-by['A0']['U'] if o['U'] is not None and by.get('A0',{}).get('U') is not None else None
  rows.append(dict(group=o['run_id'],target_split=r['slot']['frozen_G_fraction'],actual_split=o.get('G_fraction'),G_ASR=o.get('G_completed'),S_ASR=o.get('S_completed'),window_status=o['window_status'],legal_U=o['U'],delta_U=delta,eligible=o['eligible'],stability=o.get('stability'),offered=o.get('offered'),completed=o.get('completed'),failures=o.get('offered',0)-o.get('completed',0),**o.get('components',{})))
  # components U may carry diagnostic value only if original eligible; legal_U is authoritative after additional telemetry checks.
  for t,v in o.get('per_task',{}).items():
   arr=[x['e2e_latency_ms'] for x in r['phases']['measurement']['request_records'] if x['task']==t and x['status']=='completed']
   b=by.get('A0',{}).get('per_task',{}).get(t,{})
   tasks.append(dict(group=o['run_id'],task=t,failures=v['n_offered']-v['n_completed'],**v,p50_ms=float(np.percentile(arr,50)) if arr else None,p99_ms=float(np.percentile(arr,99)) if arr else None,max_ms=max(arr,default=None),mean_delta_ms=v['mean_ms']-b['mean_ms'] if v['mean_ms'] is not None and b.get('mean_ms') is not None else None,p95_delta_ms=v['p95_ms']-b['p95_ms'] if v['p95_ms'] is not None and b.get('p95_ms') is not None else None,legal=o['eligible']))
  paths.extend(dict(group=o['run_id'],**p) for p in o.get('paths',[]))
  sensors.extend(dict(group=o['run_id'],**s) for s in r['telemetry']['samples'])
 csvsave('system_utility_comparison.csv',rows,['group','legal_U','delta_U']);csvsave('per_task_latency_comparison.csv',tasks,['group','task']);csvsave('worker_service_wait.csv',paths);csvsave('worker_interference_analysis.csv',interference);csvsave('GPU_CPU_environment.csv',sensors)
 positive=[r['arm'] for r in comparisons if r['positive'] and matched and alllegal];legal_positive=[r['arm'] for r in comparisons if r.get('delta_U') is not None and r['delta_U']>0]
 gate='INCOMPLETE_OR_INVALID_EXPERIMENT' if not alllegal or not matched else ('POSITIVE_FEASIBILITY_CANDIDATE_REQUIRES_INDEPENDENT_REPLICATION' if positive else 'NO_POSITIVE_FEASIBILITY_CANDIDATE')
 history=read(OUT/'protected_history_metadata.json');changed=[p for p,v in history.items() if not Path(p).exists() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=v]
 core=[r['path'] for r in read(OUT/'protected_core_hashes.json') if ref(r['path'])['sha256']!=r['sha256']]
 protected=not changed and not core
 if not protected:gate='PROTECTED_HISTORY_INTEGRITY_FAILURE'
 stats=[]
 if alllegal and matched:
  stats=bootstrap(raws,obs);save('statistical_diagnostics.json',dict(results=stats,repeatability_proven=False,independent_matched_contrasts=1,model_combinations=1,original_holdout_support_satisfied=False))
 g=dict(gate=gate,formal_complete=complete,all_three_legal_stable=alllegal,same_offered_workload=matched,positive_candidates=positive,legal_positive_delta_arms=legal_positive,comparisons=comparisons,history_preserved=protected,changed_history=changed,changed_core=core,original_statistical_support_satisfied=False,repeatability_proven=False,Stage_D=False,Module31B_started=False,Module31C_started=False,statistical_diagnostics=stats)
 save('gate_decision.json',g)
 validation=dict(formal=[dict(group=o['run_id'],window_status=o['window_status'],checks=o['validation'],journal=o['journal_audit'],max_check_seconds=r.get('control_latency_seconds',{}).get('max_check'),max_publish_seconds=r.get('control_latency_seconds',{}).get('max_publish'),telemetry_max_gap_seconds=o.get('telemetry_max_gap_seconds'),raw=ref(OUT/'raw_runs'/(o['run_id']+'.json'))) for o,r in zip(obs,raws)],history_preserved=protected,history_files=len(history),same_offered_workload=matched,formal_complete=complete)
 save('telemetry_validation.json',validation)
 (OUT/'telemetry_validation.md').write_text('# 遥测与窗口合法性\n\n'+f'正式窗口完成 {len(obs)}/3；三臂同轨迹={matched}；历史保护={protected}（{len(history)} 文件 metadata 与核心/Gate SHA256 核验）。\n\n'+ '\n'.join(f"- {x['group']}: {x['window_status']}；journal={x['journal']['passed']}；check 最大={x['max_check_seconds']}s；publish 最大={x['max_publish_seconds']}s；采样最大间隙={x['telemetry_max_gap_seconds']}s；检查={x['checks']}。" for x in validation['formal'])+'\n\n只使用 SafeMonitorV2；250ms 慢传感器控制检查证据见 nonblocking_control_acceptance.json，三组 GPU smoke 见 smoke_acceptance*.json。完整预热与正式 journal 终态逐请求对照，不截尾补算；原始 perf/unix/worker 时间戳、完整请求、采样、关闭/释放证据见 raw_runs。窗口内吞吐与完整排空 cohort 吞吐分开，预热/path probe 排除。完整效用由原 optimizer.components 与独立 arithmetic 复算，误差要求 <1e-12。INVALID 的合法 U=NULL。\n')
 env=[]
 for o,r in zip(obs,raws):
  ph=r.get('phases',{}).get('measurement')
  if not ph:continue
  w=ph['window'];ss=[s for s in r['telemetry']['samples'] if w['start_time_ns']<=s['time_ns']<=w['end_time_ns']]
  env.append(dict(group=o['run_id'],**{k+'_mean':float(np.mean([s[k] for s in ss])) for k in ['gpu_util_percent','memory_util_percent','owned_used_mib','worker_CPU_core_percent','temperature_gpu_c','sm_clock_mhz']},peak_owned_mib=max(s['owned_used_mib'] for s in ss)))
 csvsave('resource_comparison.csv',env)
 text=['# 共置干扰与 ASR 执行/排队','', '均值的 S 分母会随分流改变；下面额外使用保留 S 的同请求对照和被分流请求的 S-only 反事实，避免输入选择混淆。原生 wait=start-submit，service=end-start。CPU 占用来自 worker 树，100%=一 CPU 核；NVML 为采样占用，不能唯一归因为 GPU 因果。','']
 for o in obs:
  for p in o.get('paths',[]):
   if p['task']=='asr':text.append(f"- {o['run_id']} {p['model']} ASR: n={p['completed']}，service mean/P95={p['service']['mean']}/{p['service']['p95']}ms，wait mean/P95={p['wait']['mean']}/{p['wait']['p95']}ms。")
 for x in interference:
  if x['task']=='asr':text.append(f"- {x['arm']} {x['worker']}: same n={x['n_same_worker']}，保留 S service/wait 差={x['paired_service_delta_ms']}/{x['paired_wait_delta_ms']}ms；分流 G 相对同输入原 S service/wait 差={x['diverted_vs_S_control_service_delta_ms']}/{x['diverted_vs_S_control_wait_delta_ms']}ms。")
 for x in tasks:
  if x['group']!='A0':text.append(f"- {x['group']} {x['task']}: mean/P95 Δ={x['mean_delta_ms']}/{x['p95_delta_ms']}ms。")
 text += ['', '资源均值/峰值见 resource_comparison.csv，逐样本见 GPU_CPU_environment.csv；所有背景任务变化保留，不以改善任务替代完整六任务效用。单次序列 A0→A1→A2 未隔离运行次序/温漂；物理环境 Gate 单独检查。']
 (OUT/'interference_analysis.md').write_text('\n'.join(text)+'\n')
 advice='不建议基于此负收益点进入 31B/31C；如将来有新科学问题需另行授权设计。' if gate=='NO_POSITIVE_FEASIBILITY_CANDIDATE' else ('可考虑 31B 的独立配对复现；31C 应以复现、SLO 和统计支持通过为前提。本次不执行。' if positive else '当前证据不完整或不合法，先解决阻断并按同冻结配置完成本模块，不能宣称实验完成。')
 (OUT/'gate_decision.md').write_text('# Module31A Gate\n\n'+f'**{gate}**\n\n合法正 ΔU 臂={legal_positive}；通过原探索 Gate 的候选={positive}。原门槛：ΔU>0.01、三臂合法稳定、G≥20、各背景≥72、温度差≤5°C、SM clock 差≤15%；质量/六任务/完整 cohort/资源/恢复约束不变。\n\n一次配对轨迹、一个模型组合不满足原独立 holdout 统计支持规则。区块 bootstrap 仅为窗内探索诊断，不是可重复收益证明。未引入新的绝对时延 SLO，完整报告背景 P95 退化。\n\n'+advice+'\n\nStage D、31B–31E 未启动，正式优化器、权重和历史 Gate 未改。\n')
 lines=['# Module31A Qwen2.5-Omni + Whisper-small 合法同版本对照','',f'最终 Gate：**{gate}**。正式窗口 {len(obs)}/3；三臂合法稳定={alllegal}；同负载={matched}。','', '|组|目标/实际分流|G/S ASR|合法 U|ΔU vs A0|状态|','|---|---|---|---:|---:|---|']
 for x in rows:lines.append(f"|{x['group']}|{x['target_split']:.0%}/{(x['actual_split'] or 0):.4%}|{x['G_ASR']}/{x['S_ASR']}|{x['legal_U']}|{x['delta_U']}|{x['window_status']} / {x['stability']}|")
 lines+=['','ASR=1 QPS，其余五任务=.2 QPS，同一 Module30 C1 seed=301001 冻结轨迹；各组完整 600 秒，30 秒预热另计，最长 60 秒排空，五副本预算 34567.5534 MiB。SafeMonitorV2 全臂同版本；初版 NULL 不补算。','', f'历史保护={protected}。Git commit 不可用（目录无 Git 元数据），源码快照、模型全文件 SHA256、环境/依赖及轨迹均冻结于 experiment_manifest.json；这是明确记录的版本管理限制。','',advice,'','统计：单次对照不能证明可重复收益；原独立统计支持规则未满足。窗内 30 秒配对区块 bootstrap（2000次）见 statistical_diagnostics.json（仅三臂全合法时生成）。','', '运行证据：campaign.log、campaign_status.json、smoke_acceptance.json、raw_runs/{SMOKE_A0,SMOKE_A1,SMOKE_A2,A0,A1,A2}.json、request_journals/*.jsonl、run_attempts/*.json、thermal_prelaunch/*.json。完整计量见 per_task_latency_comparison.csv、worker_service_wait.csv、resource_comparison.csv、worker_interference_analysis.csv。','', '未执行 Stage D 或 31B–31E。']
 if (OUT/'campaign_failure.json').exists():lines+=['','阻断：'+read(OUT/'campaign_failure.json')['error']]
 (OUT/'module31a_summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':generate()
