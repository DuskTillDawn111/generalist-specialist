from .common import *
import numpy as np
from .campaign import setup
from module26.common import cohort
from module25.common import utility_inputs

def generate():
 f=read(OUT/'experiment_manifest.json');_,report,r31=setup()
 raws=[read(OUT/'raw_runs'/(s['run_id']+'.json')) for s in f['slots'] if (OUT/'raw_runs'/(s['run_id']+'.json')).exists()];obs=[r31.assess(r) for r in raws]
 save('run_observations.json',obs)
 comp,inter,changes=report.compare(raws,obs);comp=[x for x in comp if x['candidate']=='C1'];save('pair_comparisons.json',comp)
 by={o['run_id']:o for o in obs};rows=[];tasks=[];paths=[];resources=[];sensors=[];backlogs=[]
 for r,o in zip(raws,obs):
  base=by.get('A0',{});delta=o['U']-base['U'] if o['eligible'] and base.get('eligible') else None
  rows.append(dict(run_id=o['run_id'],target_split=r['slot']['frozen_G_fraction'],actual_split=o.get('G_fraction'),G_ASR=o.get('G_completed'),S_ASR=o.get('S_completed'),U=o['U'],delta_U=delta,eligible=o['eligible'],technical_valid=o['technical_valid'],stability=o.get('stability'),offered=o.get('offered'),completed=o.get('completed'),within_qps=o.get('within_qps'),cohort_qps=o.get('completed_qps'),**{k:v for k,v in o.get('components',{}).items() if k!='U'}))
  for t,v in o.get('per_task',{}).items():
   b=base.get('per_task',{}).get(t,{});tasks.append(dict(run_id=o['run_id'],task=t,**v,mean_delta_ms=v['mean_ms']-b['mean_ms'] if b and v['mean_ms'] is not None else None,p95_delta_ms=v['p95_ms']-b['p95_ms'] if b and v['p95_ms'] is not None else None))
  paths.extend(dict(run_id=o['run_id'],**p) for p in o.get('paths',[]));sensors.extend(dict(run_id=o['run_id'],**v) for v in r['telemetry']['samples']);backlogs.extend(report.backlog(r))
  w=r.get('phases',{}).get('measurement',{}).get('window',{});ss=[v for v in r['telemetry']['samples'] if w.get('start_time_ns',1)<=v['time_ns']<=w.get('end_time_ns',0)]
  resources.append(dict(run_id=o['run_id'],safe_memory_mib=o.get('safe_memory_mib'),sampled_peak_mib=o.get('sampled_peak_mib'),errors=r['errors'],**{key:report.stats(ss,key) for key in ['temperature_gpu_c','sm_clock_mhz','gpu_util_percent','power_w','worker_CPU_core_percent','owned_used_mib']}))
 csvsave('system_utility_comparison.csv',rows,['run_id','U','delta_U']);csvsave('per_task_latency_comparison.csv',tasks,['run_id','task']);csvsave('worker_service_wait.csv',paths,['run_id','worker','task']);csvsave('worker_interference_analysis.csv',inter);csvsave('task_utility_changes.csv',changes);csvsave('resource_comparison.csv',resources);csvsave('GPU_CPU_environment.csv',sensors,['run_id']);csvsave('backlog_timeseries.csv',backlogs,['run_id','worker','offset_seconds'])
 decom=[]
 rawby={r['slot']['run_id']:r for r in raws}
 if 'A0' in rawby:
  base={r['request_id']:r for r in rawby['A0']['phases'].get('measurement',{}).get('request_records',[]) if r['task']=='asr' and r['status']=='completed'}
  for aid in ['A1','A2']:
   if aid not in rawby:continue
   rr=[r for r in rawby[aid]['phases'].get('measurement',{}).get('request_records',[]) if r['task']=='asr' and r['status']=='completed' and r['request_id'] in base];n=len(rr)
   if not n:continue
   same=[r for r in rr if r['model']==base[r['request_id']]['model']];div=[r for r in rr if r not in same]
   def diff(sub,key):return sum(r[key]-base[r['request_id']][key] for r in sub)
   sw=diff(same,'worker_queue_wait_ms');ss=diff(same,'worker_latency_ms');gs=diff(div,'worker_latency_ms');gw=diff(div,'worker_queue_wait_ms')
   total=diff(rr,'e2e_latency_ms');host=total-sw-ss-gs-gw
   decom.append(dict(arm=aid,n=n,n_retained_S=len(same),n_diverted_G=len(div),S_wait_reduction_mean_ms=-sw/len(same) if same else None,S_wait_reduction_total_seconds=-sw/1000,S_service_delta_mean_ms=ss/len(same) if same else None,G_added_execution_mean_ms=gs/len(div) if div else None,G_added_execution_total_seconds=gs/1000,G_added_wait_mean_ms=gw/len(div) if div else None,G_added_wait_total_seconds=gw/1000,host_transport_delta_total_seconds=host/1000,ASR_e2e_mean_delta_ms=total/n,reconstruction_residual_ms=total-(sw+ss+gs+gw+host),scope='Native same-input paired accounting; host residual includes scheduling/transport/result receipt'))
 csvsave('native_latency_decomposition.csv',decom)
 matched=len(raws)==3 and all(cohort(r)==cohort(raws[0]) for r in raws)
 legal=len(obs)==3 and all(o['eligible'] for o in obs)
 positive=[x for x in comp if x.get('positive') and matched and legal]
 before=read(OUT/'protected_before.json');changed=[p for p,v in before.items() if not Path(p).exists() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=v]
 hashes=[r['path'] for r in f['sources']+f['frozen_inputs'] if ref(r['path'])['sha256']!=r['sha256']]
 protected=not changed and not hashes
 save('historical_integrity_after.json',dict(passed=protected,protected_files=len(before),changed=changed,hashchanged=hashes))
 gate='HEAVY_S_EXPLORATORY_POSITIVE' if positive and protected else 'HEAVY_S_NO_POSITIVE_GAIN' if legal and matched and protected and all(x.get('environment_comparable') and x.get('sample_gate') for x in comp) else 'HEAVY_S_INCONCLUSIVE'
 save('module32_gate.json',dict(gate=gate,formal_windows=len(obs),all_three_legal_stable=legal,matched_trace=matched,positive_arms=positive,pairs=comp,historical_integrity=protected,independent_replication=False,optimizer_modified=False,candidate2_GPU_executed=False))
 if positive:save('frozen_positive_candidate.json',dict(label=gate,selected=positive,experiment_manifest=ref(OUT/'experiment_manifest.json'),independent_replication=False))
 if legal:save('statistical_diagnostics.json',r31.bootstrap(raws,obs))
 lines=['# Module 32：Heavy Specialist Positive-Split Feasibility','',f'Gate：**{gate}**。正式窗口 {len(obs)}/3；三臂合法稳定={legal}；匹配轨迹={matched}；历史保护={protected}。','', '候选 1：Qwen2.5-Omni + Whisper-large-v3。Text→RoBERTa SST-2，Image→ConvNeXt tiny，OCR→PARSeq，LID/VQA→Qwen；五个单副本覆盖六任务。Whisper-large-v3 使用已有 Module13B FP32、batch=1、native attention 和生成设置，不新增推理等待或低效参数。所有臂同一运行版本、B3=34567.5534 MiB、效用、模型池和输入。','', 'Whisper-large-v3 WER=0.03948844514247252，目标门槛=0.042181，原始200条质量输出已复算。Module14B 的 FAIL_QUALITY 是相对 Distil baseline 的严格高质量筛选，不能误当目标 ASR 准入失败。候选2仅保留 CPU 配对审查；候选1预检通过后不扩展 GPU 搜索。','']
 if (OUT/'frozen_load.json').exists():
  load=read(OUT/'frozen_load.json');lines +=[f"冻结负载：ASR={load['ASR_qps']} QPS，其余五任务各0.2 QPS；600秒/臂，30秒预热另计，最多60秒排空。执行顺序 A1(5%)→A0(0%)→A2(10%)，基线置于中间；一次序列探索无法独立隔离次序效应。",'']
 lines+=['|臂|实际G/S ASR|合法U|ΔU vs S-only|状态|','|---|---|---:|---:|---|']
 for r in rows:lines.append(f"|{r['run_id']}|{r['G_ASR']}/{r['S_ASR']} ({r['actual_split']})|{r['U']}|{r['delta_U']}|{r['stability']} / eligible={r['eligible']}|")
 lines+=['','效用完整沿用 Module20A：Q/T/L/P/M=0/.25/.30/.30/.15，六任务 offered QPS 加权和原参考时延；原 stability_v2、完整cohort、资源和恢复判据不变。ΔU 只在对应双方合法稳定时计算。原+0.01、G样本≥20、各背景≥72、温度差≤5°C、频率差≤15%探索 Gate 单独审查。','']
 for x in decom:lines.append(f"- {x['arm']}：保留S的等待平均减少 {x['S_wait_reduction_mean_ms']:.3f} ms，总减少 {x['S_wait_reduction_total_seconds']:.3f}s；转交G新增执行 {x['G_added_execution_mean_ms']:.3f}ms/请求、总 {x['G_added_execution_total_seconds']:.3f}s；G新增等待 {x['G_added_wait_mean_ms']:.3f}ms/请求；保留S服务时间变化 {x['S_service_delta_mean_ms']:.3f}ms。")
 lines+=['','Module31A 在ASR=1QPS、Whisper-small且LID→S时，S执行227.57ms、等待36.06ms；5%/10%分流的同输入S等待减少1.472/4.816ms，G新增执行724.551/736.142ms。本轮更重S、不同ASR QPS及LID→G，因此只作机制对照，不能把跨模块差异归因于模型重量单一因素。','']
 for aid in ['A0','A1','A2']:
  if aid not in by:continue
  aa=[p for p in by[aid].get('paths',[]) if p['task']=='asr'];lines.append(f"{aid} ASR service/wait："+'；'.join(f"{p['model']} n={p['completed']}，执行mean/P95={p['service']['mean']}/{p['service']['p95']}ms，等待mean/P95={p['wait']['mean']}/{p['wait']['p95']}ms" for p in aa)+ '。')
 for x in comp:
  relevant=[c for c in changes if c['arm']==x['arm']]
  bg=sum(c['mean_U_delta']+c['p95_U_delta'] for c in relevant if c['task']!='asr' and c['mean_U_delta'] is not None)
  asr=sum(c['mean_U_delta']+c['p95_U_delta'] for c in relevant if c['task']=='asr' and c['mean_U_delta'] is not None)
  lines +=['',f"{x['arm']} ASR L/P效用变化={asr}，其余五任务L/P变化={bg}；T/M差和环境可比性见 pair_comparisons.json。"]
 lines+=['','Text/Image/OCR/LID/VQA逐任务mean/P95变化完整保存在 per_task_latency_comparison.csv 和 task_utility_changes.csv；共置同输入S服务膨胀、G新增成本见 worker_interference_analysis.csv。温度、频率、功率、显存、CPU和异常逐样本见 GPU_CPU_environment.csv；有共置差异不等于已经唯一识别GPU因果。','', 'Stage B 的40秒真实闭环容量、S/G原生服务mean/P95、背景任务延迟、队列稳定性、观察到的G余量保存在 colocation_profile.json。闭环吞吐是短窗共置观测，G剩余QPS是观测需求的推算，不能宣称长窗容量认证。预检负载选择完全不使用ΔU。','']
 if (OUT/'campaign_failure.json').exists():lines+=['停止原因：'+read(OUT/'campaign_failure.json')['error'],'']
 if positive:
  lines+=['本轮已找到探索性正例。正式优化器建议：以实测共置服务需求及其不确定性生成部署候选，保留不部署重S、纯S和纯G对照；根据S队列压力、G可用容量和背景任务的归一化mean/tail损失预测完整系统U，联合选择重S部署与连续分流比例；仅在质量、预算和所有worker稳定条件下接受收益，候选置信不足时主动追加受控画像。模型身份、ASR QPS和分流比例均作为输入变量，禁止硬编码本轮胜者。此处只提出建议，正式优化器没有修改。','']
 else:lines+=['未形成可接受探索性正例，不提出或实施正式优化器修改；保留所有负结果与不合法窗口。','']
 lines+=['统计仅为单模型组合、单次匹配轨迹探索。窗内30秒配对区块bootstrap不包含独立跨运行不确定性，不满足原独立holdout支持规则。未宣称独立复现，未执行独立长窗口，完成后停止。','', '产物：model_pair_review、quality_admission_provenance、colocation_profile、frozen_load、raw_runs、request_journals、system_utility_comparison、native_latency_decomposition、module32_gate，以及冻结manifest和保护审计。']
 (OUT/'module32_summary.md').write_text('\n'.join(lines)+'\n')
 (OUT/'gate_decision.md').write_text(f'# Module32 Gate\n\n**{gate}**\n\n三臂合法稳定={legal}，匹配轨迹={matched}，历史保护={protected}。独立复现与正式优化器修改均未执行。\n')
 print('GATE',gate,flush=True)
if __name__=='__main__':generate()
