import sys,time,collections,traceback
from pathlib import Path
OUT=Path('/root/autodl-tmp/generalist-specialist/results/module32_heavy_specialist');sys.path[:0]=[str(OUT),str(OUT.parents[1])]
from m32.common import *
from m32.campaign import setup
from module26.common import cohort
from module25.common import utility_inputs
import numpy as np

def main():
 f=read(OUT/'experiment_manifest.json');definition=read(OUT/'module32_gate_definition.json');_,report,r31=setup()
 # Keep the exploratory source report, then apply the user's pre-formal pairwise Gate definition.
 try:
  from m32.report import generate
  generate()
 except Exception as ex:save('source_report_failure.json',dict(error=repr(ex),traceback=traceback.format_exc()))
 rawby={};obs=[]
 for s in f['slots']:
  p=OUT/'raw_runs'/(s['run_id']+'.json')
  if p.exists():
   r=read(p);rawby[s['run_id']]=r;obs.append(r31.assess(r))
 by={o['run_id']:o for o in obs};control=by.get('A0',{});pairs=[]
 d,refs=utility_inputs();history=read(OUT/'historical_integrity_after.json') if (OUT/'historical_integrity_after.json').exists() else {'passed':False}
 audit=read(OUT/'final_data_acceptance.json') if (OUT/'final_data_acceptance.json').exists() else {}
 protected=history['passed'] and audit.get('checks',{}).get('full_checkpoint_hashes',False) and audit.get('checks',{}).get('historical_metadata',False)
 formal_rows=[];tasks=[];paths=[];decomp=[]
 for aid in ['A0','A1','A2']:
  o=by.get(aid);s=next((s for s in f['slots'] if s['run_id']==aid),None)
  row=dict(run_id=aid,state='NOT_EXECUTED' if o is None else o['window_status'],eligible=bool(o and o['eligible']),U=o['U'] if o else None,delta_U=None,target_split=s['frozen_G_fraction'] if s else [0,.05,.1][int(aid[-1])],actual_split=o.get('G_fraction') if o else None,G_ASR=o.get('G_completed') if o else None,S_ASR=o.get('S_completed') if o else None,stability=o.get('stability') if o else None)
  if o:
   row.update(**{k:v for k,v in o.get('components',{}).items() if k not in ['U','Q_norm','Q_contribution']},Q_contribution=0.,Q_norm=None,offered=o.get('offered'),completed=o.get('completed'),within_qps=o.get('within_qps'),cohort_qps=o.get('completed_qps'),safe_memory_mib=o.get('safe_memory_mib'),sampled_peak_mib=o.get('sampled_peak_mib'),drain_seconds=o.get('drain_seconds'))
   if o['eligible'] and control.get('eligible'):row['delta_U']=o['U']-control['U']
   for t,v in o.get('per_task',{}).items():
    b=control.get('per_task',{}).get(t,{});r=dict(run_id=aid,task=t,**v,eligible=o['eligible'])
    if b and v['mean_ms'] is not None and b['mean_ms'] is not None:r.update(mean_delta_ms=v['mean_ms']-b['mean_ms'],p95_delta_ms=v['p95_ms']-b['p95_ms'],mean_U_delta=-d['weights']['L']*f['slots'][0]['lambda_rps'][t]/sum(f['slots'][0]['lambda_rps'].values())*(v['mean_ms']-b['mean_ms'])/refs[t] if row['delta_U'] is not None else None,p95_U_delta=-d['weights']['P']*f['slots'][0]['lambda_rps'][t]/sum(f['slots'][0]['lambda_rps'].values())*(v['p95_ms']-b['p95_ms'])/refs[t] if row['delta_U'] is not None else None)
    tasks.append(r)
   paths.extend(dict(run_id=aid,**p) for p in o.get('paths',[]))
  formal_rows.append(row)
  if aid=='A0':continue
  legal=bool(o and o['eligible'] and control.get('eligible'))
  matched=legal and cohort(rawby[aid])==cohort(rawby['A0']) and all(rawby[aid]['slot'][k]==rawby['A0']['slot'][k] for k in ['deployment','lambda_rps','safe_memory_budget_mib'])
  delta=o['U']-control['U'] if matched else None
  pairs.append(dict(arm=aid,legal_matched_pair=matched,delta_U=delta,positive=bool(matched and protected and delta>.01),G_ASR=o.get('G_completed') if o else None,S_ASR=o.get('S_completed') if o else None,actual_split=o.get('G_fraction') if o else None,auxiliary_G_sample_below20=bool(o and o.get('G_completed',0)<20),independent_replication=False))
  if aid in rawby and 'A0' in rawby and rawby[aid].get('phases',{}).get('measurement') and rawby['A0'].get('phases',{}).get('measurement'):
   aa=[r for r in rawby[aid]['phases']['measurement']['request_records'] if r['task']=='asr' and r['status']=='completed'];bb={r['request_id']:r for r in rawby['A0']['phases']['measurement']['request_records'] if r['task']=='asr' and r['status']=='completed'};aa=[r for r in aa if r['request_id'] in bb];same=[r for r in aa if r['model']==bb[r['request_id']]['model']];gg=[r for r in aa if r['model']!=bb[r['request_id']]['model']]
   def diff(rs,key):return sum(r[key]-bb[r['request_id']][key] for r in rs)
   if aa:
    sw=diff(same,'worker_queue_wait_ms');se=diff(same,'worker_latency_ms');ge=diff(gg,'worker_latency_ms');gw=diff(gg,'worker_queue_wait_ms');et=diff(aa,'e2e_latency_ms');host=et-sw-se-ge-gw
    decomp.append(dict(arm=aid,n=len(aa),n_retained_S=len(same),n_diverted_G=len(gg),S_wait_reduction_mean_ms=-sw/len(same) if same else None,S_wait_reduction_total_seconds=-sw/1000,S_service_delta_mean_ms=se/len(same) if same else None,G_execution_delta_vs_same_input_S_mean_ms=ge/len(gg) if gg else None,G_execution_delta_vs_same_input_S_total_seconds=ge/1000,G_wait_delta_vs_same_input_S_mean_ms=gw/len(gg) if gg else None,G_wait_delta_vs_same_input_S_total_seconds=gw/1000,G_absolute_execution_total_seconds=sum(r['worker_latency_ms'] for r in gg)/1000,G_absolute_wait_total_seconds=sum(r['worker_queue_wait_ms'] for r in gg)/1000,G_absolute_execution_mean_ms=float(np.mean([r['worker_latency_ms'] for r in gg])) if gg else None,G_absolute_wait_mean_ms=float(np.mean([r['worker_queue_wait_ms'] for r in gg])) if gg else None,host_transport_delta_seconds=host/1000,ASR_E2E_mean_delta_ms=et/len(aa),reconstruction_residual_ms=et-(sw+se+ge+gw+host),legal=matched))
 alllegal=len(obs)==3 and all(o['eligible'] for o in obs);allmatched=alllegal and all(p['legal_matched_pair'] for p in pairs);positive=[p for p in pairs if p['positive']]
 gate='HEAVY_S_EXPLORATORY_POSITIVE' if positive else 'HEAVY_S_NO_POSITIVE_GAIN' if allmatched and protected else 'HEAVY_S_INCONCLUSIVE'
 save('module32_gate.json',dict(gate=gate,gate_definition=ref(OUT/'module32_gate_definition.json'),formal_windows_attempted=len(obs),all_three_legal_stable=alllegal,positive_arms=positive,pairs=pairs,historical_integrity=protected,candidate1_formal_executed=False,formal_candidate=f.get('formal_candidate'),independent_replication=False,optimizer_modified=False))
 save('three_arm_gpu_measurement.json',dict(arms=formal_rows,scope='Each planned formal arm is 600 seconds with matched traces, 30s warmup, 60s drain. NOT_EXECUTED rows are explicit missing measurements, never simulated results.'))
 save('pair_comparisons_user_gate.json',pairs);csvsave('system_utility_comparison.csv',formal_rows,['run_id','U','delta_U']);csvsave('per_task_latency_comparison.csv',tasks,['run_id','task']);csvsave('worker_service_wait.csv',paths,['run_id','task','worker']);csvsave('native_latency_decomposition.csv',decomp,['arm','n','S_wait_reduction_mean_ms','G_absolute_execution_total_seconds','G_absolute_wait_total_seconds','legal'])
 stage=read(OUT/'colocation_profile.json');stage_rows=[]
 for o in stage:
  raw=read(OUT/'raw_runs'/(o['run_id']+'.json'));s=raw['slot'];row=dict(run_id=o['run_id'],candidate=s['candidate'],ASR_qps=s['lambda_rps']['asr'],G_target_fraction=s['frozen_G_fraction'],valid_session=o['valid_session'],passed_precheck=o['passed'],stability=o.get('stability',{}).get('stability_state_v2'),offered=o.get('offered'),completed=o.get('completed'),drain_seconds=o.get('drain_seconds'),safe_memory_mib=o.get('safe_memory_mib'),peak_memory_mib=o.get('peak_memory_mib'),actual_closed_loop_capacity_qps=o.get('capacity',{}).get('actual_ASR_capacity_qps'),U=None)
  for role,model in [('S',s['Specialist']),('G',s['Generalist'])]:
   p=next((p for p in o.get('paths',[]) if p['task']=='asr' and p['worker']==model+'#0'),None)
   if p:row.update(**{role+'_ASR_'+k:v for k,v in dict(n=p['n'],service_mean_ms=p['service']['mean'],service_p95_ms=p['service']['p95'],wait_mean_ms=p['wait']['mean'],wait_p95_ms=p['wait']['p95']).items()})
  stage_rows.append(row)
 csvsave('colocation_profile.csv',stage_rows,['run_id','candidate','U'])
 lines=['# Module 32：Heavy Specialist Positive-Split Feasibility','',f'最终 Gate：**{gate}**。候选1正式比较=0；正式候选={f.get("formal_candidate")}；正式已测臂={len(obs)}/3；三臂均合法稳定={alllegal}。','', '所有新数据、脚本与审计位于本目录。Module14F–31A历史文件、模型权重与正式优化器保护审计='+str(protected)+'。不执行独立长窗口复现。','', '## 配对与容量','', '两配对均通过目标ASR历史质量准入：Qwen WER=0.0417321，Phi WER=0.0195199，Whisper-large-v3 WER=0.0394884，冻结门槛0.042181。大模型S原始200条质量已复算。原14B严格高质量pool筛选与目标任务准入是不同判据；没有放宽门槛。','', 'C1：Qwen、heavy S、RoBERTa、ConvNeXt、PARSeq各1，LID/VQA→Qwen。最初实测S容量1.875QPS，但后续S-only在1.68和唯一一次下调0.92QPS均未完整排空；G 0.6QPS诊断亦失败，0.2QPS完整。所有过载窗口NULL U，未用其作效用对照。','', 'C2：Phi、heavy S、Whisper-small、RoBERTa、ConvNeXt、PARSeq各1，LID→Whisper-small，VQA→Phi。S实测闭环容量0.575QPS，mean/P95=1.669/3.539s；0.8QPS上界诊断完整排空但原判据hard-unstable。按测前80%容量规则选择0.46QPS；G .15QPS诊断完整稳定，native ASR mean/P95=3.972/6.366s，基于实测服务推算剩余ASR能力约.234QPS，非长窗容量认证。','']
 lines+=['|预检|ASR QPS|完成/到达|S执行mean ms|S等待mean ms|预检通过|原始稳定性|','|---|---:|---|---:|---:|---|---|']
 for x in stage_rows:lines.append(f"|{x['run_id']}|{x['ASR_qps']}|{x.get('completed')}/{x.get('offered')}|{x.get('S_ASR_service_mean_ms')}|{x.get('S_ASR_wait_mean_ms')}|{x['passed_precheck']}|{x['stability']}|")
 lines+=['','共置服务需求有明显变化，历史standalone S=494.913ms，不能直接用作本轮稳定容量。Qwen同输入25样本native服务944→3117ms且预测文本一致；worker源码及Qwen/Specialist/OCR依赖版本与31A一致，温度/频率/驻留诊断未唯一识别膨胀因果。源数据与完整所有背景延迟见 colocation_profile.json、S_service_expansion_by_arrival_block.csv、G_service_drift_audit.json。','', '## 冻结与三臂','']
 if f['slots']:
  s=f['slots'][0];lines +=[f"正式冻结ASR={s['lambda_rps']['asr']}QPS，五背景各.2QPS，B3={s['safe_memory_budget_mib']}MiB；同一六模型池、各1副本、输入、预算、原Module20A效用。每臂600秒，30秒warmup另计，最大60秒drain，顺序5%→S-only→10%。请求轨迹与SHA在frozen_load/experiment_manifest中；负载选择从未使用ΔU。",'']
 else:lines+=['没有通过完整压力预检，正式负载未冻结，600秒三臂没有启动；缺测不能填为负收益或仿真值。','']
 lines+=['|臂|目标/实际G份额|实际G/S ASR|合法U|合法ΔU|稳定性|','|---|---|---|---:|---:|---|']
 for x in formal_rows:lines.append(f"|{x['run_id']}|{x['target_split']}/{x['actual_split']}|{x['G_ASR']}/{x['S_ASR']}|{x['U']}|{x['delta_U']}|{x['stability']} / {x['state']}|")
 lines+=['','严格复用Module20A：Q/T/L/P/M=0/.25/.30/.30/.15，offered QPS权重、逐任务原参考时延、窗口吞吐损失、safe-memory定义均不变；质量只硬准入，原stability_v2与完整cohort/有界排空/资源判据不变。仅合法匹配双方计算ΔU。Module32按用户冻结的“一个合法配置ΔU>+.01”接受探索性正例；旧Module30样本及环境额外检查保留为诊断，不替代该定义。','', '## 效用与机制','']
 for x in decomp:lines +=[f"- {x['arm']}：同输入保留S等待减少 {x['S_wait_reduction_mean_ms']:.3f}ms/请求、总{x['S_wait_reduction_total_seconds']:.3f}s；G新增绝对执行/等待={x['G_absolute_execution_total_seconds']:.3f}/{x['G_absolute_wait_total_seconds']:.3f}s。G相对原S执行增量={x['G_execution_delta_vs_same_input_S_mean_ms']:.3f}ms/请求；保留S执行变化={x['S_service_delta_mean_ms']:.3f}ms；ASR整体mean变化={x['ASR_E2E_mean_delta_ms']:.3f}ms。"]
 for p in pairs:
  aid=p['arm'];sub=[x for x in tasks if x['run_id']==aid and x.get('mean_U_delta') is not None];target=sum(x['mean_U_delta']+x['p95_U_delta'] for x in sub if x['task']=='asr');background=sum(x['mean_U_delta']+x['p95_U_delta'] for x in sub if x['task']!='asr');lines +=['',f"{aid}：ASR mean/P95效用贡献变化={target}，五背景贡献变化={background}；T/M贡献另见system_utility_comparison.csv。全部Text/Image/OCR/LID/VQA mean/P95变化保留，不按改善任务挑选。"];p['ASR_L_P_delta']=target if p['legal_matched_pair'] else None;p['background_L_P_delta']=background if p['legal_matched_pair'] else None
 lines+=['','Module31A：S执行227.57ms、等待36.06ms，5%/10%同输入S等待减少1.472/4.816ms。本轮预检S等待更大；正式可释放等待、G代价、共置服务膨胀以及背景抵消程度以本轮匹配分解为准。跨模块G和LID路径、负载及运行状态不同，只比较机制，不宣称单一模型重量因果。','']
 if positive:
  lines+=['探索性正例已冻结于frozen_positive_candidate.json。正式优化器仅提出建议：通过实测共置service/tail、S队列压力、G可用容量及背景干扰的条件模型，联合选择是否部署heavy S和路由；保留不部署heavy S、纯S、纯G与连续混合候选，按完整U和稳定/预算/质量约束比较。模型身份、QPS与分流比例均为可优化输入，禁止硬编码本轮胜者；不修改或执行正式优化器。','']
  save('frozen_positive_candidate.json',dict(label=gate,candidate=f.get('formal_candidate'),positive_arms=positive,matched_pool=f['slots'][0]['deployment'],rates=f['slots'][0]['lambda_rps'],experiment_manifest=ref(OUT/'experiment_manifest.json'),gate_definition=ref(OUT/'module32_gate_definition.json'),independent_replication=False))
 else:lines+=['未找到满足本轮Gate的合法探索性正例。未提出或实施正式优化器修改；不合法与负收益结果全部保留。','']
 if any(p['auxiliary_G_sample_below20'] for p in pairs):lines+=['5%实际G样本较少，G路径P95及单次效应不确定；保留实际份额与全部样本，不外推可复现性。','']
 if (OUT/'campaign_failure.json').exists():lines+=['停止原因：'+read(OUT/'campaign_failure.json')['error'],'']
 lines+=['## 审计与限制','', '最初READY任务声明适配遗漏在测量前修复，保留initial_ready_failure；G容量探测一次降载、无请求备用worker的审计修正、C1唯一压力负载调整均有归档与时间戳。没有重跑已测GPU窗口、按正收益选种子、人工延时、修改原推理设置或增加无意义计算。所有正式窗口仅一次探索。','', '温度、频率、功率、显存、CPU与异常原始样本见raw_runs、GPU_CPU_environment.csv与resource_comparison.csv；完整请求journal及collector/sender/reader退出、GPU释放见原始审计。显存为NVML采样，非瞬时allocator最大值。Git元数据不可用，源码/checkpoint SHA及依赖冻结代替版本身份记录。','', '窗内bootstrap仅诊断，不含独立跨运行不确定性，不满足Module20A独立holdout规则。未自动执行独立长窗口、修改正式优化器或启动下一模块；到此停止。']
 (OUT/'module32_summary.md').write_text('\n'.join(lines)+'\n');(OUT/'gate_decision.md').write_text(f'# Module32 Gate\n\n**{gate}**\n\n定义在正式窗口前冻结于module32_gate_definition.json；对应双方合法稳定才计算ΔU。独立复现与正式优化器修改均未执行。\n')
 save('pair_comparisons_user_gate.json',pairs)
 save('artifact_manifest.json',dict(finished_unix_ns=time.time_ns(),artifacts=[ref(p) for p in sorted(OUT.glob('*')) if p.is_file() and p.name!='artifact_manifest.json'],source_scripts=[ref(p) for p in sorted(OUT.glob('*.py'))],history_protected=protected,gate=gate))
 print('CANONICAL_MODULE32_GATE',gate,pairs,flush=True)
if __name__=='__main__':main()
