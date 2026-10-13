"""Final evidence narrative, no changed inference or scientific thresholds."""
import sys, itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32c.common import *
from m32c.report import generate
from module25.common import utility_inputs
generate();g=read(OUT/'module32c_gate.json');f=read(OUT/'experiment_manifest.json');obs=read(OUT/'run_observations.json');by={o['run_id']:o for o in obs};pairs=list(csv.DictReader((OUT/'gpu_independent_pairs.csv').open()));tasks=list(csv.DictReader((OUT/'per_task_utility_decomposition.csv').open()));asr=list(csv.DictReader((OUT/'Whisper_paired_queue_service.csv').open()));res=list(csv.DictReader((OUT/'gpu_resource_comparison.csv').open()));sensitivity=list(csv.DictReader((OUT/'utility_sensitivity_diagnostic.csv').open()));trigger=list(csv.DictReader((OUT/'dynamic_Phi_ASR_trigger_execution.csv').open()));vqa=list(csv.DictReader((OUT/'gpu_VQA_execution_overlap.csv').open()));s=g.get('statistics');shap=[];timing=[];associations=[];dose=[]
for pair in range(1,4):
 sid=f'P{pair}_S';did=f'P{pair}_D'
 if sid not in by or did not in by:continue
 a=read(OUT/'raw_runs'/f'{did}.json');b=read(OUT/'raw_runs'/f'{sid}.json')
 if not a.get('phases',{}).get('measurement') or not b.get('phases',{}).get('measurement'):continue
 ra=a['phases']['measurement']['request_records'];bb={x['request_id']:x for x in b['phases']['measurement']['request_records']};ga=[x for x in ra if x['model']==G and x['task']=='asr' and x['status']=='completed']
 for t in ['text','visual_qa']:
  z=[x for x in ra if x['task']==t and x['status']=='completed' and bb[x['request_id']]['status']=='completed']
  def vector(x):return [x['worker_queue_wait_ms'],x['worker_latency_ms'],x['e2e_latency_ms']-x['worker_queue_wait_ms']-x['worker_latency_ms']]
  bv=np.array([vector(bb[x['request_id']]) for x in z]);dv=np.array([vector(x) for x in z]);contrib=np.zeros(3)
  for perm in itertools.permutations(range(3)):
   current=bv.copy();prior=np.percentile(current.sum(axis=1),95)
   for j in perm:current[:,j]=dv[:,j];q=np.percentile(current.sum(axis=1),95);contrib[j]+=(q-prior)/6;prior=q
  shap.append(dict(pair=pair,task=t,queue_p95_change_ms=contrib[0],handler_service_p95_change_ms=contrib[1],host_residual_p95_change_ms=contrib[2],total_p95_change_ms=np.percentile(dv.sum(axis=1),95)-np.percentile(bv.sum(axis=1),95),algebraic_NOT_executed_counterfactual=True))
  for j,key in enumerate(['queue','handler_service','host_residual']):timing.append(dict(pair=pair,task=t,component=key,S_mean_ms=np.mean(bv[:,j]),D_mean_ms=np.mean(dv[:,j]),mean_change_ms=np.mean(dv[:,j]-bv[:,j]),S_p95_ms=np.percentile(bv[:,j],95),D_p95_ms=np.percentile(dv[:,j],95),component_p95_not_additive=True))
  for flag in [False,True]:
   selected=[x for x in z if (sum(overlap(interval(x),interval(y)) for y in ga)>0)==flag]
   associations.append(dict(pair=pair,task=t,overlapped_Phi_ASR_handler=flag,n=len(selected),paired_service_change_ms=stats(x['worker_latency_ms']-bb[x['request_id']]['worker_latency_ms'] for x in selected),paired_E2E_change_ms=stats(x['e2e_latency_ms']-bb[x['request_id']]['e2e_latency_ms'] for x in selected),observational_duration_selection_confound=True,causal_mechanism_not_identified=True))
 dts=[x for x in tasks if int(x['pair'])==pair];tc=next((x for x in dts if x['task']=='text'),{});ac=next((x for x in dts if x['task']=='asr'),{});vc=next((x for x in dts if x['task']=='visual_qa'),{})
 dose.append(dict(pair=pair,G_ASR=len(ga),Text_P95_contribution=tc.get('P_delta_U'),ASR_total_contribution=ac.get('L_P_delta_U'),VQA_total_contribution=vc.get('L_P_delta_U'),future_VQA_arrivals=sum(int(x['future_VQA_arrivals_during_handler']) for x in trigger if int(x['pair'])==pair),limitations='Three groups cannot identify dose-response or isolate GPU contention; no future routing features'))
csvsave('Text_VQA_p95_component_attribution.csv',shap);csvsave('Text_VQA_service_queue_decomposition.csv',timing);csvsave('Text_VQA_Phi_overlap_association.csv',associations);csvsave('routing_dose_utility_attribution.csv',dose)
lines=['# Module32C：独立配对复现与效用归因','',f'最终 Gate：**{g["gate"]}**。完整三组={g["complete_three_pairs"]}；双方合法稳定={g["all_legal_stable"]}；每组真实G/S={g["real_G_and_S_each_dynamic"]}；三组ΔU均>+.01={g["all_three_delta_above_001"]}。','', 'Stage A CPU-only审计通过：Module32B完整文件SHA、raw、journal、worker身份/输出、GPU遥测与全部六任务U复算均合格。Text P95正贡献+.044249（总Text+.040279）主要来自handler服务变化，排队很小；原Text P95由115样本第109/110排序请求插值。具体请求、原四次触发与代数分解见 utility_attribution.md、text_tail_audit.md。','', '## 冻结独立配对','', '原Phi + Whisper-large-v3六任务池、六单副本、ASR=.46QPS、五背景各.2QPS、B3=34567.5534MiB、原质量准入不变。原32B路由源码、全部阈值、音频元数据、背压规则按字节相同；只把计量固定600s分母改成冻结窗口2400s，CPU对照复算验证，无原优化器改动。','', '三个预先固定新seed=320301/320302/320303，组内相同输入、Poisson到达、route/replica uniforms；相同 frozen sample pool 的新顺序/到达轨迹，不是新数据集质量验证。预定顺序P1 S→D、P2 D→S、P3 S→D，无seed替换。每臂30s warmup完全排空、原热控制、2400s正式、有界60s排空，完整cohort，原Module20A U与稳定性阈值，不稳定/不合法U=NULL。','', '|组|S-only U|Dynamic U|ΔU|G/S ASR|合法稳定|温差°C|','|---|---:|---:|---:|---|---|---:|']
for x in pairs:lines.append(f'|{x["pair"]}|{x["S_only_U"]}|{x["dynamic_U"]}|{x["delta_U"]}|{x["G_ASR"]}/{x["S_ASR"]}|{x["both_legal_stable"]}|{x["mean_temperature_difference_c"]}|')
lines+=['','## 独立组统计','']
if s:lines+= [f'三个独立配对ΔU均值={s["mean_delta_U"]:+.9f}，样本SD={s["sample_SD"]:.9f}，95% Student-t区间=[{s["CI95_low"]:+.9f}, {s["CI95_high"]:+.9f}]，df=2。统计单位是配对组，不把数千请求当独立复现。仅n=3，正态近似、序列环境漂移与同输入池共享均限制外推；区间不是GPU干扰机制的因果证明。']
else:lines+=['完整合法三组不足，不报告预定三组均值/置信区间为有效复现证据。']
lines+=['','## Text 尾与总效用来源','', '|组|Text mean贡献|Text P95贡献|ASR L/P贡献|VQA L/P贡献|诊断减Text P95|','|---|---:|---:|---:|---:|---:|']
for x in sensitivity:
 p=int(x['pair']);ac=next(t for t in tasks if int(t['pair'])==p and t['task']=='asr');vc=next(t for t in tasks if int(t['pair'])==p and t['task']=='visual_qa');lines.append(f'|{p}|{x["Text_mean_contribution"]}|{x["Text_p95_contribution"]}|{ac["L_P_delta_U"]}|{vc["L_P_delta_U"]}|{x["diagnostic_without_Text_p95_contribution"]}|')
lines+=['','仅合法配对才列上述正式贡献；不合法配对保持NULL，另以明确标记的ineligible_task_arithmetic_diagnostic.csv提供完整六任务代数诊断。减去Text贡献只是代数敏感性，官方U仍含六任务全部原权重；没有删请求、重定义效用或执行反事实路由。全部测量和T/M分项见 per_task_utility_decomposition.csv、gpu_independent_pairs.csv与完整raw。','', 'Text与VQA服务/等待/E2E分别统计；P95分项不能相加。Text_VQA_p95_component_attribution.csv是匹配已测请求向量的代数Shapley，并非执行过的counterfactual。直接Phi ASR overlap按同host Unix区间；不存在直接重叠也不能排除共享GPU/host资源或状态漂移。没有内核级共享资源干预，所以Text改善与路由因果机制仍不能唯一识别。','']
if s:lines.append(f'Text P95改善在三组是否均正={s["all_three_Text_p95_positive"]}；三组Text P95效用贡献均值={s["Text_p95_mean_contribution"]:+.9f}。完整尾请求与两个分位支撑样本见 gpu_text_VQA_tail_requests.csv、gpu_text_VQA_quantile_support.csv。')
lines+=['','## Whisper 实际收益与 VQA 非抢占风险','', '|组|保留S数|G数|保留S等待均值减少ms|保留S等待累计减少s|G相对同输入S新增执行s|','|---|---:|---:|---:|---:|---:|']
for x in asr:lines.append(f'|{x["pair"]}|{x["retained_S_n"]}|{x["diverted_G_n"]}|{x["S_wait_reduction_mean_ms"]}|{x["S_wait_total_saved_seconds"]}|{x["G_extra_service_vs_same_input_S_seconds"]}|')
lines+=['','G真实接纳状态、实时预测/实际handler时间、实际等待和同输入基线均保留。当前空Phi不能知道未来VQA，不可抢占执行会阻塞新VQA；下面的未来暴露均是事后诊断，路由器不读取。']
for x in dose:
 p=x['pair'];blocked=[v for v in vqa if v['run_id']==f'P{p}_D' and float(v['Phi_ASR_handler_overlap_in_wait_ms'])>0];o=by[f'P{p}_D'];b=by[f'P{p}_S'];lines.append(f'- P{p}: G执行{x["G_ASR"]}次；期间新VQA到达{x["future_VQA_arrivals"]}；VQA等待覆盖Phi ASR的请求{len(blocked)}；VQA E2E mean/P95 S={b["per_task"]["visual_qa"]["mean_ms"]:.3f}/{b["per_task"]["visual_qa"]["p95_ms"]:.3f}ms，D={o["per_task"]["visual_qa"]["mean_ms"]:.3f}/{o["per_task"]["visual_qa"]["p95_ms"]:.3f}ms。')
lines+=['','## 稳定性、环境与停止','', '每个系统/worker沿用原stability_v2判据，每任务一秒backlog轨迹完整记录；没有用排空截尾补出正值。吞吐分within-window和full-cohort两口径。所有600秒开发数据只用于审计，不与新2400秒窗口混作配对基线。','', 'GPU UUID/驱动/内存与各worker Python环境冻结，温度、SM/显存时钟、功率、CPU、VRAM与外来进程逐样本保存。组内原5°C/15%热状态诊断见gpu_independent_pairs.csv，并报告绝对服务需求可能随顺序/环境漂移。Unix与perf时间仅按各自定义计算，handler含原CUDA同步与预处理，不声称纯kernel服务时间。','']
if g['gate']=='DYNAMIC_REPLICATION_CONFIRMED':lines+=['三组与组级区间达到预定复现要求；仍只有三组、单GPU及同样本池，Text机制未被实验隔离。下一阶段可另行设计Fixed10和更强基线，以及依据实测边际系统成本自主启用协作的优化器机制；本轮不执行。']
elif g['gate']=='POSITIVE_TREND_EVIDENCE_INSUFFICIENT':lines+=['三组均>+.01，但组级置信区间跨零：只报告正向趋势与证据不足，不宣称独立复现成功，不进入正式优化器或下一阶段。']
elif g['gate']=='DYNAMIC_REPLICATION_FAILED':lines+=['未满足三组均>+.01的独立复现要求。所有负值/不足门槛组完整保留，不改阈值、不替换种子，不据此进入正式优化器或下一阶段。']
else:lines+=['数据合法性、完整性或真实协作未支持有效三组复现；保留完整已尝试结果和未执行原因，不宣称失败策略的效用为有效正例。']
if (OUT/'long_window_router_failure.json').exists():
 failure=read(OUT/'long_window_router_failure.json');diag=read(OUT/'ineligible_full_cohort_arithmetic_diagnostic.json')
 lines+=['','## 本轮停止原因与未完成配对','',f'实际执行2/6次正式GPU运行，仅P1完整两臂；P2/P3未执行。P1_D路由wrapper P95开销={failure["observed_ms"]["p95"]:.6f}ms，超过原32B冻结1ms验收阈值，触发原技术错误停止政策。Dynamic虽然完整、steady且10次真实分流，正式U与配对ΔU均为NULL，不用原始代数数值挽救验收。','', '平均GPU温度还相差7.006°C，原5°C环境诊断未通过；SM时钟差0.282%，无外来GPU进程。温差不是独立增加的停止阈值，但进一步限制因果解释。','', '原32B状态投影扫描累计runtime.pending，其中完成请求未删除。wrapper分段P95随2400s窗口增长：前300s约.288ms，最后300s约1.335ms。CPU-only源码/重放画像支持历史扫描扩展问题；纯router.decide很快但不能替换原完整wrapper验收。未修改冻结实现。','', f'仅非合法完整cohort代数差={diag["DIAGNOSTIC_arithmetic_difference"]:+.9f}，Text P95诊断贡献={diag["DIAGNOSTIC_Text_P95_contribution"]:+.9f}；二者明确不是正式ΔU。P1 Text P95由58.638增至62.398ms，原Text尾改善未在这一已测轨迹出现。缺少三组有效差值，不报告三组均值/置信区间。','', '后续若另行开展研究，先处理宿主在线状态历史扫描并独立冻结/验收，再设计新的配对与环境控制；不得放宽阈值或沿用当前非法差值宣称成功。本轮只提出此诊断，不实现修复、不重跑GPU。','', '完整失败诊断见long_window_failure_diagnosis.md；六个计划slot状态见planned_run_disposition.json，后两组新轨迹原样保留，未换种子。']
 g.update(stop_reason='Original frozen routing-wrapper P95<1ms acceptance failed at P1_D',measured_GPU_runs=2,planned_GPU_runs=6,unexecuted_pairs=[2,3],observed_G_ASR_first_pair=10,observed_ASR_first_pair=1070,observed_real_G_and_S_in_measured_dynamic=True,Dynamic_wrapper_P95_ms=failure['observed_ms']['p95'],formal_dynamic_U=None,formal_pair_delta_U=None,planned_three_pair_mean_CI=None,parameters_and_source_unchanged=True,scientific_replication_complete=False,history_integrity_passed=read(OUT/'final_data_acceptance.json')['integrity_passed'])
 save('module32c_gate.json',g)
 lines.insert(4,'原计划三组六臂，实际一组两臂完成。P1_D触发原32B冻结1ms路由开销验收失败后停止；其正式U与ΔU=NULL，后两组NOT_EXECUTED，无三组均值/置信区间。完整性保护通过，实验验收未通过。')
lines+=['','最终全权重SHA、历史全部Module32/32B SHA和Module14F–31A/正式优化器metadata保护、组内输入和逐决策因果复算见 final_data_acceptance.json。该文件passed包括六次完成和技术验收，本轮为false；integrity_passed单独表示文件/权重/环境/因果重放完整性。所有既有Gate不变；GPU已释放，未启动下一阶段，到此停止。']
(OUT/'module32c_summary.md').write_text('\n'.join(lines)+'\n');save('module32c_summary.json',dict(gate=g,Text_VQA_p95_attribution=shap,routing_dose=dose,sensitivity=sensitivity,stopped=True,optimizer_modified=False,next_stage_started=False))
(OUT/'stability_environment_report.md').write_text('# 稳定性与环境\n\n'+ '\n'.join(f'- {o["run_id"]}: legal={o["eligible"]}, technical={o["technical_valid"]}, stability={o.get("stability")}, offered/completed={o.get("offered")}/{o.get("completed")}, U={o["U"]}, VRAM safe/peak={o.get("safe_memory_mib")}/{o.get("sampled_peak_mib")}, drain={o.get("drain_seconds")}s, checks={o.get("validation")}, journal={o["journal_audit"]["passed"]}' for o in obs)+'\n\n原判据、完整request cohort、worker/sys/task backlog、原热控制不变。环境逐样本与组内温差/SM时钟诊断见GPU CSV；最终审计见final_data_acceptance.json。\n')
print('FINAL_REPORT',g['gate'],flush=True)
