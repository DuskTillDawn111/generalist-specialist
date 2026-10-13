"""Post-campaign CPU review only. Never launch GPU or modify frozen implementation."""
import math,statistics,json
from .common import *

def avg(xs):return statistics.fmean(xs) if xs else None
def num(x):return 'NULL' if x is None else f'{x:.4f}'
def taskpaths(o,t):return [p for p in o.get('paths',[]) if p['task']==t]
def make_decomposition(raws,observations):
 by={o['run_id']:o for o in observations};rows=[]
 for cid in ['C1','C2']:
  group=[r for r in raws if r['slot']['candidate']==cid];control=next((r for r in group if r['slot']['arm']=='s_only'),None)
  if not control or not control.get('phases',{}).get('measurement'):continue
  lookup={r['request_id']:r for r in control['phases']['measurement']['request_records']}
  for raw in group:
   if raw['slot']['arm']=='s_only' or not raw.get('phases',{}).get('measurement'):continue
   o=by[raw['slot']['run_id']];b=by[control['slot']['run_id']]
   for task in raw['slot']['lambda_rps']:
    pp=[(r,lookup[r['request_id']]) for r in raw['phases']['measurement']['request_records'] if r['task']==task and r['status']=='completed' and lookup[r['request_id']]['status']=='completed']
    if not pp:continue
    n=len(pp);same=[p for p in pp if p[0]['model']==p[1]['model']];diverted=[p for p in pp if p[0]['model']!=p[1]['model']]
    def term(ps,key):return sum(r[key]-c[key] for r,c in ps)/n
    def residual(r):return r['e2e_latency_ms']-r['worker_latency_ms']-r['worker_queue_wait_ms']-r['schedule_delay_ms']
    d=dict(candidate=cid,arm=o['arm'],task=task,n_matched=n,n_retained=len(same),n_diverted=len(diverted),legal=bool(o['eligible'] and b['eligible']),
     same_worker_service_term_ms=term(same,'worker_latency_ms'),diverted_execution_term_ms=term(diverted,'worker_latency_ms'),
     same_worker_wait_term_ms=term(same,'worker_queue_wait_ms'),diverted_wait_term_ms=term(diverted,'worker_queue_wait_ms'),
     schedule_term_ms=term(pp,'schedule_delay_ms'),transport_collection_residual_term_ms=avg([residual(r)-residual(c) for r,c in pp]),
     mean_e2e_delta_ms=term(pp,'e2e_latency_ms'),scope='Whole task mean; each subgroup sum divided by total matched task count. Native wait includes submit-to-start transport.')
    value=sum(d[k] for k in ['same_worker_service_term_ms','diverted_execution_term_ms','same_worker_wait_term_ms','diverted_wait_term_ms','schedule_term_ms','transport_collection_residual_term_ms'])
    assert abs(value-d['mean_e2e_delta_ms'])<1e-8,(d,value)
    rows.append(d)
 return rows

def main():
 assert (OUT/'campaign_completed.json').exists(),'No post-review while GPU campaign is active'
 assert read(OUT/'v2_supervisor_process.json')['status']=='COMPLETE_NO_NEXT_MODULE','Preparation or GPU measurement is still active'
 gate=read(OUT/'module30_gate.json');f=read(OUT/'frozen_gpu_manifest.json');fv=read(OUT/'frozen_gpu_manifest_v2.json')
 raws=[read(p) for p in sorted((OUT/'raw_runs').glob('*.json'))];obs=read(OUT/'run_observations.json');by={o['run_id']:o for o in obs}
 import csv
 with (OUT/'system_utility_comparison.csv').open() as stream:comparisons=list(csv.DictReader(stream))
 with (OUT/'task_utility_changes.csv').open() as stream:changes=list(csv.DictReader(stream))
 with (OUT/'worker_interference_analysis.csv').open() as stream:interference=list(csv.DictReader(stream))
 decomposition=make_decomposition(raws,obs);csvsave('native_latency_decomposition.csv',decomposition,['candidate','arm','task','legal'])
 # Keep published controller drafts before adding a clearer numerical review.
 drafts=OUT/'controller_generated_reports';drafts.mkdir(exist_ok=True)
 import shutil
 for name in ['module30_summary.md','positive_split_mechanism.md','optimizer_integration_plan.md']:
  if not (drafts/name).exists():shutil.copy2(OUT/name,drafts/name)
 legalcomparisons=[c for c in comparisons if c.get('delta_U') not in ('',None)]
 best=max(legalcomparisons,key=lambda c:float(c['delta_U']),default=None)
 started_ids={r['slot']['run_id'] for r in raws if r.get('phases',{}).get('measurement') or r.get('aborted_phase',{}).get('label')=='measurement'}
 missing=[s['run_id'] for s in f['slots'] if s['run_id'] not in started_ids]
 c2_full=len([o for o in obs if o['candidate']=='C2'])==3 and all(o['eligible'] for o in obs if o['candidate']=='C2')
 completion=read(OUT/'unmeasured_slot_completion_result.json') if (OUT/'unmeasured_slot_completion_result.json').exists() else None
 # Verify exact utility additivity only for eligible full-cohort comparisons.
 additivity=[]
 for c in legalcomparisons:
  rows=[r for r in changes if r['candidate']==c['candidate'] and r['arm']==c['arm']]
  taskcost=sum(float(r['mean_U_delta'])+float(r['p95_U_delta']) for r in rows)
  total=taskcost+float(c['T_delta'])+float(c['M_delta'])
  assert abs(total-float(c['delta_U']))<1e-10
  additivity.append(dict(candidate=c['candidate'],arm=c['arm'],task_LP_delta=taskcost,T_delta=float(c['T_delta']),M_delta=float(c['M_delta']),summed_delta_U=total,reported_delta_U=float(c['delta_U'])))
 schedule=[]
 for raw in raws:
  ph=raw.get('phases',{}).get('measurement')
  if not ph:continue
  for t in raw['slot']['lambda_rps']:
   rs=[r for r in ph['request_records'] if r['task']==t and r['status']=='completed'];values=[r['schedule_delay_ms'] for r in rs]
   import numpy as np
   schedule.append(dict(run_id=raw['slot']['run_id'],telemetry_version=by[raw['slot']['run_id']]['telemetry_version'],task=t,n=len(rs),mean_schedule_delay_ms=avg(values),p95_schedule_delay_ms=float(np.percentile(values,95)) if values else None,max_schedule_delay_ms=max(values) if values else None,eligible=by[raw['slot']['run_id']]['eligible']))
 csvsave('dispatch_observer_audit.csv',schedule,['run_id','telemetry_version','task'])
 save('final_cpu_analysis_acceptance.json',dict(passed=True,native_mean_decomposition_exact=True,original_utility_additivity=additivity,GPU_measurements_started=0,frozen_implementation_untouched=True,post_analysis_code=ref(ROOT/'module30/final_review.py')))
 # Every formal slot, including two affected original windows, remains in the budget.
 assert len(raws)<=6 and len({r['slot']['run_id'] for r in raws})==len(raws)
 expected=['telemetry_fix_audit.md','candidate_pool_ranking.csv','candidate_selection_rationale.md','frozen_gpu_manifest.json','gpu_feasibility_runs.csv','worker_interference_analysis.csv','system_utility_comparison.csv','positive_split_mechanism.md','optimizer_integration_plan.md','module30_gate.json','module30_summary.md']
 assert all((OUT/n).exists() for n in expected)
 v1=read(OUT/'v1_observer_affected_diagnostics.json')
 lines=['# Module30 GPU 实测正分流可行性筛选','',f"最终 Gate：**{gate['gate']}**。已停止；没有启动下一 Module 或独立 GPU 复现。",'',
 'CPU 测前排名：① Qwen2.5-Omni + Whisper-small ASR；② Phi-4-multimodal + Whisper-small ASR。前者 G ASR 更快，后者有 Module15 真实分流画像；排名不代表已存在正效用。39 个已准入共同任务配对的依据、缺失画像和淘汰理由全部保留。', '',
 '固定条件：ASR 1 QPS，Text/Image/OCR/LID/VQA 各 0.2 QPS；每组 G、Whisper-small、RoBERTa-base、ConvNeXt-tiny、PARSeq 各一副本。显存预算 34567.5534 MiB；目标路由 0/5/10%；候选各一条冻结 Poisson 轨迹及匹配输入。每个正式窗口 600 秒、30 秒预热、最长 60 秒排空。所有质量准入、20A 效用和稳定性阈值未改。', '',
 f"正式窗口实际记录：{len(raws)}/6；完整 {gate['formal_runs_completed']}，可用于物理可行性比较的合法稳定窗口 {gate['legal_stable_runs']}。前两个 C1 窗口受初版遥测控制锁影响，合法 U 为 NULL；原始完整 cohort 和算式保留。C1 的 10% 窗口与它们跨版本，禁止计算合法 ΔU。C2 三臂完整合法稳定：{c2_full}；未执行窗口：{missing}。没有重复已经测量的窗口、换种子或新增点。", '',
 '|组合|路由|合法 U|ΔU 对 S-only|G / S 实际 ASR 数|状态|','|---|---|---:|---:|---:|---|']
 for o in obs:
  c=next((c for c in comparisons if c['candidate']==o['candidate'] and c['arm']==o['arm']),None)
  delta=float(c['delta_U']) if c and c.get('delta_U') not in ('',None) else None
  lines.append(f"|{o['candidate']} {o['Generalist']} + Small|{o['arm']}|{num(o['U'])}|{num(delta)}|{o.get('G_completed',0)} / {o.get('S_completed',0)}|{o.get('stability')} / eligible={o['eligible']}|")
 lines+=['', '最终六问：',
 '1. **哪个组合最有潜力？** 测前首选 Qwen + Whisper-small ASR，但 C1 受遥测问题限制，没有完整同版本对照，不能由本轮确认其正收益潜力。'+(f"现有合法匹配对照中最好的是 {best['Generalist']} + Whisper-small、{best['arm']}，ΔU={float(best['delta_U']):+.4f}。" if best else '本轮没有可排序的合法 ΔU。'),
 f"2. **是否实际观测到 G/S 优于 S-only？** 合格探索性正收益：{'是' if gate['positive_candidates'] else '否'}。只接受全对照合法稳定、双方实际执行及 ΔU>+0.01；初版受影响的完整算式差值 {v1.get('diagnostic_observed_U_difference')} 仅作技术诊断。",
 '3. **主要物理变化是什么？** 下面给出同输入服务、等待与背景变化；native_latency_decomposition.csv 将目标 mean 的 S 保留请求执行/等待、G 转交代价、提交与采集残差逐项加总，严格等于实测 mean 差。P95 和系统效用另外按原定义计算。']
 for c in legalcomparisons:
  cid,arm=c['candidate'],c['arm'];d=next((d for d in decomposition if d['candidate']==cid and d['arm']==arm and d['task']=='asr'),None)
  a=next(o for o in obs if o['candidate']==cid and o['arm']==arm);b=next(o for o in obs if o['candidate']==cid and o['arm']=='s_only')
  sp=next(p for p in a['paths'] if p['task']=='asr' and p['model']=='whisper-small');bp=next(p for p in b['paths'] if p['task']=='asr' and p['model']=='whisper-small');gp=next(p for p in a['paths'] if p['task']=='asr' and p['model']==a['Generalist'])
  lines.append(f"   - {cid}/{arm}：S service {bp['service']['mean']:.2f}→{sp['service']['mean']:.2f} ms，S wait {bp['wait']['mean']:.2f}→{sp['wait']['mean']:.2f} ms；G service/wait {gp['service']['mean']:.2f}/{gp['wait']['mean']:.2f} ms。目标 mean/P95 {b['per_task']['asr']['mean_ms']:.2f}/{b['per_task']['asr']['p95_ms']:.2f}→{a['per_task']['asr']['mean_ms']:.2f}/{a['per_task']['asr']['p95_ms']:.2f} ms；ΔU={float(c['delta_U']):+.4f}。")
 bg=[r for r in changes if r['legal']=='True' and r['task']!='asr'];worst=min(bg,key=lambda r:float(r['mean_U_delta'])+float(r['p95_U_delta']),default=None)
 if worst:
  cost=float(worst['mean_U_delta'])+float(worst['p95_U_delta'])
  lines.append(f"4. **背景任务是否损失？** 最差单任务变化为 {worst['candidate']}/{worst['arm']} {worst['task']}：mean {float(worst['mean_delta_ms']):+.2f} ms，P95 {float(worst['p95_delta_ms']):+.2f} ms，ΔU 分项 {cost:+.4f}。所有五背景任务完整分项见下表；观察变化不等于已隔离 GPU 因果。")
 else:lines.append('4. **背景任务是否损失？** 没有合法背景反事实；初版窗口差值标记为不可用于物理判断。')
 lines+=['', '|组合/路由|背景任务|Δmean ms|ΔP95 ms|ΔU（L+P）|','|---|---|---:|---:|---:|']
 for r in bg:lines.append(f"|{r['candidate']}/{r['arm']}|{r['task']}|{float(r['mean_delta_ms']):+.2f}|{float(r['p95_delta_ms']):+.2f}|{float(r['mean_U_delta'])+float(r['p95_U_delta']):+.4f}|")
 lines+=['',f"5. **值得独立长窗口验证的候选？** {'有，见 exploratory_positive_freeze.json；仍仅开发探索证据。' if gate['positive_candidates'] else '没有通过本轮门槛的候选。C1 技术上未定，不包装成负收益证明，也不因未定自动补测。'}", 
 '6. **无正收益的原因更接近什么？** 只对 C2 已完成的合法匹配窗口评估：以 S-only 的真实服务需求和等待识别负载点是否形成可释放排队，以 G/S 服务差与保留 S 同输入膨胀识别执行及共置代价。下述数值支持本轮归因；不推广到未测负载或未完成路由。']
 base=next((o for o in obs if o['candidate']=='C2' and o['arm']=='s_only' and o['eligible']),None)
 if base:
  sw=next(w for w in base['workers'] if w['worker']=='whisper-small#0');sp=next(p for p in base['paths'] if p['task']=='asr' and p['model']=='whisper-small')
  lines.append(f"   - C2 S-only Specialist 完整 cohort 原生服务需求/600s={sw['busy_fraction']:.4f}（非 GPU utilization），ASR service={sp['service']['mean']:.2f} ms，真实 wait={sp['wait']['mean']:.2f} ms。若需求比远低于 1 且 wait 小，当前点的排队红利有限；分流执行差和共置外部代价更易压过它。")
 lines+=['', '遥测审计：Module29 live request 无锁 deepcopy 的原方法已在 CPU 重现同型错误；修复采用 runtime.lock 一致快照、采样/日志串行化、fsync 后提交和原子文件。初版又发现 check 共用长 I/O 锁；v2 将控制状态与 I/O 锁分离。两个版本各做超过原故障时刻的长压；v2 另含慢传感器、完整/中止 cohort、并发写盘和错误注入。每次 GPU 原始记录与 durable journal 逐请求核对，技术失效 U 为 NULL。', '',
 '限制：只有两个固定模型池、每池一个负载点、单匹配轨迹与小 G 样本；C1 缺合法同版本反事实。Module14F–29 全部历史文件与权重保存，Module29 Gate 不改，正式优化器不改。Stage D 仅在有合格正候选时输出 CPU 建议；独立长窗口验证另行执行。', '',
 '关键文件：telemetry_fix_audit.md、candidate_pool_ranking.csv、frozen_gpu_manifest.json（原冻结）、frozen_gpu_manifest_v2.json（仅基础设施修正及未执行槽）、gpu_feasibility_runs.csv、system_utility_comparison.csv、worker_interference_analysis.csv、native_latency_decomposition.csv、module30_gate.json。']
 (OUT/'module30_summary.md').write_text('\n'.join(lines)+'\n')
 mechanism=(OUT/'positive_split_mechanism.md').read_text().replace('# Module30 正分流物理机制','# Module30 G/S 实测效用与干扰机制',1)
 mechanism+='\n## 完整 mean 物理分解与原效用加总\n\n目标 mean 逐项贡献（每一项以全体目标请求数归一；G 子组不能误作全任务均值）：\n\n'
 mechanism+='|组合/路由|保留 S 执行|转交 G 执行|保留 S 等待|转交 G 等待|提交/采集残差|Δmean ms|\n|---|---:|---:|---:|---:|---:|---:|\n'
 for d in decomposition:
  if d['task']=='asr' and d['legal']:
   mechanism+=f"|{d['candidate']}/{d['arm']}|{d['same_worker_service_term_ms']:+.3f}|{d['diverted_execution_term_ms']:+.3f}|{d['same_worker_wait_term_ms']:+.3f}|{d['diverted_wait_term_ms']:+.3f}|{d['schedule_term_ms']+d['transport_collection_residual_term_ms']:+.3f}|{d['mean_e2e_delta_ms']:+.3f}|\n"
 mechanism+='\n所有分解恒等式及 ΔU=各任务 L/P 分项+T+M 在 final_cpu_analysis_acceptance.json 中通过独立加总。P95 不作错误的线性等待分解。\n'
 (OUT/'positive_split_mechanism.md').write_text(mechanism)
 if completion:
  extra='\n准备流程记录：原控制器在 C2_split05 启动前冷却超时后停止；另在 GPU 完全空闲时有界等待同一个冻结冷态条件，只允许补齐此前未测量的原槽，没有重复已测窗口、改阈值或扩展搜索。结果见 unmeasured_slot_completion_result.json，完整方法偏离与温度序列见 frozen_gpu_manifest_unmeasured_slot_completion.json。\n'
  if missing:extra+='\n筛选不完整：C1 两个窗口受初版观察器影响；C2_split05 未进入正式测量。NO_POSITIVE_FEASIBILITY_CANDIDATE 表示本轮没有可认证的正组合，不表示所有冻结比例或未测负载已被负收益排除。缺失窗口没有请求、没有可用效用或延迟统计；没有截尾或替代 U。\n'
  for name in ['module30_summary.md','positive_split_mechanism.md','optimizer_integration_plan.md']:
   p=OUT/name;p.write_text(p.read_text()+extra)
 # Explicit placeholder for an unmeasured slot: unavailable metrics are blank/NULL,
 # never a zero-valued latency or an invented offered/completed request cohort.
 with (OUT/'gpu_feasibility_runs.csv').open() as stream:runs=list(csv.DictReader(stream))
 for r in runs:r['formal_measurement_started']=r['run_id'] in started_ids
 for slot in f['slots']:
  if slot['run_id'] not in by:
   runs.append(dict(run_id=slot['run_id'],candidate=slot['candidate'],arm=slot['arm'],Generalist=slot['Generalist'],Specialist=slot['Specialist'],eligible=False,U=None,
    status='NOT_MEASURED_PRELAUNCH_THERMAL_TIMEOUT',formal_measurement_started=False,offered=None,completed=None,G_completed=None,S_completed=None,
    unavailable_reason='Unchanged cold gate never passed before bounded preparation ended'))
 csvsave('gpu_feasibility_runs.csv',runs,['run_id','candidate','arm','eligible','U','formal_measurement_started'])
 if gate['positive_candidates']:
  positive=read(OUT/'exploratory_positive_freeze.json');positive.update(corrected_manifest=ref(OUT/'frozen_gpu_manifest_v2.json'),implementation_v2=fv['implementation_v2'],physical_comparison_version='v2_only');save('exploratory_positive_freeze.json',positive)
  plan=(OUT/'optimizer_integration_plan.md').read_text();plan+='\n本次实际数值和逐任务效用加总见 module30_summary.md 与 native_latency_decomposition.csv；冻结代码必须同时包含 frozen_gpu_manifest_v2.json 的 implementation_v2。不能用 v1 窗口训练收益判据。\n';(OUT/'optimizer_integration_plan.md').write_text(plan)
 gate.update(final_CPU_review_passed=True,latency_decomposition=ref(OUT/'native_latency_decomposition.csv'),utility_additivity=ref(OUT/'final_cpu_analysis_acceptance.json'),report_summary=ref(OUT/'module30_summary.md'))
 gate.update(stopped=True,provisional=False,completion_in_progress=False,unexecuted_formal_windows=missing,C2_all_three_legal_stable=c2_full)
 gate.update(actual_formal_windows_started=len(started_ids),actual_completed_formal_windows=sum(r['status']=='COMPLETED' for r in raws),
  screening_complete=not missing and gate['C1_controls_comparable'],all_candidate_routes_ruled_out=False,
  methodological_preparation_deviation=ref(OUT/'frozen_gpu_manifest_unmeasured_slot_completion.json') if completion else None,
  nonmeasurement_preparation_result=completion,measured_window_retry_count=0)
 save('module30_gate.json',gate)
 print('FINAL_REVIEW',gate['gate'],'best=',best,'worst_background=',worst,flush=True)
if __name__=='__main__':main()
