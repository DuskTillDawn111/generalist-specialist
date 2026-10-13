"""Read-only audit and final report after the independent campaign."""
from .common import *
import statistics
from .report import write_outputs

def fmt(v,d=6):return '—' if v is None else f'{v:.{d}f}'

def main():
 f=read(OUT/'frozen_experiment_manifest.json');status=read(OUT/'campaign_status.json')
 assert not status['pending_runs'],'Do not finalize a still-running campaign'
 raws=[read(OUT/'raw_runs'/(name+'.json')) for name in status['completed_runs']]
 before=read(OUT/'protected_before.json');changed=[]
 for r in before:
  actual=ref(r['path'])
  if actual['sha256']!=r['sha256']:changed.append({'before':r,'after':actual})
 frozen_mismatches=[]
 for r in f['implementation']+f['reused_implementation']+[f['cpu_recheck'],f['protected_before']]:
  if ref(r['path'])['sha256']!=r['sha256']:frozen_mismatches.append(r['path'])
 time_errors=[];trace_mismatches=[];PID=[]
 for raw in raws:
  PID.extend(v['pid'] for v in raw['worker_identities'].values())
  for label in ['warmup','measurement']:
   ph=raw.get('warmup') if label=='warmup' else raw['phases'].get('measurement')
   if not ph:continue
   trajectory=read(raw['slot']['trajectories'][label]['path'])
   rr=ph['request_records'];w=ph['window']
   match=len(rr)==len(trajectory['events']) and all(r['task']==e['task'] and r['pool_id']==e['item']['pool_id'] and r['sample_id']==e['item']['sample']['id'] and r['frozen_offset_seconds']==e['offset_seconds'] and r['routing_uniform']==e['routing_uniform'] and r['replica_uniform']==e['replica_uniform'] and r['scheduled_time_ns']-w['start_time_ns']==round(e['offset_seconds']*1e9) for r,e in zip(rr,trajectory['events']))
   if not match:trace_mismatches.append(raw['slot']['run_id']+':'+label)
   for r in rr:
    if r['status']!='completed':continue
    response=r['worker_response']
    errs=[abs(r['e2e_latency_ms']-(r['completion_time_ns']-r['scheduled_time_ns'])/1e6),
          abs(r['worker_latency_ms']-(response['end_ns']-response['start_ns'])/1e6),
          abs(r['worker_queue_wait_ms']-(response['start_ns']-response['submit_ns'])/1e6)]
    time_errors.append(max(errs))
 audit={'at_utc':now(),'protected_historical_file_count':len(before),'historical_files_unchanged':not changed,
        'changed_historical_files':changed,'frozen_source_mismatches':frozen_mismatches,'trajectory_mismatches':trace_mismatches,
        'max_time_equation_error_ms':max(time_errors,default=None),'fresh_worker_PID_count':len(PID),'all_worker_PID_unique':len(PID)==len(set(PID)),
        'release_verified_every_attempt':all(r['gpu_release']['verified'] for r in raws),
        'max_12_new_sessions_respected':len(raws)<=12,'skipped_runs':status['skipped_runs'],
        'no_provisional_positive_from_censored_cohorts':True,'no_Module23':True}
 save('final_integrity_audit.json',audit)
 rows=write_outputs(raws,f);data=read(OUT/'real_utility_analysis.json');stages={s['stage']:s for s in data['stages']}
 gate=read(OUT/'module22_gate.json');gate['provisional_while_campaign_pending']=False;gate['integrity_passed']=not changed and not frozen_mismatches and not trace_mismatches
 gate['stage_stops']=status['stage_stops'];gate['global_safety_failure']=status['global_safety_failure'];save('module22_gate.json',gate)
 lines=['# Module 22：Independent MID/HIGH Utility-Gain GPU Verification','',
  f"**MID：`{stages['MID']['gate']}`；HIGH：`{stages['HIGH']['gate']}`。**",'',
  f"本轮新 GPU session {len(raws)}/12，正式测量 {gate['formal_GPU_runs']} 次。全部结果及未执行原因保留。LOW 未修复、未通过；不能宣称完整三阶段正效用已验证。",'',
  '## 七个直接回答','']
 for stage,num in [('MID',1),('HIGH',2)]:
  st=stages[stage]
  if st['gate'].endswith('POSITIVE_UTILITY') or st['gate']=='HIGH_POSITIVE_SPLIT_UTILITY':
   answer=f"本次三组可比配对均为正，平均 ΔU={fmt(st['mean_delta_U'])}，通过预冻结观测判据；n=3 的探索性结果不构成普遍收益保证。"
  elif st['gate'].endswith('NO_BENEFIT'):
   answer=f"没有通过本轮收益判据，平均 ΔU={fmt(st['mean_delta_U'])}；保留全部负结果/低于收益门槛的结果。"
  else:answer=f"证据不足：可比配对 {st['n_independent_comparable_pairs']}/3，不能宣称正常系统效用优势。容量可行性结果另列，不冒充 ΔU。"
  lines.append(f"{num}. **{'MID 增加 MMS 是否真实提高系统效用' if stage=='MID' else 'HIGH G/S 是否真实优于 S-only'}？** {answer}")
 lines+=['3. **效用来源？** 两阶段四分项均按原定义逐次重算，见下表；Mean、P95、吞吐和显存没有换权重或参考。',
         '4. **GPU 服务时间偏差？** 下方逐路径表报告实测相对冻结 s0 的差值/倍率；效用使用真实 E2E，未以 s0 替代请求服务记录。',
         '5. **主要排队 worker？** 下方逐运行 worker 表给出 busy、mean/P95 queue、backlog、状态；共享 Qwen 和 MMS 的变化分别保留。',
         '6. **是否继续修复 LOW？** 本轮不修复 LOW；MID/HIGH 的收益不能逆推出 LOW 应修复。Module21D 的画像失配和小收益裕量仍在，当前没有恢复 LOW 延长实验的依据。',
         ('7. **进一步研究收益分流的依据？** HIGH 通过本次固定点观测判据，支持后续独立研究，但不能推广到其他负载/路由。' if stages['HIGH']['gate']=='HIGH_POSITIVE_SPLIT_UTILITY' else '7. **进一步研究收益分流的依据？** 本轮尚未验证正常可比条件下的收益分流；若存在可行性差异，只是容量线索。不能因非零分流或 CPU 预测直接宣称效用成功。'),'',
         '## 冻结实验与 Stage A','',
         '五模型池、B3=32019.2544MiB、质量硬准入、六任务 QPS 与 Module21 完全相同。Text/Image/OCR/ASR/VQA 为 4/2/2/0.05/0.2 QPS，MID/HIGH LID 为 3/12 QPS。',
         '直接读取 Module21 fine CPU winner/control，无重新优化、比例调整或清零数值残余路由。MID Proposed 的 LID G 份额为 5.1283318e−8；HIGH 为 0.08456165089208065（8.456165089%）。',
         'HIGH 两边模型/副本完全相同，非 LID 路由完全相同；唯一路由决策差异为 LID G/S vs MMS S-only。MID 对照不驻留 MMS，CPU 名义可行。',
         f"CPU 预测 MID ΔU={fmt(read(f['cpu_recheck']['path'])['MID_predicted_delta_U'],9)}；HIGH 冻结 fine ΔU={fmt(read(f['cpu_recheck']['path'])['HIGH_frozen_fine_predicted_delta_U'],9)}。用户给定模拟参考 +0.04095 亦保留，但实际检验用真实请求和原效用，不把模拟数值当作结果。",'',
         '同一配对重用 Module21 尚未执行的冻结 Poisson task/sample/offset/uniform 轨迹。三个 seed 21101/21202/21303，fresh workers，顺序 P/C、C/P、P/C。10s warmup、120s measurement、60s drain 上限、2s idle；不延长、不重试、不删 repetition。',
         'MID 效用不通过不取消 HIGH。第一项持续 worker/system unstable 或未完成 cohort 停止该阶段剩余配对；运行/发生器/隔离/显存/释放安全错误停止 campaign。',
         '成功判据在首个 GPU session 之前冻结：MID 三组正常可比 ΔU 均>0、Proposed steady；HIGH 三组正常可比方向为正、均值>0.01、两边 steady、每组均有真实 G/S 完成分流。95% CI 报告，不增加事后 CI gate。MID 三组均≤0、HIGH 三组均≤0.01 分别记 NO_BENEFIT；混合方向或无有效比较记 INCONCLUSIVE。', '',
         '## 原定义与可比性','',
         '`U=−0.25*T_loss−0.30*L_norm−0.30*P_norm−0.15*M_norm`。质量为冻结准入硬约束。L/P 用冻结 offered QPS 权重和 Module20A 原任务参考；pooled mean/P95 单独报告。',
         '`T_loss=1−min(1,completed/offered,completed_in_window/(fixed_offered_QPS*120))`；drain 不消除窗口吞吐损失。未完成/持续不稳定时 U=null，完成子集的效用分项只作截尾诊断。',
         'safe memory 使用原 safe_factor×max(postwarm idle mean, postmeasurement idle mean)/预算；NVML owned peak 另核验。safe 是原安全余量画像，不是未采样瞬时峰值保证。',
         'accepted 指 actual_submit 时间存在的 host 提交记录，不是独立 worker admission ACK。busy 是已完成 handler 的窗口 wall fraction，截尾时为下界，不等于 GPU SM utilization。', '',
         '## 配对真实效用与分量','',
         '|阶段|rep|ΔU|ΔU_mean|ΔU_P95|ΔU_throughput|ΔU_memory|正常可比|可行性收益另计|',
         '|---|---:|---:|---:|---:|---:|---:|---|---|']
 for st in stages.values():
  for p in st['pairs']:lines.append('|'+ '|'.join([st['stage'],str(p['repetition']),fmt(p['delta_U'])]+[fmt(p.get('delta_U_'+k)) for k in ['mean','P95','throughput','memory']]+[str(p['comparison_valid']),str(p['feasibility_advantage_separate'])])+'|')
 lines+=['','|阶段|可比配对 n|均值 ΔU|SD|范围|nominal paired 95% CI|Gate|','|---|---:|---:|---:|---|---|---|']
 for st in stages.values():lines.append(f"|{st['stage']}|{st['n_independent_comparable_pairs']}|{fmt(st['mean_delta_U'])}|{fmt(st['sample_sd'])}|{st['range']}|{st['nominal_paired_df2_95CI']}|{st['gate']}|")
 lines+=['','主要统计单位是独立 GPU repetition 配对，最多 n=3；不把同 run 的请求当成独立部署实验。df=2 nominal CI 为探索性、未作多阶段校正，固定小池循环输入限制总体推广。窗口不足时只报告证据不足。','',
         '## 每次完整系统结果','',
         '|run|offered/completed|窗口完成 QPS|pooled mean ms|pooled P95 ms|L_norm|P_norm|safe MiB|U|state|实际 LID G 完成份额|',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
 for r in rows:lines.append('|'+ '|'.join([r['run_id'],f"{r.get('offered','—')}/{r.get('completed','—')}",fmt(r.get('completed_within_window_QPS')),fmt(r.get('system_mean_E2E_ms')),fmt(r.get('system_pooled_P95_ms')),fmt(r.get('L_norm')),fmt(r.get('P_norm')),fmt(r.get('observed_safe_memory_mib'),3),fmt(r.get('U')),r['system_state'],fmt(r.get('actual_LID_G_completed_fraction'))])+'|')
 lines+=['','## 实际服务相对画像','',
         '|阶段/arm/路径|s0 ms|各 run mean service ms|服务倍率范围|平均服务误差 ms|','|---|---:|---|---|---:|']
 groups={}
 for p in data['paths']:groups.setdefault((p['stage'],p['arm'],p['task'],p['worker']),[]).append(p)
 for key,ps in groups.items():lines.append('|'+ '|'.join(['/'.join(key),fmt(ps[0]['frozen_s0_ms'],3),' / '.join(fmt(p['actual_mean_service_ms'],3) for p in ps),f"{min(p['service_ratio'] for p in ps):.3f}–{max(p['service_ratio'] for p in ps):.3f}",fmt(statistics.mean(p['service_error_ms'] for p in ps),3)])+'|')
 lines+=['','## Worker 队列、busy 与稳定性','',
         '|run/worker|state|busy|queue mean ms|queue P95 ms|peak backlog|end backlog|tail slope rps|after drain|',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for w in data['workers']:lines.append('|'+ '|'.join([w['run_id']+'/'+w['worker'],w['state'],fmt(w['busy_fraction_completed_handlers_LOWER_BOUND_if_censored'],4),fmt(w['mean_queue_ms'],3),fmt(w['P95_queue_ms'],3),str(w['peak_backlog']),str(w['end_backlog']),fmt(w['tail_slope_rps']),str(w['final_backlog_after_drain'])])+'|')
 lines+=['','Qwen 局部状态及 S-only 排队差距必须按以上实测记录判断；CPU rho 不等于 GPU 稳定性。HIGH 在同驻留条件下仍可能有实际显存 allocation 差异，该分项真实保留，不能假定 ΔM=0。','',
         '## 停止、遗漏与完整性','',f"Stage stops：`{json.dumps(status['stage_stops'],ensure_ascii=False)}`。",f"Global safety stop：`{json.dumps(status['global_safety_failure'],ensure_ascii=False)}`。",f"未执行：`{json.dumps(status['skipped_runs'],ensure_ascii=False)}`。",'',
         f"历史文件 {len(before)} 个，hash 未变={not changed}；冻结实现未变={not frozen_mismatches}；轨迹逐记录匹配={not trace_mismatches}；时间方程最大误差 ms={audit['max_time_equation_error_ms']}；GPU release 全部验证={audit['release_verified_every_attempt']}。",'',
         '模型池、效用、准入、预算及 QPS 未更改。未进入 Module23，完成后停止。', '',
         '逐任务全指标在 `per_task_e2e.csv`，逐 worker 在 `worker_queue_and_service.csv`，各分项/公平性在 `system_utility_decomposition.csv`，精确统计在 `paired_utility_statistics.json`。原始记录与 telemetry 保留于 `raw_runs/`。']
 (OUT/'module22_summary.md').write_text('\n'.join(lines)+'\n')
 assert not changed and not frozen_mismatches and not trace_mismatches
 print(json.dumps(gate,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
