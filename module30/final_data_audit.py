"""Final CPU data acceptance and empirical interpretation; no GPU launch."""
import csv,math
from .common import *
def main():
 assert read(OUT/'v2_supervisor_process.json')['status']=='COMPLETE_NO_NEXT_MODULE'
 f=read(OUT/'frozen_gpu_manifest.json');obs=read(OUT/'run_observations.json');by={o['run_id']:o for o in obs}
 raws=[read(p) for p in sorted((OUT/'raw_runs').glob('*.json'))]
 assert len(raws)==6 and {r['slot']['run_id'] for r in raws}=={s['run_id'] for s in f['slots']}
 total=0;details=[]
 for r in raws:
  o=by[r['slot']['run_id']];p=r['phases']['measurement'];rs=p['request_records'];total+=len(rs)
  assert p['window']['duration_seconds']==600 and all(q['status']=='completed' for q in rs)
  assert r['valid_session'] and r['gpu_release']['verified'] and not r['errors']
  assert o['journal_audit']['passed'] and o['matched_trace'] and o['native_timings_consistent']
  assert all(v['exit_code']==0 and v['bye_received'] and not v['forced'] for v in r['shutdown'].values())
  assert r['telemetry']['collector_exited'] and not r['telemetry']['errors']
  assert len(r['replica_to_family'])==5 and all(q>0 for q in r['slot']['lambda_rps'].values())
  bad=[]
  for v in r['telemetry']['samples']:
   assert not v['foreign_pids']
   for proc in v['processes']:
    if proc['pid'] in v['owned_pids'] and (proc['used_mib'] is None or not math.isfinite(proc['used_mib'])):bad.append(proc)
  assert not bad
  for k in ['postwarm_idle','postmeasurement_idle']:
   assert r[k] and all(v['owned_used_mib']>0 and v['each_owned_pid_has_cuda_context'] for v in r[k])
  details.append(dict(run_id=r['slot']['run_id'],complete_requests=len(rs),journal_terminal=o['journal_audit']['terminal'],request_accounting_passed=True,
   native_timings_passed=True,owned_memory_numeric=True,clean_exit_and_GPU_release=True,eligible=o['eligible'],U=o['U']))
 matched={}
 for cid in ['C1','C2']:
  group=[r for r in raws if r['slot']['candidate']==cid]
  def signature(r,key):return [(q['task'],q['sample_id'],q['pool_id']) for q in r[key]]
  probe=[signature(r,'path_probe_records') for r in group];assert probe[0]==probe[1]==probe[2]
  for label in ['warmup','measurement']:
   cohorts=[r['warmup'] if label=='warmup' else r['phases']['measurement'] for r in group]
   ss=[[(q['task'],q['sample_id'],q['pool_id'],q['frozen_offset_seconds'],q['routing_uniform'],q['replica_uniform']) for q in p['request_records']] for p in cohorts]
   assert ss[0]==ss[1]==ss[2]
  matched[cid]=dict(path_probes_identical=True,warmup_and_formal_trace_and_inputs_identical=True)
 before=read(OUT/'v1_controller_outputs/telemetry_fix_gate.json')['Module29_gate']
 assert ref(before['path'])['sha256']==before['sha256']
 assert read(OUT/'historical_integrity_after.json')['passed']
 assert read(OUT/'final_cpu_analysis_acceptance.json')['passed']
 recovery=read(OUT/'frozen_gpu_manifest_unmeasured_slot_completion.json');assert recovery['cold_ready'] and recovery['measured_window_retries']==0
 samples=recovery['idle_samples'];assert all(v['temperature_gpu_c']<=38 for v in samples[-3:])
 idle=(samples[-1]['time_ns']-samples[0]['time_ns'])/1e9
 result=dict(passed=True,formal_windows=6,formal_seconds_each=600,total_formal_requests=total,all_offered_requests_completed=True,
  formal_retry_count=0,valid_for_utility_comparison=4,observer_affected_windows=2,matched_inputs=matched,per_run=details,
  original_Module29_gate_unchanged=True,historical_files_preserved=17642,additional_idle_seconds_between_first_last_samples=idle,
  additional_idle_samples=len(samples),original_cold_condition_passed=True,CPU_audit_code=ref(ROOT/'module30/final_data_audit.py'))
 save('final_data_acceptance.json',result)
 conclusion='''\n## 原因判断\n\n本轮没有探索性正收益。当前固定ASR=1QPS点的S-only原生服务需求仅约20.7%窗口，真实ASR等待约38ms；没有足够排队红利抵消慢G执行。Phi-ASR平均执行在两种分流中约1.44–1.74秒，S约0.21–0.22秒。G的平均剩余容量虽然很多，长ASR仍会阻塞同一串行worker的VQA。\n\n因此原因更接近 **G/S性能差距 + 当前负载点S压力不足 + G共享队列对VQA的外部代价**。S同输入共置服务膨胀确实存在，但不是唯一或主要解释；10%同输入S服务增加约5.72ms（2.75%）、目标S等待增加约0.19ms。5%和10%的VQA效用损失分别约-0.1356/-0.3102，均大于任何背景改善。不是S等待下降却被其它项抵消：目标S等待在这两个窗口都没有下降。\n\nC2三臂完整合法稳定，两比例均负收益；C1因两窗观察器影响缺少合法同版本反事实，不能声称已否定Qwen/Small组合所有比例。没有通过门槛的独立长窗口验证候选；Stage D未执行，正式优化器未改。全部结论限于本次固定池、负载和轨迹。\n'''
 for name in ['module30_summary.md','positive_split_mechanism.md']:
  p=OUT/name;p.write_text(p.read_text()+conclusion)
 p=OUT/'telemetry_fix_audit.md';p.write_text(p.read_text()+f'\n## 最终数据验收\n\n额外空闲准备首末采样间隔 {idle:.3f}秒，{len(samples)}个采样，最后3个均≤38°C，随后原300秒启动前及驻留≤40°C检查通过；无阈值放宽。第六窗口只执行一次，600秒，1162请求全完成。全轮6个正式600秒窗口、{total}个正式请求全完成，warmup/正式输入轨迹以及native prime样本逐候选三臂完全相同。所有6个journal核对、native时间等式、线程/worker清洁退出及GPU释放通过；NVML owned-process显存全部数值有效。2个初版观察器窗口U仍为NULL，另4个合法U。详见final_data_acceptance.json。\n')
 gate=read(OUT/'module30_gate.json');gate.update(final_data_acceptance=ref(OUT/'final_data_acceptance.json'),formal_windows_started=6,formal_windows_completed=6,
  formal_window_retry_count=0,prelaunch_failure_count=1,additional_idle_preparation=True,report_summary=ref(OUT/'module30_summary.md'),
  C2_status='COMPLETE_LEGAL_STABLE_NONPOSITIVE',Stage_B_execution_complete=True,screening_complete=False,C1_technical_limitation_preserved=True)
 save('module30_gate.json',gate)
 print('FINAL_DATA_ACCEPTANCE',total,'requests; 6 x 600 seconds; matching input/prime; unchanged history; no GPU launched')
if __name__=='__main__':main()
