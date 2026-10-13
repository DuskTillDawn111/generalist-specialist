"""Raw six-task serving evidence, honest utilization units, final gate and stop."""
from collections import Counter
import numpy as np
from .common import *
from .gpu import memory_values

def summarize(raw,f):
 slot=raw['slot'];ph=raw['phases'].get('measurement',{});records=ph.get('request_records',[]);mapping=raw['replica_to_family'];data=read(f['catalog']['path']);roles={m:data['models'][m]['role'] for m in data['models']};memory=memory_values(raw,data['safe_factor']);asr=[r for r in records if r['task']=='asr'];tot=len(asr);g=sum(roles[mapping[r['model']]]=='generalist' for r in asr);s=tot-g;gc=sum(r['status']=='completed' and roles[mapping[r['model']]]=='generalist' for r in asr);sc=sum(r['status']=='completed' and roles[mapping[r['model']]]=='specialist' for r in asr);deployed_asrS=[m for m in slot['deployment'] if roles[m]=='specialist' and 'asr' in data['models'][m]['supported_tasks']]
 condition=(not deployed_asrS and gc>0 and s==0) if slot['stage']=='LOW' else (bool(deployed_asrS) and tot>0 and s/tot>=.8) if slot['stage']=='MID' else (bool(deployed_asrS) and gc>0 and sc>0)
 six=all(any(r['task']==k and r['status']=='completed' for r in records) for k in slot['lambda_rps']);st=ph.get('stability',{}).get('stability_state_v2');m=ph.get('metrics',{});passed=bool(raw['valid_session'] and memory['passed'] and condition and six and st=='steady')
 row={'run_id':slot['run_id'],'stage':slot['stage'],'repetition':slot['repetition'],'seed':slot['seed'],'actual_deployment':slot['deployment'],'actual_worker_PID':{i:v['pid'] for i,v in raw['worker_identities'].items()},'CPU_max_rho':slot['CPU_max_rho'],'actual_ASR_G_fraction':g/tot if tot else None,'actual_ASR_S_fraction':s/tot if tot else None,'ASR_G_completed':gc,'ASR_S_completed':sc,'six_tasks_completed':six,'valid_session':raw['valid_session'],'stability':st,'stage_condition':condition,'memory_passed':memory['passed'],'passed':passed,'mean_E2E_ms':m.get('mean_e2e_ms'),'P95_E2E_ms':m.get('p95_e2e_ms'),'queue_mean_ms':m.get('mean_queue_wait_ms'),'queue_P95_ms':m.get('p95_queue_wait_ms'),'completion_ratio':m.get('completion_ratio'),'within_window_completion_ratio':m.get('within_window_completion_ratio'),'drain_seconds':ph.get('window',{}).get('drain_seconds'),'stability_detail':ph.get('stability'),'errors':raw['errors'],'release_verified':raw['gpu_release']['verified'],'raw':ref(OUT/'raw_runs'/(slot['run_id']+'.json'))}
 route=[];e2e=[];models=[];dur=ph.get('window',{}).get('duration_seconds',120.)
 for task,q in slot['lambda_rps'].items():
  cohort=[r for r in records if r['task']==task];N=len(cohort)
  for fam in slot['deployment']:
   rs=[r for r in cohort if mapping[r['model']]==fam]
   route.append({'run_id':slot['run_id'],'stage':slot['stage'],'task':task,'model':fam,'role':roles[fam],'predicted_QPS':slot['routing'].get(task,{}).get(fam,0),'predicted_fraction':slot['routing'].get(task,{}).get(fam,0)/q,'actual_offered_requests':len(rs),'actual_completed_requests':sum(r['status']=='completed' for r in rs),'actual_offered_QPS':len(rs)/dur,'actual_fraction':len(rs)/N if N else None})
  e2e.append({'run_id':slot['run_id'],'stage':slot['stage'],'task':task,**ph.get('per_task_metrics',{}).get(task,{})})
 e2e.append({'run_id':slot['run_id'],'stage':slot['stage'],'task':'ALL','mean_e2e_latency_ms':m.get('mean_e2e_ms'),'p95_e2e_latency_ms':m.get('p95_e2e_ms')})
 start=ph.get('window',{}).get('start_unix_ns',0);end=start+int(dur*1e9)
 samples=[v for v in raw['telemetry']['samples'] if v['phase']=='measurement' and start<=v['unix_ns']<end]
 for fam,count in slot['deployment'].items():
  ids={i for i,v in mapping.items() if v==fam};pids={v['pid'] for i,v in raw['worker_identities'].items() if i in ids};rs=[r for r in records if r['model'] in ids];done=[r for r in rs if r['status']=='completed'];busy=sum(max(0,min(r.get('worker_response',{}).get('end_ns',0),end)-max(r.get('worker_response',{}).get('start_ns',0),start)) for r in done)/1e9
  gpu=[u['sm_percent'] for v in samples for u in v.get('per_process_gpu_utilization',[]) if u['pid'] in pids];cpu=[p['cpu_core_percent'] for v in samples for p in v['host_processes'] if p['root_worker_pid'] in pids and p['cpu_core_percent'] is not None];rho=sum(len([r for r in rs if r['task']==k])/dur*data['models'][fam]['service_time_ms'][k]/1000 for k in data['models'][fam]['supported_tasks'])/data['capacity_factors'][fam][str(count)]
  models.append({'run_id':slot['run_id'],'stage':slot['stage'],'model':fam,'worker_count':count,'offered_requests':len(rs),'completed_requests':len(done),'within_window_completed':sum(r['completion_time_ns']<=ph['window']['end_time_ns'] for r in done),'cohort_throughput_rps':len(done)/dur,'handler_busy_wall_fraction':busy/(dur*count),'calibrated_actual_offered_rho':rho,'per_process_GPU_SM_percent_mean':statistics.mean(gpu) if gpu else None,'GPU_utilization_status':'MEASURED_NVML_PROCESS_SAMPLES' if gpu else 'UNAVAILABLE_OR_NO_VALID_PROCESS_SAMPLES','host_CPU_core_percent_mean':statistics.mean(cpu) if cpu else None,'device_GPU_percent_mean':statistics.mean(v['gpu_util_percent'] for v in samples) if samples else None,'queue_wait_ms_mean':statistics.mean(r['worker_queue_wait_ms'] for r in done) if done else None,'queue_wait_ms_P95':float(np.percentile([r['worker_queue_wait_ms'] for r in done],95)) if done else None})
 return row,route,e2e,{'run_id':slot['run_id'],'stage':slot['stage'],'predicted_safe_mib':slot['CPU_safe_memory_mib'],**memory},models

def main():
 f=read(OUT/'frozen_gpu_manifest.json');verify(f.get('implementation',[])+f.get('reused_implementation',[]));rows=[];routes=[];e2e=[];mem=[];models=[]
 for slot in f['slots']:
  path=OUT/'raw_runs'/(slot['run_id']+'.json')
  if path.exists():
   a,b,c,d,e=summarize(read(path),f);rows.append(a);routes+=b;e2e+=c;mem.append(d);models+=e
 save('gpu_validation_rows.json',rows)
 for name,values,fields in [('gpu_runs.csv',rows,['run_id','stage','stability','passed']),('per_task_routing.csv',routes,['run_id','task','model']),('e2e_summary.csv',e2e,['run_id','task','mean_e2e_latency_ms','p95_e2e_latency_ms']),('memory_summary.csv',mem,['run_id','budget_mib','passed']),('per_model_metrics.csv',models,['run_id','model','cohort_throughput_rps','handler_busy_wall_fraction'])]:csvsave(name,values,fields)
 passed=len(rows)==9 and all(r['passed'] for r in rows);gate='FULLPOOL_THREE_PHASE_REAL_VALIDATED' if passed else 'FULLPOOL_GPU_REJECTED' if rows else 'FULLPOOL_THEORY_ONLY';failure=next((r for r in rows if not r['passed']),None)
 stages=[{'stage':stage,'runs':sum(r['stage']==stage for r in rows),'steady':sum(r['stage']==stage and r['stability']=='steady' for r in rows),'passed':sum(r['stage']==stage and r['passed'] for r in rows)} for stage in ['LOW','MID','HIGH']]
 verify(read(OUT/'protected_before.json')['files']);pids=[pid for r in rows for pid in r['actual_worker_PID'].values()];assert len(set(pids))==len(pids)
 save('module16_gate.json',{'at_utc':now(),'gate':gate,'CPU_three_phase_reproduced':read(OUT/'cpu_witness_revalidation.json')['three_phase_reproduced'],'stages':stages,'GPU_runs':len(rows),'first_failure_stage':failure['stage'] if failure else None,'first_failure':failure,'campaign':read(OUT/'campaign_status.json') if (OUT/'campaign_status.json').exists() else None,'full_pool_Generalists_competed':['qwen','phi4mm','minicpm'],'full_14F_Heavy_pool_used':True,'original_V3_unchanged_independent_CPU_baseline':True,'independent_resource_policy_same_J_certified':True,'latency_SLO_claimed':False,'new_models_or_capacity_profiling':0,'frozen_inputs_moved_after_GPU':False,'history_integrity_verified':True,'all_worker_PIDs_independent':True,'stop':True})
 text='# Module 16：Full-Pool Three-Phase GPU Validation\n\nGate：**'+gate+'**。完成后停止。\n\n固定预算27980.1894 MiB；text=1.1664,image=0.3729,ocr=0.5501,lid=12.0,visual_qa=0.3603 QPS；ASR LOW=0.12,MID=1.00,HIGH=3.80。Qwen/Phi/MiniCPM和14F完整Heavy pool全部参与CPU竞争，无强制模型；quality、真实实测副本表、safe-factor、rho≤0.8与J目标不变，无人为延迟SLO。\n\n'
 text+='资源优先：全池MILP最小化safe residency，并枚举全部同等最小资源state后按原J选择。HIGH原Module9数值结果未通过1e-8秒证书；独立同目标CPU Newton精化通过同一严格凸LP最优性证书。未修改Module9或原V3，原V3独立CPU dry-run与数值失败完整保留；不把独立精化称为原V3成功。CPU复核与竞争见cpu_witness_revalidation.json。\n\n'
 for p in read(OUT/'cpu_witness_revalidation.json')['points']:
  w=p['full_pool_result']['winner'];text+=f"{p['stage']}：最优部署 {w['deployment'] if w else None}；safe={w['safe_memory_mib'] if w else None} MiB；max rho={w['max_rho'] if w else None}；J={w['J_seconds'] if w else None}秒。\n\n"
 text+='冻结后顺序LOW→MID→HIGH；每点最多三次独立进程，10秒warmup+120秒测量，seeds16101/16202/16303，原完整六任务样本流、native handlers、scheduled arrival-to-result E2E及stability_v2阈值。首个unstable或运行/显存不可行立即停止，不改点、不重试。\n\n|run|steady状态|ASR G实际份额|ASR G/S完成数|六任务完成|safe预算|通过|\n|---|---|---:|---|---|---|---|\n'
 for r in rows:text+=f"|{r['run_id']}|{r['stability']}|{r['actual_ASR_G_fraction']}|{r['ASR_G_completed']}/{r['ASR_S_completed']}|{r['six_tasks_completed']}|{r['memory_passed']}|{r['passed']}|\n"
 text+='\nper_task_routing.csv来自实际request记录，e2e_summary.csv逐任务报告E2E mean/P95；memory_summary.csv单列factor×两个idle窗口与sampled owned peak。per_model_metrics.csv包含throughput、真实worker handler busy wall fraction、基于实测s0/g的offered rho、host CPU和NVML process GPU utilization；如设备不支持process SM利用率明确留空，handler busy不是GPU SM占用率。device utilization单列，不能归因到个别模型。raw保留队列/尾部增长、完整请求、actual PIDs、READY和遥测，所有进程独立且释放核验。\n\n'
 if failure:text+=f"首个未通过阶段：{failure['stage']} / {failure['run_id']}。CPU预测可行而GPU观察={failure['stability']}，错误={failure['errors']}，显存通过={failure['memory_passed']}。具体队列/运行原因见gate.first_failure和raw，不把数值预测当真实稳定性。未移动冻结工作负载或路由。\n\n"
 text+='这只验证三个离散120秒development serving点与独立重复，不宣称连续稳定边界、未定义延迟SLO或未测副本能力。全部历史保护文件hash核验。\n';(OUT/'module16_summary.md').write_text(text)
 save('artifact_manifest.json',{'at_utc':now(),'gate':gate,'artifacts':[ref(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'],'implementation':[ref(p) for p in sorted((ROOT/'module16').glob('*.py'))],'status':'COMPLETE_AND_STOPPED'})
 print('FINAL_GATE',gate,stages,flush=True)
if __name__=='__main__':main()
