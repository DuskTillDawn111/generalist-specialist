"""Six bounded exploratory arms. Await infrastructure gate; immutable native workers."""
import os,time,copy,traceback,fcntl
from types import SimpleNamespace
import numpy as np
from .common import *
from .telemetry import SafeMonitor
from module22.gpu import PoissonConfig
from module21.trajectory import choose
from module11b.measure import prime
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from joint_deploy.measurement.run_measurement import new_record,wait_until
from joint_deploy.measurement.metrics import annotate_records,summarize
from replica_scaling.telemetry import snapshot,release
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS

class AuditedRuntime(DeploymentRuntime):
 def close(self):
  report=super().close()
  live=[name for name,t in self.senders.items() if t.is_alive()]
  if self.collector and self.collector.is_alive():live.append('collector')
  live += ['reader:'+name for name,w in self.workers.items() if w.process and not w.reader_done.is_set()]
  if live:self.errors.append(dict(error='RUNTIME_THREADS_DID_NOT_EXIT',threads=live))
  return report

def phase(runtime,config,slot,plan,mapping,label,monitor):
 trajectory=read(slot['trajectories'][label]['path']);assert ref(slot['trajectories'][label]['path'])['sha256']==slot['trajectories'][label]['sha256']
 events=trajectory['events'];duration=trajectory['duration'];start=time.perf_counter_ns();end=start+round(duration*1e9);unix=time.time_ns()
 records=[]
 for i,e in enumerate(events):
  r=new_record(label,i,e['task'],choose(e,slot['routing'],slot['deployment']),e['item'],start+round(e['offset_seconds']*1e9))
  r.update(frozen_offset_seconds=e['offset_seconds'],routing_uniform=e['routing_uniform'],replica_uniform=e['replica_uniform']);records.append(r)
 # Publish complete planned cohort; collector writes all record mutations under runtime.lock.
 monitor.phase=label;monitor.set_active(dict(label=label,planned_start_time_ns=start,planned_end_time_ns=end,planned_duration_seconds=duration,request_records=records))
 for r,e in zip(records,events):
  wait_until(r['scheduled_time_ns']);monitor.check()
  with runtime.lock:
   if runtime.errors:raise RuntimeError(runtime.errors)
   pending=[p for p in runtime.pending.values() if p['status'] in ('queued','submitted')]
   oldest=max(((time.perf_counter_ns()-p['enqueued_time_ns'])/1e9 for p in pending),default=0)
  if len(pending)>300 or oldest>120:raise RuntimeError('FROZEN_BACKLOG_SAFETY_ABORT')
  runtime.enqueue(r,e['item']['sample'])
 wait_until(end);monitor.check();arrival_end=time.perf_counter_ns();monitor.phase='drain' if label=='measurement' else 'warmup_drain'
 runtime.drain(records,60.);drain=time.perf_counter_ns()
 with runtime.lock:annotate_records(records)
 monitor.flush_active()
 window=dict(start_time_ns=start,start_unix_ns=unix,end_time_ns=end,planned_end_time_ns=end,duration_seconds=duration,planned_duration_seconds=duration,
  arrival_loop_end_time_ns=arrival_end,drain_end_time_ns=drain,drain_seconds=(drain-end)/1e9,stop=None,cohort='All frozen matched offered requests including full bounded drain')
 with runtime.lock:per,system,models=summarize(records,config,window)
 tmp=dict(request_records=records,measurement_window=window,status='completed' if all(r['status']=='completed' for r in records) else 'completed_with_failures',errors=runtime.errors,drain_timeout_seconds=60.,worker_identities={i:{'pid':w.process.pid} for i,w in runtime.workers.items()})
 aliases={i:SimpleNamespace(service_time_ms=plan['models'][m]['service_time_ms']) for i,m in mapping.items()};ev=reconstruct(tmp,aliases,DEFAULT_THRESHOLDS)
 monitor.set_active(None)
 return dict(window=window,request_records=records,per_task_metrics=per,system_metrics=system,stability=classify(ev,DEFAULT_THRESHOLDS),stability_evidence=ev,trajectory=slot['trajectories'][label])

def cool(slot,f,monitor=None):
 from module26.gpu import thermal
 resident=monitor is not None;p=f['thermal_protocol'];limit=p['resident_cool_timeout_seconds'] if resident else p['max_wait_seconds'];maxc=p['resident_premeasurement_temperature_max_c'] if resident else p['prelaunch_temperature_max_c'];required=p['resident_consecutive_samples'] if resident else p['consecutive_idle_samples_required'];begin=time.monotonic();samples=[];consecutive=0
 while time.monotonic()-begin<limit:
  v=monitor.sample() if resident else snapshot()
  if not resident:v.update(thermal())
  samples.append(v)
  if not resident and (v['processes'] or v['used_mib']>128):raise RuntimeError('PRELAUNCH_GPU_ISOLATION_FAILURE')
  consecutive=consecutive+1 if v['temperature_gpu_c']<=maxc else 0
  if consecutive>=required:return dict(passed=True,samples=samples,elapsed_seconds=time.monotonic()-begin)
  if monitor:monitor.check()
  time.sleep(2.)
 raise RuntimeError('THERMAL_LAUNCH_FAILURE')

def safe_idle(raw,key,monitor,f):
 raw[key]=monitor.idle_window(key,2.);monitor.check()
 if f['safe_factor']*np.mean([v['owned_used_mib'] for v in raw[key]])>raw['slot']['safe_memory_budget_mib']:raise RuntimeError('SAFE_MEMORY_BUDGET_EXCEEDED')

def run(slot,f):
 path=OUT/'raw_runs'/(slot['run_id']+'.json');assert not path.exists(),'No retries or overwritten raw'
 save('run_attempts/'+slot['run_id']+'.json',dict(slot=slot,started_unix_ns=time.time_ns(),pid=os.getpid()))
 pre=cool(slot,f);save('thermal_prelaunch/'+slot['run_id']+'.json',pre);baseline=pre['samples'][-1];plan=f['plan']
 assert all(baseline[k]==f['GPU'][k] for k in ['uuid','name','driver','total_mib'])
 for k,v in plan['environment_overrides'].items():os.environ[k]=v
 mapping={f'{m}#{i}':m for m,n in slot['deployment'].items() for i in range(n)}
 routes={t:{i:q/slot['deployment'][m] for m,q in row.items() for i,mm in mapping.items() if mm==m} for t,row in slot['routing'].items()}
 config=PoissonConfig(deployment={i:1 for i in mapping},workload_rps=slot['lambda_rps'],routing_rps=routes,arrival_process='poisson',random_seed=slot['seed'],
  warmup_seconds=30.,measurement_seconds=600.,startup_timeout_seconds=300.,drain_timeout_seconds=60.,shutdown_timeout_seconds=15.,routing_source='Module30 frozen matched Poisson trajectory and independent route uniforms')
 config.validate({i:plan['models'][m]['supported_tasks'] for i,m in mapping.items()})
 runtime=AuditedRuntime(config,{i:plan['registry'][m] for i,m in mapping.items()});monitor=SafeMonitor(runtime,slot,slot['safe_memory_budget_mib'])
 raw=dict(slot=slot,status='STARTING',errors=[],phases={},baseline_GPU=baseline,replica_to_family=mapping,worker_identities={},transport_config=config.as_dict(),frozen_manifest=ref(OUT/'frozen_gpu_manifest.json'));begin=time.perf_counter();monitor.start()
 try:
  runtime.start();monitor.check();raw['worker_identities']={i:dict(family=m,pid=runtime.workers[i].process.pid,READY=runtime.ready[i]) for i,m in mapping.items()}
  assert len({r['pid'] for r in raw['worker_identities'].values()})==len(mapping)
  raw['path_probe_records']=prime(runtime,config,plan,mapping,slot);monitor.check()
  raw['warmup']=phase(runtime,config,slot,plan,mapping,'warmup',monitor);assert all(r['status']=='completed' for r in raw['warmup']['request_records'])
  monitor.phase='postwarm_thermal_control';raw['resident_thermal_control']=cool(slot,f,monitor);safe_idle(raw,'postwarm_idle',monitor,f)
  print('MEASUREMENT_START',slot['run_id'],600,flush=True);raw['phases']['measurement']=phase(runtime,config,slot,plan,mapping,'measurement',monitor)
  safe_idle(raw,'postmeasurement_idle',monitor,f)
  if not all(r['status']=='completed' for r in raw['phases']['measurement']['request_records']):raise RuntimeError('REQUEST_ERROR_OR_INCOMPLETE_COHORT')
  raw['status']='COMPLETED'
 except (Exception,KeyboardInterrupt) as exc:
  raw['status']='FAILED';raw['errors'].append(repr(exc));raw['exception_traceback']=traceback.format_exc()
  with monitor.sample_lock:
   if monitor.active_phase:raw['aborted_phase']=monitor.active_phase;raw['aborted_phase']['abort_time_ns']=time.perf_counter_ns()
 finally:
  # Allow ordered RESULT/BYE/EXIT collection before marking any surviving cohort incomplete.
  try:raw['shutdown']=runtime.close()
  except Exception as exc:raw['errors'].append('shutdown:'+repr(exc))
  if raw.get('aborted_phase'):
   with runtime.lock:
    for r in raw['aborted_phase']['request_records']:
     if r['status'] in ('planned','queued','submitted'):r.update(status='incomplete',error='Entire frozen cohort retained after abort')
    annotate_records(raw['aborted_phase']['request_records'])
   try:monitor.flush_active()
   except Exception as exc:raw['errors'].append('final_journal:'+repr(exc))
  try:monitor.close()
  except Exception as exc:raw['errors'].append('monitor_close:'+repr(exc))
  raw['errors']+=runtime.errors+monitor.errors
  raw['telemetry']=dict(samples=monitor.samples,errors=monitor.errors,error_details=monitor.error_details,interval_seconds=.5,
   collector_exited=not monitor.thread.is_alive(),peak_scope='NVML sampled owned high-water, not instantaneous allocator maximum')
  raw['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process])
  raw['valid_session']=bool(raw['status']=='COMPLETED' and raw['gpu_release']['verified'] and not raw['errors'] and raw['telemetry']['collector_exited'] and len(raw.get('shutdown',{}))==len(mapping) and all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in raw['shutdown'].values()))
  raw['elapsed_seconds']=time.perf_counter()-begin;save('raw_runs/'+path.name,raw)
 return raw

def freeze():
 harness=read(OUT/'cpu_harness_acceptance.json');preflight=read(OUT/'cpu_preflight.json')
 assert harness['passed'] and preflight['passed']
 stress=read(OUT/'telemetry_stress_result.json');life=read(OUT/'cpu_lifecycle_audit.json');race=read(OUT/'legacy_race_reproduction.json')
 assert stress['passed'] and stress['elapsed_seconds']>1997.196239099 and life['passed']
 assert race['reproduced_error']
 f=read(OUT/'cpu_screening.json');f['source_refs']+=preflight['quality_evidence'];gpu=snapshot();assert not gpu['processes'] and gpu['used_mib']<=128
 f.update(status='FROZEN_BEFORE_GPU',frozen_unix_ns=time.time_ns(),GPU=gpu,module='Module30 GPU-Measured G/S Positive-Split Feasibility Screening',stability_thresholds=DEFAULT_THRESHOLDS,
  infrastructure_gate='PASSED',infrastructure_evidence=[ref(OUT/n) for n in ['telemetry_stress_result.json','cpu_lifecycle_audit.json','legacy_race_reproduction.json','cpu_harness_acceptance.json','cpu_preflight.json']],
  implementation=[ref(p) for p in sorted((ROOT/'module30').glob('*.py'))],
  reused_implementation=[ref(ROOT/p) for p in ['joint_deploy/measurement/deployment_runtime.py','joint_deploy/measurement/metrics.py','performance_model/local_calibration/stability_v2.py','module20/optimizer.py','module22/gpu.py','module11b/measure.py','module11b/telemetry.py','replica_scaling/telemetry.py']]+[ref(v['script']) for v in f['plan']['registry'].values()],
  failure_policy=dict(technical_error='STOP_ENTIRE_CAMPAIGN_NO_RETRY',unstable='retain NULL U, continue remaining frozen arms to diagnose same candidate; no expansion',backlog_safety=dict(max_pending=300,max_oldest_seconds=120),prelaunch_failure='STOP_NO_REPLACEMENT'),
  positive_gate=dict(delta_U_min=.01,all_three_arms_legal_stable=True,min_target_G_completed=20,min_background_task_completed=72,
   postwarm_temperature_difference_max_c=5.,measurement_mean_temperature_difference_max_c=5.,measurement_mean_sm_clock_relative_difference_max=.15),
  optimizer_modified=False,independent_replication_automatic=False,next_module_automatic=False)
 f['plan']['GPU']=gpu
 save('telemetry_fix_gate.json',dict(passed=True,stress=stress,lifecycle=life,race=race,historical_code_preserved=True,Module29_gate=ref(ROOT/'results/module29_dynamic_replication/module29_gate.json')))
 (OUT/'telemetry_fix_audit.md').write_text('''# 遥测修复审计

Module29在1997.196239秒记录dictionary changed size during iteration；原始无堆栈，不能唯一识别具体触发语句。历史DurableMonitor对sender/collector正在修改的live request执行未加锁deepcopy；CPU普通字典并发实验重现同型异常，见legacy_race_reproduction.json。

新SafeMonitor在runtime.lock下深拷贝完整请求快照，退出锁后序列化/写盘；sample_lock串行后台与同步冷却采样、CPU状态、日志与phase切换，完成样本最后发布。先冻结完整planned cohort再启用日志；日志fsync成功后才提交persisted键，原子raw/progress使用唯一临时文件、fsync、replace与目录fsync。遥测/写盘异常保存堆栈并进入硬失败；不允许技术失败或不完整窗口算合法U。全请求期末与异常后均flush并核对journal；未到达槽也保留为incomplete。

新AuditedRuntime复用原native transport与推理，额外核验sender/collector/reader退出；先清洁关闭、按序接收RESULT/BYE/EXIT再标未完成，核验owned PID释放。CPU fixture验证200个请求无丢失且清洁退出。首次fixture协议前缀写错导致启动等待，修正为原READY|/RESULT|/BYE|协议后通过；初始日志保留，未启动GPU。最初默认Python缺scipy，改用项目已有gsdeploy环境；未安装依赖、失败日志保留。

2100秒CPU压力超过原故障时刻：两个持续请求字典更新线程、后台采样及同步采样并发；逐版本快照一致、JSONL完整、终态无重复/遗漏、线程退出。sensor异常、fsync失败与不退出collector注入均可见；写盘失败不提交persisted状态。数值结果见telemetry_stress_result.json。CPU soak使用假传感器，不冒充真实GPU画像；正式每次GPU同样检查遥测与journal完整性。

历史Module14F–29代码、错误、raw、Gate和权重均不覆盖；修复为module30/telemetry.py替换实现，旧冻结监测器保留为错误证据，当前Module不再调用其无锁请求拷贝。正式优化器不改。仅上述检查通过后才冻结当前NVML设备身份并开始GPU测量。
''')
 save('frozen_gpu_manifest.json',f);return f

def main():
 lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'campaign_completed.json').exists()
 save('campaign_process.json',dict(pid=os.getpid(),started_unix_ns=time.time_ns(),status='WAITING_INFRASTRUCTURE_GATE'))
 deadline=time.monotonic()+2400
 while not (OUT/'telemetry_stress_result.json').exists():
  if time.monotonic()>deadline:raise RuntimeError('STRESS_GATE_NOT_PRODUCED_NO_GPU')
  time.sleep(2)
 f=freeze();save('campaign_process.json',dict(pid=os.getpid(),status='GPU_CAMPAIGN_RUNNING'))
 attempted=[];stop=None
 for slot in f['slots']:
  for r in f['implementation']+f['reused_implementation']+f['source_refs']:
   assert ref(r['path'])['sha256']==r['sha256'],r['path']
  attempted.append(slot['run_id']);print('START',slot['run_id'],flush=True)
  try:raw=run(slot,f)
  except Exception as exc:
   save('prelaunch_failures/'+slot['run_id']+'.json',dict(slot=slot,error=repr(exc),traceback=traceback.format_exc()));stop=dict(run_id=slot['run_id'],reason='PRELAUNCH_OR_PERSISTENCE_ERROR',error=repr(exc));break
  from .report import observation,interim
  o=observation(raw);print('FINISH',slot['run_id'],'valid',raw['valid_session'],'stable',o.get('stability'),'U',o.get('U'),'errors',raw['errors'],flush=True)
  interim()
  if not raw['valid_session'] or not raw['gpu_release']['verified']:
   stop=dict(run_id=slot['run_id'],reason='TECHNICAL_FAILURE_NO_RETRY',errors=raw['errors']);break
 save('campaign_completed.json',dict(attempted=attempted,stop=stop,skipped=[s['run_id'] for s in f['slots'] if s['run_id'] not in attempted],maximum_formal_runs=6,no_retries=True))
 from .report import finalize
 finalize()
 print('MODULE30_COMPLETE',read(OUT/'module30_gate.json')['gate'],flush=True)
if __name__=='__main__':main()
