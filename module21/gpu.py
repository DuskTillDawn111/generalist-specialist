"""Conditional frozen Poisson campaign; native workers/runtime left untouched."""
from .common import *
from .trajectory import choose
import os,time
from dataclasses import replace
from types import SimpleNamespace
from joint_deploy.measurement.schema import MeasurementConfig
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from joint_deploy.measurement.run_measurement import new_record,wait_until
from joint_deploy.measurement.metrics import annotate_records,summarize
from module11b.measure import prime
from module16.gpu import ProcessMonitor
from replica_scaling.telemetry import snapshot,release
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS

class PoissonConfig(MeasurementConfig):
 def validate(self,capabilities):
  # Reuse frozen structural checks; local harness supplies actual Poisson arrivals.
  assert self.arrival_process=='poisson'
  base=MeasurementConfig(**{**self.as_dict(),'arrival_process':'deterministic_periodic'});base.validate(capabilities)

def phase(runtime,config,slot,plan,mapping,label,monitor):
 trajectory=read(slot['trajectories'][label]['path']);assert ref(slot['trajectories'][label]['path'])['sha256']==slot['trajectories'][label]['sha256'];events=trajectory['events'];duration=trajectory['duration'];start=time.perf_counter_ns();unix=time.time_ns();end=start+round(duration*1e9);records=[];monitor.phase=label
 for i,event in enumerate(events):
  scheduled=start+round(event['offset_seconds']*1e9);worker=choose(event,slot['routing'],slot['deployment']);r=new_record(label,i,event['task'],worker,event['item'],scheduled);r.update(frozen_offset_seconds=event['offset_seconds'],routing_uniform=event['routing_uniform'],replica_uniform=event['replica_uniform']);records.append(r)
  wait_until(scheduled);runtime.enqueue(r,event['item']['sample'])
 wait_until(end);arrival_end=time.perf_counter_ns();monitor.phase='drain' if label=='measurement' else 'warmup_drain';runtime.drain(records,60.);drain=time.perf_counter_ns();annotate_records(records)
 window={'start_time_ns':start,'start_unix_ns':unix,'end_time_ns':end,'planned_end_time_ns':end,'duration_seconds':duration,'planned_duration_seconds':duration,'arrival_loop_end_time_ns':arrival_end,'drain_end_time_ns':drain,'drain_seconds':(drain-end)/1e9,'stop':None,'cohort':'all frozen Poisson requests scheduled in [start,end), full paired offered stream'}
 per_task,system,models=summarize(records,config,window);temp={'request_records':records,'measurement_window':window,'status':'completed' if all(r['status']=='completed' for r in records) else 'completed_with_failures','errors':runtime.errors,'drain_timeout_seconds':60.,'worker_identities':{i:{'pid':w.process.pid} for i,w in runtime.workers.items()}}
 aliases={i:SimpleNamespace(service_time_ms=plan['models'][m]['service_time_ms']) for i,m in mapping.items()};ev=reconstruct(temp,aliases,DEFAULT_THRESHOLDS)
 return {'window':window,'request_records':records,'per_task_metrics':per_task,'system_metrics':system,'stability':classify(ev,DEFAULT_THRESHOLDS),'stability_evidence':ev,'trajectory':slot['trajectories'][label]}

def run(slot,plan):
 path=OUT/'raw_runs'/(slot['run_id']+'.json');assert not path.exists(),'No observation overwrite or retry'
 for k,v in plan['environment_overrides'].items():os.environ[k]=v
 baseline=snapshot();assert not baseline['processes'] and baseline['used_mib']<=128,'GPU isolation failed';assert baseline['uuid']==plan['GPU']['uuid']
 mapping={f'{m}#{i}':m for m,n in slot['deployment'].items() for i in range(n)};routes={t:{i:q/slot['deployment'][m] for m,q in row.items() for i,mm in mapping.items() if mm==m} for t,row in slot['routing'].items()}
 config=PoissonConfig(deployment={i:1 for i in mapping},workload_rps=slot['lambda_rps'],routing_rps=routes,arrival_process='poisson',random_seed=slot['seed'],warmup_seconds=10.,measurement_seconds=120.,startup_timeout_seconds=300.,drain_timeout_seconds=60.,shutdown_timeout_seconds=15.,routing_source='module21_frozen_independent_uniform_Poisson_thinning');config.validate({i:plan['models'][m]['supported_tasks'] for i,m in mapping.items()})
 runtime=DeploymentRuntime(config,{i:plan['registry'][m] for i,m in mapping.items()});monitor=ProcessMonitor(runtime,.5,BUDGET);begin=time.perf_counter();monitor.start();result={'slot':slot,'status':'STARTING','errors':[],'phases':{},'baseline_GPU':baseline,'worker_identities':{},'replica_to_family':mapping,'transport_config':config.as_dict(),'frozen_manifest':ref(OUT/'frozen_manifest.json')}
 try:
  runtime.start();result['worker_identities']={i:{'family':m,'pid':runtime.workers[i].process.pid,'READY':runtime.ready[i]} for i,m in mapping.items()};assert len({r['pid'] for r in result['worker_identities'].values()})==len(mapping)
  result['path_probe_records']=prime(runtime,config,plan,mapping,slot);result['warmup']=phase(runtime,config,slot,plan,mapping,'warmup',monitor)
  if any(r['status']!='completed' for r in result['warmup']['request_records']):raise RuntimeError('Warmup cohort incomplete; no formal measurement')
  result['postwarm_idle']=monitor.idle_window('postwarm_idle',2.);result['phases']['measurement']=phase(runtime,config,slot,plan,mapping,'measurement',monitor);result['postmeasurement_idle']=monitor.idle_window('postmeasurement_idle',2.);monitor.check();result['status']='COMPLETED'
 except (Exception,KeyboardInterrupt) as exc:result['status']='FAILED';result['errors'].append(repr(exc))
 finally:
  try:result['shutdown']=runtime.close()
  except Exception as exc:result['errors'].append('shutdown: '+repr(exc))
  monitor.close();result['errors']+=runtime.errors+monitor.errors;result['telemetry']={'samples':monitor.samples,'errors':monitor.errors};result['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process]);result['valid_session']=bool(result['status']=='COMPLETED' and result['gpu_release']['verified'] and not result['errors'] and all(r['exit_code']==0 and not r['forced'] and r['bye_received'] for r in result.get('shutdown',{}).values()));result['elapsed_seconds']=time.perf_counter()-begin;save('raw_runs/'+path.name,result)
 return result

def main():
 f=read(OUT/'frozen_manifest.json');assert f['status']=='FROZEN_BEFORE_GPU';assert read(f['CPU']['path'])['Stage_A_passed'];assert len(f['slots'])<=18
 for r in f['implementation']+f['reused_implementation']+[f['CPU'],f['input_protocol'],f['catalog'],f['utility_definition'],f['utility_references'],f['sample_pool']]:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 from .report import analyze,write_outputs,stage_result
 results=[];halt=None
 for stage in RATES:
  stage_rows=[];unstable={'proposed':0,'control':0}
  for slot in [s for s in f['slots'] if s['stage']==stage]:
   print('START',slot['run_id'],flush=True);raw=read(OUT/'raw_runs'/(slot['run_id']+'.json')) if (OUT/'raw_runs'/(slot['run_id']+'.json')).exists() else run(slot,f['plan']);row,_,_=analyze(raw,f);results.append(raw);stage_rows.append(row)
   print('FINISH',slot['run_id'],'U',row['U'],'system',row['system_state'],'worker_states',row['worker_states'],'errors',raw['errors'],flush=True);write_outputs(results,f)
   if row['hard_unstable']:unstable[slot['arm']]+=1
   if raw['errors'] or not raw['valid_session'] or not row['memory_passed'] or not raw['gpu_release']['verified']:
    halt={'stage':stage,'gate':'THREE_PHASE_GPU_UNSTABLE','reason':'Runtime/memory/release failure','run':slot['run_id']};break
   if max(unstable.values())>=2:
    halt={'stage':stage,'gate':'THREE_PHASE_GPU_UNSTABLE','reason':'Repeated system or worker instability','unstable_counts':unstable};break
  if not halt:
   verdict=stage_result(stage,stage_rows)
   if not verdict['passed']:halt={'stage':stage,'gate':'THREE_PHASE_GPU_UNSTABLE' if any(r['hard_unstable'] for r in stage_rows) else 'THREE_PHASE_WITHOUT_UTILITY_GAIN','reason':'Frozen stage paired utility criterion failed','stage_result':verdict}
  save('campaign_status.json',{'at_utc':now(),'completed_runs':[r['slot']['run_id'] for r in results],'halt':halt,'skipped_runs':[s['run_id'] for s in f['slots'] if s['run_id'] not in [r['slot']['run_id'] for r in results]],'no_retry_or_point_movement':True})
  if halt:break
 write_outputs(results,f);save('module21_gate.json',{'at_utc':now(),'gate':halt['gate'] if halt else 'POSITIVE_UTILITY_THREE_PHASE_REAL_VALIDATED','Stage_A_passed':True,'GPU_started':True,'formal_GPU_runs':sum('measurement' in r['phases'] for r in results),'halt':halt,'no_Module22':True,'CPU_simulation_not_GPU_evidence':True})
 print('CAMPAIGN_STOP',halt,flush=True)
if __name__=='__main__':main()
