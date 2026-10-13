"""Exactly nine frozen point validations; unchanged workers, routing and labels."""
import copy,os,time
from .common import *
from module11b.measure import transport,prime,phase
from module11b.telemetry import CostMonitor
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from replica_scaling.telemetry import snapshot,release

def freeze():
 assert not (OUT/'gpu_validation_freeze.json').exists(),'Reuse the frozen plan.'
 f=read(OUT/'search_protocol_freeze.json');verify(f['implementation']+[f['catalog']]);w=read(OUT/'theoretical_witness.json')['witness'];d=read(OUT/'existing_empirical_catalog.json')
 base=read(ROOT/'results/module11_replica_and_cost/measurement_plan.json');new=read(ROOT/'results/module14_heavy_replica/measurement_plan.json');plan=copy.deepcopy(base);plan['models']=d['models'];plan['registry']={**base['registry'],**new['registry']};plan['warmup_seconds']=f['GPU_protocol']['warmup_seconds'];slots=[]
 for p in w['points']:
  fit=p['full_pool_result']['winner'];routing={k:{path.split('::')[1]:v for path,v in fit['routing'].items() if path.split('::')[0]==k and v>0} for k in p['rates']}
  for rep,seed in enumerate(f['GPU_protocol']['seeds'],1):
   assert all(m in plan['registry'] for m in fit['deployment'])
   slots.append({'run_id':p['stage'].lower()+'_rep'+str(rep),'stage':p['stage'],'seed':seed,'repetition':rep,'target_task':w['target_task'],'Generalist':w['Generalist'],'Specialist':w['Specialist'],'deployment':fit['deployment'],'lambda_rps':p['rates'],'routing':routing,'measurement_seconds':f['GPU_protocol']['measurement_seconds'],'safe_memory_budget_mib':w['budget_mib'],'CPU_safe_memory_mib':fit['safe_memory_mib'],'CPU_target_G_fraction':fit['routing'].get(w['target_task']+'::'+w['Generalist'],0)/p['target_rps'],'CPU_max_rho':fit['max_rho']})
 save('gpu_validation_freeze.json',{'at_utc':now(),'witness':ref(OUT/'theoretical_witness.json'),'search_policy':ref(OUT/'search_protocol_freeze.json'),'catalog':f['catalog'],'plan':plan,'slots':slots,'maximum_sessions':9,'no_outcome_retries':True,'no_point_or_route_movement':True,'new_capacity_profiling':0,'latency_SLO':None,'memory_validation':'Each repetition: frozen factor times owned postwarm idle mean AND owned postmeasurement idle mean <= unchanged budget; sampled physical owned peak <= same budget, with peak kept separate from idle*safety cost. Never scale or impute memory.','implementation':[ref(ROOT/'module15/gpu.py'),ref(ROOT/'module15/solver.py'),ref(ROOT/'module15/search.py'),ref(ROOT/'module15/baseline.py'),ref(ROOT/'module11b/measure.py'),ref(ROOT/'joint_deploy/measurement/deployment_runtime.py'),ref(ROOT/'performance_model/local_calibration/stability_v2.py')]})
 print('GPU_PLAN_FROZEN',[(s['run_id'],s['lambda_rps']) for s in slots],flush=True)

def run(slot,plan):
 target=OUT/'raw_runs'/(slot['run_id']+'.json');assert not target.exists(),'No outcome retry or overwritten observation.'
 for k,v in plan['environment_overrides'].items():os.environ[k]=v
 baseline=snapshot();assert not baseline['processes'] and baseline['used_mib']<=128
 assert baseline['uuid']==plan['GPU']['uuid']
 mapping,config,registry=transport(slot,plan);runtime=DeploymentRuntime(config,registry);monitor=CostMonitor(runtime,plan['telemetry_interval_seconds'],plan['GPU']['total_mib']);begin=time.perf_counter_ns();monitor.start()
 result={'slot':slot,'status':'STARTING','errors':[],'phases':{},'baseline_GPU':baseline,'worker_identities':{},'replica_to_family':mapping,'transport_config':config.as_dict(),'frozen_validation_plan':ref(OUT/'gpu_validation_freeze.json')}
 try:
  runtime.start();result['worker_identities']={i:{'family':m,'pid':runtime.workers[i].process.pid,'READY':runtime.ready[i]} for i,m in mapping.items()}
  assert len({v['pid'] for v in result['worker_identities'].values()})==sum(slot['deployment'].values())
  result['path_probe_records']=prime(runtime,config,plan,mapping,slot)
  result['warmup']=phase(runtime,config,plan,mapping,'warmup',config.warmup_seconds,None,monitor)
  result['postwarm_idle']=monitor.idle_window('postwarm_idle',2.)
  result['phases']['measurement']=phase(runtime,config,plan,mapping,'measurement',slot['measurement_seconds'],None,monitor)
  result['postmeasurement_idle']=monitor.idle_window('postmeasurement_idle',2.);monitor.check();result['status']='COMPLETED'
 except (Exception,KeyboardInterrupt) as exc:result['status']='FAILED';result['errors'].append(repr(exc))
 finally:
  try:result['shutdown']=runtime.close()
  except Exception as exc:result['errors'].append('shutdown:'+repr(exc))
  monitor.close();result['errors']+=runtime.errors+monitor.errors;result['telemetry']={'samples':monitor.samples,'errors':monitor.errors};result['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process]);result['valid_session']=bool(result['status']=='COMPLETED' and result['gpu_release']['verified'] and not result['errors'] and all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in result.get('shutdown',{}).values()));result['elapsed_seconds']=(time.perf_counter_ns()-begin)/1e9;save('raw_runs/'+target.name,result)
 return result

def main():
 f=read(OUT/'gpu_validation_freeze.json');verify(f['implementation']+[f['witness'],f['search_policy'],f['catalog']]);slots=f['slots'];assert len(slots)==9
 for s in slots:
  path=OUT/'raw_runs'/(s['run_id']+'.json')
  if path.exists():r=read(path)
  else:
   print('START',s['run_id'],s['lambda_rps'],flush=True);r=run(s,f['plan'])
  assert r['gpu_release']['verified'],'GPU release failed; stop additional work.'
  ph=r['phases'].get('measurement',{})
  print('FINISH',s['run_id'],r['valid_session'],ph.get('stability',{}).get('stability_state_v2'),ph.get('metrics',{}).get('mean_e2e_ms'),r['errors'],flush=True)
 print('FROZEN_NINE_SESSIONS_COMPLETE',flush=True)
if __name__=='__main__':main()
