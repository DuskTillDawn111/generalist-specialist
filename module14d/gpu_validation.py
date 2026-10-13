"""Bounded 14E validation uses frozen Module9 routes, unchanged periodic runtime."""
import copy,os,time,math,statistics
from dataclasses import replace
from .common import ROOT,OLD,OUT as DOUT,read,ref,now,verify
from . import common
from module14c.common import E,H
from module11b.measure import transport,prime,phase
from module11b.telemetry import CostMonitor
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from replica_scaling.telemetry import snapshot,release
from module11b.common import factor,stat
OUT=ROOT/'results/module14e_heavy_gpu_validation'

def save(name,data):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(__import__('json').dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def freeze():
 gate=read(DOUT/'module14d_gate.json');assert gate['automatic_GPU_validation_required']
 if (OUT/'validation_freeze.json').exists():return read(OUT/'validation_freeze.json')
 OUT.mkdir(exist_ok=True);rows=read(DOUT/'structural_replay_rows.json');cmp=read(DOUT/'structural_transition_comparison.json')['comparisons'];f=read(OLD/'screen_protocol_freeze.json');candidates=[]
 for p in cmp:
  if not p['comparable_feasible']:continue
  pair=[r for r in rows if r['workload']==p['workload'] and r['budget']==p['budget']]
  if not all(r['frozen_Module9_solver']['converged'] and r['frozen_Module9_solver'].get('operational_certificate',{}).get('passed') for r in pair):continue
  work=next(w for w in f['workloads'] if w['id']==p['workload']);candidates.append((p,pair,work))
 # One persistent reduction and one highest demand comparable witness.
 low=[v for v in candidates if v[0].get('old_NS_reduction_persists') and v[2]['regime']=='low_load']
 selected=[]
 if low:selected.append(min(low,key=lambda v:(v[0]['delta_NS'],v[0]['budget'],v[0]['workload'])))
 high=max(candidates,key=lambda v:(math.fsum(v[2]['rates'].values()),v[0]['budget']))
 if not selected or high[0]!=selected[0][0]:selected.append(high)
 assert selected
 plan=copy.deepcopy(read(DOUT/'measurement_plan.json'));base=read(ROOT/'results/module11_replica_and_cost/measurement_plan.json');h=read(DOUT/'heavy_empirical_replica_table.json');e=read(OLD/'regime_e.json');plan['models']={**e['models'],**h['models']};plan['registry']={**base['registry'],**plan['registry']}
 # New unselected Heavy checkpoints stay in pool; states using them are not
 # secretly instantiated without their frozen launcher. No outcome substitution.
 for _,rs,_ in selected:
  for r in rs:
   for m in r['replicas']:assert m in plan['registry'],m
 slots=[];contexts=[]
 for index,(p,rs,work) in enumerate(selected):
  contexts.append({'comparison':p,'workload':work,'CPU_rows':rs})
  for rep,seed in enumerate([7101,7202,7303],1):
   order=[E,H] if rep%2 else [H,E]
   for regime in order:
    row=next(r for r in rs if r['Specialist_regime']==regime);route=row['frozen_Module9_solver']['routing'];routing={k:{path.split('::')[1]:v for path,v in route.items() if path.split('::')[0]==k and v>0} for k in work['rates'] if work['rates'][k]>0};rates={k:v for k,v in work['rates'].items() if v>0}
    slots.append({'run_id':f'context{index}_{"E" if regime==E else "H"}_rep{rep}','context':work['id']+'/'+p['budget'],'Specialist_regime':regime,'deployment':row['replicas'],'lambda_rps':rates,'routing':routing,'seed':seed,'repetition':rep,'measurement_seconds':30.,'primary_task':max(rates,key=rates.get),'safe_memory_budget_mib':row['memory_budget_mib'],'CPU_J_seconds':row['J_star_seconds'],'CPU_NG':row['N_G'],'CPU_NS':row['N_S'],'CPU_G_share':row['Generalist_routed_fraction']})
 frozen={'at_utc':now(),'scope':'Module14E bounded E-vs-H real GPU; no model/context/quality/budget/J/rho changes','selection_rule':'One low-load persistent NS reduction with largest count reduction, then maximum demand comparable context; Module9 accepted routes only. All selection before 14E GPU outcomes.','contexts':contexts,'plan':plan,'slots':slots,'maximum_sessions':len(slots),'protocol':{'warmup_seconds':5,'measurement_seconds':30,'repetitions':3,'seeds':[7101,7202,7303],'capacity_method':'periodic original workload E2E stability validation; never substituted for replica capacity','no_retries_after_outcomes':True},'stop_rule':'Systematic contradiction = a representative CPU-feasible context shows H unstable or runtime-failed in all three repetitions while E is steady in all three. Observed latency/service/memory signs are reported separately, without interpreting them as GPU optimality. Isolated invalid session means inconclusive, never stable success.','source_14D_gate':ref(DOUT/'module14d_gate.json'),'implementation':[ref(ROOT/'module14d/gpu_validation.py'),ref(ROOT/'module11b/measure.py'),ref(ROOT/'joint_deploy/measurement/deployment_runtime.py')],'historical_inputs':ref(DOUT/'protected_before.json'),'no_outcome_driven_hotspot_selection':True}
 save('validation_freeze.json',frozen);return frozen

def run(slot,plan):
 target=OUT/'raw_runs'/(slot['run_id']+'.json')
 if target.exists():return read(target)
 for k,v in plan['environment_overrides'].items():os.environ[k]=v
 baseline=snapshot();assert not baseline['processes'] and baseline['used_mib']<=128
 mapping,config,registry=transport(slot,plan);runtime=DeploymentRuntime(config,registry);monitor=CostMonitor(runtime,plan['telemetry_interval_seconds'],plan['GPU']['total_mib']);monitor.start();begin=time.perf_counter_ns()
 result={'slot':slot,'status':'STARTING','errors':[],'phases':{},'baseline_GPU':baseline,'worker_identities':{},'replica_to_family':mapping,'transport_config':config.as_dict()}
 try:
  runtime.start();result['worker_identities']={i:{'family':m,'pid':runtime.workers[i].process.pid,'READY':runtime.ready[i]} for i,m in mapping.items()};result['path_probe_records']=prime(runtime,config,plan,mapping,slot)
  result['warmup']=phase(runtime,config,plan,mapping,'warmup',5.,None,monitor)
  result['postwarm_idle']=monitor.idle_window('postwarm_idle',2.)
  result['phases']['measurement']=phase(runtime,config,plan,mapping,'measurement',30.,None,monitor);monitor.check();result['status']='COMPLETED'
 except (Exception,KeyboardInterrupt) as exc:result['status']='FAILED';result['errors'].append(repr(exc))
 finally:
  try:result['shutdown']=runtime.close()
  except Exception as exc:result['errors'].append('shutdown:'+repr(exc))
  monitor.close();result['errors']+=runtime.errors+monitor.errors;result['telemetry']={'samples':monitor.samples,'errors':monitor.errors};result['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process]);result['valid_session']=bool(result['status']=='COMPLETED' and result['gpu_release']['verified'] and not result['errors'] and all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in result.get('shutdown',{}).values()));result['elapsed_seconds']=(time.perf_counter_ns()-begin)/1e9;save('raw_runs/'+target.name,result)
 return result

def main():
 frozen=freeze();verify(frozen['implementation']);rows=[]
 for slot in frozen['slots']:
  print('VALIDATE_START',slot['run_id'],slot['context'],flush=True);raw=run(slot,frozen['plan']);assert raw['gpu_release']['verified'];ph=raw['phases'].get('measurement',{});idle=raw.get('postwarm_idle',[]);mem=statistics.mean(s['owned_used_mib'] for s in idle) if idle else None
  row={**slot,'valid_session':raw['valid_session'],'errors':raw['errors'],'stability':ph.get('stability',{}).get('stability_state_v2'),'mean_E2E_ms':ph.get('metrics',{}).get('mean_e2e_ms'),'p95_E2E_ms':ph.get('metrics',{}).get('p95_e2e_ms'),'throughput_rps':ph.get('metrics',{}).get('completed_throughput_rps'),'completion_ratio':ph.get('metrics',{}).get('completion_ratio'),'per_task_metrics':ph.get('per_task_metrics',{}),'system_metrics':ph.get('system_metrics',{}),'service_time_ms':stat(r.get('worker_latency_ms') for r in ph.get('request_records',[]) if r['status']=='completed'),'safe_idle_memory_mib':factor()*mem if mem else None,'peak_GPU_memory_mib':max((s['owned_used_mib'] for s in raw['telemetry']['samples']),default=None),'raw':ref(OUT/'raw_runs'/(slot['run_id']+'.json'))}
  rows.append(row);save('validation_rows.json',rows);print('VALIDATE_FINISH',slot['run_id'],row['valid_session'],row['stability'],row['mean_E2E_ms'],flush=True)
 comparisons=[]
 for con in frozen['contexts']:
  cid=con['workload']['id']+'/'+con['comparison']['budget'];er=[r for r in rows if r['context']==cid and r['Specialist_regime']==E];hr=[r for r in rows if r['context']==cid and r['Specialist_regime']==H];all_valid=all(r['valid_session'] for r in er+hr);resource_fit=all(r['safe_idle_memory_mib'] is not None and r['safe_idle_memory_mib']<=r['safe_memory_budget_mib']+1e-6 for r in hr)
  hsteady=sum(r['stability']=='steady' and r['valid_session'] for r in hr);esteady=sum(r['stability']=='steady' and r['valid_session'] for r in er);hrejected=sum(r['stability']=='unstable' or not r['valid_session'] for r in hr);contradiction=esteady==3 and hrejected==3
  comparisons.append({'context':cid,'all_sessions_valid':all_valid,'E_steady_repetitions':esteady,'H_steady_repetitions':hsteady,'H_unstable_or_failed_repetitions':hrejected,'CPU_NS_reduction':con['comparison']['T1'],'CPU_G_share_increase':con['comparison']['T2'],'safe_memory_budget_verified':resource_fit,'systematic_operational_contradiction':contradiction,'H_mean_E2E_ms':stat(r['mean_E2E_ms'] for r in hr),'E_mean_E2E_ms':stat(r['mean_E2E_ms'] for r in er)})
 systemic=any(c['systematic_operational_contradiction'] for c in comparisons);inconclusive=not all(r['valid_session'] for r in rows);memory_reject=any(r['safe_idle_memory_mib'] is not None and r['safe_idle_memory_mib']>r['safe_memory_budget_mib']+1e-6 for r in rows if r['Specialist_regime']==H)
 status='STOP_REAL_GPU_SYSTEMATICALLY_REJECTS_CPU_TREND' if systemic else 'STOP_GPU_MEMORY_BUDGET_REJECTION' if memory_reject else 'INCONCLUSIVE_INVALID_EXECUTION' if inconclusive else 'GPU_NOT_SYSTEMATICALLY_REJECTING_CPU_TREND'
 save('module14e_gate.json',{'at_utc':now(),'gate':status,'comparisons':comparisons,'physical_sessions':len(rows),'continue_stable_overflow_search':status=='GPU_NOT_SYSTEMATICALLY_REJECTING_CPU_TREND','Generalist_sharing_gain_proven':False,'CPU_identity_counts_were_validated_as_fixed_deployments_not_GPU_optimality':True})
 (OUT/'module14e_summary.md').write_text('# Module14E\n\n'+status+'\n\n'+__import__('json').dumps(comparisons,indent=2)+'\n')
 print('VALIDATION_GATE',status,flush=True)
if __name__=='__main__':main()
