from .common import *
import numpy as np,traceback,fcntl
from module31a.campaign import TimedMonitor
from module20.optimizer import components
from module25.common import utility_inputs
from module26.common import arithmetic,cohort
from module21.trajectory import choose

def setup():
 _,report=configure()
 from . import harness
 harness.OUT=OUT;harness.save=save;harness.csvsave=csvsave
 from module31a import report as r31
 r31.OUT=OUT;r31.save=save;r31.csvsave=csvsave;r31.configure=configure
 return harness,report,r31

def execute(s,f):
 harness,_,_=setup()
 for r in f['sources']+f['frozen_inputs']:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 for p,meta in f['checkpoint_metadata'].items():assert [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==meta,p
 print('RUN_START',s['run_id'],s['lambda_rps'],s['frozen_G_fraction'],flush=True)
 raw=harness.run(s,f);m=TimedMonitor.instances[-1]
 raw['control_latency_seconds']=dict(max_check=max(m.check_seconds,default=0),max_publish=max(m.publish_seconds,default=0));raw['control_nonblocking']=max(m.check_seconds+m.publish_seconds,default=0)<.01
 save('raw_runs/'+s['run_id']+'.json',raw)
 print('RUN_FINISH',s['run_id'],raw['valid_session'],raw['errors'],flush=True);return raw

def describe_B(r,f):
 _,report,_=setup();ph=r.get('phases',{}).get('measurement');o=dict(run_id=r['slot']['run_id'],valid_session=r['valid_session'],errors=r['errors'],passed=False)
 if not ph:return o
 rs=ph['request_records'];done=[x for x in rs if x['status']=='completed'];w=ph['window'];dur=w['duration_seconds']
 safe=f['safe_factor']*max(np.mean([x['owned_used_mib'] for x in r[k]]) for k in ['postwarm_idle','postmeasurement_idle'])
 peak=max(x['owned_used_mib'] for x in r['telemetry']['samples']);paths=[]
 for t in r['slot']['lambda_rps']:
  for m in {x['model'] for x in done if x['task']==t}:
   pp=[x for x in done if x['task']==t and x['model']==m]
   paths.append(dict(task=t,worker=m,n=len(pp),service=report.stats(pp,'worker_latency_ms'),wait=report.stats(pp,'worker_queue_wait_ms'),e2e=report.stats(pp,'e2e_latency_ms'),observed_busy_fraction=sum(x['worker_latency_ms'] for x in pp)/dur/1000))
 from types import SimpleNamespace
 from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS
 workers=[]
 for m in r['worker_identities']:
  pp=[x for x in rs if x['model']==m]
  ev=reconstruct(dict(request_records=pp,measurement_window=w,status='completed',errors=r['errors'],drain_timeout_seconds=60),{m:SimpleNamespace(service_time_ms=f['plan']['models'][r['replica_to_family'][m]]['service_time_ms'])},DEFAULT_THRESHOLDS)
  workers.append(dict(worker=m,classification=classify(ev,DEFAULT_THRESHOLDS),evidence=ev))
 cl=[ph['stability']]+[x['classification'] for x in workers];stable=all(x['stability_state_v2']!='unstable' and (x['stability_state_v2']!='recoverable_congested' or x['final_recovery_confirmed']) for x in cl)
 checks=dict(valid_session=r['valid_session'],full_cohort=len(done)==len(rs),stable=stable,memory=safe<=r['slot']['safe_memory_budget_mib'] and peak<=r['slot']['safe_memory_budget_mib'],journal=report.journal_audit(r)['passed'],nonblocking=r['control_nonblocking'])
 o.update(passed=all(checks.values()),checks=checks,paths=paths,workers=workers,stability=ph['stability'],safe_memory_mib=safe,peak_memory_mib=peak,offered=len(rs),completed=len(done),within_qps=sum(x['completion_time_ns']<=w['end_time_ns'] for x in done)/dur,drain_seconds=w['drain_seconds'],capacity={k:v for k,v in r.get('capacity_probe',{}).items() if k!='request_records'},per_task=ph['per_task_metrics'])
 save('stage_B/'+o['run_id']+'.json',o);return o

def main():
 lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 f=read(OUT/'experiment_manifest.json');setup();stage=[]
 try:
  bs=slot(f,'B_S',1.4,0,90,320001,'B');bs['capacity_probe']=True
  save('stage_B_frozen_initial_slots.json',[bs,slot(f,'B_G',.6,1,90,320002,'B')])
  save('campaign_status.json',dict(status='STAGE_B',run_id='B_S',pid=os.getpid()))
  r=execute(bs,f);o=describe_B(r,f);stage.append(o);assert r['valid_session'],o
  cap=r['capacity_probe']['actual_ASR_capacity_qps'];sd=next(x['service']['mean'] for x in o['paths'] if x['task']=='asr' and x['worker']==S+'#0')/1000
  assert r['capacity_probe']['all_completed']
  gs=slot(f,'B_G',.6,1,90,320002,'B');r=execute(gs,f);go=describe_B(r,f);stage.append(go);assert go['passed'],go
  gd=next(x['service']['mean'] for x in go['paths'] if x['task']=='asr')/1000
  bg=sum(x['observed_busy_fraction'] for x in go['paths'] if x['worker']=='qwen#0' and x['task']!='asr')
  q=np.floor(min(.90*cap,.90/sd)*100)/100
  assert q>0 and bg+.1*q*gd<.8,('CAPACITY_PRECHECK',q,bg,gd)
  save('stage_B_pressure_preselection.json',dict(S_capacity_qps=cap,S_service_seconds=sd,G_service_seconds=gd,G_background_busy_ratio=bg,G_remaining_nominal_qps=(1-bg)/gd,selected_ASR_qps=q,split10_projected_G_demand=bg+.1*q*gd,rule=f['stage_B_protocol']['pressure_rate_rule'],U_not_used_for_selection=True))
  pressure=[]
  for rid,p in [('B_PRESSURE_S',0),('B_PRESSURE_10',.1)]:
   s=slot(f,rid,q,p,120,320003,'B_PRESSURE')
   # Share both arrival/input traces in this capacity precheck.
   if pressure:s['trajectories']=copy.deepcopy(pressure[0]['trajectories'])
   pressure.append(s)
  save('stage_B_pressure_slots.json',pressure)
  po=[]
  for s in pressure:
   r=execute(s,f);o=describe_B(r,f);stage.append(o);po.append(o);assert o['passed'],('PRESSURE_NOT_LEGAL_STABLE',o)
  wait=next(x['wait']['mean'] for x in po[0]['paths'] if x['task']=='asr')
  assert wait>200,('NO_OBSERVED_QUEUE_PRESSURE',wait)
  f['slots']=[]
  for rid,p in [('A0',0),('A1',.05),('A2',.1)]:
   s=slot(f,rid,q,p,600,320032,'C')
   if f['slots']:s['trajectories']=copy.deepcopy(f['slots'][0]['trajectories'])
   tr=read(s['trajectories']['measurement']['path'])['events'];counts=collections.Counter(e['task'] for e in tr);g=sum(e['task']=='asr' and choose(e,s['routing'],s['deployment'])=='qwen#0' for e in tr)
   s.update(frozen_offered_counts=dict(counts),actual_planned_G_asr=g,actual_planned_G_fraction=g/counts['asr'])
   assert all(counts[t]>=72 for t in counts if t!='asr') and (p==0 or g>=20)
   f['slots'].append(s)
  f['formal_load_frozen_unix_ns']=time.time_ns();f['stage_B_evidence']=[ref(OUT/'stage_B'/ (x['run_id']+'.json')) for x in stage]
  f['split_fractions']=[.05,.10];f['split_revision_count']=0
  save('frozen_load.json',dict(at_unix_ns=f['formal_load_frozen_unix_ns'],ASR_qps=q,background_qps=.2,B3_budget_mib=34567.5534,slots=f['slots'],split_revision_count=0,selection_evidence=f['stage_B_evidence'],no_utility_result_used_for_selection=True))
  save('experiment_manifest.json',f);save('frozen_gpu_manifest.json',f);save('stage_B_gate.json',dict(passed=True,observed_S_mean_wait_ms=wait,selected_ASR_qps=q,split10_projected_G_demand=bg+.1*q*gd,formal_candidate='C1',candidate2_GPU_executed=False))
  save('colocation_profile.json',stage)
  for rid in f['formal_order']:
   s=next(s for s in f['slots'] if s['run_id']==rid);save('campaign_status.json',dict(status='STAGE_C',run_id=rid,pid=os.getpid()))
   r=execute(s,f);_,_,r31=setup();o=r31.assess(r);save('formal_acceptance/'+rid+'.json',o)
   print('FORMAL_ACCEPT',rid,o['eligible'],o['U'],o['validation'],flush=True)
   assert o['technical_valid'],('FORMAL_TECHNICAL_FAILURE',rid,o)
  save('campaign_status.json',dict(status='COMPLETED',formal_arms=['A1','A0','A2'],independent_replication_executed=False))
 except BaseException as ex:
  save('campaign_failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));save('campaign_status.json',dict(status='STOPPED_INCONCLUSIVE',error=repr(ex)));print('STOPPED',repr(ex),flush=True)
 finally:
  save('colocation_profile.json',stage)
  from .report import generate
  generate()
if __name__=='__main__':main()
