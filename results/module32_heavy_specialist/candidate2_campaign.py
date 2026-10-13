import sys,shutil,time,fcntl,traceback,collections
from pathlib import Path
OUT=Path('/root/autodl-tmp/generalist-specialist/results/module32_heavy_specialist');sys.path[:0]=[str(OUT),str(OUT.parents[1])]
from m32.common import *
from m32.campaign import execute,describe_B,setup
import numpy as np
from module21.trajectory import choose
from replica_scaling.telemetry import snapshot
G='phi4mm'
def c2slot(f,rid,q,p,dur,seed,stage):
 s=slot(f,rid,q,p,dur,seed,stage);s.update(candidate='C2',Generalist=G)
 s['deployment'].pop('qwen');s['deployment'].update(phi4mm=1,**{'whisper-small':1})
 s['routing']['lid']={'whisper-small':.2};s['routing']['visual_qa']={G:.2};s['routing']['asr']={S:q*(1-p)}
 if p:s['routing']['asr'][G]=q*p
 return s
lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
f=read(OUT/'experiment_manifest.json');assert not f['slots'] and read(OUT/'pressure_load_revision.json')['maximum_adjustments']==1
f['pressure_load_revision_count']=1
assert not snapshot()['processes'];(OUT/'campaign_failure.json').unlink()
old=read(ROOT/'results/module30_positive_split_feasibility/frozen_gpu_manifest_v2.json');catalog=read(OUT/'selected_pool_catalog.json')
f['plan']['models'].pop('qwen');f['plan']['registry'].pop('qwen')
for m in [G,'whisper-small']:f['plan']['models'][m]=catalog['models'][m];f['plan']['registry'][m]=old['plan']['registry'][m]
f.update(formal_candidate='C2',candidate2_GPU_executed=True,candidate2_gpu_reason='C1 real S service expanded and failed S-only bounded drain at 1.68 and 0.92 QPS; no C1 formal arms executed',candidate2_stage_B_protocol=dict(initial_S_qps=.8,initial_seconds=90,capacity_seconds=40,G_probe_qps=.15,G_probe_seconds=120,pressure_rule='min(0.8*actual S closed-loop capacity,0.8/initial S native mean seconds), floor to .01QPS; no rate search or utility selection',pressure_seconds=120,maximum_pressure_adjustments=0,maximum_formal_candidates=1))
f['sources']=[ref(r['path']) for r in f['sources']]+[ref(OUT/'candidate2_campaign.py'),ref(ROOT/'candidate_pool_v2_1_adapters/worker_phi4mm.py')]
current={r['path'] for r in f['checkpoints']}
for directory in [ROOT/'models/phi4-multimodal',ROOT/'models/whisper-small']:
 for p in sorted(directory.rglob('*')):
  if p.is_file() and '.cache' not in p.parts and '__pycache__' not in p.parts and str(p.resolve()) not in current:f['checkpoints'].append(ref(p));current.add(str(p.resolve()))
f['checkpoint_metadata']={r['path']:[Path(r['path']).stat().st_size,Path(r['path']).stat().st_mtime_ns] for r in f['checkpoints']}
save('candidate2_pre_GPU_freeze.json',f);save('experiment_manifest.json',f);save('frozen_gpu_manifest.json',f)
stage=read(OUT/'colocation_profile.json')
try:
 s=c2slot(f,'B_C2_S',.8,0,90,320201,'B_C2');s['capacity_probe']=True;save('candidate2_initial_S_slot.json',s)
 save('campaign_status.json',dict(status='STAGE_B_C2',run_id=s['run_id'],pid=os.getpid()))
 r=execute(s,f);so=describe_B(r,f);stage.append(so);assert so['passed'] and r['capacity_probe']['all_completed'],('C2_INITIAL_S_NOT_VALID',so.get('checks'))
 cap=r['capacity_probe']['actual_ASR_capacity_qps'];sd=next(x['service']['mean'] for x in so['paths'] if x['task']=='asr')/1000
 s=c2slot(f,'B_C2_G',.15,1,120,320202,'B_C2_G');save('candidate2_initial_G_slot.json',s)
 save('campaign_status.json',dict(status='STAGE_B_C2',run_id=s['run_id'],pid=os.getpid()))
 r=execute(s,f);go=describe_B(r,f);stage.append(go);assert go['passed'],('C2_G_NOT_VALID',go.get('checks'))
 gd=next(x['service']['mean'] for x in go['paths'] if x['task']=='asr')/1000;bg=sum(.2*x['service']['mean']/1000 for x in go['paths'] if x['worker']==G+'#0' and x['task']!='asr')
 q=float(np.floor(min(.8*cap,.8/sd)*100)/100);fractions=[.05,.1];reason=None
 if bg+.1*q*gd>=.85:
  upper=(.75-bg)/(q*gd);fractions=[min(.025,upper/2),min(.05,upper)];reason='Measured C2 G demand has clear capacity risk at 10%; one pre-formal smaller split freeze'
  assert min(fractions)>0
 save('candidate2_pressure_preselection.json',dict(S_capacity_qps=cap,S_native_service_seconds=sd,G_native_service_seconds=gd,G_background_demand_ratio=bg,G_nominal_remaining_ASR_qps=(1-bg)/gd,selected_ASR_qps=q,split_fractions=fractions,split_change_reason=reason,U_used=False,rule=f['candidate2_stage_B_protocol']['pressure_rule']))
 pressure=[]
 for rid,p in [('B_C2_PRESSURE_S',0),('B_C2_PRESSURE_G',fractions[1])]:
  s=c2slot(f,rid,q,p,120,320203,'B_C2_PRESSURE')
  if pressure:s['trajectories']=copy.deepcopy(pressure[0]['trajectories'])
  pressure.append(s)
 save('candidate2_pressure_slots.json',pressure);po=[]
 for s in pressure:
  save('campaign_status.json',dict(status='STAGE_B_C2_PRESSURE',run_id=s['run_id'],pid=os.getpid()))
  r=execute(s,f);o=describe_B(r,f);stage.append(o);po.append(o);assert o['passed'],('C2_PRESSURE_NOT_VALID',o['run_id'],o.get('checks'))
 wait=next(x['wait']['mean'] for x in po[0]['paths'] if x['task']=='asr');assert wait>200,('C2_NO_OBSERVED_QUEUE_PRESSURE',wait)
 f['slots']=[]
 for rid,p,arm in [('A0',0,'s_only'),('A1',fractions[0],'split05'),('A2',fractions[1],'split10')]:
  s=c2slot(f,rid,q,p,600,320232,'C_EXPLORATORY');s['arm']=arm
  if f['slots']:s['trajectories']=copy.deepcopy(f['slots'][0]['trajectories'])
  ev=read(s['trajectories']['measurement']['path'])['events'];counts=collections.Counter(e['task'] for e in ev);g=sum(e['task']=='asr' and choose(e,s['routing'],s['deployment'])==G+'#0' for e in ev)
  s.update(frozen_offered_counts=dict(counts),actual_planned_G_asr=g,actual_planned_G_fraction=g/counts['asr'])
  assert all(counts[t]>=72 for t in counts if t!='asr') and (p==0 or g>=20)
  f['slots'].append(s)
 f['formal_load_frozen_unix_ns']=time.time_ns();f['split_fractions']=fractions;f['split_revision_count']=int(reason is not None);f['stage_B_evidence']=[ref(OUT/'stage_B'/ (x['run_id']+'.json')) for x in stage]
 save('frozen_load.json',dict(at_unix_ns=f['formal_load_frozen_unix_ns'],ASR_qps=q,background_qps=.2,B3_budget_mib=34567.5534,slots=f['slots'],split_revision_count=f['split_revision_count'],split_change_reason=reason,selection_evidence=f['stage_B_evidence'],no_utility_result_used_for_selection=True))
 save('experiment_manifest.json',f);save('frozen_gpu_manifest.json',f);save('stage_B_gate.json',dict(passed=True,observed_S_mean_wait_ms=wait,selected_ASR_qps=q,formal_candidate='C2',candidate1_GPU_formal_executed=False,candidate2_GPU_executed=True))
 save('colocation_profile.json',stage)
 for rid in f['formal_order']:
  s=next(s for s in f['slots'] if s['run_id']==rid);save('campaign_status.json',dict(status='STAGE_C',run_id=rid,pid=os.getpid()))
  r=execute(s,f);_,_,r31=setup();o=r31.assess(r);save('formal_acceptance/'+rid+'.json',o)
  print('FORMAL_ACCEPT',rid,o['eligible'],o['U'],o['validation'],flush=True)
  assert o['technical_valid'],('FORMAL_TECHNICAL_FAILURE',rid,o['validation'])
 save('campaign_status.json',dict(status='COMPLETED',formal_candidate='C2',formal_arms=['A1','A0','A2'],independent_replication_executed=False))
except BaseException as ex:
 save('campaign_failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));save('campaign_status.json',dict(status='STOPPED_INCONCLUSIVE',error=repr(ex)));print('STOPPED',repr(ex),flush=True)
finally:
 save('colocation_profile.json',stage)
 from m32.report import generate
 generate()
