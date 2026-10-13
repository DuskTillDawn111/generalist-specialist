"""Archival evidence and all bounded search/selection rules frozen before GPU."""
from .common import *
from module21.trajectory import arrivals
from joint_deploy.measurement.load_generator import SamplePool
from module20.optimizer import predict

def trajectories(name,rates,seed,duration,plan):
 refs={};counts={}
 for label,seconds in [('warmup',30.),('measurement',duration)]:
  pool=SamplePool(plan['sample_pool']['path'],list(rates),seed);events=arrivals(rates,seconds,seed,label)
  for e in events:e['item']=pool.next(e['task'])
  fn=f'trajectories/{name}_{label}.json';save(fn,dict(rates=rates,duration=seconds,seed=seed,label=label,events=events));refs[label]=ref(OUT/fn)
  if label=='measurement':counts=dict(Counter(e['task'] for e in events))
 return refs,counts

def slot(name,stage,point,rates,fraction,seed,rep,duration,traces):
 return dict(run_id=name,stage=stage,arm='control' if fraction==0 else 'proposed',pool_id='LID_MEDIUM_SHARED_ASR',
  point_id=point,target_task=TARGET,lambda_rps=rates,deployment={m:1 for m in MODELS},routing=routing(rates,fraction),
  G_fraction=fraction,seed=seed,repetition=rep,measurement_seconds=duration,safe_memory_budget_mib=BUDGET,trajectories=traces,
  diagnostic_only=stage=='B',optimizer_selected_configuration=False,
  training_role='EXPLORATORY_DEVELOPMENT' if stage=='B' else 'NEW_INDEPENDENT_VALIDATION_NEVER_TRAIN')

def main():
 assert not (OUT/'discovery_protocol.json').exists(),'No frozen protocol replacement'
 OUT.mkdir(parents=True,exist_ok=True);save('historical_metadata_before.json',protected_metadata())
 old=read(SOURCE);d=copy.deepcopy(old)
 for key in ['models','capacity_factors','memory_table_mib','evidence_bounds','model_evidence']:
  d[key]={m:d[key][m] for m in MODELS}
 d['module27_subset_only']=True;save('selected_pool_catalog.json',d)
 definition,refs=utility_inputs();weights=definition['weights']
 hist=read(ROOT/'results/module25_utility_difference/observations.json');hist+=read(ROOT/'results/module26_positive_probe/replication_observations.json')
 evidence=[]
 for r in hist:
  for p in r.get('paths',[]):
   evidence.append(dict(module=r['module'],run_id=r['run_id'],raw=r['raw'],model=p['model'],task=p['task'],count=p['count'],
    actual_service=p['actual_service'],actual_queue=p['actual_queue'],actual_e2e=p['actual_e2e'],
    deployment=r['deployment'],rates=r['rates'],routing=r['routing'],eligible=r['eligible'],hard_unstable=r['hard_unstable'],
    original_whole_session_group=r.get('group'),no_request_split=True))
 csvsave('historical_service_evidence.csv',evidence)
 inventory=[]
 dirs={14:'module14f_memory_ocr_x3',15:'module15_three_phase',16:'module16_fullpool_gpu',17:'module17_runtime_capacity',18:'module18_cross_task',
  19:'module19_strategy_benchmark',20:'module20_utility_optimizer',21:'module21_positive_utility','21D':'module21_low_diagnosis',22:'module22_mid_high_utility',
  23:'module23_positive_three_phase','24A':'module24_utility_calibration','24B':'module24b_high_validation',25:'module25_utility_difference',26:'module26_positive_probe'}
 for module,name in dirs.items():
  root=ROOT/'results'/name;native=sorted(root.glob('raw_runs/*.json'))
  inventory.append(dict(module=module,directory=str(root),raw_sessions_available=len(native),raw_files=[str(p) for p in native],
   normalized_path_rows=sum(v['module']==module for v in evidence),no_new_GPU_in_Stage_A=True))
 save('historical_data_availability.json',inventory)
 # Raw service observations on both routes; no summary U is used to screen.
 native=[];consumed=[SOURCE,ROOT/'results/module25_utility_difference/observations.json',ROOT/'results/module26_positive_probe/replication_observations.json',
  ROOT/'results/module20_utility_optimizer/utility_definition.json',ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json',
  ROOT/'results/module8b_quality_pool_finalization/quality_policy.json']
 for parent,pattern in [('module18_cross_task','lid_*'),('module19_strategy_benchmark','lid_*'),('module26_positive_probe','high_*')]:
  for path in sorted((ROOT/'results'/parent/'raw_runs').glob(pattern+'.json')):
   raw=read(path);consumed.append(path);ph=raw.get('phases',{}).get('measurement')
   if not ph:continue
   done=[r for r in ph['request_records'] if r['status']=='completed']
   for m,t in [(MEDIUM,'lid'),(G,'lid'),(G,'asr'),(G,'visual_qa')]:
    rows=[r for r in done if r['task']==t and raw['replica_to_family'][r['model']]==m]
    if rows:native.append(dict(raw=ref(path),run_id=raw['slot']['run_id'],model=m,task=t,n=len(rows),
      service=stats([r['worker_latency_ms'] for r in rows]),queue=stats([r['worker_queue_wait_ms'] for r in rows]),
      input_scope=raw['slot']['lambda_rps'],arrival_process=raw.get('transport_config',{}).get('arrival_process'),
      original_system_stability=ph['stability']['stability_state_v2']))
 csvsave('native_archival_path_check.csv',native)
 medium_control=[v['service']['mean_ms'] for v in native if v['model']==MEDIUM and v['task']=='lid' and 'lid_high_s_oriented' in v['run_id']]
 medium_split=[v['service']['mean_ms'] for v in native if v['model']==MEDIUM and v['task']=='lid' and 'lid_high' in v['run_id'] and 's_oriented' not in v['run_id']]
 generalist=[v['service']['mean_ms'] for v in native if v['model']==G and v['task']=='lid' and 'lid_high' in v['run_id']]
 assert len(medium_control)==3 and len(medium_split)>=3 and len(generalist)>=3
 screen=[dict(target='asr',decision='REJECT_PRIORITY',reason='Module26 original five-model ASR18 percent independently negative in3 long pairs; large G service and background waits'),
  dict(target='text',decision='REJECT_CURRENT_GRID',reason='Historical beneficial-looking J routes only about3 percent; 10/20 percent at specialist pressure can overload slow G; no positive system-U counterfactual'),
  dict(target='image',decision='REJECT_CURRENT_GRID',reason='Historical target G fraction0.36 percent; default10/20 percent cannot be safely extrapolated'),
  dict(target='ocr',decision='REJECT_EVIDENCE',reason='Qwen lacks admitted OCR; alternative G OCR service/support not measured for promising matched coexistence'),
  dict(target='lid',specialist='whisper_turbo',decision='REJECT_EVIDENCE',reason='Module16/17 shared ASR/LID congestion; no scoped Qwen/turbo joint support at desired breakpoint'),
  dict(target='lid',specialist='MMS-LID256',decision='REJECT_EVIDENCE',reason='Qwen/MMS joint route-conditioned path support absent; warm memory uncertainty and extra ASR model'),
  dict(target='lid',specialist=MEDIUM,decision='SELECT_ONE_POOL',reason='Both native LID paths measured; medium75-77ms pure vs85-87ms split, G300-365ms. Shift ASR to same medium and protect Text/Image with resident specialists; new Poisson pressure not claimed validated.')]
 csvsave('candidate_screening.csv',screen)
 base=read(ROOT/'results/module18_cross_task/frozen_task_manifests.json')['plan'];plan=copy.deepcopy(base)
 plan['registry']={m:plan['registry'][m] for m in MODELS};plan['models']=d['models'];plan['warmup_seconds']=30.
 for m in MODELS:
  source=Path(plan['registry'][m]['script']);assert source.exists() and Path(plan['registry'][m]['python']).exists()
  consumed.append(source)
  if 'results' in source.parts:
   own=OUT/'launchers'/('medium.py' if m==MEDIUM else 'parseq.py');own.parent.mkdir(parents=True,exist_ok=True);own.write_bytes(source.read_bytes());plan['registry'][m]['script']=str(own)
 plan['environment_overrides'].update(PYTHONDONTWRITEBYTECODE='1')
 points=[];slots=[];future=[];hypotheses=[]
 for pi,q in enumerate([10.5,11.5],1):
  point=f'lid_{q:g}';rates={t:.2 for t in ['text','image','ocr','asr','lid','visual_qa']};rates['lid']=q
  traces,counts=trajectories(f'{point}_B',rates,27100+pi,600.,plan)
  order=[0.,.1,.2] if pi==1 else [.2,0.,.1]
  for fraction in order:
   name=f'B_{point}_g{round(fraction*100):02d}';slots.append(slot(name,'B',point,rates,fraction,27100+pi,pi,600.,traces))
   hypotheses.append(dict(point_id=point,G_fraction=fraction,rates=rates,
    old_Module20_prediction=predict(d,{m:1 for m in MODELS},routing(rates,fraction),rates,BUDGET,refs,weights,phase=.37),
    medium_measured_LID_range_ms=[min(medium_control if fraction==0 else medium_split),max(medium_control if fraction==0 else medium_split)],
    scoped_service_demand_check={'S':q*(1-fraction)*max(medium_control if fraction==0 else medium_split)/1000+.2*d['models'][MEDIUM]['service_time_ms']['asr']/1000,
     'G':q*fraction*max(generalist)/1000+.2*.375},
    capacity_certified=False,shared_ASR_LID_joint_service_support='MISSING_EXACT_NEW_COMBINATION; candidate hypothesis only',
    no_unified_interference_factor=True))
  independent=[]
  for rep,seed in enumerate([27201,27202,27203],1):
   tr,n=trajectories(f'{point}_C_rep{rep}',rates,seed,2400.,plan);independent.append(dict(rep=rep,seed=seed,trajectories=tr,counts=n))
  points.append(dict(point_id=point,target_qps=q,rates=rates,exploration_order=order,exploration_task_counts=counts,independent_pre_generated=independent))
 save('cpu_candidate_hypotheses.json',hypotheses)
 sources=[ref(p) for p in dict.fromkeys(consumed)];save('consumed_historical_sources.json',sources)
 quality=read(ROOT/'results/module8b_quality_pool_finalization/quality_policy.json')
 for s in slots:
  assert set(s['lambda_rps'])==set(refs)
  for t,row in s['routing'].items():
   assert abs(sum(row.values())-s['lambda_rps'][t])<1e-10
   assert all(t in d['models'][m]['supported_tasks'] for m in row)
 memory=d['safe_factor']*sum(d['memory_table_mib'][m]['1'] for m in MODELS);assert memory<=BUDGET
 audit=dict(at_utc=now(),passed=True,selected_pools=1,selected_points=2,pool=MODELS,single_replicas_only=True,
  safe_memory_prediction_mib=memory,budget_mib=BUDGET,medium_pure_LID_service_ms=medium_control,
  medium_split_LID_service_ms=medium_split,Qwen_split_LID_service_ms=generalist,
  original_quality_policy_unchanged=True,utility_weights_unchanged=True,refs_unchanged=True,
  independent_arrivals_change_old_periodic_queue_collision_pattern=True,
  no_existing_positive_evidence_claim=True,no_new_predictor=True,
  reason='Near measured S queue breakpoint, G freed from ASR/Text/Image background; retain measured S/G interference separately, all future GPU outcomes unknown',
  gaps=['New joint medium-ASR/LID Poisson capacity not yet measured','Old medium gets slower under split; gains cannot be assumed','Unknown warm memory/tail/route feedback under new loads'])
 save('stage_a_evidence_audit.json',audit)
 protocol=dict(at_utc=now(),status='FROZEN_BEFORE_GPU',Stage_A=ref(OUT/'stage_a_evidence_audit.json'),catalog=ref(OUT/'selected_pool_catalog.json'),
  utility_definition=ref(ROOT/'results/module20_utility_optimizer/utility_definition.json'),utility_reference=ref(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json'),
  quality_policy=ref(ROOT/'results/module8b_quality_pool_finalization/quality_policy.json'),plan=plan,budget_mib=BUDGET,safe_factor=d['safe_factor'],
  models=MODELS,target=TARGET,weights=weights,refs=refs,points=points,exploration_slots=slots,
  maximum_exploration_startup_attempts=6,user_maximum_exploration_runs=12,exploration_seconds=600.,grid=[0.,.1,.2],
  no_budget_expansion=True,no_outcome_retries=True,no_new_pool_after_freeze=True,
  thermal_protocol=dict(prelaunch_temperature_max_c=38.,consecutive_idle_samples_required=3,max_wait_seconds=300.,between_samples_seconds=2.,
   resident_premeasurement_temperature_max_c=40.,resident_cool_timeout_seconds=180.,resident_consecutive_samples=3,
   postwarm_pair_temperature_difference_max_c=5.,clocks_power_not_modified=True),
  exploratory_selection=dict(delta_U_over=.01,both_arms_legal_stable=True,matched_cohort_residency=True,min_task_requests=72,
   actual_G_and_S_nonzero=True,conditional_60s_block_sensitivity_lower_over_zero=True,thermal_match_max_c=5.,
   ranking='Highest development delta_U, tie canonical run id; one winner only; no StageB data in independent denominator',
   claim='Exploratory candidate only. Block sensitivity is not independent confidence or cross-pool generalization'),
  independent_protocol=dict(pairs=3,measurement_seconds=2400.,seeds=[27201,27202,27203],orders=[['control','proposed'],['proposed','control'],['control','proposed']],
   all_delta_over=.01,paired_t95_lower_over_zero=True,min_task_requests=200,all_legal_stable_thermal_matched=True,
   negative_results_retained_no_extra_seeds=True,no_development_data_in_denominator=True,normal_effects_assumption_and_n3_limits=True),
  failure_policy='Fatal/runtime/quality/memory/GPU-release/thermal-launch failure stops campaign; growing-queue unstable run gets U=null and stops remaining runs at that and higher same-pool loads. Never compare censored U.',
  optimizer_protocol=dict(stage_D_only_after_independent_positive=True,implementation='Unmodified module20.optimizer.Generator and predict/predict_scenarios',
   pool='Frozen five-model pool; all measured replica counts and all admitted task routes compete',
   target_constraints=None,no_forced_GS=True,no_predictor_fit=True,
   discovery_requirement='Exact validated deployment and all-task route within1e-6 relative QPS; near route or other replicas lack GPU proof',
   scope='Heuristic formal Module20 candidate generator, retain pure endpoints and search metadata, not exhaustive continuous global certificate'),
  implementation=[ref(p) for p in sorted((ROOT/'module27').glob('*.py'))]+[ref(ROOT/p) for p in ['module20/optimizer.py','module21/trajectory.py','module22/gpu.py','module11b/measure.py','module16/gpu.py','module13b/worker.py','module14b/worker.py','joint_deploy/measurement/deployment_runtime.py','joint_deploy/worker_protocol.py'] if (ROOT/p).exists()]+[ref(v['script']) for v in plan['registry'].values()],
  old_history_immutable=True,next_module_automatic=False)
 save('discovery_protocol.json',protocol);print('FROZEN',len(slots),'runs',memory,'MiB safe predicted; independent conditional2400s x6',flush=True)

if __name__=='__main__':main()
