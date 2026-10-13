import sys,shutil,subprocess,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32b.common import *
from replica_scaling.telemetry import snapshot
from performance_model.local_calibration.stability_v2 import DEFAULT_THRESHOLDS
assert read(OUT/'dynamic_opportunity_gate.json')['stage_B_passed'];assert read(OUT/'CPU_router_acceptance.json')['passed'];assert not (OUT/'frozen_routing_manifest.json').exists()
old=read(OLD/'experiment_manifest.json');f=copy.deepcopy(old);g=snapshot();assert not g['processes'] and g['used_mib']<=128;assert g['uuid']==old['GPU']['uuid']
for r in old['sources']+old['frozen_inputs']:assert ref(r['path'])['sha256']==r['sha256'],r['path']
for p,v in old['checkpoint_metadata'].items():assert [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==v,p
shutil.copyfile(OLD/'selected_pool_catalog.json',OUT/'selected_pool_catalog.json');save('frozen_routing_parameters.json',read(OUT/'draft_routing_parameters.json'))
f['slots']=[];old10=next(s for s in old['slots'] if s['frozen_G_fraction']==.1);old0=next(s for s in old['slots'] if s['frozen_G_fraction']==0)
for aid,oldslot,arm in [('A0',old0,'s_only'),('A1',old10,'fixed10'),('A2',old0,'dynamic')]:
 s=copy.deepcopy(oldslot);s.update(run_id=aid,arm=arm,stage='Module32B_EXPLORATORY')
 if arm=='dynamic':s['routing']['asr']={S[:-2]:.46*.95,G[:-2]:.46*.05};s['frozen_G_fraction']=None;s['actual_planned_G_asr']=None;s['actual_planned_G_fraction']=None
 f['slots'].append(s)
f.update(module='Module32B VQA Interference Attribution and Background-Aware Dynamic Routing',formal_order=['A1','A0','A2'],frozen_unix_ns=time.time_ns(),GPU=g,formal_candidate='C2',stability_thresholds=DEFAULT_THRESHOLDS,maximum_formal_runs=3,no_GPU_smoke_or_extra_capacity_runs=True,optimizer_modified=False,independent_replication=False)
f['sources']=old['sources']+[ref(p) for p in sorted((OUT/'m32b').glob('*.py'))]+[ref(OUT/n) for n in ['vqa_aware_dynamic_router.py','test_router_cpu.py','freeze_exploration.py']]
f['frozen_inputs']=old['frozen_inputs']+[ref(OUT/n) for n in ['frozen_routing_parameters.json','audio_metadata_cache.json','dynamic_opportunity_gate.json','CPU_router_acceptance.json','selected_pool_catalog.json','clock_and_cuda_timing_audit.json','frozen_utility_context.json']]
f['positive_gate']=dict(dynamic_U_vs_S_only_gt=.01,dynamic_U_vs_fixed_gt=.01,all_three_legal_stable=True,matched_trajectory=True,actual_dynamic_G_and_S_execution=True,quality_memory_stability='Original Module20A unchanged',ineligible_U=None)
f['history_baseline_policy']='Historical Module32 numbers used only for CPU attribution/calibration. All three current arms remeasured under same new harness/environment and trajectory, no historical U as strict comparator.'
f['dynamic_policy_causal_schema']=['current input audio duration header','current monotonic time','queued/submitted pending model/task/host submit time/S cost estimate','past admissions and exposure budget']
save('experiment_manifest.json',f);save('frozen_gpu_manifest.json',f)
save('frozen_routing_manifest.json',dict(status='FROZEN_BEFORE_FORMAL_GPU',at_unix_ns=f['frozen_unix_ns'],parameters=ref(OUT/'frozen_routing_parameters.json'),router_source=ref(OUT/'vqa_aware_dynamic_router.py'),implementation=[ref(p) for p in sorted((OUT/'m32b').glob('*.py'))],six_models=f['slots'][0]['deployment'],rates=f['slots'][0]['lambda_rps'],budget_mib=f['slots'][0]['safe_memory_budget_mib'],slots=f['slots'],order=f['formal_order'],warmup_seconds=30.,drain_timeout_seconds=60.,measurement_seconds=600.,maximum_formal_runs=3,never_adjust_after_GPU=True,original_fixed_baseline='Module32 Frozen10, remeasured current run',causal_schema=f['dynamic_policy_causal_schema'],nonpreemptive_limit='Future VQA can arrive during accepted ASR and cannot preempt it; mean-risk/exposure predictions are uncertain, not an SLO guarantee',positive_gate=f['positive_gate'],original_utility=ref(ROOT/'results/module20_utility_optimizer/utility_definition.json'),metadata_cache_contains_inputs_not_future_arrival_or_results=True,calibration_and_replay_share_historical_trajectory=True,exploration_only_no_independent_validation=True))
print('FROZEN',f['formal_order'],g['uuid'],flush=True)
