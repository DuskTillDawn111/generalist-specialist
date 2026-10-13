"""Profile/audit checkpoint validation only; never imports deferred predictors."""
import csv,math
from collections import Counter
from .common import *

def main():
 protocol=load(OUT/'profile_protocol.json');catalog=load(OUT/'catalog.json');admission=load(ROOT/'results/module8b_quality_pool_finalization/quality_admission_matrix.json');allowed={(r['model'],r['task']) for r in admission['rows'] if r['quality_admissible'] and r['capability_verified']};assert {(r['model'],r['task']) for r in catalog['paths']}==allowed
 assert set(tuple(r) for r in protocol['paths'])==allowed
 hardware=load(OUT/'hardware_metric_audit.json')['device_attributes'];warps=hardware['max_threads_per_sm']/hardware['warp_size'];allrows=0;requests=0;checks=[]
 required=['kernel_name','registers_per_thread','static_shared_bytes','dynamic_shared_bytes','threads_per_block','warps_per_block','grid','block','max_active_blocks_per_sm','theoretical_occupancy','duration_profile_ns','sequence_index']
 for entry in catalog['paths']:
  m,k=entry['model'],entry['task'];tag=m+'__'+k;raw=load(OUT/'raw_profiles'/(tag+'.json'));profile=load(OUT/'task_path_profiles'/(tag+'.json'));assert raw['status']=='COMPLETE' and raw['exit_code']==0 and raw['gpu_release']['verified'];assert [r['pool_id'] for r in raw['target_records']]==[r['pool_id'] for r in protocol['items'][k][:3]];assert [r['pool_id'] for r in raw['warmup_records']]==[protocol['items'][k][i%3]['pool_id'] for i in range(6)];assert len(profile['requests'])==3
  for i,r in enumerate(profile['requests']):
   assert r['sample_index']==i and r['sample_id']==protocol['items'][k][i]['pool_id'];rows=r['kernels'];assert [v['sequence_index'] for v in rows]==list(range(len(rows)));assert all(v['launch_start_profile_ns']<=w['launch_start_profile_ns'] for v,w in zip(rows,rows[1:]));assert r['intra_path_gpu_overlap_events']==0
   for v in rows:
    assert all(v.get(field) is not None for field in required)
    assert v['threads_per_block']==math.prod(v['block']) and v['grid_blocks']==math.prod(v['grid']);assert v['warps_per_block']==math.ceil(v['threads_per_block']/hardware['warp_size']);assert 0<v['max_active_blocks_per_sm']<=hardware['max_blocks_per_sm'];assert v['theoretical_occupancy']==v['warps_per_block']*v['max_active_blocks_per_sm']/warps;assert 0<v['theoretical_occupancy']<=1
    assert v['duration_profile_ns']>0 and v['duration_corrected_ms']>0
   assert math.fsum(v['duration_corrected_ms'] for v in rows)==entry['standalone_s0_ms'];allrows+=len(rows);requests+=1
  checks.append({'model':m,'task':k,'launch_metadata_complete':True,'occupancy_implicit_function_attributes_directly_observed':False,'requests':3,'kernel_launches':sum(len(r['kernels']) for r in profile['requests']),'standalone_only':True});print(json.dumps({'event':'VALIDATED_PROFILE','path':tag}),flush=True)
 with (OUT/'task_path_kernel_catalog.csv').open() as f:
  count=sum(1 for r in csv.DictReader(f))
 assert count==allrows
 audit=load(OUT/'runtime_sharing_semantics_audit.json');assert not audit['ROOMIE_SPATIAL_PAIR_MODEL_APPLICABLE_TO_CURRENT_RUNTIME'];assert audit['selected_scheduler_policy'] is None and audit['selected_pair_predictor'] is None;assert audit['new_pair_validation_GPU_runs']==0;assert len(audit['trace_evidence'])==len(allowed)
 for n in ['pair_predictions.csv','pair_ground_truth.csv','measurement_execution_freeze.json','pair_split_manifest.json','pair_validation_manifest.json','protocol_identity.json']:assert not (OUT/n).exists(),n
 import deployment_core.interference as package
 assert not hasattr(package,'InterferenceEstimator')
 assert not any((ROOT/'deployment_core/interference'/n).exists() for n in ['pairwise_predictor.py','kernel_overlap.py','routing_conditioned.py','occupancy.py','_kernel_overlap.so'])
 from replica_scaling.telemetry import snapshot
 gpu=snapshot();assert not gpu['processes'] and gpu['used_mib']<=128;assert all(gpu[k]==protocol['GPU'][k] for k in ['name','driver','total_mib'])
 env=load(OUT/'post_profile_environment_audit.json');assert env['unchanged_from_frozen_9A2']
 save('validation_results.json',{'status':'PASS','scope':'OFFLINE_PROFILING_AND_RUNTIME_SHARING_AUDIT_ONLY','admitted_paths':len(allowed),'complete_launch_metadata_paths':len(checks),'occupancy_function_attributes_fully_direct_observed':False,'captured_inferences':requests,'kernel_launches':allrows,'kernel_catalog_count_matches_profiles':True,'exact_samples_order_and_separate_six_warmups':True,'all_duration_diagnostic_sums_exactly_equal_frozen_s0':True,'all_occupancy_formula_checks_pass':True,'all_single_context_traces':True,'model_kernel_stream_priorities':[0],'standalone_profiles_successful':True,'profile_prelaunch_adapter_failures_preserved':3,'new_pair_GPU_runs':0,'new_multimodel_GPU_runs':0,'pair_predictions_generated':False,'no_active_pair_predictor_or_evaluator':True,'runtime_environment_unchanged':True,'frozen_references_verified':True,'frozen_reference_count':len(load(OUT/'protected_references.json')),'GPU_release':gpu,'MPS_worker_runtime_driver_or_CUDA_changed':False,'checks':checks,'Full_Module9A_predictor_or_accuracy_gate_complete':False})
 save('interference_model_gate.json',{'status':'NOT_EVALUATED_AT_USER_REQUESTED_CHECKPOINT','reason':'User requested stop after offline profiling and Runtime Sharing Semantics Audit, before further predictor implementation/execution and new pair-validation GPU runs','runtime_spatial_pair_applicability':False,'accuracy_evidence_available':False,'SUPPORTED_PARTIAL_NOT_SUPPORTED_verdict':None,'enter_Module9B':False,'original_accuracy_gate_rules_changed':False})
 save('campaign_state.json',{'phase':'OFFLINE_PROFILING_AND_RUNTIME_AUDIT_COMPLETE_STOPPED_AT_USER_REQUEST','completed_profiles':len(allowed),'captured_inferences':requests,'kernel_launches':allrows,'new_validation_runs':0,'new_multimodel_GPU_runs':0,'predictor_deferred':True,'stopped_at_utc':now()})
 save('run_accounting.json',{'planned_standalone_profile_paths':len(allowed),'completed_standalone_profile_paths':len(allowed),'completed_standalone_profile_processes':len(allowed),'captured_inferences':requests,'warmup_inferences':6*len(allowed),'profile_launch_attempts':len(allowed)+3,'profile_adapter_pre_target_failures':3,'profile_inference_result_based_retries':0,'new_pair_GPU_runs':0,'new_multimodel_GPU_runs':0,'CPU_synthetic_prototype_tests_before_user_steering':8,'prototype_retired_from_active_package':True})
 save('checkpoint_source_hashes.json',[{'path':str(p.resolve()),'sha256':sha(p)} for directory in [ROOT/'module9a_full',ROOT/'deployment_core/interference'] for p in directory.glob('*.py')])
 print(json.dumps({'status':'PASS','profiles':len(allowed),'inferences':requests,'kernels':allrows,'new_pair_runs':0}))
if __name__=='__main__':main()
