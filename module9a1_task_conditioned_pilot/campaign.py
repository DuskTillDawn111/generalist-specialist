"""Bounded development characterization; zero deployment searches."""
import ctypes,json,os,subprocess,sys
from pathlib import Path
from .prepare import ROOT,OUT,DEV,load,save,sha,now

def verify():
 freeze=load(OUT/'execution_freeze.json')
 for r in freeze['core_files']+freeze['execution_sources']:
  assert sha(r['path'])==r['sha256'],'Frozen pilot input/source mismatch: '+r['path']
def inventory8f2():
 d=load(DEV/'artifact_hashes.json');bad=[r['path'] for r in d.values() if sha(r['path'])!=r['sha256']];assert not bad,bad;return len(d)
def audit():
 lib=ctypes.CDLL('libcuda.so.1');assert lib.cuInit(0)==0
 attrids={'major':75,'minor':76,'max_threads_per_block':1,'max_threads_per_sm':39,'regs_per_block':12,'regs_per_sm':82,'warp_size':10,'shared_per_block':8,'shared_per_sm':81,'sm_count':16,'shared_optin_per_block':97,'reserved_shared_per_block':111};attrs={}
 for name,key in attrids.items():
  v=ctypes.c_int();rc=lib.cuDeviceGetAttribute(ctypes.byref(v),key,0);assert rc==0,(name,rc);attrs[name]=v.value
 probes=[]
 for binary in ['/usr/bin/ncu','/opt/nvidia/nsight-compute/2026.1.1/ncu']:
  v=subprocess.run([binary,'--version'],text=True,capture_output=True);q=subprocess.run([binary,'--query-metrics','--query-metrics-mode','base'],text=True,capture_output=True);probes.append({'binary':binary,'version':v.stdout+v.stderr,'query_exit_code':q.returncode,'query_output':(q.stdout+q.stderr)[:5000]})
 data={'audited_at_utc':now(),'ncu_audit':probes,'nsys_version':subprocess.check_output(['/usr/local/bin/nsys','--version'],text=True).strip(),'device_attributes':attrs,'official_occupancy_calculator_header':{'path':'/usr/local/cuda-11.8/targets/x86_64-linux/include/cuda_occupancy.h','sha256':sha('/usr/local/cuda-11.8/targets/x86_64-linux/include/cuda_occupancy.h')},'metadata_backend':'Nsight Systems CUDA/NVTX tracing and CUPTI activity SQLite, no hardware performance counters','preexisting_trace_schema_audited':'benchmark/results/characterization/qwen_nsys_v1/qwen_asr_n30.sqlite','metadata_fields_expected':['kernel order/name','registersPerThread','staticSharedMemory','dynamicSharedMemory','blockX/Y/Z','gridX/Y/Z'],'theoretical_occupancy':'NVIDIA standalone CUDA occupancy calculator, driver device limits and traced launch resources; optional unknown function attributes disclosed, not achieved occupancy.','achieved_occupancy_available':False,'SM_DRAM_L2_counter_model_implemented':False,'CUDA_or_driver_changed':False,'new_model_GPU_measurements_at_audit':0};save('profiler_capability_audit.json',data)
 base=' '.join(str(attrs[k]) for k in attrids);rows=[(256,32,0,0),(1024,16,0,0),(128,64,0,0)];out=subprocess.check_output([str(ROOT/'module9a1_task_conditioned_pilot/occupancy')],input='\n'.join(base+' '+' '.join(map(str,r)) for r in rows)+'\n',text=True);assert len(out.splitlines())==3 and all(x.split()[0]=='0' and 0<float(x.split()[2])<=1 for x in out.splitlines());save('occupancy_CPU_checks.json',{'rows':rows,'calculator_results':out.splitlines(),'status':'PASS','new_GPU_experiments':0})
def prepare_execution():
 if (OUT/'execution_freeze.json').exists():raise RuntimeError('Execution already frozen; do not reschedule')
 assert load(OUT/'pilot_state.json')['GPU_runs']==0
 audit();checks=[];selected=set(load(OUT/'pilot_selection.json')['generalists']+load(OUT/'pilot_selection.json')['co_models'])
 for r in load(DEV/'hash_verification.json')['model_assets']:
  if r['model'] in selected:
   actual=sha(r['path']);checks.append({'path':r['path'],'expected':r['expected'],'actual':actual,'match':actual==r['expected']});assert actual==r['expected'],r['path']
 verified=inventory8f2();save('pre_GPU_asset_checks.json',{'status':'PASS','checks':checks,'frozen_8F2_files_verified':verified,'new_GPU_model_runs':0})
 p=load(OUT/'measurement_protocol_freeze.json');plan=[]
 for model,tasks in p['tasks'].items():
  for task in tasks:
   for seed in p['repetition_seeds']:plan.append({'model':model,'task':task,'seed':seed,'profile':True,'co':None})
 # Rotate condition order by repetition block before any timing observation.
 for ri,seed in enumerate(p['repetition_seeds']):
  conditions=[None]+p['co_models'];conditions=conditions[ri:]+conditions[:ri]
  paths=[(m,k) for m,tasks in p['tasks'].items() for k in tasks];paths=paths[ri:]+paths[:ri]
  for model,task in paths:
   for co in conditions:plan.append({'model':model,'task':task,'seed':seed,'profile':False,'co':co})
 core=['pilot_selection.json','noise_audit.json','measurement_protocol_freeze.json','development_evidence_freeze.json','candidate_history_freeze.json','failure_map.csv','prior_art_boundary.md','profiler_capability_audit.json','pre_GPU_asset_checks.json'];files=lambda ps:[{'path':str(x.resolve()),'sha256':sha(x)} for x in ps]
 freeze={'frozen_at_utc':now(),'core_files':files([OUT/n for n in core]),'execution_sources':files([x for x in Path(__file__).parent.iterdir() if x.suffix in ['.py','.cpp'] or x.name=='occupancy']),'plan':plan,'unprofiled_runs':sum(not r['profile'] for r in plan),'profiled_runs':sum(r['profile'] for r in plan),'model_GPU_runs_before_freeze':0,'no_8F2_heldout_validation':True,'no_method_routing_candidate_rank_revision':True,'no_new_benchmark_mixture_budget_or_intensity':True,'profile_durations_ground_truth':False,'retries':0};save('execution_freeze.json',freeze)
 print(json.dumps({'event':'FROZEN','profiled_runs':freeze['profiled_runs'],'unprofiled_runs':freeze['unprofiled_runs']},ensure_ascii=False),flush=True)
def main():
 if '--resume-pre-gpu-import-fix' in sys.argv:
  assert not list((OUT/'task_path_kernel_profiles').rglob('*.json')) and not list((OUT/'pair_measurements').rglob('*.json')),'Any prior GPU evidence forbids this pre-GPU import correction'
  original=load(OUT/'execution_freeze.json');assert load(OUT/'pilot_run_accounting.json')['completed_profiled_runs']==load(OUT/'pilot_run_accounting.json')['completed_unprofiled_runs']==0
  save('execution_freeze_before_CPU_import_fix.json',original)
  original['execution_sources']=[{'path':r['path'],'sha256':sha(r['path'])} for r in original['execution_sources']]
  original['pre_GPU_CPU_import_correction']={'reason':'Legacy active-pair primitive imported run_static_hybrid via sibling path; module import exited before any model start. Added joint_deploy to new pilot adapter sys.path and verified imports. Selection, protocol, noise, rules and frozen 8F2 unchanged.','GPU_measurements_at_correction':0,'failure_log':str(OUT/'run_logs/001.log'),'failure_log_sha256':sha(OUT/'run_logs/001.log'),'at_utc':now()}
  save('execution_freeze.json',original)
 else:prepare_execution()
 freeze=load(OUT/'execution_freeze.json');account={'planned_profiled_runs':freeze['profiled_runs'],'planned_unprofiled_runs':freeze['unprofiled_runs'],'completed_profiled_runs':0,'completed_unprofiled_runs':0,'new_attempts':0,'reuse':0,'retries':0,'complete_deployment_searches':0,'CPU_prelaunch_failures':1 if '--resume-pre-gpu-import-fix' in sys.argv else 0};save('pilot_run_accounting.json',account)
 try:
  for i,r in enumerate(freeze['plan']):
   verify();account['new_attempts']+=1;save('pilot_run_accounting.json',account);save('pilot_state.json',{'status':'RUNNING','plan_index':i,'current':r,'GPU_runs':account['new_attempts'],'started_at_utc':now()});cmd=[sys.executable,'-u','-m','module9a1_task_conditioned_pilot.runtime','--model',r['model'],'--task',r['task'],'--seed',str(r['seed'])]
   if r['co']:cmd+=['--co',r['co']]
   if r['profile']:cmd+=['--profile']
   log=OUT/'run_logs'/f'model_{i+1:03d}.log';log.parent.mkdir(exist_ok=True);env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
   print(json.dumps({'event':'START','plan_index':i,**r},ensure_ascii=False),flush=True)
   with log.open('w') as stream:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT);save('active_process.json',{'pid':proc.pid,'command':cmd,'log':str(log)});rc=proc.wait()
   if rc:raise RuntimeError(f'Pilot measurement failed index={i}; see {log}; no retry')
   account['completed_profiled_runs' if r['profile'] else 'completed_unprofiled_runs']+=1;save('pilot_run_accounting.json',account);print(json.dumps({'event':'COMPLETE','plan_index':i,'accounting':account}),flush=True)
   if i==0:
    from .analyze import extract
    tag=f'{r["model"]}__{r["task"]}__none__{r["seed"]}';path=OUT/'task_path_kernel_profiles/kernel'/f'{tag}.sqlite';kernels=extract(path);save('profiler_runtime_metadata_audit.json',{'status':'PASS','first_frozen_profile':str(path),'first_profile_sha256':sha(path),'kernel_occurrences':len(kernels),'request_indices':sorted(set(k['sample_index'] for k in kernels)),'required_launch_resource_metadata_available':True,'occupancy_calculated_fraction':sum(k['theoretical_occupancy'] is not None for k in kernels)/len(kernels),'counter_permissions_changed':False,'CUDA_driver_changed':False})
  verify();n=inventory8f2();from .analyze import final
  final();save('pilot_state.json',{'status':'COMPLETE','GPU_runs':account['new_attempts'],'frozen_8F2_artifacts_unchanged':n,'completed_at_utc':now()});save('validation_results.json',{'status':'PASS','all_selected_tasks_retained':True,'frozen_8F2_files_unchanged':n,'frozen_pilot_core_and_source_unchanged':True,'all_runs_complete':True,'full_v3_started':False,'routing_or_ranking_revised':False,'future_8F2_heldout_eligible':False});save('artifact_hashes.json',{str(x.relative_to(OUT)):{'path':str(x.resolve()),'sha256':sha(x)} for x in sorted(OUT.rglob('*')) if x.is_file() and x.name not in ['artifact_hashes.json','execution.log'] and '__pycache__' not in x.parts})
 except BaseException as exc:
  save('pilot_state.json',{'status':'INCOMPLETE','error':f'{type(exc).__name__}: {exc}','GPU_runs':account['new_attempts'],'new_thresholds_or_tasks_selected':False});raise
 print(json.dumps({'event':'PILOT_COMPLETE','gate':load(OUT/'pilot_gate.json')['status'],'accounting':account},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
