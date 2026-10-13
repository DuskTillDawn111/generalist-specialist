import ctypes, math, os, subprocess
import numpy as np
from .common import *
def main():
 assert not (OUT/'profile_protocol.json').exists()
 admission=load(ROOT/'results/module8b_quality_pool_finalization/quality_admission_matrix.json')
 service=load(ROOT/'results/candidate_pool_v2_1/service_demand_v2_1.json')
 from module8c_final_real_search.common import inputs
 catalog,hw,roles,cal=inputs()
 allowed=sorted((r['model'],r['task']) for r in admission['rows'] if r['quality_admissible'] and r['capability_verified'])
 assert set(allowed)=={(m,k) for m,v in catalog.items() for k in v.supported_tasks}
 context=load(ROOT/'results/module8f2_resource_static/contexts/C01.json')
 old=load(ROOT/'results/module9a1_task_conditioned_pilot/measurement_protocol_freeze.json')
 seed=min(old['repetition_seeds']);pool=[json.loads(line) for line in Path(old['input_pool']).read_text().splitlines() if line.strip()];items={}
 for k in sorted({k for m,k in allowed}):
  by=[r for r in pool if r['task_type']==k];order=np.random.default_rng(seed).permutation(len(by))[:30];items[k]=[by[int(i)] for i in order]
 for k,rs in old['input_subsets'][str(seed)].items():assert items[k]==rs
 refs={}
 for directory in ['module8f2_resource_static','module8f3_structural_dominance','module9a1_task_conditioned_pilot','module9a2_repeatability']:
  for p in sorted((ROOT/'results'/directory).rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:refs[str(p)]={'path':str(p),'sha256':sha(p)}
 for directory in ['method_v2','module8d_real_baselines','deployment_core']:
  for p in (ROOT/directory).rglob('*.py'):
   if 'interference' not in p.parts:refs[str(p)]={'path':str(p),'sha256':sha(p)}
 save('protected_references.json',list(refs.values()))
 factor=load(ROOT/'results/module8f1a_memory_breakpoints/memory_safety_factor.json')['safety_factor']
 rows=[]
 for m,k in allowed:
  a=next(r for r in admission['rows'] if (r['model'],r['task'])==(m,k));rows.append({'model':m,'task':k,'role':roles[m],'admitted_quality':a['measured_quality'],'quality_metric':a['quality_metric'],'quality_threshold':a['historical_threshold'],'standalone_s0_ms':service['models'][m]['service_time_ms'][k],'s0_source':service['models'][m]['service_time_sources'][k],'safe_memory_mib_by_replicas':{n:factor*v for n,v in cal.memory_table_mib[m].items()},'g_m_by_replicas':cal.capacity_factors[m],'g_m_scope':cal.model_evidence[m].get('capacity_scope','single replica only'),'profiling_availability':'NSYS_CUDA_NVTX','kernel_profile_status':'PENDING'})
 csvsave('task_path_catalog.csv',rows)
 save('catalog.json',{'paths':rows,'capacity_factors':cal.capacity_factors,'roles':roles,'count_reconstructed_from_admission':len(rows),'memory_factor':factor})
 lib=ctypes.CDLL('libcuda.so.1');assert lib.cuInit(0)==0
 ids={'major':75,'minor':76,'max_threads_per_block':1,'max_threads_per_sm':39,'regs_per_block':12,'regs_per_sm':82,'warp_size':10,'shared_per_block':8,'shared_per_sm':81,'sm_count':16,'shared_optin_per_block':97,'reserved_shared_per_block':111,'max_blocks_per_sm':106,'concurrent_kernels':31};attrs={}
 for n,k in ids.items():
  v=ctypes.c_int();rc=lib.cuDeviceGetAttribute(ctypes.byref(v),k,0);assert rc==0,(n,rc);attrs[n]=v.value
 probes=[]
 for binary in ['/usr/bin/ncu','/opt/nvidia/nsight-compute/2026.1.1/ncu']:
  r=subprocess.run([binary,'--query-metrics','--query-metrics-mode','base'],capture_output=True,text=True);probes.append({'binary':binary,'exit_code':r.returncode,'output':(r.stdout+r.stderr)[:5000]})
 save('hardware_metric_audit.json',{'device_attributes':attrs,'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'nsys_version':subprocess.check_output(['nsys','--version'],text=True).strip(),'nvcc_version':subprocess.check_output(['nvcc','--version'],text=True).strip(),'counter_probes':probes,'metadata_source':'CUPTI_ACTIVITY_KIND_KERNEL via Nsight Systems SQLite','available':['registers/thread','static shared','dynamic shared','threads/block','grid dimensions','block dimensions','name','launch order','occupancy from installed NVIDIA standalone occupancy API'],'not_observed':['achieved occupancy','hardware SM/DRAM/L2 counters','function maxThreadsPerBlock','exact block barrier count','runtime shared cache carveout','internal context scheduling quantum'],'occupancy_assumptions':['function maxThreadsPerBlock=device max','barriers=1 (CUDA runtime conversion default)','opt-in shared-memory permitted','default carveout'],'stack_changed':False,'counter_permissions_changed':False,'audited_at_utc':now()})
 protocol={'frozen_at_utc':now(),'paths':[list(p) for p in allowed],'registry':context['registry'],'items':items,'sample_seed':seed,'warmup_rule':'six exact requests per path: first three sample IDs repeated twice; separate from captured samples','profile_samples':3,'ground_truth_s0':'frozen unprofiled service_demand_v2_1; no profile timing refit','duration_method':'PROJECT_ENGINEERING_ADAPTATION','duration_formula':'corrected_k=s0_ms*profile_ns/sum(profile_ns); last duration receives floating-point residual','environment':{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','LD_LIBRARY_PATH':load(ROOT/'results/module8c_final_real_search/startup_recovery.json')['runtime_environment_overrides']['LD_LIBRARY_PATH']},'GPU':load(ROOT/'results/module9a2_repeatability/protocol_identity.json')['GPU'],'no_validation_GPU_results':True,'same_model_interference':False,'profiling_source_sha256':{str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}}
 save('profile_protocol.json',protocol)
 save('campaign_state.json',{'phase':'PROFILES_PREPARED','completed_profiles':0,'new_validation_runs':0})
 print(json.dumps({'paths':len(allowed),'attrs':attrs}))
if __name__=='__main__':main()
