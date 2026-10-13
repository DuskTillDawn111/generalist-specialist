"""Read-only runtime semantics audit; no pair predictor or new concurrency run."""
import ctypes,os,sqlite3,subprocess
from collections import Counter
from .common import *

def processes():
 found=[]
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  try:
   comm=(path/'comm').read_text().strip()
   if comm.startswith('nvidia-cuda-mps'):
    found.append({'pid':int(path.name),'comm':comm,'command':(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')})
  except (OSError,PermissionError):continue
 return found

def main():
 profiles=load(OUT/'profile_protocol.json');hardware=load(OUT/'hardware_metric_audit.json');lib=ctypes.CDLL('libcuda.so.1');assert lib.cuInit(0)==0;attrs={}
 for n,k in [('compute_preemption_supported',90),('stream_priorities_supported',78),('compute_mode',20),('kernel_exec_timeout',17),('concurrent_kernels',31)]:
  v=ctypes.c_int();rc=lib.cuDeviceGetAttribute(ctypes.byref(v),k,0);attrs[n]={'query_rc':rc,'value':v.value if rc==0 else None}
 version=ctypes.c_int();assert lib.cuDriverGetVersion(ctypes.byref(version))==0;attrs['driver_API_version']=version.value
 trace_evidence=[]
 for m,k in profiles['paths']:
  tag=m+'__'+k;path=OUT/'raw_profiles'/(tag+'.sqlite');raw=load(OUT/'raw_profiles'/(tag+'.json'));assert raw['status']=='COMPLETE' and raw['gpu_release']['verified'];c=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
  context=[dict(r) for r in c.execute('select * from TARGET_INFO_CUDA_CONTEXT_INFO')];streams=[dict(r) for r in c.execute('select * from TARGET_INFO_CUDA_STREAM')];active=list(c.execute('select contextId,streamId,count(*) as launches from CUPTI_ACTIVITY_KIND_KERNEL group by contextId,streamId'))
  used=[]
  for r in active:
   v=next((s for s in streams if (s['contextId'],s['streamId'])==(r['contextId'],r['streamId'])),None);priority=v['priority'] if v else None
   if priority is not None and priority>=2**31:priority-=2**32
   ctx=next((v for v in context if v['contextId']==r['contextId']),None);default=ctx and ctx['nullStreamId']==r['streamId']
   used.append({'context_id':r['contextId'],'stream_id':r['streamId'],'priority_signed':priority,'launches':r['launches'],'is_default_stream':bool(default)})
  runtime_tables=[n for n in tables if 'CUDA' in n and ('VERSION' in n or 'PROCESS' in n)];versions={n:[dict(r) for r in c.execute('select * from '+n)] for n in runtime_tables}
  trace_evidence.append({'model':m,'task':k,'worker_pid':raw['ready']['pid'],'raw_path':str(OUT/'raw_profiles'/(tag+'.json')),'trace_path':str(path),'trace_sha256':sha(path),'CUDA_contexts':context,'used_kernel_streams':used,'read_only_driver_probe':raw['ready'].get('runtime_sharing_semantics'),'runtime_metadata':versions,'capture_kind':'standalone_offline_only','profile_process_repetitions':1,'profile_requests':3});c.close()
 # Existing unprofiled pair records verify independent live worker processes and
 # host handler overlap. No assertion of concurrent SM execution follows.
 historical=[];legacy=ROOT/'results/module9a2_repeatability'
 for key in ['qwen__asr__whisper-small','qwen__visual_qa__whisper-small','qwen__lid__paddleocr']:
  rawpath=legacy/'raw_runs'/(key+'__rep1.json');r=load(rawpath);ready=r['ready'];pids={m:v['pid'] for m,v in ready.items()};historical.append({'state':key,'source_path':str(rawpath),'source_sha256':sha(rawpath),'status':r['status'],'worker_pids':pids,'distinct_processes':len(set(pids.values()))==len(pids),'minimum_host_worker_interval_overlap':min(v['active_overlap_ratio'] for v in r['target_records']),'overlap_means':'overlap of wall-clock request handler intervals; not simultaneous GPU instruction execution','new_measurement':False})
 # There is no public scheduler configuration in these unchanged native workers.
 registry=profiles['registry'];source_evidence=[];needles=['cudaStreamCreateWithPriority','torch.cuda.Stream(','torch.cuda.set_stream','CUDA_MPS','cuCtxCreate','cuCtxSetCurrent','set_stream(','stream_guard(']
 for m,spec in registry.items():
  path=Path(spec['script']);text=path.read_text();source_evidence.append({'model':m,'source':str(path),'sha256':sha(path),'explicit_sharing_controls_found':[n for n in needles if n in text]})
 workerpath=ROOT/'joint_deploy/worker_protocol.py';text=workerpath.read_text();source_evidence.append({'model':'worker_protocol','source':str(workerpath),'sha256':sha(workerpath),'serialized_line_dispatch':True,'handler_wall_clock_service_time':True,'inter_process_GPU_global_scheduler_configured':False})
 mps=processes();ports=[{'path':path,'exists':Path(path).exists()} for path in ['/tmp/nvidia-mps','/var/run/nvidia-mps']];assert not mps,'Unexpected MPS process; preserve evidence and investigate without stack changes'
 mechanisms=[
 {'mechanism':'per-kernel profiling','status':'APPLICABLE','scope':'standalone metadata/ordered task paths'},
 {'mechanism':'theoretical occupancy o=Wb*b/W','status':'APPLICABLE_STANDALONE_ONLY','scope':'isolated launch resource capacity, not achieved occupancy or wall-clock pair slowdown'},
 {'mechanism':'subtract simultaneous co-kernel SM resources to compute adjusted occupancy','status':'NOT_APPLICABLE_CROSS_CONTEXT','scope':'no simultaneous compute kernels across the independent contexts in current no-MPS runtime'},
 {'mechanism':'occupancy slowdown d_e=d*o/o_tilde','status':'NOT_JUSTIFIED_FOR_THIS_RUNTIME','scope':'cross-context waiting/preemption is not reduced same-SM co-resident warp occupancy'},
 {'mechanism':'shorter kernel completes, survivor restores full occupancy','status':'NOT_APPLICABLE_AS_CURRENT_PAIR_EXECUTION_MODEL','scope':'no concurrent cross-context kernel phase; preemption/time slicing can occur before kernel completion'},
 {'mechanism':'temporal kernel alignment','status':'APPLICABLE_AS_TRACE_DESCRIPTOR_ONLY','scope':'spatial co-residency alignment simulation lacks physical applicability; a future time-sharing event model would require separate justification'},
 {'mechanism':'pairwise approximation','status':'REUSABLE_MODELING_PRINCIPLE_UNVALIDATED','scope':'may approximate interference for a justified execution model; does not validate current occupancy dynamics'},
 {'mechanism':'median across alignments','status':'REUSABLE_STATISTICAL_AGGREGATOR_UNVALIDATED','scope':'requires meaningful scenario/event model first'},
 {'mechanism':'gamma=max(q_p/q_q,1)','status':'DOCUMENTED_ROOMIE_HEURISTIC_NOT_CURRENT_RUNTIME_LAW','scope':'kernel-count ratio does not identify context scheduling time share, CPU gaps, or preemption quantum'},
 {'mechanism':'equal spatial partition across colocated process kernels','status':'EXCLUDED','scope':'would assume an unavailable sharing mode'},
 {'mechanism':'priority-based cross-model stream scheduler','status':'NOT_ESTABLISHED','scope':'active stream priorities are recorded per path; no cross-model priority policy is configured or inferred'},
 {'mechanism':'FCFS/nonpreemptive whole-kernel alternation','status':'NOT_ESTABLISHED','scope':'no evidence for a queue quantum, strict FCFS, or switching only at kernel boundaries'},
 ]
 active_priorities=sorted({v['priority_signed'] for r in trace_evidence for v in r['used_kernel_streams'] if v['priority_signed'] is not None});green=any(v.get('isGreenContext') for r in trace_evidence for v in r['CUDA_contexts']);contextcounts=[len(r['CUDA_contexts']) for r in trace_evidence]
 payload={'status':'AUDIT_COMPLETE_PREDICTOR_DECISION_DEFERRED','audited_at_utc':now(),'runtime_sharing_class':'INDEPENDENT_PROCESS_INDEPENDENT_CUDA_CONTEXT_NO_MPS_TEMPORAL_SHARING','evidence_strength':{'independent_worker_processes':'DIRECT_EXISTING_PAIR_RECORDS_AND_CURRENT_STANDALONE_PIDS','per_process_CUDA_contexts':'DIRECT_NSYS_CONTEXT_METADATA_AND_READ_ONLY_DRIVER_PROBES','MPS_absence':'DIRECT_PROCESS_FILESYSTEM_ENVIRONMENT_SNAPSHOT','cross_context_compute_concurrency':'NVIDIA_DOCUMENTED_CONTRACT_APPLIED_TO_OBSERVED_RUNTIME','exact_driver_context_schedule':'NOT_MEASURED_OR_IDENTIFIED'},'hardware':{'name':'NVIDIA GeForce RTX 4090','driver':profiles['GPU']['driver'],'capacity_mib':profiles['GPU']['total_mib'],'toolkit':hardware['nvcc_version'],'device_attributes':attrs},'MPS':{'enabled':False,'server_processes':mps,'standard_socket_directories':ports,'current_environment':{k:v for k,v in os.environ.items() if k.startswith('CUDA_MPS_')},'enabled_by_this_module':False},'context_structure':{'single_context_per_profile':all(n==1 for n in contextcounts),'context_counts':contextcounts,'green_contexts_observed':green,'per_worker_process_CUDA_context':True,'numeric_context_id_caveat':'IDs/handles must be associated with process and trace; coincident numeric IDs in separate reports do not prove shared context'},'stream_structure':{'active_kernel_priorities_signed':active_priorities,'explicit_project_priority_policy_found':any(r['explicit_sharing_controls_found'] for r in source_evidence if r['model']!='worker_protocol'),'profiler_only_streams_not_used_as_worker_policy_evidence':True,'default_stream_probe_available_paths':sum(r['read_only_driver_probe'] is not None for r in trace_evidence)},'trace_evidence':trace_evidence,'historical_pair_process_evidence':historical,'worker_source_evidence':source_evidence,'mechanism_applicability':mechanisms,'known_semantics':['distinct processes and contexts share compute temporally, not through simultaneous SM kernel co-residency in the observed no-MPS setup','host worker requests may overlap while GPU compute time-slices','compute preemption capability is present; this does not measure actual preemption frequency','allocated VRAM remains a residency constraint distinct from SM register/shared-memory occupancy'],'unknown_semantics':['context scheduling quantum','actual fairness/time share','FCFS vs round-robin or other ordering','preemption frequency and actual context-switch overhead','cross-context cache state effects and transfer overlap','unprofiled per-kernel isolated time and CPU/launch-gap decomposition'],'selected_scheduler_policy':None,'selected_pair_predictor':None,'occupancy_pair_applicable':False,'ROOMIE_SPATIAL_PAIR_MODEL_APPLICABLE_TO_CURRENT_RUNTIME':False,'offline_kernel_catalog_retained':True,'new_pair_validation_GPU_runs':0,'new_multi_model_GPU_runs':0,'worker_runtime_driver_CUDA_or_MPS_modified':False,'user_stop_boundary_enforced':True,'source_links':{'CUDA_context_concurrency':'https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#concurrent-kernel-execution','MPS':'https://docs.nvidia.com/deploy/mps/when-to-use-mps.html','driver_context_API':'https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__CTX.html','stream_API':'https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__STREAM.html','Roomie':'https://arxiv.org/html/2607.16784v1'}}
 save('runtime_sharing_semantics_audit.json',payload)
 csvsave('roomie_mechanism_applicability.csv',mechanisms)
 save('scheduler_policy_audit.json',{'status':'DEFERRED_AFTER_RUNTIME_SHARING_SEMANTICS_AUDIT','selected_policy':None,'physically_established_semantics':'temporal sharing across independent contexts; exact policy unknown','equal_spatial_partition_allowed':False,'FCFS_or_whole_kernel_alternation_confirmed':False,'priority_policy_confirmed':False,'no_policy_selection_using_GPU_pair_outcomes':True,'uncertainty':payload['unknown_semantics'],'source':'runtime_sharing_semantics_audit.json'})
 print(json.dumps({'status':payload['status'],'profile_paths':len(trace_evidence),'active_priorities':active_priorities,'context_counts':contextcounts,'spatial_pair_model_applicable':False,'selected_policy':None,'pair_runs':0}))
if __name__=='__main__':main()
