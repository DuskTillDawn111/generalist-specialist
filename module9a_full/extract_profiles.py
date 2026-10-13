"""Nsight trace metadata, official isolated occupancy, and duration audit."""
import math,sqlite3,subprocess
from .common import *
from deployment_core.interference.task_path_profile import reconstruct

def extract(path):
 c=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
 strings={r['id']:r['value'] for r in c.execute('select * from StringIds')};ranges=[]
 for row in c.execute('select * from NVTX_EVENTS'):
  r=dict(row);label=r.get('text') or strings.get(r.get('textId'),'')
  if label.startswith('PILOT|') and '|target:' in label and r.get('end') is not None:ranges.append({'start':r['start'],'end':r['end'],'index':int(label.split('target:')[1])})
 assert len(ranges)==3,('missing request NVTX',path,len(ranges))
 tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")};launches={}
 for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
  if table in tables:
   for r in c.execute('select * from '+table):
    rr=dict(r);launches[rr['correlationId']]=min(launches.get(rr['correlationId'],rr['start']),rr['start'])
 rows=[]
 for row in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
  r=dict(row);request=next((v for v in ranges if v['start']<=r['start'] and r['end']<=v['end']),None)
  if request is None:continue
  grid=[r['gridX'],r['gridY'],r['gridZ']];block=[r['blockX'],r['blockY'],r['blockZ']]
  rows.append({'sample_index':request['index'],'kernel_name':strings[r['demangledName']],'registers_per_thread':r['registersPerThread'],'static_shared_bytes':r['staticSharedMemory'],'dynamic_shared_bytes':r['dynamicSharedMemory'],'block':block,'grid':grid,'threads_per_block':math.prod(block),'warps_per_block':math.ceil(math.prod(block)/32),'grid_blocks':math.prod(grid),'duration_profile_ns':r['end']-r['start'],'launch_start_profile_ns':launches.get(r['correlationId'],r['start']),'gpu_start_profile_ns':r['start'],'gpu_end_profile_ns':r['end'],'stream_id':r['streamId'],'context_id':r.get('contextId'),'trace_global_pid':r.get('globalPid'),'correlation_id':r['correlationId'],'shared_memory_limit_config':r.get('sharedMemoryLimitConfig'),'cluster_dimensions':[r.get('clusterX'),r.get('clusterY'),r.get('clusterZ')]})
 c.close();rows.sort(key=lambda r:(r['sample_index'],r['launch_start_profile_ns'],r['gpu_start_profile_ns'],r['correlation_id']))
 assert rows and all(r['duration_profile_ns']>0 and r['threads_per_block']>0 for r in rows)
 props=load(OUT/'hardware_metric_audit.json')['device_attributes'];names=['major','minor','max_threads_per_block','max_threads_per_sm','regs_per_block','regs_per_sm','warp_size','shared_per_block','shared_per_sm','sm_count','shared_optin_per_block','reserved_shared_per_block'];base=' '.join(str(props[k]) for k in names)
 configs=sorted({(r['threads_per_block'],r['registers_per_thread'],r['static_shared_bytes'],r['dynamic_shared_bytes']) for r in rows})
 out=subprocess.check_output([str(ROOT/'module9a1_task_conditioned_pilot/occupancy')],input='\n'.join(base+' '+' '.join(map(str,v)) for v in configs)+'\n',text=True).splitlines();assert len(out)==len(configs)
 lookup={}
 for cfg,line in zip(configs,out):
  rc,b,o,flags=line.split();assert int(rc)==0 and int(b)>0,(cfg,line)
  lookup[cfg]={'max_active_blocks_per_sm':int(b),'theoretical_occupancy':math.ceil(cfg[0]/props['warp_size'])*int(b)/(props['max_threads_per_sm']/props['warp_size']),'occupancy_API_printed_value':float(o),'occupancy_limiting_factors':int(flags),'occupancy_status':'NVIDIA_API_WITH_DISCLOSED_DEFAULT_FUNCTION_ATTRIBUTES'}
 for r in rows:r.update(lookup[(r['threads_per_block'],r['registers_per_thread'],r['static_shared_bytes'],r['dynamic_shared_bytes'])])
 return rows

def main():
 catalog=load(OUT/'catalog.json');audit=[];summary=[];p=load(OUT/'profile_protocol.json')
 if (OUT/'task_path_kernel_catalog.csv').exists():(OUT/'task_path_kernel_catalog.csv').unlink()
 for entry in catalog['paths']:
  m,k=entry['model'],entry['task'];tag=m+'__'+k;path=OUT/'raw_profiles'/(tag+'.sqlite');rows=extract(path);requests=[]
  for i in range(3):
   rr=[r for r in rows if r['sample_index']==i];assert rr
   durations=reconstruct([r['duration_profile_ns'] for r in rr],entry['standalone_s0_ms'])
   for j,(r,v) in enumerate(zip(rr,durations)):r['sequence_index']=j;r['duration_corrected_ms']=v
   intervals=sorted((r['gpu_start_profile_ns'],r['gpu_end_profile_ns']) for r in rr);prev=-1;overlaps=0
   for start,end in intervals:
    if start<prev:overlaps+=1
    prev=max(prev,end)
   requests.append({'sample_index':i,'sample_id':p['items'][k][i]['pool_id'],'kernel_count':len(rr),'kernels':rr,'intra_path_gpu_overlap_events':overlaps})
   summed=math.fsum(durations);assert math.isclose(summed,entry['standalone_s0_ms'],rel_tol=1e-13,abs_tol=1e-10)
   audit.append({'model':m,'task':k,'sample_index':i,'profile_kernel_sum_ms':sum(r['duration_profile_ns'] for r in rr)/1e6,'corrected_kernel_sum_ms':summed,'frozen_unprofiled_s0_ms':entry['standalone_s0_ms'],'absolute_anchor_error_ms':abs(summed-entry['standalone_s0_ms']),'exact_fsum_equality':summed==entry['standalone_s0_ms'],'intra_path_overlap_events':overlaps})
  result={'model':m,'task':k,'standalone_s0_ms':entry['standalone_s0_ms'],'metadata_complete':True,'requests':requests,'source':{'trace_path':str(path),'trace_sha256':sha(path),'profile_protocol_sha256':sha(OUT/('profile_protocol.json' if m not in ['convnext_tiny','distilbert','distil_whisper','minicpm'] else 'profile_protocol_before_paddle_adapter_repair.json')),'standalone_source':entry['s0_source'],'duration_method':'PROJECT_ENGINEERING_ADAPTATION','occupancy_assumptions':load(OUT/'hardware_metric_audit.json')['occupancy_assumptions']}}
  save('task_path_profiles/'+tag+'.json',result);entry['kernel_profile_status']='COMPLETE'
  print(json.dumps({'event':'EXTRACTED','path':tag,'kernels':len(rows)}),flush=True)
  summary.append({'model':m,'task':k,'profile_requests':3,'total_kernel_launches':len(rows),'unique_kernel_names':len({r['kernel_name'] for r in rows}),'kernel_counts':[r['kernel_count'] for r in requests],'complete_launch_metadata':True,'min_theoretical_occupancy':min(r['theoretical_occupancy'] for r in rows),'max_theoretical_occupancy':max(r['theoretical_occupancy'] for r in rows),'stream_ids':sorted({r['stream_id'] for r in rows}),'context_ids':sorted({r['context_id'] for r in rows if r['context_id'] is not None})})
  # Full launch catalog is streamed to disk; no diagnostic durations discarded.
  cols=['model','task','sample_index','sequence_index','kernel_name','registers_per_thread','static_shared_bytes','dynamic_shared_bytes','threads_per_block','warps_per_block','grid','block','max_active_blocks_per_sm','theoretical_occupancy','duration_profile_ns','duration_corrected_ms','stream_id','context_id','launch_start_profile_ns']
  import csv
  with (OUT/'task_path_kernel_catalog.csv').open('w' if not (OUT/'task_path_kernel_catalog.csv').exists() else 'a',newline='') as f:
   w=csv.DictWriter(f,fieldnames=cols)
   if f.tell()==0:w.writeheader()
   for r in rows:w.writerow({c:json.dumps(r[c]) if isinstance(r.get(c),(dict,list)) else m if c=='model' else k if c=='task' else r[c] for c in cols})
 save('catalog.json',catalog);csvsave('task_path_catalog.csv',catalog['paths']);csvsave('kernel_profile_summary.csv',summary)
 save('duration_reconstruction_audit.json',{'method':'PROJECT_ENGINEERING_ADAPTATION','Roomie_public_reconstruction_reproducible':False,'formula':'s0*profile_ns/sum(profile_ns)','final_entry_roundoff_correction':True,'ground_truth_s0_refit':False,'standalone_s0_source':'frozen unprofiled service_demand_v2_1','all_request_sums_match_s0':True,'audit':audit,'limitations':['total service includes CPU work, launch gaps, transfers; proportional allocation is not isolated GPU kernel duration','one independent profile process per path, three deterministic inputs; not a profile repeatability campaign','averaging across different input identities limited to these captured inputs'],'raw_vs_corrected_sensitivity':'NOT_RUN_USER_STOP_BEFORE_PAIR_PREDICTOR','outcome_based_weight_tuning':False})
 print(json.dumps({'complete_paths':len(summary),'total_kernels':sum(r['total_kernel_launches'] for r in summary)}))
if __name__=='__main__':main()
