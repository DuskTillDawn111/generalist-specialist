"""Add only two real standalone values; validate each raw measurement and gate D."""
import copy,math,statistics
from .common import *
def main():
 original=read(SERVICE_OLD);catalog=copy.deepcopy(original)
 pool=read(POOL_OUT/'candidate_pool_v2_1.json')
 targets=[json.loads(l) for l in TASK_POOL.read_text().splitlines() if l and json.loads(l)['task_type']=='visual_qa'][:30]
 expected_ids=[r['pool_id'] for r in targets]
 validations=[];pending=[]
 for name in ('phi4mm','minicpm'):
  reps=[];all_latencies=[]
  for rep in range(1,4):
   path=POOL_OUT/f'vqa_new_path_raw_measurements/r{rep}/active_single_{name}_visual_qa_n30.jsonl'
   vp=Path(str(path)+'.validation.json');sp=Path(str(path)+'.summary.json')
   if not vp.exists() or not path.exists() or not sp.exists():
    pending.append({'model':name,'repetition':rep,'reason':'missing raw/summary/validation'});continue
   validation=read(vp);summary=read(sp)
   if validation['status']!='measured':
    pending.append({'model':name,'repetition':rep,'reason':'failed measurement','evidence':ref(vp)});continue
   rows=[json.loads(l) for l in path.read_text().splitlines() if l]
   errors=[]
   if not (validation['standalone_only'] and validation['replica_count']==1 and validation['workers_started']==1):errors.append('non-standalone')
   if validation['foreign_GPU_processes'] or validation.get('monitor_error'):errors.append('isolation failure')
   if not validation['GPU_memory_released'] or validation['shutdown']!={'exit_code':0,'forced':False}:errors.append('unload failure')
   if len(rows)!=30 or [r['pool_id'] for r in rows]!=expected_ids:errors.append('wrong target IDs')
   if summary['protocol']!='active_single_profile_v1' or summary['target_count']!=30:errors.append('wrong protocol')
   requests=validation['requests']
   if len(requests)!=31 or requests[0]['request_id']!=-1 or any(r['request_id']!=i for i,r in enumerate(requests[1:])):errors.append('warmup/submission policy')
   if requests[0]['sample']!=targets[0]['sample'] or any(r['sample']!=t['sample'] for r,t in zip(requests[1:],targets)):errors.append('sample/preprocessing inputs changed')
   if any(r['worker_latency_ms']<=0 or not math.isclose((r['end_ns']-r['start_ns'])/1e6,r['worker_latency_ms'],abs_tol=1e-12) for r in rows):errors.append('worker timer mismatch')
   if not validation['load_succeeded']:errors.append('load failed')
   if any(e['kind']=='ERROR' for e in validation['events']):errors.append('worker ERROR')
   if validation['original_profiler']['sha256']!=sha(ROOT/'joint_deploy/run_active_single_profile.py'):errors.append('Stage-1 code changed')
   latencies=[r['worker_latency_ms'] for r in rows]
   if not math.isclose(statistics.mean(latencies),summary['mean_worker_latency_ms'],abs_tol=1e-9):errors.append('raw/summary mean mismatch')
   assert not errors,(name,rep,errors)
   all_latencies+=latencies
   reps.append({'repetition':rep,'measurement_count':30,'mean_worker_latency_ms':statistics.mean(latencies),
                'sample_std_ms':statistics.stdev(latencies),'raw':ref(path),'summary':ref(sp),'validation':ref(vp)})
   validations.append({'model':name,'repetition':rep,'status':'PASS','target_ids_match':True,'one_warmup_excluded':True,
                       'standalone_isolation_verified':True,'memory_released':True,'raw_summary_mean_match':True,
                       'valid_closed_set_outputs':sum(r['valid_closed_set_output'] for r in rows),
                       'correct':sum(r['correct'] for r in rows),'quality_note':'profiling correctness is raw evidence, not a replacement for 200-sample quality audit'})
  if len(reps)==3:
   s0=sum(r['mean_worker_latency_ms']*r['measurement_count'] for r in reps)/sum(r['measurement_count'] for r in reps)
   catalog['models'][name]['capabilities_verified'].append('visual_qa')
   catalog['models'][name]['service_time_ms']['visual_qa']=s0
   catalog['models'][name]['service_time_sources']['visual_qa']={'kind':'new_real_standalone_measurement','protocol':'active_single_profile_v1',
                  'value_ms':s0,'measurement_count':90,'aggregation':'sample-count-weighted mean','sample_std_ms':statistics.stdev(all_latencies),
                  'repetitions':reps,'source_revision':next(m['source_revision'] for m in pool['models'] if m['id']==name),
                  'adapter':pool['versioned_adapters'][name],'standalone_protocol':ref(POOL_OUT/'standalone_protocol.json')}
 required=[{'model_id':m['id'],'task':k} for m in pool['models'] if m['id'] in pool['formal_model_ids'] for k in m['capabilities_verified']]
 available=[p for p in required if p['task'] in catalog['models'][p['model_id']]['service_time_ms']]
 missing=[p for p in required if p not in available]
 for name,model in original['models'].items():
  for task,value in model['service_time_ms'].items():
   assert catalog['models'][name]['service_time_ms'][task]==value
   assert catalog['models'][name]['service_time_sources'][task]==model['service_time_sources'][task]
 completeness={'required_paths':len(required),'available_paths':len(available),'still_pending_paths':len(missing),'pending_paths':missing,
               'completeness_ratio':len(available)/len(required),'required_path_list':required,
               'measured_paths':len(available),'new_measured_paths':len(available)-25,'existing_paths':25,
               'new_measurement_failures':pending}
 assert len(required)==27
 catalog.update(pool_version='2.1',candidate_pool=ref(POOL_OUT/'candidate_pool_v2_1.json'),base_service_catalog=ref(SERVICE_OLD),
                status='complete' if not missing and not pending else 'pending',frozen=not missing and not pending,completeness=completeness,
                existing_service_demands_modified=0,new_paths_measured=len(available)-25)
 validation={'status':'PASS' if not missing and not pending else 'PENDING','original_25_values_unchanged':True,'original_25_provenance_unchanged':True,
             'repetitions':validations,'pending':pending,'no_synthetic_fallback':True,'protocol':ref(POOL_OUT/'standalone_protocol.json'),
             'new_vqa_s0_ms':{m:catalog['models'][m]['service_time_ms'].get('visual_qa') for m in ('phi4mm','minicpm')},
             'part_D_allowed':not missing and not pending}
 for folder in (POOL_OUT,OUT):
  write(folder/'service_demand_v2_1.json',catalog,freeze=catalog['frozen'])
  write(folder/'vqa_service_demand_validation.json',validation,freeze=validation['status']=='PASS')
  write(folder/'completeness.json',completeness,freeze=completeness['completeness_ratio']==1)
 print('Service-demand completeness',len(available),'/',len(required),'new VQA s0 ms',validation['new_vqa_s0_ms'])
 if not validation['part_D_allowed']:raise RuntimeError('Required new measurement pending; no Part D')
if __name__=='__main__':main()
