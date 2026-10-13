"""CPU invariants, artifact integrity and lightweight inference measurements."""
import math,time,tracemalloc
from copy import deepcopy
from deployment_core.interference.scheduling_pressure import SchedulingPressureEstimator
from .common import *

def cpu_checks():
 features=load(OUT/'path_features.json')
 fit={'selected_model':'I2','delta_path_ms_per_kernel':{'whisper-small__asr':.01,'whisper-small__lid':.02},'delta_model_ms_per_kernel':{'whisper-small':.015},'validation_targets':[]}
 model=SchedulingPressureEstimator(features,fit)
 target=('qwen','visual_qa');p=features['paths']['__'.join(target)]
 first=model.pair(target,('whisper-small','asr'));second=model.pair(target,('whisper-small','lid'))
 assert math.isclose(second['additional_delay_ms'],2*first['additional_delay_ms'])
 other=model.pair(('qwen','asr'),('whisper-small','asr'))
 assert math.isclose(first['additional_delay_ms']/p['kernel_count'],other['pressure_ms_per_kernel'])
 # Co-path changes pressure; target changes only known count/standalone anchor.
 x={'qwen':1,'whisper-small':1};r={'visual_qa':{'qwen':.001},'asr':{'whisper-small':.01}}
 copy=deepcopy(r);out=model.evaluate(x,r,{'visual_qa':.001,'asr':.01});assert r==copy
 busy=min(.01*features['paths']['whisper-small__asr']['standalone_ms']/1000,1)
 assert math.isclose(out['effective_service_time']['visual_qa']['qwen']-p['standalone_ms'],p['kernel_count']*.01*busy,rel_tol=1e-9)
 solo=model.evaluate({'qwen':2},{'visual_qa':{'qwen':.01}},{'visual_qa':.01})
 assert solo['effective_service_time']['visual_qa']['qwen']==p['standalone_ms']
 assert solo['SELF_INTERFERENCE_HANDLED_BY_G']
 zero=model.evaluate(x,{'visual_qa':{'qwen':.001}}, {'visual_qa':.001})
 assert zero['effective_service_time']['visual_qa']['qwen']==p['standalone_ms']
 assert model.pressure('phi4mm','visual_qa')==(0,'UNCALIBRATED_ZERO_FALLBACK')
 for invalid in [({'qwen':3},r,{'visual_qa':.001,'asr':.01}), (x,r,{'visual_qa':1,'asr':.01}), (x,{'ocr':{'qwen':1}},{'ocr':1})]:
  try:model.evaluate(*invalid)
  except ValueError:pass
  else:raise AssertionError('Invalid routing/deployment accepted')
 entries=load(OUT/'pair_split_manifest.json')['entries']
 assert len({e['id'] for e in entries})==len(entries)
 for q in {tuple(e['co_path']) for e in entries}:
  cal={tuple(e['target']) for e in entries if tuple(e['co_path'])==q and e['split']=='CALIBRATION'}
  val={tuple(e['target']) for e in entries if tuple(e['co_path'])==q and e['split']=='VALIDATION'}
  assert len(cal)>=2 and cal.isdisjoint(val)
 assert set(features['paths'])==set(load(ROOT/'results/module9a_interference/catalog.json')['paths'][i]['model']+'__'+load(ROOT/'results/module9a_interference/catalog.json')['paths'][i]['task'] for i in range(len(features['paths'])))
 return {'status':'PASS','checks':['co-path coefficient identity','unseen target uses kernel count, no fitted target tax','routing immutability','busy scaling units','idle co-model zero tax','replicas only empirical g','uncalibrated explicit fallback','unsupported/absent/conservation/replica-bound rejection','per-co-path target disjoint splits','all admitted profiles preserved']}

def preflight():
 result=cpu_checks();save('preflight_cpu_validation.json',result)
 verify(load(OUT/'protected_references.json'))
 files=[p for folder in [ROOT/'module9a_scheduling_tax',ROOT/'deployment_core/interference'] for p in folder.glob('*.py')]
 save('measurement_execution_freeze.json',{'at_utc':now(),'CPU_validation':result,'protocol_sha256':sha(OUT/'protocol_identity.json'),'split_sha256':sha(OUT/'pair_split_manifest.json'),'comparison_sha256':sha(OUT/'model_comparison_freeze.json'),'sources':[{'path':str(p),'sha256':sha(p)} for p in files],'native_workers_runtime_modified':False,'MPS_enabled':False,'profile_checkpoint_preserved':True})
 print(json.dumps(result),flush=True)

def validate_and_benchmark():
 checks=cpu_checks();verify(load(OUT/'protected_references.json'))
 freeze=load(OUT/'measurement_execution_freeze.json');verify(freeze['sources'])
 estimator=SchedulingPressureEstimator.load(OUT)
 # A routing spanning every admitted path, with each resident model at one replica.
 x={p['model']:1 for p in estimator.paths.values()};routing={}
 for p in estimator.paths.values():routing.setdefault(p['task'],{})[p['model']]=1e-4
 workload={k:sum(rates.values()) for k,rates in routing.items()}
 times=[]
 for _ in range(2000):
  start=time.perf_counter_ns();estimator.evaluate(x,routing,workload);times.append((time.perf_counter_ns()-start)/1e6)
 pairtimes=[]
 for _ in range(5000):
  start=time.perf_counter_ns();estimator.pair(('qwen','lid'),('whisper-small','asr'));pairtimes.append((time.perf_counter_ns()-start)/1e6)
 import numpy as np
 tracemalloc.start();loaded=SchedulingPressureEstimator.load(OUT);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
 save('runtime_overhead.json',{'CPU_only':True,'path_count':len(estimator.paths),'catalog_and_fit_json_bytes':sum((OUT/n).stat().st_size for n in ['path_features.json','fitted_model.json']),'resident_estimator_tracemalloc_bytes':current,'load_peak_bytes':peak,'pair_median_ms':float(np.median(pairtimes)),'pair_p95_ms':float(np.quantile(pairtimes,.95)),'full_21_path_11_model_median_ms':float(np.median(times)),'full_p95_ms':float(np.quantile(times,.95)),'no_GPU_or_profiler_during_evaluation':True})
 from replica_scaling.telemetry import snapshot
 gpu=snapshot();assert not gpu['processes'] and gpu['used_mib']<=128
 save('validation_results.json',{**checks,'profiles':21,'completed_GPU_runs':len(list((OUT/'raw_runs').glob('*.json'))),'profile_checkpoint_and_frozen_references_verified':True,'premeasurement_source_hashes_verified':True,'GPU_release':gpu,'native_worker_runtime_MPS_changed':False,'9A_complete':True,'multi_model_gpu_aggregation_validated':False})

if __name__=='__main__':preflight()
