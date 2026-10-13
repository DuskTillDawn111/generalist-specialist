from .common import *
from replica_scaling.telemetry import snapshot

def main():
 assert not (OUT/'experiment_manifest.json').exists()
 protected={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in ROOT.rglob('*') if p.is_file() and OUT not in p.parents and any(x in p.parts for x in ['results','models','module20','deployment_core','deployment_search']) and '__pycache__' not in p.parts}
 save('protected_before.json',protected)
 old=read(ROOT/'results/module31a_qwen_whisper_legal_control/experiment_manifest.json');f=copy.deepcopy(old)
 gpu=snapshot();assert not gpu['processes'] and gpu['used_mib']<128
 f.update(module='Module32 Heavy Specialist Positive-Split Feasibility',GPU=gpu,slots=[],maximum_formal_runs=3,formal_order=['A1','A0','A2'],formal_seconds=600,Stage_D=True,scope='One exploratory candidate, no independent replication',optimizer_modified=False,independent_replication_automatic=False)
 f['plan']['GPU']=gpu
 f['plan']['models']['qwen']['supported_tasks']=['asr','lid','visual_qa']
 f['plan']['models'][S]=dict(supported_tasks=['asr'],idle_memory_mib=7310.,service_time_ms={'asr':494.912528},role='specialist')
 f['plan']['models'].pop('whisper-small')
 # Argument adapter only: reuse the measured Module13B worker, model and native settings.
 launch=OUT/'launchers/whisper_large_v3.py';launch.parent.mkdir(exist_ok=True)
 launch.write_text("import sys,runpy\nfrom pathlib import Path\nROOT=Path('/root/autodl-tmp/generalist-specialist')\nsys.path.insert(0,str(ROOT))\nsys.argv=[str(ROOT/'module13b/worker.py'),'--model-id','openai/whisper-large-v3','--task','asr','--directory',str(ROOT/'models/module13b/openai--whisper-large-v3')]\nrunpy.run_path(sys.argv[0],run_name='__main__')\n")
 f['plan']['registry'].pop('whisper-small');f['plan']['registry'][S]={'python':str(ROOT/'envs/specialist_env/bin/python'),'script':str(launch)}
 catalog=read(ROOT/'results/module31a_qwen_whisper_legal_control/selected_pool_catalog.json');catalog['models'][S]=f['plan']['models'][S];save('selected_pool_catalog.json',catalog)
 policy=read(ROOT/'results/module8b_quality_pool_finalization/quality_policy.json');thresholds={r['task']:r['historical_threshold'] for r in policy['tasks']}
 quality=list(csv.DictReader((ROOT/'results/module13_real_specialists/real_quality_results.csv').open()));srow=next(r for r in quality if r['model_id']=='openai/whisper-large-v3' and r['task']=='asr')
 raw=[json.loads(l) for l in Path(srow['raw']).read_text().splitlines()];wer=sum(r['word_errors'] for r in raw)/sum(r['reference_words'] for r in raw);assert len(raw)==200 and abs(wer-float(srow['quality_value']))<1e-12 and wer<=thresholds['asr']
 base=read(ROOT/'results/module8b_quality_pool_finalization/quality_admission_matrix.json')['rows'];rows=[]
 for t,m in [('asr','qwen'),('asr','phi4mm'),('text','roberta_sst2'),('image','convnext_tiny'),('lid','qwen'),('lid','whisper-small'),('visual_qa','qwen'),('visual_qa','phi4mm')]:
  r=next(r for r in base if r['task']==t and r['model']==m);assert r['planning_admissible'];rows.append({**r,'passed':True})
 rows.append(dict(task='asr',model=S,passed=True,measured_quality=wer,historical_threshold=thresholds['asr'],quality_metric='wer',raw=ref(srow['raw']),count=len(raw),source=ref(ROOT/'results/module13_real_specialists/real_quality_results.csv')))
 ocr=list(csv.DictReader((ROOT/'results/module14_high_quality_heavy/ocr_gapfill_measurements.csv').open()));rr=next(r for r in ocr if r['model_id']+'@'+r['revision']==OCR);assert float(rr['quality'])>=thresholds['ocr'] and rr['measurement_complete']=='True';rows.append(dict(task='ocr',model=OCR,passed=True,measured_quality=float(rr['quality']),historical_threshold=thresholds['ocr'],raw=ref(rr['quality_raw'])))
 csvsave('quality_admission_provenance.csv',rows)
 pairs=[]
 for g in ['qwen','phi4mm']:
  models=dict(f['plan']['models']);models[g]=catalog['models'][g]
  dep=[g,S,'roberta_sst2','convnext_tiny',OCR]+(['whisper-small'] if g=='phi4mm' else [])
  mem=sum(models[m]['idle_memory_mib'] if m in models else catalog['models'][m]['idle_memory_mib'] for m in dep)*f['safe_factor']
  bg=(catalog['models'][g]['service_time_ms']['visual_qa']*.2+(catalog['models'][g]['service_time_ms']['lid']*.2 if g=='qwen' else 0))/1000
  pairs.append(dict(candidate='C1' if g=='qwen' else 'C2',G=g,S=S,G_WER=next(r['measured_quality'] for r in rows if r['model']==g and r['task']=='asr'),S_WER=wer,ASR_WER_threshold=thresholds['asr'],quality_passed=True,six_tasks_covered=True,single_replica_deployment={m:1 for m in dep},safe_table_memory_mib=mem,B3_budget_mib=34567.5534,memory_passed=mem<=34567.5534,S_standalone_mean_ms=494.912528,S_standalone_p95_ms=960.25260895,G_standalone_ASR_mean_ms=catalog['models'][g]['service_time_ms']['asr'],G_background_historical_demand_ratio=bg,G_nominal_remaining_ratio=1-bg,decision='MEASURE_FIRST' if g=='qwen' else 'CONDITIONAL_FALLBACK_NOT_GPU_EXECUTED',capacity_scope='Historical standalone demand is screening evidence, not certified co-resident capacity'))
 assert pairs[0]['memory_passed'];save('model_pair_review.json',pairs);csvsave('model_pair_review.csv',pairs)
 summaries=[]
 for p in sorted((ROOT/'results').glob('module*/*summary.md')):
  prefix=p.parent.name.split('_')[0][6:]
  if prefix and (prefix.startswith('14') or any(prefix.startswith(str(i)) for i in range(15,32))):summaries.append(dict(source=ref(p),text=p.read_text()))
 save('historical_review_14F_31A.json',summaries)
 f['stage_B_protocol']=dict(background_qps=.2,initial_S_qps=1.4,initial_seconds=90,G_probe_qps=.6,G_probe_seconds=90,closed_loop_capacity_seconds=40,capacity_max_outstanding_per_ASR_worker=1,pressure_rate_rule='min(0.90*S closed-loop QPS,0.90/S initial co-resident mean seconds), rounded down to 0.01 QPS',pressure_seconds=120,split_precheck=.10,max_rate_adjustments=1,pressure_required='S mean native wait > 200 ms',no_added_inference_delay=True,formal_order=['A1','A0','A2'],formal_arm_seconds=600,warmup_seconds=30,drain_seconds=60)
 f['checkpoints']=[r for r in old['checkpoints'] if '/whisper-small/' not in r['path']]+[ref(p) for p in sorted((ROOT/'models/module13b/openai--whisper-large-v3').rglob('*')) if p.is_file() and '.cache' not in p.parts]
 f['checkpoint_metadata']={r['path']:[Path(r['path']).stat().st_size,Path(r['path']).stat().st_mtime_ns] for r in f['checkpoints']}
 paths={Path(r['path']) for r in old['sources']}|set((OUT/'m32').glob('*.py'))|{launch,ROOT/'module13b/worker.py'}
 f['sources']=[ref(p) for p in sorted(paths)]
 f['frozen_inputs']=[ref(ROOT/p) for p in ['results/module20_utility_optimizer/utility_definition.json','results/module20_utility_optimizer/utility_reference_manifest.json','results/module11_replica_and_cost/frozen_samples.jsonl','results/module13_real_specialists/real_quality_results.csv','results/module13_real_specialists/real_latency_results.csv','results/module13_real_specialists/real_memory_results.csv','results/module8b_quality_pool_finalization/quality_policy.json','results/module8b_quality_pool_finalization/quality_admission_matrix.json']]+[ref(srow['raw'])]
 f['frozen_unix_ns']=time.time_ns();f['slots']=[]
 save('experiment_manifest.json',f);save('frozen_gpu_manifest.json',f)
 save('stage_A_gate.json',dict(passed=True,candidate='C1',quality=True,memory=True,coverage=True,raw_S_WER_recomputed=wer,stage_B_not_certified_yet=True))
 print('STAGE_A_PASS',pairs,flush=True)
if __name__=='__main__':main()
