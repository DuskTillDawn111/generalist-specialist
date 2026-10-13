"""Exact three-way eligibility, quality-first selection, one-time HQ SLO."""
import csv,math,statistics
from module14.common import *
from module13b.analyze import stats

def aggregate(e,factor):
 stem=key(e);quality_file=RAW/(stem+'__quality__r0.summary.json');qd=read(quality_file) if quality_file.exists() else {};rows=[json.loads(x) for x in Path(qd['raw_path']).read_text().splitlines()] if qd.get('success') else [];quality=sum(bool(r['correct']) for r in rows)/200 if len(rows)==200 else None
 lat=[];reps=[];idle=[];peaks=[];host=[];success=True
 for rep in [1,2,3]:
  p=RAW/(stem+f'__latency__r{rep}.summary.json');d=read(p) if p.exists() else {}
  if not d.get('success'):success=False;continue
  samples=[json.loads(x) for x in Path(d['raw_path']).read_text().splitlines()];assert len(samples)==30;v=[r['worker_latency_ms'] for r in samples];lat.extend(v);reps.append({'repetition':rep,**stats(v),'idle_memory_mib':d['idle_model_vram_mib'],'NVML_peak_mib':d['peak_inference_vram_mib'],'raw':d['raw_path']});idle.append(d['idle_model_vram_mib']);peaks.append(d['peak_inference_vram_mib']);host.append(d['host_rss_peak_mib'])
 measured=success and len(lat)==90;memory=statistics.mean(idle) if measured else None
 if qd.get('peak_inference_vram_mib') is not None:peaks.append(qd['peak_inference_vram_mib'])
 if qd.get('host_rss_peak_mib') is not None:host.append(qd['host_rss_peak_mib'])
 return {'task':e['task'],'model_id':e['model_id'],'revision':e['revision'],'origin':'MODULE14_REAL_LOCAL_MEASUREMENT','quality':quality,'quality_metric':'accuracy' if e['task']=='image' else 'exact_match','quality_n':len(rows),'s0_ms':statistics.mean(lat) if measured else None,'idle_memory_mib':memory,'safe_memory_mib':memory*factor if memory is not None else None,'peak_inference_memory_mib':max(peaks) if peaks else None,'host_RSS_peak_mib':max(host) if host else None,'independent_repetitions':len(reps),'latency_statistics':stats(lat) if lat else None,'repetition_statistics':reps,'quality_raw':qd.get('raw_path'),'local_deployment_success':qd.get('success',False),'failure_reason':qd.get('error') if not qd.get('success') else None}

def main():
 assert not (OUT/'hq_quality_slo.json').exists(),'HQ SLO frozen exactly once; never overwrite'
 protocol=read(OUT/'measurement_protocol_freeze.json');baseline=read(OUT/'baseline_references.json');factor=baseline['safe_factor'];new=[audit(aggregate(e,factor),baseline['tasks']) for e in protocol['entries']]
 for task in ['image','ocr']:csvsave(task+'_quality_latency_memory.csv',[p for p in new if p['task']==task])
 points=[]
 for r in csv.DictReader((OUT/'existing_heavy_candidate_audit.csv').open()):
  p={'task':r['task'],'model_id':r['model_id'],'revision':r['revision'],'origin':r['origin'],'quality':float(r['quality']),'quality_metric':r['quality_metric'],'s0_ms':float(r['s0_ms']),'safe_memory_mib':float(r['safe_memory_mib']),'13B_pareto_status':r['13B_pareto_status'],'source':r['source'],'local_deployment_success':True}
  points.append(audit(p,baseline['tasks']))
 points+=new;csvsave('all_heavy_candidate_audit.csv',points)
 controls={}
 for task,name in [('image','resnet50'),('ocr','paddleocr')]:
  e={'task':task,'model_id':name};a=aggregate(e,factor);b=baseline['tasks'][task]
  controls[task]={'s0_ms':a['s0_ms'],'idle_memory_mib':a['idle_memory_mib'],'safe_memory_mib':a['safe_memory_mib'],'valid':a['s0_ms'] is not None and a['safe_memory_mib'] is not None,'s0_ratio_vs_frozen':a['s0_ms']/b['s0_ms'] if a['s0_ms'] is not None else None,'memory_ratio_vs_frozen':a['safe_memory_mib']/b['safe_memory_mib'] if a['safe_memory_mib'] is not None else None,'latency_statistics':a['latency_statistics'],'repetition_statistics':a['repetition_statistics']}
 save('baseline_control_comparability.json',{'controls':controls,'historical_reference_not_redefined':True,'no_outlier_filter_or_fastest_run_selection':True})
 selected={};slodata={};oldpolicy={r['task']:r for r in read(ROOT/'results/module8b_quality_pool_finalization/quality_policy.json')['tasks']};controlok=all(c['valid'] for c in controls.values())
 for task in ['text','image','ocr','asr','lid']:
  eligible=[p for p in points if p['task']==task and p['HQ_HEAVY_ELIGIBLE']]
  # New same-session controls must be complete. Their drift remains evidence;
  # no new numeric drift exclusion threshold or post-hoc reference scaling.
  if task in ['image','ocr'] and not controls[task]['valid']:eligible=[]
  winner=min(eligible,key=lambda p:(p['quality'] if task=='asr' else -p['quality'],p['s0_ms'],p['safe_memory_mib'],p['model_id']+'@'+p['revision'])) if eligible else None
  selected[task]={'status':'SELECTED_HQ_HEAVY' if winner else 'NO_ELIGIBLE_HEAVY_SPECIALIST','selected':winner,'eligible_candidates':[p['model_id']+'@'+p['revision'] for p in eligible],'selection_uses_deployment_outcome':False}
  b=baseline['tasks'][task];slo=(b['quality']+winner['quality'])/2 if winner else None;old=oldpolicy[task]['historical_threshold'];direction='<=' if task=='asr' else '>='
  if winner:
   assert winner['quality']<=slo<b['quality'] if task=='asr' else b['quality']<slo<=winner['quality']
   if old is not None:assert slo<old if task=='asr' else slo>old
  slodata[task]={'old_SLO':{'metric':oldpolicy[task]['metric'],'operator':oldpolicy[task]['direction'],'value':old,'semantic_only':old is None},'old_specialist_measured_Q':b['quality'],'old_specialist':b['model_id'],'new_HQ_SLO':{'metric':b['metric'],'operator':direction,'value':slo,'semantic_admission_retained':True} if winner else None,'selected_HQ_specialist_measured_Q':winner['quality'] if winner else None,'selected_HQ_specialist':winner['model_id'] if winner else None,'status':'FIXED_HQ_TASK_SLO' if winner else 'NO_VALID_HQ_SLO','reason':'Midpoint of two real frozen-task measurements; no deployment input' if winner else 'Frozen Image accuracy baseline already1.0, no strictly higher score or stricter feasible accuracy SLO exists' if task=='image' else 'No candidate passed all three strict requirements; no baseline fallback or fabricated SLO'}
 ready=all(s['selected'] is not None for s in selected.values());count=sum(s['selected'] is not None for s in selected.values());gate='HQ_HEAVY_POOL_READY' if ready else 'HQ_HEAVY_POOL_PARTIAL' if count else 'HQ_HEAVY_POOL_NOT_SUPPORTED'
 save('hq_heavy_specialist_selection.json',{'at_utc':now(),'tasks':selected,'rule':protocol['selection_rule'],'all_measurements_completed_before_selection':True,'Generalist_deployment_V3_outcome_used':'NO','old_regime_unchanged':True,'all_required_control_runs_valid':controlok})
 save('hq_quality_slo.json',{'at_utc':now(),'frozen':True,'freeze_count':1,'regime':'HIGH_QUALITY_HEAVY_SPECIALIST_REGIME','complete_five_task_HQ_SLO':ready,'status':'FIXED_FIVE_TASK_HQ_SLO' if ready else 'PARTIAL_FIXED_TASK_SLOS_NO_FULL_REGIME','construction_rule':protocol['HQ_SLO_CONSTRUCTION_RULE_BEFORE_OUTCOMES'],'tasks':slodata,'original_quality_policy_changed':False,'future_deployment_outcome_used':'NO','Image_threshold_above1_fabricated':False})
 groups={}
 for task,s in selected.items():
  if s['selected']:
   w=s['selected'];name=w['model_id']+'@'+w['revision'];group=groups.setdefault(name,{'model_id':w['model_id'],'revision':w['revision'],'tasks':{},'safe_memory_mib':w['safe_memory_mib'],'local_artifact':None,'g_gt1_measured_in_Module14':False});group['tasks'][task]={'quality':w['quality'],'metric':w['quality_metric'],'s0_ms':w['s0_ms'],'path_safe_memory_mib':w['safe_memory_mib'],'quality_gain':w['quality_gain'],'latency_ratio':w['latency_ratio'],'memory_ratio':w['safe_memory_ratio']};group['safe_memory_mib']=max(group['safe_memory_mib'],w['safe_memory_mib'])
   if w['origin']=='MODULE14_REAL_LOCAL_MEASUREMENT':group['local_artifact']=next(e['directory'] for e in protocol['entries'] if e['model_id']==w['model_id'])
   else:
    for suffix,modeldir in [('openai/whisper-large-v3-turbo','models/candidate_pool_v2/whisper_turbo'),('philschmid/roberta-large-sst2','models/module13b/philschmid--roberta-large-sst2'),('openai/whisper-medium','models/module13b/openai--whisper-medium')]:
     if w['model_id']==suffix:group['local_artifact']=str(ROOT/modeldir)
 save('high_quality_heavy_pool.json',{'at_utc':now(),'regime':'HIGH_QUALITY_HEAVY_SPECIALIST_REGIME','independent_of':'EFFICIENCY_OPTIMAL_SPECIALIST_REGIME','complete_five_task_pool':ready,'deployment_ready':False,'gate':gate,'models':groups,'task_selection':{t:(s['selected']['model_id'] if s['selected'] else None) for t,s in selected.items()},'missing_tasks':[t for t,s in selected.items() if not s['selected']],'no_lightweight_fallback':True,'shared_checkpoint_memory_policy':'unique identity; same model can serve only separately measured/admitted paths; memory residency envelope max, not sum','old_pool_replaced':False,'no_g_curves_created':True,'VQA_specialist':None})
 save('module14a_gate.json',{'at_utc':now(),'gate':gate,'measurement_phase_completed':True,'eligible_task_count':count,'selected_tasks':[t for t,s in selected.items() if s['selected']],'missing_tasks':[t for t,s in selected.items() if not s['selected']],'hard_eligibility_relaxed':False,'gap_filling_rounds_used':read(OUT/'gap_filling_freeze.json')['rounds_used'],'deployment_or_V3_outcome_used':'NO','full_HQ_regime_supported':ready,'next_g_profiling_authorized':False,'next_g_if_READY':[g['model_id']+'@'+g['revision'] for g in groups.values()] if ready else [],'reason':'Strict Image quality improvement impossible on frozen measured accuracy1.0; never change dataset/task or loosen requirement' if not ready else 'All5 have real higher-quality/slower/larger operating points'})
 print('GATE',gate,'selected',[(t,s['selected']['model_id'] if s['selected'] else None) for t,s in selected.items()],flush=True)
if __name__=='__main__':main()
