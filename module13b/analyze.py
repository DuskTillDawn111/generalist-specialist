"""Frozen admission, real Pareto/distance selection, no deployment outcomes."""
import csv, hashlib, json, math, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/module13_real_specialists';RAW=OUT/'raw13b'
def read(p):return json.loads(Path(p).read_text())
def save(n,d):(OUT/n).write_text(json.dumps(d,indent=2)+'\n')
def csvsave(n,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/n).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def key(e):return e['task']+'__'+e['model_id'].replace('/','--')
def quantile(v,q):
    v=sorted(v);p=(len(v)-1)*q;lo=int(p);hi=min(lo+1,len(v)-1);return v[lo]+(v[hi]-v[lo])*(p-lo)
def stats(v):
    mean=statistics.mean(v);std=statistics.stdev(v) if len(v)>1 else 0
    return {'count':len(v),'mean_ms':mean,'median_ms':statistics.median(v),'p95_ms':quantile(v,.95),'std_ms':std,'CV':std/mean}
def distance(task,gamma,mu,rule):
    def gap(v,limits):return max(math.log(limits[0])-math.log(v),0,math.log(v)-math.log(limits[1]))
    return min(max(gap(gamma,r['gamma']),gap(mu,r['mu'])) for r in rule['target_rectangles_by_task'][task])
def main():
    entries=read(OUT/'measurement_shortlist.json')['entries'];rule=read(OUT/'specialist_selection_rule.json');baseline=read(OUT/'baseline_references.json')['tasks'];factor=read(ROOT/'results/module8f1a_memory_breakpoints/memory_safety_factor.json')['safety_factor']
    pre=read(OUT/'cpu_preflight_results.json')['entries'];qrows=[];lrows=[];mrows=[];points=[];controls={}
    for task in ['asr','lid','text']:
        name=baseline[task]['ratio_reference'];e={'task':task,'model_id':name};vals=[];idle=[];valid=True
        for rep in [1,2,3]:
            summary=RAW/(key(e)+f'__latency__r{rep}.summary.json')
            d=read(summary) if summary.exists() else {}
            if not d.get('success'):valid=False;continue
            vals.extend(json.loads(x)['worker_latency_ms'] for x in Path(d['raw_path']).read_text().splitlines());idle.append(d['idle_model_vram_mib'])
        controls[task]={'success':valid and len(vals)==90,'model_id':name,'historical_s0_ms':rule['ratio_denominators'][task]['latency_ms'],'historical_idle_mib':rule['ratio_denominators'][task]['idle_memory_mib'],'statistics':stats(vals) if vals else None,'idle_mib':statistics.mean(idle) if idle else None}
        if vals:controls[task]['s0_over_frozen_reference']=statistics.mean(vals)/controls[task]['historical_s0_ms']
        if idle:controls[task]['idle_over_frozen_reference']=statistics.mean(idle)/controls[task]['historical_idle_mib']
    save('baseline_control_comparability.json',{'controls':controls,'all_controls_valid':all(v['success'] for v in controls.values()),'no_rescaling_old_baselines':True,'drift_is_reported_not_recalibrated':True,'numeric_drift_admission_threshold_added':False})
    for e in entries:
        task=e['task'];stem=key(e);qpath=RAW/(stem+'__quality__r0.summary.json');qd=read(qpath) if qpath.exists() else {};results=[json.loads(x) for x in Path(qd['raw_path']).read_text().splitlines()] if qd.get('success') else []
        metric='wer' if task=='asr' else 'accuracy';q=None;passed=False;reason='LOCAL_DEPLOYMENT_FAILED or incomplete quality measurement'
        if len(results)==200:
            q=sum(r['word_errors'] for r in results)/sum(r['reference_words'] for r in results) if task=='asr' else sum(bool(r['correct']) for r in results)/200
            passed=q<=.042181 if task=='asr' else q>=.925 if task=='text' else True
            reason='' if passed else 'Original frozen numeric quality admission failed'
        qrows.append({'task':task,'model_id':e['model_id'],'revision':e['revision'],'quality_value':q,'quality_metric':metric,'admission_pass':passed,'failure_reason':reason,'quality_n':len(results),'threshold':.042181 if task=='asr' else .925 if task=='text' else None,'semantics_only_admission':task=='lid','raw':qd.get('raw_path')})
        reps=[];v=[];idle=[];peaks=[];rss=[];valid=True
        for rep in [1,2,3]:
            path=RAW/(stem+f'__latency__r{rep}.summary.json');d=read(path) if path.exists() else {}
            if not d.get('success'):valid=False;continue
            rv=[json.loads(x)['worker_latency_ms'] for x in Path(d['raw_path']).read_text().splitlines()];assert len(rv)==30
            reps.append({'repetition':rep,**stats(rv),'raw':d['raw_path']});v.extend(rv);idle.append(d['idle_model_vram_mib']);peaks.append(d['peak_inference_vram_mib']);rss.append(d['host_rss_peak_mib'])
        s0=statistics.mean(v) if valid and len(v)==90 else None;mem=statistics.mean(idle) if valid and len(idle)==3 else None
        lrows.append({'task':task,'model_id':e['model_id'],'revision':e['revision'],'success':s0 is not None,'s0_ms':s0,**(stats(v) if v else {}),'independent_repetitions':len(reps),'repetition_statistics':reps,'quality_admitted':passed,'selection_eligible':passed and s0 is not None,'estimator':'sample-count weighted mean of 3x30 requests','same_session_control_ratio':s0/controls[task]['statistics']['mean_ms'] if s0 is not None and controls[task]['success'] else None})
        torchstats=[]
        quality_peak=qd.get('peak_inference_vram_mib')
        if quality_peak is not None:peaks.append(quality_peak)
        if qd.get('host_rss_peak_mib') is not None:rss.append(qd['host_rss_peak_mib'])
        for rep in [1,2,3]:
            f=RAW/(stem+f'__latency__r{rep}.worker.log')
            if f.exists():torchstats.extend(json.loads(x.split('|',1)[1]) for x in f.read_text().splitlines() if x.startswith('MEMORY_END|'))
        mrows.append({'task':task,'model_id':e['model_id'],'revision':e['revision'],'success':mem is not None,'idle_model_vram_mib':mem,'peak_inference_vram_mib':max(peaks) if peaks else None,'safe_memory_mib':mem*factor if mem is not None else None,'safe_factor':factor,'idle_by_repetition_mib':idle,'latency_peak_by_repetition_mib':peaks[:len(reps)],'quality_peak_inference_vram_mib':quality_peak,'host_rss_peak_mib':max(rss) if rss else None,'torch_memory_including_warmup':torchstats,'NVML_peak_sampling_ms':10,'safe_memory_definition':'unchanged factor * measured post-warmup idle process VRAM; not a guaranteed upper bound on peak'})
        reference=rule['ratio_denominators'][task];gamma=s0/reference['latency_ms'] if s0 is not None else None;mu=mem/reference['idle_memory_mib'] if mem is not None else None
        points.append({'task':task,'model_id':e['model_id'],'revision':e['revision'],'origin':'NEW_MEASURED_CANDIDATE','quality':q,'quality_utility':-q if task=='asr' and q is not None else q,'quality_metric':metric,'admission_pass':passed,'s0_ms':s0,'idle_memory_mib':mem,'safe_memory_mib':mem*factor if mem is not None else None,'gamma_real':gamma,'mu_real':mu,'valid_operating_point':s0 is not None and mem is not None,'selection_eligible':passed and s0 is not None and mem is not None,'target_distance':distance(task,gamma,mu,rule) if gamma and mu else None,'first_split_gamma_distance_log':max(math.log(1.2125)-math.log(gamma),0,math.log(gamma)-math.log(1.25)) if gamma else None,'stronger_gamma_distance_log':max(math.log(1.9375)-math.log(gamma),0,math.log(gamma)-math.log(2)) if gamma else None})
    frontiers={};selections={};allpoints=points[:]
    for task in ['asr','lid','text']:
        taskpoints=[p for p in points if p['task']==task]
        for b in baseline[task]['models']:
            if not b['quality_admitted']:continue
            q=b['quality'];mem=b['measured_idle_memory_mib'];s0=b['measured_service_ms'];ref=rule['ratio_denominators'][task];gamma=s0/ref['latency_ms'];mu=mem/ref['idle_memory_mib']
            p={'task':task,'model_id':b['source'],'revision':b['source_revision'],'legacy_planning_id':b['id'],'origin':'FROZEN_EFFICIENCY_BASELINE','quality':q,'quality_utility':-q if task=='asr' else q,'quality_metric':'wer' if task=='asr' else 'accuracy','admission_pass':True,'s0_ms':s0,'idle_memory_mib':mem,'safe_memory_mib':mem*factor,'gamma_real':gamma,'mu_real':mu,'valid_operating_point':True,'selection_eligible':True,'target_distance':distance(task,gamma,mu,rule)}
            taskpoints.append(p);allpoints.append(p)
        eligible=[p for p in taskpoints if p['selection_eligible']]
        for p in taskpoints:
            dominators=[q['model_id']+'@'+q['revision'] for q in eligible if q['quality_utility']>=p['quality_utility'] and q['s0_ms']<=p['s0_ms'] and q['safe_memory_mib']<=p['safe_memory_mib'] and (q['quality_utility']>p['quality_utility'] or q['s0_ms']<p['s0_ms'] or q['safe_memory_mib']<p['safe_memory_mib'])] if p['selection_eligible'] else []
            p['pareto_status']='REAL_DOMINATED' if dominators else 'REAL_PARETO_OPTIMAL' if p['selection_eligible'] else 'NOT_ADMITTED_OR_MISSING_MEASUREMENT';p['dominated_by']=dominators
        frontier=[p for p in taskpoints if p['pareto_status']=='REAL_PARETO_OPTIMAL'];bestdistance=min(p['target_distance'] for p in frontier);ties=[p for p in frontier if p['target_distance']<=bestdistance+1e-12];winner=min(ties,key=lambda p:(-p['quality_utility'],p['safe_memory_mib'],p['model_id']+'@'+p['revision']))
        selections[task]={'status':'SELECTED_NEW_BALANCED' if winner['origin']=='NEW_MEASURED_CANDIDATE' else 'BASELINE_RETAINED','winner':winner,'frontier_count':len(frontier),'tie_candidates':[p['model_id'] for p in ties],'rule_sha256':hashlib.sha256((OUT/'specialist_selection_rule.json').read_bytes()).hexdigest()};frontiers[task]={'universe':taskpoints,'frontier':[p['model_id']+'@'+p['revision'] for p in frontier]}
    csvsave('real_quality_results.csv',qrows);csvsave('real_latency_results.csv',lrows);csvsave('real_memory_results.csv',mrows);csvsave('real_operating_points.csv',allpoints)
    save('real_pareto_frontiers.json',{'quality_utility':'-WER for ASR, accuracy for LID/Text','memory_axis':'frozen-safe-factor * real idle (ordering identical to idle)','tasks':frontiers})
    selection={'tasks':selections,'selection_before_structural_replay':True,'selected_to_make_Generalist_win':'NO','Generalist_outcomes_used':False,'all_controls_valid':all(c['success'] for c in controls.values()),'13A_rule_preserved':True,'regime_catalog_policy':read(OUT/'module13b_execution_freeze.json')['regime_catalog_policy']}
    if not selection['all_controls_valid']:selection['selection_valid']=False;selection['reason']='Baseline control failed; comparability blocked, winners tentative only'
    else:selection['selection_valid']=True
    save('balanced_specialist_selection.json',selection)
    print(json.dumps({k: {'status':v['status'],'winner':v['winner']['model_id'],'distance':v['winner']['target_distance']} for k,v in selections.items()},indent=2))
if __name__=='__main__':main()
