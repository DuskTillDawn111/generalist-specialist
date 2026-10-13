"""Module 23: fresh CPU search using the unchanged Module 20 utility optimizer.

No GPU imports; writes only this directory. Enumerate every measured placement,
then optimize every admitted route with the existing periodic FIFO predictor.
The only route choice in this admitted pool is ASR. Never imports Module21/22.
"""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar
from module20.optimizer import Generator, predict, predict_scenarios

G = 'phi4mm'
S = 'openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1'
TEXT = 'roberta_sst2'
IMAGE = 'convnext_tiny'
OCR = 'baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt'
LID = 'facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a'
MODELS = [G, S, TEXT, IMAGE, OCR, LID]
BUDGET = 32019.2544
RATES = {stage: dict(text=4., image=2., ocr=5., asr=q, lid=.3, visual_qa=.2)
         for stage, q in [('LOW', .02), ('MID', .50), ('HIGH', 1.55)]}

def read(path):
    return json.loads(Path(path).read_text())

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def reference(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)

def csvsave(name, rows, fields):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(fields + [k for r in rows for k in r])))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in r.items()} for r in rows)

def historical_metadata():
    # Full metadata audit; source hashes below protect all scientific inputs used.
    files = list((ROOT / 'results').rglob('*'))
    files += [p for folder in ROOT.glob('module*') if folder.is_dir() for p in folder.rglob('*')]
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in files if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts}

def best(fits):
    valid = [f for f in fits if (f.get('prediction') or {}).get('eligible')]
    return min(valid, key=lambda f: (-f['prediction']['U'], f['prediction']['safe_memory_mib'],
                                    json.dumps([f['deployment'], f['routing']], sort_keys=True))) if valid else None

def quality_checks(d, policy):
    checks = []
    matrices = {
        'old': list(csv.DictReader((ROOT/'results/quality_workload_audit/quality_matrix.csv').open())),
        'specialist': list(csv.DictReader((ROOT/'results/module13_real_specialists/real_operating_points.csv').open())),
        'ocr': list(csv.DictReader((ROOT/'results/module14_high_quality_heavy/ocr_gapfill_measurements.csv').open())),
    }
    policies = {p['task']: p for p in policy['tasks']}
    evidence = {(G,'asr'): ('old','phi4mm','wer'), (IMAGE,'image'): ('old','convnext_tiny','top1_accuracy'),
                (TEXT,'text'): ('specialist','textattack/roberta-base-SST-2','quality'),
                (S,'asr'): ('specialist','openai/whisper-large-v3','quality'),
                (LID,'lid'): ('specialist','facebook/mms-lid-256','quality'),
                (OCR,'ocr'): ('ocr','baudm/parseq:parseq','quality')}
    for (m,t), (table,identity,metric) in evidence.items():
        row = next(r for r in matrices[table] if r.get('model',r.get('model_id'))==identity and r['task']==t)
        value = float(row[metric]); p = policies[t]; threshold = p['historical_threshold']
        passed = threshold is None or (value <= threshold if p['direction']=='<=' else value >= threshold)
        checks.append(dict(model=m, task=t, value=value, threshold=threshold,
                           policy_status=p['policy_status'], passed=passed))
    raw = read(ROOT/'results/quality_workload_audit/raw_quality/phi4mm.json')
    assert raw['status']=='COMPLETE' and raw['tasks']['visual_qa']['status']=='COMPLETE'
    checks.append(dict(model=G,task='visual_qa',threshold=None,policy_status=policies['visual_qa']['policy_status'],
                       passed=True, evidence='Frozen complete visual_qa evaluation; no numeric threshold'))
    assert all(c['passed'] for c in checks)
    assert {(c['model'],c['task']) for c in checks} == {(m,t) for m in MODELS for t in d['models'][m]['supported_tasks']}
    # Explicitly retain excluded capabilities as excluded by original quality.
    exclusions = []
    for t,metric in [('text','accuracy'),('image','top1_accuracy'),('ocr','exact_match')]:
        r=next(r for r in matrices['old'] if r['model']==G and r['task']==t)
        assert float(r[metric]) < policies[t]['historical_threshold']
        exclusions.append(dict(model=G,task=t,value=float(r[metric]),threshold=policies[t]['historical_threshold'],reason='FROZEN_QUALITY_FAILURE'))
    return checks, exclusions

def routing(x, rates, fraction):
    rr = {t: {} for t in rates}
    for t,q in rates.items():
        if t=='asr':
            if fraction>0: rr[t][G]=q*fraction
            if fraction<1: rr[t][S]=q*(1-fraction)
        else:
            paths=[m for m in x if t in D['models'][m]['supported_tasks']]
            assert len(paths)==1
            rr[t][paths[0]]=q
    return rr

def optimize(x, rates):
    start=time.perf_counter(); diagnostics=[]; evaluations=0
    q=rates['asr']; sg=D['models'][G]['service_time_ms']['asr']/1000
    sv=D['models'][G]['service_time_ms']['visual_qa']/1000
    cg=D['capacity_factors'][G][str(x[G])]
    lo,hi=(0.,1.) if S in x else (1.,1.)
    # Only the original rho<1 capacity domain, no operational .8 constraint.
    hi=min(hi,(cg-rates['visual_qa']*sv-1e-10)/(q*sg))
    if S in x:
        cs=D['capacity_factors'][S][str(x[S])]
        ss=D['models'][S]['service_time_ms']['asr']/1000
        lo=max(lo,1-(cs-1e-10)/(q*ss))
    if hi<lo:
        return dict(deployment=x,status='NO_CAPACITY_FEASIBLE_ROUTE',solve_seconds=time.perf_counter()-start,
                    prediction=None,ASR_G_fraction_domain=[lo,hi])
    def evaluate(f):
        nonlocal evaluations
        evaluations+=1
        rr=routing(x,rates,float(f))
        return predict(D,x,rr,rates,BUDGET,REFS,WEIGHTS,phase=.37)
    candidates=[]
    def objective(z):
        f=float(np.asarray(z).reshape(-1)[0]);p=evaluate(f)
        if p['eligible']:
            candidates.append((f,p))
            return -p['U']
        return 1e6
    objective(lo);objective(hi)
    if hi-lo>1e-12:
        # Starts sample the domain; they never constrain the winning share.
        for f in np.linspace(lo,hi,65): objective(f)
        for seed in [23001,23002]:
            fit=differential_evolution(objective,[(lo,hi)],seed=seed,popsize=10,maxiter=80,
                                       tol=1e-9,atol=1e-11,polish=False)
            objective(fit.x)
            diagnostics.append(dict(method='continuous_DE',seed=seed,nfev=fit.nfev,
                                    success=bool(fit.success),message=str(fit.message)))
        leaders=sorted(candidates,key=lambda c:-c[1]['U'])[:3]
        for f,p in leaders:
            width=(hi-lo)/32
            fit=minimize_scalar(objective,bounds=(max(lo,f-width),min(hi,f+width)),method='bounded',
                                options={'xatol':1e-10,'maxiter':100})
            objective(fit.x)
            diagnostics.append(dict(method='bounded_scalar_polish',nfev=fit.nfev,success=bool(fit.success)))
    if not candidates:
        return dict(deployment=x,status='NO_ELIGIBLE_PREDICTION',prediction=None,solve_seconds=time.perf_counter()-start)
    f,p=min(candidates,key=lambda c:(-c[1]['U'],c[0]))
    rr=routing(x,rates,f)
    p=predict_scenarios(D,x,rr,rates,BUDGET,REFS,WEIGHTS)
    assert p['eligible']
    return dict(deployment=x,routing=rr,ASR_G_fraction=f,prediction=p,
                status='ALL_PLACEMENTS_CONTINUOUS_NUMERICAL_SEARCH_NO_GLOBAL_CERTIFICATE',
                ASR_G_fraction_domain=[lo,hi],solve_seconds=time.perf_counter()-start,
                predictor_evaluations=evaluations,solver_diagnostics=diagnostics)

def main():
    global D,REFS,WEIGHTS
    assert not (OUT/'cpu_optimizer_recheck.json').exists(), 'Refuse to overwrite completed experiment'
    before=historical_metadata(); save('historical_metadata_before.json',before)
    start=time.perf_counter()
    source=ROOT/'results/module14f_memory_ocr_x3/heavy_empirical_x3_reconciled.json'
    original=read(source);D=dict(original)
    for key in ['models','capacity_factors','memory_table_mib','evidence_bounds','model_evidence']:
        D[key]={m:original[key][m] for m in MODELS}
    D['module23_subset_only']=True;save('six_model_catalog.json',D)
    definition=read(ROOT/'results/module20_utility_optimizer/utility_definition.json')
    manifest=read(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json')
    WEIGHTS=definition['weights'];REFS=manifest['latency_reference_ms']
    assert WEIGHTS==dict(Q=0.,T=.25,L=.30,P=.30,M=.15)
    assert list(RATES['LOW']) == manifest['task_pool']
    policy=read(ROOT/'results/module8b_quality_pool_finalization/quality_policy.json')
    admission,exclusions=quality_checks(D,policy)
    inputs=[source,ROOT/'module20/optimizer.py',ROOT/'module20/common.py',
            ROOT/'results/module20_utility_optimizer/utility_definition.json',
            ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json',
            ROOT/'results/module8b_quality_pool_finalization/quality_policy.json',
            ROOT/'results/quality_workload_audit/quality_matrix.csv',
            ROOT/'results/quality_workload_audit/raw_quality/phi4mm.json',
            ROOT/'results/module13_real_specialists/real_operating_points.csv',
            ROOT/'results/module14_high_quality_heavy/ocr_gapfill_measurements.csv']
    frozen=[reference(p) for p in inputs]
    save('cpu_protocol.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),status='FROZEN_BEFORE_CPU_SEARCH',
        models=MODELS,rates=RATES,budget_mib=BUDGET,weights=WEIGHTS,latency_references_ms=REFS,inputs=frozen,
        offered_arrival_process='Independent per-task Poisson (GPU design)',
        CPU_predictor='Unchanged module20.optimizer.predict_scenarios: periodic FIFO surrogate; not Poisson or GPU observations',
        task_order=list(RATES['LOW']),all_admitted_paths_free=True,target_fraction_selection_constraints=None,
        all_measured_counts_compete=True,DE_seeds=[23001,23002],DE_maxiter=80,DE_popsize=10,
        route_domain='Full rho<1 domain, only frozen quality/budget/count/coverage constraints',
        ranking='Maximum predicted U, then smaller safe memory, then canonical JSON identity',
        gate='LOW no ASR S; MID autonomous ASR S; HIGH mixed ASR; legal controls and predicted delta_U>0.01 each',
        stage_B_and_C='Only after Stage A passes; formal runs maximum 18, three pairs per stage; each delta_U>0.01',
        optimizer_limits='Exhaustive measured placements; continuous numerical route search, no global optimality certificate'))
    states=[];legal=[]
    for counts in itertools.product(*(list(map(int,D['capacity_factors'][m])) for m in MODELS)):
        x={m:n for m,n in zip(MODELS,counts) if n}
        assert all(str(n) in D['memory_table_mib'][m] for m,n in x.items())
        mem=D['safe_factor']*sum(D['memory_table_mib'][m][str(n)] for m,n in x.items())
        cover={t for m in x for t in D['models'][m]['supported_tasks']}
        reasons=[]
        if mem>BUDGET+1e-6:reasons.append('SAFE_MEMORY_BUDGET')
        if set(RATES['LOW'])-cover:reasons.append('TASK_COVERAGE')
        states.append(dict(deployment=x,safe_memory_mib=mem,reject_reasons=reasons,legal_counts=True))
        if not reasons:legal.append(x)
    save('all_deployment_states.json',states)
    gen=Generator(D,REFS);stages={};comparison=[];component_rows=[];task_rows=[]
    for stage,rates in RATES.items():
        stage_start=time.perf_counter();seeds=[]
        for cap in manifest['generator']['capacity_caps']:
            r=gen.solve(rates,BUDGET,'asr',None,cap,WEIGHTS)
            if 'deployment' in r:r['prediction']=predict_scenarios(D,r['deployment'],r['routing'],rates,BUDGET,REFS,WEIGHTS)
            seeds.append(r)
        fits=[]
        for i,x in enumerate(legal):
            fit=optimize(x,rates);fit['candidate_id']=f'{stage}_D{i:03d}';fits.append(fit)
            print('CPU',stage,i+1,'/',len(legal),'U',fit['prediction']['U'] if fit['prediction'] else None,flush=True)
        # Rank original free MILP candidates alongside full enumeration.
        for i,r in enumerate(seeds):
            if 'deployment' in r:
                r={**r,'ASR_G_fraction':r['routing']['asr'].get(G,0.)/rates['asr'],
                   'candidate_id':f'{stage}_ORIGINAL_MILP_{i}'};fits.append(r)
        winner=best(fits);assert winner
        if stage=='LOW':control=best([f for f in fits if S in f['deployment']])
        elif stage=='MID':control=best([f for f in fits if S not in f['deployment']])
        else:
            rr={t:dict(row) for t,row in winner['routing'].items()};rr['asr']={S:rates['asr']}
            control=dict(deployment=winner['deployment'],routing=rr,ASR_G_fraction=0.,candidate_id='HIGH_S_ONLY_IDENTICAL_RESIDENCY',
                         prediction=predict_scenarios(D,winner['deployment'],rr,rates,BUDGET,REFS,WEIGHTS)) if S in winner['deployment'] else None
        delta=winner['prediction']['U']-control['prediction']['U'] if control and control['prediction']['eligible'] else None
        structural=(S not in winner['deployment'] if stage=='LOW' else S in winner['deployment'] if stage=='MID'
                    else 1e-7<winner['ASR_G_fraction']<1-1e-7)
        checks=dict(all_six_tasks_covered=set(winner['routing'])==set(rates),quality_admission=True,
                    safe_memory_budget=winner['prediction']['safe_memory_mib']<=BUDGET+1e-6,
                    legal_counts=all(str(n) in D['capacity_factors'][m] and str(n) in D['memory_table_mib'][m] for m,n in winner['deployment'].items()),
                    flow_conservation=winner['prediction']['route_conservation'],nominal_capacity=winner['prediction']['max_rho']<1,
                    expected_structure=structural,predicted_gain_gt_001=delta is not None and delta>.01)
        for arm,fit in [('selected',winner),('baseline',control)]:
            if fit is None:continue
            p=fit['prediction'];component_rows.append(dict(stage=stage,arm=arm,evidence='CPU_PREDICTION_ONLY',
                **{k:p.get(k) for k in ['U','T_norm','L_norm','P_norm','M_norm','T_contribution','L_contribution','P_contribution','M_contribution']},
                actual_GPU_U=None))
            for t,metrics in p.get('per_task',{}).items():task_rows.append(dict(stage=stage,arm=arm,task=t,
                evidence='CPU_PREDICTION_ONLY',predicted_mean_E2E_ms=metrics['mean_ms'],predicted_P95_E2E_ms=metrics['p95_ms'],
                actual_service_ms=None,actual_mean_E2E_ms=None,actual_P95_E2E_ms=None))
        stages[stage]=dict(rates=rates,winner=winner,baseline=control,predicted_delta_U=delta,checks=checks,
            Stage_A_stage_passed=all(checks.values()),candidates=fits,original_free_MILP=seeds,
            wall_seconds=time.perf_counter()-stage_start,continuous_global_optimality_certificate=False)
        comparison.append(dict(stage=stage,evidence='CPU_PREDICTION_ONLY_GPU_NOT_RUN',
            selected_ASR_G_fraction=winner['ASR_G_fraction'],selected_ASR_S_resident=S in winner['deployment'],
            predicted_selected_U=winner['prediction']['U'],predicted_baseline_U=control['prediction']['U'] if control else None,
            predicted_delta_U=delta,actual_delta_U=None,expected_structure_passed=structural,actual_GPU_positive_utility='NOT_TESTED'))
        save('cpu_stage_'+stage+'.json',stages[stage]);print('STAGE_RESULT',stage,comparison[-1],flush=True)
    passed=all(s['Stage_A_stage_passed'] for s in stages.values())
    result=dict(at_utc=datetime.now(timezone.utc).isoformat(),module=23,CPU_only=True,GPU_evidence=False,
        Stage_A_passed=passed,all_states_count=len(states),coverage_budget_legal_deployments=len(legal),
        all_legal_deployments_competed=True,all_admitted_routes_free=True,quality_checks=admission,excluded_paths=exclusions,
        frozen_inputs=frozen,original_optimizer='module20.optimizer.Generator + predict_scenarios unchanged',
        supplemental_search='Enumerate all legal placements and freely optimize ASR fraction with same original predictor/utility',
        no_historical_Poisson_results_reused=True,predictor_arrival_limitation='Periodic surrogate cannot certify actual Poisson queues',
        stage_summary=comparison,stages=stages,wall_seconds=time.perf_counter()-start,
        continuous_global_optimality_certificate=False)
    save('cpu_optimizer_recheck.json',result)
    csvsave('system_utility_comparison.csv',comparison,['stage','evidence','actual_delta_U'])
    csvsave('utility_components.csv',component_rows,['stage','arm','evidence','actual_GPU_U'])
    csvsave('per_task_metrics.csv',task_rows,['stage','arm','task','evidence','actual_service_ms'])
    csvsave('gpu_paired_runs.csv',[],['run_id','stage','arm','repetition','status','offered','completed','actual_U'])
    csvsave('worker_stability.csv',[],['run_id','worker','state','busy_fraction','end_backlog','actual_service_mean_ms','actual_service_p95_ms'])
    if not passed:
        reason='Stage A autonomous phase/gain requirements failed; user requires stopping before Stage B/C.'
        save('colocation_preflight.json',dict(status='NOT_RUN_STAGE_A_FAILED',Stage_B_passed=None,GPU_started=False,
            reason=reason,actual_safe_memory_mib=None,six_model_colocation_verified=False,
            predicted_six_model_x1_safe_memory_mib=D['safe_factor']*sum(D['memory_table_mib'][m]['1'] for m in MODELS),
            predicted_service_ms={m:D['models'][m]['service_time_ms']['asr'] for m in [G,S]},
            actual_service_distributions=None,predictor_GPU_service_deviation=None))
        save('frozen_gpu_manifest.json',dict(status='NOT_FROZEN_STAGE_A_FAILED',frozen=False,reason=reason,
            slots=[],actual_formal_runs=0,max_formal_runs=18,measurement_duration_seconds=None,
            power_plan_status='NOT_ACTIVATED_NO_GPU_EXPERIMENT_ALLOWED',
            sparse_LOW_ASR_note='At .02 QPS, 120s has only 2.4 expected ASR arrivals; 10000s yields 200 expected arrivals, approximately 10 upper-5% observations. Neither is a power guarantee.',
            no_seed_selection=True,no_optional_stopping=True))
        save('module23_gate.json',dict(module=23,gate='THREE_PHASE_CPU_NOT_REPRODUCED',Stage_A_passed=False,
            Stage_B_passed=None,Stage_C_passed=None,GPU_started=False,formal_GPU_runs=0,
            stage_checks={k:v['checks'] for k,v in stages.items()},positive_utility_three_phase_established=False,
            reason=reason,no_Module24=True,historical_experiments_modified=False))
    after=historical_metadata()
    assert before==after, 'Historical files changed during CPU validation'
    for ref in frozen:assert reference(ref['path'])['sha256']==ref['sha256']
    save('historical_integrity.json',dict(files_checked=len(before),metadata_unchanged=True,
        scientific_input_hashes_unchanged=True,scope='All previous results and module source metadata; full SHA256 of scientific inputs used. No full historical byte rehash.'))
    print('STAGE_A_PASSED',passed,'wall_seconds',result['wall_seconds'],flush=True)
    if passed:
        raise RuntimeError('Stage A passed: proceed to preflight; stopping outputs intentionally not created')

if __name__=='__main__':
    main()
