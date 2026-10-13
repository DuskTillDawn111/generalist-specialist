"""CPU acceptance checks; these are software checks, not new scientific trials."""
import ast
import os
import subprocess
import sys
from unittest.mock import patch
from .common import *
from .predictor import UtilityDifferencePredictor

def main():
    checks=[]
    def passed(name,**evidence):
        checks.append(dict(check=name,passed=True,**evidence))
        print('PASS',name,flush=True)

    required=['module24_prediction_error_attribution.md','colocation_prediction_model.json',
        'utility_difference_predictor.py','independent_ranking_validation.csv','prediction_uncertainty.csv',
        'high_first_pool_search.csv','frozen_positive_candidates.json','gpu_paired_validation.csv',
        'module25_gate.json','module25_summary.md']
    assert all((OUT/f).is_file() and (OUT/f).stat().st_size for f in required)
    for path in [*(ROOT/'module25').glob('*.py'),OUT/'utility_difference_predictor.py']:
        ast.parse(path.read_text(),filename=str(path))
    assert (OUT/'utility_difference_predictor.py').read_bytes()==(ROOT/'module25/predictor.py').read_bytes()
    passed('required_outputs_and_python_syntax',required_outputs=len(required))

    obs=read(OUT/'observations.json');lookup={o['uid']:o for o in obs}
    report=read(OUT/'validation_report.json');protocol=read(OUT/'research_protocol.json')
    contrasts=read(OUT/'contrast_definition_manifest.json')
    final=UtilityDifferencePredictor.load(OUT/'colocation_prediction_model.json')
    for fold in report['folds']:
        fit=set(fold['fit_uids']);cal=set(fold['calibration_uids']);held=set(fold['held_combinations'])
        assert not fit&cal
        assert not {lookup[u]['deployment_workload_group'] for u in fit}&{lookup[u]['deployment_workload_group'] for u in cal}
        assert not {lookup[u]['checkpoint_group'] for u in fit|cal}&held
        for c in contrasts:
            if set(c['held_combinations'])==held:
                assert not {c['a_uid'],c['b_uid']}&(fit|cal)
        saved=read(OUT/'fold_models'/(hashlib.sha256(fold['fold'].encode()).hexdigest()[:12]+'.json'))
        assert set(saved['training_uids'])==fit
    held24={o['uid'] for o in obs if o['checkpoint_group']==protocol['final_holdout_checkpoint_pool']}
    assert len(held24)==6 and not held24&set(final.training_uids)
    assert protocol['final_holdout_checkpoint_pool'] not in final.training_combinations
    passed('whole_combination_and_workload_separation',outer_folds=len(report['folds']),all_24_pool_sessions_excluded=len(held24))

    # Independently calculate raw U without calling extract/components/replay.
    definition,refs=utility_inputs();weights=definition['weights'];recomputed={}
    for uid in ['M24A:high_gs18_diagnostic_rep1','M24A:high_s_only_diagnostic_attempt2',
                'M24B:high_proposed_rep1','M24B:high_control_rep1']:
        ob=lookup[uid];raw=read(ob['raw']['path']);phase=raw['phases']['measurement']
        records=phase['request_records'];done=[r for r in records if r['status']=='completed']
        assert len(done)==len(records) and not ob['hard_unstable']
        total=sum(ob['rates'].values());within=sum(r['completion_time_ns']<=phase['window']['end_time_ns'] for r in done)
        loss=1-min(1.,len(done)/len(records),within/(total*phase['window']['planned_duration_seconds']))
        L=P=0.
        for t,q in ob['rates'].items():
            values=[r['e2e_latency_ms'] for r in done if r['task']==t]
            L+=q/total*np.mean(values)/refs[t];P+=q/total*np.percentile(values,95)/refs[t]
        mem=ob['catalog']['safe_factor']*max(np.mean([v['owned_used_mib'] for v in raw[k]]) for k in ['postwarm_idle','postmeasurement_idle'])/ob['budget_mib']
        U=-weights['T']*loss-weights['L']*L-weights['P']*P-weights['M']*mem
        assert abs(U-ob['U'])<1e-12
        assert abs(sum(ob['actual_components'][k+'_contribution'] for k in ['T','L','P','M'])-U)<1e-12
        recomputed[uid]=float(U)
    attrib=read(OUT/'module24_prediction_error_attribution.json')
    for m,aid,bid in [('24A','high_gs18_diagnostic_rep1','high_s_only_diagnostic_attempt2'),('24B','high_proposed_rep1','high_control_rep1')]:
        delta=recomputed['M'+m+':'+aid]-recomputed['M'+m+':'+bid]
        assert abs(delta-attrib['experiments'][m]['actual_delta_U'])<1e-12
    passed('four_raw_full_cohort_utility_recomputations',U=recomputed,pooled_P95_not_substituted=True)

    rows=list(csv.DictReader((OUT/'independent_ranking_validation.csv').open()));groups=defaultdict(list)
    for r in rows:
        if r['primary']=='True':groups[r['contrast_group']].append(r)
    assert len(groups)==5
    available=[]
    for group,rr in groups.items():
        a=np.mean([float(r['actual_delta_U']) for r in rr]);assert a<0
        if all(r['predicted_delta_NEW'] for r in rr):
            p=np.mean([float(r['predicted_delta_NEW']) for r in rr]);available.append((a,p))
    assert len(available)==3 and sum(a*p>0 for a,p in available)==2
    assert sum(p>0 for a,p in available)==1 and sum(p>.01 for a,p in available)==0
    assert all(not r['interval_NEW'] and r['decision']=='ABSTAIN_UNCERTAIN' for r in rows)
    new=next(r for r in report['metrics'] if r['algorithm']=='NEW' and r['scope']=='PRIMARY_GS_MINUS_S')
    assert abs(np.mean([abs(a-p) for a,p in available])-new['delta_U_MAE'])<1e-12
    passed('independent_metric_denominators_and_no_fabricated_intervals',primary_groups=5,point_available=3,
        point_correct=2,abstentions=2,wrong_sign_positive=1,false_point_choices_above001=0)

    a=lookup['M24B:high_proposed_rep1'];b=lookup['M24B:high_control_rep1']
    pred=final.difference(a['catalog'],a['deployment'],a['routing'],b['routing'],a['rates'],a['budget_mib'],refs,weights,hardware_uuid=a['hardware_uuid'])
    assert pred['interval'] is None and not pred['support']
    doubles=[v for v in pred['proposed']['service_paths'].values() if v['replicas']==2]
    absent=[v for v in doubles if v['support']=='NO_MEASURED_CHECKPOINT_TASK_COUNT']
    assert absent and all(v['uncertain_joint_replica_scaling'] and v['support']!='MEASURED_PATH_CONDITIONAL' for v in doubles)
    assert 'predicted_effective_capacity' in pred['proposed']
    bad=copy.deepcopy(a['routing']);bad['ocr']={TEXT:a['rates']['ocr']}
    assert final.predict(a['catalog'],a['deployment'],bad,a['rates'],a['budget_mib'],refs,weights)['U'] is None
    bad=copy.deepcopy(a['routing']);bad['asr'][G]+=.1
    assert final.predict(a['catalog'],a['deployment'],bad,a['rates'],a['budget_mib'],refs,weights)['U'] is None
    x=copy.deepcopy(a['deployment']);x[G]=99
    assert final.predict(a['catalog'],x,a['routing'],a['rates'],a['budget_mib'],refs,weights)['U'] is None
    assert final.predict(a['catalog'],a['deployment'],a['routing'],a['rates'],1.,refs,weights)['U'] is None
    save('api_smoke_query.json',dict(catalog=str(OUT/'api_smoke_catalog.json'),deployment=a['deployment'],routing=a['routing'],
        control_routing=b['routing'],rates=a['rates'],budget_mib=a['budget_mib'],hardware_uuid=a['hardware_uuid'],arrival_process='poisson'))
    save('api_smoke_catalog.json',a['catalog'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    cli=subprocess.run([sys.executable,str(OUT/'utility_difference_predictor.py'),'--query',str(OUT/'api_smoke_query.json')],env=env,capture_output=True,text=True,check=True)
    result=json.loads(cli.stdout);assert abs(result['delta_U']-pred['delta_U'])<1e-12
    save('api_smoke_result.json',dict(scope='SOFTWARE_API_CHECK_NOT_ADDITIONAL_INDEPENDENT_VALIDATION',prediction=result))
    passed('public_api_cli_and_hard_constraints',unmeasured_double_paths=len(absent),unknown_replica_count_rejected=True,
        quality_violation_rejected=True,flow_violation_rejected=True,memory_violation_rejected=True)

    # Verify both sides respond to the complete route, using already fitted paths.
    ob=next(o for o in obs if o['module']==19 and o['deployment'].get(IMAGE)==1 and o['deployment'].get(G)==1)
    r0=copy.deepcopy(ob['routing']);r1=copy.deepcopy(ob['routing']);q=ob['rates']['image']
    r0['image']={IMAGE:q};r1['image']={IMAGE:q/2,G:q/2}
    route_changes=[]
    for m,t in [(G,'visual_qa'),(IMAGE,'image')]:
        s0=final.service(ob['catalog'],ob['deployment'],r0,m,t);s1=final.service(ob['catalog'],ob['deployment'],r1,m,t)
        assert s0['route_sensitive'] and s1['route_sensitive'] and abs(s0['mean_ms']-s1['mean_ms'])>1e-9
        route_changes.append(dict(model=m,task=t,control_mean_ms=s0['mean_ms'],split_mean_ms=s1['mean_ms']))
    passed('generalist_background_and_specialist_route_response',scope='FUNCTIONAL_CHECK_NOT_EFFECT_VALIDATION',paths=route_changes)

    # Analytical M/G/1 two tasks sharing each physical replica: service20/40ms.
    # This verifies sharing and thinning with a specified service distribution,
    # and does not assert unmeasured real replicas scale at this ratio.
    services={('synthetic','text'):dict(mean_ms=20.,quantiles_ms=[20.],transport_ms=0.),
              ('synthetic','image'):dict(mean_ms=40.,quantiles_ms=[40.],transport_ms=0.)}
    rates={'text':4.,'image':2.};synthetic=[]
    for n in [1,2]:
        rho=(4*.02+2*.04)/n;wait=(4*.02**2+2*.04**2)/n/(2*(1-rho))
        p=final._poisson({}, {'synthetic':n},{t:{'synthetic':q} for t,q in rates.items()},rates,100.,
            {'text':20.,'image':40.},weights,services,{'synthetic':rho},10.)
        assert len(p['worker_queues'])==n and p['eligible']
        assert all(abs(w['mean_ms']-wait*1000)<1e-12 for w in p['worker_queues'].values())
        assert abs(p['per_task']['text']['mean_ms']-(20+wait*1000))<1e-12
        assert abs(p['per_task']['image']['mean_ms']-(40+wait*1000))<1e-12
        synthetic.append(dict(replicas=n,rho=rho,analytic_wait_ms=wait*1000))
    passed('shared_worker_queue_and_replica_thinning_analytical_check',cases=synthetic)

    from .optimizer import optimize
    with patch('module25.optimizer.Generator',side_effect=AssertionError('Search must never start')),patch('module25.optimizer.legal_deployments',side_effect=AssertionError('Enumeration must never start')):
        try:optimize({}, {},0.,{},weights,final,report)
        except RuntimeError as error:assert 'PREDICTIVE_GATE_FAILED' in str(error)
        else:raise AssertionError('Failed predictive gate allowed Stage D')
    gate=read(OUT/'module25_gate.json');candidates=read(OUT/'frozen_positive_candidates.json')
    assert not gate['Stage_D_executed'] and not gate['Stage_E_executed'] and gate['new_GPU_acquisition_sessions']==0
    assert not candidates['candidates'] and not candidates['frozen']
    passed('failed_predictive_gate_blocks_optimizer_before_enumeration',new_GPU_sessions=0)

    before=read(OUT/'historical_metadata_before.json');after=protected_metadata()
    changed=[p for p,v in before.items() if after.get(p)!=v];assert not changed
    inventory=read(OUT/'source_inventory.json');verify(inventory)
    passed('historical_metadata_and_source_SHA256_unchanged',historical_files=len(before),scientific_sources=len(inventory))
    save('final_verification.json',dict(passed=True,checks=checks,CPU_only=True,
        scientific_holdout_scores_refitted_or_modified=False,new_scientific_trials=0,
        scope='Implementation/data-integrity acceptance; not additional predictive validation'))
    save('execution_source_manifest.json',dict(source=[ref(p) for p in sorted((ROOT/'module25').glob('*.py'))],
        required_outputs=[ref(OUT/f) for f in required],python=sys.executable,
        verification_command='PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 envs/data_env/bin/python -m module25.verify',
        stop_gate=gate['status'],historical_experiments_unchanged=True))
    print('MODULE25_VERIFICATION_COMPLETE_NO_GPU_NO_SEARCH',flush=True)

if __name__=='__main__':main()
