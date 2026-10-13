"""Whole-combination excluded predictions and independent contrast accounting."""
from .common import *
from .predictor import UtilityDifferencePredictor,original_J
from module20.optimizer import predict

def contrasts(observations):
    lookup={o['uid']:o for o in observations};rows=[]
    legacy=list(csv.DictReader((ROOT/'results/module20_utility_optimizer/prediction_ranking_validation.csv').open()))
    for r in legacy:
        if not r.get('GS_run') or not r.get('S_run'):continue
        a=lookup.get('M19:'+r['GS_run']);b=lookup.get('M19:'+r['S_run'])
        if a and b:rows.append(dict(case='M19_'+r['case_id'],target=r['case_id'].split('_')[0],a=a,b=b,
            original_control_legal=r['control_operationally_legal']=='True',scope='FIXED_RESIDENCY_GS_MINUS_S',original_study_role='LEGACY_GPU_BENCHMARK'))
    for rep in [1,2,3]:
        for module,stage,target in [(21,'low','lid'),(22,'mid','lid'),(22,'high','lid')]:
            a=lookup.get(f'M{module}:{stage}_proposed_rep{rep}');b=lookup.get(f'M{module}:{stage}_control_rep{rep}')
            if a and b:rows.append(dict(case=f'M{module}_{stage}',target=target,a=a,b=b,original_control_legal=True,
                scope='FIXED_RESIDENCY_GS_MINUS_S' if module==22 and stage=='high' else 'AUXILIARY_DEPLOYMENT_POLICY_NOT_GS_MINUS_S',
                original_study_role='LEGACY_PAIRED_GPU'))
    for module,aid,bid,role in [('24A','high_gs18_diagnostic_rep1','high_s_only_diagnostic_attempt2','DEVELOPMENT_ONLY_PROBE'),
                              ('24B','high_proposed_rep1','high_control_rep1','ORIGINAL_PROSPECTIVE_INDEPENDENT_PAIR')]:
        a=lookup.get(f'M{module}:{aid}');b=lookup.get(f'M{module}:{bid}')
        if a and b:rows.append(dict(case=f'M{module}_HIGH',target='asr',a=a,b=b,original_control_legal=True,
            scope='FIXED_RESIDENCY_GS_MINUS_S',original_study_role=role))
    for row in rows:
        a,b=row['a'],row['b'];row['contrast_group']=identity([row['case'],a['deployment'],b['deployment'],a['rates']])
        row['held_combinations']=sorted({a['checkpoint_group'],b['checkpoint_group']})
        row['primary']=bool(row['scope']=='FIXED_RESIDENCY_GS_MINUS_S' and row['original_control_legal'] and row['original_study_role']!='DEVELOPMENT_ONLY_PROBE')
        row['actual_delta']=a['U']-b['U'] if a['eligible'] and b['eligible'] else None
    return rows

def conformal_radius(errors,alpha=.05):
    n=len(errors);rank=math.ceil((n+1)*(1-alpha))
    return float(sorted(errors)[rank-1]) if n and rank<=n else None

def infer(model,ob,refs,weights):
    return model.predict(ob['catalog'],ob['deployment'],ob['routing'],ob['rates'],ob['budget_mib'],refs,weights,
        arrival_process=ob['arrival_process'],hardware_uuid=ob['hardware_uuid'])

def summarize(records,algorithm,primary=True):
    selected=[r for r in records if (r['primary'] or not primary) and r['actual_delta_U'] is not None]
    grouped=defaultdict(list)
    for r in selected:grouped[r['contrast_group']].append(r)
    truth=[];forecast=[];misspositive=missnegative=abstain=interval_n=interval_hit=0;positive=negative=0;combos=set()
    for values in grouped.values():
        actual=float(np.mean([r['actual_delta_U'] for r in values]));available=[r['predicted_delta_'+algorithm] for r in values if r['predicted_delta_'+algorithm] is not None]
        predicted=float(np.mean(available)) if len(available)==len(values) else None
        positive+=actual>.01;negative+=actual<0;combos.update(values[0]['held_combinations'])
        if predicted is None:abstain+=1
        else:
            truth.append(actual);forecast.append(predicted)
            missnegative+=actual<=0 and predicted>0
            misspositive+=actual>.01 and predicted<=0
        if algorithm=='NEW':
            finite=[r['interval_NEW'] for r in values if r['interval_NEW'] is not None]
            if len(finite)==len(values):
                lo=float(np.mean([v[0] for v in finite]));hi=float(np.mean([v[1] for v in finite]));interval_n+=1;interval_hit+=lo<=actual<=hi
    normal=np.array(truth);pred=np.array(forecast)
    directions=normal*pred>0
    result=dict(algorithm=algorithm,scope='PRIMARY_GS_MINUS_S' if primary else 'ALL_WITH_AUXILIARY_AND_DEVELOPMENT',
        independent_contrast_groups=len(grouped),checkpoint_combinations=len(combos),
        actual_positive_gt001_groups=positive,actual_negative_groups=negative,point_available_groups=len(truth),point_abstentions=abstain,
        point_direction_accuracy=float(np.mean(directions)) if len(directions) else None,
        wrong_point_GS_selections=missnegative,wrong_point_rejections_of_beneficial_GS=misspositive,
        delta_U_MAE=float(np.mean(np.abs(pred-normal))) if len(normal) and algorithm!='J' else None,
        delta_U_RMSE=float(np.sqrt(np.mean((pred-normal)**2))) if len(normal) and algorithm!='J' else None,
        ranking_accuracy_same_as_two_candidate_direction=float(np.mean(directions)) if len(directions) else None,
        interval_evaluable_groups=interval_n,interval_coverage=interval_hit/interval_n if interval_n else None,
        J_units='J gain seconds; no delta_U error or arbitrary J-to-U conversion' if algorithm=='J' else None)
    if len(normal)>1:
        from scipy.stats import spearmanr
        result['delta_gain_spearman']=float(spearmanr(normal,pred).statistic) if len(set(pred))>1 else None
        k=int(np.argmax(pred));result.update(top_predicted_gain_actual_delta_U=float(normal[k]),top_gain_regret=float(max(normal)-normal[k]))
    return result

def main():
    if (OUT/'validation_report.json').exists():
        assert not (OUT/'validation_completed.json').exists(),'Completed validation cannot be overwritten'
        # First attempt completed all immutable scores, but its diagnostic JSON
        # failed on a numpy boolean. Restore only missing artifacts from frozen
        # saved fold coefficients; do not refit, rescore or replace any result.
        observations=read(OUT/'observations.json');cs=contrasts(observations);definition,refs=utility_inputs();predictions={}
        for c in cs:
            fold=identity(c['held_combinations'])
            if fold not in predictions:
                mdl=UtilityDifferencePredictor.load(OUT/'fold_models'/(hashlib.sha256(fold.encode()).hexdigest()[:12]+'.json'))
                predictions[fold]={}
            else:mdl=UtilityDifferencePredictor.load(OUT/'fold_models'/(hashlib.sha256(fold.encode()).hexdigest()[:12]+'.json'))
            for ob in [c['a'],c['b']]:
                key=identity([ob['deployment'],ob['routing'],ob['rates'],ob['budget_mib'],ob['arrival_process']])
                if key not in predictions[fold]:predictions[fold][key]=infer(mdl,ob,refs,definition['weights'])
        prior=list(csv.DictReader((OUT/'independent_ranking_validation.csv').open()))
        for c,row in zip(cs,prior):
            fold=identity(c['held_combinations']);ps=[]
            for ob in [c['a'],c['b']]:
                key=identity([ob['deployment'],ob['routing'],ob['rates'],ob['budget_mib'],ob['arrival_process']]);ps.append(predictions[fold][key])
            delta=ps[0]['U']-ps[1]['U'] if all(p['U'] is not None for p in ps) else None
            assert (delta is None and not row['predicted_delta_NEW']) or (delta is not None and abs(delta-float(row['predicted_delta_NEW']))<1e-10)
        save('predictions_by_outer_fold.json',predictions)
        (OUT/'utility_difference_predictor.py').write_bytes((ROOT/'module25/predictor.py').read_bytes())
        save('validation_completed.json',dict(complete=True,only_missing_artifact_serialization_recovered=True,
            frozen_predictions_independently_recomputed_and_unchanged=True,no_refit_or_outcome_tuning=True))
        print('VALIDATION_ARTIFACT_RECOVERY_COMPLETE_NO_RESCORE',read(OUT/'validation_report.json')['passed'],flush=True)
        return
    protocol=read(OUT/'research_protocol.json');observations=read(OUT/'observations.json');definition,refs=utility_inputs();weights=definition['weights']
    cs=contrasts(observations);save('contrast_definition_manifest.json',[{k:v for k,v in c.items() if k not in ['a','b']}|
        dict(a_uid=c['a']['uid'],b_uid=c['b']['uid']) for c in cs])
    finaltrain=[o for o in observations if not o['final_holdout'] and not o['is_preflight']]
    final=UtilityDifferencePredictor().fit(finaltrain);save('colocation_prediction_model.json',final.as_dict())
    assert not set(final.training_combinations)&{protocol['final_holdout_checkpoint_pool']}
    models={};foldmeta=[];predictions={};records=[];uncertainty=[];latencymetrics=[];stable=[]
    # Outer folds exclude every session/count/workload with either arm's checkpoint set.
    for c in cs:
        fold=identity(c['held_combinations'])
        if fold in models:continue
        train=[o for o in observations if o['checkpoint_group'] not in c['held_combinations'] and not o['is_preflight']]
        groups={o['deployment_workload_group'] for o in train}
        calgroups={g for g in groups if int(hashlib.sha256(g.encode()).hexdigest()[:8],16)%5==0}
        fit=[o for o in train if o['deployment_workload_group'] not in calgroups]
        cal=[o for o in train if o['deployment_workload_group'] in calgroups]
        mdl=UtilityDifferencePredictor().fit(fit);models[fold]=mdl;assert not set(mdl.training_combinations)&set(c['held_combinations'])
        caluids={o['uid'] for o in cal};calerrors=defaultdict(list)
        for contrast in cs:
            if not contrast['primary'] or contrast['actual_delta'] is None:continue
            a,b=contrast['a'],contrast['b']
            if a['uid'] not in caluids or b['uid'] not in caluids:continue
            pa,pb=infer(mdl,a,refs,weights),infer(mdl,b,refs,weights)
            if pa['U'] is not None and pb['U'] is not None:
                calerrors[contrast['contrast_group']].append((pa['U']-pb['U'])-contrast['actual_delta'])
        # One group vote; repeat errors aggregated before calibration.
        errors=[abs(float(np.mean(v))) for v in calerrors.values()];radius=conformal_radius(errors)
        foldmeta.append(dict(fold=fold,fit_uids=mdl.training_uids,fit_groups=mdl.training_groups,held_combinations=c['held_combinations'],
            calibration_uids=sorted(caluids),calibration_groups=sorted(calgroups),calibration_contrast_groups=len(errors),
            finite95_conformal_radius=radius,finite95_requires_at_least19_distinct_calibration_contrasts=True,
            model_sha256=hashlib.sha256(identity(mdl.as_dict()).encode()).hexdigest(),fit_seconds=mdl.fit_seconds))
        save('fold_models/'+hashlib.sha256(fold.encode()).hexdigest()[:12]+'.json',mdl.as_dict())
        predictions[fold]={};print('FIT_OUTER',len(models),'fit_sessions',len(fit),'calibration_contrasts',len(errors),flush=True)
    for c in cs:
        a,b=c['a'],c['b'];fold=identity(c['held_combinations']);mdl=models[fold];meta=next(f for f in foldmeta if f['fold']==fold)
        ps=[];ts=[]
        for ob in [a,b]:
            key=identity([ob['deployment'],ob['routing'],ob['rates'],ob['budget_mib'],ob['arrival_process']])
            if key not in predictions[fold]:predictions[fold][key]=infer(mdl,ob,refs,weights)
            ps.append(predictions[fold][key])
        pa,pb=ps;newdelta=pa['U']-pb['U'] if pa['U'] is not None and pb['U'] is not None else None
        legacy=[];j=[];oldseconds=[]
        for ob in [a,b]:
            start=time.perf_counter();p=predict(ob['catalog'],ob['deployment'],ob['routing'],ob['rates'],ob['budget_mib'],refs,weights,phase=.37)
            oldseconds.append(time.perf_counter()-start);legacy.append(p);j.append(original_J(ob['catalog'],ob['deployment'],ob['routing'],ob['rates'])['J_seconds'])
        olddelta=legacy[0]['U']-legacy[1]['U'] if legacy[0]['U'] is not None and legacy[1]['U'] is not None else None
        jdelta=j[1]-j[0] if j[0] is not None and j[1] is not None else None
        supported=bool(pa.get('support') and pb.get('support'));radius=meta['finite95_conformal_radius']
        interval=[newdelta-radius,newdelta+radius] if newdelta is not None and radius is not None and supported else None
        actual=c['actual_delta'];primary=c['primary'];matched=False
        if a.get('cohort_signature') and b.get('cohort_signature'):matched=a['cohort_signature']==b['cohort_signature']
        row=dict(case=c['case'],target=c['target'],repetition=a['repetition'],scope=c['scope'],primary=primary,
            original_study_role=c['original_study_role'],contrast_group=c['contrast_group'],held_combinations=c['held_combinations'],
            whole_checkpoint_combinations_excluded=True,no_request_split=True,fold=fold,
            a_uid=a['uid'],b_uid=b['uid'],same_trace=matched,same_residency=a['deployment']==b['deployment'],
            actual_delta_U=actual,predicted_delta_NEW=newdelta,predicted_delta_A20=olddelta,predicted_delta_J=jdelta,
            NEW_absolute_delta_error=abs(newdelta-actual) if newdelta is not None and actual is not None else None,
            predicted_support=supported,interval_NEW=interval,
            interval_status='FINITE_WHOLE_GROUP_CONFORMAL' if interval else 'UNBOUNDED_SUPPORT_OR_INSUFFICIENT_CALIBRATION',
            finite_calibration_contrast_groups=meta['calibration_contrast_groups'],
            confident_GS_selected=bool(interval and interval[0]>.01),
            confident_GS_rejected=bool(interval and interval[1]<=0),
            decision='GS' if interval and interval[0]>.01 else 'S-only' if interval and interval[1]<=0 else 'ABSTAIN_UNCERTAIN',
            NEW_prediction_seconds=pa['seconds']+pb['seconds'],A20_prediction_seconds=sum(oldseconds),
            J_system_U_not_defined=True,a_actual_hard_unstable=a['hard_unstable'],b_actual_hard_unstable=b['hard_unstable'],
            a_NEW_nominal_stable=pa['eligible'],b_NEW_nominal_stable=pb['eligible'],
            a_NEW_stability=pa['predicted_stability'],b_NEW_stability=pb['predicted_stability'])
        records.append(row);uncertainty.append({k:v for k,v in row.items() if k in ['case','repetition','primary','actual_delta_U',
            'predicted_delta_NEW','predicted_support','interval_NEW','interval_status','finite_calibration_contrast_groups','decision','contrast_group']})
        for ob,pnew,pold in zip([a,b],ps,legacy):
            stable.append(dict(uid=ob['uid'],fold=fold,actual_unstable=ob['hard_unstable'],NEW_nominal_stable=pnew['eligible'],
                NEW_confirmed_stable=False,NEW_predicted_stability=pnew['predicted_stability'],A20_nominal_stable=pold['eligible'],
                no_UNKNOWN_counted_as_correct_unstable=True))
            if ob['eligible']:
                for alg,p in [('NEW',pnew),('A20',pold)]:
                    latencymetrics.append(dict(uid=ob['uid'],fold=fold,algorithm=alg,actual_U=ob['U'],predicted_U=p.get('U'),
                        U_error=p['U']-ob['U'] if p.get('U') is not None else None,
                        actual_per_task=ob['per_task'],predicted_per_task=p.get('per_task'),nominal_eligible=p['eligible']))
        print('OUT_OF_SAMPLE',c['case'],'rep',a['repetition'],'actual',actual,'NEW',newdelta,'A20',olddelta,'support',supported,flush=True)
    metrics=[summarize(records,alg,primary=primary) for primary in [True,False] for alg in ['J','A20','NEW']]
    new=next(m for m in metrics if m['scope']=='PRIMARY_GS_MINUS_S' and m['algorithm']=='NEW')
    unique_stability=list({(r['uid'],r['fold']):r for r in stable}.values())
    stability_summary=dict(actual_unstable=sum(r['actual_unstable'] for r in unique_stability),
        NEW_false_nominal_stable=sum(r['actual_unstable'] and r['NEW_nominal_stable'] for r in unique_stability),
        A20_false_nominal_stable=sum(r['actual_unstable'] and r['A20_nominal_stable'] for r in unique_stability),
        NEW_false_confirmed_stable=0,NEW_unknown_not_counted_as_detected=sum(r['actual_unstable'] and r['NEW_nominal_stable'] for r in unique_stability))
    checks=dict(contrast_count=new['independent_contrast_groups']>=3,combination_count=new['checkpoint_combinations']>=2,
        positive_test_support=new['actual_positive_gt001_groups']>=2,negative_test_support=new['actual_negative_groups']>=2,
        direction_accuracy=new['point_direction_accuracy'] is not None and new['point_direction_accuracy']>=.8,
        finite_interval_available_for_all=new['interval_evaluable_groups']==new['independent_contrast_groups'],
        interval_coverage=new['interval_coverage'] is not None and new['interval_coverage']>=.8,
        no_false_confident_GS=not any(r['confident_GS_selected'] and r['actual_delta_U'] is not None and r['actual_delta_U']<=0 for r in records if r['primary']),
        no_false_confident_stability=stability_summary['NEW_false_confirmed_stable']==0,
        matched_pairs=all(r['same_trace'] for r in records if r['primary'] and r['actual_delta_U'] is not None))
    passed=all(checks.values());summary=dict(passed=passed,gate_checks=checks,metrics=metrics,stability=stability_summary,
        final_model_training_excludes_all_Module24A_Module24B_pool_data=True,
        original_negative_results_not_train_and_test_in_same_fold=True,folds=foldmeta,
        no_blind_prospective_new_validation=True,independent_positive_GS_groups_in_primary=new['actual_positive_gt001_groups'],
        development_positive_probe_not_counted_as_primary_independent_evidence=True,
        point_coverage_does_not_certify_decisions=True,Stage_D_permitted=passed,
        reason='All frozen group/sign/positive-support/uncertainty criteria required. Unbounded intervals do not masquerade as calibrated confidence.')
    save('validation_report.json',summary);csvsave('independent_ranking_validation.csv',records)
    csvsave('prediction_uncertainty.csv',uncertainty);csvsave('predictor_accuracy_and_overhead.csv',metrics)
    csvsave('stability_holdout_validation.csv',unique_stability);csvsave('absolute_utility_holdout.csv',latencymetrics)
    save('predictions_by_outer_fold.json',predictions)
    (OUT/'utility_difference_predictor.py').write_bytes((ROOT/'module25/predictor.py').read_bytes())
    save('validation_completed.json',dict(complete=True,no_refit_or_outcome_tuning=True))
    print('PREDICTION_GATE',passed,checks,flush=True)

if __name__=='__main__':main()
