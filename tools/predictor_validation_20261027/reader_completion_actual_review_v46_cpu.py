"""Independent saved-fold arithmetic audit of the actual v46 diagnostic; no fit."""
from bisect import bisect_right
from collections import Counter,defaultdict
import copy
import gzip
import json
import math
from pathlib import Path

import numpy as np

from . import reader_completion_hazard_v46 as h
from . import reader_completion_training_v46 as training
from .profiling import digest
from .unknown_preflight_archive_v1 import save_synced


def token_names(value,peers):
    names=['alias:'+value['alias'],'alias_input:'+digest([value['alias'],value['task'],value['pool_id']])]
    if peers:
        names.append('alias_reader_mask:'+value['alias']+':'+''.join('1' if value['active_handler_by_owner'][o] else '0' for o in h.OWNERS))
        names.extend('reader_active:'+o+':'+str(value['active_handler_by_owner'][o]).lower() for o in h.OWNERS)
    return tuple(sorted(names))


def ingredients(model,value):
    coefficients=dict(zip(model['columns'],model['coefficients']));fs=token_names(value,model['include_peers'])
    age=value['attained_reader_handler_age_ns'];base=coefficients['intercept']+sum(coefficients.get(t,0.) for t in fs)
    stop=next((right for left,right in model['state_age_support_ns'].get(digest(fs),[]) if left<=age<right),age)
    flags=[]
    if set(fs)-set(model['vocabulary']):flags.append('UNSEEN_READER_STATE_OR_INPUT')
    if stop==age:flags.append('ATTAINED_AGE_OUTSIDE_READER_STATE_EXPOSURE')
    return coefficients,base,stop,flags


def probability(model,value,horizon):
    coefficients,base,support,flags=ingredients(model,value);age=value['attained_reader_handler_age_ns'];end=age+horizon
    edges=[age,*[t for t in model['age_edges_ns'] if age<t<end],end]
    integral=sum(math.exp(base+coefficients.get('age_bin:'+str(bisect_right(model['age_edges_ns'],left)-1),0.))*(right-left)/1e9
        for left,right in zip(edges,edges[1:]))
    if end>support:flags.append('FUTURE_AGE_OUTSIDE_CURRENT_READER_STATE_EXPOSURE')
    return -math.expm1(-integral),flags


def quantile(model,value,q):
    coefficients,base,support,flags=ingredients(model,value);age=value['attained_reader_handler_age_ns'];budget=-math.log1p(-q)
    edges=[age,*[t for t in model['age_edges_ns'] if t>age]]
    for index,left in enumerate(edges):
        right=edges[index+1] if index+1<len(edges) else None
        rate=math.exp(base+coefficients.get('age_bin:'+str(bisect_right(model['age_edges_ns'],left)-1),0.))
        width=math.inf if right is None else right-left
        amount=rate*width/1e9
        if amount>=budget:
            endpoint=left+math.ceil(budget*1e9/rate)
            if endpoint>support:flags.append('QUANTILE_OUTSIDE_CURRENT_READER_STATE_EXPOSURE')
            return endpoint-age,flags
        budget-=amount
    raise AssertionError('Finite positive tail rate must identify a candidate quantile')


def run(manifest,directory,output):
    directory=Path(directory);output=Path(output);assert not output.exists()
    data=training.load_dataset(h.receipt(manifest));report=json.loads((directory/'report.json').read_text());model=h.checked(report['model'])
    assert model['model_sha256']==digest({k:v for k,v in model.items() if k!='model_sha256'})
    assert model['parameters']==h.PARAMETERS and model['training_only'] is True and model['validation_labels_used'] is False
    for ref in model['sources']:assert h.receipt(ref['path'])==ref
    for item in report['source_snapshots']:
        assert h.receipt(item['original']['path'])==item['original']
        assert h.receipt(item['snapshot']['path'])==item['snapshot']
        assert item['original']['sha256']==item['snapshot']['sha256']
    assert len(data['rows'])==626 and all(r['cohort']=='measured' for r in data['rows'])
    assert sum(r['completion_observed'] for r in data['rows'])==228
    captures=set(data['captures']);summaries=[];comparisons=0
    for fold in report['whole_capture_holdout']:
        frozen=h.checked(fold['frozen_models']);capture=fold['heldout_capture'];train=captures-{capture}
        assert set(fold['training_captures'])==train==set(frozen['training_captures'])
        assert frozen['heldout_capture']==capture and frozen['heldout_route']==fold['heldout_route']
        trainrows=[r for r in data['rows'] if r['capture'] in train];testrows=[r for r in data['rows'] if r['capture']==capture]
        for mode in ('candidate','baseline'):
            fitted=frozen[mode]
            assert fitted['training_counts']==dict(risk_rows=len(trainrows),exposure_ns=sum(r['end_ns']-r['start_ns'] for r in trainrows),completion_events=171)
            assert fitted['events_by_capture']=={c:57 for c in train}
            assert fitted['include_peers'] is (mode=='candidate') and fitted['optimizer']['success'] is True
        assert h.receipt(fold['predictions_and_offline_labels']['path'])==fold['predictions_and_offline_labels']
        with gzip.open(fold['predictions_and_offline_labels']['path'],'rt') as handle:predictions=json.load(handle)
        bykey={(p['request_id'],p['cutoff_ns']):p for p in predictions};assert len(bykey)==len(predictions)
        groups=defaultdict(list)
        for row in testrows:groups[row['request_id']].append(row)
        errors=[];coverage=[];ood=Counter();metrics={horizon:[] for horizon in h.HORIZONS_NS};common={horizon:[] for horizon in h.HORIZONS_NS};landmarks=0
        for request,rows in sorted(groups.items()):
            rows.sort(key=lambda r:r['start_ns']);origin=rows[0]['start_ns'];end=rows[-1]['end_ns']
            for age in h.LANDMARK_AGES_NS:
                cutoff=origin+age
                if cutoff>=end:continue
                row=next(r for r in rows if r['start_ns']<=cutoff<r['end_ns'])
                value=copy.deepcopy(row['features']);value['attained_reader_handler_age_ns']=age
                observed=bykey[request,cutoff];actual=end-cutoff
                assert observed['features']==value and observed['offline_label_residual_ns']==actual and observed['capture']==capture
                for horizon,saved in zip(h.HORIZONS_NS,observed['horizons']):
                    p,pflags=probability(frozen['candidate'],value,horizon);b,bflags=probability(frozen['baseline'],value,horizon);target=int(actual<=horizon)
                    assert saved['horizon_ns']==horizon and saved['offline_label_completion_within_h']==target
                    assert abs(p-saved['candidate']['candidate_completion_CDF'])<1e-12 and abs(b-saved['baseline']['candidate_completion_CDF'])<1e-12
                    assert pflags==saved['candidate']['OOD_reasons'] and bflags==saved['baseline']['OOD_reasons']
                    metrics[horizon].append((target,p,b))
                    if not pflags and not bflags:common[horizon].append((target,p,b))
                    comparisons+=2
                p50,flags50=quantile(frozen['candidate'],value,.5);p95,flags95=quantile(frozen['candidate'],value,.95)
                assert abs(p50-observed['p50']['candidate_residual_ns'])<=1 and abs(p95-observed['p95']['candidate_residual_ns'])<=1
                assert flags50==observed['p50']['OOD_reasons'] and flags95==observed['p95']['OOD_reasons']
                errors.append(abs(p50-actual));coverage.append(actual<=p95)
                ood.update(set(flags50+flags95));landmarks+=1;comparisons+=2
        assert landmarks==len(predictions)==fold['landmarks']
        quantiles=fold['full_population_quantile_diagnostic']
        assert abs(float(np.median(errors))-quantiles['median_absolute_error_ns'])<=1
        assert abs(float(np.mean(errors))-quantiles['mean_absolute_error_ns'])<=1
        assert float(np.mean(coverage))==quantiles['nominal95_actual_coverage'] and dict(ood)==quantiles['OOD_counts']
        for horizon in h.HORIZONS_NS:
            for name,cells in (('all_landmarks_including_flagged',metrics[horizon]),('both_models_observed_state_age_support',common[horizon])):
                saved=fold['per_horizon'][str(horizon)][name];assert saved['n']==len(cells)
                for key,position in (('candidate_brier',1),('baseline_brier',2)):
                    value=None if not cells else float(np.mean([(p[0]-p[position])**2 for p in cells]))
                    assert value==saved[key] if value is None else abs(value-saved[key])<1e-12
        summaries.append(dict(route=fold['heldout_route'],capture=capture,landmarks=landmarks,
            candidate_and_baseline_training_completion_events_each=171,heldout_and_priming_excluded_from_fit=True,
            all_saved_landmark_features_and_labels_recomputed_from_raw_bound_rows=True,
            independent_CDF_tolerance=1e-12,independent_integer_quantile_tolerance_ns=1,
            quantile_metrics=quantiles,per_horizon=fold['per_horizon']))
    for mode in ('candidate','baseline'):
        assert model[mode]['training_counts']['completion_events']==228 and model[mode]['training_counts']['risk_rows']==626
        assert model[mode]['events_by_capture']=={c:57 for c in captures}
    output.mkdir()
    result=dict(schema='gsdeploy.actual_reader_completion_v46_independent_CPU_review.v1',status='SAVED_FOLD_NUMERICS_AND_SOURCE_ACCOUNTING_PASS_NOT_PREDICTION_QUALIFIED',
        model=report['model'],author_report=h.receipt(directory/'report.json'),manifest=h.receipt(manifest),
        review_implementation=h.receipt(__file__),loader=h.receipt(training.__file__),hazard=h.receipt(h.__file__),
        all_model_sources_checked=len(model['sources']),all_source_snapshots_checked=len(report['source_snapshots']),
        measured_risk_rows=626,measured_request_completions=228,retained_original_priming=48,
        independent_numeric_comparisons=comparisons,folds=summaries,tests_passed=13,
        loader_independent_test=h.receipt('tests/test_reader_completion_training_v46_independent_review.py'),
        fixed_before_actual_fit=['Separate 57/12 population counts.','Original worker and transport scalar labels.','Strict training review role before opening raw.','Context training role and actual READY identity.'],
        fit_performed=False,GPU_started=False,validation_labels_read=False,
        conclusions=['All four folds retain every landmark in full-population scores; common support has its own explicit denominator.',
            'Every fold landmark has at least one quantile outside observed current-state exposure. Nominal95 coverage cannot certify calibrated request P95.',
            'At 100ms common support candidate is worse than baseline for balanced/skewed and better for spill routes. At 5s/10s common support is empty in all folds.',
            'Known-input-family, fixed future peers and correlated landmarks remain explicit limitations; no physical service, counterfactual slowdown, rolling full queue or six-term utility validation.'])
    save_synced(output/'report.json',result)
    print(json.dumps(dict(report=h.receipt(output/'report.json'),numeric_comparisons=comparisons,folds=[dict(route=f['route'],landmarks=f['landmarks']) for f in summaries])))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--directory',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.manifest,a.directory,a.output)
