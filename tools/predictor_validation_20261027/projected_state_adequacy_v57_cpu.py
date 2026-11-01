"""Training-only descriptive adequacy checks, without refit or tuning v57."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path

import numpy as np

from .implementation_source_closure_v27 import implementation_sources,strict_references
from .native_collect import receipt
from .save_hazard_v13 import EDGES_NS
from .unknown_preflight_archive_v1 import save_synced

OWNERS=('generalist_2#0','ocr#0','text#0','text#1')
# Diagnostic extension of the pre-existing v13 grid, fixed before streaming
# risk here. No model coefficients or hyperparameters are changed.
AGE_EDGES=(*EDGES_NS,100_000_000,500_000_000,1_000_000_000,2_000_000_000,5_000_000_000,10_000_000_000)
RESCALED_GRID=(.1,.25,.5,1.,2.,3.,5.)


def read(ref):
    if receipt(ref['path'])!=ref:raise ValueError('Source byte mismatch')
    return json.loads(Path(ref['path']).read_text())


def split_age(entry,exit,event):
    """Sum exact time over (entry,exit], with an edge event in the left bin."""
    for index in range(bisect_right(AGE_EDGES,entry)-1,bisect_left(AGE_EDGES,exit)):
        lo=max(entry,AGE_EDGES[index]);hi=min(exit,AGE_EDGES[index+1]) if index+1<len(AGE_EDGES) else exit
        if hi>lo:yield index,hi-lo,int(event and hi==exit)


def deviance(observed,expected):
    if expected==0:return 0. if observed==0 else None
    return 2*(observed*math.log(observed/expected)-observed+expected) if observed else 2*expected


def product_limit(observations,grid=RESCALED_GRID):
    """Right-censored cumulative-hazard residuals, retaining censoring in risk sets."""
    times=defaultdict(lambda:[0,0])
    for value,event in observations:times[value][0 if event else 1]+=1
    records=sorted(times.items());at_risk=len(observations);survival=1.;cursor=0;result=[]
    for threshold in grid:
        while cursor<len(records) and records[cursor][0]<=threshold:
            time,(events,censored)=records[cursor]
            if events:survival*=1-events/at_risk
            at_risk-=events+censored;cursor+=1
        result.append(dict(cumulative_hazard=threshold,estimated_survival=survival,constant_intensity_expected_survival=math.exp(-threshold),
            descriptive_absolute_difference=abs(survival-math.exp(-threshold)),at_risk_after_threshold=at_risk))
    event_values=[value for value,event in observations if event]
    return dict(envelopes=len(observations),events=len(event_values),right_censored=len(observations)-len(event_values),
        event_only_quantiles_diagnostic_not_censor_adjusted={str(q):float(np.quantile(event_values,q)) for q in (.1,.25,.5,.75,.9,.99)} if event_values else {},
        exponential_reference_quantiles={str(q):-math.log1p(-q) for q in (.1,.25,.5,.75,.9,.99)},product_limit_survival=result)


def run(model_path,report_path,output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    model_ref=receipt(model_path);report_ref=receipt(report_path);model=read(model_ref);report=read(report_ref)
    if (model_ref['sha256']!='ba035125b2d53c711a6996325718ff008a209d13821b83833fa9ddc3339181dc'
            or report_ref['sha256']!='11833f3134c2ec6777162b40416d553a893cfc0d32b8c093fe95a40e143ae2bd'
            or model['training_only'] is not True or model['validation_labels_used'] is not False):
        raise ValueError('Exact frozen actual v57 training model/report required')
    captures=set(model['captures']);folds=report['leave_one_whole_capture_out']
    if len(folds)!=4 or {f['held_out_capture'] for f in folds}!=captures:
        raise ValueError('Exactly one fold per original capture required')
    if (report['by_state']!=model['fitted'] or report['summary']!=model['coverage']
            or model['original_training_manifest']['sha256']!='4cbb2725b4ede8e8fc948844feb83daa38b97cd0e8cac81aa16eda93258735b4'):
        raise ValueError('Actual report/model/original manifest binding differs')
    folds={f['held_out_capture']:f for f in folds}
    age=defaultdict(lambda:dict(exposure_ns=0,events=0,pooled_expected=0.,heldout_expected=0.,heldout_unknown_exposure_ns=0,heldout_unknown_events=0))
    episodes=defaultdict(list);sources=[model_ref,report_ref,*model['sources']];stream_totals=Counter()
    for capture,entry in model['captures'].items():
        projected=read(entry['report'])
        if projected['training_only'] is not True or projected['validation_labels_used'] is not False or projected['capture']!=capture:
            raise ValueError('Training role before risk open required')
        ref=projected['risk_chronology'];sources.extend([entry['report'],ref]);read_ref=receipt(ref['path'])
        if read_ref!=ref:raise ValueError('Risk source drift')
        with gzip.open(ref['path'],'rt') as stream:
            for line in stream:
                row=json.loads(line);stream_totals['rows_read']+=1
                if row['capture']!=capture:raise ValueError('Risk capture mismatch')
                if not row['exposure_qualified']:
                    stream_totals['unresolved_rows']+=1;continue
                bits=''.join('1' if row['shared_left_features']['handler_active'][o] else '0' for o in OWNERS);key=row['held_owner']+':'+bits
                width=row['end_ns']-row['start_ns'];event=int(row['physical_SAVE_END_observed'])
                if width<=0 or width!=row['age_exit_ns']-row['age_entry_ns']:raise ValueError('Exact positive age exposure required')
                stream_totals['qualified_rows']+=1;stream_totals['exposure_ns']+=width;stream_totals['events']+=event
                pooled=model['fitted'][key];heldout=folds[capture]['fitted'][key]
                split=list(split_age(row['age_entry_ns'],row['age_exit_ns'],event))
                if sum(x[1] for x in split)!=width or sum(x[2] for x in split)!=event:raise ValueError('Age bin conservation failure')
                for index,ns,n in split:
                    cell=age[capture,key,index];cell['exposure_ns']+=ns;cell['events']+=n;cell['pooled_expected']+=pooled['hazard_per_second']*ns/1e9
                    if heldout['hazard_per_second'] is None:
                        cell['heldout_unknown_exposure_ns']+=ns;cell['heldout_unknown_events']+=n
                    else:cell['heldout_expected']+=heldout['hazard_per_second']*ns/1e9
                identity=(capture,row['held_owner'],tuple(row['driver_context']),row['restore_start_ns'])
                episodes[identity].append((row['start_ns'],row['end_ns'],event,key,row['age_entry_ns']))
        if receipt(ref['path'])!=ref:raise ValueError('Source changed during streaming')
    expected=model['coverage']['totals']
    if any(stream_totals[k]!=expected[k] for k in ('exposure_ns','events')) or stream_totals['qualified_rows']!=expected['rows']:
        raise ValueError('Exact training population differs')
    count_rows=[];all_cells=[]
    for capture,fold in folds.items():
        scored=[];unscored=[]
        for key,d in fold['diagnostics'].items():
            cell=dict(state=key,**d)
            if 'expected_SAVE' in d:scored.append(cell);all_cells.append(dict(capture=capture,route=fold['held_out_route'],**cell))
            elif d['heldout_exposure_ns']>0:unscored.append(cell)
        count_rows.append(dict(route=fold['held_out_route'],capture=capture,scored_strata=len(scored),
            covered95=sum(c['observed_within_predictive95'] for c in scored),nominal_coverage=.95,
            actual_coverage=sum(c['observed_within_predictive95'] for c in scored)/len(scored),
            scored_observed_SAVE=sum(c['observed_SAVE'] for c in scored),expected_SAVE=sum(c['expected_SAVE'] for c in scored),
            misses=[c for c in scored if not c['observed_within_predictive95']],unscored_exposed_strata=unscored))
    rescaled=defaultdict(list);episode_counts=Counter()
    for (capture,owner,context,restore),rows in episodes.items():
        rows.sort();episode_counts['total_envelopes']+=1
        if any(a[1]!=b[0] for a,b in zip(rows,rows[1:])) or any(row[2] for row in rows[:-1]):
            raise ValueError('Contiguous feature splits with terminal SAVE only required')
        episode_counts['delayed_entry_envelopes']+=int(rows[0][4]>0)
        event=bool(rows[-1][2]);h=0.;heldout_h=0.;unknown=False
        for lo,hi,_,key,_ in rows:
            h+=model['fitted'][key]['hazard_per_second']*(hi-lo)/1e9
            rate=folds[capture]['fitted'][key]['hazard_per_second']
            if rate is None:unknown=True
            else:heldout_h+=rate*(hi-lo)/1e9
        route=model['captures'][capture]['route'];rescaled['pooled:ALL'].append((h,event));rescaled['pooled:'+route].append((h,event))
        if unknown:
            episode_counts['heldout_unscorable_envelopes']+=1;episode_counts['heldout_unscorable_event_envelopes']+=int(event)
        else:
            rescaled['heldout:ALL'].append((heldout_h,event));rescaled['heldout:'+route].append((heldout_h,event))
    age_rows=[];pooled_age=defaultdict(lambda:dict(exposure_ns=0,events=0,expected_SAVE=0.))
    for (capture,key,index),cell in sorted(age.items()):
        age_rows.append(dict(capture=capture,route=model['captures'][capture]['route'],state=key,age_bin=index,
            age_lo_ns=AGE_EDGES[index],age_hi_ns=AGE_EDGES[index+1] if index+1<len(AGE_EDGES) else None,
            pooled_Poisson_deviance=deviance(cell['events'],cell['pooled_expected']),**cell))
        pooled_cell=pooled_age[key,index];pooled_cell['exposure_ns']+=cell['exposure_ns'];pooled_cell['events']+=cell['events'];pooled_cell['expected_SAVE']+=cell['pooled_expected']
    pooled_rows=[dict(state=key,age_bin=index,age_lo_ns=AGE_EDGES[index],age_hi_ns=AGE_EDGES[index+1] if index+1<len(AGE_EDGES) else None,
        deviance=deviance(cell['events'],cell['expected_SAVE']),**cell) for (key,index),cell in sorted(pooled_age.items())]
    large=[cell for cell in pooled_rows if cell['expected_SAVE']>=5]
    sources=strict_references([sources,implementation_sources(__file__)])
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Frozen training source changed')
    out.mkdir(parents=True,exist_ok=False);save_synced(out/'age-bin-counts.json',dict(edges_ns=AGE_EDGES,per_capture=age_rows,pooled_by_state=pooled_rows))
    save_synced(out/'rescaled-survival.json',{key:product_limit(values) for key,values in sorted(rescaled.items())})
    result=dict(schema='gsdeploy.projected_state_v57_training_adequacy_diagnostic.v1',status='MATERIAL_MODEL_MISMATCH_DETECTED_NO_REFIT',
        model=model_ref,author_report=report_ref,arithmetic_PASS_is_scientific_adequacy=False,
        four_fold_coverage=count_rows,total_scored_strata=len(all_cells),total_covered95=sum(c['observed_within_predictive95'] for c in all_cells),
        observed_coverage=sum(c['observed_within_predictive95'] for c in all_cells)/len(all_cells),nominal_coverage=.95,
        age_bins=receipt(out/'age-bin-counts.json'),age_bin_expected_ge5_cells=len(large),
        pooled_age_bin_deviance=sum(c['deviance'] for c in large),
        largest_age_bin_mismatches=sorted(large,key=lambda c:c['deviance'],reverse=True)[:20],
        rescaled_survival=receipt(out/'rescaled-survival.json'),episode_counts=dict(episode_counts),stream_counts=dict(stream_totals),
        supplementary_actual_reviewer_checks=dict(exact_four_unique_wholecapture_folds=True,report_by_state_equals_actual_model=True,
            report_summary_equals_full_model_coverage=True,original_v41_manifest_SHA_pinned=True),
        finding='Nominal 95% whole-capture predictive counts cover only 45/71 exposed scoreable state/capture cells. Large event-count discrepancies reject practical adequacy of current conditional constant-rate/cross-route exchangeability assumptions. Correct N/T arithmetic is insufficient.',
        limitations=['The four folds share training captures and state cells are dependent, so no independent binomial coverage p-value is asserted.',
            'Age-bin deviance is descriptive; bins share estimated rates, dependent contexts and a projected clock. No chi-square significance is asserted.',
            'Time rescaling uses integrated hazard across mask changes and preserves right censoring via product-limit survival. It assumes noninformative entry/censoring and correctly observed conditioning history.',
            'Pooled residuals are in sample; heldout residuals exclude complete envelopes crossing a state with no training exposure, reported explicitly.',
            'Mismatches can reflect age effects, state insufficiency, route effects, clock error, non-Poisson dynamics or dependent alternating physical marks. This diagnostic does not identify a unique cause.',
            'Age endpoint events are assigned to the left exposure bin. No model parameters are fitted, selected or changed.'],
        sources=sources,source_bytes_unchanged=True,fit_performed=False,model_parameters_changed=False,GPU_started=False,validation_labels_read=False)
    save_synced(out/'report.json',result);return receipt(out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('model','report','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();print(json.dumps(run(args.model,args.report,args.output)))
