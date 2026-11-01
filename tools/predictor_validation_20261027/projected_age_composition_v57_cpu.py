"""Fixed original-age-grid whole-capture diagnostic; no candidate fit or tuning.

Compare independent holder/mask/age Gamma-Poisson count predictions with
v57 constant-state predictions on exactly the same supported heldout cells.
Unsupported age exposure is UNKNOWN and never borrowed from another bin.
"""
import argparse
from bisect import bisect_left,bisect_right
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path

from scipy.signal import fftconvolve
from scipy.stats import nbinom
import numpy as np

from .implementation_source_closure_v27 import implementation_sources,strict_references
from .native_collect import receipt
from .save_hazard_v13 import EDGES_NS
from .unknown_preflight_archive_v1 import save_synced

OWNERS=('generalist_2#0','ocr#0','text#0','text#1')
KEYS=tuple(o+':'+format(i,'04b') for o in OWNERS for i in range(16))


def split(entry,exit,event):
    for index in range(bisect_right(EDGES_NS,entry)-1,bisect_left(EDGES_NS,exit)):
        lo=max(entry,EDGES_NS[index]);hi=min(exit,EDGES_NS[index+1]) if index+1<len(EDGES_NS) else exit
        if hi>lo:yield index,hi-lo,int(event and hi==exit)


def prediction(n,t,h,observed):
    if not t:return dict(status='UNKNOWN_NO_TRAINING_EXPOSURE')
    if not h:return dict(status='NO_HELDOUT_EXPOSURE')
    a=n+.5;p=t/(t+h);bounds=[int(x) for x in nbinom.ppf([.025,.975],a,p)]
    logprob=math.lgamma(observed+a)-math.lgamma(a)-math.lgamma(observed+1)+a*math.log(p)+observed*math.log1p(-p)
    return dict(status='TRAINING_ONLY_MARGINAL_COUNT_PREDICTIVE',observed=observed,expected=a*h/t,
        variance=a*h/t*(1+h/t),bounds95=bounds,covered95=bounds[0]<=observed<=bounds[1],log_predictive_probability=logprob,
        shape=a,p=p)


def sum_bounds(cells):
    """Convolve independent age-bin NB counts; do not use FFT tail mass as log score."""
    distribution=np.array([1.]);tail_bound=0.
    for cell in cells:
        a,p=cell['shape'],cell['p'];high=int(nbinom.ppf(1-1e-12,a,p))
        if high>2_000_000:raise ValueError('Count convolution exceeds explicit diagnostic memory bound')
        pmf=nbinom.pmf(np.arange(high+1),a,p);tail_bound+=float(nbinom.sf(high,a,p))
        distribution=fftconvolve(distribution,pmf);distribution=np.maximum(distribution,0.)
    mass=float(distribution.sum());cumulative=np.cumsum(distribution)
    if abs(1-mass)>1e-8:raise ValueError('Predictive convolution mass loss')
    return dict(bounds95=[int(np.searchsorted(cumulative,q)) for q in (.025,.975)],
        convolved_mass=mass,analytic_omitted_tail_upper_bound=tail_bound,
        numerical_method='NB component arrays truncated beyond 1-1e-12 CDF; FFT convolution, negative roundoff clipped; no FFT log scores')


def summarize(cells):
    return dict(scored_cells=len(cells),covered95=sum(c['covered95'] for c in cells),
        coverage=sum(c['covered95'] for c in cells)/len(cells) if cells else None,
        observed=sum(c['observed'] for c in cells),expected=sum(c['expected'] for c in cells),
        sum_marginal_log_predictive_probability=sum(c['log_predictive_probability'] for c in cells))


def run(model_path,report_path,output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    model_ref=receipt(model_path);report_ref=receipt(report_path)
    if (model_ref['sha256']!='ba035125b2d53c711a6996325718ff008a209d13821b83833fa9ddc3339181dc'
            or report_ref['sha256']!='11833f3134c2ec6777162b40416d553a893cfc0d32b8c093fe95a40e143ae2bd'):
        raise ValueError('Exact frozen v57 actuals required')
    model=json.loads(Path(model_path).read_text());report=json.loads(Path(report_path).read_text())
    counts={capture:{(key,index):dict(exposure_ns=0,events=0) for key in KEYS for index in range(len(EDGES_NS))} for capture in model['captures']}
    sources=[model_ref,report_ref,*model['sources']];rows=0
    for capture,entry in model['captures'].items():
        if receipt(entry['report']['path'])!=entry['report']:raise ValueError('Projection receipt drift')
        projected=json.loads(Path(entry['report']['path']).read_text())
        if projected['capture']!=capture or projected['training_only'] is not True or projected['validation_labels_used'] is not False:
            raise ValueError('Training role before original risk read required')
        ref=projected['risk_chronology'];sources.extend([entry['report'],ref])
        if receipt(ref['path'])!=ref:raise ValueError('Risk source drift')
        with gzip.open(ref['path'],'rt') as stream:
            for line in stream:
                row=json.loads(line);rows+=1
                if row['capture']!=capture:raise ValueError('Capture mismatch')
                if not row['exposure_qualified']:continue
                mask=row['shared_left_features']['handler_active']
                if set(mask)!=set(OWNERS) or any(type(v) is not bool for v in mask.values()):raise ValueError('Exact bool mask required')
                key=row['held_owner']+':'+''.join('1' if mask[o] else '0' for o in OWNERS)
                parts=list(split(row['age_entry_ns'],row['age_exit_ns'],row['physical_SAVE_END_observed']))
                if sum(v[1] for v in parts)!=row['end_ns']-row['start_ns'] or sum(v[2] for v in parts)!=int(row['physical_SAVE_END_observed']):
                    raise ValueError('Original fixed-age exposure/events not conserved')
                for index,t,n in parts:counts[capture][key,index]['exposure_ns']+=t;counts[capture][key,index]['events']+=n
        if receipt(ref['path'])!=ref:raise ValueError('Risk changed during full read')
        for key in KEYS:
            if (sum(counts[capture][key,i]['exposure_ns'] for i in range(len(EDGES_NS)))!=model['by_capture'][capture][key]['exposure_ns']
                    or sum(counts[capture][key,i]['events'] for i in range(len(EDGES_NS)))!=model['by_capture'][capture][key]['events']):
                raise ValueError('Original 64-state sufficient statistics changed')
    fold_reports=[];cell_rows=[];state_rows=[];all_age=[];all_constant=[];all_full_age=[];all_full_constant=[]
    for capture in sorted(counts):
        training=[c for c in sorted(counts) if c!=capture];route=model['captures'][capture]['route']
        age_cells=[];constant_cells=[];unseen=[];fold_state=[]
        for key in KEYS:
            constant_t=sum(model['by_capture'][c][key]['exposure_ns'] for c in training)
            constant_n=sum(model['by_capture'][c][key]['events'] for c in training)
            state_age=[];state_constant=[];unknown_t=0;unknown_n=0;observed=0;common_t=0
            for i in range(len(EDGES_NS)):
                test=counts[capture][key,i];h=test['exposure_ns'];n=test['events']
                if not h:continue
                t=sum(counts[c][key,i]['exposure_ns'] for c in training);train_n=sum(counts[c][key,i]['events'] for c in training)
                a=prediction(train_n,t,h,n);b=prediction(constant_n,constant_t,h,n)
                base=dict(capture=capture,route=route,state=key,age_bin=i,age_lo_ns=EDGES_NS[i],
                    age_hi_ns=EDGES_NS[i+1] if i+1<len(EDGES_NS) else None,
                    training_exposure_ns=t,training_events=train_n,heldout_exposure_ns=h,heldout_events=n)
                if not t:
                    unseen.append(base);unknown_t+=h;unknown_n+=n
                else:
                    age_cells.append(a);constant_cells.append(b);state_age.append(a);state_constant.append(b);observed+=n;common_t+=h
                    cell_rows.append(dict(**base,age_conditioned=a,constant_same_cell=b))
            if state_age:
                age_summary=summarize(state_age);constant_summary=summarize(state_constant)
                const_total=prediction(constant_n,constant_t,common_t,observed)
                convolution=sum_bounds(state_age);bounds=convolution['bounds95'];age_total=dict(observed=observed,
                    expected=sum(a['expected'] for a in state_age),bounds95=bounds,covered95=bounds[0]<=observed<=bounds[1],
                    **{k:v for k,v in convolution.items() if k!='bounds95'})
                value=dict(capture=capture,route=route,state=key,common_age_exposure_ns=common_t,common_age_observed_SAVE=observed,
                    unsupported_heldout_exposure_ns=unknown_t,unsupported_heldout_SAVE=unknown_n,
                    age_composition_standardized=age_total,constant_on_identical_age_exposure=const_total,
                    age_composite=age_summary,constant_composite=constant_summary,
                    age_reduces_absolute_count_bias=abs(observed-age_total['expected'])<abs(observed-const_total['expected']))
                fold_state.append(value);state_rows.append(value)
                if unknown_t==0:all_full_age.append(age_total);all_full_constant.append(const_total)
        all_age.extend(age_cells);all_constant.extend(constant_cells)
        fold_reports.append(dict(capture=capture,route=route,age_conditioned=summarize(age_cells),constant_same_support=summarize(constant_cells),
            positive_event_age_cells=summarize([a for a in age_cells if a['observed']>0]),
            positive_event_constant_cells=summarize([a for a in constant_cells if a['observed']>0]),
            unsupported_age_cells=unseen,unsupported_exposure_ns=sum(v['heldout_exposure_ns'] for v in unseen),
            unsupported_SAVE=sum(v['heldout_events'] for v in unseen),
            state_counts_same_support=dict(cells=len(fold_state),age_covered95=sum(v['age_composition_standardized']['covered95'] for v in fold_state),
                constant_covered95=sum(v['constant_on_identical_age_exposure']['covered95'] for v in fold_state),
                age_reduces_absolute_bias=sum(v['age_reduces_absolute_count_bias'] for v in fold_state))))
    sources=strict_references([sources,implementation_sources(__file__)])
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Frozen source changed')
    out.mkdir(parents=True,exist_ok=False)
    save_synced(out/'age-cell-predictions.json',cell_rows);save_synced(out/'state-composition-predictions.json',state_rows)
    save_synced(out/'all-fixed-age-counts.json',[dict(capture=capture,state=key,age_bin=index,**value) for capture,table in counts.items() for (key,index),value in table.items()])
    result=dict(schema='gsdeploy.projected_age_composition_wholecapture_diagnostic.v1',status='TRAINING_ONLY_FIXED_GRID_COMPARISON_NOT_MODEL_QUALIFICATION',
        model=model_ref,author_report=report_ref,original_fixed_EDGES_NS=EDGES_NS,prior='fixed Jeffreys Gamma(N+.5,T) per supported age cell, diagnostic only',
        folds=fold_reports,combined_age_cells=summarize(all_age),combined_constant_same_support=summarize(all_constant),
        combined_positive_event_age_cells=summarize([a for a in all_age if a['observed']>0]),
        combined_positive_event_constant_cells=summarize([a for a in all_constant if a['observed']>0]),
        fully_supported_whole_state_cells=dict(cells=len(all_full_age),age_covered95=sum(a['covered95'] for a in all_full_age),
            constant_covered95=sum(a['covered95'] for a in all_full_constant)),
        original_risk_rows_read=rows,age_cell_predictions=receipt(out/'age-cell-predictions.json'),
        state_composition_predictions=receipt(out/'state-composition-predictions.json'),all_fixed_age_counts=receipt(out/'all-fixed-age-counts.json'),
        limitations=['Predictive comparisons are four overlapping whole-training-capture holdouts, never independent validation.',
            'Each marginal age-cell log score is valid. Their sum is a descriptive composite score: constant-state cells share a Gamma latent rate and are dependent.',
            'Independent age-bin posteriors change the prior dimension; score improvement alone is not proof that all route effects are caused by age composition.',
            'Whole-state age count intervals convolve independent age-bin NB predictions on identical supported exposure; unsupported bins remain UNKNOWN and excluded explicitly.',
            'Complete-support whole-state coverage has a different denominator from v57 original 71 state/capture cells; do not silently compare denominators.',
            'No mask/holder/age pooling, no alternative bin search, no parameters changed, no runtime candidate model produced.',
            'Age-dependent event-time adequacy, source-clock calibration, correlated dynamics and actual GPU/queue accuracy require separate assessment.'],
        sources=sources,source_bytes_unchanged=True,fit_performed=False,runtime_model_produced=False,hyperparameters_tuned=False,
        GPU_started=False,validation_labels_read=False)
    save_synced(out/'report.json',result);return receipt(out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('model','report','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();print(json.dumps(run(args.model,args.report,args.output)))
