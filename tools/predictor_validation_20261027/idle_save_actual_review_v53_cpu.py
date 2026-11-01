"""Recompute frozen v53 counts/posteriors/folds without fitting or scipy stats."""
import argparse
from collections import defaultdict
import gzip
import io
import json
import math
from pathlib import Path
from statistics import NormalDist
import unittest

from .implementation_source_closure_v27 import implementation_sources,strict_references
from .learned_availability_forecast_v20 import verify_sources
from .native_collect import receipt
from .projected_handler_context_v42 import OWNERS
from .unknown_preflight_archive_v1 import save_synced


def empty():return dict(rows=0,exposure_ns=0,events=0,unresolved_rows=0,unresolved_exposure_ns=0,
    unresolved_events=0,age_min_ns=None,age_max_ns=None)


def combine(counts,captures,owner):
    values=[counts[c][owner] for c in captures];result=empty()
    for key in ('rows','exposure_ns','events','unresolved_rows','unresolved_exposure_ns','unresolved_events'):
        result[key]=sum(v[key] for v in values)
    mins=[v['age_min_ns'] for v in values if v['age_min_ns'] is not None]
    maxs=[v['age_max_ns'] for v in values if v['age_max_ns'] is not None]
    result['age_min_ns']=min(mins) if mins else None;result['age_max_ns']=max(maxs) if maxs else None
    return result


def run(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).absolute();out.mkdir(parents=True,exist_ok=False)
    base=root/'artifacts/predictor-idle-physical-save-v53-20261028-training01'
    model_ref=receipt(base/'model.json');report_ref=receipt(base/'report.json')
    if model_ref['sha256']!='ca82687c16f883844a91b2c7ab9b83a3ab4fa3bc5cb8312d2ffd8e32969a9590':
        raise ValueError('Exact frozen actual v53 model required')
    model=json.loads((base/'model.json').read_text());report=json.loads((base/'report.json').read_text())
    refs=[model_ref,report_ref,receipt(base/'training-manifest.json'),*model['sources']]
    counts={capture:{owner:empty() for owner in OWNERS} for capture in model['captures']}
    mark_count=0
    for capture,entry in model['captures'].items():
        verify_sources([entry['report']]);adapter=json.loads(Path(entry['report']['path']).read_text())
        refs.extend([entry['report'],adapter['risk_chronology'],adapter['mark_records']])
        verify_sources([adapter['risk_chronology'],adapter['mark_records']])
        with gzip.open(adapter['risk_chronology']['path'],'rt') as stream:
            for line in stream:
                row=json.loads(line);mask=row['shared_left_features']['handler_active']
                if set(mask)!=set(OWNERS) or any(type(x) is not bool for x in mask.values()):
                    raise ValueError('Complete four-owner projected bool mask required')
                if any(mask.values()):continue
                if row['capture']!=capture:raise ValueError('Risk capture mismatch')
                value=counts[capture][row['held_owner']];width=row['end_ns']-row['start_ns']
                if width<=0 or width!=row['age_exit_ns']-row['age_entry_ns']:
                    raise ValueError('Exposure mismatch')
                if not row['exposure_qualified']:
                    value['unresolved_rows']+=1;value['unresolved_exposure_ns']+=width
                    value['unresolved_events']+=int(row['physical_SAVE_END_observed']);continue
                value['rows']+=1;value['exposure_ns']+=width;value['events']+=int(row['physical_SAVE_END_observed'])
                lo,hi=row['age_entry_ns'],row['age_exit_ns']
                value['age_min_ns']=lo if value['age_min_ns'] is None else min(value['age_min_ns'],lo)
                value['age_max_ns']=hi if value['age_max_ns'] is None else max(value['age_max_ns'],hi)
        with gzip.open(adapter['mark_records']['path'],'rt') as stream:
            for line in stream:
                row=json.loads(line)
                mark_count+=int(all(x is False for x in row['features']['handler_active'].values()))
    if counts!=model['by_capture'] or mark_count!=model['idle_mark_rows'] or mark_count!=0:
        raise ValueError('Independent source N/T/counts/ages differ')
    comparisons=0
    def compare(actual,expected):
        nonlocal comparisons
        if isinstance(expected,float):
            if not math.isclose(actual,expected,rel_tol=1e-11,abs_tol=1e-13):
                raise ValueError(f'Independent numeric mismatch: {actual} != {expected}')
        elif actual!=expected:raise ValueError('Independent exact value mismatch')
        comparisons+=1
    def posterior_check(saved,stats):
        for key,value in stats.items():compare(saved[key],value)
        if stats['exposure_ns']==0:
            compare(saved['status'],'UNKNOWN_NO_EXPOSURE');compare(saved['hazard_per_ms'],None);return
        if stats['events']!=0:raise ValueError('Normal-square identity is only used for actual zero-event shape one-half')
        seconds=stats['exposure_ns']/1e9
        compare(saved['shape'],.5);compare(saved['rate_seconds'],seconds)
        compare(saved['hazard_per_second'],.5/seconds);compare(saved['hazard_per_ms'],.5/seconds/1000)
        for p,bound in zip((.025,.975),saved['credible95_per_second']):
            compare(bound,NormalDist().inv_cdf((1+p)/2)**2/(2*seconds))
        compare(saved['zero_event_frequentist_one_sided95_upper_per_second'],-math.log(.05)/seconds)
    captures=sorted(counts)
    for owner in OWNERS:posterior_check(model['fitted'][owner],combine(counts,captures,owner))
    heldout_rows=[]
    for fold in report['leave_one_whole_capture_out']:
        held=fold['held_out_capture'];training=[c for c in captures if c!=held]
        compare(fold['training_captures'],training)
        for owner in OWNERS:
            stats=combine(counts,training,owner);posterior_check(fold['fitted'][owner],stats)
            target=counts[held][owner];diagnostic=fold['diagnostics'][owner]
            if stats['exposure_ns']==0:
                compare(diagnostic['status'],'UNKNOWN_NO_TRAINING_EXPOSURE');continue
            if target['exposure_ns']==0:
                compare(diagnostic['status'],'NO_HELDOUT_EXPOSURE');continue
            train_s,test_s=stats['exposure_ns']/1e9,target['exposure_ns']/1e9
            p=train_s/(train_s+test_s);probability=math.sqrt(p)
            compare(diagnostic['observed_SAVE'],0)
            compare(diagnostic['expected_SAVE'],.5*test_s/train_s)
            compare(diagnostic['predictive_zero_probability'],probability)
            compare(diagnostic['log_predictive_probability'],math.log(probability))
            quantiles=[]
            for cutoff in (.025,.975):
                k=0;mass=probability;cumulative=mass
                while cumulative<cutoff:
                    mass*=((k+.5)/(k+1))*(1-p);k+=1;cumulative+=mass
                    if k>100000:raise ValueError('Predictive recurrence did not converge')
                quantiles.append(k)
            compare(diagnostic['posterior_predictive95_count'],quantiles)
            compare(diagnostic['observed_within_predictive95'],quantiles[0]<=0<=quantiles[1])
            heldout_rows.append(dict(capture=held,owner=owner,training_exposure_ns=stats['exposure_ns'],
                heldout_exposure_ns=target['exposure_ns'],expected_count=.5*test_s/train_s,
                zero_probability=probability,predictive95=quantiles))
    modules=['tests.test_idle_save_v53_independent_review','tests.test_idle_physical_save_hazard_v53']
    refs=strict_references([refs,receipt(__file__),implementation_sources(root/'tools/predictor_validation_20261027/projected_handler_forecast_v54.py'),
        [receipt(root.joinpath(*name.split('.')).with_suffix('.py')) for name in modules]])
    verify_sources(refs)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestLoader().loadTestsFromNames(modules))
    (out/'tests.txt').write_text(stream.getvalue());verify_sources(refs)
    review=dict(schema='gsdeploy.idle_SAVE_v53_actual_independent_CPU_review.v1',status='PASS' if result.wasSuccessful() else 'FAIL',
        actual_model=model_ref,actual_report=report_ref,sources=refs,source_bytes_unchanged=True,
        source_counts_by_capture=counts,source_totals=report['summary'],
        independent_exact_or_numeric_comparisons=comparisons,
        independent_numeric_method='N=0 Gamma(shape=.5,rate=T) credible quantiles from squared inverse standard-normal CDF; negative-binomial predictive counts from recurrence, not scipy.stats or production posterior/predictive helpers.',
        independently_recomputed_heldout_cells=heldout_rows,
        exposed_heldout_diagnostics=len(heldout_rows),
        tests_run=result.testsRun,test_log=receipt(out/'tests.txt'),new_independent_tests=4,
        actual_owner_rates_per_second={owner:value['hazard_per_second'] for owner,value in model['fitted'].items()},
        original_nonidle_queries_kept=True,all12false_empty_frontier_only=True,OCR_no_exposure_UNKNOWN=True,
        old_active_and_idle_rates_integrated_before_changes=True,no_budget_redraw_or_restore_age_reset=True,
        v54_original_v44_mark_retained=True,idle_mark_extrapolation_disabled=True,
        limitations=report['limitations'],
        interpretation='Training exposure constrains a predeclared positive constant idle rate. All-zero leaveout coverage does not validate constant intensity, time projection, idle marks, active GPU service, full-queue accuracy or utility.',
        fit_performed=False,full_forecast_executed=False,GPU_started=False,validation_labels_read=False)
    save_synced(out/'report.json',review)
    if not result.wasSuccessful():raise RuntimeError('Independent CPU regressions failed')
    return receipt(out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(run(parser.parse_args().output)))
