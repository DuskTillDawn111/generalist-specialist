"""Re-evaluate frozen H600 training folds with event-completion numerics v2.

No refit, new outcomes, threshold changes or replacement of the original report.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import time

from . import run_aware_report as original
from .native_flow import checked, receipt, save
from .probe_validation import read


def run(original_report, output):
    from .h600_event_predict_v2 import predict
    from . import h600_event_predict_v2 as helper
    from . import h600_event_simulator_v2 as numerical
    from .duration_selection import evaluate
    from .ocr_trained_validation import LIMITS
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    previous=read(original_report)
    if (previous.get('schema')!='gsdeploy.run_aware_H600_training_seed_crossfit_error_report.v1'
            or previous.get('independent_validation') is not False
            or previous.get('thresholds_tuned') is not False
            or previous.get('requested')!=9600):
        raise ValueError('Original complete-domain H600 training CV report required')
    refs=[receipt(original_report),receipt(__file__),receipt(numerical.__file__),receipt(helper.__file__),*previous['source_receipts']]
    for ref in refs:checked(ref)
    audit=original.audit_manifest(checked(previous['manifest']))
    folds,resource_refs=original.resource_folds(checked(previous['resource_manifest']),audit['contracts'])
    refs.extend([*audit['source_receipts'],*resource_refs])
    quality_ref=next(r for r in previous['source_receipts'] if Path(r['path']).name=='training-v8-quality.json')
    quality_doc=read(checked(quality_ref));quality=quality_doc.get('loss_by_family_task',quality_doc)
    addendum=read(checked(previous['threshold_addendum']))
    if addendum['thresholds']!=LIMITS:raise ValueError('Original thresholds required')
    rows=[];by_seed={seed:[] for seed in audit['contracts']}
    for case in audit['cases']:
        plan=read(checked(case['heldout_plan']))
        candidate=next(c for c in plan['candidates'] if c['id']==case['candidate_id'])
        proposed=copy.deepcopy(candidate);fold=folds[case['seed']]
        override=fold['candidate_overrides'][candidate['id']]
        for key in ('predicted_startup_joules','predicted_prephase_peak_mib'):proposed[key]=override[key]
        prediction=predict(plan,proposed,read(checked(case['model'])),read(checked(fold['resources'])),quality)
        # Actual execution is read only after the public-input forecast.
        raw,scored=original.core.scored_heldout(case['heldout_raw'])
        row=original.core.compare(plan,candidate,prediction,raw)
        row.update(**case,heldout_scored=scored,independent_validation=False,
            resource_prediction_source=fold['resources'],
            predicted_candidate_resource_estimates={key:proposed[key] for key in
                ('predicted_startup_joules','predicted_prephase_peak_mib')})
        rows.append(row);by_seed[case['seed']].append(row);refs.append(scored)
        print('NUMERICAL_V2_CASE',case['flow_id'],row['known'],flush=True)
    rankings={str(seed):original.core.ranking(items,audit['contracts'][seed]['baseline'],
        tie_tolerance=audit['contracts'][seed]['tie_tolerance']) for seed,items in sorted(by_seed.items())}
    value=copy.deepcopy(previous)
    value.update(schema='gsdeploy.run_aware_H600_numerical_v2_training_report.v1',
        created_ns=time.perf_counter_ns(),previous_report=receipt(original_report),
        numerical_source=receipt(numerical.__file__),entry_source=receipt(__file__),
        numerical_change='Complete the segment at its selected representable completion timestamp; unchanged service coefficients, event budget, metrics and thresholds',
        candidates=rows,by_seed=rankings,known=sum(r['known'] for r in rows),
        complete_prediction_coverage=all(r['known']==r['requested'] for r in rows),
        predictive_fitting_performed=False,thresholds_tuned=False,
        model_verifications=audit['model_verifications'],source_receipts=refs,
        scope='Frozen H600 whole-seed training CV; only event-completion arithmetic changed; original failed report preserved')
    value['original_threshold_assessment']=evaluate(value,
        {'seeds':sorted(audit['contracts']),'routes':sorted(original.ROUTES)},LIMITS,audit['contracts'])
    for ref in refs:checked(ref)
    output.mkdir(parents=True,exist_ok=False);save(output/'report.json',value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-report',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();value=run(args.original_report,args.output)
    print({key:value[key] for key in ('known','requested','complete_prediction_coverage')})
