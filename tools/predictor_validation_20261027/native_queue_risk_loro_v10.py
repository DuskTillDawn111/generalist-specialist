"""Leave one TRAINING run out; no validation relabeling or calibration claim."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.recovery_risk import (
    RecoveryRiskProvider, _label, _vector, fit_recovery_risk,
)
from gsdeploy.resources import identity
from .background_collect import receipt, save
from .multi_native_composite_v5 import checked

PARAMETERS=dict(n_estimators=40,learning_rate=.2,minimum_leaf=2)


def counts(values):
    found=Counter(values)
    return {str(label):found[label] for label in (0,1)}


def metrics(rows):
    known=[r for r in rows if r['status']=='KNOWN']
    positive=[r['risk_score'] for r in known if r['label']==1]
    negative=[r['risk_score'] for r in known if r['label']==0]
    return dict(total=len(rows),supported=len(known),unknown=len(rows)-len(known),
        support_coverage=len(known)/len(rows) if rows else None,
        all_class_counts=counts(r['label'] for r in rows),
        supported_class_counts=counts(r['label'] for r in known),
        unknown_class_counts=counts(r['label'] for r in rows if r['status']=='UNKNOWN'),
        rank_ROC_AUC=(sum((a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative))
                      if positive and negative else None),
        AUC_unknown_reason=None if positive and negative else 'BOTH_SUPPORTED_CLASSES_REQUIRED',
        Brier_score_error=(sum((r['risk_score']-r['label'])**2 for r in known)/len(known) if known else None),
        Brier_scope='Squared score error on supported training-run-out rows only; does not establish probability calibration',
        calibrated_probability=False,independent_validation=False,
        group_disjoint=False,confidence_interval=None)


def run(report_path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source_paths=[Path(__file__),Path('gsdeploy/forecasting/recovery_risk.py').resolve()]
    before=[receipt(p) for p in source_paths]
    for p in source_paths:
        dest=output/'sources'/p.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
    report_ref=receipt(report_path);original=checked(report_ref)
    if original['status']!='FITTED_TRAINING_ONLY_UNCALIBRATED' or not original['source_closure_unchanged']:
        raise ValueError('Frozen completed risk03 training fit required')
    extracted=checked(original['rows']);original_model=checked(original['model'])
    rows=[r for e in extracted for r in e['rows']]
    if len(extracted)!=4 or len(rows)!=68 or any(e['omitted'] for e in extracted):
        raise ValueError('All four complete 17-cutoff training runs required')
    target=original['protocol']['target'];context=original['execution_context_sha256']
    regenerated=fit_recovery_risk(rows,context,**target,**PARAMETERS)
    if regenerated!=original_model:
        raise ValueError('Fixed original hyperparameters do not reproduce risk03 model')
    registries={};evidence=[report_ref,original['rows'],original['model'],original['work_feature_provenance']]
    for item in extracted:
        review=checked(item['review']);plan=checked(review['evidence']['plan'])
        evidence.extend([item['review'],review['evidence']['plan'],review['evidence']['raw']])
        run_rows=item['rows'];rid=plan['run_id'];offsets=plan['cutoff_offsets_ns']
        if (len(run_rows)!=17 or len(offsets)!=17 or {r['run_id'] for r in run_rows}!={rid}
                or {r['group_id'] for r in run_rows}!={plan['group_id']}
                or [r['decision_time_ns']-r['features']['declared_at_ns'] for r in run_rows]!=offsets
                or any(r['split']!='training' or r['kind']!='measured' for r in run_rows)):
            raise ValueError('Original complete training cutoff registry changed')
        registries[rid]=run_rows
    if len(registries)!=4 or len({r['id'] for r in rows})!=68:
        raise ValueError('Distinct complete original runs and IDs required')
    folds=[];predictions=[]
    for rid,held in registries.items():
        train=[r for r in rows if r['run_id']!=rid]
        if len(train)!=51 or len({r['run_id'] for r in train})!=3 or len({r['group_id'] for r in train})!=2:
            raise ValueError('Exactly three training runs spanning two families required')
        model=fit_recovery_risk(train,context,**target,**PARAMETERS)
        model_path=output/(rid+'-model.json');save(model_path,model)
        provider=RecoveryRiskProvider(model,context);forecast=[]
        for row in held:
            _,vector,label=_label(row,context,'training',**target)
            result=dict(id=row['id'],run_id=rid,group_id=row['group_id'],split=row['split'],
                cutoff_offset_ns=row['decision_time_ns']-row['features']['declared_at_ns'],
                label=label,model_sha256=model['model_sha256'])
            try:
                prediction=provider.predict(row['features'],row['decision_time_ns'])
                result.update(status='KNOWN',risk_score=prediction['risk_score'])
            except UnknownEvidence as exc:
                result.update(status='UNKNOWN',risk_score=None,reason=str(exc),
                    outside_support=[dict(feature=name,value=float(value),minimum=lo,maximum=hi)
                        for name,value,lo,hi in zip(model['feature_names'],vector,model['support_min'],model['support_max'])
                        if value<lo or value>hi])
            forecast.append(result)
        predictions.extend(forecast)
        folds.append(dict(held_training_run=rid,held_group=held[0]['group_id'],
            train_runs=model['provenance']['run_ids'],train_groups=model['provenance']['group_ids'],
            training_count=len(train),training_class_counts=model['training_class_counts'],
            original_cutoffs_preserved=len(forecast)==17,
            held_group_overlaps_training=held[0]['group_id'] in model['provenance']['group_ids'],
            model=receipt(model_path),predictions=forecast,metrics=metrics(forecast)))
    after=[receipt(p) for p in source_paths]
    if before!=after:raise ValueError('LORO producer changed during diagnostic')
    for reference in evidence:checked(reference)
    result=dict(schema='gsdeploy.native_queue_risk_leave_one_training_run_out.v10',
        status='COMPLETE_TRAINING_ONLY_RUN_OUT_DIAGNOSTIC',original_fit=report_ref,evidence=evidence,
        hyperparameters=PARAMETERS,original_model_reproduced_exactly=True,
        rows_split_unchanged='training',group_disjoint=False,independent_validation=False,
        all_68_cutoffs_preserved=len(predictions)==68,folds=folds,pooled_metrics=metrics(predictions),
        family_holdout_gap='Only burst_A and burst_B, with two runs each: leaving either family out leaves two training runs, below the fixed minimum of three runs and two groups. A fresh third family is required for group-disjoint validation.',
        pooled_score_scope='Each row uses its own three-run model; pooled ranking is a descriptive diagnostic across four fitted models.',
        target_tuned=False,hyperparameters_tuned=False,evaluate_holdout_called=False,
        risk_probability_calibrated=False,physical_risk_accuracy_established=False,
        six_term_utility_changed=False,stability_rule_changed=False,admission_changed=False,
        source_before=before,source_after=after,source_closure_unchanged=True,GPU_started=False)
    save(output/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-report',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.fit_report,args.output)
    print(json.dumps(dict(status=result['status'],pooled_metrics=result['pooled_metrics'])))


if __name__=='__main__':main()
