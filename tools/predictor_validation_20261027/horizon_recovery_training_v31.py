"""Extract true horizon binary targets; optional four-natural-capture diagnostic."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from . import horizon_recovery_risk_v31 as risk
from .idle_owner_selection_audit_v20 import read
from .owner_lifecycle_mark_v21 import receipt


def extract_one(report_path,domain):
    report=read(report_path)
    if report.get('training_only') is not True or report.get('validation_labels_used') is not False or report.get('model_fitted') is not False:
        raise ValueError('Actual v30 training-only observations required')
    refs=[receipt(report_path),report['features'],report['labels'],report['split_manifest']]
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Changed visible-risk source')
    with gzip.open(report['features']['path'],'rt',encoding='utf-8') as stream:
        prefix={row['id']:row for row in (json.loads(line) for line in stream)}
    if len(prefix)!=report['counts']['prefix_snapshots']:raise ValueError('Prefix row count/identity mismatch')
    rows=[];counts=Counter();seen=set()
    with gzip.open(report['labels']['path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            label=json.loads(line);source=prefix[label['id']];horizon=label['offline_labels']['horizon_ns']
            key=label['id'],horizon
            if key in seen:raise ValueError('Duplicate prefix/horizon label')
            seen.add(key)
            row=risk.training_row(source,label,domain)
            counts[row['label_status']]+=1
            if row['label_status']!='NOT_AT_RISK_CURRENT_BACKLOG_EMPTY':rows.append(row)
    if len(seen)!=report['counts']['label_rows']:raise ValueError('Risk label source population mismatch')
    if [receipt(ref['path']) for ref in refs]!=refs:raise ValueError('Source bytes changed during risk extraction')
    return rows,dict(source_report=refs[0],run_id=report['run_id'],group_id=report['group_id'],
        source_prefix_count=len(prefix),label_status_counts=dict(counts),retained_positive_backlog_rows=len(rows),sources=refs)


def run(domain_path,visible_reports,output,*,fit_diagnostic=False):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    domain=read(domain_path);risk.check_domain(domain,require_actual=False)
    refs=[receipt(domain_path),receipt(__file__),receipt(risk.__file__),receipt(risk.legacy.__file__)]
    rows=[];extractions=[]
    for report in visible_reports:
        new,provenance=extract_one(report,domain);rows.extend(new);extractions.append(provenance);refs.extend(provenance['sources'])
    if len({row['id'] for row in rows})!=len(rows):raise ValueError('Duplicate rows/runs across input reports')
    folds=[];models=[];fit_requested=fit_diagnostic
    if fit_diagnostic:
        risk.check_domain(domain,require_actual=True)
        if set(r['original_execution_context_sha256'] for r in rows)!=set(domain['member_execution_context_sha256']) or len({r['run_id'] for r in rows})!=4:
            raise UnknownEvidence('ALL_FOUR_DISTINCT_ACTUAL_NATURAL_CAPTURE_INPUTS_REQUIRED')
        for horizon in risk.HORIZONS_NS:
            population=[row for row in rows if row['horizon_ns']==horizon]
            for heldout in sorted({row['run_id'] for row in population}):
                train,test=risk.whole_capture_partition(population,heldout)
                try:
                    model=risk.candidate_fit(train,domain,horizon_ns=horizon)
                    result=risk.capture_holdout(model,test,domain)
                    rate=sum(row['label'] for row in train if row['label'] is not None)/sum(row['label'] is not None for row in train)
                    common=[p for p in result['predictions'] if p['label'] is not None and p['risk_score'] is not None]
                    result['training_rate_baseline_on_candidate_common_support']=dict(training_rate=rate,n=len(common),
                        brier_score=None if not common else sum((rate-p['label'])**2 for p in common)/len(common))
                    models.append(dict(horizon_ns=horizon,heldout_run_id=heldout,model=model))
                    folds.append(dict(horizon_ns=horizon,heldout_run_id=heldout,status='TRAINING_CAPTURE_DIAGNOSTIC_ONLY',**result))
                except UnknownEvidence as exc:
                    folds.append(dict(horizon_ns=horizon,heldout_run_id=heldout,status='UNKNOWN',reason=str(exc)))
    after=[receipt(ref['path']) for ref in refs]
    if refs!=after:raise ValueError('Risk inputs changed during preparation/fit')
    output.mkdir(parents=True)
    with gzip.open(output/'positive-backlog-horizon-rows.jsonl.gz','wt',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    (output/'fold-models.json').write_text(json.dumps(dict(schema=risk.SCHEMA+'.fold_models',models=models),indent=2)+'\n')
    result=dict(schema=risk.SCHEMA,status='SAME_FAMILY_WHOLE_CAPTURE_TRAINING_DIAGNOSTIC' if fit_requested else 'TRUE_HORIZON_ROWS_READY_NO_MODEL',
        risk_domain_sha256=domain['risk_domain_sha256'],actual_execution_qualification=domain['actual_execution_qualification'],
        original_context_SHA_preserved=True,extractions=extractions,rows=receipt(output/'positive-backlog-horizon-rows.jsonl.gz'),
        fold_models=receipt(output/'fold-models.json'),folds=folds,source_receipts_before=refs,source_receipts_after=after,
        known_target_class_counts={str(h):dict(Counter(str(row['label']) for row in rows if row['horizon_ns']==h)) for h in risk.HORIZONS_NS},
        protocol=dict(population='Current positive reader-visible backlog only',
            target='First visible waiting-queue clearance strictly later than H',horizons_ns=list(risk.HORIZONS_NS),
            exact_recovery_censor_at_H_is_known_positive=True,administrative_censor_before_H_is_unknown=True,
            already_observed_clearance_before_H_is_known_negative=True,
            fit_hyperparameters=dict(n_estimators=40,learning_rate=.2,minimum_leaf=2),split_unit='whole capture/run_id',
            shared_recipe_content_groups_preserved=True,feature_scaling='Raw positive public/visible covariates; input_size never converted into time'),
        training_only=True,GPU_started=False,model_fitted=bool(models),validation_labels_used=False,
        calibrated_probability=False,input_generalization_claimed=False,physical_predictor_qualified=False,
        six_term_utility_changed=False,rho_changed=False,rolling_forecast_implemented=False,
        limitations=['All recipes/content may be shared; whole-capture holdout is a training diagnostic, not input-family generalization.',
            'Reader receipt delay means visible queue may differ from physical queue.',
            'Dense receipt cutoffs are correlated and can have unequal counts per capture; no independent-window confidence interval.',
            'Only train-supported feature rows receive candidate scores; UNKNOWN/coverage and baseline common support are explicit.',
            'The known binary event at H does not reveal exact censored clearance time or certify GPU progress.',
            'No old recovery_risk/gate/utility/rho behavior is modified.'])
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--risk-domain',required=True)
    parser.add_argument('--visible-report',action='append',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--fit-diagnostic',action='store_true')
    args=parser.parse_args();result=run(args.risk_domain,args.visible_report,args.output,fit_diagnostic=args.fit_diagnostic)
    print(json.dumps(dict(status=result['status'],known_target_class_counts=result['known_target_class_counts'],model_fitted=result['model_fitted']),indent=2))
