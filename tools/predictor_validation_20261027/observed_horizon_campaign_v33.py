"""Four actual capture diagnostics, sealed observer replay, and final candidates.

All captures are training-family evidence. Final candidates are fit once with
fixed hyperparameters and never evaluated on their own training outcomes.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil

from gsdeploy.forecast_errors import UnknownEvidence
from . import horizon_recovery_risk_v31 as risk
from . import horizon_recovery_training_v31 as training
from . import joint_horizon_risk_v35 as joint
from . import observed_horizon_risk_sidecar_v32 as sidecar
from .idle_owner_selection_audit_v20 import read
from .owner_lifecycle_mark_v21 import receipt

SCHEMA='gsdeploy.observed_horizon_campaign.v33'


def run(domain_path,visible_reports,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    domain=read(domain_path);risk.check_domain(domain,require_actual=True)
    if len(visible_reports)!=4:raise ValueError('Exactly four complete actual natural reports required')
    visible={}
    refs=[receipt(__file__),receipt(training.__file__),receipt(joint.__file__),receipt(domain_path),*sidecar.static_source_closure()]
    for path in visible_reports:
        report=read(path)
        if report['run_id'] in visible:raise ValueError('Distinct whole captures required')
        visible[report['run_id']]=path
        refs.extend([receipt(path),report['features'],report['labels'],report['split_manifest']])
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Frozen campaign source mismatch')
    # v31 validates all original domain members and full real rows before fit.
    diagnostic=training.run(domain_path,visible_reports,output/'training-diagnostic',fit_diagnostic=True)
    fold_models=read(diagnostic['fold_models']['path'])['models'];sealed=[]
    for index,item in enumerate(fold_models):
        root=output/'whole-capture-replays'/str(index)
        root.mkdir(parents=True)
        model_path=root/'frozen-model.json'
        model_path.write_text(json.dumps(item['model'],indent=2)+'\n')
        heldout=item['heldout_run_id'];report=visible[heldout]
        prediction=sidecar.run(model_path,domain_path,report,root/'prediction')
        evaluation=sidecar.evaluate_sealed(root/'prediction/report.json',report,model_path,root/'evaluation')
        sealed.append(dict(horizon_ns=item['horizon_ns'],heldout_run_id=heldout,
            model=receipt(model_path),prediction=receipt(root/'prediction/report.json'),
            evaluation=receipt(root/'evaluation/report.json'),scored_n=evaluation['scored_n'],
            unknown_n=evaluation['unknown_n'],not_at_risk_n=evaluation['not_at_risk_n'],
            brier_score=evaluation['brier_score'],rank_auc=evaluation['rank_auc']))
    joint_evaluations=[]
    for index,heldout in enumerate(sorted(visible)):
        group=[item for item in sealed if item['heldout_run_id']==heldout]
        if len(group)!=3:
            joint_evaluations.append(dict(heldout_run_id=heldout,status='JOINT_UNKNOWN_MISSING_FROZEN_HORIZON_MODEL'))
            continue
        root=output/'joint-horizon-replays'/str(index)
        joint_report=joint.seal([item['prediction']['path'] for item in group],root/'prediction')
        evaluation=joint.evaluate(root/'prediction/report.json',visible[heldout],root/'evaluation')
        joint_evaluations.append(dict(heldout_run_id=heldout,
            prediction=receipt(root/'prediction/report.json'),evaluation=receipt(root/'evaluation/report.json'),
            raw_and_fixed_projection_per_horizon=evaluation['per_horizon'],
            raw_monotonicity_violation_n=joint_report['raw_monotonicity_violation_n'],
            maximum_raw_monotonicity_violation=joint_report['maximum_raw_monotonicity_violation'],
            joint_unknown_n=evaluation['joint_unknown_n']))
    with gzip.open(diagnostic['rows']['path'],'rt',encoding='utf-8') as stream:
        rows=[json.loads(line) for line in stream]
    finals=[];final_root=output/'final-candidates';final_root.mkdir()
    for horizon in risk.HORIZONS_NS:
        population=[row for row in rows if row['horizon_ns']==horizon]
        try:
            candidate=risk.candidate_fit(population,domain,horizon_ns=horizon)
            model_path=final_root/('horizon-'+str(horizon)+'.json')
            model_path.write_text(json.dumps(candidate,indent=2)+'\n')
            finals.append(dict(horizon_ns=horizon,status='FROZEN_ALL_FOUR_TRAINING_CAPTURE_CANDIDATE',
                model=receipt(model_path),training_class_counts=candidate['training_class_counts'],
                training_run_ids=candidate['provenance']['run_ids']))
        except UnknownEvidence as exc:
            finals.append(dict(horizon_ns=horizon,status='UNKNOWN',reason=str(exc)))
    after=[receipt(ref['path']) for ref in refs]
    if after!=refs:raise ValueError('Frozen campaign sources changed')
    project=Path(__file__).resolve().parents[2];snapshots=[]
    for source_path in sorted({ref['path'] for ref in refs if Path(ref['path']).suffix=='.py'}):
        source=Path(source_path);dest=output/'source-snapshot'/source.relative_to(project)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        original=receipt(source);snapshot=receipt(dest)
        if original['sha256']!=snapshot['sha256']:raise ValueError('Campaign source snapshot differs')
        snapshots.append(dict(original=original,snapshot=snapshot))
    if [receipt(ref['path']) for ref in refs]!=refs:raise ValueError('Frozen campaign sources changed while snapshotting')
    complete=(len(sealed)==12 and len(joint_evaluations)==4 and all('evaluation' in item for item in joint_evaluations)
        and all(f['status']=='FROZEN_ALL_FOUR_TRAINING_CAPTURE_CANDIDATE' for f in finals))
    report=dict(schema=SCHEMA,status='FOUR_CAPTURE_DIAGNOSTICS_AND_FINAL_CANDIDATES_FROZEN' if complete else 'PARTIAL_DIAGNOSTICS_EXPLICIT_UNKNOWN',
        risk_domain_sha256=domain['risk_domain_sha256'],diagnostic=receipt(output/'training-diagnostic/report.json'),
        sealed_whole_capture_replays=sealed,final_candidates=finals,
        joint_horizon_replays=joint_evaluations,joint_horizon_policy=joint.POLICY,
        source_receipts_before=refs,source_receipts_after=after,training_only=True,GPU_started=False,
        source_snapshots=snapshots,
        final_model_training_outcomes_scored=False,validation_label_model_selection=False,
        fit_hyperparameters=dict(n_estimators=40,learning_rate=.2,minimum_leaf=2),
        calibrated_probability=False,input_generalization_claimed=False,independent_accuracy_established=False,
        physical_predictor_qualified=False,rolling_forecast_implemented=False,six_term_utility_changed=False,rho_changed=False,
        limitations=['Whole-capture diagnostics share recipe/content; correlated windows do not establish independent accuracy.',
            'Final candidates bind only this exact four-member execution domain; a new execution context needs a new explicit qualified domain before prediction.',
            'No parameter/model selection based on held-out outcomes; final candidates use all four training captures once.',
            'Scores remain uncalibrated and UNKNOWN outside training coordinate support.'])
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--risk-domain',required=True);parser.add_argument('--visible-report',action='append',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();report=run(args.risk_domain,args.visible_report,args.output)
    print(json.dumps(dict(status=report['status'],sealed_folds=len(report['sealed_whole_capture_replays']),final_candidates=len(report['final_candidates'])),indent=2))
