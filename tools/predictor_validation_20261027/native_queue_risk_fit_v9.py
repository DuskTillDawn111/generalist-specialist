"""Fit optional risk from strictly reviewed native TRAINING captures only.

This command is CPU-only. It rederives reference coordinates and rows, freezes
the model when both classes/runs/groups exist, and never reports training fit
as held-out accuracy or changes the original six-term/stability objective.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.recovery_risk import fit_recovery_risk
from .background_collect import receipt,save
from .multi_native_composite_v5 import checked
from .native_queue_risk_rows_v9 import ReconstructedReferenceWork,reviewed_rows


def fit(review_paths,bundle_path,solo_configuration_path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    refs=[receipt(p) for p in review_paths]
    if not refs:raise ValueError('Strictly reviewed training runs required')
    source_paths=[Path(__file__),Path(__file__).with_name('native_queue_risk_rows_v9.py'),
        Path(__file__).with_name('native_queue_busy_replay_v9.py'),
        Path(__file__).with_name('dag_busy_support_v9.py'),
        Path('gsdeploy/forecasting/recovery_risk.py').resolve()]
    before=[receipt(p) for p in source_paths]
    plan_ref=checked(refs[0])['evidence']['plan']
    provider=ReconstructedReferenceWork(receipt(bundle_path),plan_ref,receipt(solo_configuration_path))
    save(output/'work-feature-provenance.json',provider.provenance())
    extracted=[reviewed_rows(ref,provider) for ref in refs]
    contexts={e['protocol']['execution_context_sha256'] for e in extracted}
    protocols={e['protocol']['protocol_sha256'] for e in extracted}
    if len(contexts)!=1 or len(protocols)!=1:
        raise ValueError('One execution context and one frozen target protocol required')
    rows=[r for e in extracted for r in e['rows']]
    save(output/'rows.json',extracted)
    target=extracted[0]['protocol']['target'];status='UNKNOWN';reason=None;model_ref=None
    try:
        model=fit_recovery_risk(rows,next(iter(contexts)),**target)
        save(output/'model.json',model);model_ref=receipt(output/'model.json')
        status='FITTED_TRAINING_ONLY_UNCALIBRATED'
    except UnknownEvidence as exc:
        reason=str(exc)
    after=[receipt(p) for p in source_paths]
    if before!=after:raise ValueError('Risk producer sources changed during reconstruction')
    result=dict(schema='gsdeploy.native_queue_risk_training_diagnostic.v9',status=status,unknown_reason=reason,
        reviews=refs,execution_context_sha256=next(iter(contexts)),
        protocol=extracted[0]['protocol'],rows=receipt(output/'rows.json'),model=model_ref,
        work_feature_provenance=receipt(output/'work-feature-provenance.json'),
        source_before=before,source_after=after,source_closure_unchanged=True,
        usable_rows=len(rows),omitted_rows=sum(len(e['omitted']) for e in extracted),
        distinct_runs=len({r['run_id'] for r in rows}),distinct_groups=len({r['group_id'] for r in rows}),
        training_class_counts=(model['training_class_counts'] if model_ref else None),
        independent_validation=False,risk_probability_calibrated=False,
        stability_rule_changed=False,six_term_utility_changed=False,GPU_started=False)
    save(output/'report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviews',nargs='+',required=True);p.add_argument('--bundle',required=True)
    p.add_argument('--solo-configuration',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    result=fit(args.reviews,args.bundle,args.solo_configuration,args.output)
    print(json.dumps(dict(status=result['status'],usable_rows=result['usable_rows'],
        omitted_rows=result['omitted_rows'],model=result['model'],GPU_started=False)))


if __name__=='__main__':main()
