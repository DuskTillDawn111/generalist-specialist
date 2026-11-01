"""Retrospective training diagnostic of the frozen H90 model on H600 flows.

No fitting or prospective-validation claim. Public arrival and input traces
alone enter the forecast. Old H90 power/startup/memory are explicit transfer
hypotheses, not qualified H600 resources. Completed training outcomes score
forecasts only; they never change the frozen model.
"""
import argparse
from pathlib import Path
from .flow_crossfit_report import predict,compare,scored_heldout
from .native_flow import receipt,checked,save
from .probe_validation import read
from .training_campaign_v14 import verify_campaign
from .variant_full_audit import verify_variant_full_model


def run(campaign_path,model_path,resources_path,quality_path,run_id,output):
    if Path(output).exists():raise FileExistsError(output)
    campaign=verify_campaign(campaign_path)
    job=next(j for j in campaign['main_jobs'] if j['run_id']==run_id)
    raw_path=Path(job['output'])/'raw.json'
    raw,raw_ref=scored_heldout(receipt(raw_path))
    plan=read(checked(job['plan']))
    if plan.get('phase')!='predictor_training' or plan.get('split')!='training':
        raise ValueError('Training-only diagnostic required')
    candidate=next(c for c in plan['candidates'] if c['id']==job['candidate_id'])
    model=read(model_path);resources=read(resources_path);quality=read(quality_path)
    # Full recursive verification once per report, no mutation of frozen model.
    audit=verify_variant_full_model(model_path)
    if any(r['path']==str(raw_path.resolve()) for r in audit['training_raw_receipts']):
        raise ValueError('H600 outcome is already part of the supposed H90 model')
    prediction=predict(plan,candidate,model,resources,quality.get('loss_by_family_task',quality))
    comparison=compare(plan,candidate,prediction,raw)
    value={'schema':'gsdeploy.training_only_H90_to_H600_transfer_diagnostic.v1',
        'campaign':receipt(campaign_path),'run_id':run_id,'plan':job['plan'],'raw':raw_ref,
        'model':receipt(model_path),'resources':receipt(resources_path),'quality':receipt(quality_path),
        'source':receipt(__file__),'model_verification':audit,'comparison':comparison,'prediction':prediction,
        'independent_validation':False,'GPU_started':False,'fitting_performed':False,
        'old_H90_resource_transfer_validated':False,'common_startup_omission_not_absolute_energy':True,
        'final_gain_eligible':False,'prediction_created_after_training_collection':True}
    save(output,value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','model','resources','quality','run-id','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();d=run(a.campaign,a.model,a.resources,a.quality,a.run_id,a.output)
    print(d['run_id'],d['comparison']['timing_errors']['service'],d['comparison']['queue_error'])
