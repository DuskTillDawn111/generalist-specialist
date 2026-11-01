"""Additional one-input ChatQA training shape; original priming stays fixed."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from .native_collect import receipt,save,context_identity
from .operator_profile_campaign_v2 import audit as source_audit
from .profiling import digest
from .smoke_verifier_v3 import check_receipt
from .startup_collect_v4 import collect


def prepare(reference,selection_report,output):
    reference,selection_report,output=Path(reference).resolve(),Path(selection_report).resolve(),Path(output).resolve()
    selected=json.loads(selection_report.read_text())
    if selected.get('purpose')!='ADDITIONAL_TRAINING_NOT_HELDOUT_VALIDATION':raise ValueError('Explicit additional training selection required')
    for name in ('selected_input','catalog','prior_split_selection'):
        if not check_receipt(selected[name]):raise ValueError('Changed selection evidence')
    item=json.loads(Path(selected['selected_input']['path']).read_text());pool_id=item['pool_id']
    if pool_id!=selected['selected']['pool_id'] or pool_id in selected['excluded_pool_ids'] or item['task_type']!='text':raise ValueError('Eligible exact text training input required')
    for domain in ('metadata','light-kernel','natural'):source_audit(reference/'modalities'/domain)
    output.mkdir(parents=True,exist_ok=False)
    extra=[]
    for index,path in enumerate([Path(__file__).resolve(),selection_report,Path(selected['selected_input']['path'])]):
        frozen=output/'extra-source-snapshot'/str(index)/path.name;frozen.parent.mkdir(parents=True)
        shutil.copyfile(path,frozen);extra.append(dict(original_path=str(path),frozen=receipt(frozen)))
    runs={}
    for domain in ('metadata','light-kernel','natural'):
        old=reference/'modalities'/domain
        plan=copy.deepcopy(json.loads((old/'plan.json').read_text()));context=copy.deepcopy(json.loads((old/'context.json').read_text()))
        if pool_id in {p['pool_id'] for p in plan['inputs']}:raise ValueError('New shape must not silently enter inherited priming')
        frozen_pool=next(s['frozen']['path'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl'))
        matching=[r for r in map(json.loads,Path(frozen_pool).read_text().splitlines()) if r['pool_id']==pool_id]
        if matching!=[item]:raise ValueError('Selected input must exactly match the already frozen public runtime catalog')
        context['execution_sources']+=copy.deepcopy(extra);context['sources']+=[s['frozen'] for s in extra]
        context.update(additional_training_shape=dict(pool_id=pool_id,selection=receipt(selection_report),
            expected_input_tokens=selected['selected']['input_tokens'],priming_added=False,
            scope='ADDITIONAL_TRAINING_ONLY_NOT_INDEPENDENT_VALIDATION'))
        sha=context_identity(context)
        job=dict(context='solo-text#0-new-M',inputs={'text#0':pool_id},tasks={'text#0':'text'},
            start_offsets_ms={'text#0':0},split='training',repeat=0,phase='additional-new-shape-training',
            validation_view='ADDITIONAL_TRAINING_NOT_HELDOUT_VALIDATION')
        job['id']=digest(job)
        plan.update(schema='gsdeploy.additional_shape_training_plan.v4',execution_context_sha256=sha,
            shard='additional-new-shape',jobs=[job],paired_jobs_sha256=digest([job]),
            additional_job_bound_training_input=dict(pool_id=pool_id,selected_input=selected['selected_input'],
                expected_input_tokens=selected['selected']['input_tokens'],inherited_priming_inputs_unchanged=True),
            claim='ONE_NEW_SHAPE_TRAINING_REQUEST_NOT_TWO_RUN_BASELINE_NOT_INDEPENDENT_VALIDATION')
        plan.pop('plan_sha256',None);plan['plan_sha256']=digest(plan)
        run=output/domain;run.mkdir();save(run/'context.json',context);save(run/'plan.json',plan)
        runs[domain]=dict(plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),
            execution_context_sha256=sha,jobs=1,requests=1)
    result=dict(schema='gsdeploy.additional_shape_training_campaign.v4',status='PREPARED_CPU_ONLY',runs=runs,
        reference_campaign=receipt(reference/'campaign.json'),selection=receipt(selection_report),
        training_only=True,validation_started=False,GPU_started=False,
        original_physical_owners=4,additional_pool_id=pool_id,expected_input_tokens=selected['selected']['input_tokens'],
        priming_cohort='Exactly inherited plan.inputs; new job-bound text input is not primed.',
        startup_prediction_reuse='No implicit reuse: this wrapper/source packet produces new context identities; startup raw is collected again.',
        independent_capture_requirement='Single new observed shape per domain only; does not create a two-run baseline.',
        copied_prior_source_snapshots=False)
    save(output/'campaign.json',result)
    for domain in runs:audit(output/domain)
    return result


def audit(run):
    run=Path(run).resolve();result=source_audit(run);plan=json.loads((run/'plan.json').read_text())
    if plan.get('schema')!='gsdeploy.additional_shape_training_plan.v4' or plan.get('split')!='training' or len(plan['jobs'])!=1:
        raise ValueError('Exact one-job additional training shape required')
    job=plan['jobs'][0]
    if set(job['inputs'])!={'text#0'} or job['tasks']!={'text#0':'text'} or job['phase']!='additional-new-shape-training':raise ValueError('Only retained text owner zero new-shape solo permitted')
    if next(iter(job['inputs'].values())) in {p['pool_id'] for p in plan['inputs']}:raise ValueError('New shape must not alter inherited priming')
    return result


def execute(run):
    run=Path(run).resolve();audit(run);plan=json.loads((run/'plan.json').read_text())
    raw=collect(run,trace=plan['domain']!='natural')
    return dict(status=raw['status'],raw=receipt(run/'raw.json'),requests=len(raw['request_records']))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--reference',required=True);p.add_argument('--selection-report',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('audit');p.add_argument('--run',required=True)
    p=sub.add_parser('execute');p.add_argument('--run',required=True);p.add_argument('--execute',required=True,action='store_true')
    args=parser.parse_args();r=prepare(args.reference,args.selection_report,args.output) if args.op=='prepare' else audit(args.run) if args.op=='audit' else execute(args.run)
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
