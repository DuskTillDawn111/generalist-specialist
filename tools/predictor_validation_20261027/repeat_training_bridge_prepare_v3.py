"""Prepare a new independent capture with the exact audited domain identity.

Frozen source copies move to a new output directory. The original verified
large-weight-stat receipt stays fixed because its path is part of the current
context hash. This does not reuse a measured trial, process, or trace.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from .kernel_training_campaign_v3 import DOMAINS, audit
from .native_collect import context_identity, receipt, save
from .profiling import digest
from .smoke_verifier_v3 import check_receipt


def relocated_context(context, output):
    context=copy.deepcopy(context)
    for index,source in enumerate(context['execution_sources']):
        old=source['frozen']
        if not check_receipt(old):raise ValueError('Changed reference frozen source')
        target=Path(output)/'source-snapshot'/str(index)/Path(old['path']).name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if receipt(target)['sha256']!=old['sha256']:raise ValueError('Conflicting frozen source')
        else:shutil.copyfile(old['path'],target)
        source['frozen']=receipt(target)
    context['sources']=[s['frozen'] for s in context['execution_sources']]
    return context


def prepare(reference,output):
    reference,output=Path(reference).resolve(),Path(output).resolve()
    old=json.loads((reference/'campaign.json').read_text())
    checked={}
    for domain in DOMAINS:
        audit(reference/domain)
        report=reference/domain/'strict-training-review-v3/report.json'
        doc=json.loads(report.read_text())
        if doc.get('status')!='PASS' or not check_receipt(doc['raw'],reference/domain/'raw.json'):
            raise ValueError('Passing reference training capture required: '+domain)
        checked[domain]=receipt(report)
    output.mkdir(parents=True,exist_ok=False)
    runs={}
    for domain in DOMAINS:
        context=json.loads((reference/domain/'context.json').read_text())
        context=relocated_context(context,output)
        plan=json.loads((reference/domain/'plan.json').read_text())
        if context_identity(context)!=plan['execution_context_sha256']:
            raise ValueError('Relocation changed execution identity')
        selection=plan['input_selection']
        if not check_receipt(selection):raise ValueError('Changed reference selection')
        selected=output/'input-selection.json'
        if not selected.exists():shutil.copyfile(selection['path'],selected)
        plan['input_selection']=receipt(selected)
        plan.pop('plan_sha256',None);plan['plan_sha256']=digest(plan)
        run=output/domain;run.mkdir()
        save(run/'context.json',context);save(run/'plan.json',plan)
        runs[domain]=dict(plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),
                          execution_context_sha256=context_identity(context),jobs=len(plan['jobs']))
    if not check_receipt(old['design']):raise ValueError('Changed reference design')
    shutil.copyfile(old['design']['path'],output/'training-design.json')
    result=dict(schema=old['schema'],status='PREPARED_CPU_ONLY',runs=runs,
                design=receipt(output/'training-design.json'),validation_started=False,full_training_executable=False)
    save(output/'campaign.json',result)
    for domain in DOMAINS:audit(output/domain)
    save(output/'repeat-preparation.json',dict(schema='gsdeploy.independent_training_capture_preparation.v3',
        reference_campaign=receipt(reference/'campaign.json'),reference_capture_reviews=checked,
        evaluator=receipt(__file__),same_domain_context_identity_preserved=True,
        reused_measurements=False,independent_startup_capture_required=True,
        stable_large_weight_receipt='Reference receipt retained exactly; execution audit checks its content and current actual weight stat.'))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.reference,args.output),indent=2))


if __name__=='__main__':main()
