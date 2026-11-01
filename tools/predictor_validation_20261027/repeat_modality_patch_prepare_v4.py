"""CPU preparation for a fresh G2 modality capture, reusing frozen sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .native_collect import receipt,save,context_identity
from .small_training_patch_campaign_v4 import audit
from .smoke_verifier_v3 import check_receipt


def prepare(reference,output,review_paths):
    reference,output=Path(reference).resolve(),Path(output).resolve()
    reviews={}
    for domain in ('metadata','light-kernel','natural'):
        run=reference/'modalities'/domain;audit(run)
        path=Path(review_paths[domain]).resolve();report=json.loads(path.read_text())
        if report.get('schema')!='gsdeploy.startup_training_collection_strict_review.v4' or report.get('status')!='PASS':
            raise ValueError('Full strict reference capture PASS required: '+domain)
        for field,name in [('raw','raw.json'),('plan','plan.json'),('context','context.json')]:
            if not check_receipt(report[field],run/name):raise ValueError('Reference review actual evidence mismatch')
        if not all(check_receipt(s) for s in report['evaluator_sources']):raise ValueError('Changed reference evaluator')
        reviews[domain]=receipt(path)
    output.mkdir(parents=True,exist_ok=False);runs={}
    for domain in ('metadata','light-kernel','natural'):
        old=reference/'modalities'/domain;run=output/'modalities'/domain;run.mkdir(parents=True)
        # References inside these documents stay byte-identical, including
        # the original large-weight stat and selected-input receipts.
        for name in ('plan.json','context.json'):shutil.copyfile(old/name,run/name)
        plan=json.loads((run/'plan.json').read_text());context=json.loads((run/'context.json').read_text())
        if context_identity(context)!=plan['execution_context_sha256']:raise ValueError('Execution identity changed')
        audit(run)
        runs['modalities/'+domain]=dict(plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),
            execution_context_sha256=plan['execution_context_sha256'],jobs=len(plan['jobs']),requests=4)
    usage=shutil.disk_usage(output)
    result=dict(schema='gsdeploy.small_startup_training_patch_campaign.v4',status='PREPARED_CPU_ONLY',runs=runs,
        independent_capture_count=0,no_gpu_started=True,validation_started=False,
        reference_campaign=receipt(reference/'campaign.json'),reference_reviews=reviews,evaluator=receipt(__file__),
        source_snapshot_copied=False,measurements_or_processes_reused=False,
        raw_capture_policy='Every execution must launch fresh workers and write exclusive new raw/traces; prior raw is never copied.',
        execution_order=['modalities/metadata','modalities/light-kernel','modalities/natural'],
        free_disk_bytes_at_preparation=usage.free,free_disk_reserve_bytes=12*2**30,
        size_scenario=dict(reference_metadata_bytes=1032721990,reference_light_bytes=574799100,
            reference_natural_bytes=4386464,estimated_additional_bytes=3*2**30,
            guaranteed_upper_bound=False,remaining_after_reserve_and_scenario_bytes=usage.free-15*2**30),
        inherited_budget_limit='Original frozen runner checks disk/time at phase/job boundaries, not a continuously enforced watchdog.')
    save(output/'campaign.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True);parser.add_argument('--output',required=True)
    for domain in ('metadata','light-kernel','natural'):parser.add_argument('--'+domain+'-review',required=True)
    args=parser.parse_args();paths={d:getattr(args,d.replace('-','_')+'_review') for d in ('metadata','light-kernel','natural')}
    result=prepare(args.reference,args.output,paths)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
