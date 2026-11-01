"""Training-only exact-eight native tuple intake; complete native re-review.

No caller Boolean certifies a row. Every source-bound native review is rerun,
actual stdout/ownership/window facts rederive, and every planned attempt enters
the intact whole-block tuple catalog. No numerical fit or GPU launch.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import tempfile

from . import twoFIFO_public181_resource_execute_v1 as executor
from . import twoFIFO_public181_resource_review_v1 as reviewer
from . import twoFIFO_public181_resource_campaign_v1 as campaign
from . import twoFIFO_public181_quality_labels_v1 as quality
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .stage_response_features_v1 import checked,receipt
from .twoFIFO_training_manifest_v1 import require

SCHEMA='gsdeploy.twoFIFO_public181_resource_native_tuple_intake.v1'
PASS='PASS_COMPLETE_PUBLIC181_RESOURCE_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION'


def load_complete(manifest_path,review_paths):
    manifest_ref=receipt(manifest_path);manifest=executor.audit(manifest_path)
    require(len(review_paths)<=8 and len({str(Path(p).resolve()) for p in review_paths})==len(review_paths),
            'At most eight distinct native source reviews required; unobserved plans remain UNKNOWN')
    refs=[receipt(path) for path in review_paths];documents=[checked(ref) for ref in refs]
    # All review and raw metadata roles qualify before any joint resource label.
    require(all(d.get('schema')==reviewer.SCHEMA and d.get('status') in (PASS,'NOT_ELIGIBLE')
        and d.get('resource_manifest')==manifest_ref and d.get('validation_labels_read') is False
        and d.get('training_feature_extraction_eligible') is (d.get('status')==PASS)
        and (d.get('status')!=PASS or d.get('issues')==[]) for d in documents),
        'Complete explicit training-only native PASS reports required before tuple labels')
    expected={a['plan']['path']:a for a in manifest['attempts']}
    supplied=[d['inputs']['plan.json']['path'] for d in documents]
    require(len(expected)==8 and len(supplied)==len(set(supplied)) and set(supplied)<=set(expected),
            'Each supplied native source plan at most once; all eight planned rows retained')
    for document in documents:
        raw=checked(document['raw']);plan=checked(document['inputs']['plan.json'])
        require(raw.get('schema')==executor.RAW_SCHEMA and raw.get('split')=='training'
            and raw.get('backend')=='native' and raw.get('kind')=='measured'
            and plan.get('schema')==executor.SCHEMA+'.attempt' and plan.get('split')=='training',
            'Native raw and plan roles required before any resource label')
    rows=[];sources=[manifest_ref]
    for report_ref,document in zip(refs,documents):
        for source in document['sources']:campaign.verify(source)
        item=expected[document['inputs']['plan.json']['path']]
        require(item['plan']==document['inputs']['plan.json'] and document['execution'] is not None,
                'Exact native source execution/plan binding required')
        with tempfile.TemporaryDirectory(prefix='public181-native-resource-review-') as directory:
            rebuilt=reviewer.review(document['raw']['path'],Path(directory)/'review',
                document['execution']['path'],manifest_path=manifest_path)
            require(checked(rebuilt)==document, 'Native source review cannot rederive exactly')
        raw=checked(document['raw']);plan=checked(item['plan'])
        eligible=document['status']==PASS
        derived=reviewer.resource_facts(raw) if eligible else None
        require(not eligible or derived==document['observed_resources'], 'Actual source resource tuple differs from native re-review')
        actual_quality=quality.measure(raw,plan,document['raw']) if eligible else None
        rows.append(dict(schema=campaign.SCHEMA+'.training_joint_tuple',split='training',
            **{k:item[k] for k in ('attempt_id','capture_id','execution_id','candidate_id','plan')},
            collection_protocol_identity=manifest['collection_protocol_identity'],
            public_case_sha256=plan['public_case_sha256'],joint_tuple=copy.deepcopy(derived['joint_tuple']) if eligible else None,
            resource_status='OBSERVED_NATIVE_FIXED_WINDOW' if eligible else 'UNKNOWN_NATIVE_RESOURCE_OBSERVATION',
            unknown_reasons=[] if eligible else copy.deepcopy(document['issues']),
            native_review=report_ref,raw=document['raw'],actual_native_quality=actual_quality))
        sources.extend([report_ref,document['sources']])
    present={row['attempt_id'] for row in rows}
    for item in manifest['attempts']:
        if item['attempt_id'] not in present:
            plan=checked(item['plan'])
            rows.append(dict(schema=campaign.SCHEMA+'.training_joint_tuple',split='training',
                **{k:item[k] for k in ('attempt_id','capture_id','execution_id','candidate_id','plan')},
                collection_protocol_identity=manifest['collection_protocol_identity'],
                public_case_sha256=plan['public_case_sha256'],joint_tuple=None,
                resource_status='UNKNOWN_NATIVE_RESOURCE_OBSERVATION',
                unknown_reasons=['MISSING_NATIVE_REVIEW_OR_UNOBSERVED_PLAN'],native_review=None,raw=None,
                actual_native_quality=None))
    order={a['attempt_id']:index for index,a in enumerate(manifest['attempts'])}
    rows.sort(key=lambda row:order[row['attempt_id']])
    # This is an explicit structural interface view, not relabelled native raw.
    interface=dict(training_attempts=manifest['attempts'],
        independent_validation_references=manifest['independent_validation_references'],
        collection_protocol_identity=manifest['collection_protocol_identity'])
    complete=all(row['joint_tuple'] is not None for row in rows)
    resource_rows=[{k:v for k,v in row.items() if k not in
        ('actual_native_quality','resource_status','unknown_reasons')} for row in rows]
    catalog=campaign.paired_resource_draws(resource_rows,interface) if complete else None
    sources=strict_references([sources,implementation_sources(__file__)])
    for source in sources:campaign.verify(source)
    campaign.verify(manifest_ref)
    return dict(schema=SCHEMA,status='COMPLETE_EIGHT_NATIVE_RESOURCE_TUPLES_REDERIVED_NO_FIT' if complete else
        'UNKNOWN_NATIVE_RESOURCE_CATALOG_EIGHT_PLAN_DENOMINATOR_RETAINED',
        manifest=manifest_ref,rows=rows,paired_resource_catalog=catalog,sources=sources,
        actual_native_review_rederived=True,original_observed_scopes_rederived=complete,
        observed_resource_attempts=sum(row['joint_tuple'] is not None for row in rows),
        unknown_resource_attempts=sum(row['joint_tuple'] is None for row in rows),
        unknown_attempts_removed_or_survivor_mass_renormalized=False,
        native_source_transfer_basis='EXECUTED_COMMANDS_READY_STDOUT_OWNERSHIP_PROTECTED_LEDGER_AND_EXACT_WINDOWS',
        training_attempts=8,training_blocks=2,training_offered_requests=1448,
        joint_tuple_from_same_attempt_preserved=True,startup_amortization_applied=False,
        native_output_sources_and_per_task_error_reference_totals_retained=True,quality_model_fitted=False,
        safe_factor_applied=False,queue_resource_probability_coupling_inferred=False,
        support='EXACT_FOUR_FROZEN_PUBLIC_CASES_ONLY',fit_started=False,GPU_started=False,
        validation_labels_read=False,prediction_accuracy_evaluated=False,full_U_accuracy_evaluated=False,
        final_gain_eligible=False)


def run(manifest_path,review_paths,output):
    result=load_complete(manifest_path,review_paths)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    return campaign.save(output/'report.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--review',action='append',default=[])
    parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(run(args.manifest,args.review,args.output)))
