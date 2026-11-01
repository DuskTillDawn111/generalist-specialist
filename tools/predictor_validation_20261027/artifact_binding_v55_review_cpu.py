"""Review v48 evidence and v55 starts/producer-closure changes without sealing."""
import argparse
from collections import Counter
import gzip
import io
import json
from pathlib import Path
import resource
import unittest

from . import projected_handler_campaign_seal_v55 as v
from .learned_availability_forecast_v20 import verify_sources


def run(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).absolute();out.mkdir(parents=True,exist_ok=False)
    author=root/'artifacts/predictor-projected-handler-campaign-v48-20261028-cpu01'
    author_report=v.checked(v.receipt(author/'report.json'))
    for ref in author_report['artifacts'].values():v.checked_bytes(ref)
    author_before=v.checked(v.receipt(author/'source-before.json'))
    if author_before!=v.checked(v.receipt(author/'source-after.json')):
        raise ValueError('Author source snapshots differ')
    verify_sources(author_before)
    if v.receipt(Path(v.__file__).with_name('projected_handler_campaign_seal_v48.py'))['sha256']!='10affeb8a7b1d0eecd290f75ebd2738d0c48a3f2fededd84af0159a195694b62':
        raise ValueError('Frozen original v48 changed')
    training=v.checked(v.receipt(author/'training-check.json'))
    legacy=v.legacy._complete_index(training['legacy']['index']['path'])
    manifest=v.checked(legacy['artifacts']['source-manifest.json'])
    original_index=v.checked(legacy['index'])
    independent=[]
    # Reconstruct the exact full measured+priming identity ledger directly;
    # qualify every plan before reading its training outcome raw.
    captures=[*manifest['captures'],*training['supplemental_captures']]
    raw_refs=[]
    for capture in captures:
        raw_ref=capture['raw']
        if 'plan' in capture:plan_ref=capture['plan']
        else:plan_ref=v.checked(capture['review'])['evidence']['plan.json']
        plan=v.checked(plan_ref)
        if plan.get('split')!='training' or plan.get('training_only') is not True:
            raise ValueError('Qualify training role before raw read')
        raw=v.checked(raw_ref);raw_refs.extend([raw_ref,plan_ref])
        measured,priming=raw['request_records'],raw['priming_request_records']
        expected_measured=capture.get('measured_requests',capture.get('measured'))
        expected_priming=capture.get('priming_requests',capture.get('priming'))
        if len(measured)!=expected_measured or len(priming)!=expected_priming:
            raise ValueError('Exact measured and priming populations required')
        independent.extend(dict(run_id='raw-sha256:'+raw_ref['sha256'],**{k:r[k] for k in ('request_id','pool_id','task')})
            for r in measured+priming)
    campaign=root/'artifacts/predictor-projected-handler-v44-20261028-preflight01'
    with gzip.open(campaign/'shared_spill_high-forecast.json.gz','rt') as stream:small=json.load(stream)
    config=small['configuration'];plan_ref=small['plan'];plan=v.checked(plan_ref)
    index,extra_refs,extra=v.supplemental_training(config,legacy)
    signature=lambda rows:Counter(tuple(sorted(row.items())) for row in rows)
    if signature(index['records'])!=signature(independent) or len(independent)!=1281:
        raise ValueError('Independent complete training identity union differs')
    availability,availability_refs=v.availability_training(config,legacy)
    author_availability=training['availability']
    if availability!=author_availability:raise ValueError('Actual availability training lineage differs')
    summary_report=v.checked(v.receipt(campaign/'shared_balanced-progress-report.json'))
    summary=next(c for c in summary_report['cells'] if c['candidate_id']=='shared_balanced')
    parser,parser_refs=v.parser_runtime('/tmp/v16-preflight-readonly-ijson')
    projected=v.forecast_projection(summary,parser)
    public_ref=v.receipt(Path(plan_ref['path']).with_name('public-shared_balanced.json'))
    routes_ref=v.receipt(Path(plan_ref['path']).with_name('routes-shared_balanced.json'))
    public=v.checked(public_ref)['forecast_inputs'];routes=v.checked(routes_ref)
    bundle_ref=config['base_v15_configuration']['base_model_configuration']['component_bundle'];bundle=v.checked(bundle_ref)
    defaults=v.original_flow_defaults(plan,bundle['forecast_defaults']);quality,quality_refs=v.quality_from_bundle(bundle)
    rows=v.bind_public(plan,'shared_balanced',public,routes,defaults,plan['weights'])
    candidate=next(c for c in plan['candidates'] if c['id']=='shared_balanced')
    stored_ref=v.receipt(campaign/'shared_balanced-unsealed-cell.json');stored=v.checked(stored_ref)
    cell=v.verify_projection(projected,summary,stored,plan,plan_ref,config,rows,candidate,public['phase_start_ns'],quality)
    modules=['tests.test_projected_handler_campaign_seal_v55_independent_review','tests.test_projected_handler_campaign_seal_v48']
    refs=v.strict_references([author_before,list(author_report['artifacts'].values()),v.receipt(author/'report.json'),
        raw_refs,extra_refs,availability_refs,parser_refs,quality_refs,plan_ref,bundle_ref,public_ref,routes_ref,
        stored_ref,summary['result']['artifact'],v.receipt(campaign/'report.json'),v.implementation_sources(v.__file__),
        v.receipt(__file__),[v.receipt(root.joinpath(*name.split('.')).with_suffix('.py')) for name in modules]])
    verify_sources(refs)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestLoader().loadTestsFromNames(modules))
    (out/'tests.txt').write_text(stream.getvalue());verify_sources(refs)
    report=dict(schema='gsdeploy.artifact_binding_implementation_independent_review.v55',
        status='PASS_IMPLEMENTATION_REVIEW_NO_PRODUCTION_PREPARATION_OR_SEAL' if result.wasSuccessful() else 'FAIL',
        producer=v.receipt(v.__file__),original_v48=v.receipt(Path(v.__file__).with_name('projected_handler_campaign_seal_v48.py')),
        sources=refs,source_bytes_unchanged=True,tests_run=result.testsRun,test_log=v.receipt(out/'tests.txt'),
        independent_new_tests=6,old_tests=7,
        frozen_v48_findings=[dict(issue='START values not type/equality checked',fixed_in_v55=True,
            regression='Strings, null, bool, fractional and mismatching integer starts pass frozen v48 while unchanged metrics/gate remain equal; v55 rejects.'),
            dict(issue='Dynamic producer import closure not required in preflight source snapshot',fixed_in_v55=True,
            regression='Entrypoint-only receipt list fails v55, complete factory/preflight closure passes, before/after mismatch or byte drift fails.')],
        training_audit=dict(legacy_records=len(original_index['records']),supplemental_records=276,
            combined_records=len(independent),independent_raw_captures=len(captures),
            direct_original_raw_identity_union_equals_preparer=True,actual_availability_raw_runs=len(availability),
            actual_availability_in_complete_legacy_training_index=True,training_roles_checked_before_raw_open=True),
        actual_balanced=dict(forecast=summary['result'],full_gzip_CRC_SHA_length_verified=True,
            complete57_numeric_start_and_receipt_consistency_verified=True,
            original_six_term_metrics_recomputed_equal=True,original_gate_cell_recomputed_equal=cell==stored,
            requests=len(rows)),
        seal_contract=dict(schema=v.SCHEMA,independent_review_schema='gsdeploy.projected_handler_artifact_binding_independent_review.v55',
            model_id=v.MODEL_ID,forecast_contract_schema='gsdeploy.projected_handler_artifact_contract.v55',
            additional_execution_manifest_schema='gsdeploy.additional_execution_sources.v55',
            preflight_cells_reused=True,provider_constructed_after_freeze=False,
            source_before_after_and_original_execution_bytes_checked=True,
            old_v48_review_or_contract_not_silently_accepted=True,
            UNKNOWN_refused_before_parser_training_or_output=True,
            seal_requires_exact_independent_review_preparation_binding=True,
            current_four_candidate_v44_preflight_is_not_sealable=True),
        limitations=['This is an implementation review, not the schema required to approve an actual production preparation.',
            'Actual four-candidate COMPLETE numeric forecasts and exact independent preparation review remain necessary.',
            'Existing preflight values are reused honestly; this is not fresh provider execution after protocol freeze.',
            'Future GPU collection must start after the new seal and satisfy original gate bindings.',
            'Physical model accuracy, idle mark support and six-term improvement remain unvalidated.'],
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        production_preparation_issued=False,protocol_sealed=False,GPU_started=False,validation_outcomes_read=False)
    v.io.save_synced(out/'report.json',report)
    if not result.wasSuccessful():raise RuntimeError('Independent regression failed')
    return v.receipt(out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(run(parser.parse_args().output)))
