"""Fixed four-route CLI for independently stored native/shared training captures."""
import argparse
import json
from pathlib import Path
import re

from . import whole57_lifecycle_training_adapter_v35 as native
from . import whole57_shared_observable_risk_v36 as shared
from .implementation_source_closure_v27 import implementation_sources
from .owner_lifecycle_mark_v21 import receipt
from .unknown_preflight_archive_v1 import ROOT,save_synced

ROUTES=('balanced','skewed','spill_low','spill_high')


def run(route,review_sha,archive_manifest_reference,output):
    if route not in ROUTES:raise ValueError('One of the four original light routes required')
    review_path=ROOT/'artifacts/predictor-whole57-joint-training-v23-20261028-reviewed01/whole57-joint-light-v23'/route/'report.json'
    review_reference=dict(path=str(review_path),sha256=review_sha)
    review=native.checked(review_reference)
    plan=native.checked(review['evidence']['plan.json'])
    if plan.get('route_id')!=route or plan.get('domain')!='whole57-joint-light-v23':
        raise ValueError('Exact original light route identity required')
    expected=str(review_path.parent/'parsed-measured.json')
    if review.get('trace',{}).get('parsed',{}).get('path')!=expected:raise ValueError('Exact derived source path required')
    out=Path(output).absolute()
    if out.parent!=ROOT/'artifacts' or not re.fullmatch('predictor-whole57-native-route-v33-'+route+r'-[A-Za-z0-9_-]+',out.name):
        raise ValueError('Exclusive fresh per-route training output required')
    out.mkdir(exist_ok=False);sources=implementation_sources(__file__)
    save_synced(out/'intent.json',dict(route=route,review=review_reference,archive_manifest=archive_manifest_reference,
        implementations=sources,training_only=True,GPU_started=False))
    try:
        native_report=native.extract(review_reference,archive_manifest_reference,out/'native')
        native_reference=receipt(out/'native/report.json')
        shared_report=shared.extract(native_reference,out/'shared')
        for ref in sources:
            if receipt(ref['path'])!=ref:raise ValueError('Route adapter implementation changed')
        result=dict(schema='gsdeploy.whole57_native_route_training_capture.v33',status='NATIVE_AND_SHARED_TRAINING_CAPTURE_EXTRACTED_NO_FIT',
            route=route,capture=native_report['capture'],review=review_reference,archive_manifest=archive_manifest_reference,
            native_report=native_reference,shared_report=receipt(out/'shared/report.json'),
            implementations=sources,physical_SAVE_events=shared_report['physical_SAVE_event_count'],
            qualified_next_mark_labels=shared_report['mark_record_count'],mark_target_counts=shared_report['mark_target_counts'],
            original_full57_retained=True,independent_capture_count=1,fit_performed=False,validation_labels_used=False,
            same_holder_guard_changed=False,physical_SM_service_identified=False,GPU_started=False)
        save_synced(out/'report.json',result);return result
    except BaseException as exc:
        save_synced(out/'failed.json',dict(status='FAILED_NO_FIT_OR_ORIGINAL_MUTATION',route=route,reason=str(exc),
            completed_native_report=receipt(out/'native/report.json') if (out/'native/report.json').is_file() else None,
            GPU_started=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route',choices=ROUTES,required=True);p.add_argument('--review-sha256',required=True)
    p.add_argument('--archive-manifest',required=True);p.add_argument('--archive-manifest-sha256',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();r=run(args.route,args.review_sha256,
        dict(path=str(Path(args.archive_manifest).absolute()),sha256=args.archive_manifest_sha256),args.output)
    print(json.dumps({k:r[k] for k in ('status','route','capture','physical_SAVE_events','mark_target_counts')}))
