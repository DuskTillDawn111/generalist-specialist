"""Join unchanged passing trace checks with independently corrected host review.

Only the specific prior non-active observed-zero telemetry false failure is
eligible. Original raw, SQLite, parsed trace, failed report and evaluators stay
immutable. Nothing is re-exported or removed from the measured sample set.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .native_collect import receipt,save
from .smoke_verifier_v3 import check_receipt


def join(prior_path,host_path,output):
    prior_path,host_path=Path(prior_path).resolve(),Path(host_path).resolve();output=Path(output).resolve()
    prior=json.loads(prior_path.read_text());host=json.loads(host_path.read_text())
    if prior.get('schema')!='gsdeploy.startup_training_collection_strict_review.v4' or prior.get('issues')!=['INVALID_STARTUP_OWNED_BOARD_SAMPLE']:
        raise ValueError('Exactly the independently superseded startup sample issue is eligible')
    if host.get('schema')!='gsdeploy.host_startup_collection_strict_review.v4' or host.get('status')!='PASS':
        raise ValueError('Independent host startup strict review PASS required')
    if any(prior[k]!=host[k] for k in ('raw','plan','context')):raise ValueError('Exact same raw/context/plan required')
    for report in (prior,host):
        for name in ('raw','plan','context'):
            if not check_receipt(report[name]):raise ValueError('Changed actual evidence')
        if not all(check_receipt(s) for s in report['evaluator_sources']):raise ValueError('Changed evaluator source')
    for owner in prior['owners'].values():
        for name in ('sqlite','parsed','operator_kernel_index'):
            if name in owner and not check_receipt(owner[name]):raise ValueError('Changed canonical trace evidence')
    result=copy.deepcopy(prior)
    result.update(status='PASS',issues=[],original_strict_review=receipt(prior_path),
        independent_host_startup_review=receipt(host_path),
        review_scope='UNCHANGED_PASSING_TRACE_CHECKS_PLUS_CORRECTED_HOST_NONACTIVE_OBSERVED_ZERO_RULE',
        evaluator_sources=[receipt(__file__)],zero_samples_imputed=False,original_samples_removed=False)
    output.mkdir(parents=True,exist_ok=False);save(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prior-review',required=True);parser.add_argument('--host-review',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();r=join(args.prior_review,args.host_review,args.output)
    print(json.dumps(dict(status=r['status'],report=receipt(Path(args.output)/'report.json'))))


if __name__=='__main__':main()
