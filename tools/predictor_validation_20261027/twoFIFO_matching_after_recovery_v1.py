"""Queue the independently reviewed matching campaign after exact12 completes.

Never interleave campaigns, retry a started capture, or wait for predictor
accuracy before collecting missing training inputs. Numerical fitting remains
in the separate CPU pipeline. This entry requires explicit --execute.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from . import twoFIFO_matching_execute_v1 as matching
from . import twoFIFO_recovery_phase12_v1 as baseline
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .stage_response_features_v1 import checked, receipt
from .twoFIFO_recovery_downstream_driver_v1 import controller_identity, save, wait_for_final
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard

SCHEMA = 'gsdeploy.twoFIFO_matching_after_recovery.v1'
require = matching.require


def checked_review(path, manifest_ref):
    ref = receipt(path); review = checked(ref)
    require(review.get('schema') == 'gsdeploy.twoFIFO_matching_owned_producer_independent_review.v1'
        and review.get('status') == 'PASS_CPU_CONTRACTS_GUARD_FIX_VERIFIED_NOT_NATIVE_QUALIFICATION'
        and review.get('manifest') == manifest_ref
        and review.get('unresolved_blocking_findings') == []
        and review.get('test_summary',{}).get('status') == 'PASS',
        'Exact independent matching guard-fix review required')
    for source in strict_references([review, ref]): matching._verify(source)
    return ref, review


def run(fit_manifest, matching_manifest, independent_review, output, matching_output,
        *, controller_pid, poll_seconds=30, run_command=subprocess.run):
    require(type(poll_seconds) in (int,float) and 0 < poll_seconds <= 60,
            'Finite positive polling interval at most sixty seconds required')
    fit_ref, manifest_ref = receipt(fit_manifest), receipt(matching_manifest)
    # A queued driver has no authority to resume any existing matching attempt.
    manifest = matching.audit(matching_manifest)
    require(all(not Path(a['claim']).exists() and not Path(a['collection']).exists()
                for a in manifest['attempts']), 'Only a completely unstarted matching campaign may be queued')
    review_ref, review = checked_review(independent_review, manifest_ref)
    fit = baseline.audit(fit_manifest)
    refs = strict_references([fit_ref, manifest_ref, review_ref, review,
        manifest['sources'], [a['plan'] for a in manifest['attempts']],
        implementation_sources(__file__)])
    root = Path(output).resolve(); root.mkdir(parents=True,exist_ok=False)
    target = Path(matching_output).resolve()
    command = [sys.executable,'-m','tools.predictor_validation_20261027.twoFIFO_matching_execute_v1',
               'run','--manifest',str(Path(matching_manifest).resolve()),'--output',str(target),'--execute']
    claim = dict(schema=SCHEMA,fit_manifest=fit_ref,matching_manifest=manifest_ref,
        independent_review=review_ref,source=receipt(__file__),sources=refs,
        baseline_final_path=fit['expected_continuation_report_path'],matching_output=str(target),
        command=command,no_retries=True,GPU_started=False)
    save(root/'claim.json',claim)
    result = dict(schema=SCHEMA,matching_manifest=manifest_ref,independent_review=review_ref,
        capture_launch_invoked=False,automatic_retry=False,numerical_fit_run=False,
        independent_prediction_accuracy_validated=False,final_gain_eligible=False)
    try:
        require(not target.exists(),'Never reuse a matching continuation output')
        with SourceChangeGuard(refs) as guard:
            fit = baseline.audit(fit_manifest); matching.audit(matching_manifest)
            identity = (controller_identity(controller_pid,fit['recovery_manifest']['path'])
                        if not Path(claim['baseline_final_path']).is_file() else None)
            save(root/'controller-identity.json',identity)
            final_ref, final = wait_for_final(claim['baseline_final_path'],controller_pid=controller_pid,
                recovery_manifest=fit['recovery_manifest']['path'],poll_seconds=poll_seconds,
                expected_identity=identity,check_sources=guard.check)
            require(baseline.complete_final(fit,final),'Exact12 baseline incomplete/stopped: no new GPU campaign')
            matching._verify(final_ref); guard.check()
            matching.audit(matching_manifest); checked_review(independent_review,manifest_ref)
            require(all(not Path(a['claim']).exists() and not Path(a['collection']).exists()
                        for a in manifest['attempts']), 'Matching campaign started by another process: stop')
            require(not target.exists(),'Matching continuation output appeared while waiting')
            save(root/'matching-started.json',dict(command=command,baseline_final=final_ref))
            result.update(baseline_final=final_ref,capture_launch_invoked=True)
            with (root/'matching.log').open('x') as log:
                process = run_command(command,stdout=log,stderr=subprocess.STDOUT,check=False)
            guard.check(); matching._verify(final_ref)
            require(process.returncode==0,'Matching command failed; no automatic retry')
            report_ref = receipt(target/'report.json'); report = checked(report_ref)
            require(report.get('schema')==matching.CONT_SCHEMA+'.run'
                and report.get('manifest')==manifest_ref
                and report.get('status')=='COMPLETE_ALL_FIFTEEN_MATCHING_NATIVE_ATTEMPTS_REVIEWED_NO_FIT'
                and len(report.get('completed_attempts',[]))==15
                and report.get('numerical_fit_run') is False,
                'Matching stopped/incomplete; do not claim complete training capture')
            result.update(status='COMPLETE_MATCHING_CAPTURE_AFTER_EXACT12_NOT_PREDICTOR_QUALIFICATION',
                          matching_report=report_ref)
    except Exception as exc:
        result.update(status='STOPPED_NO_RETRY',error=type(exc).__name__+': '+str(exc))
    save(root/'report.json',result)
    return receipt(root/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fit-manifest','matching-manifest','independent-review','output','matching-output'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--controller-pid',type=int,required=True)
    parser.add_argument('--execute',required=True,action='store_true')
    args=parser.parse_args()
    print(run(args.fit_manifest,args.matching_manifest,args.independent_review,args.output,
              args.matching_output,controller_pid=args.controller_pid))
