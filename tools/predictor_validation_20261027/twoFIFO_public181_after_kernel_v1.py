"""Queue the reviewed eight resource startups after the pinned kernel chain.

No capture, fit or failed claim is retried. Both native collection and training
tuple intake are separate commands; successful training is not validation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import kernel_dag_response_replication_continuation_v1 as kernel
from . import twoFIFO_public181_resource_execute_v1 as executor
from . import twoFIFO_public181_resource_intake_v1 as intake
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .stage_response_features_v1 import checked, receipt
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard

SCHEMA = 'gsdeploy.twoFIFO_public181_resource_after_kernel.v1'
MODULE = 'tools.predictor_validation_20261027.kernel_dag_response_replication_continuation_v1'
REVIEW_SCHEMA = 'gsdeploy.twoFIFO_public181_resource_native_independent_review.v1'
REVIEW_STATUS = 'PASS_CPU_PREPARED_NATIVE_REAUTHENTICATION_INTERFACES_NOT_NATIVE_QUALIFICATION'
require = executor.require
save = executor.campaign.save


def verify(ref):
    require(receipt(ref['path']) == ref, 'Pinned resource continuation source changed')
    return ref


def producer_idle(manifest):
    require(all(not Path(a['claim']).exists() and not Path(a['collection']).exists()
                for a in manifest['attempts']), 'Only wholly unstarted resource training may be queued')


def reviewed(ref, manifest_ref):
    document = checked(verify(ref))
    require(document.get('schema') == REVIEW_SCHEMA and document.get('status') == REVIEW_STATUS
            and document.get('manifest') == manifest_ref
            and document.get('test_summary', {}).get('status') == 'PASS'
            and document.get('unresolved_blocking_findings') == [],
            'Exact independently reviewed resource source preparation required')
    for source in strict_references([ref, document]): verify(source)
    return document


def kernel_identity(pid, manifest_ref, claim_ref, *, proc_root=Path('/proc')):
    require(type(pid) is int and pid > 0, 'One positive kernel continuation PID required')
    proc = Path(proc_root)/str(pid)
    command = (proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    require(len(command) == 7 and command[1:5] == ['-m', MODULE, 'run', '--manifest']
            and command[-1] == '--execute', 'Exact kernel continuation command required')
    path = Path(command[5])
    path = path if path.is_absolute() else (proc/'cwd').resolve()/path
    require(path.resolve() == Path(manifest_ref['path']).resolve(), 'Foreign kernel continuation manifest')
    manifest = checked(verify(manifest_ref)); claim = checked(verify(claim_ref))
    require(claim == dict(manifest=manifest_ref, queue_identity=manifest['queue_identity'], no_retries=True)
            and Path(claim_ref['path']).resolve() == Path(manifest['output'])/'claim.json',
            'Exact exclusive kernel continuation claim required')
    return dict(pid=pid, start_ticks=(proc/'stat').read_text().rsplit(')', 1)[1].split()[19],
                command=command, manifest=manifest_ref, claim=claim_ref)


def prepare(resource_manifest, independent_review, kernel_manifest, kernel_pid, output):
    resource_ref = receipt(resource_manifest); resource = executor.audit(resource_manifest)
    producer_idle(resource)
    review_ref = receipt(independent_review); review = reviewed(review_ref, resource_ref)
    kernel_ref = receipt(kernel_manifest); parent, campaign = kernel.audit(kernel_manifest)
    claim_ref = receipt(Path(parent['output'])/'claim.json')
    identity = kernel_identity(kernel_pid, kernel_ref, claim_ref)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    sources = strict_references([resource_ref, resource, review_ref, review, kernel_ref, parent,
                                 campaign, claim_ref, implementation_sources(__file__)])
    for source in sources: verify(source)
    result = dict(schema=SCHEMA, status='CPU_PREPARED_WAITING_EXPLICIT_EXECUTE',
        resource_manifest=resource_ref, independent_review=review_ref,
        kernel_manifest=kernel_ref, kernel_claim=claim_ref, kernel_identity=identity,
        kernel_final_path=str(Path(parent['output'])/'report.json'),
        output=str(output/'run-v1'), sources=sources, no_retries=True,
        numerical_fit_success_required=False, GPU_started=False, validation_labels_read=False)
    return save(output/'manifest.json', result)


def audit(path):
    ref = receipt(path); manifest = checked(ref)
    require(manifest.get('schema') == SCHEMA and manifest.get('status') == 'CPU_PREPARED_WAITING_EXPLICIT_EXECUTE'
            and manifest.get('no_retries') is True and manifest.get('validation_labels_read') is False
            and manifest.get('numerical_fit_success_required') is False,
            'Explicit fixed resource continuation preparation required')
    resource = executor.audit(verify(manifest['resource_manifest'])['path']); producer_idle(resource)
    review = reviewed(manifest['independent_review'], manifest['resource_manifest'])
    parent, campaign = kernel.audit(verify(manifest['kernel_manifest'])['path'])
    require(manifest['kernel_final_path'] == str(Path(parent['output'])/'report.json')
            and manifest['kernel_identity']['manifest'] == manifest['kernel_manifest']
            and manifest['kernel_identity']['claim'] == manifest['kernel_claim']
            and Path(manifest['output']).resolve() == Path(path).resolve().parent/'run-v1',
            'Canonical pinned kernel writer and final required')
    claim = checked(verify(manifest['kernel_claim']))
    require(claim == dict(manifest=manifest['kernel_manifest'], queue_identity=parent['queue_identity'], no_retries=True)
            and Path(manifest['kernel_claim']['path']).resolve() == Path(parent['output'])/'claim.json',
            'Exact pinned kernel continuation claim required')
    for source in manifest['sources']: verify(source)
    require(manifest['sources'] == strict_references([manifest['resource_manifest'], resource,
        manifest['independent_review'], review, manifest['kernel_manifest'], parent,
        campaign, manifest['kernel_claim'], implementation_sources(__file__)]),
        'Full exact source cohort and continuation implementation closure required')
    return manifest, resource, parent, campaign


def completed_kernel(manifest, parent, campaign):
    published = kernel.load_publication(manifest['kernel_final_path'])
    if published is None: return None
    ref, result = published
    require(result.get('schema') == kernel.SCHEMA+'.run'
            and result.get('manifest') == manifest['kernel_manifest']
            and result.get('status') == 'COMPLETE_THREE_NATIVE_CAPTURES_AND_CPU_DIAGNOSTICS_NO_FIT'
            and result.get('fit_started') is False and result.get('automatic_retry') is False
            and result.get('validation_labels_read') is False,
            'Kernel chain stopped or incomplete; resource campaign must not launch')
    require([row['capture_id'] for row in result['completed_captures']] == [r['capture_id'] for r in campaign['runs']]
            and [row['name'] for row in result['completed_cpu_stages']] == [a['name'] for a in
                kernel.cpu_stages(parent['campaign']['path'], campaign,
                                  parent['device_attributes']['path'], parent['output'])],
            'All three pinned captures and twelve CPU stages required')
    for source in strict_references([ref, result]): verify(source)
    return ref, result


def wait_kernel(manifest, parent, campaign, check_sources, *, poll_seconds=30,
                identity_reader=kernel_identity, pause=time.sleep):
    require(type(poll_seconds) in (int, float) and 0 < poll_seconds <= 60, 'Bounded polling required')
    while True:
        check_sources()
        try:
            identity = identity_reader(manifest['kernel_identity']['pid'], manifest['kernel_manifest'], manifest['kernel_claim'])
        except (FileNotFoundError, ProcessLookupError): identity = None
        require(identity is None or identity == manifest['kernel_identity'], 'Kernel writer PID/command/start ticks changed')
        completed = completed_kernel(manifest, parent, campaign)
        if completed is not None: return completed
        require(identity is not None, 'Kernel writer exited without complete final; no resource launch')
        pause(poll_seconds)


def actions(manifest):
    root = Path(manifest['output']); collected = root/'native-resource'
    command = [sys.executable, '-m', 'tools.predictor_validation_20261027.twoFIFO_public181_resource_execute_v1',
               'run', '--manifest', manifest['resource_manifest']['path'], '--output', str(collected), '--execute']
    return dict(command=command, output=collected, report=collected/'report.json', intake_output=root/'native-intake')


def run(manifest_path, *, poll_seconds=30, run_command=subprocess.run):
    manifest, resource, parent, campaign = audit(manifest_path)
    root = Path(manifest['output']); root.mkdir(parents=True, exist_ok=False)
    manifest_ref = receipt(manifest_path)
    save(root/'claim.json', dict(schema=SCHEMA+'.claim', manifest=manifest_ref,
         kernel_identity=manifest['kernel_identity'], no_retries=True))
    result = dict(schema=SCHEMA+'.run', manifest=manifest_ref, status='WAITING_PINNED_KERNEL_FINAL',
        GPU_started=False, automatic_retry=False, fit_started=False, validation_labels_read=False,
        independent_prediction_accuracy_validated=False, final_gain_eligible=False)
    try:
        with SourceChangeGuard(strict_references([manifest_ref, manifest])) as guard:
            final_ref, final = wait_kernel(manifest, parent, campaign, guard.check, poll_seconds=poll_seconds)
            guard.check(); producer_idle(executor.audit(manifest['resource_manifest']['path']))
            with SourceChangeGuard(strict_references([final_ref, final])) as parent_guard:
                action = actions(manifest); save(root/'native-started.json', dict(command=action['command'], kernel_final=final_ref))
                result['GPU_started'] = True
                with (root/'native.log').open('x') as stream:
                    process = run_command(action['command'], stdout=stream, stderr=subprocess.STDOUT, check=False)
                guard.check(); parent_guard.check()
                require(process.returncode == 0, 'Resource collection command failed; no retry')
                collected_ref = receipt(action['report']); collected = checked(collected_ref)
                require(collected.get('schema') == executor.CONT_SCHEMA
                        and collected.get('manifest') == manifest['resource_manifest']
                        and collected.get('status') == 'COMPLETE_EIGHT_NATIVE_RESOURCE_ATTEMPTS_REVIEWED_NO_FIT'
                        and collected.get('fit_started') is False and collected.get('validation_labels_read') is False
                        and [r['attempt_id'] for r in collected['completed_attempts']] == [a['attempt_id'] for a in resource['attempts']],
                        'Complete exact eight native training attempts required')
                save(root/'native-complete.json', dict(report=collected_ref))
                result.update(kernel_final=final_ref, native_report=collected_ref)
                command = [sys.executable, '-m', 'tools.predictor_validation_20261027.twoFIFO_public181_resource_intake_v1',
                           '--manifest', manifest['resource_manifest']['path']]
                for row in collected['completed_attempts']: command += ['--review', row['training_review']['path']]
                command += ['--output', str(action['intake_output'])]
                save(root/'intake-started.json', dict(command=command))
                # Newly published native evidence joins the live watch before
                # re-review/labels. Hash restoration after a transient edit
                # cannot authenticate a mixed snapshot of those inputs.
                training_refs = strict_references([collected_ref, collected,
                    [checked(verify(row['training_review'])) for row in collected['completed_attempts']]])
                with SourceChangeGuard(training_refs) as training_guard:
                    training_guard.check()
                    with (root/'intake.log').open('x') as stream:
                        process = run_command(command, stdout=stream, stderr=subprocess.STDOUT, check=False,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                    training_guard.check(); guard.check(); parent_guard.check(); verify(collected_ref)
                    require(process.returncode == 0, 'Training tuple intake failed; no automatic retry')
                    intake_ref = receipt(action['intake_output']/'report.json'); document = checked(intake_ref)
                    require(document.get('schema') == intake.SCHEMA and document.get('manifest') == manifest['resource_manifest']
                            and document.get('status') == 'COMPLETE_EIGHT_NATIVE_RESOURCE_TUPLES_REDERIVED_NO_FIT'
                            and document.get('observed_resource_attempts') == 8 and document.get('unknown_resource_attempts') == 0,
                            'Exact eight native tuples must rederive before complete status')
                    training_guard.check()
                    result.update(status='COMPLETE_EIGHT_NATIVE_RESOURCE_TRAINING_AND_INTAKE_NOT_VALIDATION', intake=intake_ref)
    except Exception as exc:
        result.update(status='STOPPED_NO_RETRY_PREFIX_PRESERVED', error=type(exc).__name__+': '+str(exc))
    return save(root/'report.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='op', required=True)
    cmd = sub.add_parser('prepare')
    for name in ('resource-manifest', 'independent-review', 'kernel-manifest', 'output'): cmd.add_argument('--'+name, required=True)
    cmd.add_argument('--kernel-pid', required=True, type=int)
    cmd = sub.add_parser('audit'); cmd.add_argument('--manifest', required=True)
    cmd = sub.add_parser('run'); cmd.add_argument('--manifest', required=True); cmd.add_argument('--execute', required=True, action='store_true')
    args = parser.parse_args()
    if args.op == 'prepare': result = prepare(args.resource_manifest, args.independent_review, args.kernel_manifest, args.kernel_pid, args.output)
    elif args.op == 'audit': result = audit(args.manifest)[0]
    else: result = run(args.manifest)
    print(json.dumps(result))
