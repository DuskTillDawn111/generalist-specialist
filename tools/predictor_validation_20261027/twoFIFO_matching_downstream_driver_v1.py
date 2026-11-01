"""Wait for the reviewed matching campaign, then consume each CPU fit once.

The waiting writer is the exact queued campaign driver, not an arbitrary live
PID. A partial JSON publication permits waiting only while that writer retains
its recorded process identity. No GPU work or retries are performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from . import twoFIFO_matching_phase12_v1 as response
from . import twoFIFO_matching_generated_v1 as generated
from . import twoFIFO_matching_resource_v1 as resource
from . import twoFIFO_matching_resource_rescore_v1 as rescore
from . import twoFIFO_matching_after_recovery_v1 as queued
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .stage_response_features_v1 import checked, receipt
from .twoFIFO_recovery_downstream_driver_v1 import execute_stages, save
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard

SCHEMA = 'gsdeploy.twoFIFO_matching_downstream_driver.v1'
require = response.require


def writer_identity(pid, queue_claim, manifest, *, proc_root=Path('/proc')):
    require(type(pid) is int and pid > 0, 'Positive original queue writer PID required')
    proc = Path(proc_root)/str(pid)
    command = (proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    require(command[1:3] == ['-m', 'tools.predictor_validation_20261027.twoFIFO_matching_after_recovery_v1']
        and command[-1:] == ['--execute'] and len(command) == 16,
        'Exact queued matching writer command required')
    pairs = list(zip(command[3:-1:2], command[4:-1:2]))
    args = dict(pairs)
    require(len(args) == len(pairs) == 6 and set(args) == {'--fit-manifest', '--matching-manifest',
        '--independent-review', '--output', '--matching-output', '--controller-pid'},
        'Unique exact queued writer arguments required')
    def resolved(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else ((proc/'cwd').resolve()/path).resolve()
    claim_path = Path(queue_claim).resolve()
    claim = checked(receipt(claim_path))
    require(claim.get('schema') == queued.SCHEMA and claim.get('source') == receipt(queued.__file__)
        and claim.get('matching_manifest') == manifest['executor_manifest']
        and str(Path(claim['matching_output'])/'report.json') == manifest['expected_continuation_report_path']
        and resolved(args['--output']) == claim_path.parent
        and resolved(args['--matching-manifest']) == Path(manifest['executor_manifest']['path'])
        and resolved(args['--matching-output']) == Path(claim['matching_output'])
        and receipt(resolved(args['--fit-manifest'])) == claim['fit_manifest']
        and receipt(resolved(args['--independent-review'])) == claim['independent_review'],
        'Queued writer/claim/final/executor identity differs')
    ticks = (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]
    return dict(pid=pid, command=command, start_ticks=ticks)


def wait_for_final(path, *, writer_pid, queue_claim, manifest, check_sources,
                   expected_identity=None, poll_seconds=30):
    require(type(poll_seconds) in (int, float) and 0 < poll_seconds <= 60,
            'Finite positive polling interval at most sixty seconds required')
    final = Path(path); original = expected_identity
    while True:
        check_sources()
        if final.is_file():
            try:
                payload = final.read_bytes(); document = json.loads(payload)
                ref = dict(path=str(final.resolve()), sha256=hashlib.sha256(payload).hexdigest())
                if receipt(final) == ref:
                    return ref, document
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        current = writer_identity(writer_pid, queue_claim, manifest)
        if original is None: original = current
        require(current == original, 'Original matching queue writer identity changed')
        time.sleep(poll_seconds)


def stages(fit_manifest, generated_design, resource_design, output):
    root = Path(output).resolve()
    fit, forecast, resources, score = [root/name for name in ('fit', 'generated', 'resources', 'rescore')]
    def command(module, *args):
        return [sys.executable, '-m', 'tools.predictor_validation_20261027.'+module, *map(str,args)]
    return [
        dict(name='fit', command=command('twoFIFO_matching_phase12_v1','run','--manifest',fit_manifest,
            '--output',fit,'--fit'),result=str(fit/'report.json'),
            statuses=['COMPLETE_MATCHING_SOURCE_BOUND_PHASE12_TRAINING_DIAGNOSTICS']),
        dict(name='generated', command=command('twoFIFO_matching_generated_v1','run','--design',generated_design,
            '--fit-report',fit/'report.json','--output',forecast),result=str(forecast/'report.json'),
            statuses=['COMPLETE_MATCHING_CONDITIONAL_TRAINING_LEAVEOUT',
                      'COMPLETE_MATCHING_UNQUALIFIED_FIT_ALL_WORLDS_UNKNOWN']),
        dict(name='resources', command=command('twoFIFO_matching_resource_v1','run','--design',resource_design,
            '--output',resources,'--fit'),result=str(resources/'model.json'),
            statuses=['COMPLETE_MATCHING_RESOURCE_TRAINING_NOT_QUALIFICATION']),
        dict(name='rescore', command=command('twoFIFO_matching_resource_rescore_v1',
            '--completed-report',forecast/'report.json','--resource-model',resources/'model.json',
            '--output',score),result=str(score/'report.json'),
            statuses=['COMPLETE_MATCHING_ALLCASE_RESOURCE_TRAINING_DIAGNOSTICS'])]


def run(fit_manifest, generated_design, resource_design, queue_claim, output, *, writer_pid,
        poll_seconds=30, run_stages=execute_stages):
    manifest = response.audit(fit_manifest)
    forecast, resources = generated.audit(generated_design), resource.audit(resource_design)
    inputs = {k:receipt(p) for k,p in dict(fit_manifest=fit_manifest,
        generated_design=generated_design,resource_design=resource_design,queue_claim=queue_claim).items()}
    require(manifest.get('independent_source_review_complete') is True
        and forecast['fit_manifest'] == resources['fit_manifest'] == inputs['fit_manifest'],
        'One independently reviewed matching fit manifest required for all stages')
    claim = checked(inputs['queue_claim'])
    require(claim.get('schema') == queued.SCHEMA and claim['matching_manifest'] == manifest['executor_manifest']
        and str(Path(claim['matching_output'])/'report.json') == manifest['expected_continuation_report_path'],
        'Queue claim must bind the exact matching executor and final')
    refs = strict_references([inputs, manifest['sources'], forecast['sources'], resources['sources'],
        claim, implementation_sources(__file__)])
    for ref in refs: response.verify(ref)
    root = Path(output).resolve(); root.mkdir(parents=True,exist_ok=False)
    actions = stages(fit_manifest,generated_design,resource_design,root)
    save(root/'claim.json',dict(schema=SCHEMA,inputs=inputs,sources=refs,actions=actions,
        writer_pid=writer_pid,GPU_started=False,no_retries=True))
    result = dict(schema=SCHEMA,inputs=inputs,completed=[],GPU_started=False,automatic_retry=False,
        independent_prediction_accuracy_validated=False,final_gain_eligible=False)
    try:
        with SourceChangeGuard(refs) as guard:
            final_path = manifest['expected_continuation_report_path']
            identity = (writer_identity(writer_pid,queue_claim,manifest) if not Path(final_path).is_file() else None)
            save(root/'writer-identity.json',identity)
            final_ref, final = wait_for_final(final_path,writer_pid=writer_pid,queue_claim=queue_claim,
                manifest=manifest,check_sources=guard.check,expected_identity=identity,poll_seconds=poll_seconds)
            require(response._complete_final(manifest,final),'Incomplete or foreign matching final: no CPU fit')
            guard.check(); response.verify(final_ref)
            response.audit(fit_manifest); generated.audit(generated_design); resource.audit(resource_design)
            result['continuation'] = final_ref
            for action in actions:
                guard.check(); response.verify(final_ref)
                run_stages([action],root,completed=result['completed'])
                guard.check(); response.verify(final_ref)
            result['status'] = 'COMPLETE_MATCHING_FOUR_CPU_TRAINING_STAGES_NOT_PREDICTOR_QUALIFICATION'
    except Exception as exc:
        result.update(status='STOPPED_MATCHING_DOWNSTREAM_NO_RETRY',error=type(exc).__name__+': '+str(exc))
    save(root/'report.json',result)
    return receipt(root/'report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('fit-manifest','generated-design','resource-design','queue-claim','output'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--writer-pid',type=int,required=True)
    args = parser.parse_args()
    print(run(args.fit_manifest,args.generated_design,args.resource_design,args.queue_claim,args.output,
              writer_pid=args.writer_pid))
