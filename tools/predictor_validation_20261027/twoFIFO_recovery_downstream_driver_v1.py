"""Wait for the existing recovery capture; run the predeclared CPU stages once.

This driver never starts or retries GPU collection, changes a prepared design,
or treats a successful process exit as predictor qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import twoFIFO_recovery_phase12_v1 as response
from . import twoFIFO_recovery_generated_v1 as generated
from . import twoFIFO_recovery_resource_v1 as resource
from .stage_response_features_v1 import checked, receipt

SCHEMA = 'gsdeploy.twoFIFO_recovery_downstream_driver.v1'
require = response.require


def controller_identity(pid, recovery_manifest):
    require(type(pid) is int and pid > 0, 'One positive controller PID required')
    proc = Path('/proc')/str(pid)
    command = (proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    expected = ['-m', 'tools.predictor_validation_20261027.twoFIFO_prelaunch_recovery_v1',
                'run', '--manifest']
    require(len(command) == 7 and command[1:5] == expected and command[-1] == '--execute',
            'Exact running recovery controller command required')
    path = Path(command[5])
    if not path.is_absolute():
        path = (proc/'cwd').resolve()/path
    require(path.resolve() == Path(recovery_manifest).resolve(), 'Controller uses a foreign recovery manifest')
    start_ticks = (proc/'stat').read_text().rsplit(')',1)[1].split()[19]
    return dict(pid=pid, command=command, start_ticks=start_ticks)


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def wait_for_final(final, *, controller_pid, recovery_manifest, poll_seconds, check_sources,
                   expected_identity=None):
    """Wait for JSON publication, without retrying any collection or fit.

    The immutable writer creates its final before json.dump finishes. A partial
    JSON file is readable only after the same writer publishes complete bytes.
    Malformed evidence with no live matching writer fails closed.
    """
    final = Path(final)
    identity = expected_identity
    while True:
        check_sources()
        if final.is_file():
            try:
                payload = final.read_bytes()
                document = json.loads(payload)
                ref = dict(path=str(final.resolve()), sha256=hashlib.sha256(payload).hexdigest())
                if receipt(final) == ref:
                    return ref, document
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Only a partial publication by the still-live original writer
                # permits waiting. Schema/identity/status errors are not caught.
                pass
        current = controller_identity(controller_pid, recovery_manifest)
        if identity is None:
            identity = current
        require(current == identity, 'Recovery controller process identity changed')
        time.sleep(poll_seconds)


def stages(fit_manifest, generated_design, resource_design, output):
    """The same four commands predeclared in the recovery execution document."""
    root = Path(output).resolve()
    fit, forecast, resources, score = [root/name for name in ('fit', 'generated', 'resources', 'rescore')]
    prefix = [sys.executable, '-m', 'tools.predictor_validation_20261027.']
    def command(module, *args):
        return [*prefix[:2], prefix[2] + module, *map(str, args)]
    return [
        dict(name='fit', command=command('twoFIFO_recovery_phase12_v1', 'run', '--manifest', fit_manifest,
            '--output', fit, '--fit'), result=str(fit/'report.json'),
            statuses=['COMPLETE_RECOVERY_SOURCE_BOUND_PHASE12_TRAINING_DIAGNOSTICS']),
        dict(name='generated', command=command('twoFIFO_recovery_generated_v1', 'run', '--design', generated_design,
            '--fit-report', fit/'report.json', '--output', forecast), result=str(forecast/'report.json'),
            statuses=['COMPLETE_REBOUND_CONDITIONAL_TRAINING_LEAVEOUT',
                      'COMPLETE_REBOUND_UNQUALIFIED_FIT_ALL_WORLDS_UNKNOWN']),
        dict(name='resources', command=command('twoFIFO_recovery_resource_v1', 'run', '--design', resource_design,
            '--output', resources, '--fit'), result=str(resources/'model.json'), statuses=[resource.COMPLETE_STATUS]),
        dict(name='rescore', command=command('twoFIFO_recovery_resource_rescore_v1',
            '--completed-report', forecast/'report.json', '--resource-model', resources/'model.json',
            '--output', score), result=str(score/'report.json'),
            statuses=['COMPLETE_REBOUND_ALLCASE_RESOURCE_TRAINING_DIAGNOSTICS'])]


def execute_stages(actions, output, *, run_command=subprocess.run, completed=None):
    completed = [] if completed is None else completed
    for action in actions:
        require(not Path(action['result']).parent.exists(), 'Never reuse a started downstream output directory')
        save(Path(output)/(action['name']+'-started.json'), action)
        with (Path(output)/(action['name']+'.log')).open('x') as stream:
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES='')
            process = run_command(action['command'], stdout=stream, stderr=subprocess.STDOUT,
                                  env=env, check=False)
        require(process.returncode == 0, 'Downstream stage failed; no retry: ' + action['name'])
        result_ref = receipt(action['result'])
        result = checked(result_ref)
        require(result.get('status') in action['statuses'], 'Downstream stage held/stopped: ' + action['name'])
        completed.append(dict(name=action['name'], result=result_ref, status=result['status']))
        save(Path(output)/(action['name']+'-complete.json'), completed[-1])
    return completed


def run(fit_manifest, generated_design, resource_design, output, *, controller_pid, poll_seconds=30):
    require(type(poll_seconds) in (int, float) and 0 < poll_seconds <= 60, 'Poll interval must be at most 60 seconds')
    require(type(controller_pid) is int and controller_pid > 0, 'One positive controller PID required')
    manifest = response.audit(fit_manifest)
    forecast = generated.audit(generated_design)
    resources = resource.audit(resource_design)
    inputs = {k:receipt(p) for k,p in dict(fit_manifest=fit_manifest,
        generated_design=generated_design, resource_design=resource_design).items()}
    require(forecast['fit_manifest'] == resources['fit_manifest'] == inputs['fit_manifest'],
            'All stages must use the same prepared fit manifest')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    actions = stages(fit_manifest, generated_design, resource_design, output)
    claim = dict(schema=SCHEMA, inputs=inputs, source=receipt(__file__), actions=actions,
        controller_pid=controller_pid, GPU_started=False, automatic_retry=False,
        final_path=manifest['expected_continuation_report_path'])
    save(output/'claim.json', claim)
    result = dict(schema=SCHEMA, inputs=inputs, completed=[], GPU_started=False,
        automatic_retry=False, independent_prediction_accuracy_validated=False, final_gain_eligible=False)
    try:
        final = Path(claim['final_path'])
        def check_sources():
            require(receipt(__file__) == claim['source'], 'Downstream driver source changed')
            for ref in inputs.values():
                require(receipt(ref['path']) == ref, 'Prepared downstream input changed while waiting')
        identity = (controller_identity(controller_pid, manifest['recovery_manifest']['path'])
                    if not final.is_file() else None)
        save(output/'controller-identity.json', identity)
        final_ref, document = wait_for_final(final, controller_pid=controller_pid,
            recovery_manifest=manifest['recovery_manifest']['path'], poll_seconds=poll_seconds,
            check_sources=check_sources, expected_identity=identity)
        require(response.complete_final(manifest, document),
                'Recovery final is incomplete/stopped; do not start any downstream fit')
        for ref in inputs.values():
            require(receipt(ref['path']) == ref, 'Prepared downstream input changed')
        require(receipt(__file__) == claim['source'], 'Downstream driver source changed')
        require(receipt(final) == final_ref, 'Recovery final changed after publication')
        result['continuation'] = final_ref
        execute_stages(actions, output, completed=result['completed'])
        result['status'] = 'COMPLETE_FOUR_CPU_TRAINING_DIAGNOSTIC_STAGES_NOT_QUALIFICATION'
    except Exception as exc:
        result.update(status='STOPPED_DOWNSTREAM_NO_RETRY', error=type(exc).__name__ + ': ' + str(exc))
    save(output/'report.json', result)
    return receipt(output/'report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-manifest', required=True)
    parser.add_argument('--generated-design', required=True)
    parser.add_argument('--resource-design', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--controller-pid', required=True, type=int)
    args = parser.parse_args()
    print(run(args.fit_manifest, args.generated_design, args.resource_design,
              args.output, controller_pid=args.controller_pid))
