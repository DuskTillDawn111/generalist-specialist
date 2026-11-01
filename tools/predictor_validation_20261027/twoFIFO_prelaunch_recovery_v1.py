"""CPU-predeclared one-shot recovery of a proven unlaunched tail.

This is a new policy, not a replay of the STOPPED_NO_RETRY continuation.
The old prelaunch claim remains immutable. No started capture can be relaunched.
The exact old child, supervisor, native reviewer and macros remain unchanged.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys
import tempfile
from . import twoFIFO_source_rebound_execute_v1 as old
from . import twoFIFO_source_rebound_review_v1 as native
from . import twoFIFO_source_rebound_macros_v1 as macros
from .twoFIFO_observed_resources_v2 import observed
from .stage_response_features_v1 import receipt, checked
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA = 'gsdeploy.twoFIFO_prelaunch_only_recovery.v1'
POLICY = 'NEW_ONE_SHOT_PRELAUNCH_ONLY_RECOVERY_NO_STARTED_CAPTURE_RETRY'
FINAL = 'COMPLETE_EXACT12_PREFIX4_RECOVERY_TAIL8_NO_FIT'
RECOVERY_CLAIM = 'prelaunch-recovery-launch-claim-v1.json'
RECOVERY_ENVELOPE = 'prelaunch-recovery-execution-envelope-v1.json'
require = old.require
ORIGINAL_EXECUTOR_SHA = '320b6b654a089fee905ade85560fa26fabe3ecfbe13e09f544a5c9b4ad7bda01'
ORIGINAL_SUPERVISOR_SHA = '854dfc77b43b2bf7af87935bbe145e7202b81d53359143f61834048d72f4e6a5'
ORIGINAL_MANIFEST = old.ROOT/'artifacts/predictor-twoFIFO-source-rebound-executor-v1-20261028-cpu11/manifest.json'
ORIGINAL_REPORT = ORIGINAL_MANIFEST.parent/'run01/report.json'
ERROR = 'ValueError: 18GiB preflight reserve required; never reduced'
STOPPED_SHA = '2f9ebffdf04b55e3f269fd34ae8c75f08a267babab6fc6b76ee035b1e6870cb0'
MANIFEST_SHA = 'b483b59ea1c549414a0d94830ecd11ec5f52f26d8190ce9f13fcf6629f5fb2b0'
FIFTH_CLAIM_SHA = 'eabefbb787681e8aa15b5338dcc7c06453ebd75cf06c5fc91f85335e47fe473a'


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def verify(ref):
    old._verify(ref); return checked(ref)


def expected_claim(manifest_ref, item):
    return dict(schema=old.SCHEMA+'.claim', manifest=manifest_ref, plan=item['plan'],
        execution_id=item['execution_id'], attempt_id=item['attempt_id'], no_retries=True)


def proof(manifest_path, report_path):
    """Authenticate exact synchronous preflight failure before mkdir/Popen.

    Current absence alone is not the proof: it is bound to the exact final
    source, unique exception, five execute invocations, completed prefix4 and
    immutable canonical claim. Current absence then gates eligibility.
    """
    require(Path(manifest_path).resolve() == ORIGINAL_MANIFEST and Path(report_path).resolve() == ORIGINAL_REPORT,
        'Only the explicit cpu11/run01 preflight boundary is recoverable')
    require(receipt(old.__file__)['sha256'] == ORIGINAL_EXECUTOR_SHA
        and receipt(old.supervisor.__file__)['sha256'] == ORIGINAL_SUPERVISOR_SHA,
        'Exact final executor and supervisor preflight semantics required')
    mr = receipt(manifest_path); require(mr['sha256'] == MANIFEST_SHA, 'Pinned cpu11 manifest changed')
    manifest = old.audit(manifest_path)
    rr = receipt(report_path); require(rr['sha256'] == STOPPED_SHA, 'Pinned STOPPED report changed')
    report = checked(rr)
    require(report.get('schema') == old.CONT_SCHEMA+'.run' and report.get('manifest') == mr
        and report.get('status') == 'STOPPED_NO_RETRY' and report.get('error') == ERROR
        and report.get('execute_invocations') == 5 and report.get('attempt_id') == manifest['attempts'][4]['attempt_id']
        and report.get('no_retries') is True and report.get('numerical_fit_run') is False
        and report.get('automatic_fit') is False and report.get('validation_labels_read') is False,
        'Exact disk preflight-only failure required; no arbitrary STOPPED recovery')
    prefix = report.get('completed_attempts')
    require(type(prefix) is list and len(prefix) == 4 and
        [r['attempt_id'] for r in prefix] == [i['attempt_id'] for i in manifest['attempts'][:4]],
        'Exactly the unchanged completed first block required')
    prefix_sources = []
    for row, item in zip(prefix, manifest['attempts']):
        require(row['execution']['path'] == str(Path(item['collection'])/'execution.json'), 'Prefix canonical execution required')
        old.completed_execution(verify(row['execution']), mr, item)
        review = verify(row['training_review']); doc = verify(row['macros'])
        prefix_sources.extend([review.get('sources',[]),doc.get('sources',[]),review.get('raw')])
        require(review.get('schema') == native.SCHEMA and review.get('source_rebound_manifest') == mr
            and review.get('execution') == row['execution'] and review.get('issues') == []
            and review.get('training_feature_extraction_eligible') is True
            and review.get('status') == 'PASS_COMPLETE_SOURCE_REBOUND_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION',
            'Complete prefix native review required')
        require(doc.get('source_rebound_manifest') == mr and doc.get('attempt_id') == item['attempt_id']
            and doc.get('execution_id') == item['execution_id'] and doc.get('measured_requests') ==
            sum(c['offered'] for c in checked(item['plan'])['cases']), 'Prefix macro identity required')
    first = manifest['attempts'][4]
    claim = receipt(first['claim'])
    require(claim['sha256'] == FIFTH_CLAIM_SHA and checked(claim) == expected_claim(mr, first), 'Exact immutable fifth prelaunch claim required')
    return dict(original_manifest=mr, stopped_report=rr, prefix=prefix, prelaunch_claim=claim,
        tail=manifest['attempts'][4:], original_sources=manifest['sources'],prefix_sources=strict_references(prefix_sources))


def no_started_tail(proven):
    for index, item in enumerate(proven['tail']):
        require(not (Path(item['plan']['path']).parent/RECOVERY_CLAIM).exists(), 'Canonical recovery authorization already consumed')
        require(not (Path(item['plan']['path']).parent/RECOVERY_ENVELOPE).exists(), 'Recovery envelope already exists')
        require(not Path(item['collection']).exists(), 'A started/existing tail capture cannot enter recovery')
        if index == 0:
            require(receipt(item['claim']) == proven['prelaunch_claim'], 'Original prelaunch claim changed')
        else:
            require(not Path(item['claim']).exists(), 'Later claim already exists; no recovery/retry')


def prepare(manifest_path, report_path, output):
    proven = proof(manifest_path, report_path); no_started_tail(proven)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    result = dict(schema=SCHEMA, status='CPU_PREDECLARED_EXPLICIT_EXECUTE_REQUIRED', policy=POLICY,
        proof=proven, run_output=str(ORIGINAL_MANIFEST.parent/'run02'),
        recovery_claims=[str(Path(i['plan']['path']).parent/RECOVERY_CLAIM) for i in proven['tail']],
        sources=strict_references([implementation_sources(__file__), proven]),
        old_failed_report_preserved=True, old_fifth_claim_preserved=True,
        no_started_capture_retries=True, recovery_launch_limit=8, automatic_fit=False,
        GPU_started=False, numerical_fit_run=False, validation_labels_read=False,
        supervisor_budget=old.supervisor.BUDGET, preflight_free_bytes_required=old.supervisor.PREFLIGHT)
    write(output/'manifest.json', result); audit(output/'manifest.json'); return receipt(output/'manifest.json')


def audit(path):
    result = checked(receipt(path)); proven = proof(result['proof']['original_manifest']['path'], result['proof']['stopped_report']['path'])
    require(result.get('schema') == SCHEMA and result.get('status') == 'CPU_PREDECLARED_EXPLICIT_EXECUTE_REQUIRED'
        and result.get('policy') == POLICY and result.get('proof') == proven
        and result.get('run_output') == str(ORIGINAL_MANIFEST.parent/'run02')
        and result.get('recovery_claims') == [str(Path(i['plan']['path']).parent/RECOVERY_CLAIM) for i in proven['tail']]
        and result.get('old_failed_report_preserved') is True and result.get('old_fifth_claim_preserved') is True
        and result.get('no_started_capture_retries') is True and result.get('recovery_launch_limit') == 8
        and result.get('automatic_fit') is False and result.get('GPU_started') is False
        and result.get('numerical_fit_run') is False and result.get('validation_labels_read') is False
        and result.get('supervisor_budget') == old.supervisor.BUDGET
        and result.get('preflight_free_bytes_required') == old.supervisor.PREFLIGHT,
        'Exact new prelaunch-only policy and canonical paths required')
    require(result['sources'] == strict_references([implementation_sources(__file__), proven]), 'Full recovery implementation/evidence source closure drift')
    for ref in result['sources']: old._verify(ref)
    return result


def recheck(row, manifest_path, output):
    """Rederive every numeric/native predicate, power fact and macro."""
    previous = verify(row['training_review'])
    rebuilt = native.review(previous['raw']['path'], output/'native-review', row['execution']['path'], manifest_path=manifest_path)
    require(checked(rebuilt) == previous, 'Original prefix native review must rederive exactly')
    require(macros.extract(row['training_review']['path']) == verify(row['macros']), 'Original prefix macros must rederive exactly')
    resources = observed(checked(previous['raw'])); write(output/'observed-resources.json', resources)
    require(resources['startup']['energy_joules'] is not None and all(r['measurement_energy_joules'] is not None for r in resources['cases']),
        'Original startup/measurement power-gap gate required')


def launch(manifest_path, item, index, *, guard):
    recovery_ref = receipt(manifest_path); recovery = audit(manifest_path); proven = recovery['proof']
    mr = proven['original_manifest']; original = old.audit(mr['path'])
    require(item == original['attempts'][4+index] and item == proven['tail'][index], 'Exact unlaunched fixed-order tail item required')
    collection = Path(item['collection']); require(not collection.exists(), 'Never relaunch an existing/started collection')
    # This canonical path belongs to the original attempt, so a second new
    # recovery manifest cannot bypass an earlier consumed authorization.
    authorization = Path(item['plan']['path']).parent/RECOVERY_CLAIM
    require(not authorization.exists(), 'New recovery authorization already consumed; no retry')
    require(shutil.disk_usage(collection.parent).free >= old.supervisor.PREFLIGHT, '18GiB recovery preflight insufficient; no authorization consumed')
    if index == 0:
        require(receipt(item['claim']) == proven['prelaunch_claim'], 'Preserved original fifth claim changed')
    else:
        require(not Path(item['claim']).exists(), 'Unlaunched tail claim already exists')
    # This fresh receipt declares the policy exception explicitly. It is not
    # a replacement for, or a rewrite of, the old fifth claim.
    write(authorization, dict(schema=SCHEMA+'.one_shot_authorization', recovery_manifest=recovery_ref,
        policy=POLICY, original_manifest=mr, item=item, preserved_claim=proven['prelaunch_claim'] if index == 0 else None,
        already_started_capture_retry=False))
    if index > 0: write(item['claim'], expected_claim(mr, item))
    command = [sys.executable, '-m', 'tools.predictor_validation_20261027.twoFIFO_source_rebound_execute_v1',
        '_child', '--manifest', mr['path'], '--plan', item['plan']['path'], '--output', item['collection'], '--execute']
    def gate():
        guard.check(); old._verify(recovery_ref); audit(manifest_path); old.audit(mr['path'])
        old.audit_plan(item['plan']['path']); old.supervisor.public_environment_gate(); guard.check()
    authorization_ref = receipt(authorization)
    with old.SourceChangeGuard([authorization_ref, receipt(item['claim'])]) as local_guard:
        def guarded_gate():
            local_guard.check(); old._verify(authorization_ref); gate(); local_guard.check()
        execution = old.supervisor.supervise(command, collection,
            environment={'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'},
            before_launch=guarded_gate, after_run=guarded_gate, metadata=dict(schema=old.SCHEMA+'.execution_binding',
                manifest=mr, plan=item['plan'], execution_id=item['execution_id'], attempt_id=item['attempt_id'],
                raw_schema=old.RAW_SCHEMA, downstream_adapter_required=True, automatic_fit=False))
    # The original child/execution schema still authenticates the original plan.
    # This separate envelope is mandatory for new-policy provenance/intake.
    envelope = dict(schema=SCHEMA+'.execution_envelope', policy=POLICY, recovery_manifest=recovery_ref,
        authorization=receipt(authorization), execution=receipt(collection/'execution.json'), item=item,
        original_manifest=mr, source_guard='KERNEL_INOTIFY_AND_FULL_CLOSURE_BEFORE_AFTER', no_started_capture_retry=True)
    write(Path(item['plan']['path']).parent/RECOVERY_ENVELOPE, envelope)
    return execution, receipt(Path(item['plan']['path']).parent/RECOVERY_ENVELOPE)


def run(manifest_path, *, execute=False):
    recovery_ref = receipt(manifest_path); recovery = audit(manifest_path); proven = recovery['proof']
    output = Path(recovery['run_output']); output.mkdir(parents=True, exist_ok=False)
    # Share the original continuation mutex; the supervisor alone owns the GPU
    # lease, which it holds until cleanup and all descendant reaping complete.
    lock = Path(proven['original_manifest']['path']).parent
    old_manifest = old.audit(proven['original_manifest']['path'])
    lock = Path(old_manifest['campaign']['path']).parent/'source-rebound-continuation.lock'
    result = dict(schema=SCHEMA+'.run', policy=POLICY, recovery_manifest=recovery_ref,
        original_manifest=proven['original_manifest'], stopped_report=proven['stopped_report'],
        prefix_attempts=proven['prefix'], completed_tail=[], completed_attempts=list(proven['prefix']),
        execute_invocations=0, no_started_capture_retries=True, numerical_fit_run=False,
        automatic_fit=False, validation_labels_read=False, fit_eligible=False)
    with lock.open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            no_started_tail(proven)
            if not execute:
                result['status'] = 'READY_NEW_POLICY_EXPLICIT_EXECUTE_REQUIRED'
            else:
                with old.SourceChangeGuard([recovery_ref, recovery['sources']]) as guard:
                    for row in proven['prefix']:
                        directory = output/('prefix-recheck-'+row['attempt_id']); directory.mkdir()
                        recheck(row, proven['original_manifest']['path'], directory)
                    for index, item in enumerate(proven['tail']):
                        guard.check(); result['attempt_id'] = item['attempt_id']
                        result['execute_invocations'] += 1
                        execution, envelope = launch(manifest_path, item, index, guard=guard)
                        old.completed_execution(execution, proven['original_manifest'], item)
                        directory = output/item['attempt_id']; directory.mkdir()
                        rr = native.review(Path(item['collection'])/'raw.json', directory/'native-review',
                            Path(item['collection'])/'execution.json', manifest_path=proven['original_manifest']['path'])
                        reviewed = checked(rr)
                        require(reviewed.get('issues') == [] and reviewed.get('training_feature_extraction_eligible') is True
                            and reviewed.get('status') == 'PASS_COMPLETE_SOURCE_REBOUND_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION',
                            'Native recovery attempt not eligible; stop before next attempt')
                        resources = observed(checked(reviewed['raw'])); write(directory/'observed-resources.json', resources)
                        require(resources['startup']['energy_joules'] is not None and all(
                            r['measurement_energy_joules'] is not None for r in resources['cases']), 'Original power-gap gate required')
                        doc = macros.extract(rr['path']); write(directory/'macros-v1.json', doc)
                        row = dict(attempt_id=item['attempt_id'], execution=receipt(Path(item['collection'])/'execution.json'),
                            training_review=rr, macros=receipt(directory/'macros-v1.json'), recovery_envelope=envelope)
                        result['completed_tail'].append(row); result['completed_attempts'].append(row)
                        guard.check()
                    result.update(status=FINAL, next_gate='NEW_EXPLICIT_RECOVERY_INTAKE_REQUIRED')
        except BaseException as exc:
            result.update(status='STOPPED_NEW_POLICY_NO_RETRY', error=type(exc).__name__+': '+str(exc))
        write(output/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='op', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--original-manifest', required=True); p.add_argument('--stopped-report', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('audit'); p.add_argument('--manifest', required=True)
    p = sub.add_parser('run'); p.add_argument('--manifest', required=True); p.add_argument('--execute', action='store_true')
    a = parser.parse_args()
    result = prepare(a.original_manifest,a.stopped_report,a.output) if a.op == 'prepare' else audit(a.manifest) if a.op == 'audit' else run(a.manifest,execute=a.execute)
    print(json.dumps(result))


if __name__ == '__main__': main()
