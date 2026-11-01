"""External one-shot SIGCONT gate for the already stopped resource waiter.

This never starts/replaces a collector or changes an existing claim. The exact
18GiB check permits the first launch only; it promises no eight-capture capacity.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
import os
from pathlib import Path
import platform
import shutil
import signal
import time

from . import twoFIFO_public181_after_kernel_v1 as queue
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .stage_response_features_v1 import checked, receipt
from .twoFIFO_owned_collect_v3 import atomic_save
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard

SCHEMA = 'gsdeploy.external_resource_capacity_gate.v1'
HOLD = Path(__file__).resolve().parents[2]/'artifacts/predictor-public181-resource-capacity-gate-v1-20261029-cpu01/hold.json'
HOLD_SHA = '5dddabd64075fff4d184dca24cba47be4c5cf0db00fe086a1255acc748c7e9c0'
PID = 691457
START_TICKS = '616998763'
PREFLIGHT = queue.executor.supervisor.PREFLIGHT
require = queue.require


def audit(hold_path):
    ref = receipt(hold_path)
    require(Path(ref['path']) == HOLD and ref['sha256'] == HOLD_SHA, 'Exact original hold required')
    hold = checked(ref)
    identity = hold['resource_identity']
    require(hold.get('schema') == SCHEMA and hold.get('status') ==
            'PINNED_UNSTARTED_RESOURCE_WAITER_HELD_FOR_EXTERNAL_CAPACITY_GATE'
            and identity['pid'] == PID and identity['start_ticks'] == START_TICKS
            and hold.get('no_native_started') is True
            and hold.get('existing_sources_or_claims_changed') is False
            and hold.get('no_capture_or_fit_retried') is True
            and hold['required_original_preflight_bytes'] == PREFLIGHT == 18*2**30,
            'Exact stopped identity and unchanged original 18GiB contract required')
    require(identity['manifest'] == hold['resource_queue_manifest'], 'Queue identity manifest differs')
    manifest, resource, parent, campaign = queue.audit(identity['manifest']['path'])
    queue.verify(identity['manifest']); queue.verify(identity['claim'])
    require(manifest['resource_manifest'] == hold['resource_manifest']
            and manifest['kernel_final_path'] == hold['kernel_final_path']
            and manifest['output'] == identity['output'] and len(resource['attempts']) == 8,
            'Exact kernel final, output and eight resource attempts required')
    claim = checked(identity['claim'])
    require(Path(identity['claim']['path']) == Path(manifest['output'])/'claim.json'
            and claim == dict(schema=queue.SCHEMA+'.claim', manifest=identity['manifest'],
                              kernel_identity=manifest['kernel_identity'], no_retries=True),
            'Original queue exclusive claim changed')
    refs = strict_references([ref, hold, manifest, resource, parent, campaign,
                              implementation_sources(__file__)])
    return hold, manifest, resource, parent, campaign, refs


def process_state(identity, *, proc_root=Path('/proc')):
    root = Path(proc_root)/str(identity['pid'])
    fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
    command = (root/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    require(fields[19] == identity['start_ticks'] and command == identity['command']
            and str((root/'cwd').resolve()) == identity['cwd'], 'Original PID identity changed or reused')
    require(fields[0] == 'T', 'Original resource waiter must remain stopped')
    require(all(not (p/'children').read_text().strip() for p in (root/'task').iterdir()),
            'Stopped waiter has children; no signal allowed')
    return fields[0]


def unstarted(manifest, resource):
    queue.producer_idle(resource)
    root = Path(manifest['output'])
    require(not any((root/name).exists() for name in
                    ('native-started.json', 'native-resource', 'native-intake', 'intake-started.json', 'report.json')),
            'Resource queue already advanced; no signal allowed')


def kernel_outcome(manifest, parent, campaign):
    publication = queue.kernel.load_publication(manifest['kernel_final_path'])
    if publication is None:
        try:
            identity = queue.kernel_identity(manifest['kernel_identity']['pid'],
                                             manifest['kernel_manifest'], manifest['kernel_claim'])
        except (FileNotFoundError, ProcessLookupError):
            raise ValueError('Kernel writer exited without canonical final; remain stopped')
        require(identity == manifest['kernel_identity'], 'Kernel writer identity changed')
        return None
    ref, result = publication
    if result.get('status') == 'STOPPED_NO_RETRY_PREFIX_PRESERVED':
        require(result.get('schema') == queue.kernel.SCHEMA+'.run'
                and result.get('manifest') == manifest['kernel_manifest']
                and result.get('fit_started') is False and result.get('automatic_retry') is False
                and result.get('validation_labels_read') is False, 'Foreign kernel failure report')
        for source in strict_references([ref, result]): queue.verify(source)
        return 'KERNEL_FAILED_RELEASE_TO_ORIGINAL_HOLD', ref, result
    ref, result = queue.completed_kernel(manifest, parent, campaign)
    return 'KERNEL_COMPLETE', ref, result


def _pidfd_syscall(number, *args):
    # This host's Python 3.10 omits the os/signal pidfd wrappers. Linux x86_64
    # and aarch64 expose the same verified syscall numbers (434 and 424).
    require(platform.system() == 'Linux' and platform.machine() in ('x86_64', 'aarch64'),
            'Reviewed Linux pidfd ABI required; no os.kill fallback')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(number, *args)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def open_pidfd(pid, flags):
    return _pidfd_syscall(434, pid, flags)


def send_continue(pid, sig, pidfd):
    # A process descriptor also closes the PID-reuse race after the last check.
    require(sig == signal.SIGCONT, 'Only SIGCONT is permitted')
    _pidfd_syscall(424, pidfd, sig, ctypes.c_void_p(), 0)


def run(hold_path=HOLD, *, poll_seconds=30, pause=time.sleep,
        disk_usage=shutil.disk_usage, identity_reader=process_state,
        outcome_reader=kernel_outcome, pidfd_open=open_pidfd,
        pidfd_close=os.close, signal_sender=send_continue):
    require(type(poll_seconds) in (int, float) and 0 < poll_seconds <= 60, 'Bounded polling required')
    hold, manifest, resource, parent, campaign, refs = audit(hold_path)
    root = Path(hold_path).resolve().parent/'run-v1'
    root.mkdir(exist_ok=False)  # A second invocation cannot resume again.
    report = dict(schema=SCHEMA+'.run', hold=receipt(hold_path), status='WAITING_KERNEL_AND_FIRST_LAUNCH_CAPACITY',
                  original_first_launch_preflight_bytes=PREFLIGHT,
                  first_launch_only_not_eight_attempt_completion_guarantee=True,
                  declared_eight_attempt_incremental_scenario_bytes=8*queue.executor.supervisor.BUDGET['incremental_storage_scenario_bytes'],
                  native181_measured_storage_estimate_bytes=None,
                  native181_storage_estimate_status='NO_COMPLETED_RESOURCE181_OR_NATIVE4_SAMPLE',
                  additional_arbitrary_reserve_bytes=0, signal_calls=0, automatic_retry=False,
                  collector_started_by_gate=False, existing_claims_modified=False)
    atomic_save(root/'status.json', report)
    fd = None
    try:
        with ExitStack() as watches:
            guard = watches.enter_context(SourceChangeGuard(refs))
            guard.check()
            require(audit(hold_path) == (hold, manifest, resource, parent, campaign, refs),
                    'Guarded hold/source snapshot changed')
            # Open first, then verify /proc identity: a reused PID cannot inherit
            # the descriptor even if it exits immediately before the signal.
            fd = pidfd_open(hold['resource_identity']['pid'], 0)
            final_guard = None; pinned_outcome = None
            while True:
                guard.check()
                identity_reader(hold['resource_identity']); unstarted(manifest, resource)
                outcome = outcome_reader(manifest, parent, campaign)
                if outcome is not None:
                    if final_guard is None:
                        final_guard = watches.enter_context(SourceChangeGuard(strict_references(outcome[1:])))
                        pinned_outcome = outcome
                    require(outcome == pinned_outcome, 'Kernel final changed during capacity wait')
                    final_guard.check()
                free = disk_usage(Path(manifest['output']).parent).free
                failed = outcome is not None and outcome[0] == 'KERNEL_FAILED_RELEASE_TO_ORIGINAL_HOLD'
                if outcome is not None and (failed or free >= PREFLIGHT):
                    # Reauthenticate immediately before the single signal. A
                    # failed kernel only releases the original queue's HOLD path.
                    require(outcome_reader(manifest, parent, campaign) == outcome, 'Kernel outcome drift')
                    guard.check(); final_guard.check()
                    identity_reader(hold['resource_identity']); unstarted(manifest, resource)
                    free = disk_usage(Path(manifest['output']).parent).free
                    require(failed or free >= PREFLIGHT, 'Capacity fell before resume; remain stopped')
                    report.update(status='RESUME_INTENT_NO_RETRY', actual_free_bytes=free,
                                  kernel_final=outcome[1], resume_reason=outcome[0])
                    atomic_save(root/'resume-intent.json', report)
                    guard.check(); final_guard.check()
                    identity_reader(hold['resource_identity']); unstarted(manifest, resource)
                    require(failed or disk_usage(Path(manifest['output']).parent).free >= PREFLIGHT,
                            'Capacity fell at signal boundary; remain stopped')
                    report['signal_calls'] = 1
                    signal_sender(hold['resource_identity']['pid'], signal.SIGCONT, fd)
                    report['status'] = ('RESUMED_ORIGINAL_WAITER_FOR_KERNEL_FAILURE_HOLD' if failed
                                        else 'RESUMED_ORIGINAL_WAITER_AT_ORIGINAL_FIRST_LAUNCH_PREFLIGHT')
                    break
                report.update(status='WAITING_KERNEL_FINAL' if outcome is None else 'WAITING_ORIGINAL_18GiB_PREFLIGHT',
                              actual_free_bytes=free)
                atomic_save(root/'status.json', report)
                pause(poll_seconds)
    except BaseException as exc:
        report.update(status='HOLD_NO_AUTOMATIC_RESUME', error=type(exc).__name__+': '+str(exc))
    finally:
        if fd is not None: pidfd_close(fd)
    atomic_save(root/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hold', default=str(HOLD))
    parser.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    report = run(args.hold)
    print(report['status'])
    return int(report['status'] == 'HOLD_NO_AUTOMATIC_RESUME')


if __name__ == '__main__': raise SystemExit(main())
