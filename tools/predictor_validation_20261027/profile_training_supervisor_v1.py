"""One reviewed-budget process boundary for isolated kernel/natural/batch profiles.

prepare/audit never load models. Only execute --execute starts a collection.
The outer supervisor owns the GPU lock through actual process-group exit.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .native_collect import receipt, save
from .implementation_source_closure_v27 import implementation_sources
from .unified_corun_collect_v8_1 import cleanup_group, group_members, wait_bounded
from .batch_profile_collect_v3 import _await_group_exit
from .profile_process_ownership_v1 import OwnedDescendants, NONCE_ENV, subreaper

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'gsdeploy.profile_training_supervision.v1'
BUDGET = dict(free_disk_reserve_bytes=12*2**30, disk_abort_margin_bytes=2**30,
    incremental_storage_scenario_bytes=6*2**30, total_timeout_seconds=2400.,
    watchdog_interval_seconds=.2, cleanup_grace_seconds=8.)
PREFLIGHT = 18*2**30
LOCK = ROOT / 'artifacts/.gpu-experiment.lock'
LEASE_FD = 'GS_PROFILE_TRAINING_LEASE_FD'
LEASE_PARENT = 'GS_PROFILE_TRAINING_LEASE_PARENT'


def read(path):
    return json.loads(Path(path).read_text())


def checked(ref):
    if set(ref) != {'path', 'sha256'} or receipt(ref['path']) != ref:
        raise ValueError('Immutable profile reference changed')
    return read(ref['path'])


def validate_lease(fd):
    """Child must share its live parent's locked open-file description."""
    if (type(fd) is not int or str(fd) != os.environ.get(LEASE_FD)
            or os.environ.get(LEASE_PARENT) != str(os.getppid())):
        raise ValueError('A live outer profile supervisor lease is required')
    own = os.fstat(fd)
    expected = LOCK.stat()
    parent = os.stat(f'/proc/{os.getppid()}/fd/{fd}')
    if (own.st_dev, own.st_ino) != (expected.st_dev, expected.st_ino) or (
            parent.st_dev, parent.st_ino) != (own.st_dev, own.st_ino):
        raise ValueError('Supervisor lease descriptor identity mismatch')
    # Read existing kernel lock state before doing anything to the descriptor.
    # Acquiring flock here would let an unlocked parent masquerade as a live
    # supervisor. fdinfo also distinguishes an independently opened same-inode
    # descriptor from the parent's inherited locked open-file description.
    expected_lock = ('FLOCK','ADVISORY','WRITE',str(os.getppid()),
        f'{os.major(own.st_dev):02x}:{os.minor(own.st_dev):02x}:{own.st_ino}','0','EOF')
    def existing_locks(path):
        entries = set()
        for line in Path(path).read_text().splitlines():
            fields = line.split()
            if 'FLOCK' in fields and '->' not in fields:
                at = fields.index('FLOCK')
                entries.add(tuple(fields[at:]))
        return entries
    try:
        observations = [existing_locks('/proc/locks'),
            existing_locks(f'/proc/{os.getppid()}/fdinfo/{fd}'),
            existing_locks(f'/proc/self/fdinfo/{fd}')]
    except OSError as exc:
        raise ValueError('Cannot verify existing parent experiment lock') from exc
    if any(expected_lock not in entries for entries in observations):
        raise ValueError('Parent must already hold the inherited exclusive experiment lock')


def binding(kind, target, split=None):
    """Build only known entry points; arbitrary commands are never manifests."""
    if kind == 'paired':
        from . import paired_profile_campaign_v4 as campaign
        plan = campaign.audit(target)
        return dict(kind=kind, target=str(Path(target).resolve()), split=None,
            plan=receipt(Path(target)/'plan.json'), context=receipt(Path(target)/'context.json'),
            mode=plan['mode'], python=sys.executable,
            environment={'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'})
    if kind == 'batch':
        from . import batch_profile_collect_v3 as batch
        if split not in ('training', 'validation'):
            raise ValueError('Explicit batch split required')
        plan = batch.audit(target)
        base = batch.v2.audit(batch.checked(plan['text_owner_plan']))
        if plan['execution_blockers'] or plan['supervisor_timeout_seconds'] != 2400.:
            raise ValueError('Unblocked batch plan with exact2400s outer budget required')
        registry = read(ROOT/'artifacts/static-study-v6-20261021/inputs/worker-registry.json')
        python = registry['text']['command'][0]
        if not Path(python).is_file():
            raise ValueError('Pinned text interpreter unavailable')
        return dict(kind=kind, target=str(Path(target).resolve()), split=split,
            plan=receipt(target), context=plan['text_owner_plan'], mode='natural_batch',
            python=python, python_registry=receipt(ROOT/'artifacts/static-study-v6-20261021/inputs/worker-registry.json'),
            environment={**base['owner_environment'], 'OMP_NUM_THREADS':'1',
                         'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'})
    raise ValueError('Only paired or batch profile training is supported')


def prepare(kind, target, output, *, split=None):
    item = binding(kind, target, split)
    result = dict(schema=SCHEMA, status='CPU_PREPARED_EXPLICIT_EXECUTE_REQUIRED',
        binding=item, budget=BUDGET, preflight_free_bytes_required=PREFLIGHT,
        disk_abort_floor_bytes=13*2**30, timeout_scope='per run; startup through cleanup',
        sources=implementation_sources(__file__), GPU_started=False,
        no_retries=True, fit_eligible=False, final_gain_eligible=False)
    save(output, result)
    return result


def audit(manifest_path):
    manifest = read(manifest_path)
    if (manifest.get('schema') != SCHEMA or manifest.get('status') != 'CPU_PREPARED_EXPLICIT_EXECUTE_REQUIRED'
            or manifest.get('budget') != BUDGET or manifest.get('preflight_free_bytes_required') != PREFLIGHT
            or manifest.get('disk_abort_floor_bytes') != 13*2**30
            or manifest.get('no_retries') is not True):
        raise ValueError('Exact training supervisor schema/budget required')
    item = manifest['binding']
    if binding(item['kind'], item['target'], item['split']) != item:
        raise ValueError('Profile plan/mode/interpreter/environment drift')
    if manifest['sources'] != implementation_sources(__file__):
        raise ValueError('Complete supervisor implementation closure drift')
    return manifest


def wait_from_baseline(process,run,budget,*,initial_free,begin):
    """Reviewed wait_bounded policy with one caller-owned disk/time baseline.

    The old helper initializes its baseline at entry. Keeping the original
    prelaunch values closes both launch-growth and helper-entry race windows.
    Existing group inspection and cleanup helpers are reused unchanged.
    """
    samples=[]
    while True:
        free=shutil.disk_usage(run).free;elapsed=time.monotonic()-begin
        samples.append(dict(elapsed_seconds=elapsed,free_bytes=free))
        reason=None
        if free<budget['free_disk_reserve_bytes']+budget['disk_abort_margin_bytes']:reason='DISK_MARGIN'
        elif initial_free-free>budget['incremental_storage_scenario_bytes']:reason='STORAGE_SCENARIO_EXCEEDED'
        elif elapsed>budget['total_timeout_seconds']:reason='TIMEOUT'
        if reason is not None:
            return dict(status=reason,initial_free_bytes=initial_free,samples=samples,
                cleanup=cleanup_group(process,budget['cleanup_grace_seconds']))
        code=process.poll()
        if code is not None:
            survivors=group_members(process.pid)
            return dict(status='EXITED' if not survivors else 'LIVE_DESCENDANTS_AFTER_EXIT',
                return_code=code,initial_free_bytes=initial_free,samples=samples,
                cleanup=cleanup_group(process,budget['cleanup_grace_seconds']) if survivors else dict(events=[],live_survivors=[]))
        time.sleep(budget['watchdog_interval_seconds'])


def supervise(command, directory, *, environment, lock_path=LOCK, budget=None,
              preflight=PREFLIGHT, before_launch=None, after_run=None, metadata=None):
    """CPU-testable machinery; production execute supplies exact audited constants.

    Keep the parent lock on every exit and interrupt until all owned live group
    members disappear. Uninterruptible processes extend cleanup, never release
    the lock or turn failure into success.
    """
    budget = dict(BUDGET if budget is None else budget)
    directory = Path(directory)
    result = dict(status='FAILED_NO_RETRY', command=command, budget=budget,
                  collection_subprocess_launched=False, started_unix_ns=time.time_ns())
    if metadata is not None:
        result['profile_execution'] = metadata
    process = None
    owners = None; previous_subreaper = None
    with Path(lock_path).open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if shutil.disk_usage(directory.parent).free < preflight:
            raise ValueError('18GiB preflight reserve required; never reduced')
        directory.mkdir(exist_ok=False)
        initial_free = shutil.disk_usage(directory).free
        begin = time.monotonic()
        if before_launch is not None:
            before_launch()
        child_env = {**os.environ, **environment, LEASE_FD:str(lease.fileno()),
            LEASE_PARENT:str(os.getpid()), 'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONUNBUFFERED':'1'}
        nonce=uuid.uuid4().hex;child_env[NONCE_ENV]=nonce
        save(directory/'execution-start.json', result)
        try:
            previous_subreaper=subreaper(True)
            owners=OwnedDescendants(nonce)
            with (directory/'stdout.log').open('x') as stdout, (directory/'stderr.log').open('x') as stderr:
                process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=stdout,
                    stderr=stderr, start_new_session=True, pass_fds=(lease.fileno(),))
                result.update(collection_subprocess_launched=True, process_group=process.pid)
                owners.observe(process.pid)
                result['watchdog'] = wait_from_baseline(process,directory,budget,
                    initial_free=initial_free,begin=begin)
                if (result['watchdog']['status'] == 'EXITED'
                        and result['watchdog']['return_code'] == 0):
                    result['status'] = 'CHILD_EXITED_REQUIRES_PROFILE_REVIEW'
        except BaseException as exc:
            result['error'] = repr(exc)
        finally:
            try:
                if process is not None and group_members(process.pid):
                    result['cleanup'] = cleanup_group(process, budget['cleanup_grace_seconds'])
                    result['status'] = 'FAILED_NO_RETRY'
            except BaseException as exc:
                result.update(status='FAILED_NO_RETRY', cleanup_error=repr(exc))
            finally:
                if process is not None:
                    _await_group_exit(process.pid, directory)
                    while process.poll() is None:
                        try:
                            process.wait(timeout=.1)
                        except (KeyboardInterrupt, subprocess.TimeoutExpired):
                            continue
                    result.update(live_group_pids=group_members(process.pid),
                        experiment_lock_held_through_group_exit=True, child_reaped=True)
                    while True:
                        try:
                            owned_exit=owners.finish(process,budget['cleanup_grace_seconds'],directory)
                            result['owned_descendant_exit']=owned_exit
                            if owned_exit['initial_live_owned']:
                                result['status']='FAILED_NO_RETRY'
                            break
                        except BaseException as exc:
                            result.update(status='FAILED_NO_RETRY',owned_cleanup_error=repr(exc))
                    result['all_owned_descendants_exited_and_reaped']=True
                if previous_subreaper is not None:
                    subreaper(previous_subreaper)
            if after_run is not None:
                try:
                    after_run()
                    result['final_source_audit'] = 'PASS'
                except BaseException as exc:
                    result.update(status='FAILED_NO_RETRY', final_source_audit='FAIL', source_error=repr(exc))
            result.update(ended_unix_ns=time.time_ns(), free_bytes_at_end=shutil.disk_usage(directory).free)
            save(directory/'execution.json', result)
    return result


def execution_gate(manifest):
    item = manifest['binding']
    if item['kind'] == 'paired':
        from .paired_profile_campaign_v4 import execution_gate as paired_gate
        paired_gate(item['target'])
    else:
        from . import batch_profile_collect_v3 as batch
        plan = batch.audit(item['target'])
        if (Path(item['target']).parent/item['split']/'collection').exists():
            raise FileExistsError('Batch collection already attempted; no retry')
        # Independent batch validation requires a training-only frozen model
        # contract, which this mechanism-training supervisor does not produce.
        if item['split'] != 'training':
            raise ValueError('Batch holdout blocked until an independent frozen predictor gate exists')
    return item


def public_environment_gate():
    if any(name.startswith('CUDA_MPS_') for name in os.environ):
        raise ValueError('Inherited CUDA_MPS controls are outside default shared profile identity')
    if os.environ.get('CUDA_VISIBLE_DEVICES','0') != '0':
        raise ValueError('Original public device0 identity required')


def execute(manifest_path, output):
    public_environment_gate()
    manifest_ref = receipt(manifest_path); manifest = audit(manifest_path)
    item = execution_gate(manifest)
    command = [item['python'], '-m', 'tools.predictor_validation_20261027.profile_training_supervisor_v1',
        '_child', '--manifest', str(Path(manifest_path).resolve()), '--execute']
    result = supervise(command, output, environment=item['environment'],
        before_launch=lambda: (public_environment_gate(), checked(manifest_ref), audit(manifest_path), execution_gate(manifest)),
        after_run=lambda: (checked(manifest_ref), audit(manifest_path)),
        metadata=dict(schema=SCHEMA+'.execution_binding', manifest=manifest_ref, binding=item))
    # Child output alone never qualifies its physical profiles or predictor.
    result['profile_review_required'] = True
    return result


def completed_execution(execution_ref, target):
    execution = checked(execution_ref)
    bound = execution.get('profile_execution', {})
    if bound.get('schema') != SCHEMA+'.execution_binding':
        raise ValueError('Exact profile supervisor execution binding required')
    manifest = checked(bound['manifest']); audit(bound['manifest']['path'])
    if (bound.get('binding') != manifest['binding']
            or bound['binding']['target'] != str(Path(target).resolve())
            or execution.get('status') != 'CHILD_EXITED_REQUIRES_PROFILE_REVIEW'
            or execution.get('live_group_pids') != [] or execution.get('child_reaped') is not True
            or execution.get('experiment_lock_held_through_group_exit') is not True
            or execution.get('all_owned_descendants_exited_and_reaped') is not True
            or execution.get('final_source_audit') != 'PASS'
            or execution.get('watchdog', {}).get('status') != 'EXITED'
            or execution['watchdog'].get('return_code') != 0):
        raise ValueError('Exact completed target/source/process-boundary receipt required')
    return execution


def child(manifest_path):
    public_environment_gate()
    validate_lease(int(os.environ.get(LEASE_FD, '-1')))
    manifest = audit(manifest_path)
    item = execution_gate(manifest)
    if item['kind'] == 'paired':
        from .paired_profile_campaign_v4 import collect
        return collect(item['target'], lease_fd=int(os.environ[LEASE_FD]))
    from . import batch_profile_collect_v3 as batch
    directory = Path(item['target']).parent/item['split']/'collection'
    directory.mkdir(exist_ok=False)
    return batch._collect_attempt(item['target'], item['split'], execute=True, _reserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='op', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--kind', choices=('paired','batch'), required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--split', choices=('training','validation'))
    p.add_argument('--output', required=True)
    p = sub.add_parser('audit'); p.add_argument('--manifest', required=True)
    p = sub.add_parser('execute'); p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True); p.add_argument('--execute', action='store_true', required=True)
    p = sub.add_parser('_child'); p.add_argument('--manifest', required=True)
    p.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    if args.op == 'prepare':
        result = prepare(args.kind, args.target, args.output, split=args.split)
    elif args.op == 'audit':
        audit(args.manifest); result = dict(status='CPU_AUDIT_PASS')
    elif args.op == 'execute':
        result = execute(args.manifest, args.output)
    else:
        result = child(args.manifest)
    print(json.dumps(dict(status=result['status'])))
    if args.op == '_child':
        return 0 if result['status'] in ('completed','COLLECTED_REQUIRES_NEW_PROFILE_QUALIFICATION') else 1
    return 1 if result['status'] in ('FAILED_NO_RETRY','failed') else 0


if __name__ == '__main__':
    raise SystemExit(main())
