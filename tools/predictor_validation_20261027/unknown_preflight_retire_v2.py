"""Auditable retirement of four exactly restorable UNKNOWN forecast copies.

Imports the frozen v1 archive/restore contract unchanged. No removal happens on
import or during inspect. retire requires an explicit execute switch. All four
originals and archives are completely verified before a durable intent; each
unlink repeats exact identity verification. Training/raw paths are never targets.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time

from . import unknown_preflight_archive_v1 as archive

SCHEMA='gsdeploy.exact_unknown_preflight_archive_retirement.v2'


def verify_original(record):
    p=record['original']['path'];before=archive.file_stat(p)
    if before!=record['original_stat'] or before['size']!=record['original_size']:
        raise ValueError('Original direct lstat differs from archived identity: '+p)
    actual=archive.receipt(p)
    if actual!=record['original'] or archive.file_stat(p)!=before:
        raise ValueError('Original direct SHA/lstat changed: '+p)
    return dict(original=actual,lstat=before,direct_full_SHA_and_length_verified=True)


def verify_archive(record):
    p=record['archive']['path'];before=archive.file_stat(p)
    if before!=record['archive_stat'] or before['size']!=record['archive_size']:
        raise ValueError('Archive lstat differs from manifest')
    if archive.receipt(p)!=record['archive']:raise ValueError('Archive direct SHA mismatch')
    verification=archive.verify_gzip(p,record['original']['sha256'],record['original_size'])
    if archive.file_stat(p)!=before or archive.receipt(p)!=record['archive']:
        raise ValueError('Archive changed during complete roundtrip revalidation')
    return dict(archive=record['archive'],archive_stat=before,verification=verification,
        compressed_SHA_and_complete_decompression_reverified=True)


def existing_original(record):
    """Accept restored bytes with new inode/timestamps; refuse any changed bytes."""
    p=Path(record['original']['path'])
    if not p.exists() and not p.is_symlink():return dict(original=record['original'],missing=True)
    before=archive.file_stat(p)
    if before['size']!=record['original_size'] or archive.receipt(p)!=record['original'] or archive.file_stat(p)!=before:
        raise ValueError('Existing original differs from exact archived bytes: '+str(p))
    return dict(original=record['original'],missing=False,lstat=before,direct_full_SHA_and_length_verified=True)


def verify_all(manifest_path,manifest_sha,*,recovery=False):
    archive.file_stat(manifest_path)
    reference,manifest=archive.checked_manifest(manifest_path,manifest_sha)
    if manifest.get('originals_retained') is not True:
        raise ValueError('Initial archive must declare all originals retained')
    if archive.receipt(archive.SOURCE/'report.json')!=manifest['source_report']:
        raise ValueError('Original UNKNOWN report identity changed')
    if archive.receipt(archive.__file__)!=manifest.get('tool'):
        raise ValueError('Original archive producer implementation identity changed')
    evidence=[]
    for record in manifest['originals']:
        evidence.append(dict(candidate_id=record['candidate_id'],
            archive_verification=verify_archive(record),
            original_verification=existing_original(record) if recovery else verify_original(record)))
    if archive.receipt(manifest_path)!=reference:
        raise ValueError('Manifest changed during full archive/original revalidation')
    return reference,manifest,evidence


def inspect(manifest_path,manifest_sha):
    ref,manifest,evidence=verify_all(manifest_path,manifest_sha)
    return dict(schema=SCHEMA,status='READ_ONLY_ALL_FOUR_ORIGINALS_AND_ARCHIVES_REVERIFIED',
        manifest=ref,evidence=evidence,source_deletion_executed=False,
        exact_fixed_targets=[r['original'] for r in manifest['originals']],
        original_total_bytes=sum(r['original_size'] for r in manifest['originals']),
        training_or_raw_files_touched=False,GPU_started=False,tool=archive.receipt(__file__))


def restore_command(manifest_ref,output):
    return [sys.executable,'-m','tools.predictor_validation_20261027.unknown_preflight_retire_v2',
        'recover','--manifest',manifest_ref['path'],'--manifest-sha256',manifest_ref['sha256'],
        '--output',str(output),'--execute']


def current_target_presence(records):
    # Also records unlink success followed by fsync/receipt failure: absence is
    # observed directly and is never inferred solely from progress receipts.
    result=[]
    for record in records:
        p=Path(record['original']['path'])
        try:stat=archive.file_stat(p);result.append(dict(original=record['original'],missing=False,lstat=stat))
        except FileNotFoundError:result.append(dict(original=record['original'],missing=True))
        except BaseException as exc:result.append(dict(original=record['original'],missing=False,inspection_error=str(exc)))
    return result


def checked_output(output,operation):
    out=Path(output).absolute()
    if out.parent!=archive.ROOT/'artifacts' or not re.fullmatch('predictor-v16-preflight02-lossless-'+operation+r'-[A-Za-z0-9_-]+',out.name):
        raise ValueError('Exclusive '+operation+' audit namespace required')
    if out.exists() or out.is_symlink():raise FileExistsError('Exclusive audit directory must be new')
    return out


def recover(manifest_path,manifest_sha,output):
    out=checked_output(output,'recovery')
    manifest_ref,manifest,evidence=verify_all(manifest_path,manifest_sha,recovery=True)
    with (Path(manifest_ref['path']).parent/'retirement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest_ref,manifest,evidence=verify_all(manifest_path,manifest_sha,recovery=True)
        out.mkdir(parents=True,exist_ok=False);archive.fsync_directory(out.parent)
        intent_path=out/'intent.json';tool=archive.receipt(__file__)
        archive.save_synced(intent_path,dict(schema=SCHEMA,status='RECOVERY_INTENT_FSYNC_BEFORE_PUBLISH',
            manifest=manifest_ref,tool=tool,verified=evidence,overwrite_existing=False))
        restored=[]
        try:
            for record in manifest['originals']:
                if archive.receipt(manifest_path)!=manifest_ref:raise ValueError('Manifest changed before recovery')
                verify_archive(record);state=existing_original(record)
                if state['missing']:
                    result=archive._restore_one(record,record['original']['path']);restored.append(result)
                    archive.save_synced(out/('restored-'+record['candidate_id']+'.json'),result)
            final_ref,_,verified=verify_all(manifest_path,manifest_sha,recovery=True)
            if any(e['original_verification']['missing'] for e in verified):raise ValueError('Recovery incomplete')
            if archive.receipt(__file__)!=tool:raise ValueError('Recovery implementation changed')
            result=dict(schema=SCHEMA,status='ALL_FOUR_EXACT_BYTES_PRESENT_NO_OVERWRITES',
                manifest=final_ref,intent=archive.receipt(intent_path),restored=restored,verified=verified,
                archives_retained=True,training_or_raw_files_touched=False,GPU_started=False)
            archive.save_synced(out/'recovery-receipt.json',result);return result
        except BaseException as exc:
            archive.save_synced(out/'failed-recovery.json',dict(schema=SCHEMA,status='FAILED_RECOVERY_NO_OVERWRITES',
                manifest=manifest_ref,intent=archive.receipt(intent_path),restored=restored,
                current_targets=current_target_presence(manifest['originals']),type=type(exc).__name__,reason=str(exc),
                archives_retained=True,GPU_started=False));raise


def _retire_records(records,intent_path,manifest_ref,*,before_unlink=None):
    """Narrow deletion primitive, invoked only after outer exact allowlist checks.

    The private callback is for independent tiny race/failure fixtures. It runs
    before the final complete per-file verification, never after that check.
    """
    retired=[]
    for index,record in enumerate(records):
        if before_unlink is not None:before_unlink(index,record)
        if archive.receipt(manifest_ref['path'])!=manifest_ref:
            raise ValueError('Manifest changed before original unlink')
        # Revalidate the full gzip stream again, immediately before touching
        # this original. This does not depend on an earlier sample or flag.
        verified_archive=verify_archive(record)
        verified_original=verify_original(record)
        original=Path(record['original']['path'])
        if archive.file_stat(original)!=record['original_stat']:
            raise ValueError('Original identity changed immediately before unlink')
        os.unlink(original);archive.fsync_directory(original.parent)
        evidence=dict(candidate_id=record['candidate_id'],original=record['original'],
            original_stat=record['original_stat'],archive=record['archive'],archive_stat=record['archive_stat'],
            archive_verification=verified_archive,original_verification=verified_original,
            original_unlinked=True,original_parent_directory_fsynced=True,
            audit_intent=archive.receipt(intent_path),unlinked_unix_ns=time.time_ns())
        retired.append(evidence)
        archive.save_synced(Path(intent_path).parent/('retired-'+record['candidate_id']+'.json'),evidence)
    return retired


def retire(manifest_path,manifest_sha,output):
    out=checked_output(output,'retirement')
    manifest_ref,manifest,evidence=verify_all(manifest_path,manifest_sha)
    # A dedicated archive-local lock serializes cooperating retirement attempts.
    # It does not claim to lock an arbitrary external writer to historical files.
    archive_directory=Path(manifest_ref['path']).parent
    with (archive_directory/'retirement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest_ref,manifest,evidence=verify_all(manifest_path,manifest_sha)
        out.mkdir(parents=True,exist_ok=False);archive.fsync_directory(out.parent)
        tool=archive.receipt(__file__);restore_output=archive.ROOT/'artifacts'/('predictor-v16-preflight02-lossless-recovery-'+out.name.removeprefix('predictor-v16-preflight02-lossless-retirement-'))
        command=restore_command(manifest_ref,restore_output)
        intent=dict(schema=SCHEMA,status='INTENT_FSYNC_BEFORE_ANY_ORIGINAL_UNLINK',manifest=manifest_ref,
            tool=tool,archive_tool=manifest['tool'],created_unix_ns=time.time_ns(),
            exact_fixed_targets=[dict(candidate_id=r['candidate_id'],original=r['original'],original_size=r['original_size'],original_stat=r['original_stat'],archive=r['archive']) for r in manifest['originals']],
            all_four_direct_original_SHA_lstat_and_full_gzip_roundtrip_verified=evidence,
            restore_command=command,restore_command_shell=shlex.join(command),
            partial_failure_recovery='recover verifies all four archives and existing original bytes, restores only missing originals, refuses overwrite, and is safe after partial or complete retirement.',
            source_report=manifest['source_report'],training_or_raw_files_touched=False,GPU_started=False)
        intent_path=out/'intent.json';archive.save_synced(intent_path,intent);retired=[]
        try:
            # The intent is complete, durable, and source-bound before deletion.
            retired=_retire_records(manifest['originals'],intent_path,manifest_ref)
            if any(not x['missing'] for x in current_target_presence(manifest['originals'])):
                raise ValueError('An original path is present after retirement')
            if archive.receipt(manifest_path)!=manifest_ref:raise ValueError('Manifest changed after retirement')
            if archive.receipt(__file__)!=tool:raise ValueError('Retirement implementation changed during execution')
            final=dict(schema=SCHEMA,status='ALL_FOUR_EXACT_ORIGINAL_COPIES_RETIRED_ARCHIVES_RETAINED',
                manifest=manifest_ref,intent=archive.receipt(intent_path),retired=retired,
                restore_command=command,archive_retained=True,training_or_raw_files_touched=False,GPU_started=False)
            archive.save_synced(out/'retirement-receipt.json',final);return final
        except BaseException as exc:
            # Per-file receipts are authoritative even if an error occurred
            # before the helper could return its accumulated list.
            progress=[]
            for path in sorted(out.glob('retired-*.json')):
                with path.open() as f:progress.append(json.load(f))
            archive.save_synced(out/'failed-retirement.json',dict(schema=SCHEMA,status='FAILED_PARTIAL_RETIREMENT_RECOVERABLE',
                manifest=manifest_ref,intent=archive.receipt(intent_path),retired=progress,
                current_targets=current_target_presence(manifest['originals']),restore_command=command,
                type=type(exc).__name__,reason=str(exc),archive_retained=True,
                training_or_raw_files_touched=False,GPU_started=False));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    for op in ('inspect','retire','recover'):
        parser=sub.add_parser(op);parser.add_argument('--manifest',required=True);parser.add_argument('--manifest-sha256',required=True);parser.add_argument('--output',required=True)
        if op!='inspect':parser.add_argument('--execute',action='store_true',required=True)
    args=p.parse_args()
    if args.op=='inspect':result=inspect(args.manifest,args.manifest_sha256);archive.save_synced(args.output,result)
    elif args.op=='retire':result=retire(args.manifest,args.manifest_sha256,args.output)
    else:result=recover(args.manifest,args.manifest_sha256,args.output)
    print(json.dumps(dict(status=result['status'],GPU_started=False)))


if __name__=='__main__':main()
