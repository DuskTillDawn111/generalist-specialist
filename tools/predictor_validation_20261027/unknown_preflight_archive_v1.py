"""Exact allowlist lossless archive/restore for four UNKNOWN v16 JSON files.

The plan command reads metadata only. Archive and restore require explicit CLI
execution switches. No source deletion, training/raw mutation or GPU work exists.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/predictor-learned-availability-v16-20261028-preflight02'
REPORT_SHA='058000df9eeff201e7955926c0e8179ef76d2f98b03131fdbf03b83445a683c6'
ALLOWED={
    'shared_balanced':('53e9fee8011e5e83a42f7697d3e10d1ce65d5a90d33e34b0821c84f4cce2c605',2308856507),
    'shared_skewed':('4cb5d8dc6ae3d15ea4036aa34085384adabd36ce0d90dde3d5e558f720ac3336',2288820955),
    'shared_spill_low':('0b9ad3a751f419e3ae8db9636eaf6eb4ee7a145f4f2a6629a0754a06ca91eb43',2437523661),
    'shared_spill_high':('6882132a8bdf7cd398cde53513d3693bba3ee98e7625ad6b9469df8c6917ed99',2538255709),
}
BUFFER=1024*1024
SCHEMA='gsdeploy.exact_unknown_preflight_lossless_archive.v1'


def file_stat(path):
    path=Path(path);s=path.lstat()
    if not stat.S_ISREG(s.st_mode):raise ValueError('Regular non-symlink evidence file required: '+str(path))
    return dict(path=str(path.absolute()),device=s.st_dev,inode=s.st_ino,size=s.st_size,
        mode=s.st_mode,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def receipt(path):
    p=Path(path).absolute();h=hashlib.sha256()
    with p.open('rb') as f:
        while block:=f.read(BUFFER):h.update(block)
    return dict(path=str(p),sha256=h.hexdigest())


def fsync_directory(path):
    descriptor=os.open(Path(path),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def save_synced(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    fsync_directory(Path(path).parent)


def exact_sources():
    report_ref=receipt(SOURCE/'report.json')
    if report_ref['sha256']!=REPORT_SHA:raise ValueError('Exact historical UNKNOWN report required')
    with Path(report_ref['path']).open() as f:report=json.load(f)
    if (report.get('predictions_sealed') is not False or report.get('GPU_started') is not False
            or report.get('completeness',{}).get('status')!='INCOMPLETE_PROHIBITS_SEAL_AND_COLLECTION'
            or len(report['candidates'])!=4):raise ValueError('Only these four unsealed UNKNOWN results qualify')
    sources=[]
    for candidate in report['candidates']:
        cid=candidate['candidate_id'];sha,size=ALLOWED[cid];p=SOURCE/('forecast-'+cid+'.json')
        ref=dict(path=str(p),sha256=sha)
        if candidate['status']!='UNKNOWN' or candidate['result']!=ref:
            raise ValueError('Allowlisted candidate identity changed')
        identity=file_stat(p)
        if identity['size']!=size:raise ValueError('Original evidence size changed')
        sources.append(dict(candidate_id=cid,original=ref,original_size=size,original_stat=identity))
    if {r['candidate_id'] for r in sources}!=set(ALLOWED):raise ValueError('Exact four UNKNOWN originals required')
    return report_ref,sources


def dryrun(archive_directory):
    target=Path(archive_directory).absolute()
    if target.parent!=ROOT/'artifacts' or not re.fullmatch(r'predictor-v16-preflight02-lossless-archive-[A-Za-z0-9_-]+',target.name):
        raise ValueError('Archive output must use its exclusive preflight02 archive namespace')
    if target.exists():raise FileExistsError('Archive destination must be new')
    parent=target.parent
    while not parent.exists():parent=parent.parent
    report,sources=exact_sources();total=sum(r['original_size'] for r in sources)
    # A deliberately conservative storage scenario; gzip can grow incompressible
    # input. This is not an expected compression ratio or a formal codec bound.
    scenario=sum(r['original_size']+r['original_size']//100+1024*1024 for r in sources)
    free=shutil.disk_usage(parent).free
    return dict(schema=SCHEMA,status='METADATA_ONLY_DRYRUN_NO_ARCHIVING',source_report=report,
        sources=sources,archive_directory=str(target),gzip_level=6,buffer_bytes=BUFFER,
        original_total_bytes=total,current_free_bytes=free,
        conservative_incompressible_archive_storage_scenario_bytes=scenario,
        free_after_conservative_scenario_bytes=free-scenario,
        scenario_is_formal_upper_bound=False,actual_archive_size_unknown=True,
        space_reclaimed_by_archiving_without_original_deletion_bytes=0,
        potential_reclaim_after_separately_authorized_source_retirement_bytes=total,
        original_full_SHA_rehashed_by_this_dryrun=False,
        actual_archive_requires_original_SHA_check_and_full_decompression=True,
        original_evidence_modified=False,source_deletion_supported=False,training_or_raw_files_touched=False,
        GPU_started=False,tool=receipt(__file__))


def verify_gzip(path,expected_sha,expected_size):
    digest=hashlib.sha256();size=0
    with gzip.open(path,'rb') as stream:
        while block:=stream.read(BUFFER):
            digest.update(block);size+=len(block)
            if size>expected_size:raise ValueError('Decompressed evidence exceeds exact original size')
    if digest.hexdigest()!=expected_sha or size!=expected_size:
        raise ValueError('Full gzip decompression SHA256/length mismatch')
    return dict(decompressed_sha256=digest.hexdigest(),decompressed_bytes=size,
        full_stream_decompressed=True,gzip_CRC_checked=True,exact_original_bytes_verified=True)


def _archive_one(source,target,*,level=6):
    """Generic byte-stream primitive; production caller enforces exact allowlist."""
    original=Path(source['original']['path']);target=Path(target)
    if file_stat(original)!=source['original_stat']:raise ValueError('Original source identity changed before compression')
    descriptor=os.open(original,os.O_RDONLY|os.O_NOFOLLOW)
    digest=hashlib.sha256();size=0
    with os.fdopen(descriptor,'rb') as stream,target.open('xb') as archive:
        with gzip.GzipFile(filename='',mode='wb',fileobj=archive,compresslevel=level,mtime=0) as compressed:
            while block:=stream.read(BUFFER):compressed.write(block);digest.update(block);size+=len(block)
        archive.flush();os.fsync(archive.fileno())
    fsync_directory(target.parent)
    if (digest.hexdigest()!=source['original']['sha256'] or size!=source['original_size']
            or file_stat(original)!=source['original_stat']):
        raise ValueError('Original evidence SHA/length/stat changed while compressing')
    archive_stat=file_stat(target);archive_ref=receipt(target)
    verification=verify_gzip(target,source['original']['sha256'],source['original_size'])
    if file_stat(target)!=archive_stat or receipt(target)!=archive_ref:
        raise ValueError('Archive bytes changed during full decompression verification')
    return dict(**source,archive=archive_ref,archive_size=archive_stat['size'],archive_stat=archive_stat,
        gzip_level=level,gzip_mtime=0,full_original_SHA_verified_during_compression=True,
        compressed_file_and_parent_directory_fsynced=True,verification=verification)


def archive(directory):
    plan=dryrun(directory);target=Path(plan['archive_directory']);target.mkdir(parents=True,exist_ok=False)
    fsync_directory(target.parent);save_synced(target/'archive-plan.json',plan)
    began=time.monotonic();tool=receipt(__file__);records=[]
    try:
        for source in plan['sources']:
            record=_archive_one(source,target/(Path(source['original']['path']).name+'.gz'))
            records.append(record);save_synced(target/('verified-'+source['candidate_id']+'.json'),record)
        for source in plan['sources']:
            if file_stat(source['original']['path'])!=source['original_stat']:
                raise ValueError('An original changed before archive manifest was sealed')
        if receipt(__file__)!=tool:raise ValueError('Archive implementation changed during execution')
        manifest=dict(schema=SCHEMA,status='ALL_FOUR_FULL_DECOMPRESSION_VERIFIED_FSYNCED',
            source_report=plan['source_report'],originals=records,archive_plan=receipt(target/'archive-plan.json'),
            tool=tool,buffer_bytes=BUFFER,seconds=time.monotonic()-began,originals_retained=True,
            original_evidence_modified=False,training_or_raw_files_touched=False,GPU_started=False)
        save_synced(target/'manifest.json',manifest);return manifest
    except BaseException as exc:
        save_synced(target/'failed-archive.json',dict(schema=SCHEMA,status='FAILED_ARCHIVE_ORIGINALS_RETAINED',
            type=type(exc).__name__,reason=str(exc),verified_records=records,originals_retained=True,GPU_started=False))
        raise


def checked_manifest(path,sha):
    manifest_ref=receipt(path)
    if manifest_ref['sha256']!=sha:raise ValueError('Exact archive manifest SHA required')
    with Path(path).open() as f:manifest=json.load(f)
    if (manifest.get('schema')!=SCHEMA or manifest.get('status')!='ALL_FOUR_FULL_DECOMPRESSION_VERIFIED_FSYNCED'
            or manifest.get('source_report')!=dict(path=str(SOURCE/'report.json'),sha256=REPORT_SHA)
            or len(manifest.get('originals',[]))!=4):raise ValueError('Complete exact four-file archive manifest required')
    seen=set()
    for record in manifest['originals']:
        cid=record['candidate_id'];sha,size=ALLOWED[cid];seen.add(cid)
        if (record['original']!=dict(path=str(SOURCE/('forecast-'+cid+'.json')),sha256=sha)
                or record['archive']['path']!=str(Path(path).absolute().parent/('forecast-'+cid+'.json.gz'))
                or record['original_size']!=size or record['verification']['decompressed_sha256']!=sha
                or record['verification']['decompressed_bytes']!=size
                or record['verification']['exact_original_bytes_verified'] is not True):
            raise ValueError('Manifest original identity/verification differs from exact allowlist')
    if seen!=set(ALLOWED):raise ValueError('Duplicate or missing manifest candidate')
    return manifest_ref,manifest


def _restore_one(record,target):
    """Create exact bytes at a missing path; never overwrite an existing file."""
    target=Path(target)
    if target.exists() or target.is_symlink():raise FileExistsError('Restore refuses to overwrite any existing path')
    if file_stat(record['archive']['path'])!=record['archive_stat'] or receipt(record['archive']['path'])!=record['archive']:
        raise ValueError('Archive receipt/stat changed before restoration')
    temp=target.with_name(target.name+'.restore-partial-'+str(os.getpid()))
    digest=hashlib.sha256();size=0
    try:
        with gzip.open(record['archive']['path'],'rb') as source,temp.open('xb') as output:
            while block:=source.read(BUFFER):
                digest.update(block);size+=len(block)
                if size>record['original_size']:raise ValueError('Restored stream exceeds exact original size')
                output.write(block)
            output.flush();os.fsync(output.fileno())
        if digest.hexdigest()!=record['original']['sha256'] or size!=record['original_size']:
            raise ValueError('Restored SHA256/length mismatch; target not published')
        if receipt(record['archive']['path'])!=record['archive'] or file_stat(record['archive']['path'])!=record['archive_stat']:
            raise ValueError('Archive changed during restoration; target not published')
        # link() atomically refuses an existing target, including one created
        # after the initial existence check. Only this owned temporary is removed.
        os.link(temp,target);temp.unlink();fsync_directory(target.parent)
        restored=receipt(target)
        if restored['sha256']!=record['original']['sha256']:raise ValueError('Restored target changed after publication')
        return dict(original=record['original'],restored=restored,restored_size=size,
            original_historical_stat=record['original_stat'],restored_stat=file_stat(target),
            bytes_exact=True,historical_inode_or_timestamps_recreated=False,file_and_directory_fsynced=True)
    except BaseException:
        # Keep a failed partial as explicit failed evidence; never touch original.
        raise


def restore(manifest_path,manifest_sha,output):
    manifest_ref,manifest=checked_manifest(manifest_path,manifest_sha)
    # Precheck all targets before creating any, then recheck atomically per file.
    for record in manifest['originals']:
        p=Path(record['original']['path'])
        if p.exists() or p.is_symlink():raise FileExistsError('All original paths must be absent; existing evidence is never overwritten')
    out=Path(output).absolute();out.mkdir(parents=True,exist_ok=False);records=[]
    try:
        for record in manifest['originals']:records.append(_restore_one(record,record['original']['path']))
        report=dict(schema=SCHEMA,status='EXACT_ORIGINAL_BYTES_RESTORED_NO_OVERWRITES',manifest=manifest_ref,
            originals=records,training_or_raw_files_touched=False,GPU_started=False)
        save_synced(out/'restore-receipt.json',report);return report
    except BaseException as exc:
        save_synced(out/'failed-restore.json',dict(schema=SCHEMA,status='FAILED_RESTORE_NO_OVERWRITES',
            manifest=manifest_ref,restored=records,type=type(exc).__name__,reason=str(exc),GPU_started=False));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    plan=sub.add_parser('plan');plan.add_argument('--archive-directory',required=True);plan.add_argument('--output',required=True)
    execute=sub.add_parser('archive');execute.add_argument('--archive-directory',required=True);execute.add_argument('--execute',action='store_true',required=True)
    restore_parser=sub.add_parser('restore');restore_parser.add_argument('--manifest',required=True);restore_parser.add_argument('--manifest-sha256',required=True);restore_parser.add_argument('--output',required=True);restore_parser.add_argument('--execute',action='store_true',required=True)
    args=p.parse_args()
    if args.op=='plan':
        result=dryrun(args.archive_directory);save_synced(args.output,result)
    elif args.op=='archive':result=archive(args.archive_directory)
    else:result=restore(args.manifest,args.manifest_sha256,args.output)
    print(json.dumps(dict(status=result['status'],GPU_started=False)))


if __name__=='__main__':main()
