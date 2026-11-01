"""Two fixed archive allowlists; reuse frozen byte/retirement primitives unchanged."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import shutil
import sys

from . import unknown_preflight_archive_v1 as a
from . import unknown_preflight_retire_v2 as t

ROOT=a.ROOT
SOURCE=ROOT/'artifacts/predictor-reference-feasibility-v22-20261028-cpu01'
REPORT_SHA='229ad41bf7f6c14b82c1dee23303670a9b82d52ad3a202edb3c2ddadff70d41d'
ALLOWED={
 'shared_balanced':('92a165cd6fdd08075af0cd36a3524a1c069ed8519000cb294344a780cf5d830c',336997406),
 'shared_skewed':('ddb47b03ce8e8d11562c70e0e3f8889fe504a4530062f68c88e6f30c574755ff',327014388),
 'shared_spill_low':('4580193a2ae39ce5961119d2babec8671f4cce75bf29daa80e61191f82d315d5',350302880),
 'shared_spill_high':('61c764667738659492bc552339116ea87ffedaeb0301a719d4d6d5aa67c9849e',350505457),
}
SCHEMA='gsdeploy.exact_reference_feasibility_v22_lossless_archive.v1'
PREFIX='predictor-v22-reference-lossless-'
FLOOR=12*2**30
DATASET='v22'
IS_DERIVED=False


def select_dataset(name):
    # Two hardcoded snapshots only; no arbitrary path/report/hash CLI inputs.
    global DATASET,IS_DERIVED,SOURCE,REPORT_SHA,ALLOWED,SCHEMA,PREFIX
    if name=='v22':
        if DATASET!='v22':raise ValueError('Select dataset once per process')
        return
    if name!='light-balanced-v23' or DATASET!='v22':raise ValueError('Unknown or repeated dataset selection')
    DATASET=name;IS_DERIVED=True
    SOURCE=ROOT/'artifacts/predictor-whole57-joint-training-v23-20261028-reviewed01/whole57-joint-light-v23/balanced'
    REPORT_SHA='b16ef5367798c8406f1c9cd1b4d464f1b99a9fe965d17984adced29269611a3b'
    ALLOWED={'parsed_measured':('123857415e45f555012d6161eba63d34379ced98233654b8da60146238864df6',2059465474)}
    SCHEMA='gsdeploy.exact_light_balanced_derived_v23_lossless_archive.v1'
    PREFIX='predictor-v23-light-balanced-derived-lossless-'


def original_path(cid):
    return SOURCE/('parsed-measured.json' if IS_DERIVED else 'forecast-'+cid+'.json')


def copy_scope():
    return dict(raw_or_collection_source_files_touched=False,selected_copy_is_derived_training=IS_DERIVED)



def output_path(path,operation,*,new=True):
    path=Path(path).absolute()
    if path.parent!=ROOT/'artifacts' or not re.fullmatch(PREFIX+operation+r'-[A-Za-z0-9_-]+',path.name):
        raise ValueError('Exclusive fixed-v22 '+operation+' namespace required')
    if new and (path.exists() or path.is_symlink()):raise FileExistsError(path)
    return path


def producer_refs():return [a.receipt(p) for p in (__file__,a.__file__,t.__file__)]


def exact_sources():
    report_ref=a.receipt(SOURCE/'report.json')
    if report_ref['sha256']!=REPORT_SHA:raise ValueError('Exact fixed v22 report required')
    report=json.loads((SOURCE/'report.json').read_text())
    if IS_DERIVED:
        cid='parsed_measured';sha,size=ALLOWED[cid];path=original_path(cid);identity=a.file_stat(path)
        expected=dict(path=str(path),sha256=sha)
        if (report.get('status')!='PASS' or report.get('issues')!=[] or report.get('requests')!=57
                or report.get('training_only') is not True or report.get('validation_started') is not False
                or report.get('model_fitted') is not False or report.get('trace',{}).get('parsed')!=expected
                or identity['size']!=size):raise ValueError('Exact completed light/balanced derived review required')
        return report_ref,[dict(candidate_id=cid,original=expected,original_size=size,original_stat=identity)]
    if (report.get('status')!='RELAXED_REFERENCE_FEASIBILITY_EXECUTED' or report.get('GPU_started') is not False
            or report.get('protocol_sealed') is not False or report.get('full_prediction_preflight') is not False
            or report.get('physical_service_or_gain_validated') is not False or len(report.get('cases',[]))!=4):
        raise ValueError('Only four unsealed relaxed reference diagnostics qualify')
    sources=[]
    for case in report['cases']:
        cid=case['candidate_id'];sha,size=ALLOWED[cid];path=original_path(cid)
        expected=dict(path=str(path),sha256=sha);identity=a.file_stat(path)
        if case['result']!=expected or identity['size']!=size:raise ValueError('Fixed diagnostic identity changed')
        sources.append(dict(candidate_id=cid,original=expected,original_size=size,original_stat=identity))
    if {r['candidate_id'] for r in sources}!=set(ALLOWED):raise ValueError('Four distinct exact diagnostics required')
    return report_ref,sources


def plan(directory):
    target=output_path(directory,'archive');report,sources=exact_sources()
    total=sum(s['original_size'] for s in sources);scenario=total+total//100+4*a.BUFFER
    free=shutil.disk_usage(target.parent).free
    return dict(schema=SCHEMA,status='METADATA_ONLY_NO_ARCHIVE_OR_REMOVAL',source_report=report,sources=sources,
        archive_directory=str(target),original_total_bytes=total,current_free_bytes=free,
        incompressible_storage_scenario_bytes=scenario,free_disk_floor_bytes=FLOOR,
        floor_plus_storage_scenario_available=free>=FLOOR+scenario,producers=producer_refs(),
        original_full_SHA_checked=False,**copy_scope(),GPU_started=False)


def archive(directory):
    declaration=plan(directory)
    if not declaration['floor_plus_storage_scenario_available']:raise ValueError('Disk floor plus archive scenario unavailable')
    target=Path(declaration['archive_directory']);target.mkdir(exist_ok=False);a.fsync_directory(target.parent)
    a.save_synced(target/'archive-plan.json',declaration);records=[]
    try:
        for source in declaration['sources']:
            records.append(a._archive_one(source,target/(Path(source['original']['path']).name+'.gz')))
            a.save_synced(target/('verified-'+source['candidate_id']+'.json'),records[-1])
        for record in records:t.verify_original(record)
        if (a.receipt(SOURCE/'report.json')!=declaration['source_report']
                or producer_refs()!=declaration['producers']):raise ValueError('Archive bound sources changed')
        result=dict(schema=SCHEMA,status='ALL_SELECTED_FULL_DECOMPRESSION_VERIFIED_FSYNCED',
            source_report=declaration['source_report'],originals=records,originals_retained=True,
            producers=declaration['producers'],**copy_scope(),GPU_started=False)
        a.save_synced(target/'manifest.json',result);return result
    except BaseException as exc:
        a.save_synced(target/'failed-archive.json',dict(status='FAILED_ORIGINALS_RETAINED',reason=str(exc),verified=records));raise


def verify_all(path,sha,*,recovery=False):
    path=Path(path).absolute();output_path(path.parent,'archive',new=False);a.file_stat(path)
    reference=a.receipt(path)
    if path.name!='manifest.json' or reference['sha256']!=sha:raise ValueError('Exact v22 manifest SHA required')
    manifest=json.loads(path.read_text())
    if (manifest.get('schema')!=SCHEMA or manifest.get('status')!='ALL_SELECTED_FULL_DECOMPRESSION_VERIFIED_FSYNCED'
            or manifest.get('originals_retained') is not True or len(manifest.get('originals',[]))!=len(ALLOWED)
            or manifest.get('source_report')!=dict(path=str(SOURCE/'report.json'),sha256=REPORT_SHA)
            or a.receipt(SOURCE/'report.json')!=manifest['source_report'] or manifest.get('producers')!=producer_refs()):
        raise ValueError('Fixed diagnostic manifest/report/producer contract changed')
    seen=set();verified=[]
    for record in manifest['originals']:
        cid=record['candidate_id'];digest,size=ALLOWED[cid];seen.add(cid)
        if (record['original']!=dict(path=str(original_path(cid)),sha256=digest)
                or record['original_size']!=size
                or record['archive']['path']!=str(path.parent/(original_path(cid).name+'.gz'))):
            raise ValueError('Manifest differs from exact four-file allowlist')
        verified.append(dict(candidate_id=cid,archive=t.verify_archive(record),
            original=t.existing_original(record) if recovery else t.verify_original(record)))
    if seen!=set(ALLOWED) or a.receipt(path)!=reference:raise ValueError('Duplicate candidate or changed manifest')
    return reference,manifest,verified


def recover_command(reference,output):
    return [sys.executable,'-m','tools.predictor_validation_20261027.reference_feasibility_archive_v22','--dataset',DATASET,'recover',
        '--manifest',reference['path'],'--manifest-sha256',reference['sha256'],'--output',str(output),'--execute']


def transaction(operation,manifest_path,manifest_sha,output):
    if operation not in ('retirement','recovery'):raise ValueError('Exact retire/recover operation required')
    out=output_path(output,operation)
    manifest_path=Path(manifest_path).absolute();output_path(manifest_path.parent,'archive',new=False)
    with (manifest_path.parent/'retirement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        reference,manifest,verified=verify_all(manifest_path,manifest_sha,recovery=operation=='recovery')
        out.mkdir(exist_ok=False);a.fsync_directory(out.parent)
        recovery_out=ROOT/'artifacts'/(PREFIX+'recovery-'+out.name.removeprefix(PREFIX+operation+'-'))
        if operation=='recovery':recovery_out=recovery_out.with_name(recovery_out.name+'-retry')
        while recovery_out.exists() or recovery_out.is_symlink():recovery_out=recovery_out.with_name(recovery_out.name+'-retry')
        command=recover_command(reference,recovery_out);intent=out/'intent.json'
        a.save_synced(intent,dict(schema=SCHEMA,status='DURABLE_INTENT_BEFORE_'+operation.upper(),
            manifest=reference,verified=verified,producers=producer_refs(),restore_command=command,
            restore_command_shell=shlex.join(command),**copy_scope()))
        progress=[]
        try:
            if operation=='retirement':progress=t._retire_records(manifest['originals'],intent,reference)
            else:
                for record in manifest['originals']:
                    if a.receipt(manifest_path)!=reference:raise ValueError('Manifest changed before restore')
                    t.verify_archive(record)
                    if t.existing_original(record)['missing']:
                        restored=a._restore_one(record,record['original']['path']);progress.append(restored)
                        a.save_synced(out/('restored-'+record['candidate_id']+'.json'),restored)
            _,_,final=verify_all(manifest_path,manifest_sha,recovery=True)
            expected_missing=operation=='retirement'
            if any(v['original']['missing']!=expected_missing for v in final):raise ValueError('Unexpected final target presence')
            result=dict(schema=SCHEMA,status='ALL_SELECTED_EXACT_COPIES_RETIRED_ARCHIVES_RETAINED' if expected_missing else 'ALL_SELECTED_EXACT_BYTES_PRESENT_NO_OVERWRITES',
                manifest=reference,intent=a.receipt(intent),progress=progress,final_verification=final,
                restore_command=command,archives_retained=True,**copy_scope(),GPU_started=False)
            a.save_synced(out/(operation+'-receipt.json'),result);return result
        except BaseException as exc:
            a.save_synced(out/('failed-'+operation+'.json'),dict(schema=SCHEMA,status='FAILED_PARTIAL_OPERATION_RECOVERABLE',
                manifest=reference,intent=a.receipt(intent),reason=str(exc),current_targets=t.current_target_presence(manifest['originals']),
                restore_command=command,archives_retained=True,GPU_started=False));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',choices=('v22','light-balanced-v23'),default='v22');sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('plan');p.add_argument('--archive-directory',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('archive');p.add_argument('--archive-directory',required=True);p.add_argument('--execute',action='store_true',required=True)
    for operation in ('inspect','retire','recover'):
        p=sub.add_parser(operation);p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',required=True)
        if operation!='inspect':p.add_argument('--execute',action='store_true',required=True)
    args=parser.parse_args();select_dataset(args.dataset)
    if args.op=='plan':result=plan(args.archive_directory);a.save_synced(args.output,result)
    elif args.op=='archive':result=archive(args.archive_directory)
    elif args.op=='inspect':
        ref,_,evidence=verify_all(args.manifest,args.manifest_sha256)
        result=dict(status='READ_ONLY_ALL_SELECTED_REVERIFIED',manifest=ref,evidence=evidence);a.save_synced(args.output,result)
    else:result=transaction('retirement' if args.op=='retire' else 'recovery',args.manifest,args.manifest_sha256,args.output)
    print(json.dumps(dict(status=result['status'],GPU_started=False)))


if __name__=='__main__':main()
