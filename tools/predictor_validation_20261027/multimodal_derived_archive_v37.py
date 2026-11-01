"""Losslessly archive exactly two old multimodal training derived JSON copies.

Original raw/NSYS/SQLite/reports and collection sources are never removed.
Each operation binds an explicit final review SHA and a fixed route/path.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

from . import unknown_preflight_archive_v1 as a
from . import unknown_preflight_retire_v2 as t

ROOT = a.ROOT
ROUTES = ('packet01', 'packet02')
EXPECTED={
 'packet01':('68324dc9e1556978c13215c44f2556463f4d3e202a69c5b27e8d3aefb96b1bbe','d71cefddc18a9ce25d36c332f256ff4a06a9d758b3d4df5115748e9cf6cd8e7d',1174236655),
 'packet02':('cf1d9974e7e9d173e2a225e4589a1457e1bdee37835cf5148581b60db6e59e4c','e720926a3ce4d2d43a78b691672c11ad40b2376dec2cb15635a858bfed9f84d2',1174236656),
}
SCHEMA = 'gsdeploy.exact_multimodal_training_derived_lossless_archive.v37'


def locations(route):
    if route not in ROUTES:
        raise ValueError('Only two exact old multimodal training packets qualify')
    source = ROOT/'artifacts/predictor-gpu-20261027'/('multimodal-availability-training-v8_1-20261028-'+route)/'unified-light-v8_1/strict-review-v8_1'
    output = ROOT/'artifacts'/('predictor-multimodal-derived-lossless-v37-'+route+'-20261028')
    return source, output


def producers():
    return [a.receipt(path) for path in (__file__, a.__file__, t.__file__)]


def source_record(route, report_sha):
    source, _ = locations(route)
    report_ref = a.receipt(source/'report.json')
    fixed_report,fixed_parsed,fixed_size=EXPECTED[route]
    if report_sha!=fixed_report or report_ref['sha256'] != report_sha:
        raise ValueError('Exact completed review SHA required')
    report = json.loads((source/'report.json').read_text())
    parsed = source/'parsed-measured.json'
    expected = report.get('trace', {}).get('parsed')
    if (report.get('schema')!='gsdeploy.unified_corun_strict_review.v8_1' or report.get('status') != 'PASS' or report.get('issues') != []
            or report.get('training_only') is not True
            or report.get('independent_capture_count')!=1
            or report.get('validation_started') is not False or report.get('model_fitted') is not False
            or not isinstance(expected, dict) or expected.get('path') != str(parsed)
            or set(expected) != {'path', 'sha256'}):
        raise ValueError('Complete PASS training review and exact derived path required')
    identity = a.file_stat(parsed)
    if expected['sha256']!=fixed_parsed or identity['size']!=fixed_size:
        raise ValueError('Exact frozen multimodal derived SHA/size required')
    return report_ref, dict(candidate_id='parsed_measured', original=expected,
                           original_size=identity['size'], original_stat=identity)


def archive(route, report_sha):
    report_ref, record = source_record(route, report_sha)
    _, output = locations(route)
    if shutil.disk_usage(output.parent).free < 12*2**30 + record['original_size']*1.01 + 4*a.BUFFER:
        raise ValueError('Archive storage scenario plus disk reserve unavailable')
    output.mkdir(exist_ok=False); a.fsync_directory(output.parent)
    bound = producers()
    a.save_synced(output/'intent.json', dict(schema=SCHEMA, route=route,
        source_report=report_ref, original=record, producers=bound,
        operation='ARCHIVE_ONLY_ORIGINAL_RETAINED'))
    try:
        verified = a._archive_one(record, output/'parsed-measured.json.gz')
        t.verify_original(verified)
        if a.receipt(report_ref['path']) != report_ref or producers() != bound:
            raise ValueError('Review or archive implementation changed')
        result = dict(schema=SCHEMA, route=route, source_report=report_ref,
            originals=[verified], producers=bound, original_retained=True,
            status='FULL_ORIGINAL_SHA_AND_GZIP_ROUNDTRIP_VERIFIED',
            raw_or_trace_or_report_modified=False, GPU_started=False)
        a.save_synced(output/'manifest.json', result)
        return result
    except BaseException as exc:
        a.save_synced(output/'failed-archive.json', dict(reason=str(exc), original_retained=True))
        raise


def checked(route, manifest_sha, *, recovery=False):
    source, output = locations(route)
    manifest_ref = a.receipt(output/'manifest.json')
    if manifest_ref['sha256'] != manifest_sha:
        raise ValueError('Exact archive manifest SHA required')
    doc = json.loads((output/'manifest.json').read_text())
    if (doc.get('schema') != SCHEMA or doc.get('route') != route
            or doc.get('original_retained') is not True
            or doc.get('status') != 'FULL_ORIGINAL_SHA_AND_GZIP_ROUNDTRIP_VERIFIED'
            or doc.get('producers') != producers() or len(doc.get('originals', [])) != 1
            or doc.get('source_report', {}).get('path') != str(source/'report.json')
            or a.receipt(source/'report.json') != doc['source_report']):
        raise ValueError('Exact derived archive contract changed')
    record = doc['originals'][0]
    report = json.loads((source/'report.json').read_text())
    if (record.get('candidate_id') != 'parsed_measured'
            or record.get('original') != report['trace']['parsed']
            or record['original']['path'] != str(source/'parsed-measured.json')
            or record['archive']['path'] != str(output/'parsed-measured.json.gz')
            or record['original']['sha256']!=EXPECTED[route][1] or record['original_size']!=EXPECTED[route][2]
            or doc['source_report']['sha256']!=EXPECTED[route][0]):
        raise ValueError('Only the exact completed derived copy qualifies')
    evidence = dict(archive=t.verify_archive(record),
                    original=t.existing_original(record) if recovery else t.verify_original(record))
    if a.receipt(output/'manifest.json') != manifest_ref:
        raise ValueError('Manifest changed during verification')
    return manifest_ref, doc, evidence


def transaction(operation, route, manifest_sha, output):
    if operation not in ('retirement','recovery'):raise ValueError('Exact retirement/recovery operation required')
    _, archive_dir = locations(route)
    out = Path(output).absolute()
    prefix = 'predictor-multimodal-derived-'+operation+'-v37-'+route+'-'
    if out.parent != ROOT/'artifacts' or not out.name.startswith(prefix):
        raise ValueError('Exclusive derived operation audit directory required')
    with (archive_dir/'retirement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ref, doc, evidence = checked(route, manifest_sha, recovery=operation == 'recovery')
        out.mkdir(exist_ok=False); a.fsync_directory(out.parent)
        restore_out = ROOT/'artifacts'/('predictor-multimodal-derived-recovery-v37-'+route+'-'+
                                        hashlib.sha256(str(out).encode()).hexdigest()[:16])
        while restore_out.exists() or restore_out.is_symlink():
            restore_out = restore_out.with_name(restore_out.name+'-retry')
        command = [sys.executable, '-m', 'tools.predictor_validation_20261027.multimodal_derived_archive_v37',
                   'recover', '--route', route, '--manifest-sha256', manifest_sha,
                   '--output', str(restore_out), '--execute']
        a.save_synced(out/'intent.json', dict(manifest=ref, verified=evidence,
            producers=producers(), restore_command=command, restore_command_shell=shlex.join(command)))
        try:
            if operation == 'retirement':
                progress = t._retire_records(doc['originals'], out/'intent.json', ref)
            else:
                progress = []
                record = doc['originals'][0]
                if t.existing_original(record)['missing']:
                    progress.append(a._restore_one(record, record['original']['path']))
            _, _, final = checked(route, manifest_sha, recovery=True)
            if final['original']['missing'] != (operation == 'retirement'):
                raise ValueError('Final original presence disagrees with operation')
            result = dict(status='DERIVED_COPY_RETIRED_ARCHIVE_RETAINED' if operation == 'retirement'
                          else 'EXACT_DERIVED_BYTES_PRESENT', manifest=ref,
                          progress=progress, final_verification=final, restore_command=command,
                          raw_or_trace_or_report_modified=False, GPU_started=False)
            a.save_synced(out/'receipt.json', result)
            return result
        except BaseException as exc:
            a.save_synced(out/'failed.json', dict(reason=str(exc), manifest=ref,
                current_targets=t.current_target_presence(doc['originals']), restore_command=command))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('archive', 'inspect', 'retire', 'recover'))
    parser.add_argument('--route', choices=ROUTES, required=True)
    parser.add_argument('--review-sha256'); parser.add_argument('--manifest-sha256')
    parser.add_argument('--output'); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.operation == 'inspect':
        _, _, result = checked(args.route, args.manifest_sha256, recovery=True)
    else:
        if not args.execute:
            parser.error('An explicit --execute switch is required')
        result = archive(args.route, args.review_sha256) if args.operation == 'archive' else transaction(
            'retirement' if args.operation == 'retire' else 'recovery',
            args.route, args.manifest_sha256, args.output)
    print(json.dumps(result if args.operation == 'inspect' else dict(status=result['status'])))


if __name__ == '__main__': main()
