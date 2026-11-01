"""Losslessly archive only completed whole57 derived parsed JSON copies.

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
ROUTES = ('skewed', 'spill_low', 'spill_high')
SCHEMA = 'gsdeploy.completed_whole57_derived_lossless_archive.v32'


def locations(route):
    if route not in ROUTES:
        raise ValueError('Only the three declared remaining light training routes qualify')
    source = ROOT/'artifacts/predictor-whole57-joint-training-v23-20261028-reviewed01/whole57-joint-light-v23'/route
    output = ROOT/'artifacts'/('predictor-whole57-derived-lossless-v32-'+route+'-20261028')
    return source, output


def producers():
    return [a.receipt(path) for path in (__file__, a.__file__, t.__file__)]


def source_record(route, report_sha):
    source, _ = locations(route)
    report_ref = a.receipt(source/'report.json')
    if report_ref['sha256'] != report_sha:
        raise ValueError('Exact completed review SHA required')
    report = json.loads((source/'report.json').read_text())
    parsed = source/'parsed-measured.json'
    expected = report.get('trace', {}).get('parsed')
    if (report.get('status') != 'PASS' or report.get('issues') != []
            or report.get('requests') != 57 or report.get('training_only') is not True
            or report.get('validation_started') is not False or report.get('model_fitted') is not False
            or not isinstance(expected, dict) or expected.get('path') != str(parsed)
            or set(expected) != {'path', 'sha256'}):
        raise ValueError('Complete PASS training review and exact derived path required')
    identity = a.file_stat(parsed)
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
            or record['archive']['path'] != str(output/'parsed-measured.json.gz')):
        raise ValueError('Only the exact completed derived copy qualifies')
    evidence = dict(archive=t.verify_archive(record),
                    original=t.existing_original(record) if recovery else t.verify_original(record))
    if a.receipt(output/'manifest.json') != manifest_ref:
        raise ValueError('Manifest changed during verification')
    return manifest_ref, doc, evidence


def transaction(operation, route, manifest_sha, output):
    _, archive_dir = locations(route)
    out = Path(output).absolute()
    prefix = 'predictor-whole57-derived-'+operation+'-v32-'+route+'-'
    if out.parent != ROOT/'artifacts' or not out.name.startswith(prefix):
        raise ValueError('Exclusive derived operation audit directory required')
    with (archive_dir/'retirement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ref, doc, evidence = checked(route, manifest_sha, recovery=operation == 'recovery')
        out.mkdir(exist_ok=False); a.fsync_directory(out.parent)
        restore_out = ROOT/'artifacts'/('predictor-whole57-derived-recovery-v32-'+route+'-'+
                                        hashlib.sha256(str(out).encode()).hexdigest()[:16])
        while restore_out.exists() or restore_out.is_symlink():
            restore_out = restore_out.with_name(restore_out.name+'-retry')
        command = [sys.executable, '-m', 'tools.predictor_validation_20261027.completed_derived_archive_v32',
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
