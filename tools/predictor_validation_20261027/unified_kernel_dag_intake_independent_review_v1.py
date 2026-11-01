"""Independent CPU-only artifact audit; never call the intake compiler or fit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from gsdeploy.resources import identity
from .kernel_competition_features_v2 import FEATURES, competition_features
from .kernel_sequence_audit_v2 import IDENTITY_FIELDS


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def native(kernel):
    return tuple(kernel[field] for field in IDENTITY_FIELDS)


def review(report_path, output):
    output = Path(output).resolve()
    output.mkdir(exist_ok=False, parents=True)
    report = json.loads(Path(report_path).read_text())
    refs = report['sources'] + list(report['artifacts'].values()) + [ref(report_path)]
    receipts = {(r['path'], r['sha256']) for r in refs}
    for path, sha in sorted(receipts):
        require(digest(path) == sha, 'Source/artifact changed: ' + path)
    # Read-only SQL is an independent full-session denominator and interval oracle.
    connection = sqlite3.connect('file:' + report['sqlite']['path'] + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    sql = {}
    signatures = {}
    key_to_rowid = {}
    devices = Counter()
    for row in connection.execute('select rowid,* from CUPTI_ACTIVITY_KIND_KERNEL'):
        left, right = row['start'], row['end']
        require(type(left) is int and type(right) is int and 0 <= left < right,
                'Invalid real SQLite interval')
        key = (left, right, (row['globalPid'] >> 24) & 0xffffff, row['streamId'],
               row['correlationId'], row['deviceId'], row['contextId'])
        require(all(type(value) is int for value in key), 'Noninteger real native identity')
        require(key not in key_to_rowid, 'Duplicate SQL native identity')
        key_to_rowid[key] = row['rowid']
        sql[row['rowid']] = key
        signatures[row['rowid']] = (tuple(row['block' + x] for x in 'XYZ'),
            tuple(row['grid' + x] for x in 'XYZ'), row['registersPerThread'],
            row['staticSharedMemory'], row['dynamicSharedMemory'],
            tuple(row['cluster' + x] for x in 'XYZ'))
        devices[row['deviceId']] += 1
    connection.close()
    occupancy_path = next(r['path'] for r in report['sources']
                          if 'occupancy-declaration-unified-training' in r['path'] and r['path'].endswith('/audit.json'))
    occupancy = json.loads(Path(occupancy_path).read_text())
    hardware = occupancy['hardware']
    require(identity(hardware) == report['hardware_sha256'], 'Full hardware content differs')
    member_group = {}
    for trace in occupancy['traces']:
        require(trace['source'] == report['sqlite'] and trace['unchanged'] is True,
                'Occupancy source is not this exact native SQLite')
        require(trace['device']['id'] in devices, 'Occupancy device absent')
        for group in trace['groups']:
            representative = group['representative_launch']['sqlite_rowid']
            require(group['member_count'] == len(group['member_sqlite_rowids']), 'Group denominator')
            declaration = group['resolution']['calculation']['declaration']
            require(declaration['hardware'] == hardware, 'Occupancy group hardware differs')
            patch = group['resolution']['calculation']['kernel_patch']
            require(group['resolution']['known'] is True and not declaration['assumed_fields'],
                    'Unknown/assumed occupancy group')
            for rowid in group['member_sqlite_rowids']:
                require(rowid in sql and rowid not in member_group, 'Missing/duplicate occupancy member')
                require(signatures[rowid] == signatures[representative], 'Borrowed non-equivalent occupancy')
                member_group[rowid] = (representative, patch)
    require(set(member_group) == set(sql), 'Incomplete full-session occupancy denominator')
    del occupancy
    gc.collect()

    catalog = json.loads(Path(report['artifacts']['catalog.json']['path']).read_text())
    require(catalog['training_only'] is True and catalog['validation_labels_read'] is False,
            'Catalog role')
    raw = json.loads(Path(report['raw']['path']).read_text())
    owner_pid = {alias: row['pid'] for alias, row in raw['ready'].items()}
    launches = {}
    measured_keys = set()
    per_owner = Counter()
    streams = defaultdict(set)
    for profile in catalog['profiles']:
        require(profile['split'] == 'training' and not profile['unresolved'] and profile['cyclic'] is False,
                'Incomplete/nontraining DAG')
        for node in profile['nodes']:
            if node['kind'] != 'gpu':
                continue
            kernel = node['kernel']
            key = native(kernel)
            require(key in key_to_rowid and key not in measured_keys, 'Catalog native bijection')
            require(kernel['pid'] == owner_pid[profile['alias']], 'Owner PID mismatch')
            require(kernel['source_sqlite'] == report['sqlite'], 'Foreign catalog SQLite')
            rowid = key_to_rowid[key]
            block, grid, registers, shared, dynamic, cluster = signatures[rowid]
            require((tuple(kernel['block']), tuple(kernel['grid']), kernel['registers_per_thread'],
                kernel['static_shared_bytes'], kernel['dynamic_shared_bytes'],
                tuple(kernel['cluster'] or [0, 0, 0])) == signatures[rowid], 'Static SQL launch differs')
            representative, patch = member_group[rowid]
            evidence = kernel['allocation_evidence']
            require(evidence['source'] == report['sqlite'] and evidence['signature_equivalence_checked'] is True,
                    'Occupancy attribution differs')
            member = evidence['exact_member_launch']
            require(member['sqlite_rowid'] == rowid and member['start'] == key[0] and member['end'] == key[1]
                and member['context_id'] == key[6] and member['correlation_id'] == key[4]
                and ((member['global_pid'] >> 24) & 0xffffff) == key[2], 'Member launch is not exact')
            require(evidence['representative_launch']['sqlite_rowid'] == representative, 'Wrong static group')
            for field in ('allocated_registers_per_block', 'allocated_shared_bytes', 'architecture_blocks_per_sm_ceiling'):
                require(kernel[field] == patch[field], 'Incorrect exact occupancy patch')
            require(kernel['hardware_id'] == hardware['hardware_id'], 'Kernel hardware differs')
            launch_id = profile['request_id'], node['id']
            require(launch_id not in launches, 'Repeated exact node')
            launches[launch_id] = dict(owner=profile['alias'], key=key, static=kernel)
            measured_keys.add(key)
            per_owner[profile['alias']] += 1
            streams[profile['alias']].add(key[3])
    require(len(launches) == report['offered_kernel_count'] == report['accepted_kernel_count'], 'Catalog denominator')
    del catalog, raw, signatures, member_group
    gc.collect()

    # Rebuild the measured/outside union using SQL complement, with half-open endpoints.
    endpoints = defaultdict(lambda: [0, 0])
    for key in key_to_rowid:
        role = 0 if key in measured_keys else 1
        endpoints[key[0]][role] += 1
        endpoints[key[1]][role] -= 1
    active = [0, 0]
    previous = None
    outside_overlap = 0
    for stamp, changes in sorted(endpoints.items()):
        if previous is not None and active[0] > 0 and active[1] > 0:
            outside_overlap += stamp - previous
        active = [a + b for a, b in zip(active, changes)]
        require(all(a >= 0 for a in active), 'Negative interval sweep')
        previous = stamp
    require(active == [0, 0] and outside_overlap == 0, 'Unaccounted real outside peer')
    del endpoints, sql, key_to_rowid, measured_keys
    gc.collect()

    capture = json.loads(Path(report['artifacts']['capture.json']['path']).read_text())
    require(capture['split'] == 'training' and capture['validation_labels_read'] is False, 'Capture role')
    seen = set()
    for launch in capture['kernels']:
        ident = launch['request_id'], launch['node_id']
        require(ident in launches and ident not in seen, 'Capture node bijection')
        canonical = launches[ident]
        require(launch['owner'] == canonical['owner'] and launch['kernel'] == canonical['static'], 'Capture static differs')
        require((launch['start_ns'], launch['end_ns']) == canonical['key'][:2], 'Mapped/non-native lifetime')
        require(launch['clock_domain'] == 'native-common-session-sqlite:' + report['sqlite']['sha256'], 'Clock domain')
        seen.add(ident)
    require(seen == set(launches), 'Capture full denominator')
    del capture, seen
    gc.collect()

    paths = json.loads(Path(report['artifacts']['paths.json']['path']).read_text())
    require(paths['features'] == list(FEATURES) and paths['rejected_rows'] == [], 'Path feature/reject mismatch')
    seen = set()
    path_by_node = {}
    contrasts = []
    for observation in paths['observations']:
        ident = observation['request_id'], observation['node_id']
        require(ident in launches and ident not in seen, 'Path node bijection')
        seen.add(ident)
        launch = launches[ident]
        pieces = observation['path']
        require(pieces and pieces[0]['left_ns'] == launch['key'][0]
                and pieces[-1]['right_ns'] == launch['key'][1], 'Incomplete native path')
        last = launch['key'][0]
        for piece in pieces:
            require(piece['left_ns'] == last and piece['right_ns'] > last and piece['reason'] is None
                and len(piece['vector']) == 6, 'Path gap/unsupported/column drop')
            last = piece['right_ns']
            contrasts.append(np.asarray(piece['vector']) - pieces[0]['vector'])
        path_by_node[ident] = pieces
    require(seen == set(launches), 'Full path denominator')
    path_rank = int(np.linalg.matrix_rank(np.asarray(contrasts)))
    # Independent active-set sweep, comparing every emitted interval against direct static covariates.
    starts, ends = defaultdict(list), defaultdict(list)
    for ident, launch in launches.items():
        starts[launch['key'][0]].append(ident)
        ends[launch['key'][1]].append(ident)
    active = set()
    previous = None
    positions = Counter()
    checked_segments = 0
    same_owner_peer_segments = 0
    for stamp in sorted(set(starts) | set(ends)):
        if previous is not None and stamp > previous:
            for ident in active:
                pieces = path_by_node[ident]
                position = positions[ident]
                while pieces[position]['right_ns'] <= previous:
                    position += 1
                positions[ident] = position
                piece = pieces[position]
                require(piece['left_ns'] <= previous < stamp <= piece['right_ns'], 'Active path boundary mismatch')
                peers = [launches[other]['static'] for other in active if other != ident]
                expected = competition_features(launches[ident]['static'], peers, hardware)
                require(expected['known'] and np.allclose(piece['vector'],
                    [expected['vector'][field] for field in FEATURES], rtol=0, atol=1e-14), 'Active peer vector mismatch')
                checked_segments += 1
                same_owner_peer_segments += any(launches[other]['owner'] == launches[ident]['owner']
                                               for other in active if other != ident)
        active.difference_update(ends[stamp])
        active.update(starts[stamp])
        previous = stamp
    require(not active and path_rank == 6 and same_owner_peer_segments > 0, 'Incomplete OCR peer/rank evidence')
    diagnostics = report['diagnostics']
    require(diagnostics['baseline_groups'] == diagnostics['singleton_baseline_groups'] == len(launches)
        and diagnostics['nuisance_adjusted_derivative_rank'] == 0 and diagnostics['fit_started'] is False,
        'Singleton nuisance derivative barrier')
    require(all(report[field] is False for field in ('GPU_started', 'fit_started', 'validation_labels_read',
        'effective_SM_service_identified', 'independent_prediction_accuracy_evaluated',
        'physically_validated', 'deployment_qualified')), 'Overclaimed training diagnostics')
    result = dict(schema='gsdeploy.unified_kernel_dag_intake_independent_review.v1', status='PASS',
        reviewed_report=ref(report_path), reviewer_source=ref(__file__), checked_receipts=len(receipts),
        whole_session_kernels=report['full_session_inventory']['full_session_kernels'],
        real_invalid_native_intervals=0, real_device_counts=dict(devices),
        canonical_nodes=len(launches), per_owner_kernels=dict(per_owner),
        kernel_streams={owner: sorted(value) for owner,value in streams.items()},
        outside_overlap_wall_ns=outside_overlap, independently_checked_active_segments=checked_segments,
        same_owner_peer_segments=same_owner_peer_segments, path_rank=path_rank,
        nuisance_adjusted_derivative_rank=0, fit_started=False, GPU_started=False,
        validation_labels_read=False, prediction_accuracy_evaluated=False, deployment_qualified=False,
        limitations=['Static covariate function is shared; active-set sweep and full source/SQL/node attribution are independent.',
                    'Single native capture has one response per exact node; no response fit or accuracy qualification.'])
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.report, args.output)))
