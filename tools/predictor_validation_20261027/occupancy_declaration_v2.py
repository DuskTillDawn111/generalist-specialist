"""Offline CC8.9 occupancy declaration minimization against a pinned header.

Separates real observations from calculation representatives for parameters
proved irrelevant to cudaOccMaxActiveBlocksPerMultiprocessor. No CUDA APIs,
profiler processes, permissions, or frozen source changes are involved.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import uuid

from .occupancy_calculator_v2 import calculate_occupancy, receipt


HEADER_SHA256 = '2abf47c8e7be865463040cdb20dc146243606db220223baf2acd3d51984c610a'
CUDA_HEADER_SHA256 = 'b30a4a84da0bd6d034933fc430955a46432b219d1a61f9a9875b31e8b0494c12'
CC89_SHARED_BUCKETS = (0, 8192, 16384, 32768, 65536, 102400)


def _unknown(reason, **extra):
    return dict(status='UNKNOWN', known=False, reason=reason,
                physically_validated=False, deployment_qualified=False, **extra)


def _positive_int(value):
    return type(value) is int and value > 0


def effective_carveout_representatives(effective_bytes, shared_bytes_per_sm):
    """All integer percentages with exactly this header preference capacity.

    This is a mathematical projection of an observed executed capacity. It
    neither estimates nor reconstructs the actual requested percentage.
    """
    if not _positive_int(shared_bytes_per_sm) or shared_bytes_per_sm != 102400:
        raise ValueError('Pinned CC8.9 shared-memory capacity required')
    if type(effective_bytes) is not int or effective_bytes not in CC89_SHARED_BUCKETS:
        raise ValueError('Observed effective shared capacity is not a CC8.9 header bucket')
    def bucket(percentage):
        value = percentage*shared_bytes_per_sm//100
        return next(size for size in CC89_SHARED_BUCKETS if size >= value)
    return [value for value in range(101) if bucket(value) == effective_bytes]


def resolve_declaration(kernel, hardware, function_attributes, device_state, *, calculator,
                        effective_shared_bytes=None, effective_shared_evidence=None,
                        completed_launch_evidence=None):
    """Reduce required fields for the exact pinned CC8.9 max-active-block API.

    Real function maxThreadsPerBlock remains unknown, but a source-bound completed
    launch can prove its lower bound and supply an equivalent representative.
    OPTIN maxDynamicShared can likewise be projected only with the completed
    launch proof and a valid hardware/shared-limit threshold proof. Representatives
    never become observed function/device attributes.
    DEFAULT maxDynamicShared, GC and pre-Hopper barrier count are inapplicable.
    With a known nonnegative carveout projection, cache preference is ignored.
    """
    if ((hardware.get('compute_capability_major'), hardware.get('compute_capability_minor')) != (8, 9)):
        return _unknown('ARCHITECTURE_RULES_NOT_AUDITED')
    header = calculator.get('header', {})
    if header.get('sha256') != HEADER_SHA256 or receipt(header['path']) != header:
        return _unknown('PINNED_OCCUPANCY_HEADER_REQUIRED')
    original_function, original_state = copy.deepcopy(function_attributes), copy.deepcopy(device_state)
    if (any(function_attributes.get(name) is not None for name in ('max_threads_per_block', 'shmem_limit_config'))
            or (function_attributes.get('shmem_limit_config') == 'OPTIN' and
                function_attributes.get('max_dynamic_shared_bytes') is not None)):
        if not function_attributes.get('evidence'):
            return _unknown('OBSERVED_FUNCTION_DECLARATION_EVIDENCE_REQUIRED')
    if effective_shared_bytes is None and any(device_state.get(name) is not None
                                              for name in ('cache_config', 'carveout_config')):
        if not device_state.get('evidence'):
            return _unknown('OBSERVED_DEVICE_STATE_EVIDENCE_REQUIRED')
    function, state = copy.deepcopy(function_attributes), copy.deepcopy(device_state)
    derivations = []

    def project(target, field, value, rule, applicability='not_applicable', evidence=None):
        rows = function if target == 'function' else state
        original = original_function if target == 'function' else original_state
        rows[field] = value
        derivations.append(dict(field=target+'.'+field, observed_value=original.get(field),
                                calculation_representative=value, applicability=applicability,
                                rule=rule, header=header, evidence=evidence))

    if function.get('partitioned_gc_config') is None:
        project('function', 'partitioned_gc_config', 'OFF',
                'cudaOccPartitionedGlobalCachingModeSupport disables GC outside5.2/5.3 and6.x except6.0; cudaOccPartitionedGCExpected forces OFF on8.9')
    if function.get('num_block_barriers') is None:
        project('function', 'num_block_barriers', 0,
                'cudaOccMaxActiveBlocksPerMultiprocessor applies barrier limits only when computeMajor>=9')
    completed_proof_valid = False
    if completed_launch_evidence is not None:
        identity = completed_launch_evidence.get('launch', {}) if isinstance(completed_launch_evidence, dict) else {}
        valid = (type(identity.get('start')) is int and type(identity.get('end')) is int and
                 identity['end'] > identity['start'] and completed_launch_evidence == kernel.get('static_evidence'))
        block = kernel.get('block')
        if not valid or not isinstance(block, (list, tuple)) or len(block) != 3 or not all(_positive_int(v) for v in block):
            return _unknown('COMPLETED_LAUNCH_PROOF_NOT_BOUND_TO_EXACT_LAUNCH')
        threads = math.prod(block)
        if threads > hardware.get('max_threads_per_block', 0):
            return _unknown('COMPLETED_LAUNCH_PROOF_CONFLICTS_WITH_DEVICE_THREADS_LIMIT')
        completed_proof_valid = True
    if function.get('max_threads_per_block') is None and completed_proof_valid:
        project('function', 'max_threads_per_block', threads,
                'Completed exact CUPTI launch proves actual funcMaxThreads>=observed block threads; pinned max-active-block API only checks funcMaxThreads>0 and does not otherwise read it',
                applicability='lower_bound_equivalent_occupancy', evidence=completed_launch_evidence)
    if function.get('max_dynamic_shared_bytes') is None and function.get('shmem_limit_config') == 'DEFAULT':
        project('function', 'max_dynamic_shared_bytes', 0,
                'DEFAULT disables the dynamic-size comparison; cudaOccSMemPerBlock DEFAULT ignores maxSmemUsagePerCTA')
    if (function.get('max_dynamic_shared_bytes') is None and
            function.get('shmem_limit_config') == 'OPTIN' and completed_proof_valid):
        contract_path = Path(header['path']).with_name('cuda.h')
        if not contract_path.is_file() or receipt(contract_path)['sha256'] != CUDA_HEADER_SHA256:
            return _unknown('PINNED_DYNAMIC_SHARED_ATTRIBUTE_CONTRACT_REQUIRED')
        static, dynamic, reserved = (kernel.get('static_shared_bytes'), kernel.get('dynamic_shared_bytes'),
                                     hardware.get('reserved_shared_bytes_per_block'))
        default, optin = hardware.get('shared_bytes_per_block'), hardware.get('shared_bytes_per_block_optin')
        if (any(type(value) is not int or value < 0 for value in (static, dynamic, reserved)) or
                not _positive_int(default) or not _positive_int(optin) or optin < default):
            return _unknown('SHARED_LIMIT_EQUIVALENCE_PROOF_INPUTS_REQUIRED')
        lower, upper = dynamic, optin-static
        total = static+reserved+dynamic
        allocated = ((total+127)//128)*128
        threshold = default-static-reserved
        if lower > upper or allocated > optin+reserved:
            return _unknown('COMPLETED_LAUNCH_CONFLICTS_WITH_OPTIN_HARDWARE_BOUND')
        # The only possible m-dependence after m>=dynamic is the shared-limit
        # choice. If m=dynamic chooses DEFAULT, that limit must already fit;
        # increasing it cannot change whether this launch is feasible. If it
        # chooses OPTIN, every allowed larger m remains in that same branch.
        if total <= default and allocated > default+reserved:
            return _unknown('DEFAULT_TO_OPTIN_EQUIVALENCE_NOT_PROVEN')
        project('function', 'max_dynamic_shared_bytes', dynamic,
                'Completed exact OPTIN launch proves maxDynamic>=dynamic; only remaining dependence is DEFAULT/OPTIN per-block-limit selection. The requested allocation fits DEFAULT whenever the lower-bound representative selects DEFAULT; otherwise every allowed maximum selects OPTIN. SM allocation and residency are independent of this maximum.',
                applicability='lower_bound_observationally_equivalent_occupancy', evidence=completed_launch_evidence)
        derivations[-1]['proof'] = dict(lower_bound=lower, physical_upper_bound=upper,
            upper_bound_basis='cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES plus static shared cannot exceed device optin per-block limit',
            attribute_contract_source=receipt(contract_path),
            default_to_optin_threshold=threshold, rounded_block_allocation=allocated,
            lower_bound_selects='DEFAULT' if total <= default else 'OPTIN',
            allowed_branch_boundary_points=sorted({point for point in (lower, upper, threshold-1, threshold, threshold+1)
                                                  if lower <= point <= upper}),
            excluded_ineligible_control=lower-1 if lower else None,
            actual_max_dynamic_shared_bytes_known=False)
    if effective_shared_bytes is not None:
        if not effective_shared_evidence:
            return _unknown('EFFECTIVE_SHARED_CAPACITY_SOURCE_REQUIRED')
        try:
            representatives = effective_carveout_representatives(effective_shared_bytes, hardware.get('shared_bytes_per_sm'))
        except ValueError as exc:
            return _unknown('EFFECTIVE_SHARED_CAPACITY_NOT_REPRESENTABLE', detail=str(exc))
        static, dynamic, reserved = (kernel.get('static_shared_bytes'), kernel.get('dynamic_shared_bytes'),
                                     hardware.get('reserved_shared_bytes_per_block'))
        if any(type(value) is not int or value < 0 for value in (static, dynamic, reserved)):
            return _unknown('LAUNCH_SHARED_ALLOCATION_EVIDENCE_REQUIRED')
        rounded = ((static+dynamic+reserved+127)//128)*128
        if rounded > effective_shared_bytes:
            return _unknown('EXECUTED_SHARED_CAPACITY_BELOW_ROUNDED_BLOCK_ALLOCATION')
        project('device_state', 'carveout_config', representatives[0],
                'Observed driver-selected sharedMemoryExecuted equals the pinned header rounded preference bucket; all listed percentages yield that exact capacity',
                applicability='equivalent_projection_of_observed_effective_capacity', evidence=effective_shared_evidence)
        derivations[-1]['equivalent_percentages'] = representatives
        derivations[-1]['actual_requested_carveout_known'] = original_state.get('carveout_config') is not None
    if type(state.get('carveout_config')) is int and 0 <= state['carveout_config'] <= 100:
        if state.get('cache_config') is None:
            project('device_state', 'cache_config', 'NONE',
                    'cudaOccSMemPreferenceVoltaPlus reads cacheConfig only when carveoutConfig==-1')
    missing = ['function.'+name for name in ('max_threads_per_block', 'partitioned_gc_config',
               'shmem_limit_config', 'max_dynamic_shared_bytes', 'num_block_barriers') if function.get(name) is None]
    missing += ['device_state.'+name for name in ('cache_config', 'carveout_config') if state.get(name) is None]
    evidence = 'arch_aware_declaration_v2:'+hashlib.sha256(json.dumps(dict(
        original_function=original_function, original_state=original_state, derivations=derivations),
        sort_keys=True).encode()).hexdigest()
    function['evidence'] = dict(original=original_function.get('evidence'), resolution=evidence)
    state['evidence'] = dict(original=original_state.get('evidence'), resolution=evidence)
    report = dict(original_function_attributes=original_function, original_device_state=original_state,
                  calculation_function_attributes=function, calculation_device_state=state,
                  derivations=derivations, remaining_required_fields=missing,
                  real_function_attributes_completed=False,
                  requested_state_reconstructed=False, source=receipt(__file__))
    if missing:
        return _unknown('RELEVANT_FUNCTION_OR_STATE_FIELDS_STILL_UNKNOWN', **report)
    calculation = calculate_occupancy(kernel, hardware, function, state, calculator=calculator)
    return dict(status='ARCH_AWARE_STATIC_CEILING' if calculation['known'] else 'UNKNOWN',
                known=calculation['known'], calculation=calculation, **report,
                physically_validated=False, deployment_qualified=False)


def read_sqlite_launches(path, hardware):
    """Read real launch/configuration fields and exact GPU UUID binding.

    The SQLite cacheConfig enum is retained, without inferring the missing
    requested carveout from it. Underlying CUPTI cache fields can be stale under
    an explicit carveout. Observed effective shared capacity avoids that ambiguity.
    """
    path = Path(path).resolve()
    source = receipt(path)
    connection = sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        devices = [dict(row) for row in connection.execute('select * from TARGET_INFO_GPU')]
        expected_uuid = str(uuid.UUID(hardware['hardware_id'].removeprefix('GPU-')))
        matched = []
        for row in devices:
            try:
                candidate = str(uuid.UUID(str(row.get('uuid')).removeprefix('GPU-')))
            except ValueError:
                continue
            if candidate == expected_uuid:
                matched.append(row)
        if len(matched) != 1:
            return _unknown('NO_UNIQUE_EXACT_TRACE_HARDWARE_UUID_MATCH', source=source, rows=[])
        device = matched[0]
        for name, observed in (('compute_capability_major', 'computeMajor'),
                               ('compute_capability_minor', 'computeMinor'), ('sm_count', 'smCount')):
            if device.get(observed) != hardware.get(name):
                return _unknown('EXACT_UUID_DEVICE_PROPERTIES_DISAGREE', source=source, rows=[])
        modes = {row['id']: row['name'] for row in connection.execute('select * from ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG')}
        launch_modes = {row['id']: row['name'] for row in connection.execute('select * from ENUM_CUDA_KERNEL_LAUNCH_TYPE')}
        rows = []
        columns = {row[1] for row in connection.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
        required = {'rowid', 'start', 'end', 'deviceId', 'contextId', 'correlationId', 'globalPid',
                    'blockX', 'blockY', 'blockZ', 'gridX', 'gridY', 'gridZ',
                    'registersPerThread', 'staticSharedMemory', 'dynamicSharedMemory',
                    'cacheConfig', 'sharedMemoryLimitConfig', 'sharedMemoryExecuted', 'launchType'}
        if required-{'rowid'}-columns:
            return _unknown('MISSING_REAL_SQLITE_LAUNCH_COLUMNS', source=source, rows=[])
        for row in connection.execute('select rowid,* from CUPTI_ACTIVITY_KIND_KERNEL order by start,end,rowid'):
            if row['deviceId'] != device['id']:
                continue
            identity = dict(sqlite_rowid=row['rowid'], start=row['start'], end=row['end'],
                            context_id=row['contextId'], correlation_id=row['correlationId'], global_pid=row['globalPid'])
            mode = modes.get(row['sharedMemoryLimitConfig'])
            decoded = {'CUDA_SHARED_MEM_LIMIT_DEFAULT': 'DEFAULT', 'CUDA_SHARED_MEM_LIMIT_OPTIN': 'OPTIN'}.get(mode)
            cluster = [row['cluster'+axis] for axis in 'XYZ'] if all('cluster'+axis in columns for axis in 'XYZ') else None
            kernel = dict(hardware_id=hardware['hardware_id'], static_evidence=dict(source=source, launch=identity),
                          block=[row['block'+axis] for axis in 'XYZ'], grid=[row['grid'+axis] for axis in 'XYZ'],
                          registers_per_thread=row['registersPerThread'], static_shared_bytes=row['staticSharedMemory'],
                          dynamic_shared_bytes=row['dynamicSharedMemory'],
                          cooperative_launch=launch_modes.get(row['launchType']) != 'CUDA_KERNEL_LAUNCH_TYPE_REGULAR',
                          cluster=None if cluster == [0, 0, 0] else cluster)
            rows.append(dict(launch_identity=identity, kernel=kernel,
                function_attributes=dict(shmem_limit_config=decoded, evidence=dict(source=source, launch=identity,
                    field='sharedMemoryLimitConfig', raw=row['sharedMemoryLimitConfig'], enum_name=mode)),
                device_state=dict(evidence=dict(source=source, launch=identity), cache_config=None, carveout_config=None),
                effective_shared_bytes=row['sharedMemoryExecuted'],
                effective_shared_evidence=dict(source=source, launch=identity, field='sharedMemoryExecuted'),
                observed_raw_cache_config=row['cacheConfig'],
                observed_launch_type=launch_modes.get(row['launchType']),
                actual_function_attributes_unknown=['max_threads_per_block', 'max_dynamic_shared_bytes']))
            if type(row['start']) is int and type(row['end']) is int and row['end'] > row['start']:
                rows[-1]['completed_launch_evidence'] = kernel['static_evidence']
        return dict(status='READ_ONLY_LAUNCH_METADATA', known=True, source=source, device=device,
                    uuid_binding='UNIQUE_EXACT_UUID', rows=rows,
                    physically_validated=False, deployment_qualified=False)
    finally:
        connection.close()
