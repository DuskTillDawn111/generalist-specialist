"""Read-only CUDA device metadata through an explicit driver-API allowlist.

cuInit initializes the driver; no CUDA context is created or retained. No
runtime API, allocation, module load, stream, event, or kernel API is resolved.
This is a metadata query, not a GPU workload/experiment or occupancy proof.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid


SCHEMA = 'gsdeploy.cuda_device_attributes.v2'
DEFAULT_HEADER = Path('/usr/local/cuda/include/cuda.h')
DEFAULT_TRACE = Path(__file__).resolve().parents[2] / 'artifacts/predictor-gpu-20261027/hardware-probe/tiny-cuda-nvtx.sqlite'
ATTRIBUTES = {
    'max_threads_per_block': ('CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK', 1),
    'shared_bytes_per_block': ('CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK', 8),
    'warp_size': ('CU_DEVICE_ATTRIBUTE_WARP_SIZE', 10),
    'registers_per_block': ('CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK', 12),
    'sm_count': ('CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT', 16),
    'max_threads_per_sm': ('CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR', 39),
    'compute_capability_major': ('CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR', 75),
    'compute_capability_minor': ('CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR', 76),
    'shared_bytes_per_sm': ('CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR', 81),
    'registers_per_sm': ('CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR', 82),
    'shared_bytes_per_block_optin': ('CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN', 97),
    'max_blocks_per_sm': ('CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR', 106),
    'reserved_shared_bytes_per_block': ('CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK', 111),
}
ALLOWED_APIS = ('cuInit', 'cuDriverGetVersion', 'cuDeviceGetCount', 'cuDeviceGet',
                'cuDeviceGetName', 'cuDeviceGetUuid', 'cuDeviceGetAttribute')


class CUuuid(ctypes.Structure):
    _fields_ = [('bytes', ctypes.c_ubyte * 16)]


def file_receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def header_evidence(path):
    path = Path(path).resolve()
    text = path.read_text()
    observed = {}
    for field, (name, expected) in ATTRIBUTES.items():
        match = re.search(r'\b' + name + r'\s*=\s*(\d+)\s*,', text)
        observed[field] = dict(name=name, expected_value=expected,
                               observed_value=int(match.group(1)) if match else None)
    return dict(source=file_receipt(path), attributes=observed,
                matches_expected=all(row['observed_value'] == row['expected_value'] for row in observed.values()))


def _configure(cuda):
    signatures = {
        'cuInit': [ctypes.c_uint],
        'cuDriverGetVersion': [ctypes.POINTER(ctypes.c_int)],
        'cuDeviceGetCount': [ctypes.POINTER(ctypes.c_int)],
        'cuDeviceGet': [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        'cuDeviceGetName': [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int],
        'cuDeviceGetUuid': [ctypes.POINTER(CUuuid), ctypes.c_int],
        'cuDeviceGetAttribute': [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
    }
    for name, args in signatures.items():
        function = getattr(cuda, name)
        function.argtypes, function.restype = args, ctypes.c_int


def query_attributes(cuda):
    """Accept a CDLL or CPU fake implementing only ALLOWED_APIS."""
    _configure(cuda)
    calls = []
    def call(name, args, *, device_ordinal=None, attribute=None):
        rc = int(getattr(cuda, name)(*args))
        record = dict(api=name, returncode=rc, success=rc == 0)
        if device_ordinal is not None: record['device_ordinal'] = device_ordinal
        if attribute is not None: record['attribute'] = attribute
        calls.append(record)
        return rc, record
    rc, _ = call('cuInit', (0,))
    result = dict(driver_initialized=rc == 0, api_calls=calls, devices=[], driver_api_version=None,
                  device_count=None, context_creation_requested=False, context_retention_requested=False,
                  kernel_launches_requested=0, memory_allocations_requested=0,
                  gpu_workload_executed=False, query_kind='read_only_cuda_device_metadata')
    if rc:
        result['status'] = 'UNKNOWN_DRIVER_INITIALIZATION_FAILED'
        return result
    version = ctypes.c_int()
    rc, record = call('cuDriverGetVersion', (ctypes.byref(version),))
    if rc == 0: result['driver_api_version'] = version.value; record['value'] = version.value
    count = ctypes.c_int()
    rc, record = call('cuDeviceGetCount', (ctypes.byref(count),))
    if rc or count.value < 0 or count.value > 1024:
        result['status'] = 'UNKNOWN_DEVICE_COUNT'
        return result
    record['value'] = count.value; result['device_count'] = count.value
    for ordinal in range(count.value):
        handle = ctypes.c_int()
        rc, record = call('cuDeviceGet', (ctypes.byref(handle), ordinal), device_ordinal=ordinal)
        device = dict(ordinal=ordinal, handle=None, name=None, uuid=None, uuid_hex=None,
                      attributes={}, hardware=None, hardware_limits_status='UNKNOWN')
        result['devices'].append(device)
        if rc: device['reason'] = 'DEVICE_HANDLE_QUERY_FAILED'; continue
        device['handle'] = handle.value; record['value'] = handle.value
        name = ctypes.create_string_buffer(256)
        rc, record = call('cuDeviceGetName', (name, len(name), handle.value), device_ordinal=ordinal)
        if rc == 0: device['name'] = name.value.decode('utf-8', errors='replace'); record['value'] = device['name']
        device_uuid = CUuuid()
        rc, record = call('cuDeviceGetUuid', (ctypes.byref(device_uuid), handle.value), device_ordinal=ordinal)
        if rc == 0:
            raw = bytes(device_uuid.bytes)
            record['value_hex'] = raw.hex()
            if any(raw):
                device['uuid_hex'] = raw.hex()
                device['uuid'] = 'GPU-' + str(uuid.UUID(bytes=raw))
        for field, (attribute, number) in ATTRIBUTES.items():
            value = ctypes.c_int()
            rc, record = call('cuDeviceGetAttribute', (ctypes.byref(value), number, handle.value),
                              device_ordinal=ordinal, attribute=attribute)
            valid = rc == 0 and value.value >= (0 if field in ('compute_capability_minor', 'reserved_shared_bytes_per_block') else 1)
            device['attributes'][field] = dict(attribute_enum=number, returncode=rc,
                                                value=value.value if valid else None)
            if rc == 0: record['value'] = value.value
        limits = {field: row['value'] for field, row in device['attributes'].items()}
        warp, threads = limits.get('warp_size'), limits.get('max_threads_per_sm')
        limits['max_warps_per_sm'] = threads // warp if warp and threads and threads % warp == 0 else None
        if device['uuid'] and all(value is not None for value in limits.values()):
            device['hardware_limits_status'] = 'OBSERVED_DEVICE_LIMITS'
            device['hardware'] = dict(hardware_id=device['uuid'], **limits)
        else:
            device['reason'] = 'MISSING_VALID_IDENTITY_OR_LIMITS_OR_EXACT_WARP_DERIVATION'
        device['max_warps_derivation'] = 'max_threads_per_sm / warp_size; accepted only for an exact integer quotient'
        device['architecture_allocation_status'] = 'UNKNOWN_KERNEL_SPECIFIC_EVIDENCE_NOT_QUERIED'
    result['status'] = ('METADATA_QUERY_COMPLETE' if all(d['hardware'] for d in result['devices'])
                        else 'PARTIAL_DEVICE_METADATA')
    return result


def _uuid_hex(value):
    if isinstance(value, bytes):
        return value.hex() if len(value) == 16 and any(value) else None
    if not isinstance(value, str): return None
    text = value[4:] if value.startswith('GPU-') else value
    try:
        parsed = uuid.UUID(text)
    except ValueError:
        return None
    return parsed.hex if parsed.int else None


def trace_uuid_mapping(path, devices):
    if path is None or not Path(path).is_file():
        return dict(status='UNKNOWN_TRACE_NOT_SUPPLIED_OR_MISSING', rows=[])
    source = file_receipt(path)
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        if 'TARGET_INFO_CUDA_DEVICE' not in tables:
            return dict(status='UNKNOWN_TRACE_DEVICE_TABLE_MISSING', source=source, rows=[])
        rows = []
        for row in connection.execute('select * from TARGET_INFO_CUDA_DEVICE'):
            raw = dict(row); raw_uuid = raw.get('uuid'); normalized = _uuid_hex(raw_uuid)
            matches = [d for d in devices if normalized and normalized == d.get('uuid_hex')]
            rows.append(dict(trace_cuda_id=raw.get('cudaId'), trace_gpu_id=raw.get('gpuId'),
                trace_uuid=raw_uuid.hex() if isinstance(raw_uuid, bytes) else raw_uuid,
                normalized_uuid_hex=normalized,
                status='EXACT_UUID_MATCH' if len(matches) == 1 else 'UNKNOWN_NO_UNIQUE_EXACT_UUID_MATCH',
                queried_ordinal=matches[0]['ordinal'] if len(matches) == 1 else None,
                hardware_id=matches[0]['uuid'] if len(matches) == 1 else None))
    return dict(status='EXACT_UUIDS_ONLY', source=source, rows=rows,
                ordinal_or_name_or_sm_count_fallback_allowed=False)


def _loaded_driver_sources():
    sources = []
    try:
        lines = Path('/proc/self/maps').read_text().splitlines()
    except OSError:
        return sources
    paths = {line.split(None, 5)[-1] for line in lines if len(line.split(None, 5)) == 6
             and Path(line.split(None, 5)[-1]).name.startswith('libcuda.so')}
    for path in sorted(paths):
        try: sources.append(file_receipt(path))
        except OSError: pass
    return sources


def audit(output, *, header=DEFAULT_HEADER, trace=DEFAULT_TRACE, library='libcuda.so.1', library_loader=None):
    if Path(output).exists() or Path(output).is_symlink(): raise FileExistsError(output)
    output = Path(output).resolve()
    if trace is not None and output.is_relative_to(Path(trace).resolve().parent):
        raise ValueError('Metadata output cannot be inside the historical trace directory')
    evidence = header_evidence(header)
    if not evidence['matches_expected']:
        raise ValueError('CUDA header attribute enums do not match the declared query schema')
    trace_before = file_receipt(trace) if trace is not None and Path(trace).is_file() else None
    output.mkdir(parents=True, exist_ok=False)
    load_error = None
    try:
        cuda = (library_loader or ctypes.CDLL)(library)
        query = query_attributes(cuda)
    except (OSError, AttributeError) as exc:
        load_error = dict(type=type(exc).__name__, message=str(exc))
        query = dict(status='UNKNOWN_DRIVER_API_UNAVAILABLE', devices=[], api_calls=[])
    driver_sources = _loaded_driver_sources()
    proc_version = Path('/proc/driver/nvidia/version')
    driver_version = file_receipt(proc_version) if proc_version.is_file() else None
    driver_text = proc_version.read_text() if proc_version.is_file() else None
    for device in query['devices']:
        if device.get('hardware'):
            binding = dict(uuid=device['uuid'], attributes=device['attributes'], header=evidence['source'],
                           driver_binaries=driver_sources, driver_version=driver_text,
                           driver_api_version=query.get('driver_api_version'))
            digest = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
            device['hardware']['evidence'] = 'cuda_device_metadata_v2:' + digest
            device['hardware_evidence_sha256'] = digest
    mapping = trace_uuid_mapping(trace, query['devices'])
    trace_after = file_receipt(trace) if trace_before else None
    result = dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
                  query_kind='READ_ONLY_CUDA_DEVICE_METADATA_NOT_GPU_EXPERIMENT',
                  gpu_experiment_started=False, context_creation_requested=False,
                  context_retention_requested=False, kernel_launches_requested=0,
                  permissions_changed=False, allowed_apis=list(ALLOWED_APIS),
                  library_requested=library, library_load_error=load_error,
                  source_code=file_receipt(__file__), header=evidence,
                  driver_binary_sources=driver_sources, driver_version_source=driver_version,
                  driver_version_text=driver_text, query=query, trace_uuid_mapping=mapping,
                  trace_source_before=trace_before, trace_source_after=trace_after,
                  historical_trace_unchanged=trace_before == trace_after,
                  architecture_allocation_status='UNKNOWN_KERNEL_SPECIFIC_ALLOCATION_NOT_QUERIED',
                  occupancy_status='UNKNOWN_NO_KERNEL_ALLOCATION_EVIDENCE',
                  actual_model_baseline_status='UNKNOWN_NOT_QUERIED',
                  physically_validated=False, deployment_qualified=False,
                  interpretation='Observed static hardware limits only. cuInit driver initialization is explicit; no context creation, GPU workload, counter test, or achieved occupancy claim.')
    with (output / 'device-attributes.json').open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    with (output / 'output-manifest.json').open('x') as stream:
        json.dump(dict(files=[file_receipt(output / 'device-attributes.json')]), stream, indent=2); stream.write('\n')
    if trace_before != trace_after: raise RuntimeError('Historical trace changed during read-only metadata query')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--header', type=Path, default=DEFAULT_HEADER)
    parser.add_argument('--trace', type=Path, default=DEFAULT_TRACE)
    parser.add_argument('--library', default='libcuda.so.1')
    args = parser.parse_args(argv)
    result = audit(args.output, header=args.header, trace=args.trace, library=args.library)
    print(json.dumps(dict(output=str(Path(args.output).resolve() / 'device-attributes.json'),
                          status=result['query']['status'], query_kind=result['query_kind'],
                          context_creation_requested=False, gpu_experiment_started=False)))
    return 0


if __name__ == '__main__': raise SystemExit(main())
