"""Read-only CPU audit of existing profiler evidence and local permissions.

This command reparses an EXISTING SQLite trace. It does not call nsys/ncu,
CUDA, torch, nvidia-smi, model code, a shell, or a privilege-changing API.
All new files go in a new output directory; legacy artifacts remain untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3

from .trace_export import parse_sqlite


SCHEMA = 'gsdeploy.capability_audit.v2'
DEFAULT_PROBE = Path(__file__).resolve().parents[2] / 'artifacts/predictor-gpu-20261027/hardware-probe'
CAP_SYS_ADMIN_BIT = 21


def file_receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _read_optional(path):
    try:
        text = Path(path).read_text()
        return dict(status='READ', path=str(Path(path).resolve()), text=text,
                    captured_sha256=hashlib.sha256(text.encode()).hexdigest())
    except (OSError, UnicodeError) as exc:
        return dict(status='UNKNOWN', path=str(Path(path).absolute()), error=type(exc).__name__)


def permission_evidence(driver_text, process_text):
    """Decode only explicit observed permission fields; never probe counters."""
    restriction = re.search(r'^RmProfilingAdminOnly:\s*(\d+)\s*$', driver_text, re.M)
    caps = re.search(r'^CapEff:\s*([0-9a-fA-F]+)\s*$', process_text, re.M)
    restriction = int(restriction.group(1)) if restriction else None
    cap_hex = caps.group(1) if caps else None
    has_admin = bool(int(cap_hex, 16) & (1 << CAP_SYS_ADMIN_BIT)) if cap_hex else None
    if restriction == 1 and has_admin is False:
        status = 'ADMIN_PROFILING_RESTRICTION_WITHOUT_CAP_SYS_ADMIN'
    elif restriction is None or has_admin is None:
        status = 'UNKNOWN_PERMISSION_METADATA'
    else:
        status = 'METADATA_DOES_NOT_ESTABLISH_COUNTER_ACCESS'
    selected = {}
    for key in ('Uid', 'Gid', 'CapEff', 'CapBnd', 'NoNewPrivs', 'Seccomp'):
        match = re.search(r'^' + key + r':\s*(.*?)\s*$', process_text, re.M)
        selected[key] = match.group(1) if match else None
    return dict(status=status, rm_profiling_admin_only=restriction, cap_eff_hex=cap_hex,
                cap_sys_admin_effective=has_admin, cap_sys_admin_bit=CAP_SYS_ADMIN_BIT,
                selected_process_fields=selected,
                counter_access_current='UNKNOWN_NOT_REPROBED', permissions_changed=False,
                interpretation='Observed admin-only restriction and effective capability are diagnostic metadata; UID 0 alone is not CAP_SYS_ADMIN or an access test.')


def sqlite_device_metadata(path):
    # Metadata only: no CUDA driver/device API is imported or initialized.
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
        table = 'TARGET_INFO_CUDA_DEVICE'
        if table not in tables:
            return dict(status='UNKNOWN', reason='CUDA_DEVICE_METADATA_TABLE_ABSENT')
        columns = [r[1] for r in connection.execute('pragma table_info(TARGET_INFO_CUDA_DEVICE)')]
        return dict(status='OBSERVED_TRACE_METADATA', columns=columns,
                    devices=[dict(row) for row in connection.execute('select * from TARGET_INFO_CUDA_DEVICE')],
                    scope='Historical trace device IDs; not automatically mapped to present device ordinal/UUID')


def _kernel_inventory(parsed):
    # parse_sqlite aliases attributed kernels in nvtx_ranges and phases; choose
    # the unanchored source once rather than double-counting those references.
    return [kernel for item in parsed['nvtx_ranges'] for kernel in item['kernels']] + parsed['unattributed_kernels']


def audit(output, *, probe_directory=DEFAULT_PROBE, proc_root=Path('/proc')):
    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError(output)
    output, probe, proc_root = Path(output).resolve(), Path(probe_directory).resolve(), Path(proc_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(probe):
        raise ValueError('Output must not be inside the legacy evidence directory')
    if not probe.is_dir():
        raise FileNotFoundError(probe)
    sqlite_path, receipt_path = probe / 'tiny-cuda-nvtx.sqlite', probe / 'receipt.json'
    if not sqlite_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError('Existing tiny SQLite trace and hardware-probe receipt required')
    before = [file_receipt(path) for path in sorted(probe.rglob('*')) if path.is_file()]
    historical = json.loads(receipt_path.read_text())
    if historical.get('schema') != 'gsdeploy.hardware_probe.v1':
        raise ValueError('Explicit historical hardware-probe receipt required')
    records = {record['name']: record for record in historical.get('commands', [])}
    output.mkdir(parents=True, exist_ok=False)

    parsed = parse_sqlite(sqlite_path)  # Opens original SQLite mode=ro; no export/probe.
    _write_json(output / 'reparsed-kernels.json', parsed)
    devices = sqlite_device_metadata(sqlite_path)
    _write_json(output / 'historical-cuda-device-metadata.json', devices)

    raw = {}
    for name, path in (('driver-params', proc_root / 'driver/nvidia/params'),
                       ('process-status', proc_root / 'self/status'),
                       ('driver-version', proc_root / 'driver/nvidia/version')):
        item = _read_optional(path)
        if item['status'] == 'READ':
            captured = output / (name + '.txt')
            with captured.open('x') as stream:
                stream.write(item['text'])
            item['capture'] = file_receipt(captured)
        raw[name] = item
    permission = permission_evidence(raw['driver-params'].get('text', ''), raw['process-status'].get('text', ''))
    proc_devices = []
    for path in sorted((proc_root / 'driver/nvidia/gpus').glob('*/information')):
        captured = _read_optional(path)
        proc_devices.append(captured)
    _write_json(output / 'current-proc-device-metadata.json', proc_devices)

    counter_csv = _read_optional(probe / 'tiny-counters.csv')
    denied = 'ERR_NVGPUCTRPERM' in counter_csv.get('text', '')
    stdout = _read_optional(probe / 'nsys.stdout.log')
    historical_workload = None
    for line in stdout.get('text', '').splitlines():
        if line.startswith('{'):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and 'gpu' in candidate:
                historical_workload = candidate
                break
    kernels = _kernel_inventory(parsed)
    geometry = {key: sum(kernel.get(key) is not None for kernel in kernels) for key in ('block', 'grid')}
    allocation_missing = [i for i, kernel in enumerate(kernels) if
                          kernel.get('allocated_registers_per_block') is None or
                          kernel.get('allocated_shared_bytes') is None or not kernel.get('allocation_evidence')]
    hardware_required = ['warp_size', 'max_threads_per_block', 'max_threads_per_sm', 'max_warps_per_sm',
                         'max_blocks_per_sm', 'registers_per_sm', 'shared_bytes_per_sm']

    historical_hash_checks = []
    for source in [historical, json.loads((probe / 'export-receipt.json').read_text())
                   if (probe / 'export-receipt.json').is_file() else {}]:
        for record in source.get('files', []):
            # Match basename to this explicit source directory; never follow
            # arbitrary historical paths or commands outside it.
            current = probe / Path(record['path']).name
            if record.get('sha256'):
                historical_hash_checks.append(dict(path=str(current), expected_sha256=record['sha256'],
                                                   match=current.is_file() and file_receipt(current)['sha256'] == record['sha256']))
    after = [file_receipt(path) for path in sorted(probe.rglob('*')) if path.is_file()]
    unchanged = before == after
    nsys_binary = shutil.which('nsys')
    ncu_command = records.get('ncu', {}).get('command', [])
    ncu_binary = ncu_command[0] if ncu_command else None
    metric_index = ncu_command.index('--metrics') if '--metrics' in ncu_command else None
    requested_metrics = (ncu_command[metric_index + 1].split(',')
                         if metric_index is not None and metric_index + 1 < len(ncu_command) else [])
    result = dict(
        schema=SCHEMA, status='CPU_READ_ONLY_AUDIT_COMPLETE' if unchanged else 'SOURCE_CHANGED_DURING_AUDIT',
        created_utc=datetime.now(timezone.utc).isoformat(), output_directory=str(output),
        GPU_started=False, new_gpu_experiments_run=False, new_profiler_invocations=0,
        device_queries_executed=[], permissions_changed=False,
        source_inventory_before=before, source_inventory_after=after, legacy_files_unchanged=unchanged,
        historical_receipt_hash_checks=historical_hash_checks,
        historical_source_integrity=('MATCH' if historical_hash_checks and all(c['match'] for c in historical_hash_checks)
                                     else 'MISMATCH' if any(not c['match'] for c in historical_hash_checks) else 'UNKNOWN'),
        source_code=[file_receipt(Path(__file__)), file_receipt(Path(__file__).with_name('trace_export.py'))],
        historical_nsys=dict(returncode=records.get('nsys', {}).get('returncode'),
                             trace_observation='OBSERVED_KERNELS' if kernels else 'NO_PARSED_KERNELS',
                             kernel_count=len(kernels), historical_workload=historical_workload,
                             scope='Past tiny synthetic workload; no new nsys invocation'),
        current_nsys_installation=dict(executable_path=nsys_binary, invoked=False,
                                       current_collection_success='UNKNOWN_NOT_REPROBED'),
        historical_ncu=dict(returncode=records.get('ncu', {}).get('returncode'),
                            err_nvgpuctrperm_observed=denied,
                            status='HISTORICAL_PERMISSION_DENIED' if denied else 'UNKNOWN',
                            counter_csv_source={k: v for k, v in counter_csv.items() if k != 'text'},
                            requested_metrics=requested_metrics,
                            current_counter_access='UNKNOWN_NOT_REPROBED'),
        current_ncu_installation=dict(historical_executable_path=ncu_binary,
                                      executable_present=bool(ncu_binary and Path(ncu_binary).is_file() and os.access(ncu_binary, os.X_OK)),
                                      invoked=False),
        permission_evidence=permission,
        proc_evidence={name: {k: v for k, v in item.items() if k != 'text'} for name, item in raw.items()},
        cuda_device_metadata=devices,
        device_identity_binding=dict(status='UNKNOWN',
            historical_name=historical_workload.get('gpu') if historical_workload else None,
            present_proc_devices=len(proc_devices),
            reason='Trace device UUID may be absent; current physical minor numbers do not prove historical CUDA ordinal mapping'),
        kernel_static_resources=dict(total_kernels=len(kernels), observed_geometry_counts=geometry,
            architecture_allocation_missing_indices=allocation_missing,
            missing_hardware_limit_evidence=hardware_required,
            occupancy_status='UNKNOWN',
            reason='Launch geometry does not provide architecture-rounded allocation or complete device residency limits'),
        actual_model_baselines=dict(status='UNKNOWN', supplied=False,
            reason='Only the prior tiny synthetic probe is audited; no training-bound target-model unprofiled isolated ordered kernel baseline is supplied',
            synthetic_probe_transfer_allowed=False),
        compute_dram_complementarity=dict(status='UNKNOWN',
            reason='No usable compute/DRAM counters or configuration-bound target-model throughput evidence'),
        physically_validated=False, deployment_qualified=False,
        interpretation='Read-only evidence inventory; no permission repair, counter unlock, current profiler success, actual occupancy, or target-model prediction validation is claimed')
    _write_json(output / 'audit.json', result)
    _write_json(output / 'output-manifest.json', dict(schema='gsdeploy.capability_audit_outputs.v2',
        files=[file_receipt(path) for path in sorted(output.iterdir()) if path.is_file()]))
    if not unchanged:
        raise RuntimeError('Historical source inventory changed during audit; see audit.json')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe-directory', type=Path, default=DEFAULT_PROBE)
    parser.add_argument('--proc-root', type=Path, default=Path('/proc'),
                        help='Read-only proc tree; override supports CPU fixture tests')
    args = parser.parse_args(argv)
    result = audit(args.output, probe_directory=args.probe_directory, proc_root=args.proc_root)
    print(json.dumps(dict(output=str(Path(args.output).resolve() / 'audit.json'), status=result['status'],
                          GPU_started=False, legacy_files_unchanged=result['legacy_files_unchanged'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
