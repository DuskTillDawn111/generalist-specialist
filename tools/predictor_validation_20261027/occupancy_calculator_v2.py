"""CPU-only wrapper around NVIDIA's architecture-aware occupancy header.

Compiles a host C++ program using g++; never links CUDA libraries or invokes
device code. All function launch settings must be explicit and evidenced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


DEFAULT_HEADER = Path('/usr/local/cuda/include/cuda_occupancy.h')
GC = {'OFF': 0, 'ON': 1, 'ON_STRICT': 2}
SHMEM = {'DEFAULT': 0, 'OPTIN': 1}
CACHE = {'NONE': 0, 'SHARED': 1, 'L1': 2, 'EQUAL': 3}
HARDWARE = ('compute_capability_major', 'compute_capability_minor', 'max_threads_per_block',
            'max_threads_per_sm', 'registers_per_block', 'registers_per_sm', 'warp_size',
            'shared_bytes_per_block', 'shared_bytes_per_sm', 'sm_count',
            'shared_bytes_per_block_optin', 'reserved_shared_bytes_per_block', 'max_blocks_per_sm')
FUNCTION = ('max_threads_per_block', 'partitioned_gc_config', 'shmem_limit_config',
            'max_dynamic_shared_bytes', 'num_block_barriers')
STATE = ('cache_config', 'carveout_config')


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def build_calculator(output, *, header=DEFAULT_HEADER, compiler='g++'):
    """Build a new CPU executable plus source/compiler/header receipts."""
    output = Path(output)
    if output.exists() or output.is_symlink(): raise FileExistsError(output)
    header = Path(header).resolve()
    if not header.is_file(): raise FileNotFoundError(header)
    if header.name != 'cuda_occupancy.h': raise ValueError('Expected an explicit cuda_occupancy.h source')
    compiler_path = shutil.which(compiler)
    if compiler_path is None: raise FileNotFoundError('Host C++ compiler unavailable: ' + compiler)
    source = Path(__file__).with_suffix('.cpp').resolve()
    output.mkdir(parents=True, exist_ok=False); binary = output.resolve() / 'occupancy-host'
    command = [compiler_path, '-std=c++11', '-O2', '-Wall', '-Wextra', '-I', str(header.parent), str(source), '-o', str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    record = dict(schema='gsdeploy.host_occupancy_calculator_build.v2', command=command,
                  returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                  header=receipt(header), source=receipt(source), wrapper=receipt(__file__),
                  compiler=receipt(compiler_path), links_cuda=False, GPU_workload_executed=False)
    if result.returncode == 0: record['binary'] = receipt(binary)
    _write(output / 'build-receipt.json', record)
    if result.returncode: raise RuntimeError('Host-only occupancy compilation failed: ' + result.stderr)
    return record


def _unknown(reason, **extra):
    return dict(status='UNKNOWN', known=False, reason=reason, physically_validated=False,
                deployment_qualified=False, **extra)


def _integer(value, name, *, zero=False, negative_one=False):
    if type(value) is not int or value < (-1 if negative_one else 0 if zero else 1) or value > 2**31 - 1:
        raise ValueError('Invalid bounded integer ' + name)
    return value


def calculate_occupancy(kernel, hardware, function_attributes, device_state, *, calculator,
                        assumed_fields=(), assumption_evidence=None, allow_assumptions=False):
    """Return architecture allocation/ceiling only for the complete declaration.

    Missing per-function opt-in/carveout settings are UNKNOWN. Explicitly
    assumed settings require a separate opt-in and produce no kernel_patch.
    No real kernel function is loaded or queried by this CPU calculation.
    """
    missing = ['hardware.' + name for name in (*HARDWARE, 'hardware_id', 'evidence') if hardware.get(name) is None]
    missing += ['kernel.' + name for name in ('hardware_id', 'static_evidence', 'block', 'registers_per_thread',
                'static_shared_bytes', 'dynamic_shared_bytes') if kernel.get(name) is None]
    missing += ['function.' + name for name in (*FUNCTION, 'evidence') if function_attributes.get(name) is None]
    missing += ['device_state.' + name for name in (*STATE, 'evidence') if device_state.get(name) is None]
    if missing: return _unknown('EXPLICIT_LAUNCH_AND_DEVICE_FIELDS_REQUIRED', missing_fields=missing)
    if kernel['hardware_id'] != hardware['hardware_id']:
        return _unknown('HARDWARE_IDENTITY_MISMATCH')
    if any(not row.get(key) for row, key in ((kernel, 'static_evidence'), (hardware, 'evidence'),
                                           (function_attributes, 'evidence'), (device_state, 'evidence'))):
        return _unknown('NONEMPTY_RESOURCE_EVIDENCE_REQUIRED')
    cluster = kernel.get('cluster')
    if cluster is not None:
        if not isinstance(cluster, (tuple, list)) or len(cluster) != 3: raise ValueError('Three cluster dimensions required')
        cluster_blocks = math.prod(_integer(value, 'cluster dimension') for value in cluster)
    else:
        cluster_blocks = 1
    if kernel.get('cooperative_launch') or cluster_blocks != 1:
        return _unknown('CLUSTER_OR_COOPERATIVE_RESIDENCY_NOT_MODELED')
    if type(allow_assumptions) is not bool: raise ValueError('Boolean assumption opt-in required')
    allowed_assumptions = {'function.' + key for key in FUNCTION} | {'device_state.' + key for key in STATE}
    if not isinstance(assumed_fields, (list, tuple)) or any(key not in allowed_assumptions for key in assumed_fields):
        raise ValueError('Assumptions must explicitly identify function/device-state fields')
    if assumed_fields and (not allow_assumptions or not assumption_evidence):
        return _unknown('EXPLICIT_ASSUMPTION_OPT_IN_AND_SOURCE_REQUIRED', assumed_fields=list(assumed_fields))
    for key in HARDWARE:
        _integer(hardware[key], 'hardware.' + key, zero=key in ('compute_capability_minor', 'reserved_shared_bytes_per_block'))
    block = kernel['block']
    if not isinstance(block, (list, tuple)) or len(block) != 3: raise ValueError('Three block dimensions required')
    threads = math.prod(_integer(value, 'block dimension') for value in block)
    _integer(threads, 'block threads')
    for key in ('registers_per_thread', 'static_shared_bytes', 'dynamic_shared_bytes'):
        _integer(kernel[key], 'kernel.' + key, zero=True)
    for key in ('max_threads_per_block', 'max_dynamic_shared_bytes', 'num_block_barriers'):
        _integer(function_attributes[key], 'function.' + key, zero=key != 'max_threads_per_block')
    for key, choices in (('partitioned_gc_config', GC), ('shmem_limit_config', SHMEM)):
        if function_attributes[key] not in choices: raise ValueError('Unsupported explicit function.' + key)
    if device_state['cache_config'] not in CACHE: raise ValueError('Unsupported explicit cache_config')
    carveout = _integer(device_state['carveout_config'], 'carveout_config', negative_one=True)
    if carveout > 100: raise ValueError('Carveout percentage exceeds100')
    binary = calculator.get('binary')
    if not binary or receipt(binary['path']) != binary:
        return _unknown('HOST_CALCULATOR_BINARY_EVIDENCE_MISMATCH')
    if receipt(calculator['header']['path']) != calculator['header']:
        return _unknown('HOST_CALCULATOR_HEADER_CHANGED')
    values = [hardware[key] for key in HARDWARE[:-1]]
    values += [function_attributes['max_threads_per_block'], kernel['registers_per_thread'],
               kernel['static_shared_bytes'], GC[function_attributes['partitioned_gc_config']],
               SHMEM[function_attributes['shmem_limit_config']], function_attributes['max_dynamic_shared_bytes'],
               function_attributes['num_block_barriers'], CACHE[device_state['cache_config']], carveout,
               threads, kernel['dynamic_shared_bytes']]
    proc = subprocess.run([binary['path']], input=' '.join(map(str, values)) + '\n',
                          text=True, capture_output=True, timeout=10)
    if proc.returncode: return _unknown('HOST_CALCULATOR_PROCESS_FAILED', returncode=proc.returncode, stderr=proc.stderr)
    calculated = json.loads(proc.stdout)
    if calculated['returncode']:
        return _unknown('HEADER_REJECTED_DEVICE_OR_LAUNCH', calculation=calculated)
    if calculated['block_limits']['blocks'] != hardware['max_blocks_per_sm']:
        return _unknown('HEADER_ARCHITECTURE_BLOCK_LIMIT_DISAGREES_WITH_DEVICE', calculation=calculated)
    binding = dict(kernel=kernel, hardware=hardware, function_attributes=function_attributes,
                   device_state=device_state, calculation=calculated, header=calculator['header'], binary=binary,
                   assumed_fields=list(assumed_fields), assumption_evidence=assumption_evidence)
    source = 'host_occupancy_calculator_v2:' + hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    patch = dict(allocated_registers_per_block=calculated['allocated_registers_per_block'],
                 allocated_shared_bytes=calculated['allocated_shared_bytes'], allocation_evidence=source,
                 architecture_blocks_per_sm_ceiling=calculated['active_blocks_per_sm'])
    return dict(status='ASSUMED_STATIC_HYPOTHESIS' if assumed_fields else 'HOST_CALCULATED_STATIC_CEILING', known=True,
                calculation=calculated, kernel_patch=patch if not assumed_fields else None,
                declaration=binding, allocation_evidence=source, assumed_fields=list(assumed_fields),
                physically_validated=False, deployment_qualified=False,
                actual_concurrent_residency_proven=False,
                interpretation='Architecture-aware header calculation for explicit launch declarations; no achieved occupancy, throughput, or real function-attribute observation is implied.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--header', default=DEFAULT_HEADER, type=Path)
    parser.add_argument('--allow-assumptions', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink(): raise FileExistsError(args.output)
    request = json.loads(args.request.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    calculator = build_calculator(args.output / 'build', header=args.header)
    result = calculate_occupancy(request['kernel'], request['hardware'], request['function_attributes'],
        request['device_state'], calculator=calculator, assumed_fields=request.get('assumed_fields', ()),
        assumption_evidence=request.get('assumption_evidence'), allow_assumptions=args.allow_assumptions)
    _write(args.output / 'result.json', dict(schema='gsdeploy.host_occupancy_calculation.v2', result=result,
                                            request=receipt(args.request), GPU_workload_executed=False))
    print(json.dumps(dict(output=str((args.output / 'result.json').resolve()), status=result['status'], GPU_workload_executed=False)))
    return 0


if __name__ == '__main__': raise SystemExit(main())
