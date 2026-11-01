"""Read frozen SQLite launch metadata and prove host-header branch sensitivity."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .occupancy_calculator_v2 import DEFAULT_HEADER, build_calculator, calculate_occupancy, receipt
from .occupancy_declaration_v2 import read_sqlite_launches, resolve_declaration


def branch_sensitivity(calculator):
    """Synthetic host-only formula checks; never target-model evidence."""
    hardware = dict(hardware_id='synthetic-cc89', evidence='synthetic-device',
        compute_capability_major=8, compute_capability_minor=9, max_threads_per_block=1024,
        max_threads_per_sm=1536, registers_per_block=65536, registers_per_sm=65536,
        warp_size=32, shared_bytes_per_block=49152, shared_bytes_per_sm=102400,
        sm_count=128, shared_bytes_per_block_optin=101376,
        reserved_shared_bytes_per_block=1024, max_blocks_per_sm=24)
    kernel = dict(hardware_id='synthetic-cc89', static_evidence='synthetic-launch',
                  block=[128,1,1], grid=[4096,1,1], registers_per_thread=32,
                  static_shared_bytes=8192, dynamic_shared_bytes=0)
    function = dict(max_threads_per_block=1024, partitioned_gc_config='OFF',
                    shmem_limit_config='DEFAULT', max_dynamic_shared_bytes=0,
                    num_block_barriers=1, evidence='synthetic-function')
    state = dict(cache_config='NONE', carveout_config=32, evidence='synthetic-state')
    base = (kernel, hardware, function, state)
    cases = []
    def calculate(values):
        result = calculate_occupancy(*values, calculator=calculator)
        if not result['known']:
            raise RuntimeError('Synthetic sensitivity declaration unexpectedly unsupported')
        return result['calculation']
    reference = calculate(base)
    for field, variants in [('partitioned_gc_config', ['OFF','ON','ON_STRICT']),
                            ('num_block_barriers', [0,1,64]),
                            ('max_dynamic_shared_bytes', [0,100000,2**31-1]),
                            ('max_threads_per_block', [128,256,1024])]:
        for value in variants:
            changed = copy.deepcopy(base); changed[2][field] = value
            calculated = calculate(changed)
            cases.append(dict(case='cc89_'+field, value=value, calculation=calculated,
                              expected='identical_to_reference', passed=calculated == reference))
    for value in ('NONE','SHARED','L1','EQUAL'):
        changed = copy.deepcopy(base); changed[3]['cache_config'] = value
        calculated = calculate(changed)
        cases.append(dict(case='cc89_cache_with_explicit_carveout', value=value,
                          calculation=calculated, expected='identical_to_reference', passed=calculated == reference))
    changed = copy.deepcopy(base)
    changed[0].update(static_shared_bytes=0, dynamic_shared_bytes=60000)
    changed[2]['shmem_limit_config'] = 'OPTIN'; changed[3]['carveout_config'] = 100
    first = calculate(changed); changed[2]['max_dynamic_shared_bytes'] = 60000
    second = calculate(changed)
    cases.append(dict(case='cc89_optin_without_launch_proof_limit_is_relevant', first=first, second=second,
                      passed=(first['active_blocks_per_sm'],second['active_blocks_per_sm']) == (0,1)))
    for static, dynamic in ((0,0), (0,1024), (0,48128), (0,48129),
                             (1024,47104), (1024,47105), (8192,60000)):
        changed = copy.deepcopy(base)
        changed[0].update(static_shared_bytes=static, dynamic_shared_bytes=dynamic)
        changed[2].update(shmem_limit_config='OPTIN', max_dynamic_shared_bytes=dynamic)
        changed[3]['carveout_config'] = 100
        reference_optin = calculate(changed)
        threshold = hardware['shared_bytes_per_block']-static-hardware['reserved_shared_bytes_per_block']
        upper = hardware['shared_bytes_per_block_optin']-static
        for maximum in sorted({dynamic, upper, threshold-1, threshold, threshold+1}):
            if not dynamic <= maximum <= upper:
                continue
            changed[2]['max_dynamic_shared_bytes'] = maximum
            result = calculate(changed)
            cases.append(dict(case='cc89_optin_completed_launch_equivalence', static_shared_bytes=static,
                              dynamic_shared_bytes=dynamic, max_dynamic_representative=maximum,
                              lower_bound=dynamic, physical_upper_bound=upper, branch_threshold=threshold,
                              calculation=result, passed=result == reference_optin))
        if dynamic:
            changed[2]['max_dynamic_shared_bytes'] = dynamic-1
            result = calculate(changed)
            cases.append(dict(case='cc89_optin_excluded_ineligible_control', static_shared_bytes=static,
                              dynamic_shared_bytes=dynamic, max_dynamic_representative=dynamic-1,
                              calculation=result, passed=result['active_blocks_per_sm'] == 0))
    changed = copy.deepcopy(base)
    changed[0]['static_shared_bytes'] = 0
    changed[1].update(compute_capability_major=9, compute_capability_minor=0,
                      max_threads_per_sm=2048, max_blocks_per_sm=32,
                      shared_bytes_per_sm=233472, shared_bytes_per_block_optin=232448)
    changed[3]['carveout_config'] = 100
    changed[2]['num_block_barriers'] = 0; first = calculate(changed)
    changed[2]['num_block_barriers'] = 8; second = calculate(changed)
    cases.append(dict(case='cc90_barriers_are_relevant', first=first, second=second,
                      passed=second['active_blocks_per_sm'] < first['active_blocks_per_sm']))
    changed = copy.deepcopy(base)
    changed[0].update(block=[96,1,1], registers_per_thread=16, static_shared_bytes=0)
    changed[1].update(compute_capability_major=5, compute_capability_minor=2,
                      max_threads_per_sm=2048, max_blocks_per_sm=32,
                      shared_bytes_per_sm=98304, shared_bytes_per_block_optin=49152,
                      reserved_shared_bytes_per_block=0)
    changed[3]['carveout_config'] = -1
    changed[2]['partitioned_gc_config'] = 'OFF'; first = calculate(changed)
    changed[2]['partitioned_gc_config'] = 'ON_STRICT'; second = calculate(changed)
    cases.append(dict(case='cc52_partitioned_gc_is_relevant', first=first, second=second,
                      passed=first['active_blocks_per_sm'] != second['active_blocks_per_sm']))
    return dict(evidence_kind='cpu_fixture', target_model_evidence=False, reference=reference,
                cases=cases, all_passed=all(case['passed'] for case in cases))


def audit(sqlite_paths, device_attributes, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    calculator = build_calculator(output/'build', header=DEFAULT_HEADER)
    devices = json.loads(Path(device_attributes).read_text())['query']['devices']
    if len(devices) != 1 or devices[0].get('hardware') is None:
        raise ValueError('One explicit retained hardware metadata declaration required')
    hardware = devices[0]['hardware']
    traces = []
    for path in sqlite_paths:
        read = read_sqlite_launches(path, hardware)
        if not read['known']:
            traces.append(read)
            continue
        groups = {}
        for row in read['rows']:
            # All occupancy-relevant observations and successful launch status
            # agree inside a group; retain actual SQLite member row IDs.
            signature = dict(kernel={k:v for k,v in row['kernel'].items() if k != 'static_evidence'},
                mode=row['function_attributes']['shmem_limit_config'],
                effective_shared_bytes=row['effective_shared_bytes'],
                completed_launch=bool(row.get('completed_launch_evidence')))
            key = json.dumps(signature, sort_keys=True)
            if key not in groups:
                groups[key] = dict(representative=row, member_sqlite_rowids=[])
            groups[key]['member_sqlite_rowids'].append(row['launch_identity']['sqlite_rowid'])
        known = 0
        summaries = []
        for group in groups.values():
            row = group['representative']
            resolution = resolve_declaration(row['kernel'], hardware, row['function_attributes'],
                row['device_state'], calculator=calculator,
                effective_shared_bytes=row['effective_shared_bytes'],
                effective_shared_evidence=row['effective_shared_evidence'],
                completed_launch_evidence=row.get('completed_launch_evidence'))
            count = len(group['member_sqlite_rowids'])
            known += count if resolution['known'] else 0
            summaries.append(dict(member_count=count, member_sqlite_rowids=group['member_sqlite_rowids'],
                                  representative_launch=row['launch_identity'], resolution=resolution))
        after = receipt(path)
        if after != read['source']:
            raise ValueError('Source SQLite changed during read-only audit')
        traces.append(dict(source=read['source'], source_after=after, unchanged=True,
            uuid_binding=read['uuid_binding'], device=read['device'], launch_count=len(read['rows']),
            known_launch_count=known, unknown_launch_count=len(read['rows'])-known,
            distinct_declaration_groups=len(groups), groups=summaries))
    report = dict(schema='gsdeploy.arch_aware_occupancy_declaration_audit.v2',
        hardware_source=receipt(device_attributes), hardware=hardware,
        implementation_sources=[receipt(__file__), receipt(Path(__file__).with_name('occupancy_declaration_v2.py'))],
        build=calculator, sensitivity=branch_sensitivity(calculator), traces=traces,
        GPU_workload_executed=False, profiling_counters_requested=False,
        physically_validated=False, deployment_qualified=False,
        interpretation='Static ceilings observationally equivalent for these exact completed launches; real function maxima and requested carveout remain unmeasured; smoke is not baseline training evidence.')
    path = output/'audit.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'output-manifest.json').write_text(json.dumps(dict(outputs=[receipt(path)]),indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sqlite', action='append', required=True, type=Path)
    parser.add_argument('--device-attributes', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.sqlite, args.device_attributes, args.output)
    print(json.dumps(dict(sensitivity_passed=report['sensitivity']['all_passed'], traces=[
        {key:trace[key] for key in ('launch_count','known_launch_count','unknown_launch_count','distinct_declaration_groups')}
        if 'launch_count' in trace else trace for trace in report['traces']], GPU_workload_executed=False)))


if __name__ == '__main__':
    main()
