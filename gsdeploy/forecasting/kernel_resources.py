"""CPU-only static residency constraints and explicitly hypothetical progress.

No device API, profiler, model fit or runtime action is performed here. A known
static occupancy is not a throughput measurement or a deployment certificate.
Roomie's occupancy-duration relation is available only with explicit opt-in.
"""
from __future__ import annotations

import math


HARDWARE_FIELDS = ('warp_size', 'sm_count', 'max_threads_per_block',
                   'max_threads_per_sm', 'max_warps_per_sm', 'max_blocks_per_sm',
                   'registers_per_sm', 'shared_bytes_per_sm')


def _unknown(*reasons, **extra):
    return dict(status='UNKNOWN', known=False, reasons=list(reasons),
                physically_validated=False, deployment_qualified=False, **extra)


def _integer(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(name + ' must be a ' + ('nonnegative' if zero else 'positive') + ' integer')
    return value


def _number(value, name, *, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < (0 if zero else 1e-300):
        raise ValueError(name + ' must be finite and ' + ('nonnegative' if zero else 'positive'))
    return value


def _geometry(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(name + ' must contain exactly three dimensions')
    return math.prod(_integer(x, name) for x in value)


def static_occupancy(kernel, hardware):
    """Return a per-SM static residency ceiling, never achieved occupancy.

    Required kernel fields: hardware_id, static_evidence, block[3], grid[3],
    registers_per_thread, static_shared_bytes, dynamic_shared_bytes. Exact
    architecture-rounded allocated_registers_per_block/allocated_shared_bytes
    and allocation_evidence are required when the resource is nonzero. They
    must come from an architecture-specific calculator/API, not naive products.
    Even with these fields, unmodeled partition/scheduling constraints mean the
    returned number is a ceiling for the declared constraints, not guaranteed
    attainable residency. Missing evidence stays UNKNOWN; malformed data raises.
    """
    missing = [k for k in (*HARDWARE_FIELDS, 'hardware_id', 'evidence') if hardware.get(k) is None]
    missing += ['kernel.' + k for k in ('hardware_id', 'static_evidence', 'block', 'grid',
                'registers_per_thread', 'static_shared_bytes', 'dynamic_shared_bytes') if kernel.get(k) is None]
    if missing:
        return _unknown('MISSING_STATIC_EVIDENCE', missing_fields=missing)
    for k in HARDWARE_FIELDS:
        _integer(hardware[k], k)
    if not hardware['evidence'] or not kernel['static_evidence']:
        return _unknown('EMPTY_STATIC_EVIDENCE')
    if any(not isinstance(value, str) or not value for value in (hardware['hardware_id'], kernel['hardware_id'])):
        return _unknown('EXPLICIT_HARDWARE_IDENTITY_REQUIRED')
    if kernel['hardware_id'] != hardware['hardware_id']:
        return _unknown('HARDWARE_IDENTITY_MISMATCH')
    if kernel.get('cluster') is not None and _geometry(kernel['cluster'], 'cluster') != 1:
        return _unknown('CLUSTER_RESIDENCY_NOT_MODELED')
    if kernel.get('cooperative_launch', False):
        return _unknown('COOPERATIVE_LAUNCH_NOT_MODELED')
    threads = _geometry(kernel['block'], 'block')
    grid = _geometry(kernel['grid'], 'grid')
    registers = _integer(kernel['registers_per_thread'], 'registers_per_thread', zero=True)
    shared = sum(_integer(kernel[k], k, zero=True) for k in ('static_shared_bytes', 'dynamic_shared_bytes'))
    warps = (threads + hardware['warp_size'] - 1) // hardware['warp_size']
    raw_regs = registers * threads
    constraints = dict(block_slots=hardware['max_blocks_per_sm'],
                       threads=hardware['max_threads_per_sm'] // threads,
                       warps=hardware['max_warps_per_sm'] // warps,
                       registers=(hardware['registers_per_sm'] // raw_regs if raw_regs else hardware['max_blocks_per_sm']),
                       shared=(hardware['shared_bytes_per_sm'] // shared if shared else hardware['max_blocks_per_sm']))
    if threads > hardware['max_threads_per_block']:
        constraints['threads_per_block'] = 0
    arithmetic_upper = min(constraints.values())
    alloc_regs = kernel.get('allocated_registers_per_block', 0 if raw_regs == 0 else None)
    alloc_shared = kernel.get('allocated_shared_bytes', 0 if shared == 0 else None)
    if alloc_regs is None or alloc_shared is None or ((raw_regs or shared) and not kernel.get('allocation_evidence')):
        return _unknown('ARCHITECTURE_ALLOCATION_EVIDENCE_REQUIRED',
                        arithmetic_block_upper_bound=arithmetic_upper,
                        arithmetic_constraints=constraints,
                        theoretical_occupancy=None,
                        note='Naive registers/thread times threads omits architecture allocation granularity.')
    _integer(alloc_regs, 'allocated_registers_per_block', zero=True)
    _integer(alloc_shared, 'allocated_shared_bytes', zero=True)
    if alloc_regs < raw_regs or alloc_shared < shared:
        raise ValueError('Allocated resources cannot be below unrounded requirements')
    constraints['registers'] = hardware['registers_per_sm'] // alloc_regs if alloc_regs else hardware['max_blocks_per_sm']
    constraints['shared'] = hardware['shared_bytes_per_sm'] // alloc_shared if alloc_shared else hardware['max_blocks_per_sm']
    # The architecture calculator also checks register subpartitions and
    # launch-specific limits. Rounded totals alone do not preserve that bound.
    architecture_blocks = kernel.get('architecture_blocks_per_sm_ceiling')
    if architecture_blocks is not None:
        _integer(architecture_blocks, 'architecture_blocks_per_sm_ceiling', zero=True)
        if not kernel.get('allocation_evidence'):
            return _unknown('ARCHITECTURE_RESIDENCY_CEILING_EVIDENCE_REQUIRED')
        constraints['architecture_calculator'] = architecture_blocks
    blocks = min(constraints.values())
    return dict(status='STATIC_CEILING' if blocks else 'STATICALLY_INFEASIBLE', known=True,
                blocks_per_sm=blocks, theoretical_occupancy=blocks*warps/hardware['max_warps_per_sm'],
                threads_per_block=threads, warps_per_block=warps, grid_blocks=grid,
                allocated_registers_per_block=alloc_regs, allocated_shared_bytes=alloc_shared,
                constraints=constraints, limiting_resources=[k for k, v in constraints.items() if v == blocks],
                grid_limited_gpu_warp_fraction_upper_bound=min(grid, blocks*hardware['sm_count'])*warps /
                    (hardware['sm_count']*hardware['max_warps_per_sm']),
                hardware_id=hardware['hardware_id'],
                evidence=[hardware['evidence'], kernel['static_evidence'], kernel.get('allocation_evidence')],
                interpretation='Theoretical residency ceiling for explicit constraints; not achieved occupancy, throughput or concurrency guarantee.',
                physically_validated=False, deployment_qualified=False)


def check_resident_blocks(kernels, block_counts, hardware):
    """Necessary per-SM packing constraints for an explicit integer placement.

    This does not select actual CUDA scheduling or prove cross-context kernels
    can share an SM. Caller must supply that execution-capability evidence.
    """
    if set(kernels) != set(block_counts):
        raise ValueError('Kernel and block-count identities must match exactly')
    profiles = {k: static_occupancy(v, hardware) for k, v in kernels.items()}
    if any(not p['known'] for p in profiles.values()):
        return _unknown('STATIC_PACKING_INPUT_UNKNOWN', kernels=profiles)
    for name, count in block_counts.items():
        _integer(count, 'block_count.' + name, zero=True)
    totals = dict(blocks=sum(block_counts.values()), threads=0, warps=0, registers=0, shared=0)
    for name, p in profiles.items():
        n = block_counts[name]
        totals['threads'] += n*p['threads_per_block']
        totals['warps'] += n*p['warps_per_block']
        totals['registers'] += n*p['allocated_registers_per_block']
        totals['shared'] += n*p['allocated_shared_bytes']
    limits = dict(blocks=hardware['max_blocks_per_sm'], threads=hardware['max_threads_per_sm'],
                  warps=hardware['max_warps_per_sm'], registers=hardware['registers_per_sm'],
                  shared=hardware['shared_bytes_per_sm'])
    exceeded = [k for k in totals if totals[k] > limits[k]]
    exceeded += ['kernel_limit.' + k for k, n in block_counts.items() if n > profiles[k]['blocks_per_sm']]
    return dict(status='CONSTRAINTS_SATISFIED' if not exceeded else 'CONSTRAINTS_EXCEEDED', known=True,
                feasible_under_declared_constraints=not exceeded, exceeded=exceeded,
                totals=totals, limits=limits, actual_concurrent_residency_proven=False,
                physically_validated=False, deployment_qualified=False)


def _allocate(active, profiles, hardware, policy):
    order = list(active)
    assigned = {k: 0 for k in order}
    used = dict(blocks=0, threads=0, warps=0, registers=0, shared=0)
    limits = dict(blocks=hardware['max_blocks_per_sm'], threads=hardware['max_threads_per_sm'],
                  warps=hardware['max_warps_per_sm'], registers=hardware['registers_per_sm'],
                  shared=hardware['shared_bytes_per_sm'])
    def add(name):
        p = profiles[name]
        demand = dict(blocks=1, threads=p['threads_per_block'], warps=p['warps_per_block'],
                      registers=p['allocated_registers_per_block'], shared=p['allocated_shared_bytes'])
        if assigned[name] >= p['blocks_per_sm'] or any(used[k]+demand[k] > limits[k] for k in limits):
            return False
        assigned[name] += 1
        for k in used:
            used[k] += demand[k]
        return True
    if policy == 'equal_block_round_robin':
        while True:
            changed = [add(name) for name in order]
            if not any(changed):
                break
    else:  # explicit FCFS at a common notional SM; no preemption of a kernel
        for name in order:
            while add(name):
                pass
    return assigned


def allocate_block_progress(kernels, hardware, *, policy='equal_block_round_robin'):
    """Declared full-grid packing and hypothetical progress for an active set.

    Zero progress is a valid packing outcome; a caller whose event engine
    cannot model a waiting GPU segment must reject that context explicitly.
    This function alone grants no permission to use occupancy as throughput.
    """
    if policy not in ('equal_block_round_robin', 'fcfs_blocks'):
        raise ValueError('Explicit supported block-allocation policy required')
    profiles = {name: static_occupancy(k, hardware) for name, k in kernels.items()}
    if any(not p['known'] for p in profiles.values()):
        return _unknown('STATIC_PACKING_INPUT_UNKNOWN', kernels=profiles)
    if any(not p['blocks_per_sm'] for p in profiles.values()):
        return _unknown('STATICALLY_INFEASIBLE_ACTIVE_KERNEL')
    if any(p['grid_blocks'] < hardware['sm_count']*p['blocks_per_sm'] for p in profiles.values()):
        return _unknown('SMALL_GRID_DISTRIBUTION_NOT_MODELED')
    assigned = _allocate(kernels, profiles, hardware, policy)
    return dict(status='ANALYTICAL_HYPOTHESIS', known=True, blocks_per_sm=assigned,
                assumed_relative_progress={name: assigned[name]/p['blocks_per_sm']
                                           for name, p in profiles.items()},
                physically_validated=False, deployment_qualified=False)


def predict_sequence_alignment(sequences, hardware, *, policy='equal_block_round_robin',
                               allow_inverse_occupancy_assumption=False,
                               concurrency_evidence=None, max_events=100000):
    """One explicit-offset alignment of independent serial kernel sequences.

    Each sequence: id, offset_ns, kernels. Each kernel: static fields plus
    solo_duration_ns, solo_timing_basis='unprofiled_isolated', timing_evidence,
    and optional gap_before_ns (fixed non-GPU predecessor/launch gap).
    Synthetic/unit examples may use basis='synthetic' but are labeled as such.
    This is an unvalidated occupancy-proportional fluid progress model, NOT
    Roomie Algorithm 1, a real scheduler simulator or a hardware throughput law.
    """
    if policy not in ('equal_block_round_robin', 'fcfs_blocks'):
        raise ValueError('Explicit supported block-allocation policy required')
    _integer(max_events, 'max_events')
    if type(allow_inverse_occupancy_assumption) is not bool:
        raise ValueError('Explicit boolean occupancy hypothesis opt-in required')
    if not allow_inverse_occupancy_assumption:
        return _unknown('OCCUPANCY_IS_NOT_THROUGHPUT_OPT_IN_REQUIRED')
    if not concurrency_evidence:
        return _unknown('CROSS_CONTEXT_CONCURRENCY_EVIDENCE_REQUIRED')
    if not sequences or len({s['id'] for s in sequences}) != len(sequences):
        raise ValueError('Nonempty unique sequence identities required')
    parsed = {}; synthetic = False
    for seq in sequences:
        _number(seq['offset_ns'], 'offset_ns', zero=True)
        if not isinstance(seq['id'], str) or not seq['id'] or not seq['kernels']:
            raise ValueError('Nonempty sequence id and kernels required')
        kernels = []
        for k in seq['kernels']:
            p = static_occupancy(k, hardware)
            if not p['known']:
                return _unknown('SEQUENCE_STATIC_INPUT_UNKNOWN', sequence=seq['id'], detail=p)
            if not p['blocks_per_sm']:
                return _unknown('SEQUENCE_STATICALLY_INFEASIBLE', sequence=seq['id'])
            if p['grid_blocks'] < hardware['sm_count']*p['blocks_per_sm']:
                return _unknown('SMALL_GRID_DISTRIBUTION_NOT_MODELED', sequence=seq['id'])
            if k.get('solo_duration_ns') is None or not k.get('timing_evidence'):
                return _unknown('ISOLATED_KERNEL_TIMING_REQUIRED', sequence=seq['id'])
            if k.get('solo_timing_basis') not in ('unprofiled_isolated', 'synthetic'):
                return _unknown('PROFILED_OR_COLOCATED_TIMING_NOT_ISOLATED_BASELINE', sequence=seq['id'])
            synthetic |= k['solo_timing_basis'] == 'synthetic'
            duration = _number(k['solo_duration_ns'], 'solo_duration_ns')
            gap = _number(k.get('gap_before_ns', 0), 'gap_before_ns', zero=True)
            kernels.append((k, p, duration, gap))
        parsed[seq['id']] = dict(kernels=kernels, index=0, remaining=kernels[0][2],
                                 ready=seq['offset_ns']+kernels[0][3], offset=seq['offset_ns'],
                                 baseline=sum(k[2]+k[3] for k in kernels))
    clock = 0.; finishes = {}; transitions = []
    # Stable sequence order is explicit; ties use supplied id order, not wall-clock races.
    for _ in range(max_events):
        if len(finishes) == len(parsed):
            break
        active = [name for name, v in parsed.items() if name not in finishes and v['ready'] <= clock]
        future = [v['ready'] for name, v in parsed.items() if name not in finishes and v['ready'] > clock]
        if not active:
            clock = min(future)
            continue
        profiles = {name: parsed[name]['kernels'][parsed[name]['index']][1] for name in active}
        ordered = sorted(active, key=lambda name: (parsed[name]['ready'], list(parsed).index(name)))
        assigned = _allocate(ordered, profiles, hardware, policy)
        speeds = {name: assigned[name]/profiles[name]['blocks_per_sm'] for name in active}
        end = min([clock+parsed[name]['remaining']/speeds[name] for name in active if speeds[name] > 0] + future)
        delta = end-clock
        if not math.isfinite(delta) or delta <= 0:
            return _unknown('INVALID_SEQUENCE_EVENT')
        transitions.append(dict(start_ns=clock, end_ns=end,
                                kernel_indices={name:parsed[name]['index'] for name in active},
                                blocks_per_sm=assigned, assumed_relative_progress=speeds))
        done = []
        for name in active:
            value = parsed[name]
            # Preserve selected completion even when subtraction leaves a small residual.
            complete = speeds[name] > 0 and clock+value['remaining']/speeds[name] <= end
            value['remaining'] = max(0., value['remaining']-delta*speeds[name])
            if complete:
                done.append(name)
        clock = end
        for name in done:
            v = parsed[name]; v['index'] += 1
            if v['index'] == len(v['kernels']):
                finishes[name] = clock
            else:
                _, _, duration, gap = v['kernels'][v['index']]
                v.update(remaining=duration, ready=clock+gap)
        if len(finishes) == len(parsed):
            break
    else:
        return _unknown('SEQUENCE_EVENT_BUDGET_EXCEEDED_NO_TRUNCATION')
    if len(finishes) != len(parsed):
        return _unknown('SEQUENCE_INCOMPLETE')
    results = {}
    for name, v in parsed.items():
        latency = finishes[name]-v['offset']; base = v['baseline']
        results[name] = dict(finish_ns=finishes[name], latency_ns=latency, solo_latency_ns=base,
                             slowdown_ratio=latency/base,
                             relative_increase_over_solo=(latency-base)/base,
                             fraction_of_colocated_latency=(latency-base)/latency)
    return dict(status='ANALYTICAL_HYPOTHESIS', known=True, policy=policy, sequences=results,
                transitions=transitions, synthetic_inputs=synthetic,
                assumption='Uniform full-grid per-SM packing; relative progress=allocated blocks/solo static ceiling; instantaneous repricing and no DRAM/cache slowdown.',
                concurrency_evidence=concurrency_evidence, roomie_algorithm_1_reproduction=False,
                independently_validated=False, physically_validated=False, deployment_qualified=False)


def complementarity_candidate(config_a, config_b, *, dominance_ratio=1.2, memory_budget_bytes=None):
    """Configuration-bound compute/DRAM heuristic; never inferred from VRAM.

    Each config needs config_id, hardware_id, counter_evidence,
    counter_config_id, compute_fraction and dram_fraction (0..1 measured
    throughput fractions on the same hardware), and counter_metric_signature
    binding the metric definitions, normalization and collection scope. No occupancy substitution is
    accepted. Additive fractions are a screening hypothesis, not interference
    prediction, isolation, scheduler feasibility or a certified pruning bound.
    """
    _number(dominance_ratio, 'dominance_ratio')
    if dominance_ratio <= 1:
        raise ValueError('Dominance ratio must exceed one')
    labels = []
    for c in (config_a, config_b):
        required = ('config_id', 'hardware_id', 'counter_config_id', 'counter_evidence',
                    'compute_fraction', 'dram_fraction', 'counter_metric_signature')
        if any(c.get(k) is None or c.get(k) == '' for k in required):
            return _unknown('CONFIGURATION_BOUND_COMPUTE_AND_DRAM_COUNTERS_REQUIRED')
        if c['counter_config_id'] != c['config_id']:
            return _unknown('COUNTER_CONFIGURATION_MISMATCH')
        if not isinstance(c['counter_metric_signature'], str) or not c['counter_metric_signature']:
            return _unknown('EXPLICIT_COUNTER_METRIC_SIGNATURE_REQUIRED')
        if c.get('compute_basis') != 'measured_compute_throughput' or c.get('dram_basis') != 'measured_dram_throughput':
            return _unknown('OCCUPANCY_OR_MEMORY_CAPACITY_IS_NOT_THROUGHPUT')
        compute = _number(c['compute_fraction'], 'compute_fraction', zero=True)
        dram = _number(c['dram_fraction'], 'dram_fraction', zero=True)
        if max(compute, dram) > 1:
            raise ValueError('Measured fractions must be normalized to [0,1]')
        labels.append('COMPUTE' if compute > dominance_ratio*dram else
                      'DRAM' if dram > dominance_ratio*compute else 'BALANCED')
    if config_a['hardware_id'] != config_b['hardware_id']:
        return _unknown('COUNTER_HARDWARE_MISMATCH')
    if config_a['counter_metric_signature'] != config_b['counter_metric_signature']:
        return _unknown('COUNTER_METRIC_DEFINITIONS_OR_NORMALIZATION_MISMATCH')
    sums = {k: config_a[k]+config_b[k] for k in ('compute_fraction', 'dram_fraction')}
    capacity = dict(status='UNKNOWN', reason='Separate resident-memory sizes/budget not supplied')
    if memory_budget_bytes is not None:
        _integer(memory_budget_bytes, 'memory_budget_bytes')
        if all(c.get('resident_bytes') is not None for c in (config_a, config_b)):
            total = sum(_integer(c['resident_bytes'], 'resident_bytes', zero=True) for c in (config_a, config_b))
            capacity = dict(status='FITS_DECLARED_BYTES' if total <= memory_budget_bytes else 'EXCEEDS_DECLARED_BYTES',
                            resident_bytes_sum=total, memory_budget_bytes=memory_budget_bytes,
                            measured_peak_bound_proven=False)
    complementary = set(labels) == {'COMPUTE', 'DRAM'}
    fits = all(v <= 1 for v in sums.values())
    proposed = complementary and fits and capacity['status'] != 'EXCEEDS_DECLARED_BYTES'
    return dict(status='HEURISTIC_CANDIDATE' if proposed else 'NOT_PREFERRED_BY_HEURISTIC', known=True,
                config_ids=[config_a['config_id'], config_b['config_id']], labels=labels,
                complementary=bool(complementary), summed_throughput_fractions=sums,
                additive_screen_fits=fits, memory_capacity=capacity,
                independent_validation_required=True, candidate_domain_pruning_allowed=False,
                physical_interference_avoided=False, physically_validated=False, deployment_qualified=False,
                note='Unknown static memory remains explicit; fitting additive throughput fractions does not prove joint interference or execution capability.')
