"""Real board memory screen before expensive candidate quality collection.
This is an early rejection gate, not matched-workload cost admission or a
certificate of replica/colocation feasibility. Cached/reserved GPU memory is
not subtracted from observed board occupancy.
"""
import math


def screen_pool_memory(study, profiles):
    low, high = study['admission_requirements']['specialist_generalist_memory_ratio']
    budget = study['gpu_memory_mib']
    factor = study['safe_factor']
    if not (0 < low <= high and budget > 0 and factor >= 1):
        raise ValueError('Invalid study memory contract')
    generals = [c['slot'] for c in study['generalists']]
    specialists = [c['slot'] for c in study['specialists']]
    if len(set(generals + specialists)) != len(generals + specialists):
        raise ValueError('Candidate slots must be unique')
    issues, measured = [], {}
    if not generals or not specialists:
        issues.append({'reason': 'MISSING_GENERALIST_OR_SPECIALIST_POOL'})
    for slot in generals + specialists:
        data = profiles.get(slot, {})
        resident = data.get('resident_after_warmup_mib')
        peak = data.get('peak_board_used_mib')
        valid = (data.get('status') == 'COMPLETE' and data.get('kind') == 'measured'
                 and data.get('scope') == 'standalone_same_model_replicas'
                 and data.get('replicas') == 1 and data.get('telemetry_errors') == []
                 and all(type(v) in (float, int) and math.isfinite(v) and v > 0
                         for v in (resident, peak)))
        if not valid:
            issues.append({'slot': slot, 'reason': 'MISSING_VALID_SINGLE_REPLICA_MEMORY_EVIDENCE'})
            continue
        if peak < resident or max(peak, resident) * factor > budget:
            issues.append({'slot': slot, 'reason': 'INVALID_OR_OVER_BUDGET_BOARD_MEMORY'})
            continue
        measured[slot] = resident
    comparisons = []
    for generalist in generals:
        for specialist in specialists:
            g, s = measured.get(generalist), measured.get(specialist)
            ratio = s / g if g is not None and s is not None else None
            passed = ratio is not None and low <= ratio <= high
            comparisons.append({'generalist': generalist, 'specialist': specialist,
                                'specialist_generalist_ratio': ratio, 'passed': passed})
            if ratio is not None and not passed:
                issues.append({'generalist': generalist, 'specialist': specialist,
                               'reason': 'RESIDENT_MEMORY_RATIO_OUTSIDE_STUDY_RANGE'})
    # The intersection is a planning consequence of measured specialists; it
    # does not assert that a proposed generalist configuration achieves it.
    specialist_values = [measured[s] for s in specialists if s in measured]
    bounds = None
    if specialists and len(specialist_values) == len(specialists):
        bounds = {'minimum_mib': max(specialist_values) / high,
                  'maximum_mib': min(specialist_values) / low}
    passed = not issues and all(c['passed'] for c in comparisons)
    return {'status': 'PRELIMINARY_MEMORY_SCREEN_PASSED' if passed else 'POOL_REJECTED_BY_MEMORY_SCREEN',
            'passed': passed, 'scope': 'early_smoke_memory_rejection_only',
            'measured_resident_mib': measured, 'ratio_range': [low, high],
            'generalist_planning_interval_mib': bounds,
            'comparisons': comparisons, 'issues': issues,
            'admission_granted': False,
            'remaining': 'Full input memory stress, matched-load measurements, quality and whole-plan feasibility remain required.'}
