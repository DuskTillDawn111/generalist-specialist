"""Check actual startup-to-shutdown sample coverage without inventing peaks."""
from __future__ import annotations

from .profiling import RESIDENTS
from .smoke_verifier_v3 import finite,stamp,validate_telemetry

BOUNDARIES=('startup_begin','startup_end','priming_begin','priming_end','active_begin','active_end','shutdown_begin','shutdown_end')


def validate_startup_telemetry(raw):
    issues=[];events=raw.get('lifecycle_boundaries',[]);samples=raw.get('board_samples',[])
    if tuple(e.get('boundary') for e in events)!=BOUNDARIES:return ['EXACT_STARTUP_LIFECYCLE_BOUNDARIES_REQUIRED']
    times=[e.get('time_ns') for e in events]
    if any(not stamp(t) for t in times) or times!=sorted(times) or any(e.get('clock')!='host_perf_counter_ns' for e in events):
        return ['ORDERED_COMMON_HOST_LIFECYCLE_CLOCK_REQUIRED']
    if not samples or samples[0].get('time_ns',times[0]+1)>times[0] or samples[-1].get('time_ns',times[-1]-1)<times[-1]:
        return ['TELEMETRY_DOES_NOT_ENVELOPE_STARTUP_SHUTDOWN']
    previous=-1
    for sample in samples:
        values={int(p):v for p,v in sample.get('processes_mib',{}).items()}
        owners={int(p):a for p,a in sample.get('owned_pid_aliases',{}).items()}
        if (not stamp(sample.get('time_ns')) or sample['time_ns']<previous or not stamp(sample.get('query_duration_ns'))
                or not finite(sample.get('watts'),positive=True) or sample.get('foreign_pids')
                or any(not finite(sample.get(k)) for k in ('board_used_mib','board_total_mib','board_free_mib','owned_used_mib'))
                or not finite(sample.get('board_total_mib'),positive=True)
                or sample['board_used_mib']+sample['board_free_mib']>sample['board_total_mib']+1e-3
                or not set(owners)<=set(values) or set(owners.values())-set(RESIDENTS)
                or any(not finite(v,positive=True) for v in values.values())
                or abs(sum(values[p] for p in owners)-sample['owned_used_mib'])>1e-6
                or set(sample.get('owned_aliases',[]))!=set(owners.values())):
            issues.append('INVALID_STARTUP_OWNED_BOARD_SAMPLE');break
        previous=sample['time_ns']
    # The existing strict active-domain test continues to demand all four
    # READY PIDs throughout the actual observation, but never during loading.
    start,end=raw.get('observation_start_ns'),raw.get('observation_end_ns')
    if not stamp(start) or not stamp(end) or not times[4]<=start<end<=times[5]:
        issues.append('ACTIVE_WINDOW_OUTSIDE_LIFECYCLE_BOUNDARIES')
    else:
        active=[s for s in samples if times[2]<=s['time_ns']<=times[6]]
        issues+=validate_telemetry({**raw,'board_samples':active})
    return issues
