"""Offline operational stability labels. No latency threshold or fitted predictor."""
from dataclasses import asdict, dataclass
import math


DEFAULT_THRESHOLDS = {
    'completion_ratio_min': .98,
    'drain_unstable_seconds': 30.,
    'drain_boundary_seconds': 5.,
    'backlog_bins': 10,
    'growth_min_requests': 10.,
    'growth_offered_fraction': .02,
    'growth_increasing_fraction': .75,
    'growth_end_fraction': .5,
    'end_backlog_unstable_min_requests': 30,
    'end_backlog_unstable_fraction': .10,
    'end_backlog_boundary_min_requests': 3,
    'end_backlog_boundary_fraction': .01,
    'boundary_growth_min_requests': 3.,
    'boundary_growth_offered_fraction': .005,
}


@dataclass
class StabilityLabel:
    stable: bool
    reason: str
    category: str
    stable_region_latency_eligible: bool
    evidence: dict
    thresholds: dict

    def as_dict(self):
        return asdict(self)


def label_stability(data, thresholds):
    if set(thresholds) != set(DEFAULT_THRESHOLDS):
        raise ValueError('Provide every documented stability threshold explicitly')
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0
           for v in thresholds.values()):
        raise ValueError('Invalid stability threshold')
    n=thresholds['backlog_bins']
    if type(n) is not int or n<4: raise ValueError('backlog_bins must be integer >=4')
    records=data.get('request_records',[]); window=data.get('measurement_window')
    system=data.get('system_metrics') or {}; reasons=[]; boundary=[]; bins=[]
    ratio=system.get('completion_ratio')
    if ratio is None or ratio<thresholds['completion_ratio_min']: reasons.append('completion_ratio_missing_or_low')
    incomplete=sum(r['status'] not in ('completed','error') for r in records)
    errors=sum(r['status']=='error' for key in ('request_records','warmup_request_records','priming_request_records') for r in data.get(key,[]))
    if incomplete: reasons.append('incomplete_requests')
    if errors or data.get('errors'): reasons.append('worker_or_run_error')
    if data.get('status') not in ('completed','completed_with_failures'): reasons.append('run_failed')
    if any(v.get('forced') or v.get('exit_code')!=0 or not v.get('bye_received') for v in data.get('shutdown',{}).values()): reasons.append('unclean_shutdown')
    end_backlog=None; growth=None; drain=None
    if window:
        start,end=window['start_time_ns'],window['end_time_ns']
        if end<=start: raise ValueError('Invalid measurement window')
        # Scheduled cohort backlog includes host queues and worker queues.
        events=sorted([(r['scheduled_time_ns'],1) for r in records]+
                      [(r['completion_time_ns'],-1) for r in records if r.get('completion_time_ns') is not None])
        active=cursor=0; last=start
        for i in range(n):
            left=start+round((end-start)*i/n); edge=start+round((end-start)*(i+1)/n); area=0
            while cursor<len(events) and events[cursor][0]<edge:
                t,delta=events[cursor]; area+=active*(t-last); active+=delta; last=t; cursor+=1
            area+=active*(edge-last); last=edge
            bins.append({'offset_seconds':(edge-start)/1e9,'mean_backlog':area/(edge-left),'backlog_at_end':active})
        end_backlog=active; means=[b['mean_backlog'] for b in bins[n//2:]]; growth=means[-1]-means[0]
        inc=sum(b>a for a,b in zip(means,means[1:]))/(len(means)-1)
        limit=max(thresholds['growth_min_requests'],thresholds['growth_offered_fraction']*len(records))
        if growth>=limit and inc>=thresholds['growth_increasing_fraction'] and active>=limit*thresholds['growth_end_fraction']:
            reasons.append('sustained_backlog_growth')
        if active>=max(thresholds['end_backlog_unstable_min_requests'],thresholds['end_backlog_unstable_fraction']*len(records)):
            reasons.append('large_window_end_backlog')
        if active>=max(thresholds['end_backlog_boundary_min_requests'],thresholds['end_backlog_boundary_fraction']*len(records)):
            boundary.append('window_end_backlog_warning')
        if growth>=max(thresholds['boundary_growth_min_requests'],thresholds['boundary_growth_offered_fraction']*len(records)):
            boundary.append('queue_growth_warning')
        drain=window.get('drain_seconds')
        if drain is None: reasons.append('drain_unavailable')
        elif drain>thresholds['drain_unstable_seconds']: reasons.append('drain_above_unstable_limit')
        elif drain>thresholds['drain_boundary_seconds']: boundary.append('drain_warning')
    else: reasons.append('measurement_unavailable')
    category='unstable' if reasons else 'boundary' if boundary else 'stable'
    return StabilityLabel(not reasons,'; '.join(reasons or boundary or ['complete_within_operational_limits']),category,
        category=='stable',dict(completion_ratio=ratio,incomplete_requests=incomplete,request_errors=errors,
        window_end_backlog=end_backlog,drain_seconds=drain,backlog_bins=bins,second_half_backlog_growth=growth,
        measurement_available=bool(window),backlog_definition='scheduled cohort requests not yet completed, including host queues'),dict(thresholds))
