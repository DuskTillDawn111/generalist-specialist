"""Training means for startup energy and premeasurement owned sampled peak.

Startup energy excludes priming; prephase peak includes it. No observed drain
energy or independent-validation outcomes may be substituted for either target.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from gsdeploy.resources import identity
from .empirical_resources import integrate_samples
from .native_collect import receipt, save
from .smoke_verifier_v3 import check_receipt


def targets(raw):
    if (raw.get('schema')!='gsdeploy.startup_native_phase_collection.v4'
            or raw.get('kind')!='measured' or raw.get('backend')!='native'
            or raw.get('status')!='completed'
            or any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors'))):
        raise ValueError('Error-free completed startup-aware native training raw required')
    events=raw.get('lifecycle_boundaries',[])
    names=[r.get('boundary') for r in events]
    expected=['startup_begin','startup_end','priming_begin','priming_end','active_begin',
              'active_end','shutdown_begin','shutdown_end']
    if names!=expected or any(r.get('clock')!='host_perf_counter_ns' for r in events):
        raise ValueError('Exact ordered startup/priming/active/shutdown boundaries required')
    times=[r.get('time_ns') for r in events]
    if any(type(t) is not int or t<0 for t in times) or times!=sorted(times):
        raise ValueError('Monotonic host lifecycle boundaries required')
    boundaries=dict(zip(names,times));a,b,c=(boundaries[k] for k in ('startup_begin','startup_end','active_begin'))
    if not a<b<=c:raise ValueError('Positive startup and bounded premeasurement span required')
    samples=raw.get('board_samples',[]);stamps=[r.get('time_ns') for r in samples]
    if (len(samples)<2 or any(type(t) is not int for t in stamps) or stamps!=sorted(set(stamps))
            or stamps[0]>a or stamps[-1]<c):
        raise ValueError('Telemetry must bracket startup through premeasurement; no extrapolation')
    if max(y-x for x,y in zip(stamps,stamps[1:]))>2_000_000_000:
        raise ValueError('Telemetry exceeds the unchanged two-second gap limit')
    for sample in samples:
        if sample.get('foreign_pids'):raise ValueError('Foreign GPU activity invalidates startup evidence')
        for field in ('watts','owned_used_mib'):
            value=sample.get(field)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError('Measured finite power and owned sampled memory required')
        processes={int(pid):value for pid,value in sample.get('processes_mib',{}).items()}
        owned={int(pid):alias for pid,alias in sample.get('owned_pid_aliases',{}).items()}
        if (set(owned)-set(processes) or set(processes)-set(owned)
                or set(owned.values())!=set(sample.get('owned_aliases',[]))
                or abs(sum(processes[pid] for pid in owned)-sample['owned_used_mib'])>1e-6):
            raise ValueError('Owned PID attribution and measured memory sum differ')
    peak=max(s['owned_used_mib'] for s in samples if a<=s['time_ns']<=c)
    energy=integrate_samples(samples,a,b)/1e9
    return dict(energy_joules=energy,peak_memory_mib=peak,startup_seconds=(b-a)/1e9,
        energy_window_ns=dict(start=a,end=b),peak_window_ns=dict(start=a,end=c),
        safety_factor_applied=False,peak_is_sampled=True)


def fit(review_paths,output):
    observations=[];sources=[];seen=set();contexts=set();priming_identities=set()
    for path in review_paths:
        review=json.loads(Path(path).read_text())
        if (review.get('schema') not in ('gsdeploy.startup_training_collection_strict_review.v4',
                'gsdeploy.host_startup_collection_strict_review.v4') or review.get('status')!='PASS'):
            raise ValueError('Passing startup-aware strict training review required')
        for key in ('raw','plan','context'):
            if not check_receipt(review.get(key)):raise ValueError('Changed strict review '+key)
        for ref in review.get('evaluator_sources',[]):
            if not check_receipt(ref):raise ValueError('Changed review evaluator source')
        plan=json.loads(Path(review['plan']['path']).read_text())
        if plan.get('split')!='training' or not plan.get('jobs') or any(j.get('split')!='training' for j in plan['jobs']):
            raise ValueError('Explicit exclusively training plan required')
        raw=json.loads(Path(review['raw']['path']).read_text());sha=review['raw']['sha256']
        if sha in seen:raise ValueError('Duplicate raw is not an independent startup')
        seen.add(sha);contexts.add(plan['execution_context_sha256'])
        if len(contexts)!=1 or raw.get('execution_context_sha256')!=plan['execution_context_sha256']:
            raise ValueError('Startup training execution contexts differ')
        priming=raw.get('priming_request_records',[])
        if not priming or any(r.get('status')!='completed' for r in priming):
            raise ValueError('Complete actual premeasurement priming cohort required')
        priming_identities.add(identity([{k:r[k] for k in ('model','task','pool_id')} for r in priming]))
        if len(priming_identities)!=1:
            raise ValueError('Prephase peaks require the same actual priming configuration')
        observations.append(dict(run_id='raw-sha256:'+sha,raw=review['raw'],**targets(raw)))
        sources.extend([receipt(path),review['raw'],review['plan'],review['context']])
    if not observations:raise ValueError('Startup training captures required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    known=len(observations)>=2
    result=dict(schema='gsdeploy.startup_prediction_fit.v4',status='FITTED' if known else 'UNKNOWN',
        issues=[] if known else ['AT_LEAST_TWO_INDEPENDENT_STARTUP_CAPTURES_REQUIRED'],
        execution_context_sha256=next(iter(contexts)),independent_capture_count=len(observations),
        priming_configuration_sha256=next(iter(priming_identities)),
        observations=observations,sources=sources+[receipt(__file__),receipt(Path(__file__).with_name('empirical_resources.py'))],
        startup_prediction=({k:statistics.mean(r[k] for r in observations) for k in ('energy_joules','peak_memory_mib')} if known else None),
        startup_amortization_seconds=3600,safety_factor_applied=False,
        claim='Training mean startup energy and prephase owned sampled peak; no continuous peak, confidence bound, or held-out accuracy claim.',
        GPU_started=False,validation_started=False,physical_accuracy_validated=False)
    save(output/'fit.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews',nargs='+',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=fit(args.reviews,args.output)
    print(json.dumps(dict(status=result['status'],issues=result['issues'],output=str(Path(args.output).resolve()))))


if __name__=='__main__':main()
