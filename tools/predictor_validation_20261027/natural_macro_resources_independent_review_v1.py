"""Independent raw-poll/state arithmetic audit; no candidate routines reused.

This checks the published training evidence, not prospective resource accuracy.
Raw macro boundaries were separately reviewed; their immutable receipts remain
inputs here. Power integration below uses exact rational decimal arithmetic.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def receipt(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(ref):
    require(receipt(ref['path'])==ref, 'Independent source receipt mismatch')
    return json.loads(Path(ref['path']).read_text())


def same(actual, expected, label):
    if expected is None:
        require(actual is None, label+' must remain UNKNOWN')
    elif type(expected) in (int,float):
        require(type(actual) in (int,float) and math.isclose(actual, expected, rel_tol=2e-12,abs_tol=1e-10),label)
    else:
        require(actual==expected,label)


def integral(samples, start, end):
    """Exact integral of linear interpolation, with original 2s gap rule."""
    if not start<end or samples[0]['time_ns']>start or samples[-1]['time_ns']<end:
        return None
    value=F(0)
    for a,b in zip(samples,samples[1:]):
        x,y=a['time_ns'],b['time_ns'];lo,hi=max(start,x),min(end,y)
        if lo>=hi:
            continue
        if not 0<y-x<=2_000_000_000:
            return None
        pa,pb=F(str(a['watts'])),F(str(b['watts']))
        value+=(pa*(hi-lo)+(pb-pa)*F((hi-x)**2-(lo-x)**2,2*(y-x)))/1_000_000_000
    return float(value)


def audit_table(table):
    raw=read(table['raw']);macros=read(table['training_macros'])
    read(table['context'])
    samples=raw['board_samples'];left,right=raw['observation_start_ns'],raw['observation_end_ns']
    roster=table['resident_owners'];requests={r['request_id']:r for r in raw['request_records']}
    require(len(requests)==len(macros['requests'])==50,'Complete 50-request capture required')
    spans=[];cuts={left,right}
    for request in macros['requests']:
        row=requests[request['request_id']]
        require(request['macros'][0]['left_ns']==row['observed_worker_start_time_ns']
            and request['macros'][-1]['right_ns']==row['previous_handler_next_request_ready']['perf_ns'],
            'Actual START-to-READY coverage differs')
        for macro in request['macros']:
            a,b=macro['left_ns'],macro['right_ns'];cuts.update((a,b))
            spans.append((a,b,dict(owner=row['model'],task=row['task'],pool_id=row['pool_id'],
                response_phase=macro['semantic_phase'])))
    ordered=sorted(cuts);require(len(table['slices'])==len(ordered)-1,'Global partition differs')
    for index,(a,b) in enumerate(zip(ordered,ordered[1:])):
        s=table['slices'][index]
        members=sorted([m for x,y,m in spans if x<=a and b<=y],key=lambda m:m['owner'])
        state=hashlib.sha256(json.dumps(dict(resident_owners=sorted(roster),
            actual_START_to_READY_handlers=members),sort_keys=True).encode()).hexdigest()
        for field,value in dict(left_ns=a,right_ns=b,duration_ns=b-a,members=members,state_key=state).items():
            same(s[field],value,'Raw macro partition '+field)
        same(s['observed_piecewise_linear_board_joules'],integral(samples,a,b),'Per-slice observation integral')
    require(len(samples)==len(table['polls']),'Every raw poll must remain')
    assignments=[]
    for i,(sample,poll) in enumerate(zip(samples,table['polls'])):
        at=sample['time_ns'];duration=sample['query_duration_ns'];begin=at-duration//2;end=begin+duration
        matches=[j for j,s in enumerate(table['slices']) if s['left_ns']<=begin and end<s['right_ns']]
        require(len(matches)<=1,'Poll belongs to multiple global states')
        index=matches[0] if matches else None;assignments.append(index)
        for field,value in dict(raw_poll_ordinal=i,midpoint_ns=at,poll_begin_ns=begin,poll_end_ns=end,
                watts=sample['watts'],owned_used_mib=sample['owned_used_mib'],state_slice_ordinal=index,
                state_key=table['slices'][index]['state_key'] if index is not None else None).items():
            same(poll[field],value,'Raw poll reconstruction '+field)
    for index,s in enumerate(table['slices']):
        indices=[i for i,j in enumerate(assignments) if j==index]
        pairs=[(i,i+1) for i in range(len(samples)-1) if assignments[i]==index==assignments[i+1]
            and samples[i+1]['time_ns']-samples[i]['time_ns']<=2_000_000_000]
        duration=sum(samples[j]['time_ns']-samples[i]['time_ns'] for i,j in pairs)
        energy=sum((F(str(samples[i]['watts']))+F(str(samples[j]['watts'])))/2*
            F(samples[j]['time_ns']-samples[i]['time_ns'],1_000_000_000) for i,j in pairs)
        same(s['interior_poll_ordinals'],indices,'Interior poll cohort')
        same(s['power_pair_ordinals'],[list(p) for p in pairs],'Adjacent wholly interior power pairs')
        same(s['power_training_duration_ns'],duration,'Power-pair duration')
        same(s['power_training_joules'],float(energy),'Power-pair energy')
        same(s['sampled_owned_peak_mib'],max((samples[i]['owned_used_mib'] for i in indices),default=None),'State sampled peak')
    bounds={r['boundary']:r['time_ns'] for r in raw['lifecycle_boundaries']};startup=bounds['startup_begin']
    require(right==bounds['measurement_and_drain_end'] and samples[-1]['time_ns']<=bounds['monitoring_end']
        <bounds['shutdown_begin'],'Original observation/monitor/shutdown scope')
    same(table['startup_joules'],integral(samples,startup,left),'Whole startup integral')
    same(table['observation_joules'],integral(samples,left,right),'Whole observation integral')
    same(sum(s['observed_piecewise_linear_board_joules'] for s in table['slices']),table['observation_joules'],'One board clock conservation')
    same(table['premeasurement_owned_sampled_peak_mib'],max(s['owned_used_mib'] for s in samples if startup<=s['time_ns']<=left),'Premeasurement sampled peak')
    same(table['raw_monitor_owned_sampled_peak_mib'],max(s['owned_used_mib'] for s in samples),'Whole monitor sampled peak')
    same(table['scoring_measurement_energy_joules'],None,'Absent scoring deadline')
    return dict(capture_sha256=table['capture_sha256'],requests=len(requests),macros=len(spans),polls=len(samples),
        global_slices=len(table['slices']),observation_joules=table['observation_joules'],
        startup_joules=table['startup_joules'],all_polls_reconstructed=True,raw_macro_partition_reconstructed=True,
        independent_rational_integrals_reconstructed=True)


def audit_fit(model,tables):
    groups=defaultdict(lambda:defaultdict(list))
    for table in tables:
        for s in table['slices']:
            groups[s['state_key']][table['capture_sha256']].append(s)
    require(set(groups)==set(model['states']),'Fit retains every exact full state')
    for key,captures in groups.items():
        row=model['states'][key];powers=[];peaks=[]
        require(set(captures)==set(row['per_capture']),'State capture support differs')
        for capture,slices in captures.items():
            duration=sum(s['power_training_duration_ns'] for s in slices)
            joules=math.fsum(s['power_training_joules'] for s in slices)
            peak=max((s['sampled_owned_peak_mib'] for s in slices if s['sampled_owned_peak_mib'] is not None),default=None)
            if duration:powers.append(joules*1e9/duration)
            if peak is not None:peaks.append(peak)
            for field,value in dict(power_duration_ns=duration,power_joules=joules,owned_peak_mib=peak).items():
                same(row['per_capture'][capture][field],value,'Per-capture resource '+field)
        same(row['power_capture_count'],len(powers),'Independent power capture count')
        same(row['memory_capture_count'],len(peaks),'Independent memory capture count')
        same(row['power_known'],len(powers)>=2,'Power support threshold')
        same(row['memory_known'],len(peaks)>=2,'Memory support threshold')
        for field,values in [('watts',powers),('peak_memory_mib',peaks)]:
            same(row['mean'][field],statistics.mean(values) if len(values)>=2 else None,'Equal capture resource mean')
            same(row['max'][field],max(values) if len(values)>=2 else None,'Empirical resource maximum')
    same(model['startup']['joules'],statistics.mean(t['startup_joules'] for t in tables),'Unamortized startup mean')
    same(model['startup']['premeasurement_owned_sampled_peak_mib'],statistics.mean(t['premeasurement_owned_sampled_peak_mib'] for t in tables),'Startup sampled-peak mean')


def run(report_path):
    source=receipt(report_path);report=read(source);model=read(report['model'])
    require(report['validation_labels_read'] is False and report['GPU_started'] is False,'Training-only source required')
    sources=[source,report['model'],report['diagnostics'],*report['tables'],*report['sources']]
    unique={s['path']:s for s in sources}
    for ref in unique.values():require(receipt(ref['path'])==ref,'Training closure changed')
    tables=[read(ref) for ref in report['tables']];checks=[audit_table(t) for t in tables]
    audit_fit(model,tables)
    folds=read(report['diagnostics'])
    require(len(folds)==len(tables)==3,'Complete three-capture leaveout required')
    for fold,table in zip(folds,tables):
        train=[t for t in tables if t is not table]
        require(fold['heldout_capture_sha256']==table['capture_sha256']
            and set(fold['training_captures'])=={t['capture_sha256'] for t in train},'Disjoint complete capture fold')
        supported=defaultdict(set)
        for t in train:
            for s in t['slices']:
                if s['power_training_duration_ns']>0:supported[s['state_key']].add(t['capture_sha256'])
        known=sum(s['duration_ns'] for s in table['slices'] if len(supported[s['state_key']])>=2)
        same(fold['power_known_duration_ns'],known,'Leaveout whole-clock support denominator')
        same(fold['power_unknown_duration_ns'],table['observation_duration_ns']-known,'Leaveout unknown duration retained')
        same(fold['observation_energy'],None,'Unsupported whole observation must remain UNKNOWN')
    return dict(schema='gsdeploy.natural_macro_resource_independent_arithmetic_review.v1',status='PASS',
        sources=list(unique.values())+[receipt(__file__)],checks=checks,full_fit_rederived=True,
        complete_leaveout_support_rederived=True,source_files_verified=len(unique),
        resource_prediction_qualified=False,main_queue_provider_eligible=False,final_gain_eligible=False,
        validation_labels_read=False,GPU_started=False,
        limitations=['Audits immutable training arithmetic, not future-state or hardware accuracy.',
            'Sampled peaks are not continuous capacity bounds; missing full states remain UNKNOWN.',
            'Observed macro boundaries are authenticated inputs from a separately reviewed extractor.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args();result=run(args.report)
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],captures=len(result['checks']),source_files=result['source_files_verified'])))
