"""Measure intended phase overlap; retain every missed training attempt."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.activation_completion_bands import completion_band_layout
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.native_training import digest, validate_capture


def interval_attainment(intervals):
    if len(intervals)<2 or any(type(a) is not int or type(b) is not int or a>=b for a,b in intervals):
        raise ValueError('At least two positive observed integer-clock intervals required')
    left=max(a for a,b in intervals);right=min(b for a,b in intervals)
    return dict(attained=right>left,overlap_ns=max(0,right-left),nonoverlap_gap_ns=max(0,left-right),
                latest_start_ns=left,earliest_end_ns=right)


def describe(bundle, template):
    if validate_capture(bundle)!=template['execution_context_sha256']:
        raise ValueError('Training plan and pinned baseline context differ')
    raw=bundle['raw'];profiles={p['id']:p for p in template['profiles']}
    batches={b['id']:b for b in raw['batches']};by_job=defaultdict(list);attempts=[]
    for row in raw['request_records']:by_job[row['profile_job_id']].append(row)
    for job in bundle['plan']['jobs']:
        intent=job.get('intended_phase_alignment')
        if intent is None:continue
        if intent['actual_overlap_guaranteed'] is not False or intent['future_observations_used'] is not False:
            raise ValueError('Explicit prospective-only alignment intent required')
        rows=by_job[job['id']];batch=batches[job['id']]
        expected={(m['alias'],m['task'],m['pool_id']) for m in intent['members']}
        if len(expected)!=len(rows) or {(r['model'],r['task'],r['pool_id']) for r in rows}!=expected:
            raise ValueError('All attempted requests must remain in the observed denominator')
        lookup={(r['model'],r['task'],r['pool_id']):r for r in rows};members=[];intervals=[]
        requests={(r['alias'],r['task'],r['pool_id']):r for r in job['requests']}
        for m in intent['members']:
            key=m['alias'],m['task'],m['pool_id'];row=lookup[key];profile=profiles[m['profile_id']]
            if (profile['family'],profile['task'],profile['pool_id'])!=(m['family'],m['task'],m['pool_id']):
                raise ValueError('Intended phase template input identity differs')
            scheduled=batch['start_ns']+round(requests[key]['offset_ms']*1e6)
            if row['scheduled_time_ns']!=scheduled:raise ValueError('Planned offset and actual offered clock differ')
            targets=[s for s in completion_band_layout(row,m['family']) if s.get('native_band_role')==m['role']]
            if len(targets)!=1 or targets[0]['id']!=m['training_band']['segment_id']:
                raise ValueError('Intended phase differs from complete observed layout')
            target=targets[0];intervals.append((target['start_ns'],target['end_ns']))
            members.append(dict(alias=m['alias'],family=m['family'],task=m['task'],pool_id=m['pool_id'],
                role=m['role'],request_id=row['request_id'],observed_start_ns=target['start_ns'],
                observed_end_ns=target['end_ns'],planned_offer_offset_ms=requests[key]['offset_ms'],
                observed_start_minus_planned_start_ms=(target['start_ns']-scheduled)/1e6-m['training_band']['start_ms'],
                observed_band_ms=(target['end_ns']-target['start_ns'])/1e6))
        attempts.append(dict(job_id=job['id'],members=members,**interval_attainment(intervals)))
    if not attempts:raise ValueError('No prospective phase alignment attempts in capture')
    return dict(run_id=raw['run_id'],source=bundle['source'],attempts=attempts,
                attempted_conditions=len(attempts),attained_conditions=sum(a['attained'] for a in attempts))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();base=a.base.resolve()
    design=json.loads((base/'design.json').read_text());template=json.loads((base/'template.json').read_text())
    if digest(template)!=design['training_template_sha256']:raise ValueError('Frozen planning template changed')
    captures=json.loads((base/'queue.json').read_text())['captures'];runs=[];plans=[]
    for capture in captures:
        directory=Path(capture);bundle=load_bundle(directory)
        if bundle['plan']['baseline_sources']!=design['baseline_sources']:raise ValueError('Training baselines differ')
        run=describe(bundle,template)
        if run['attempted_conditions']!=design['phase_conditions']:raise ValueError('Incomplete intended phase cohort')
        runs.append(run);plans.append(dict(path=str(directory/'plan.json'),sha256=hashlib.sha256((directory/'plan.json').read_bytes()).hexdigest()))
    by_condition=defaultdict(list)
    for run in runs:
        for attempt in run['attempts']:
            by_condition[attempt['job_id']].append(dict(run_id=run['run_id'],attained=attempt['attained'],
                overlap_ns=attempt['overlap_ns'],nonoverlap_gap_ns=attempt['nonoverlap_gap_ns']))
    if any(len(rows)!=len(runs) for rows in by_condition.values()):raise ValueError('Cross-run intended cohorts differ')
    report=dict(schema='gsdeploy.phase_alignment_attainment.v1',runs=runs,plans=plans,
        conditions=[dict(job_id=key,independent_attained_runs=sum(r['attained'] for r in rows),observations=rows)
                    for key,rows in sorted(by_condition.items())],
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='training-only intended-versus-realized alignment; all attempts retained; no fit or qualification',
        physical_predictor_qualified=False,formal_utility_qualified=False)
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps([dict(run_id=r['run_id'],attempted=r['attempted_conditions'],attained=r['attained_conditions']) for r in runs]))
