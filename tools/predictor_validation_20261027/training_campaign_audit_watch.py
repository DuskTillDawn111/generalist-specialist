"""Finite CPU watcher for exclusive training audits, never GPU or inference retry.

Polls only this prospective training campaign. Existing completed reports are
verified and reused, never overwritten. Progress updates touch one mutable
training field; statistical models, gates and source-bound raws stay unchanged.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile
import time

from .native_flow import checked, receipt, save
from .probe_validation import read
from .training_campaign_v14 import verify_campaign
from .training_run_audit import audit_run
from .long_flow_diagnostics_v14 import one as duration_report
from .training_composition_diagnostics import run as composition_report, sample_stats
from .training_idle_diagnostics import run as idle_report


def _report(path, create, source, *, raw=None):
    if not path.exists():create()
    value=read(path)
    if value.get('source')!=receipt(source):raise ValueError('Existing CPU report source changed')
    if raw is not None and value.get('raw')!=raw:raise ValueError('Existing CPU report raw changed')
    return value


def _update(progress_path,entry):
    progress_path=Path(progress_path)
    for _ in range(10):
        before=progress_path.read_bytes();value=json.loads(before);field=value['training_v14_h600']
        runs=field.setdefault('audited_runs',[]);existing=next((r for r in runs if r['run_id']==entry['run_id']),None)
        if existing is None:runs.append(entry)
        else:existing.update(entry)
        field['completed_main_runs']=len(runs)
        field['audited_measurement_requests']=sum(r['measurement_requests'] for r in runs)
        if progress_path.read_bytes()!=before:continue
        fd,temp=tempfile.mkstemp(prefix='.training-audit-progress-',suffix='.tmp',dir=progress_path.parent)
        with os.fdopen(fd,'w') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        if progress_path.read_bytes()!=before:
            Path(temp).unlink();continue
        os.replace(temp,progress_path);return
    raise RuntimeError('Concurrent progress changes prevented safe one-field update')


def audit_job(campaign_path,job,progress_path):
    root=Path(campaign_path).parent;rid=job['run_id'];folder=Path(job['output'])
    for name in ('run-audits','diagnostics'):(root/name).mkdir(exist_ok=True)
    audit_path=root/'run-audits'/(rid+'.json');duration_path=root/'diagnostics'/(rid+'.json')
    composition_path=root/'diagnostics'/(rid+'-input-composition.json')
    idle_path=root/'diagnostics'/(rid+'-idle-history.json')
    a=_report(audit_path,lambda:audit_run(folder/'raw.json',audit_path),audit_run.__code__.co_filename)
    if a['status']!='PASS' or a['plan']!=job['plan'] or a['candidate_id']!=job['candidate_id']:
        raise ValueError('Audited run differs from prospective campaign job')
    checked(a['raw']);checked(a['scored_raw']);raw=read(a['raw']['path'])
    d=_report(duration_path,lambda:duration_report(campaign_path,rid,duration_path),duration_report.__code__.co_filename)
    if d['run_id']!=rid or d['raw']!=a['scored_raw']:
        raise ValueError('Duration report differs from audited run')
    for ref in d['source_receipts']:checked(ref)
    _report(composition_path,lambda:composition_report(audit_path,composition_path),composition_report.__code__.co_filename,raw=a['raw'])
    idle=_report(idle_path,lambda:idle_report(audit_path,idle_path),idle_report.__code__.co_filename,raw=a['raw'])
    st=d['stability'];se=st['stability_evidence']
    entry=dict(run_id=rid,complete=True,measurement_requests=a['request_count'],quality_count=a['quality']['count'],
        quality_task_counts=a['quality']['task_counts'],postcollection_audit=receipt(audit_path),diagnostics=receipt(duration_path),
        input_composition_diagnostics=receipt(composition_path),idle_history_diagnostics=receipt(idle_path),
        failure_counts={**{k:len(raw[k]) for k in ('errors','runtime_errors','telemetry_errors','qualification_issues')},
            'incomplete_requests':sum(r['status']!='completed' for r in raw['request_records']),
            'quality_annotation_mismatches':0,'strict_source_mapping_failures':0},
        original_stability_passed=st['original_stability_passed'],original_stability_state=st['stability']['stability_state_v2'],
        drain_seconds=se['drain_duration'],final_offered_backlog_after_drain=se['offered_backlog_diagnostic']['final_backlog_after_drain'],
        text_arrival_thirds=[{k:x[k] for k in ('arrival_third','service_ms','wait_to_worker_ms','e2e_ms')}
            for x in d['by_arrival_third'] if x['family']=='text'],
        idle_coverage=idle['coverage'],missing_worker_predecessors=idle['missing_prior_count'])
    _update(progress_path,entry)
    return entry


def repeat_pair(campaign_path,job,output):
    if output.exists():
        value=read(output)
        if value.get('source')!=receipt(__file__):raise ValueError('Existing repeat diagnostic source changed')
        for ref in value['sources']:checked(ref)
        return value
    campaign=verify_campaign(campaign_path);repeat_plan=read(checked(job['plan']))
    primary=next(j for j in campaign['main_jobs'] if j['campaign_role']=='primary_training'
        and j['candidate_id']==job['candidate_id'] and read(checked(j['plan']))['seed']==repeat_plan['seed'])
    primary_plan=read(checked(primary['plan']))
    if primary_plan['trace']!=repeat_plan['trace'] or primary_plan['candidates']!=repeat_plan['candidates']:
        raise ValueError('Repeat comparison requires exact same full inputs/arrivals/routes/resources')
    root=Path(campaign_path).parent;paths=[root/'run-audits'/(j['run_id']+'.json') for j in (primary,job)]
    audits=[read(p) for p in paths]
    for a in audits:
        if a['status']!='PASS':raise ValueError('Both repeated runs require passed source/quality audit')
        checked(a['raw']);checked(a['source'])
    raws=[read(a['raw']['path']) for a in audits]
    rows=[{r['request_id']:r for r in raw['request_records']} for raw in raws]
    cells=defaultdict(list);records=[]
    for request in primary_plan['trace']:
        rid=request['request_id'];pair=[by_id[rid] for by_id in rows]
        if any((r['task'],r['pool_id'],r['model'])!=(request['task'],request['pool_id'],
                next(c for c in primary_plan['candidates'] if c['id']==job['candidate_id'])['routes'][rid]) for r in pair):
            raise ValueError('Pair actual input or alias mismatch')
        values=[]
        for r in pair:
            start=r['observed_worker_start_time_ns'];end=r['worker_response']['end_perf_ns'];arrival=r['scheduled_time_ns']
            values.append(dict(service_ms=(end-start)/1e6,wait_ms=(start-arrival)/1e6,e2e_ms=(r['completion_time_ns']-arrival)/1e6))
        record=dict(request_id=rid,alias=pair[0]['model'],task=request['task'],pool_id=request['pool_id'],
            primary=values[0],repeat=values[1],repeat_minus_primary={k:values[1][k]-values[0][k] for k in values[0]})
        records.append(record);cells[(record['alias'],record['task'],record['pool_id'])].append(record)
    def metrics(rows):
        return {key:dict(primary=sample_stats([r['primary'][key] for r in rows]),
            repeat=sample_stats([r['repeat'][key] for r in rows]),
            paired_repeat_minus_primary=sample_stats([r['repeat_minus_primary'][key] for r in rows]),
            paired_mean_absolute_difference_ms=sum(abs(r['repeat_minus_primary'][key]) for r in rows)/len(rows))
            for key in ('service_ms','wait_ms','e2e_ms')}
    value=dict(schema='gsdeploy.identical_trace_training_repeat_comparison.v1',source=receipt(__file__),
        campaign=receipt(campaign_path),primary_run_id=primary['run_id'],repeat_run_id=job['run_id'],request_count=len(records),
        sources=[receipt(campaign_path),*map(receipt,paths),*[a['raw'] for a in audits],*map(lambda j:j['plan'],(primary,job))],
        exact_full_input_arrival_route_resource_match=True,overall=metrics(records),
        by_alias_task_input=[dict(alias=k[0],task=k[1],pool_id=k[2],n=len(rows),metrics=metrics(rows)) for k,rows in sorted(cells.items())],
        records=records,independent_input_realizations=1,independent_validation=False,model_changed=False,
        GPU_started=False,final_gain_eligible=False,
        interpretation='One descriptive identical-trace cold-run pair; correlated requests, no calibrated confidence interval, no fitting or causal mechanism attribution')
    save(output,value);return value


def watch(campaign_path,progress_path):
    campaign=verify_campaign(campaign_path);root=Path(campaign_path).parent
    for job in campaign['main_jobs']:
        path=Path(job['output'])/'scored-raw.json'
        while True:
            if path.exists():
                try:read(path)
                except json.JSONDecodeError:pass # native writer has not closed its new file yet
                else:break
            time.sleep(60)
        entry=audit_job(campaign_path,job,progress_path)
        print(json.dumps(dict(event='training_run_independently_audited',run_id=job['run_id'],
            requests=entry['measurement_requests'],failure_counts=entry['failure_counts'],
            original_stability_passed=entry['original_stability_passed'],drain_seconds=entry['drain_seconds'])),flush=True)
        for repeated in campaign['main_jobs']:
            if repeated['campaign_role']!='identical_trace_training_repeat':continue
            output=root/'diagnostics'/(repeated['run_id']+'-repeat-pair.json')
            rp=read(checked(repeated['plan']))
            primary=next(j for j in campaign['main_jobs'] if j['campaign_role']=='primary_training'
                and j['candidate_id']==repeated['candidate_id'] and read(checked(j['plan']))['seed']==rp['seed'])
            if output.exists() or not all((root/'run-audits'/(j['run_id']+'.json')).exists() for j in (primary,repeated)):
                continue
            pair=repeat_pair(campaign_path,repeated,output)
            print(json.dumps(dict(event='identical_trace_repeat_pair_compared',repeat_run_id=repeated['run_id'],
                requests=pair['request_count'],output=str(output))),flush=True)
    save(root/'completed-independent-training-audits.json',dict(campaign=receipt(campaign_path),source=receipt(__file__),
        completed_ns=time.perf_counter_ns(),runs=len(campaign['main_jobs']),GPU_started=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--progress',required=True)
    a=p.parse_args();watch(a.campaign,a.progress)
