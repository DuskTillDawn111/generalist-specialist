"""Describe real pressure probes; these do not freeze g(n), rho or utility."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gsdeploy.energy import integrate_power
from experiments.fit_activation_native import load_bundle


def percentile(values,q):
    values=sorted(values);return values[min(len(values)-1,round(q*(len(values)-1)))] if values else None


def summarize(directory):
    bundle=load_bundle(directory);raw,plan=bundle['raw'],bundle['plan']
    if (raw.get('schema')!='gsdeploy.native_activity_capture.v1' or raw.get('kind')!='measured'
            or raw.get('backend')!='native' or raw.get('status')!='completed' or raw.get('split')!='development_capacity'
            or plan.get('mode')!='capacity' or raw.get('errors') or raw.get('runtime_errors')
            or raw.get('telemetry_errors') or raw.get('execution_context_sha256')!=plan['execution_context_sha256']
            or raw.get('resident_aliases')!=plan['aliases']
            or any(s.get('foreign_pids') for s in raw['board_samples'])
            or any(s['forced'] or s['exit_code']!=0 or not s['bye_received'] for s in raw['shutdown'].values())):
        raise ValueError('Complete native pressure-probe cohort and clean telemetry/shutdown required')
    rows={r['request_id']:r for r in raw['request_records']};batches=[]
    if len(rows)!=len(raw['request_records']) or len(raw['batches'])!=len(plan['jobs']):
        raise ValueError('Unique complete capacity cohort required')
    used=set();cohorts=[]
    for job,batch in zip(plan['jobs'],raw['batches']):
        if job['id']!=batch['id']:raise ValueError('Capacity batch order differs from plan')
        records=[rows[rid] for rid in batch['request_ids']]
        used.update(batch['request_ids'])
        expected=Counter((r['alias'],r['task'],r['pool_id']) for r in job['requests'])
        actual=Counter((r['model'],r['task'],r['pool_id']) for r in records)
        if expected!=actual or any(r['status']!='completed' for r in records):
            raise ValueError('Capacity probe dropped or changed offered requests')
        start,end=batch['start_ns'],batch['arrival_end_ns']
        offered_end=start+round(job['duration_ms']*1e6)
        if end!=offered_end:raise ValueError('Capacity arrival window changed')
        for offered,row in zip(sorted(job['requests'],key=lambda r:r['offset_ms']),records):
            if row['scheduled_time_ns']!=start+round(offered['offset_ms']*1e6):
                raise ValueError('Capacity arrival timing changed')
        task=max(job['rates'],key=job['rates'].get);by_task=defaultdict(list)
        for r in records:by_task[r['task']].append(r)
        summaries={}
        for name,owned in sorted(by_task.items()):
            handler=[r['worker_response']['worker_latency_ms'] for r in owned]
            wait=[(r['worker_response']['start_perf_ns']-r['scheduled_time_ns'])/1e6 for r in owned]
            latency=[(r['completion_time_ns']-r['scheduled_time_ns'])/1e6 for r in owned]
            completions=sum(r['completion_time_ns']<=end for r in owned)
            at_end=sum(r['scheduled_time_ns']<=end<r['completion_time_ns'] for r in owned)
            summaries[name]=dict(offered=len(owned),completed_in_arrival_window=completions,
                observed_completion_rps=completions/(job['duration_ms']/1000),
                outstanding_at_arrival_end=at_end,mean_handler_ms=statistics.mean(handler),
                mean_queue_to_handler_ms=statistics.mean(wait),p95_queue_to_handler_ms=percentile(wait,.95),
                mean_e2e_ms=statistics.mean(latency),p95_e2e_ms=percentile(latency,.95))
        energy=integrate_power(raw['board_samples'],start,batch['end_ns'],max_gap_seconds=.25)
        if energy is None:raise ValueError('Capacity batch lacks covered board energy')
        cohorts.append([dict(task=r['task'],pool_id=r['pool_id'],offset_ms=r['offset_ms']) for r in job['requests']])
        batches.append(dict(id=job['id'],pressured_task=task,nominal_solo_rho=job['nominal_solo_rho'],
            rates=job['rates'],task_metrics=summaries,arrival_through_drain_joules=energy,
            drain_seconds=(batch['end_ns']-end)/1e9,
            scope='finite development probe; not steady capacity, legal rho evidence or formal U'))
    if used!=set(rows):raise ValueError('Capacity requests absent from batch roster')
    episode_j=integrate_power(raw['board_samples'],raw['startup_begin_time_ns'],raw['observation_end_ns'],max_gap_seconds=.25)
    if episode_j is None:raise ValueError('Capacity startup-through-drain energy lacks coverage')
    in_window=[s for s in raw['board_samples'] if raw['observation_start_ns']<=s['time_ns']<=raw['observation_end_ns']]
    return dict(schema='gsdeploy.capacity_development_summary.v1',source=str(directory/'raw.json'),
        source_sha256=hashlib.sha256((directory/'raw.json').read_bytes()).hexdigest(),
        seed=plan['seed'],resident_aliases=raw['resident_aliases'],batches=batches,
        cohort_sha256=hashlib.sha256(json.dumps(cohorts,sort_keys=True).encode()).hexdigest(),
        startup_through_drain_joules=episode_j,
        observed_peak_owned_mib=max(s['owned_used_mib'] for s in in_window),
        utility_qualified=False,capacity_qualified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    summaries=[summarize(d.resolve()) for d in args.capture]
    groups=defaultdict(list)
    for summary in summaries:groups[summary['cohort_sha256']].append(summary)
    pairs=[]
    for cohort,items in groups.items():
        for a in items:
            for b in items:
                if len(a['resident_aliases'])>=len(b['resident_aliases']):continue
                pairs.append(dict(cohort_sha256=cohort,lower=a['source'],higher=b['source'],
                    delta_startup_through_drain_joules=b['startup_through_drain_joules']-a['startup_through_drain_joules'],
                    delta_observed_peak_owned_mib=b['observed_peak_owned_mib']-a['observed_peak_owned_mib'],
                    scope='matched offered inputs and offsets; duration and execution may differ'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:json.dump(dict(runs=summaries,matched_pairs=pairs),out,ensure_ascii=False,indent=2);out.write('\n')
    print(json.dumps(dict(output=str(args.output),runs=len(summaries),matched_pairs=len(pairs))))


if __name__=='__main__':main()
