"""Describe matched shared-pressure latency/energy; finite probes do not certify U."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from gsdeploy.energy import integrate_power


def metrics(values):
    values=sorted(values)
    return dict(n=len(values),mean=statistics.mean(values),p95=values[min(len(values)-1,round(.95*(len(values)-1)))],maximum=max(values))


def summarize(directory,deadlines):
    bundle=load_bundle(directory);raw=bundle['raw'];plan=bundle['plan']
    if (raw['status']!='completed' or raw['split']!='development_shared_pressure'
            or raw['execution_context_sha256']!=plan['execution_context_sha256']
            or raw['resident_aliases']!=plan['aliases']
            or any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors'))
            or any(s.get('foreign_pids') for s in raw['board_samples'])
            or any(s['forced'] or s['exit_code']!=0 or not s['bye_received'] for s in raw['shutdown'].values())):
        raise ValueError('Complete clean shared-pressure capture required')
    rows={r['request_id']:r for r in raw['request_records']};seen=set();batches=[]
    if len(rows)!=len(raw['request_records']) or len(raw['batches'])!=len(plan['jobs']):raise ValueError('Probe roster mismatch')
    for job,batch in zip(plan['jobs'],raw['batches']):
        if job['id']!=batch['id']:raise ValueError('Probe batch identity mismatch')
        observed=[rows[rid] for rid in batch['request_ids']]
        if seen&set(batch['request_ids']):raise ValueError('Repeated probe request')
        seen.update(batch['request_ids'])
        if (Counter((r['alias'],r['task'],r['pool_id']) for r in job['requests'])!=
                Counter((r['model'],r['task'],r['pool_id']) for r in observed)
                or any(r['status']!='completed' for r in observed)):
            raise ValueError('Dropped or changed shared-pressure requests')
        if batch['arrival_end_ns']-batch['start_ns']!=round(job['duration_ms']*1e6):raise ValueError('Arrival window changed')
        for offer,row in zip(sorted(job['requests'],key=lambda r:r['offset_ms']),observed):
            if row['scheduled_time_ns']!=batch['start_ns']+round(offer['offset_ms']*1e6):raise ValueError('Arrival offsets changed')
        tasks={};owners={}
        for task in sorted({r['task'] for r in observed}):
            selected=[r for r in observed if r['task']==task]
            e2e=[(r['completion_time_ns']-r['scheduled_time_ns'])/1e6 for r in selected]
            wait=[(r['worker_response']['start_perf_ns']-r['scheduled_time_ns'])/1e6 for r in selected]
            tasks[task]=dict(e2e_ms=metrics(e2e),queue_to_handler_ms=metrics(wait),
                observed_deadline_miss_fraction=sum(t>deadlines[task] for t in e2e)/len(e2e),
                completed_in_arrival_window=sum(r['completion_time_ns']<=batch['arrival_end_ns'] for r in selected),
                outstanding_at_arrival_end=sum(r['scheduled_time_ns']<=batch['arrival_end_ns']<r['completion_time_ns'] for r in selected),
                observed_routes=dict(Counter(r['model'] for r in selected)))
        for alias in sorted(plan['aliases']):
            selected=[r for r in observed if r['model']==alias]
            if not selected:continue
            occupied=sum(max(0,min(r['worker_response']['end_perf_ns'],batch['arrival_end_ns'])-
                                  max(r['worker_response']['start_perf_ns'],batch['start_ns'])) for r in selected)
            owners[alias]=dict(handler_busy_fraction_in_arrival_window=occupied/(batch['arrival_end_ns']-batch['start_ns']),
                              handler_ms=metrics([r['worker_response']['worker_latency_ms'] for r in selected]))
        joules=integrate_power(raw['board_samples'],batch['start_ns'],batch['end_ns'],max_gap_seconds=.25)
        if joules is None:raise ValueError('Uncovered probe board energy')
        samples=[s for s in raw['board_samples'] if batch['start_ns']<=s['time_ns']<=batch['end_ns']]
        batches.append(dict(id=job['id'],rates=job['rates'],asr_generalist_share=job['asr_generalist_share'],
            shared_trace_sha256=job['shared_trace_sha256'],task_metrics=tasks,owner_metrics=owners,
            arrival_through_drain_joules=joules,drain_seconds=(batch['end_ns']-batch['arrival_end_ns'])/1e9,
            observed_peak_owned_mib=max(s['owned_used_mib'] for s in samples)))
    if seen!=set(rows):raise ValueError('Unassigned probe records')
    return dict(source=bundle['source'],seed=plan['seed'],deployment=plan['resident_domain'],batches=batches)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();reference=json.loads((args.reference/'system-frozen.json').read_text())
    runs=[summarize(args.base/name,reference['latency_reference_ms']) for name in
          ('specialists-round1','hybrid-round1','hybrid-round2','specialists-round2')]
    baseline={b['shared_trace_sha256']:b for r in runs if 'generalist_2' not in r['deployment'] for b in r['batches']}
    pairs=[]
    for run in runs:
        if 'generalist_2' not in run['deployment']:continue
        for batch in run['batches']:
            old=baseline[batch['shared_trace_sha256']]
            pairs.append(dict(hybrid_source=run['source'],shared_trace_sha256=batch['shared_trace_sha256'],
                asr_generalist_share=batch['asr_generalist_share'],
                delta_arrival_through_drain_joules=batch['arrival_through_drain_joules']-old['arrival_through_drain_joules'],
                delta_mean_e2e_ms={t:batch['task_metrics'][t]['e2e_ms']['mean']-old['task_metrics'][t]['e2e_ms']['mean'] for t in batch['task_metrics']}))
    report=dict(schema='gsdeploy.shared_pressure_probe_description.v1',runs=runs,matched_pairs=pairs,
                utility_qualified=False,capacity_qualified=False,normalization_frozen=False,U=None,
                limitations=['45 second development probes; not a steady-state or rho certificate',
                             'per-batch energy includes drain and preceding thermal/clock history; startup is not allocated',
                             'same offered input/arrival cohort does not equal same model work or quality',
                             'hybrid ASR-share sweep keeps its resident model set fixed; activation benefit requires complete episodes'])
    with args.output.open('x') as out:json.dump(report,out,ensure_ascii=False,indent=2);out.write('\n')
    print(json.dumps(dict(output=str(args.output),runs=len(runs),matched_pairs=len(pairs))))


if __name__=='__main__':main()
