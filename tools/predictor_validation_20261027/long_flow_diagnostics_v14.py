"""Training-only, per-run duration and time-bucket diagnostics; no fitting."""
import argparse
import collections
import json
from pathlib import Path
import statistics

from gsdeploy.metrics import percentile
from .flow_eligibility import native_stability
from .flow_metrics import score_flow
from .flow_profiles import map_flow_collection
from .native_flow import checked, receipt, save
from .probe_validation import read
from .training_campaign_v14 import verify_campaign


def stats(values):
    return {'n':len(values),'mean':statistics.mean(values) if values else None,
        'p95':percentile(values,95) if values else None,'max':max(values) if values else None}


def summarize(raw, plan):
    by_input=collections.defaultdict(list);by_period=collections.defaultdict(list)
    for row in raw['request_records']:
        if row['status']!='completed':raise ValueError('Complete training cohorts required')
        family=plan['aliases'][row['model']]['family'];task=row['task']
        service=(row['worker_response']['end_perf_ns']-row['observed_worker_start_time_ns'])/1e6
        wait=(row['observed_worker_start_time_ns']-row['scheduled_time_ns'])/1e6
        e2e=(row['completion_time_ns']-row['scheduled_time_ns'])/1e6
        tail=(row['completion_time_ns']-row['worker_response']['end_perf_ns'])/1e6
        sample=dict(service_ms=service,wait_to_worker_ms=wait,e2e_ms=e2e,parent_tail_ms=tail)
        by_input[(family,task,row['pool_id'])].append(sample)
        period=min(2,int(3*(row['scheduled_time_ns']-raw['phase_start_ns'])/(raw['deadline_ns']-raw['phase_start_ns'])))
        by_period[(family,task,period)].append(sample)
    def unpack(groups, names):
        return [{**dict(zip(names,key)),**{metric:stats([r[metric] for r in rows]) for metric in rows[0]}}
            for key,rows in sorted(groups.items())]
    return {'by_input':unpack(by_input,('family','task','pool_id')),
        'by_arrival_third':unpack(by_period,('family','task','arrival_third')),
        'bucket_semantics':'0/1/2 = equal thirds of public scheduled arrival window; service may finish during drain',
        'waiting_semantics':'scheduled public arrival through true worker START; includes host, transport and alias queue',
        'population':'Complete offered measurement cohort; no latency trimming or tail replacement'}


def one(campaign_path, run_id, output):
    if Path(output).exists():raise FileExistsError(output)
    campaign=verify_campaign(campaign_path)
    jobs=[j for j in campaign['main_jobs'] if j['run_id']==run_id]
    if len(jobs)!=1:raise ValueError('Exact campaign run identity required')
    job=jobs[0];folder=Path(job['output']);raw_path=folder/'scored-raw.json'
    raw=read(raw_path);native=read(checked(raw['native_raw']))
    if {k:v for k,v in raw.items() if k not in ('native_raw','quality_evidence')}!=native:
        raise ValueError('Scored training raw changed native evidence')
    if raw['plan']!=job['plan']:raise ValueError('Actual training source plan changed')
    plan=read(checked(job['plan']))
    if plan.get('split')!='training' or plan.get('phase')!='predictor_training':
        raise ValueError('Prospectively declared training only')
    mapped=map_flow_collection(raw['native_raw']['path'])
    candidate=next(c for c in plan['candidates'] if c['id']==job['candidate_id'])
    value={'schema':'gsdeploy.long_flow_training_diagnostics.v1','run_id':run_id,
        'campaign_role':job['campaign_role'],'campaign':receipt(campaign_path),'plan':job['plan'],
        'raw':receipt(raw_path),'source':receipt(__file__),'source_receipts':[
            receipt(__file__),receipt(score_flow.__code__.co_filename),receipt(native_stability.__code__.co_filename),
            *mapped['mapping_report']['sources']],
        'mapping_report':mapped['mapping_report'],'seed':plan['seed'],'window_seconds':plan['window_seconds'],
        'drain_seconds':plan['drain_seconds'],'candidate_id':candidate['id'],
        'six_terms':score_flow(raw,plan),
        'stability':native_stability(plan,candidate,raw,read(checked(plan['catalog']))),
        **summarize(raw,plan),'independent_validation':False,'GPU_started':False,'final_gain_eligible':False}
    save(output,value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','run-id','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();v=one(a.campaign,a.run_id,a.output)
    print(json.dumps({'run_id':v['run_id'],'requests':v['six_terms']['offered'],
        'original_stability_passed':v['stability']['original_stability_passed'],'GPU_started':False}))
