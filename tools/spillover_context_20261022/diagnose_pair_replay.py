"""Post-hoc component substitution for two registered terminal routing runs.

Observed service substitutions explain replay sensitivity only. They are not
counterfactual GPU measurements, causal identification, or a deployed predictor.
"""
import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import statistics

spec=importlib.util.spec_from_file_location('_pair_fine_routes',Path(__file__).with_name('fine_routes.py'))
fine=importlib.util.module_from_spec(spec);spec.loader.exec_module(fine)


def replay(rows, service, release=None):
    clocks=defaultdict(float); by_task=defaultdict(list); details=[]
    for row in rows:
        arrival=row['arrival_seconds']; ready=release(row) if release else arrival
        duration=service(row)
        if not math.isfinite(duration) or duration<=0 or ready<arrival-1e-8:
            raise ValueError('Invalid service or release time')
        start=max(ready,clocks[row['alias']]); end=start+duration/1000
        clocks[row['alias']]=end; latency=(end-arrival)*1000
        by_task[row['task']].append(latency)
        details.append(dict(index=row['index'],alias=row['alias'],pool_id=row['pool_id'],
            service_ms=duration,latency_ms=latency,queue_ms=(start-ready)*1000))
    return dict(per_task={t:dict(count=len(v),mean_ms=statistics.mean(v),p95_ms=fine.percentile(v,.95))
                         for t,v in by_task.items()},rows=details)


def terms(result,config):
    total=math.fsum(config['workload'].values());refs=config['latency_reference_ms']
    return {term:math.fsum(rate/total*result['per_task'][task][metric]/refs[task]
            for task,rate in config['workload'].items())
            for term,metric in [('L','mean_ms'),('P','p95_ms')]}


def summarize_input(raw,vectors):
    groups=defaultdict(list)
    for row in raw['request_records']:
        if row['task']=='text':groups[(row['model'],row['pool_id'])].append(row)
    result=[]
    for (alias,pool),rows in sorted(groups.items()):
        values=[r['worker_response']['service_ms'] for r in rows]
        historical=vectors[alias][pool]
        result.append(dict(alias=alias,pool_id=pool,count=len(rows),observed_mean_ms=statistics.mean(values),
            observed_median_ms=statistics.median(values),observed_p95_ms=fine.percentile(values,.95),
            historical_joint_ms=historical,observed_mean_over_historical=statistics.mean(values)/historical,
            input_tokens=sorted({r['worker_response']['input_tokens'] for r in rows}),
            generated_tokens=sorted({r['worker_response']['generated_tokens'] for r in rows})))
    return result


def diagnose(comparison_path,scenario_path,output,seed):
    output=Path(output)
    if output.exists():raise FileExistsError('Never overwrite diagnostics')
    comparison=json.loads(Path(comparison_path).read_text());scenario=json.loads(Path(scenario_path).read_text())
    observations=[r for r in comparison['observations'] if r['seed']==seed and r.get('audit_status')=='VERIFIED_TERMINAL_EVIDENCE']
    if len(observations)!=2:raise ValueError('Exactly two audited terminals required')
    observations.sort(key=lambda r:bool(r['plan']['routing']['text'].get('generalist_2')))
    if len({r['cohort_sha256'] for r in observations})!=1:raise ValueError('Different cohorts')
    raws=[];inputs=[fine.exposure.receipt(Path(comparison_path)),fine.exposure.receipt(Path(scenario_path))]
    for observation in observations:
        receipt=observation['raw_receipt'];path=Path(receipt['path'])
        assert path.name=='run.json' and fine.exposure.receipt(path)==receipt
        raw=json.loads(path.read_text());assert all(r['status']=='completed' for r in raw['request_records'])
        raws.append(raw);inputs.append(receipt)
    vector_receipt=scenario['inputs'][1];baseline_receipt=scenario['inputs'][2]
    for receipt in [vector_receipt,baseline_receipt]:
        assert fine.exposure.receipt(Path(receipt['path']))==receipt;inputs.append(receipt)
    vectors=json.loads(Path(vector_receipt['path']).read_text())['service_vectors']
    baselines=json.loads(Path(baseline_receipt['path']).read_text())['catalog']['calibrated_replay_v2']['input_baselines']
    # One shared post-hoc per-input table removes case-specific alias speeds.
    pooled=defaultdict(list)
    for row in raws[0]['request_records']:
        if row['task']=='text':pooled[row['pool_id']].append(row['worker_response']['service_ms'])
    pooled_mean={pool:statistics.mean(v) for pool,v in pooled.items()}
    pooled={pool:statistics.median(v) for pool,v in pooled.items()}
    baseline_exact={i:r['worker_response']['service_ms'] for i,r in enumerate(raws[0]['request_records']) if r['task']=='text'}
    jobs=[]
    for observation,raw in zip(observations,raws):
        config=raw['config'];window=raw['measurement_window'];start=window['start_time_ns']
        share='1/32' if raw['plan']['routing']['text'].get('generalist_2') else '0'
        case=next(c for c in scenario['cases'] if c['seed']==seed and c['workload']==config['workload'] and c['generalist_share']==share)
        rows=case['rows'];actual={i:r for i,r in enumerate(raw['request_records'])}
        assert len(rows)==len(actual)
        for row in rows:
            a=actual[row['index']]
            assert (row['task'],row['pool_id'],row['alias'])==(a['task'],a['pool_id'],a['model'])
            assert abs(row['arrival_seconds']-(a['scheduled_time_ns']-start)/1e9)<1e-8
        def historical(row):
            if row['task']=='text':return vectors[row['alias']][row['pool_id']]*(.5 if row['family']=='generalist_2' else 1)
            return baselines[row['family']][row['task']]['1']['samples'][row['pool_id']]['median_service_ms']
        def native(row):return actual[row['index']]['worker_response']['service_ms']
        def handler(row):return actual[row['index']]['worker_response']['worker_latency_ms']
        def submitted(row):return (actual[row['index']]['actual_submit_time_ns']-start)/1e9
        modes=[('historical_half_G2',historical,None),
            ('actual_specialist_only',lambda r:native(r) if r['family']=='text' else historical(r),None),
            ('actual_G2_text_only',lambda r:native(r) if r['task']=='text' and r['family']=='generalist_2' else historical(r),None),
            ('actual_background_only',lambda r:native(r) if r['task']!='text' else historical(r),None),
            ('all_actual_native',native,None),('all_actual_handler',handler,None),
            ('all_actual_handler_actual_submit',handler,submitted),
            ('shared_baseline_input_specialist_actual_others',lambda r:pooled[r['pool_id']] if r['family']=='text' else native(r),None),
            ('shared_baseline_input_specialist_historical_others',lambda r:pooled[r['pool_id']] if r['family']=='text' else historical(r),None),
            ('shared_baseline_input_mean_specialist_actual_others',lambda r:pooled_mean[r['pool_id']] if r['family']=='text' else native(r),None),
            ('baseline_exact_request_specialist_actual_others',lambda r:baseline_exact[r['index']] if r['family']=='text' else native(r),None)]
        replays=[]
        for name,service,release in modes:
            result=replay(rows,service,release);replays.append(dict(name=name,terms=terms(result,config),**result))
        measured=dict(per_task={t:dict(mean_ms=v['mean_e2e_latency_ms'],p95_ms=v['p95_e2e_latency_ms']) for t,v in raw['per_task_metrics'].items()})
        exact=next(r for r in replays if r['name']=='all_actual_handler_actual_submit')
        residuals=[actual[r['index']]['e2e_latency_ms']-r['latency_ms'] for r in exact['rows']]
        scheduling=[r['schedule_delay_ms'] for r in raw['request_records']]
        alias_service=[]
        for alias in sorted(raw['replica_to_family']):
            subset=[r for r in raw['request_records'] if r['model']==alias and r['task']=='text']
            if not subset:continue
            values=[r['worker_response']['service_ms'] for r in subset]
            alias_service.append(dict(alias=alias,n=len(values),mean_ms=statistics.mean(values),p95_ms=fine.percentile(values,.95),
                historical_matching_input_mean_ms=statistics.mean(vectors[alias][r['pool_id']] for r in subset),
                baseline_input_mean_transferred_to_current_mix_ms=statistics.mean(pooled_mean[r['pool_id']] for r in subset),
                observed_minus_baseline_input_mean_ms=statistics.mean(values)-statistics.mean(pooled_mean[r['pool_id']] for r in subset)))
        jobs.append(dict(label=observation['label'],job_id=observation['job_id'],measured_terms=terms(measured,config),
            measured_per_task=measured['per_task'],alias_text_service=alias_service,per_input_service=summarize_input(raw,vectors),
            schedule_delay_ms=dict(mean=statistics.mean(scheduling),p95=fine.percentile(scheduling,.95),maximum=max(scheduling)),
            actual_handler_submit_replay_residual_ms=dict(mean=statistics.mean(residuals),p95=fine.percentile(residuals,.95),maximum=max(residuals),minimum=min(residuals)),
            replays=replays))
    comparisons=[]
    for left,right in zip(jobs[0]['replays'],jobs[1]['replays']):
        assert left['name']==right['name']
        comparisons.append(dict(name=left['name'],delta_split_minus_baseline={k:right['terms'][k]-left['terms'][k] for k in ['L','P']},
            baseline_terms=left['terms'],split_terms=right['terms']))
    result=dict(schema='gsdeploy.posthoc_pair_component_replay.v1',seed=seed,jobs=jobs,comparisons=comparisons,
        inputs=inputs,predictor_activated=False,causal_claim=False,
        measured_delta={k:jobs[1]['measured_terms'][k]-jobs[0]['measured_terms'][k] for k in ['L','P']},
        limitations=['Post-hoc actual durations are unavailable before execution; no validation/prediction claim.',
            'Component substitution freezes durations despite interference changes; counterfactual service interaction is unobserved.',
            'A single ordered pair cannot distinguish input shape, shared-GPU interaction, and temporal drift causally.',
            'Shared baseline input table uses baseline specialist medians pooled by input, an explicit transfer assumption.',
            'Baseline exact-request transfer preserves baseline temporal variability but moves service to different aliases; unobserved counterfactual.',
            'Original fine-route half-G2 sensitivity is an assumption, not fitted performance evidence.'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'manifest.json').write_text(json.dumps(dict(inputs=inputs,tools=[fine.exposure.receipt(Path(__file__)),fine.exposure.receipt(Path(fine.__file__))],
        report=fine.exposure.receipt(output/'report.json')),indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['comparison','scenarios','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    report=diagnose(args.comparison,args.scenarios,args.output,args.seed)
    print(json.dumps(dict(output=str(args.output.resolve()),measured_delta=report['measured_delta'],comparisons=report['comparisons']),indent=2))


if __name__=='__main__':main()
