"""Authorized terminal107/108 failure audit and symmetric post-hoc components.

Preserves the pre-result forecast/model and never fits. Actual durations in FIFO
substitutions are explanatory assumptions, not GPU counterfactual observations.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'artifacts/static-spillover-study-20261022'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def pct(v,q):
    if not v:return None
    v=sorted(v);i=(len(v)-1)*q;lo=math.floor(i);hi=math.ceil(i)
    return v[lo]+(v[hi]-v[lo])*(i-lo)
def summary(values):return dict(n=len(values),mean=statistics.mean(values) if values else None,median=statistics.median(values) if values else None,p95=pct(values,.95) if values else None,total=sum(values))

def replay(rows,durations,config,releases=None):
    clocks=defaultdict(float);details=[]
    for i,row in enumerate(rows):
        release=releases[i] if releases else row['arrival_seconds'];service=durations[i]
        assert service>0 and release>=row['arrival_seconds']-1e-8
        start=max(release,clocks[row['alias']]);end=start+service/1000;clocks[row['alias']]=end
        details.append(dict(index=i,task=row['task'],alias=row['alias'],pool_id=row['pool_id'],service_ms=service,
            finish_seconds=end,latency_ms=(end-row['arrival_seconds'])*1000,queue_ms=(start-release)*1000))
    horizon=config['horizon_seconds'];per_task={}
    for task in config['workload']:
        rr=[r for r in details if r['task']==task];per_task[task]=dict(mean_ms=statistics.mean(r['latency_ms'] for r in rr),p95_ms=pct([r['latency_ms'] for r in rr],.95))
    total=sum(config['workload'].values())
    terms={k:sum(rate/total*per_task[t][metric]/config['latency_reference_ms'][t] for t,rate in config['workload'].items()) for k,metric in [('L','mean_ms'),('P','p95_ms')]}
    terms['T']=1-sum(r['finish_seconds']<=horizon for r in details)/len(details)
    return dict(terms=terms,per_task=per_task,details=details)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    inputs={}
    def read(path,role):
        path=Path(path).resolve();inputs[str(path)]=dict(sha256=sha(path),role=role);return json.loads(path.read_text())
    queue_path=STUDY/'text185-fine-development-v1/gpu-runs-v1/queue-progress.json';queue=read(queue_path,'authorized_terminal_fine_receipts')
    assert queue['status']=='ALL_DECLARED_DEVELOPMENT_ATTEMPTS_COLLECTED'
    audit=read(STUDY/'staged-continuation-v1/execution/fine-terminal-audit/report.json','authorized_fine_terminal_audit')
    forecast_dir=STUDY/'local-context-seed108-forecast-v1'
    assert sha(forecast_dir/'manifest.json')=='42ead4e03d4f76955913d11e56dc192356d678296fbad5f777a287ad312a93bd'
    forecast=read(forecast_dir/'forecast.json','immutable_pre_result108_forecast')
    freeze=read(forecast_dir/'frozen-candidate.json','immutable_pre_result108_candidate')
    assert sha(forecast_dir/'forecast.json')=='500975f6ef004ed2d0bb06f4609626c2f21e9ed3527745d6cd3e64135af3d179'
    assert sha(forecast_dir/'frozen-candidate.json')==forecast['frozen_candidate_sha256']
    inventory=read(STUDY/'context-design-inventory-v5/requests.json','authorized_derived_pre108_baselines')
    baseline={}
    for r in inventory:
        if r['baseline_ms']:baseline[r['family'],r['task'],r['n'],r['sample_id']]=r['baseline_ms']
    packet=read(STUDY/'text185-fine-development-v1/experiment-packet.json','frozen_experiment_input_packet')
    source=ROOT/'tools/spillover_capacity_20261022/exposure.py';spec=importlib.util.spec_from_file_location('_reversal_cpu_exposure',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);api=module.frozen_api()
    inputs[str(source)]=dict(sha256=sha(source),role='frozen_cpu_input_reconstruction_loader')
    for rel in ['gsdeploy/trace.py','gsdeploy/routing.py','gsdeploy/runtime/workload.py']:
        path=module.SOURCE/rel;inputs[str(path)]=dict(sha256=sha(path),role='frozen_cpu_trace_input_router')
    all_results=[];failure=[];input_analysis=[];component_rows=[]
    for seed in [107,108]:
        jobs=[j for j in queue['jobs'] if j['label'].endswith(f'seed{seed}')];assert len(jobs)==2
        jobs.sort(key=lambda j:'0of1' not in j['label']);pairs=[]
        for receipt in jobs:
            assert receipt['status']=='completed' and receipt['eligible'] is True and sha(receipt['raw']['path'])==receipt['raw']['sha256']
            raw=read(receipt['raw']['path'],'explicit_authorized_terminal_development_raw')
            assert raw['status']=='completed' and raw['transport_config']['random_seed']==seed and raw['experiment_job_id']==receipt['job_id']
            a=next(j for j in audit['jobs'] if j['job_id']==receipt['job_id']);assert a['audit_status']=='VERIFIED_TERMINAL_EVIDENCE' and a['route_and_input_replay_verified']
            job=next(j for j in packet['jobs'] if j['id']==receipt['job_id']);ident=job['identity'];config=raw['config'];trace=api['trace'].make_trace(ident['workload'],ident['horizon_seconds'],seed,ident['arrival_process'])
            assert api['trace'].trace_digest(trace)==ident['trace_sha256']==raw['measurement_window']['trace_sha256']
            aliases,routes=api['trace'].expand_routes(job['plan']['deployment'],job['plan']['routing']);pool_path=Path(packet['inputs'])/'runtime-inputs.jsonl'
            assert sha(pool_path)==ident['input_pool_sha256'];inputs[str(pool_path)]=dict(sha256=sha(pool_path),role='frozen_runtime_input_pool_only')
            pool=api['workload'].SamplePool(pool_path,list(ident['workload']),seed);router=api['routing'].Router(routes);rows=[]
            start=raw['measurement_window']['start_time_ns']
            assert len(trace)==len(raw['request_records'])
            for i,((offset,task),rec) in enumerate(zip(trace,raw['request_records'])):
                sample=pool.next(task);alias=router.choose(task)
                assert (rec['task'],rec['pool_id'],rec['model'])==(task,sample['pool_id'],alias)
                assert abs((rec['scheduled_time_ns']-start)/1e9-offset)<1e-8
                assert rec['status']=='completed' and rec['worker_response']['success'] is True
                rows.append(dict(index=i,arrival_seconds=offset,task=task,pool_id=sample['pool_id'],alias=alias,family=aliases[alias]))
            pairs.append(dict(label=receipt['label'],raw=raw,rows=rows,audit=a))
        left,right=pairs;assert [(r['arrival_seconds'],r['task'],r['pool_id']) for r in left['rows']]==[(r['arrival_seconds'],r['task'],r['pool_id']) for r in right['rows']]
        base_actual=[r['worker_response']['service_ms'] for r in left['raw']['request_records']]
        mean_input=defaultdict(list);mean_alias_input=defaultdict(list);split_input=defaultdict(list)
        for rec in left['raw']['request_records']:
            if rec['task']=='text':mean_input[rec['pool_id']].append(rec['worker_response']['service_ms']);mean_alias_input[rec['model'],rec['pool_id']].append(rec['worker_response']['service_ms'])
        for rec in right['raw']['request_records']:
            if rec['task']=='text' and right['raw']['replica_to_family'][rec['model']]=='text':split_input[rec['pool_id']].append(rec['worker_response']['service_ms'])
        pooled={k:statistics.mean(v) for k,v in mean_input.items()};split_pooled={k:statistics.mean(v) for k,v in split_input.items()}
        assert set(pooled)==set(split_pooled)
        results=[]
        for pair in pairs:
            raw=pair['raw'];rows=pair['rows'];config=raw['config'];actual=[r['worker_response']['service_ms'] for r in raw['request_records']]
            handler=[r['worker_response']['worker_latency_ms'] for r in raw['request_records']]
            releases=[(r['actual_submit_time_ns']-raw['measurement_window']['start_time_ns'])/1e9 for r in raw['request_records']]
            conditions=[]
            modes=[('actual_native',actual,None),('actual_handler',handler,None),('actual_handler_actual_submit',handler,releases),
                ('baseline_exact_request_text_actual_others',[base_actual[i] if r['family']=='text' else actual[i] for i,r in enumerate(rows)],None),
                ('common_baseline_input_mean_text_actual_others',[pooled[r['pool_id']] if r['family']=='text' else actual[i] for i,r in enumerate(rows)],None),
                ('common_split_input_mean_text_actual_others',[split_pooled[r['pool_id']] if r['family']=='text' else actual[i] for i,r in enumerate(rows)],None)]
            # Scope component timing in exact same observed cohort, no service model refit.
            if seed==108:
                fc=next(c for c in forecast['cases'] if c['label']==pair['label']);fr=read(forecast_dir/(pair['label']+'-requests.json'),'immutable_pre_result108_request_forecast')
                assert len(fr)==len(rows)
                predicted=[]
                for i,(r,f) in enumerate(zip(rows,fr)):
                    assert (r['pool_id'],r['alias'],r['task'])==(f['pool_id'],f['alias'],f['task']);assert f['predicted_median_scale_service_ms'] is not None
                    predicted.append(f['predicted_median_scale_service_ms'])
                modes.extend([('frozen_pre_result_median_service',predicted,None),
                    ('forecast_actual_specialist_text_only',[actual[i] if r['family']=='text' else predicted[i] for i,r in enumerate(rows)],None),
                    ('forecast_actual_G2_text_only',[actual[i] if r['family']=='generalist_2' and r['task']=='text' else predicted[i] for i,r in enumerate(rows)],None),
                    ('forecast_actual_background_only',[actual[i] if r['task']!='text' else predicted[i] for i,r in enumerate(rows)],None)])
                measured=pair['audit']['utility'];predterms=fc['median_scale_FIFO_LPT_sensitivity']
                failure.append(dict(label=pair['label'],forecast=predterms,measured={k:measured[k+'_norm'] for k in ['L','P','T']},
                    errors={k:dict(prediction_minus_actual=predterms[k]-measured[k+'_norm'],absolute_relative_error=abs(predterms[k]-measured[k+'_norm'])/max(abs(measured[k+'_norm']),1e-12)) for k in ['L','P','T']},
                    model_frozen_sha256=forecast['frozen_candidate_sha256'],refitted=False))
                for family,task in sorted({(r['family'],r['task']) for r in rows}):
                    ix=[i for i,r in enumerate(rows) if (r['family'],r['task'])==(family,task)]
                    component_rows.append(dict(seed=seed,label=pair['label'],family=family,task=task,
                        actual_service=summary([actual[i] for i in ix]),pre_result_geometric_predictions=summary([predicted[i] for i in ix]),
                        actual_arithmetic_over_geometric_total=sum(actual[i] for i in ix)/sum(predicted[i] for i in ix)))
            replays=[]
            for name,durations,release in modes:
                result=replay(rows,durations,config,release);replays.append(dict(name=name,terms=result['terms'],per_task=result['per_task']))
                if name=='actual_native':native_details=result['details']
                if name=='actual_handler_actual_submit':
                    residual=[rec['e2e_latency_ms']-detail['latency_ms'] for rec,detail in zip(raw['request_records'],result['details'])]
                    exact_residual=summary(residual)
            aliases=[]
            for alias in sorted(set(r['alias'] for r in rows)):
                ix=[i for i,r in enumerate(rows) if r['alias']==alias and r['task']=='text']
                if not ix:continue
                by_sid=Counter(rows[i]['pool_id'] for i in ix)
                transfer=[pooled[rows[i]['pool_id']] for i in ix];services=[actual[i] for i in ix]
                family=rows[ix[0]]['family'];n=raw['plan']['deployment'][family]
                planned=[baseline[family,'text',n,rows[i]['pool_id']] for i in ix]
                partner_ix=[i for i,r in enumerate(rows) if r['family']==family and r['alias']!=alias]
                partner_baseline=sum(baseline[family,rows[i]['task'],n,rows[i]['pool_id']] for i in partner_ix)
                probs=[count/len(ix) for count in by_sid.values()]
                temporal=[]
                for begin in [0,150,300,450]:
                    ii=[i for i in ix if begin<=rows[i]['arrival_seconds']<begin+150]
                    temporal.append(dict(begin_seconds=begin,count=len(ii),actual_service_total_ms=sum(actual[i] for i in ii),
                        transferred_baseline_input_total_ms=sum(pooled[rows[i]['pool_id']] for i in ii)))
                aliases.append(dict(alias=alias,requests=len(ix),unique_inputs=len(by_sid),per_input_counts=dict(by_sid),
                    actual_service=summary(services),no_split_input_mix_transfer=summary(transfer),
                    actual_over_same_input_no_split_pooled_total=sum(services)/sum(transfer),quarter_windows=temporal,
                    planned_trace_baseline_context=dict(baseline_service=summary(planned),
                        baseline_work_seconds=sum(planned)/1000,baseline_work_over600s=sum(planned)/600000,
                        input_frequency_entropy=-sum(v*math.log(v) for v in probs),task_mix_entropy=0.,
                        same_family_partner_baseline_work_seconds=partner_baseline/1000,
                        input_baseline_cv=statistics.pstdev(planned)/statistics.mean(planned),
                        interpretation='Plan-known exact trace/input/alias baseline only; actual services not features; n-state baseline already includes replica effects'),
                    work_seconds=sum(services)/1000,work_over600s=sum(services)/600000,
                    posthoc_native_FIFO_queue_ms=summary([native_details[i]['queue_ms'] for i in ix]),
                    posthoc_native_FIFO_latency_ms=summary([native_details[i]['latency_ms'] for i in ix])))
            for (alias,sid),records in sorted(((k,[rec for rec in raw['request_records'] if (rec['model'],rec['pool_id'])==k and rec['task']=='text']) for k in {(rec['model'],rec['pool_id']) for rec in raw['request_records'] if rec['task']=='text'})):
                values=[r['worker_response']['service_ms'] for r in records]
                input_analysis.append(dict(seed=seed,label=pair['label'],alias=alias,family=raw['replica_to_family'][alias],sample_id=sid,
                    observed_service=summary(values),same_input_no_split_specialist_mean_ms=pooled[sid],
                    relative_to_same_input_no_split_mean=statistics.mean(values)/pooled[sid],
                    generated_token_counts=dict(Counter(str(r['worker_response'].get('generated_tokens')) for r in records)),
                    effective_generation_digests=dict(Counter(hashlib.sha256(json.dumps(r['worker_response'].get('effective_generation'),sort_keys=True).encode()).hexdigest() for r in records))))
            for alias in ['text#0','text#1']:
                for sid in pooled:
                    vv=[r['worker_response']['service_ms'] for r in raw['request_records'] if r['task']=='text' and r['model']==alias and r['pool_id']==sid]
                    if vv:conditions.append(dict(alias=alias,sample_id=sid,mean_ms=statistics.mean(vv),n=len(vv)))
            results.append(dict(seed=seed,label=pair['label'],job_id=raw['experiment_job_id'],raw_sha256=sha(next(j['raw']['path'] for j in jobs if j['label']==pair['label'])),
                utility=pair['audit']['utility'],actual_per_task_metrics=raw['per_task_metrics'],replays=replays,
                alias_text_balance=aliases,per_input_alias_conditions=conditions,actual_handler_submit_replay_e2e_residual_ms=exact_residual,
                scheduling_ms=summary([r['schedule_delay_ms'] for r in raw['request_records']])))
        comparisons=[]
        for a,b in zip(results[0]['replays'],results[1]['replays']):
            assert a['name']==b['name'];comparisons.append(dict(name=a['name'],delta_split_minus_nosplit={k:b['terms'][k]-a['terms'][k] for k in ['L','P','T']},nosplit=a['terms'],split=b['terms']))
        u0,u1=[r['utility'] for r in results]
        common_cells={ (c['alias'],c['sample_id']) for c in results[0]['per_input_alias_conditions']} & { (c['alias'],c['sample_id']) for c in results[1]['per_input_alias_conditions']}
        lookup=[{(c['alias'],c['sample_id']):c for c in result['per_input_alias_conditions']} for result in results]
        paired_cells=[dict(alias=k[0],sample_id=k[1],nosplit_mean_ms=lookup[0][k]['mean_ms'],split_mean_ms=lookup[1][k]['mean_ms'],
            split_over_nosplit=lookup[1][k]['mean_ms']/lookup[0][k]['mean_ms'],nosplit_n=lookup[0][k]['n'],split_n=lookup[1][k]['n']) for k in sorted(common_cells)]
        balance=[]
        for result in results:
            aa=[a for a in result['alias_text_balance'] if a['alias'].startswith('text#')]
            values=[a['work_seconds'] for a in aa]
            balance.append(dict(label=result['label'],specialist_work_seconds=sum(values),maximum_alias_work_seconds=max(values),
                max_minus_min_alias_work_seconds=max(values)-min(values),max_to_mean_alias_work_ratio=max(values)/statistics.mean(values),
                mean_alias_service_ms=[a['actual_service']['mean'] for a in aa],
                maximum_alias_FIFO_p95_ms=max(a['posthoc_native_FIFO_latency_ms']['p95'] for a in aa)))
        all_results.append(dict(seed=seed,jobs=results,comparisons=comparisons,measured_delta={k:u1[k+'_norm']-u0[k+'_norm'] for k in ['Q','L','P','T','M','E']},
            measured_delta_U=u1['U']-u0['U'],specialist_alias_balance=balance,common_alias_input_service_cells=paired_cells,
            common_alias_input_ratio_summary=summary([c['split_over_nosplit'] for c in paired_cells])))
    predicted_delta={k:forecast['cases'][1]['median_scale_FIFO_LPT_sensitivity'][k]-forecast['cases'][0]['median_scale_FIFO_LPT_sensitivity'][k] for k in ['L','P','T']}
    observed108=all_results[1]['measured_delta']
    failure_report=dict(schema='gsdeploy.frozen108_forecast_failure.v1',status='FAILED_DIRECTION_AND_LARGE_COMPONENT_ERROR',
        prediction_delta_split_minus_nosplit=predicted_delta,observed_delta_split_minus_nosplit={k:observed108[k] for k in ['L','P','T']},
        direction_correct={k:predicted_delta[k]*observed108[k]>0 for k in ['L','P']},jobs=failure,
        candidate_refit=False,old_forecast_changed=False,development_diagnostic_only=True,
        conclusion='Frozen median-scale service FIFO failed to predict this development pair; no production activation or formal rho/U use.')
    save(out/'forecast-failure.json',failure_report);save(out/'per-input-alias-service.json',input_analysis);save(out/'forecast-service-component-errors.json',component_rows)
    result=dict(schema='gsdeploy.seed107108_reversal_components.v1',seeds=all_results,forecast_failure=failure_report,
        causal_identification=False,candidate_refit=False,predictor_activated=False,gpu_started=False,
        limitations=['Actual duration substitutions are post-hoc, not observed GPU counterfactuals.',
            'Alias/input mix, queue placement, output length, temporal drift and shared-board interference can co-vary.',
            'Seed107 and108 use different arrival/input/route exposures; opposite signs do not establish a universal split benefit.',
            'Arithmetic service differences were diagnosed, never silently used to refit/retransform the frozen forecast.'])
    save(out/'report.json',result);save(out/'opened-inputs.json',inputs)
    save(out/'manifest.json',dict(script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),inputs=inputs,
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(failure=failure_report,comparisons=[dict(seed=r['seed'],measured=r['measured_delta'],U=r['measured_delta_U'],comparisons=r['comparisons']) for r in all_results])))

if __name__=='__main__':main()
