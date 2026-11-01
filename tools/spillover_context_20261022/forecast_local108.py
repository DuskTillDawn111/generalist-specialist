"""Freeze a development service candidate and forecast seed108 from inputs only.

Never reads seed108 result raw/metadata. Median-scale deterministic alias FIFO is
a development sensitivity, not a formal mean/rho, utility, or optimality result.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def percentile(values,q):
    if not values:return None
    v=sorted(values);i=(len(v)-1)*q;lo=math.floor(i);hi=math.ceil(i)
    return v[lo]+(v[hi]-v[lo])*(i-lo)

def in_context_hull(target,contexts):
    points=np.asarray(contexts,dtype=float)
    if not len(points):return False,None
    target=np.asarray(target,dtype=float)
    # Convex mixture, exact n/path context; no marginal-box extrapolation waiver.
    a=np.vstack([points.T,np.ones(len(points))]);b=np.r_[target,1.]
    result=linprog(np.zeros(len(points)),A_eq=a,b_eq=b,bounds=(0,None),method='highs')
    if not result.success:return False,None
    err=float(np.max(np.abs(a@result.x-b)))
    return err<=1e-8,dict(maximum_absolute_residual=err,nonzero_weights=[dict(context_index=i,weight=float(w)) for i,w in enumerate(result.x) if w>1e-8])

def fifo_unknown(rows,horizon,drain):
    clocks=defaultdict(float);unknown_aliases=set();output=[]
    for r in rows:
        item=dict(r);alias=r['alias'];service=r['predicted_median_scale_service_ms']
        if service is None:unknown_aliases.add(alias)
        if alias in unknown_aliases:
            item.update(fifo_status='UNKNOWN_ALIAS_HAS_UNKNOWN_SERVICE',finish_seconds=None,latency_ms=None,queue_ms=None)
        else:
            start=max(r['arrival_seconds'],clocks[alias]);end=start+service/1000;clocks[alias]=end
            item.update(fifo_status='MEDIAN_SCALE_DETERMINISTIC_SENSITIVITY',finish_seconds=end,
                latency_ms=(end-r['arrival_seconds'])*1000,queue_ms=(start-r['arrival_seconds'])*1000,
                completed_in_window=end<=horizon,completed_by_drain=end<=horizon+drain)
        output.append(item)
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development',type=Path,required=True)
    p.add_argument('--development-manifest-sha256',required=True);p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--packet-sha256',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False);opened={}
    def read(path,role):
        path=Path(path).resolve();opened[str(path)]=dict(sha256=sha(path),role=role);return json.loads(path.read_text())
    assert sha(args.development/'manifest.json')==args.development_manifest_sha256
    manifest=read(args.development/'manifest.json','pinned_development_model_manifest')
    for name,expected in manifest['files'].items():assert sha(args.development/name)==expected
    development=read(args.development/'report.json','pinned_development_fit_only')
    inventory=Path(development['inventory']['path']);assert sha(inventory/'manifest.json')==development['inventory']['manifest_sha256']
    inventory_manifest=read(inventory/'manifest.json','pinned_inventory_manifest')
    for name,expected in inventory_manifest['files'].items():assert sha(inventory/name)==expected
    inventory_report=read(inventory/'report.json','audited_inventory_definition')
    requests=read(inventory/'requests.json','authorized_development_derived_requests_excluding108')
    baselines=read(inventory/'baseline-inventory.json','frozen_closed_loop_baseline_means')
    runs=read(inventory/'source-ledger.json','authorized_development_allowlist_excluding108')
    assert not any(r['seed'] in [61,62,71,72,73,81,82,108] for r in runs)
    candidate=next(c for c in development['candidates'] if c['name']=='family_self_other')
    assert development['selection']['simplest_within_one_SE']=='family_self_other'
    assert candidate['successful_folds']==candidate['total_folds']==19
    columns=candidate['columns'];beta=np.asarray(candidate['full_development_fit']['coefficients'])
    freeze=dict(schema='gsdeploy.frozen_local_service_development.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        source_development_report=dict(path=str((args.development/'report.json').resolve()),sha256=sha(args.development/'report.json')),
        name=candidate['name'],feature_columns=columns,feature_names=candidate['features'],coefficients=beta.tolist(),
        prediction='b * exp(X beta); conditional geometric/median scale only; no arithmetic-mean retransformation',
        domain=development['supported_local_domain'],loo_macro_run_log_rmse=candidate['macro_run_log_rmse'],
        coefficient_freeze_precedes_result108_reads=True,seed108_result_files_read=False,
        predictor_activated=False,formal_rho=None,U=None)
    save(out/'frozen-candidate.json',freeze)
    # Retrospective arithmetic-total diagnostic is not a forecast calibration factor.
    domain={(d['family'],d['task'],d['n']):d for d in freeze['domain'] if d['included']}
    groups=defaultdict(list)
    for r in requests:
        key=(r['family'],r['task'],r['n'])
        if r['valid_for_fit'] and key in domain:groups[(r['run_id'],*key)].append(r)
    arithmetic=[]
    for key,rr in sorted(groups.items()):
        predicted=sum(r['baseline_ms']*math.exp(sum(r['features'][i]*coef for i,coef in zip(columns,beta))) for r in rr)
        actual=sum(r['service_ms'] for r in rr)
        arithmetic.append(dict(run_id=key[0],family=key[1],task=key[2],n=key[3],requests=len(rr),actual_arithmetic_service_total_ms=actual,
            geometric_point_prediction_total_ms=predicted,actual_over_geometric_point_total=actual/predicted,
            role='RETROSPECTIVE_IN_SAMPLE_SCALE_DIAGNOSTIC_ONLY_NOT_FORECAST_COMPENSATION'))
    save(out/'development-arithmetic-scale-diagnostic.json',arithmetic)
    assert sha(args.packet)==args.packet_sha256
    packet=read(args.packet,'existing_frozen_fine_experiment_packet_input_only');config=packet['config'];catalog=config['catalog']
    jobs=[j for j in packet['jobs'] if j['seed']==108];assert len(jobs)==2
    sample_path=Path(packet['inputs'])/'runtime-inputs.jsonl'
    opened[str(sample_path.resolve())]=dict(sha256=sha(sample_path),role='frozen_input_pool_only')
    assert sha(sample_path)==catalog['quality']['identity']['input_pool_sha256']
    input_rows=[json.loads(v) for v in sample_path.read_text().splitlines() if v.strip()]
    assert set(r['pool_id'] for r in input_rows)==set(i for ids in catalog['quality']['input_order'].values() for i in ids)
    authorized_config=read(ROOT/'artifacts/static-spillover-study-20261022/text185-fine-development-v1/system-config.json','frozen_fine_config_input_only')
    for k in ['task_replica_profiles','capacity_factors','memory_table_mib','models']:
        assert catalog[k]==authorized_config['catalog'][k]
    # CPU frozen arrival/sample/router code only. No call to quality or GPU code.
    exposure_path=ROOT/'tools/spillover_capacity_20261022/exposure.py'
    opened[str(exposure_path)]=dict(sha256=sha(exposure_path),role='isolated_frozen_cpu_loader')
    spec=importlib.util.spec_from_file_location('_forecast_exposure',exposure_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    api=module.frozen_api()
    for rel in ['gsdeploy/trace.py','gsdeploy/routing.py','gsdeploy/runtime/workload.py']:
        path=SOURCE/rel;opened[str(path)]=dict(sha256=sha(path),role='frozen_trace_sample_alias_routing_cpu_source')
    bmeans={(r['family'],r['task'],r['n']):r for r in baselines}
    bs={}
    for r in requests:
        if r['baseline_ms'] is not None:
            key=(r['family'],r['task'],r['n'],r['sample_id']);assert key not in bs or bs[key]==r['baseline_ms'];bs[key]=r['baseline_ms']
    cases=[]
    for job in sorted(jobs,key=lambda j:bool(j['plan']['routing']['text'].get('generalist_2'))):
        ident=job['identity'];x=job['plan']['deployment'];route=job['plan']['routing'];rates=ident['workload'];horizon=ident['horizon_seconds']
        assert job['phase']=='calibration' and ident['seed']==108 and ident['input_pool_sha256']==sha(sample_path)
        A={m:sum(q*bmeans[m,t,x[m]]['input_median_mean_ms']/1000/catalog['capacity_factors'][m][str(x[m])] for t,row in route.items() for family,q in row.items() if family==m) for m in x}
        M=sum(catalog['memory_table_mib'][m][str(n)] for m,n in x.items())/49140
        H={}
        for m in x:
            shares=[row[m] for row in route.values() if row.get(m,0)>0];total=sum(shares)
            H[m]=-sum(q/total*math.log(q/total) for q in shares) if total else 0.
        path_proofs={}
        for task,row in route.items():
            for m,q in row.items():
                if not q:continue
                key=(m,task,x[m]);d=domain.get(key);target=[A[m],sum(v for family,v in A.items() if family!=m),M,H[m]]
                contexts=[[v[k] for k in ['A_self','A_other','M_resident','H_self']] for v in (d or {}).get('contexts_by_run',{}).values()]
                valid,proof=in_context_hull(target,contexts)
                path_proofs[key]=dict(in_supported_context_convex_hull=valid,witness=proof,context_target=target,
                    support_status='WEAK_SUPPORT' if (d or {}).get('weak_support') else 'OBSERVED_LOCAL_SUPPORT',
                    allowed_sample_ids=(d or {}).get('sample_ids',[]))
        trace=api['trace'].make_trace(rates,horizon,108,ident['arrival_process']);assert api['trace'].trace_digest(trace)==ident['trace_sha256']
        aliases,routes=api['trace'].expand_routes(x,route);pool=api['workload'].SamplePool(sample_path,list(rates),108);router=api['routing'].Router(routes)
        rows=[]
        for index,(arrival,task) in enumerate(trace):
            sample=pool.next(task);alias=router.choose(task);m=aliases[alias];n=x[m];sid=sample['pool_id'];key=(m,task,n);proof=path_proofs[key];reason=None
            if not proof['in_supported_context_convex_hull']:reason='OUTSIDE_EXACT_PATH_N_CONTEXT_CONVEX_HULL'
            elif sid not in proof['allowed_sample_ids']:reason='INPUT_NOT_OBSERVED_IN_SUPPORTED_LOCAL_PATH_N_DOMAIN'
            elif (*key,sid) not in bs:reason='NO_EXACT_PER_INPUT_BASELINE'
            features=[float(f==m) for f in ['generalist_1','generalist_2','generalist_3','text','image','ocr','asr']]+[A[m],sum(v for family,v in A.items() if family!=m)]
            value=None if reason else bs[(*key,sid)]*math.exp(sum(features[i]*coef for i,coef in zip(columns,beta)))
            rows.append(dict(index=index,arrival_seconds=arrival,task=task,pool_id=sid,alias=alias,family=m,n=n,
                service_status='UNKNOWN' if reason else 'LOCAL_MEDIAN_SCALE_DEVELOPMENT_POINT',unknown_reason=reason,
                predicted_median_scale_service_ms=value,baseline_ms=bs.get((*key,sid)),features=features))
        forecast=fifo_unknown(rows,horizon,ident['drain_seconds']);task_metrics={}
        for task in rates:
            rr=[r for r in forecast if r['task']==task];known=[r for r in rr if r['latency_ms'] is not None]
            full=bool(rr) and len(rr)==len(known)
            task_metrics[task]=dict(offered=len(rr),service_known=sum(r['predicted_median_scale_service_ms'] is not None for r in rr),fifo_known=len(known),
                status='MEDIAN_SCALE_DEVELOPMENT_SENSITIVITY' if full else 'UNKNOWN_PARTIAL_ALIAS_FIFO',
                mean_latency_ms=statistics.mean(r['latency_ms'] for r in known) if full else None,p95_latency_ms=percentile([r['latency_ms'] for r in known],.95) if full else None,
                completion_loss_in_window=1-sum(r['completed_in_window'] for r in known)/len(rr) if full else None,
                known_prefix_only_not_full_metrics=dict(count=len(known),mean_latency_ms=statistics.mean(r['latency_ms'] for r in known) if known else None))
        full=all(r['latency_ms'] is not None for r in forecast) and all(v['offered'] for v in task_metrics.values())
        total=sum(rates.values());terms=dict(L=None,P=None,T=None)
        if full:
            terms=dict(L=sum(rates[t]/total*task_metrics[t]['mean_latency_ms']/config['latency_reference_ms'][t] for t in rates),
                P=sum(rates[t]/total*task_metrics[t]['p95_latency_ms']/config['latency_reference_ms'][t] for t in rates),
                T=1-sum(r['completed_in_window'] for r in forecast)/len(forecast))
        label=job['labels'][0];save(out/(label+'-requests.json'),forecast)
        cases.append(dict(label=label,job_id=job['id'],trace_sha256=ident['trace_sha256'],deployment=x,routing=route,
            A_by_family=A,M_resident=M,H_by_family=H,offered=len(rows),service_unknown=sum(r['predicted_median_scale_service_ms'] is None for r in rows),
            unknown_reasons=dict(Counter(r['unknown_reason'] for r in rows if r['unknown_reason'])),
            local_path_domain_proofs=[dict(family=k[0],task=k[1],n=k[2],**v) for k,v in path_proofs.items()],
            per_task=task_metrics,median_scale_FIFO_LPT_sensitivity=terms,
            LPT_status='DEVELOPMENT_SENSITIVITY' if full else 'UNKNOWN_INCOMPLETE_LOCAL_DOMAIN',
            median_scale_service_is_not_formal_mean=True,formal_rho=None,U=None))
    result=dict(schema='gsdeploy.seed108_local_service_forecast.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        frozen_candidate_sha256=sha(out/'frozen-candidate.json'),packet=dict(path=str(args.packet.resolve()),sha256=args.packet_sha256),cases=cases,
        protocol='Forecast from exact registered packet/input/seed108 trace before any seed108 raw read by this agent. Both jobs are development only; not independent confirmation.',
        limitations=['exp of mean log residual is geometric/median scale, not an arithmetic service mean.',
            'LOO log-RMSE about0.422; deterministic FIFO can amplify service error near congestion.',
            'Unknown service makes that alias FIFO suffix unknown; no true service substitution.',
            'No M/E predictor, no formal rho exclusion, no U prediction or total-utility ranking.',
            'Retrospective arithmetic ratio diagnostics were not used as compensation.'],
        seed108_result_files_read=False,seed108_queue_or_terminal_statistics_read=False,production_activated=False,gpu_started=False)
    save(out/'forecast.json',result);save(out/'opened-inputs.json',opened)
    save(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        inputs=opened,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps([dict(label=c['label'],service_unknown=c['service_unknown'],LPT=c['median_scale_FIFO_LPT_sensitivity'],per_task=c['per_task']) for c in cases]))

if __name__=='__main__':main()
