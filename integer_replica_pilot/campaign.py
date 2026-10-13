"""Exactly one W2 empirical pilot: 3 smoke attempts, <=18 search, 4 confirmations."""
import argparse
import fcntl
import json
from pathlib import Path
import time
from dataclasses import fields

from candidate_pool_service_demand.dry_run import load_completed_catalog
from deployment_core import Workload
from deployment_core.replica_calibration import load_replica_calibration
from deployment_search import IntegerGuidedDeploymentSearch
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from deployment_search.tests.fixtures import forbid_enumeration
from deployment_search.runtime_cache import save_json, file_hash
from deployment_search.result import jsonable
from joint_deploy.measurement.schema import MeasurementConfig
from .prepare import OUT, CONFIG, prepare, verify_freeze, read
from .evaluator import ReplicaRuntimeEvaluator, PilotExecutionBlocked, runner, summarize, aggregate


def once(directory,label,config,context_path,attempts_path,cap,catalog,workload):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    raw_path=directory/(label+'.json');rep_path=directory/(label+'.summary.json')
    config_path=directory/(label+'.config.json');log_path=directory/(label+'.log')
    attempts=read(attempts_path) if attempts_path.exists() else []
    previous=[a for a in attempts if a['label']==label]
    if rep_path.exists():
        rep=read(rep_path)
        if file_hash(rep['raw_result_path'])!=rep['raw_sha256']:raise RuntimeError('Saved once raw hash drift')
        return rep
    if previous:
        if not raw_path.exists():raise PilotExecutionBlocked('Interrupted once attempt; no duplicate/retry launch')
        process_path=raw_path.with_suffix('.process.json')
        if process_path.exists():
            import os
            pid=read(process_path)['pid']
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise PilotExecutionBlocked(f'Once harness {pid} still alive')
        record=previous[0];outcome=record.get('outcome',{'returncode':-1,'recovered_after_parent_stop':True})
    else:
        if len(attempts)>=cap:raise RuntimeError('Frozen once physical-attempt cap exhausted')
        record={'label':label,'status':'started','raw_result_path':str(raw_path),'started_unix':time.time()}
        attempts.append(record);save_json(attempts_path,attempts);save_json(config_path,config)
        print(json.dumps({'event':'once_start','label':label,'deployment':config['deployment']}),flush=True)
        outcome=runner(config,raw_path,log_path,config_path,context_path);record['outcome']=outcome;save_json(attempts_path,attempts)
    if not raw_path.exists():raise PilotExecutionBlocked('No once raw result; no retry')
    raw=read(raw_path)
    if any(raw.get(k)!=v for k,v in config.items() if k!='arrival_process'):raise PilotExecutionBlocked('Once raw configuration mismatch')
    rep=summarize(raw,workload,catalog,outcome,raw_path)
    record.update(status=rep['run_status'],raw_sha256=rep['raw_sha256']);save_json(attempts_path,attempts);save_json(rep_path,rep)
    print(json.dumps({'event':'once_complete','label':label,'status':rep['run_status'],'stability':rep['stability_state_v2']}),flush=True)
    return rep


def smoke(protocol,catalog):
    results=[]
    for slot in protocol['adapter_smoke_plan']:
        model,task=slot['model_id'],slot['task']
        cfg=MeasurementConfig(deployment={m:2 if m==model else 0 for m in catalog},workload_rps={task:1.},
            routing_rps={task:{m:1. if m==model else 0. for m in catalog}},random_seed=slot['random_seed'],
            **dict(protocol['measurement_settings'],measurement_seconds=2.,warmup_seconds=0.))
        rep=once(OUT/'adapter_smoke',model+'_2',cfg.as_dict(),OUT/'runtime_context.json',OUT/'smoke_attempts.json',3,catalog,Workload({task:1.}))
        raw=read(rep['raw_result_path']);ids=raw.get('worker_identities',{})
        counts=raw.get('per_replica_request_count',{})
        independent=len(ids)==2 and len({v['pid'] for v in ids.values()})==2
        cuda=all(s['each_owned_pid_has_cuda_context'] for s in raw.get('idle_before_workload',[])) and bool(raw.get('idle_before_workload'))
        passed=bool(rep['valid_runtime_repetition'] and independent and cuda and len(counts)==2 and set(counts.values())=={1} and
            raw['system_metrics']['completion_ratio']==1 and raw['gpu_release']['verified'] and
            all(not s['forced'] and s['exit_code']==0 and s['bye_received'] for s in raw['shutdown'].values()))
        results.append({'model_id':model,'deployment':cfg.deployment,'passed':passed,'distinct_PIDs':independent,
            'CUDA_process_residency_verified':cuda,'per_replica_request_count':counts,'worker_identities':ids,
            'actual_idle_mib':raw.get('memory_observations',{}).get('actual_idle_mib'),
            'memory':raw.get('memory_observations'),'GPU_release':raw.get('gpu_release'),
            'raw':{'path':rep['raw_result_path'],'sha256':rep['raw_sha256']},'summary':rep})
        save_json(OUT/'replica_adapter_validation.json',{'status':'passed' if len(results)==3 and all(r['passed'] for r in results) else 'in_progress' if passed else 'failed',
            'conditions':results,'physical_attempt_cap':3,'full_scaling_benchmark_repeated':False})
        if not passed:raise PilotExecutionBlocked('Adapter correctness smoke failed; no formal search launched')
    return results


def confirmation(trace,protocol,evaluator,catalog,workload):
    freeze={'search_trace_sha256':file_hash(OUT/'search_trace.json'),'seed':trace['seed']['deployment'],
        'final':trace['deployment'],'decision_frozen_before_confirmation':True}
    path=OUT/'post_search_freeze.json'
    if path.exists() and read(path)!=freeze:raise RuntimeError('Search decision changed before confirmation')
    save_json(path,freeze)
    states={'seed':(trace['seed']['deployment'],trace['iterations'][0]['canonical']['routing']),
            'final':(trace['deployment'],trace['canonical']['routing'])}
    rows={'seed':[],'final':[]}
    for which,index in protocol['confirmation_order']:
        x,r=states[which]
        routing={(k,m):v for k_m,v in r.items() for k,m in [k_m.split('::')]}
        identity=evaluator.identity(x,workload,routing)
        cfg=MeasurementConfig(identity['deployment'],identity['workload'],identity['routing'],
            random_seed=protocol['confirmation_seeds'][index-1],**protocol['measurement_settings']).as_dict()
        rep=once(OUT/'confirmation',which+'_rep'+str(index),cfg,OUT/'runtime_context.json',OUT/'confirmation_attempts.json',4,catalog,workload)
        rep.update(repetition=index,random_seed=protocol['confirmation_seeds'][index-1],post_search_confirmation=True)
        rows[which].append(rep)
        save_json(OUT/'confirmation_results.json',{'complete':False,'repetitions':rows,'influences_search':False})
        if not rep['valid_runtime_repetition']:raise PilotExecutionBlocked('Confirmation failed; no additional repetitions/retries')
    aggs={which:aggregate(sorted(reps,key=lambda r:r['repetition']),'independent_confirmation',OUT/'confirmation'/f'{which}_aggregate.json',states[which][0]) for which,reps in rows.items()}
    for which,agg in aggs.items():save_json(OUT/'confirmation'/f'{which}_aggregate.json',agg)
    seed,final=aggs['seed'],aggs['final']
    gain=(seed.weighted_mean_e2e-final.weighted_mean_e2e)/seed.weighted_mean_e2e if seed.weighted_mean_e2e and final.weighted_mean_e2e is not None else None
    better,reason=NoiseAwareRuntimeObjective(1.,.05).compare(final,seed,0.,0.)
    value={'complete':True,'influences_search':False,'repetitions':rows,'aggregates':aggs,
        'relative_mean_gain':gain,'mean_direction_maintained':gain is not None and gain>=0,
        'objective_direction_maintained':better,'objective_reason':reason,'search_final_equals_seed':states['seed'][0]==states['final'][0],
        'physical_attempts':len(read(OUT/'confirmation_attempts.json'))}
    save_json(OUT/'confirmation_results.json',value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.pilot.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare()
        if args.prepare_only:return
        protocol=read(OUT/'pilot_protocol.json');catalog,hardware,roles,records,_=load_completed_catalog();cal=load_replica_calibration(catalog)
        w=Workload(protocol['workload']['arrival_rates_rps'])
        try:
            if not (OUT/'replica_adapter_validation.json').exists() or read(OUT/'replica_adapter_validation.json')['status']!='passed':smoke(protocol,catalog)
            verify_freeze()
            evaluator=ReplicaRuntimeEvaluator(OUT/'search_cache',catalog,OUT/'runtime_context.json',protocol['measurement_settings'],seeds=protocol['search_seeds'])
            finalpath=OUT/'search_complete.json'
            if finalpath.exists():trace=read(finalpath)
            else:
                with forbid_enumeration() as tripwires:
                    result=IntegerGuidedDeploymentSearch(evaluator,predictor=None,runtime_objective=NoiseAwareRuntimeObjective(1.,.05),
                        trace_callback=lambda r:save_json(OUT/'search_trace.json',r.to_dict())).search(w,catalog,hardware,CONFIG,
                            roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
                trace=result.to_dict();trace['enumeration_calls']=sum(m.call_count for m in tripwires)
                save_json(OUT/'search_trace.json',trace);save_json(finalpath,trace)
            if trace['runtime_evaluation_count']>6 or trace['search_iterations']>3:raise AssertionError('Pilot budget exceeded')
            confirmation(trace,protocol,evaluator,catalog,w)
            from .report import report
            report()
            save_json(OUT/'campaign_status.json',{'status':'complete','stopped_after_module':'7C-3'})
            print(json.dumps({'event':'pilot_complete','deployment':trace['deployment'],'evaluations':trace['runtime_evaluation_count']}),flush=True)
        except (PilotExecutionBlocked,KeyboardInterrupt) as exc:
            save_json(OUT/'campaign_status.json',{'status':'blocked','reason':str(exc),'no_retry_or_budget_growth':True})
            from .report import report
            report(incomplete=True)
            print(json.dumps({'event':'pilot_blocked','reason':str(exc)}),flush=True)
            raise


if __name__=='__main__':main()
