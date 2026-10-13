"""Integer exact-cache evaluator; three timed repetitions, no retries or substitutions."""
from dataclasses import dataclass, field, fields
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from deployment_search.measurement_runtime import (MeasurementRuntimeEvaluation, aggregate_repetitions, summarize_repetition, stats)
from deployment_search.runtime import RuntimeEvaluator
from deployment_search.runtime_cache import RuntimeEvaluationCache, save_json, file_hash, digest
from deployment_core.replicas import normalize_counts
from deployment_core import Workload
from joint_deploy.measurement.schema import MeasurementConfig
from .adapter import validate_memory

ROOT=Path(__file__).resolve().parents[1]


class PilotExecutionBlocked(RuntimeError):pass


@dataclass(frozen=True)
class ReplicaRuntimeEvaluation(MeasurementRuntimeEvaluation):
    replica_count_by_model: dict=field(default_factory=dict)
    per_replica_request_count: dict=field(default_factory=dict)
    per_replica_throughput: dict=field(default_factory=dict)
    per_replica_worker_latency: dict=field(default_factory=dict)
    per_replica_queue_wait: dict=field(default_factory=dict)
    aggregate_family_metrics: dict=field(default_factory=dict)
    memory_observations: list=field(default_factory=list)
    runtime_resource_status: str='feasible'
    valid_repetition_count: int=3
    execution_status: str='completed'


def runner(config,raw_path,log_path,config_path,context_path):
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,'-m','integer_replica_pilot.runtime','--config',str(config_path),
        '--context',str(context_path),'--output',str(raw_path)]
    timeout=sum(config['deployment'].values())*(config['startup_timeout_seconds']+config['shutdown_timeout_seconds'])+config['measurement_seconds']+config['warmup_seconds']+3*config['drain_timeout_seconds']+120
    start=time.time()
    with Path(log_path).open('x') as stream:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(Path(raw_path).with_suffix('.process.json'),{'pid':proc.pid,'command':command,'start_unix':start})
        try:code=proc.wait(timeout=timeout);outcome={'returncode':code,'timed_out':False}
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();outcome={'returncode':proc.returncode,'timed_out':True}
    return dict(outcome,elapsed_seconds=time.time()-start)


def summarize(raw,workload,catalog,outcome,path):
    if raw.get('runtime_resource_status')=='RUNTIME_RESOURCE_INFEASIBLE':
        return {'run_status':'RUNTIME_RESOURCE_INFEASIBLE','valid_runtime_repetition':False,
            'raw_result_path':str(path),'raw_sha256':file_hash(path),'process_outcome':outcome,
            'memory_observations':raw.get('memory_observations'),'errors':raw.get('errors',[]),
            'stability_state_v2':None,'reason':'Resource failure is preserved separately; no timed serving value imputed.'}
    valid=raw.get('valid_execution') and raw.get('measurement_window') and not outcome.get('timed_out')
    if not valid:
        return {'run_status':raw.get('execution_status','harness_failure'),'valid_runtime_repetition':False,
            'raw_result_path':str(path),'raw_sha256':file_hash(path),'process_outcome':outcome,'errors':raw.get('errors',[]),
            'stability_state_v2':None}
    aliases={i:catalog[m] for i,m in raw['replica_to_family'].items()}
    # Same DQA thresholds and standalone s0; genuine completed unstable stays valid.
    rep=summarize_repetition(raw,workload,aliases,{'returncode':0},path)
    rep.update(run_status='completed',valid_runtime_repetition=True,process_outcome=outcome,
        replica_count_by_model=raw['replica_count_by_model'],memory_observations=raw['memory_observations'],
        **{k:raw[k] for k in ('per_replica_request_count','per_replica_throughput','per_replica_worker_latency',
            'per_replica_queue_wait','aggregate_family_metrics')})
    return rep


def aggregate(reps,key,path,counts):
    base=aggregate_repetitions(reps,key,path)
    per={}
    for key_name in ('per_replica_request_count','per_replica_throughput'):
        per[key_name]={i:stats([r[key_name][i] for r in reps]) for i in reps[0][key_name]}
    for key_name in ('per_replica_worker_latency','per_replica_queue_wait'):
        per[key_name]={i:{metric:stats([r[key_name][i][metric] for r in reps]) for metric in ('mean','p95','observations')} for i in reps[0][key_name]}
    per['aggregate_family_metrics']={m:{k:stats([r['aggregate_family_metrics'][m][k] for r in reps]) for k in
        ('request_count','completed_requests','completed_throughput_rps','cohort_completion_throughput_rps','completion_ratio')}
        for m,n in counts.items() if n}
    return ReplicaRuntimeEvaluation(**{f.name:getattr(base,f.name) for f in fields(MeasurementRuntimeEvaluation)},
        replica_count_by_model=dict(counts),memory_observations=[r['memory_observations'] for r in reps],
        valid_repetition_count=len(reps),**per)


class ReplicaRuntimeEvaluator(RuntimeEvaluator):
    def __init__(self,directory,catalog,context_path,settings,seeds=(7101,7202,7303),max_unique=6,run=runner):
        if len(seeds)!=3 or len(set(seeds))!=3:raise ValueError('Exactly three independent search seeds required')
        self.directory=Path(directory);self.catalog=catalog;self.context_path=Path(context_path)
        self.context=json.loads(self.context_path.read_text());self.context_file_hash=file_hash(self.context_path)
        self.settings,self.seeds,self.run=dict(settings),tuple(seeds),run
        self.cache=RuntimeEvaluationCache(directory,max_unique,3)

    def identity(self,deployment,workload,routing):
        caps={m:row['supported_tasks'] for m,row in self.context['models'].items()}
        counts,_=validate_memory(deployment,caps,self.context['replica_bounds'],self.context['memory_table_mib'],self.context['gpu_capacity_mib'])
        tasks=[k for k,v in workload.arrival_rate.items() if v>0 or any(k in p.supported_tasks for p in self.catalog.values())]
        offered={k:workload.arrival_rate[k] for k in sorted(tasks)}
        nested={k:{m:float(routing.get((k,m),0.)) or 0. for m in sorted(self.catalog)} for k in sorted(tasks)}
        MeasurementConfig(counts,offered,nested,random_seed=self.seeds[0],**self.settings).validate(caps)
        return {'deployment':counts,'workload':offered,'routing':nested,
            'measurement_settings':dict(self.settings,repetition_seeds=list(self.seeds)),
            'runtime_context_sha256':self.context_file_hash,'source_model_hashes':self.context['source_hashes'],
            'capacity_mode':'empirical','replica_bounds':self.context['replica_bounds'],
            'routing_policy':'family routing then deterministic family-wide round-robin',
            'aggregation':'Module 3C worst stability; equal repetition mean; sample std ddof=1; no invalid-repeat retry or fallback'}

    def evaluate(self,deployment,workload,routing):
        if file_hash(self.context_path)!=self.context_file_hash:raise RuntimeError('Frozen context changed')
        identity=self.identity(deployment,workload,routing)
        with self.cache.locked() as cache:
            key,entry=cache.reserve(identity)
            if entry['status']=='complete':
                if file_hash(entry['aggregate'])!=entry['aggregate_sha256']:raise RuntimeError('Cached aggregate drift')
                result=ReplicaRuntimeEvaluation(**json.loads(Path(entry['aggregate']).read_text()))
                if any(file_hash(r['raw_result_path'])!=r['raw_sha256'] for r in result.repetitions):raise RuntimeError('Raw cache drift')
                cache.data['cache_hits']+=1;cache.flush();return result
            folder=cache.directory/'raw_runs'/key;folder.mkdir(parents=True,exist_ok=True)
            reps=[]
            for index,seed in enumerate(self.seeds,1):
                raw_path=folder/f'rep_{index}.json';log_path=folder/f'rep_{index}.log';config_path=folder/f'rep_{index}.config.json'
                config=MeasurementConfig(identity['deployment'],identity['workload'],identity['routing'],random_seed=seed,**self.settings).as_dict()
                previous=[r for r in entry['repetitions'] if r['index']==index]
                if previous:
                    record=previous[0]
                    if record['status']=='complete':
                        if file_hash(record['runtime']['raw_result_path'])!=record['runtime']['raw_sha256']:raise RuntimeError('Prior raw hash drift')
                        reps.append(record['runtime']);continue
                    process_path=raw_path.with_suffix('.process.json')
                    if process_path.exists():
                        pid=json.loads(process_path.read_text())['pid']
                        try:os.kill(pid,0)
                        except ProcessLookupError:pass
                        else:raise PilotExecutionBlocked(f'Prior harness {pid} still running; no duplicate')
                    outcome=record.get('outcome',{'returncode':-1,'recovered_after_parent_stop':True})
                else:
                    if len(cache.data['run_attempts'])>=cache.max_unique*3:raise RuntimeError('Frozen 18 search physical-attempt cap exhausted')
                    record={'index':index,'seed':seed,'status':'started','raw_result_path':str(raw_path),'started_unix':time.time()}
                    entry['repetitions'].append(record);cache.data['run_attempts'].append({'cache_key':key,'repetition':index,'raw_result_path':str(raw_path)})
                    save_json(config_path,config);cache.flush()
                    print(json.dumps({'event':'runtime_start','cache_key':key,'rep':index,'deployment':deployment}),flush=True)
                    try:outcome=self.run(config,raw_path,log_path,config_path,self.context_path)
                    except Exception as exc:outcome={'returncode':-1,'adapter_exception':repr(exc)}
                    record['outcome']=outcome;cache.flush()
                if not raw_path.exists():
                    record.update(status='interrupted',outcome=outcome);cache.flush()
                    raise PilotExecutionBlocked('No raw measurement; preserve attempt, do not retry or substitute')
                raw=json.loads(raw_path.read_text())
                if any(raw.get(k)!=v for k,v in config.items() if k!='arrival_process'):
                    record['status']='configuration_mismatch';cache.flush();raise PilotExecutionBlocked('Raw configuration identity mismatch')
                rep=summarize(raw,workload,self.catalog,outcome,raw_path);rep.update(repetition=index,random_seed=seed)
                record['runtime']=rep
                if not rep['valid_runtime_repetition']:
                    record['status']=rep['run_status'];entry['status']=rep['run_status'];cache.flush()
                    # Preserve real resource failure, with no fake service metrics.
                    # Abort pilot rather than silently substituting another candidate.
                    raise PilotExecutionBlocked(rep['run_status']+': raw result retained; remaining repetitions not launched')
                record.update(status='complete',runtime=rep);reps.append(rep);cache.flush()
                print(json.dumps({'event':'runtime_complete','rep':index,'stability':rep['stability_state_v2'],'mean_ms':rep['weighted_mean_e2e']}),flush=True)
            path=folder/'aggregate.json';result=aggregate(reps,key,path,identity['deployment']);save_json(path,result)
            entry.update(status='complete',aggregate=str(path),aggregate_sha256=file_hash(path));cache.flush()
            print(json.dumps({'event':'state_complete','cache_key':key,'deployment':deployment,
                'stability':result.stability_state_v2,'mean_ms':result.weighted_mean_e2e}),flush=True)
            return result
