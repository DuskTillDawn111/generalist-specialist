"""Module 3C exact shared cache with separate execution and serving labels.

A completed timed measurement remains valid even when serving is genuinely
unstable. Attempts without a completed measurement never enter the aggregate.
"""
import json
import os
from pathlib import Path
import shutil
import time

from joint_deploy.measurement.schema import MeasurementConfig
from performance_model.local_calibration.stability_v2 import DEFAULT_THRESHOLDS

from .measurement_runtime import (MeasurementRuntimeEvaluator, MeasurementRuntimeEvaluation,
                                  aggregate_repetitions, measurement_subprocess, summarize_repetition)
from .runtime_cache import digest, file_hash, save_json


def execution_status(raw, outcome, log_path):
    """Decide execution validity before computing stability_v2."""
    if raw is None:
        return 'uncertain_interruption', 'No parseable harness result; external cause is unproven.'
    window = raw.get('measurement_window') or {}
    system = raw.get('system_metrics') or {}
    complete = bool(window and system and window.get('duration_seconds') == raw.get('measurement_seconds')
                    and window.get('arrival_loop_end_time_ns', 0) >= window.get('end_time_ns', 1)
                    and window.get('drain_end_time_ns', 0) >= window.get('arrival_loop_end_time_ns', 1))
    footer = False
    if Path(log_path).exists():
        tail = Path(log_path).read_text(errors='replace')[-1600:]
        footer = '"output"' in tail
    if complete and footer and not outcome.get('configuration_mismatch'):
        return 'completed', 'Timed measurement, drain, result and harness footer complete.'
    if raw.get('adapter_failure'):
        return 'uncertain_interruption', 'Adapter envelope without completed harness evidence.'
    if any('worker' in str(e).lower() or 'model' in str(e).lower() for e in raw.get('errors', [])):
        return 'model_failure', 'Worker/model failed before the completed measurement contract.'
    return 'harness_failure', 'Measurement or harness completion evidence missing.'


class CampaignRuntimeEvaluator(MeasurementRuntimeEvaluator):
    def __init__(self, directory, catalog, source_hashes, settings, algorithm_id,
                 seeds=(7101,7202,7303), max_unique=75, physical_attempt_cap=240,
                 runner=measurement_subprocess):
        super().__init__(directory, catalog, source_hashes, settings, seeds,
                         max_unique_runtime_evaluations=max_unique, runner=runner)
        self.algorithm_id = algorithm_id
        self.physical_attempt_cap = physical_attempt_cap
        self.access_path = Path(directory).parent / 'cache_access_audit.jsonl'

    def identity(self, deployment, workload, routing):
        value = super().identity(deployment, workload, routing)
        value['execution_status_policy'] = ('Data Quality Audit v1: only completed timed measurements '
            'have stability_v2; genuine completed unstable repetitions remain valid; '
            'external/uncertain/harness/model interruptions excluded from runtime labels')
        return value

    def _record(self, cache, entry, record):
        entry.setdefault('attempts', []).append(record)
        cache.data['run_attempts'].append(dict(record))
        cache.flush()

    def _update(self, cache, record):
        for item in cache.data['run_attempts']:
            if item['attempt_id'] == record['attempt_id']:
                item.update(record)
                break
        cache.flush()

    def _archive(self, cache, record):
        target = cache.directory.parent / 'interruption_archive' / f"attempt_{record['attempt_id']:04d}"
        target.mkdir(parents=True, exist_ok=True)
        evidence = {}
        for name in ('config_path','raw_result_path','log_path','process_path'):
            path = Path(record[name])
            if path.exists():
                dest = target / path.name
                shutil.copy2(path, dest)
                evidence[name] = dict(path=str(dest), sha256=file_hash(dest))
        record['archive'] = evidence
        self._update(cache, record)

    def evaluate(self, deployment, workload, routing):
        identity = self.identity(deployment, workload, routing)
        key = digest(identity)
        MeasurementConfig(identity['deployment'], identity['workload'], identity['routing'],
                          random_seed=self.seeds[0], **self.settings).validate(
                              {m:p.supported_tasks for m,p in self.catalog.items()})
        with self.cache.locked() as cache:
            existing=cache.data['configurations'].get(key)
            access=dict(timestamp_unix=time.time(),algorithm_id=self.algorithm_id,
                        cache_key=key,deployment=identity['deployment'],workload=identity['workload'],
                        requested_identity_sha256=key,
                        previous_status=existing['status'] if existing else 'absent',
                        access_reason='algorithm_selected_state')
            with self.access_path.open('a') as f:
                f.write(json.dumps(access,sort_keys=True)+'\n')
                f.flush();os.fsync(f.fileno())
            key,entry=cache.reserve(identity)
            if entry['status']=='complete':
                path=Path(entry['aggregate'])
                if file_hash(path)!=entry['aggregate_sha256']:
                    raise RuntimeError('Cached aggregate hash mismatch')
                result=MeasurementRuntimeEvaluation(**json.loads(path.read_text()))
                if len(result.repetitions)!=3 or any(r.get('run_status')!='completed' or
                        file_hash(r['raw_result_path'])!=r['raw_sha256'] for r in result.repetitions):
                    raise RuntimeError('Cached repetition identity or execution status invalid')
                cache.data['cache_hits']+=1;cache.flush()
                print(json.dumps(dict(event='cache_hit',algorithm=self.algorithm_id,key=key)),flush=True)
                return result
            directory=cache.directory/'raw_runs'/key
            directory.mkdir(parents=True,exist_ok=True)
            reps=[]
            for index,seed in enumerate(self.seeds,1):
                valid=[r for r in entry['repetitions'] if r['index']==index and r['run_status']=='completed']
                if valid:
                    rep=valid[0]['runtime']
                    if file_hash(rep['raw_result_path'])!=rep['raw_sha256']:
                        raise RuntimeError('Completed repetition hash mismatch')
                    reps.append(rep)
                    continue
                while True:
                    active=[r for r in entry.get('attempts',[]) if r['index']==index and r['status']=='started']
                    if active:
                        record=active[-1]
                        process_path=Path(record['process_path'])
                        if process_path.exists():
                            pid=json.loads(process_path.read_text())['pid']
                            try:os.kill(pid,0)
                            except ProcessLookupError:pass
                            else:raise RuntimeError(f'Prior harness pid {pid} is still alive')
                        outcome=record.get('process_outcome',dict(returncode=-1,recovered_after_parent_stop=True))
                    else:
                        if len(cache.data['run_attempts'])>=self.physical_attempt_cap:
                            raise RuntimeError('Physical attempt safety cap reached')
                        attempt_id=len(cache.data['run_attempts'])+1
                        stem=f'rep_{index}_attempt_{attempt_id:04d}'
                        raw_path=directory/(stem+'.json')
                        config_path=directory/(stem+'.config.json')
                        log_path=directory/(stem+'.log')
                        process_path=raw_path.with_suffix('.process.json')
                        cfg=MeasurementConfig(identity['deployment'],identity['workload'],identity['routing'],
                                              random_seed=seed,**self.settings)
                        cfg.validate({m:p.supported_tasks for m,p in self.catalog.items()})
                        save_json(config_path,cfg.as_dict())
                        record=dict(attempt_id=attempt_id,index=index,seed=seed,status='started',
                                    cache_key=key,algorithm_id=self.algorithm_id,
                                    raw_result_path=str(raw_path),config_path=str(config_path),
                                    log_path=str(log_path),process_path=str(process_path),
                                    started_unix=time.time())
                        self._record(cache,entry,record)
                        print(json.dumps(dict(event='run_start',algorithm=self.algorithm_id,
                                              key=key,rep=index,attempt_id=attempt_id,
                                              deployment=deployment)),flush=True)
                        try:outcome=self.runner(cfg.as_dict(),raw_path,log_path,config_path)
                        except Exception as exc:outcome=dict(returncode=-1,adapter_error=f'{type(exc).__name__}: {exc}')
                        record['process_outcome']=outcome
                        self._update(cache,record)
                    raw_path=Path(record['raw_result_path'])
                    raw=None
                    if raw_path.exists():
                        try:raw=json.loads(raw_path.read_text())
                        except (OSError,ValueError):pass
                    cfg=json.loads(Path(record['config_path']).read_text())
                    if raw is not None and any(raw.get(k)!=v for k,v in cfg.items() if k!='arrival_process'):
                        outcome=dict(outcome,configuration_mismatch=True)
                    status,reason=execution_status(raw,outcome,record['log_path'])
                    record.update(status=status,run_status=status,run_status_reason=reason,
                                  ended_unix=time.time(),process_outcome=outcome,
                                  raw_sha256=file_hash(raw_path) if raw_path.exists() else None)
                    if status!='completed':
                        record['valid_runtime_repetition']=False
                        self._archive(cache,record)
                        print(json.dumps(dict(event='invalid_attempt',algorithm=self.algorithm_id,
                            attempt_id=record['attempt_id'],run_status=status)),flush=True)
                        if sum(a['index']==index and a['status']!='completed' for a in entry['attempts'])>=3:
                            raise RuntimeError('Three invalid attempts for one repetition; review interruption archive')
                        continue
                    # Normalize only the process exit annotation: a completed
                    # result can have raw status=failed because serving overloaded.
                    rep=summarize_repetition(raw,workload,self.catalog,dict(returncode=0),raw_path)
                    rep.update(repetition=index,random_seed=seed,run_status='completed',
                               process_outcome=outcome,valid_runtime_repetition=True)
                    record.update(valid_runtime_repetition=True,runtime=rep)
                    entry['repetitions'].append(dict(index=index,seed=seed,run_status='completed',
                                                      attempt_id=record['attempt_id'],runtime=rep))
                    self._update(cache,record)
                    reps.append(rep)
                    print(json.dumps(dict(event='run_complete',algorithm=self.algorithm_id,
                        attempt_id=record['attempt_id'],run_status='completed',
                        stability=rep['stability_state_v2'],mean_ms=rep['weighted_mean_e2e'])),flush=True)
                    break
            aggregate_path=directory/'aggregate.json'
            result=aggregate_repetitions(reps,key,aggregate_path)
            save_json(aggregate_path,result)
            entry.update(status='complete',aggregate=str(aggregate_path),aggregate_sha256=file_hash(aggregate_path))
            cache.flush()
            print(json.dumps(dict(event='evaluation_complete',algorithm=self.algorithm_id,
                key=key,stability=result.stability_state_v2,
                mean_ms=result.weighted_mean_e2e)),flush=True)
            return result
