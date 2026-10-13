"""Three-repetition adapter over the existing measurement harness, never new workers."""
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from joint_deploy.measurement.schema import MeasurementConfig
from joint_deploy.measurement.run_measurement import DEFAULT_POOL, DEFAULT_PROFILE
from performance_model.local_calibration.stability_v2 import reconstruct, classify, DEFAULT_THRESHOLDS

from .runtime import RuntimeEvaluator, RuntimeEvaluation, STABILITY, weighted_task_latency
from .runtime_cache import RuntimeEvaluationCache, file_hash, save_json
from .result import jsonable

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MeasurementRuntimeEvaluation(RuntimeEvaluation):
    per_task_mean: dict = field(default_factory=dict)
    per_task_p95: dict = field(default_factory=dict)
    worker_latency: dict = field(default_factory=dict)
    replicate_statistics: dict = field(default_factory=dict)
    repetitions: list = field(default_factory=list)
    raw_result_paths: list = field(default_factory=list)
    cache_key: str = ''


def stats(values):
    present = [v for v in values if v is not None and math.isfinite(v)]
    return dict(mean=statistics.mean(present) if present else None,
                sample_std=statistics.stdev(present) if len(present) > 1 else 0. if present else None,
                values=values, valid_repetitions=len(present), total_repetitions=len(values))


def complete_mean(values):
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def failure_raw(config, error):
    return dict(config, status='failed', errors=[error], request_records=[], warmup_request_records=[],
                priming_request_records=[], measurement_window=None, per_task_metrics={},
                system_metrics=None, models={}, shutdown={}, adapter_failure=True)


def measurement_subprocess(config, raw_path, log_path, config_path):
    """Bounded existing harness subprocess; kill only its own process group on timeout."""
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, '-m', 'joint_deploy.measurement.run_measurement',
               '--config', str(config_path), '--output', str(raw_path)]
    count = sum(config['deployment'].values())
    timeout = (count * config['startup_timeout_seconds'] + config['measurement_seconds']
               + config['warmup_seconds'] + 3 * config['drain_timeout_seconds']
               + count * config['shutdown_timeout_seconds'] + 120)
    start = time.time()
    with Path(log_path).open('x') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        save_json(Path(raw_path).with_suffix('.process.json'), dict(pid=proc.pid, start_unix=start, command=command))
        try:
            code = proc.wait(timeout=timeout)
            outcome = dict(returncode=code, timed_out=False)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            outcome = dict(returncode=proc.returncode, timed_out=True)
    outcome['elapsed_seconds'] = time.time() - start
    return outcome


def summarize_repetition(raw, workload, catalog, outcome, raw_path):
    try:
        evidence = reconstruct(raw, catalog, DEFAULT_THRESHOLDS)
        label = classify(evidence, DEFAULT_THRESHOLDS)
        evidence.pop('queue_wait_values_ms', None)
    except (ValueError, KeyError, TypeError) as exc:
        evidence = dict(run_status=raw.get('status'), run_errors=raw.get('errors', []),
                        reconstruction_error=f'{type(exc).__name__}: {exc}')
        label = dict(stability_state_v2='unstable', stability_reason_v2='missing_or_invalid_measurement_evidence',
                     thresholds=dict(DEFAULT_THRESHOLDS))
    if outcome.get('returncode', 0) != 0 or outcome.get('timed_out') or outcome.get('configuration_mismatch'):
        label = dict(label, stability_state_v2='unstable', stability_reason_v2='harness_error; ' + label['stability_reason_v2'])
    tasks = raw.get('per_task_metrics') or {}
    means = {k: tasks.get(k, {}).get('mean_e2e_latency_ms') for k in workload.active_tasks}
    tails = {k: tasks.get(k, {}).get('p95_e2e_latency_ms') for k in workload.active_tasks}
    def weighted(values):
        if any(v is None or not math.isfinite(v) for v in values.values()):
            return None
        return weighted_task_latency(workload, values)
    mean, p95 = weighted(means), weighted(tails)
    if mean is None or p95 is None:
        label = dict(label, stability_state_v2='unstable', stability_reason_v2=label['stability_reason_v2'] + '; missing_task_latency')
    system = raw.get('system_metrics') or {}
    workers = {k: {name: tasks.get(k, {}).get(name) for name in
                  ('mean_worker_latency_ms', 'p95_worker_latency_ms',
                   'mean_worker_queue_wait_ms', 'p95_worker_queue_wait_ms')} for k in workload.active_tasks}
    return dict(**label, weighted_mean_e2e=mean, weighted_p95_e2e=p95,
                per_task_mean=means, per_task_p95=tails, per_task_metrics=tasks, worker_latency=workers,
                throughput=system.get('achieved_total_throughput_rps') or 0.,
                completion_ratio=system.get('completion_ratio') or 0., queue_metrics=evidence,
                system_metrics=system, raw_status=raw.get('status'), errors=raw.get('errors', []),
                process_outcome=outcome, raw_result_path=str(raw_path), raw_sha256=file_hash(raw_path))


def aggregate_repetitions(repetitions, key, aggregate_path):
    state = min((r['stability_state_v2'] for r in repetitions), key=STABILITY.get)
    metrics = {k: stats([r[k] for r in repetitions]) for k in
               ('weighted_mean_e2e', 'weighted_p95_e2e', 'throughput', 'completion_ratio')}
    task_names = sorted(repetitions[0]['per_task_mean'])
    per_mean = {k: stats([r['per_task_mean'][k] for r in repetitions]) for k in task_names}
    per_p95 = {k: stats([r['per_task_p95'][k] for r in repetitions]) for k in task_names}
    worker = {k: {metric: stats([r['worker_latency'][k][metric] for r in repetitions])
                  for metric in repetitions[0]['worker_latency'][k]} for k in task_names}
    return MeasurementRuntimeEvaluation(
        state, complete_mean([r['weighted_mean_e2e'] for r in repetitions]),
        complete_mean([r['weighted_p95_e2e'] for r in repetitions]),
        metrics['throughput']['mean'], metrics['completion_ratio']['mean'],
        queue_metrics={'per_repetition': [r['queue_metrics'] for r in repetitions]},
        raw_result_path=str(aggregate_path), per_task_mean=per_mean, per_task_p95=per_p95,
        worker_latency=worker, replicate_statistics=metrics, repetitions=repetitions,
        raw_result_paths=[r['raw_result_path'] for r in repetitions], cache_key=key)


class MeasurementRuntimeEvaluator(RuntimeEvaluator):
    def __init__(self, directory, catalog, source_hashes, settings, seeds=(7101, 7202, 7303),
                 max_unique_runtime_evaluations=6, runner=measurement_subprocess):
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError('Pilot requires three distinct repetition seeds')
        self.cache = RuntimeEvaluationCache(directory, max_unique_runtime_evaluations, len(seeds))
        self.catalog, self.source_hashes, self.settings = catalog, jsonable(source_hashes), dict(settings)
        self.seeds, self.runner = tuple(seeds), runner

    def identity(self, deployment, workload, routing):
        tasks = [k for k, rate in workload.arrival_rate.items()
                 if rate > 0 or any(k in p.supported_tasks for p in self.catalog.values())]
        nested = {k: {m: float(routing.get((k, m), 0.)) or 0. for m in sorted(self.catalog)}
                  for k in sorted(tasks)}
        return dict(deployment={m: int(deployment.get(m, 0)) for m in sorted(self.catalog)},
                    workload={k: workload.arrival_rate[k] for k in tasks}, routing=nested,
                    measurement_settings=dict(self.settings, repetition_seeds=list(self.seeds)),
                    source_model_hashes=self.source_hashes,
                    stability_thresholds=dict(DEFAULT_THRESHOLDS), aggregation='worst stability; equal repetition mean; sample std ddof=1; missing latency never imputed')

    def evaluate(self, deployment, workload, routing):
        identity = self.identity(deployment, workload, routing)
        MeasurementConfig(identity['deployment'], identity['workload'], identity['routing'],
                          random_seed=self.seeds[0], **self.settings).validate(
                              {m: p.supported_tasks for m, p in self.catalog.items()})
        with self.cache.locked() as cache:
            key, entry = cache.reserve(identity)
            if entry['status'] == 'complete':
                aggregate_path = Path(entry['aggregate'])
                if file_hash(aggregate_path) != entry['aggregate_sha256']:
                    raise RuntimeError('Cached aggregate hash mismatch; refusing a repeat measurement')
                result = MeasurementRuntimeEvaluation(**json.loads(aggregate_path.read_text()))
                if any(file_hash(r['raw_result_path']) != r['raw_sha256'] for r in result.repetitions):
                    raise RuntimeError('Cached raw result hash mismatch; refusing a repeat measurement')
                cache.data['cache_hits'] += 1
                cache.flush()
                print(json.dumps(dict(event='cache_hit', cache_key=key)), flush=True)
                return result
            directory = cache.directory / 'raw_runs' / key
            directory.mkdir(parents=True, exist_ok=True)
            reps = []
            for index, seed in enumerate(self.seeds, 1):
                raw_path = directory / f'rep_{index}.json'
                config_path = directory / f'rep_{index}.config.json'
                log_path = directory / f'rep_{index}.log'
                cfg = MeasurementConfig(identity['deployment'], identity['workload'], identity['routing'],
                                        random_seed=seed, **self.settings)
                cfg.validate({m: p.supported_tasks for m, p in self.catalog.items()})
                config = cfg.as_dict()
                matching = [r for r in entry['repetitions'] if r['index'] == index]
                if not matching:
                    if len(cache.data['run_attempts']) >= cache.max_unique * cache.repetitions:
                        raise RuntimeError('Real run budget exhausted')
                    # Persist reservation before launch: a crash never causes this repetition to be rerun.
                    record = dict(index=index, seed=seed, status='started', raw_result_path=str(raw_path),
                                  config_path=str(config_path), log_path=str(log_path), started_unix=time.time())
                    entry['repetitions'].append(record)
                    cache.data['run_attempts'].append(dict(cache_key=key, repetition=index, raw_result_path=str(raw_path)))
                    save_json(config_path, config)
                    cache.flush()
                    print(json.dumps(dict(event='run_start', cache_key=key, repetition=index,
                                          deployment=deployment, real_run_count=len(cache.data['run_attempts']))), flush=True)
                    try:
                        outcome = self.runner(config, raw_path, log_path, config_path)
                    except Exception as exc:
                        outcome = dict(returncode=-1, adapter_error=f'{type(exc).__name__}: {exc}')
                    record['process_outcome'] = outcome
                else:
                    record = matching[0]
                    outcome = record.get('process_outcome', {'returncode': -1, 'interrupted': True})
                    # Recover raw output, never launch the same repetition again.
                    process_path = raw_path.with_suffix('.process.json')
                    if record['status'] == 'started' and process_path.exists():
                        pid = json.loads(process_path.read_text())['pid']
                        try:
                            os.kill(pid, 0)
                        except ProcessLookupError:
                            pass
                        else:
                            raise RuntimeError(f'Previous measurement process {pid} is still alive; no duplicate launch')
                if not raw_path.exists():
                    save_json(raw_path, failure_raw(config, 'Measurement produced no raw result: ' + str(outcome)))
                try:
                    raw = json.loads(raw_path.read_text())
                except (ValueError, OSError) as exc:
                    # Preserve malformed original bytes; separate adapter failure envelope.
                    raw_path = directory / f'rep_{index}.adapter_failure.json'
                    if not raw_path.exists():
                        save_json(raw_path, failure_raw(config, 'Invalid raw result: ' + str(exc)))
                    raw = json.loads(raw_path.read_text())
                if any(raw.get(k) != v for k, v in config.items() if k != 'arrival_process'):
                    outcome = dict(outcome, configuration_mismatch=True)
                rep = summarize_repetition(raw, workload, self.catalog, outcome, raw_path)
                rep.update(repetition=index, random_seed=seed)
                reps.append(rep)
                record.update(status='complete', runtime=rep)
                cache.flush()
                print(json.dumps(dict(event='run_complete', cache_key=key, repetition=index,
                                      stability=rep['stability_state_v2'], mean_ms=rep['weighted_mean_e2e'])), flush=True)
            aggregate_path = directory / 'aggregate.json'
            result = aggregate_repetitions(reps, key, aggregate_path)
            save_json(aggregate_path, result)
            entry.update(status='complete', aggregate=str(aggregate_path), aggregate_sha256=file_hash(aggregate_path))
            cache.flush()
            print(json.dumps(dict(event='evaluation_complete', cache_key=key,
                                  stability=result.stability_state_v2, mean_ms=result.weighted_mean_e2e,
                                  std_ms=result.replicate_statistics['weighted_mean_e2e']['sample_std'])), flush=True)
            return result
